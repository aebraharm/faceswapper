"""Opt-in, rate-limited integrity tracing for the camera frame pipeline.

The normal camera path must stay allocation- and logging-light.  Setting
``FRAME_FRAME_DIAGNOSTICS=1`` turns on a sampled trace (one frame per configured
interval) that is attached to the WebSocket stats object and emitted as one
backend log record.  Components below the frame processor contribute to the
same trace through a context variable, without changing their public APIs.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_diagnostic_file_handler: RotatingFileHandler | None = None


def configure_diagnostics_log(path: Path | None) -> None:
    """Persist opt-in diagnostic JSON lines outside the packaged sidecar pipe.

    Electron captures the sidecar's stdout/stderr for startup errors, so those
    streams are not a reliable place for a Windows user to retrieve a live-frame
    investigation.  This handler is installed only when diagnostics are enabled.
    It stores aggregate tensor/image facts, never pixels or JPEG bytes, and keeps
    the current file plus two 2 MiB rotations.
    """
    global _diagnostic_file_handler
    if path is None:
        if _diagnostic_file_handler is not None:
            logger.removeHandler(_diagnostic_file_handler)
            _diagnostic_file_handler.close()
            _diagnostic_file_handler = None
        return
    resolved = path.expanduser().resolve()
    if _diagnostic_file_handler is not None:
        if Path(_diagnostic_file_handler.baseFilename) == resolved:
            return
        logger.removeHandler(_diagnostic_file_handler)
        _diagnostic_file_handler.close()
        _diagnostic_file_handler = None
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(resolved, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8")
    except OSError as exc:
        # Diagnostics must never prevent the local backend from starting. The
        # regular warning stream remains available to a development launcher.
        logger.warning("Could not create FRAME diagnostics log at %s: %s", resolved, exc)
        return
    # Each line is a self-contained JSON object emitted by FrameDiagnostics.log.
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    _diagnostic_file_handler = handler


_current_trace: ContextVar["FrameDiagnostics | None"] = ContextVar("frame_diagnostics", default=None)


def current_frame_diagnostics() -> "FrameDiagnostics | None":
    """Return the trace active for the current frame, if sampled for diagnostics."""
    return _current_trace.get()


def record_image(
    stage: str,
    image: Any,
    *,
    channel_order: str | None = None,
    layout: str = "HWC",
    require_nonzero: bool = False,
) -> None:
    """Append numeric/image integrity facts to the active sampled trace."""
    trace = current_frame_diagnostics()
    if trace is not None:
        trace.record_image(
            stage,
            image,
            channel_order=channel_order,
            layout=layout,
            require_nonzero=require_nonzero,
        )


def record_payload(stage: str, payload: bytes, *, decode_image: bool = False) -> None:
    """Append encoded-payload facts to the active sampled trace."""
    trace = current_frame_diagnostics()
    if trace is not None:
        trace.record_payload(stage, payload, decode_image=decode_image)


def record_error(stage: str, message: str) -> None:
    """Record an exception boundary while preserving the original exception."""
    trace = current_frame_diagnostics()
    if trace is not None:
        trace.record_error(stage, message)


def record_event(stage: str, values: dict[str, Any]) -> None:
    """Record small JSON-safe control-plane facts (detector, tracker, alignment)."""
    trace = current_frame_diagnostics()
    if trace is not None:
        trace.record_event(stage, values)


def _log_diagnostic_record(record_type: str, values: dict[str, Any]) -> None:
    """Write a small opt-in transport/lifecycle record without image payloads."""
    normalized = FrameDiagnostics._json_value(values)
    # ``type`` is the stable JSONL record discriminator. Preserve the wire/event
    # name separately so transport facts are not flattened into one record type.
    event = normalized.pop("type", None)
    if event is not None:
        normalized["event"] = event
    logger.warning(
        "%s",
        json.dumps(
            {**normalized, "type": record_type},
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def log_frontend_frame_diagnostics(values: dict[str, Any]) -> None:
    """Persist a sampled renderer decode/draw acknowledgement from the local UI."""
    _log_diagnostic_record("frontend_frame_diagnostics", values)


def log_camera_transport_diagnostics(values: dict[str, Any]) -> None:
    """Persist client capture/send or backend receipt facts before frame work begins."""
    _log_diagnostic_record("camera_transport_diagnostics", values)


def log_live_frame_started(values: dict[str, Any]) -> None:
    """Persist a live-frame trace ID before running a potentially blocking inference."""
    _log_diagnostic_record("live_camera_frame_started", values)


class FrameDiagnostics:
    """A compact JSON-safe trace for exactly one camera frame.

    A trace is intentionally created only for sampled debug frames.  The first
    invalid boundary is retained separately so a long list of downstream effects
    cannot obscure the first place the image became unusable.
    """

    def __init__(self, trace_kind: str = "unspecified") -> None:
        self.trace_id = uuid4().hex
        # A source upload and a camera frame have intentionally separate traces.
        # Keeping the lifecycle explicit prevents cached-source preparation facts
        # from being mistaken for a live LivePortrait inference.
        self.trace_kind = trace_kind
        self._stages: list[dict[str, Any]] = []
        self._first_invalid: dict[str, str] | None = None

    @contextmanager
    def activate(self) -> Iterator["FrameDiagnostics"]:
        token: Token[FrameDiagnostics | None] = _current_trace.set(self)
        try:
            yield self
        finally:
            _current_trace.reset(token)

    @staticmethod
    def _dimensions(array: np.ndarray, layout: str) -> dict[str, int] | None:
        if layout == "NCHW" and array.ndim == 4:
            return {
                "batch": int(array.shape[0]),
                "channels": int(array.shape[1]),
                "height": int(array.shape[2]),
                "width": int(array.shape[3]),
            }
        if layout == "CHW" and array.ndim == 3:
            return {"channels": int(array.shape[0]), "height": int(array.shape[1]), "width": int(array.shape[2])}
        if array.ndim == 3:
            return {"height": int(array.shape[0]), "width": int(array.shape[1]), "channels": int(array.shape[2])}
        if array.ndim == 2:
            return {"height": int(array.shape[0]), "width": int(array.shape[1]), "channels": 1}
        return None

    def _invalid(self, stage: str, reason: str) -> None:
        if self._first_invalid is None:
            self._first_invalid = {"stage": stage, "reason": reason}

    @staticmethod
    def _json_value(value: Any) -> Any:
        """Keep control-plane metadata compact and JSON serializable."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): FrameDiagnostics._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FrameDiagnostics._json_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def record_event(self, stage: str, values: dict[str, Any]) -> None:
        self._stages.append({"stage": stage, "kind": "event", **self._json_value(values)})

    def record_image(
        self,
        stage: str,
        image: Any,
        *,
        channel_order: str | None = None,
        layout: str = "HWC",
        require_nonzero: bool = False,
    ) -> None:
        array = np.asarray(image)
        detail: dict[str, Any] = {
            "stage": stage,
            "kind": "image",
            "shape": [int(item) for item in array.shape],
            "dtype": str(array.dtype),
            "layout": layout,
        }
        dimensions = self._dimensions(array, layout)
        if dimensions is not None:
            detail["dimensions"] = dimensions
        if channel_order:
            detail["channel_order"] = channel_order

        if array.size == 0:
            detail.update({"finite": False, "all_zero": True, "invalid": "empty array"})
            self._invalid(stage, "empty array")
            self._stages.append(detail)
            return
        if not np.issubdtype(array.dtype, np.number):
            detail.update({"finite": False, "invalid": "non-numeric dtype"})
            self._invalid(stage, "non-numeric dtype")
            self._stages.append(detail)
            return

        finite = np.isfinite(array)
        finite_count = int(np.count_nonzero(finite))
        detail["finite"] = finite_count == array.size
        detail["finite_values"] = finite_count
        detail["total_values"] = int(array.size)
        if finite_count:
            values = array[finite].astype(np.float64, copy=False)
            detail["min"] = float(values.min())
            detail["max"] = float(values.max())
            detail["all_zero"] = bool(np.all(values == 0.0)) and finite_count == array.size
            detail["nonzero_fraction"] = round(float(np.count_nonzero(values)) / float(array.size), 6)
        else:
            detail.update({"min": None, "max": None, "all_zero": False, "nonzero_fraction": 0.0})

        if not detail["finite"]:
            detail["invalid"] = "contains NaN or Inf"
            self._invalid(stage, "contains NaN or Inf")
        elif require_nonzero and detail["all_zero"]:
            detail["invalid"] = "unexpected all-zero output"
            self._invalid(stage, "unexpected all-zero output")

        channel_axis = 1 if layout == "NCHW" else (0 if layout == "CHW" else array.ndim - 1)
        if channel_order and array.ndim in (3, 4) and array.shape[channel_axis] in (1, 3, 4) and finite_count:
            # Move channel axis to the end and collapse spatial/batch dimensions,
            # giving per-channel range/mean independent of RGB/BGR ordering.
            channels_last = np.moveaxis(array, channel_axis, -1).reshape(-1, array.shape[channel_axis])
            channel_stats = []
            for index in range(channels_last.shape[1]):
                values = channels_last[:, index]
                values = values[np.isfinite(values)]
                channel_stats.append(
                    {
                        "min": float(values.min()) if values.size else None,
                        "max": float(values.max()) if values.size else None,
                        "mean": round(float(values.mean()), 4) if values.size else None,
                    }
                )
            detail["channels"] = channel_stats
        self._stages.append(detail)

    def record_payload(self, stage: str, payload: bytes, *, decode_image: bool = False) -> None:
        detail: dict[str, Any] = {
            "stage": stage,
            "kind": "payload",
            "bytes": len(payload),
            "jpeg_soi": payload.startswith(b"\xff\xd8"),
            "jpeg_eoi": payload.endswith(b"\xff\xd9"),
        }
        if not payload:
            detail["invalid"] = "empty payload"
            self._invalid(stage, "empty payload")
        elif not detail["jpeg_soi"] or not detail["jpeg_eoi"]:
            detail["invalid"] = "missing JPEG boundary marker"
            self._invalid(stage, "missing JPEG boundary marker")
        decoded: np.ndarray | None = None
        if decode_image:
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            detail["decode_valid"] = decoded is not None
            if decoded is None:
                detail["invalid"] = "OpenCV could not decode payload"
                self._invalid(stage, "OpenCV could not decode payload")
            else:
                detail["decoded_dimensions"] = {
                    "height": int(decoded.shape[0]),
                    "width": int(decoded.shape[1]),
                    "channels": int(decoded.shape[2]),
                }
        # Append the encoded boundary before its decoded verification so the
        # stage list matches the actual data-flow order.
        self._stages.append(detail)
        if decoded is not None:
            self.record_image(f"{stage}.decoded_bgr", decoded, channel_order="BGR")

    def record_error(self, stage: str, message: str) -> None:
        self._stages.append({"stage": stage, "kind": "error", "message": message})
        self._invalid(stage, message)

    def report(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_kind": self.trace_kind,
            "first_invalid": self._first_invalid,
            "stages": self._stages,
        }

    def log(self) -> None:
        """Emit one JSON-lines record per sampled trace, never one per boundary."""
        logger.warning(
            "%s",
            json.dumps(
                {"type": f"{self.trace_kind}_diagnostics", **self.report()},
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
