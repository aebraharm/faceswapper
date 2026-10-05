"""Per-frame detector → tracker → aligner → model → mask/blend → JPEG pipeline."""
from __future__ import annotations

from contextlib import nullcontext
import time
from typing import Any

import cv2
import numpy as np

from app.alignment.face_aligner import FaceAligner
from app.blending.compositor import FaceCompositor
from app.camera.frame_diagnostics import (
    FrameDiagnostics,
    log_frontend_frame_diagnostics,
    record_error,
    record_event,
    record_image,
    record_payload,
)
from app.camera.session import CameraSession
from app.camera.target_tracker import TargetFaceTracker
from app.detection.analyzer import FaceAnalyzer
from app.transformation.manager import TransformerManager


class FrameProcessor:
    def __init__(
        self,
        analyzer: FaceAnalyzer,
        aligner: FaceAligner,
        compositor: FaceCompositor,
        transformer: TransformerManager,
        source_faces: Any,
        camera: CameraSession,
        *,
        diagnostics_enabled: bool = False,
        diagnostics_interval_ms: int = 1_000,
    ) -> None:
        self.analyzer = analyzer
        self.aligner = aligner
        self.compositor = compositor
        self.transformer = transformer
        self.source_faces = source_faces
        self.camera = camera
        self._tracker = TargetFaceTracker()
        self._last_transformed_rgb: np.ndarray | None = None
        self._diagnostics_enabled = diagnostics_enabled
        self._diagnostics_interval_seconds = max(1, diagnostics_interval_ms) / 1_000.0
        self._next_diagnostics_at = 0.0

    def _sample_diagnostics(self) -> FrameDiagnostics | None:
        """Create at most one opt-in trace per interval, never per live frame."""
        if not self._diagnostics_enabled:
            return None
        now = time.monotonic()
        if now < self._next_diagnostics_at:
            return None
        self._next_diagnostics_at = now + self._diagnostics_interval_seconds
        return FrameDiagnostics()

    def record_frontend_diagnostics(self, values: dict[str, Any]) -> None:
        """Persist a renderer acknowledgement only for an already sampled frame."""
        if self._diagnostics_enabled:
            log_frontend_frame_diagnostics(values)

    @staticmethod
    def _record_source_representation(representation: Any) -> None:
        """Record cached numeric source state without retaining or exposing source pixels."""
        for field in ("feature_3d", "x_s", "kp_canonical", "scale"):
            value = getattr(representation, field, None)
            if value is not None:
                record_image(f"source_representation.{field}", value, layout="NCHW" if field == "feature_3d" else "HWC")

    def process_jpeg(self, payload: bytes) -> tuple[bytes, dict[str, Any]]:
        started = time.perf_counter()
        diagnostics = self._sample_diagnostics()
        scope = diagnostics.activate() if diagnostics is not None else nullcontext()
        try:
            with scope:
                # The diagnostic decode validates the exact browser JPEG payload in
                # addition to the production decode below. It only runs for sampled
                # opt-in frames, never on the normal performance-sensitive path.
                record_payload("camera_capture.jpeg", payload, decode_image=True)
                encoded = np.frombuffer(payload, dtype=np.uint8)
                bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Camera frame is not a valid JPEG image.")
                record_image("camera_capture.bgr", bgr, channel_order="BGR")
                height, width = bgr.shape[:2]
                if width < 2 or height < 2:
                    raise ValueError("Camera frame dimensions are invalid.")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                record_image("preprocessing.rgb", rgb, channel_order="RGB")
                settings = self.camera.settings()
                max_dimension = settings["processing_resolution"]
                scale = min(1.0, max_dimension / float(max(width, height)))
                if scale < 1.0:
                    work = cv2.resize(
                        rgb,
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    work = rgb.copy()
                record_image("preprocessing.work_rgb", work, channel_order="RGB")

                observations = self.analyzer.analyze(work)
                selected_preference = self.camera.status()["selected_target_index"]
                selected_index, target = self._tracker.select(observations, selected_preference)
                record_event(
                    "target_tracker.selection",
                    {
                        "requested_index": selected_preference,
                        "detected_faces": [face.as_dict() for face in observations],
                        "selected_index": selected_index,
                        "tracked_face": target.as_dict() if target is not None else None,
                        "tracked_landmarks5_shape": list(np.asarray(target.landmarks5).shape)
                        if target is not None and target.landmarks5 is not None
                        else None,
                    },
                )
                # If there is a single face the tracker chooses it automatically; multiple faces
                # remain untouched until the user selects one explicitly.
                self.camera.update_faces([face.as_dict() for face in observations], selected_index)
                output = work
                model_status = self.transformer.status()
                source_status = self.source_faces.status()
                transform_active = bool(
                    settings["transform_enabled"]
                    and model_status["loaded"]
                    and source_status["model_ready"]
                    and target is not None
                )
                skipped_inference = False
                if transform_active and target is not None:
                    representation = self.source_faces.representation()
                    if representation is not None:
                        self._record_source_representation(representation)
                        run_inference = self.camera.should_run_inference()
                        if (
                            not run_inference
                            and self._last_transformed_rgb is not None
                            and self._last_transformed_rgb.shape == work.shape
                        ):
                            # Performance mode: reuse the last composited frame instead of
                            # falling further behind on a slow (e.g. 4 GB RAM) machine.
                            output = self._last_transformed_rgb
                            record_image("performance.reused_composite_rgb", output, channel_order="RGB")
                            skipped_inference = True
                        else:
                            aligned = self.aligner.align(work, target, output_size=256)
                            record_image("alignment.target_rgb_256", aligned.image, channel_order="RGB")
                            transformed = self.transformer.transform(representation, aligned.image)
                            # LivePortrait preserves its 512px native output through
                            # this boundary; FaceCompositor performs the required
                            # 512→256 resample immediately before inverse alignment.
                            record_image("liveportrait.native_output_rgb", transformed, channel_order="RGB", require_nonzero=True)
                            output = self.compositor.blend(work, transformed, aligned, settings["intensity"])
                            record_image("compositing.output_rgb", output, channel_order="RGB")
                            self._last_transformed_rgb = output
                    else:
                        transform_active = False
                else:
                    self._last_transformed_rgb = None

                # Highlight all detected faces; only the selected target is marked in green.
                annotated = output.copy()
                for i, face in enumerate(observations):
                    x, y, box_w, box_h = face.bbox
                    color = (63, 217, 163) if i == selected_index else (246, 178, 93)
                    thickness = max(1, int(round(min(work.shape[:2]) / 360)))
                    cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, thickness, cv2.LINE_AA)
                record_image("annotation.rgb", annotated, channel_order="RGB")
                result = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                record_image("serialization.bgr", result, channel_order="BGR")
                ok, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise RuntimeError("Could not encode the processed camera frame.")
                frame = jpeg.tobytes()
                # Decode only under diagnostics to prove the outgoing byte payload
                # is a valid image before the browser sees it.
                record_payload("serialization.jpeg", frame, decode_image=True)
                latency_ms = (time.perf_counter() - started) * 1000.0
                self.camera.record_frame(latency_ms, width, height)
                camera_status = self.camera.status()
                stats: dict[str, Any] = {
                    "type": "stats",
                    "fps": camera_status["fps"],
                    "latency_ms": round(latency_ms, 1),
                    "camera_resolution": camera_status["camera_resolution"],
                    "processing_resolution": max(work.shape[1], work.shape[0]),
                    "face_count": len(observations),
                    "faces": [face.as_dict() for face in observations],
                    "selected_target": selected_index,
                    "transformation_active": transform_active,
                    "performance_mode": settings.get("performance_mode", "auto"),
                    "skipped_inference": skipped_inference,
                    "model_status": model_status,
                    "device": model_status["device"],
                }
                if diagnostics is not None:
                    stats["frame_diagnostics"] = diagnostics.report()
                    stats["frame_diagnostics_id"] = diagnostics.trace_id
                return frame, stats
        except Exception as exc:
            if diagnostics is not None:
                # The trace already contains all successful preceding boundaries;
                # this final record pinpoints an exception boundary as the first
                # invalid state when no prior integrity violation was found.
                with diagnostics.activate():
                    record_error("frame_processor", str(exc))
            raise
        finally:
            if diagnostics is not None:
                diagnostics.log()
