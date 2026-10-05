"""ONNX Runtime adapter for the LivePortrait identity-preserving reenactment model.

Pipeline (see docs/model-integration.md for the full derivation):
  prepare_source(source_face):
    f_s      = appearance_feature_extractor(source_face)             # cached feature volume
    kp_info  = motion_extractor(source_face)                         # canonical kp + pose/exp/scale/t
    x_s      = transform_keypoints(kp_canonical=kp_info.kp, kp_info) # source's own absolute keypoints
    representation = {f_s, x_s, kp_canonical=kp_info.kp, scale=kp_info.scale}

  transform(representation, target_face):
    d_info = motion_extractor(target_face)
    x_d    = transform_keypoints(kp_canonical=representation.kp_canonical,
                                  R=R(d_info.pitch/yaw/roll), exp=d_info.exp,
                                  scale=representation.scale, t=d_info.t)
    out    = warping_spade(f_s, x_s, x_d)

`x_d` reuses the *source's* canonical keypoints and own scale, but the target's
pose rotation/expression/translation -- this is what keeps the output identity
stable while letting it track the live camera's head motion and expressions.

FRAME always supplies `source_face`/`target_face` as its own OpenCV+MediaPipe
aligned 256x256 RGB crops; the InsightFace-based face detector that ships with
the reference LivePortrait implementation is never used (see models/README.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

from app.transformation.base import FaceTransformer, ModelNotConfiguredError

# LivePortrait's feature and motion graphs consume 256×256 crops. The fused
# warping/SPADE graph deliberately renders a higher-resolution 512×512 crop.
INPUT_SIZE = 256
NUM_KEYPOINTS = 21


class SourceRepresentation(NamedTuple):
    """Everything cached once per selected source face. No image pixels are kept."""

    feature_3d: np.ndarray
    x_s: np.ndarray
    kp_canonical: np.ndarray
    scale: np.ndarray


def _softmax_degrees(pred: np.ndarray) -> np.ndarray:
    """Match LivePortrait's headpose_pred_to_degree for the 66-bin classifier head."""
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx = np.arange(66, dtype=np.float32)
        exp = np.exp(pred - pred.max(axis=1, keepdims=True))
        softmax = exp / exp.sum(axis=1, keepdims=True)
        return softmax @ idx * 3.0 - 97.5
    return pred.reshape(pred.shape[0], -1)[:, 0] if pred.ndim > 1 else pred.reshape(-1)


def _rotation_matrix(pitch_deg: np.ndarray, yaw_deg: np.ndarray, roll_deg: np.ndarray) -> np.ndarray:
    """Batch (bs,3,3) rotation matrix, matching LivePortrait's utils/camera.py exactly."""
    pitch = np.asarray(pitch_deg, dtype=np.float32).reshape(-1) / 180.0 * np.pi
    yaw = np.asarray(yaw_deg, dtype=np.float32).reshape(-1) / 180.0 * np.pi
    roll = np.asarray(roll_deg, dtype=np.float32).reshape(-1) / 180.0 * np.pi
    bs = pitch.shape[0]
    ones = np.ones(bs, dtype=np.float32)
    zeros = np.zeros(bs, dtype=np.float32)

    cx, sx = np.cos(pitch), np.sin(pitch)
    rot_x = np.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], axis=1).reshape(bs, 3, 3)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_y = np.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], axis=1).reshape(bs, 3, 3)
    cz, sz = np.cos(roll), np.sin(roll)
    rot_z = np.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], axis=1).reshape(bs, 3, 3)

    rot = rot_z @ rot_y @ rot_x
    return rot.transpose(0, 2, 1)


def _transform_keypoints(
    kp_canonical: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
    exp: np.ndarray,
    scale: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Eqn.2 from the LivePortrait paper: s * (R * x_canonical + exp) + t (xy only)."""
    rot = _rotation_matrix(pitch, yaw, roll)
    transformed = kp_canonical @ rot + exp
    transformed = transformed * scale[..., None]
    transformed = transformed.copy()
    transformed[:, :, 0:2] += t[:, None, 0:2]
    return transformed.astype(np.float32)


class _KpInfo(NamedTuple):
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    t: np.ndarray
    exp: np.ndarray
    scale: np.ndarray
    kp: np.ndarray


def _configure_session_options(intra_threads: int | None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    if intra_threads and intra_threads > 0:
        options.intra_op_num_threads = intra_threads
    return options


class LivePortraitOnnxTransformer(FaceTransformer):
    """First-class adapter for the catalog LivePortrait model (see model_catalog.py)."""

    name = "LivePortrait (MIT) identity-preserving reenactment"

    def __init__(
        self,
        appearance_feature_extractor_path: Path,
        motion_extractor_path: Path,
        warping_spade_path: Path,
        provider: str = "auto",
        intra_threads: int | None = None,
    ) -> None:
        self.appearance_feature_extractor_path = appearance_feature_extractor_path
        self.motion_extractor_path = motion_extractor_path
        self.warping_spade_path = warping_spade_path
        self.requested_provider = provider
        self.intra_threads = intra_threads
        self.device = "not loaded"
        self._provider = "CPUExecutionProvider"
        self._appearance = None
        self._motion = None
        self._warp = None
        self._warp_input_order: list[str] | None = None
        self._warp_output_size: tuple[int, int] | None = None

    def load_model(self) -> None:
        for path in (self.appearance_feature_extractor_path, self.motion_extractor_path, self.warping_spade_path):
            if not path or not path.is_file():
                raise ModelNotConfiguredError(
                    f"LivePortrait model file is missing: {path}. Install the model from the Model Status panel."
                )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelNotConfiguredError(
                "ONNX Runtime is not installed. Install backend/requirements.txt first."
            ) from exc
        available = ort.get_available_providers()
        if self.requested_provider == "auto":
            candidates = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        elif self.requested_provider in available:
            candidates = [self.requested_provider]
        else:
            raise ModelNotConfiguredError(
                f"Provider {self.requested_provider} is unavailable. Available providers: {', '.join(available)}"
            )
        if not candidates:
            raise ModelNotConfiguredError("ONNX Runtime reports no usable execution providers.")

        options = _configure_session_options(self.intra_threads)
        failures: list[str] = []
        for candidate in candidates:
            self._provider = candidate
            try:
                self._appearance = ort.InferenceSession(
                    str(self.appearance_feature_extractor_path), sess_options=options, providers=[candidate]
                )
                self._motion = ort.InferenceSession(
                    str(self.motion_extractor_path), sess_options=options, providers=[candidate]
                )
                self._warp = ort.InferenceSession(str(self.warping_spade_path), sess_options=options, providers=[candidate])
                self._warp_input_order = self._resolve_warp_input_order()
                self._warp_output_size = self._resolve_warp_output_size()
                self.device = "CUDA" if "CUDA" in candidate else "CPU"
                self._warm_up()
                return
            except Exception as exc:  # noqa: BLE001 - normalized into a friendly API error below
                failures.append(f"{candidate}: {exc}")
                self._appearance = self._motion = self._warp = None
                if self.requested_provider != "auto":
                    break
        self.unload_model()
        message = "Model load failed. " + " | ".join(failures)
        if any("memory" in failure.lower() or "alloc" in failure.lower() for failure in failures):
            message += (
                " This usually means the device ran out of RAM loading the warping/generator network. "
                "Close other applications, or use a lower processing resolution / CPU-only mode."
            )
        raise ModelNotConfiguredError(message)

    def _resolve_warp_input_order(self) -> list[str]:
        """Map (feature_3d, kp_source, kp_driving) onto this graph's actual input names."""
        assert self._warp is not None
        names = [item.name for item in self._warp.get_inputs()]
        if len(names) != 3:
            raise ModelNotConfiguredError(
                f"The warping/generator graph must take exactly 3 inputs; found {len(names)}: {names}."
            )
        lowered = [name.lower() for name in names]

        def pick(*keywords: str) -> str | None:
            for name, lower in zip(names, lowered):
                if any(keyword in lower for keyword in keywords):
                    return name
            return None

        feature_name = pick("feature", "appearance", "f_s")
        source_name = pick("source")
        driving_name = pick("driving", "target")
        if feature_name and source_name and driving_name:
            return [feature_name, source_name, driving_name]
        # Fall back to the documented FasterLivePortrait ONNX export order.
        return names

    def _resolve_warp_output_size(self) -> tuple[int, int] | None:
        """Validate the fused renderer's single RGB output without assuming 256×256.

        The catalog's FasterLivePortrait export consumes 256×256 model inputs but
        its SPADE decoder renders a 512×512 RGB image. Keep this check tied to
        ONNX metadata so an incompatible graph cannot silently be treated as the
        catalog renderer; a dynamic spatial shape is verified after inference.
        """
        assert self._warp is not None
        outputs = self._warp.get_outputs()
        if len(outputs) != 1:
            raise ModelNotConfiguredError(
                f"The warping/SPADE graph must expose one rendered RGB output; found {len(outputs)}."
            )
        output = outputs[0]
        shape = output.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ModelNotConfiguredError(
                f"The warping/SPADE output '{output.name}' must be NCHW with three RGB channels; got {shape}."
            )
        height, width = shape[2:]
        if isinstance(height, int) and isinstance(width, int):
            if height <= 0 or width <= 0 or height != width:
                raise ModelNotConfiguredError(
                    f"The warping/SPADE output '{output.name}' must be a non-empty square RGB image; got {shape}."
                )
            return height, width
        return None

    def _warm_up(self) -> None:
        warm_face = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        representation = self.prepare_source(warm_face)
        self.transform(representation, warm_face)

    @staticmethod
    def _tensor(rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        chw = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return np.ascontiguousarray(chw[None, ...])

    def _extract_feature_3d(self, rgb: np.ndarray) -> np.ndarray:
        assert self._appearance is not None
        input_name = self._appearance.get_inputs()[0].name
        output = self._appearance.run(None, {input_name: self._tensor(rgb)})[0]
        return np.asarray(output, dtype=np.float32).copy()

    def _extract_kp_info(self, rgb: np.ndarray) -> _KpInfo:
        assert self._motion is not None
        input_name = self._motion.get_inputs()[0].name
        outputs = self._motion.run(None, {input_name: self._tensor(rgb)})
        by_name = {out.name.lower(): value for out, value in zip(self._motion.get_outputs(), outputs)}
        try:
            pitch, yaw, roll, t, exp, scale, kp = (
                by_name["pitch"],
                by_name["yaw"],
                by_name["roll"],
                by_name["t"],
                by_name["exp"],
                by_name["scale"],
                by_name["kp"],
            )
        except KeyError:
            # Positional fallback matching FasterLivePortrait's onnxruntime output order.
            pitch, yaw, roll, t, exp, scale, kp = outputs
        bs = np.asarray(kp).shape[0]
        return _KpInfo(
            pitch=_softmax_degrees(np.asarray(pitch))[:, None],
            yaw=_softmax_degrees(np.asarray(yaw))[:, None],
            roll=_softmax_degrees(np.asarray(roll))[:, None],
            t=np.asarray(t, dtype=np.float32).reshape(bs, -1),
            exp=np.asarray(exp, dtype=np.float32).reshape(bs, -1, 3),
            scale=np.asarray(scale, dtype=np.float32).reshape(bs, -1)[:, :1],
            kp=np.asarray(kp, dtype=np.float32).reshape(bs, NUM_KEYPOINTS, 3),
        )

    def prepare_source(self, source_face_rgb: np.ndarray) -> SourceRepresentation:
        if self._appearance is None or self._motion is None:
            raise RuntimeError("The LivePortrait model is not loaded.")
        feature_3d = self._extract_feature_3d(source_face_rgb)
        info = self._extract_kp_info(source_face_rgb)
        x_s = _transform_keypoints(info.kp, info.pitch, info.yaw, info.roll, info.exp, info.scale, info.t)
        # Own, compact arrays only: no source pixels are retained.
        return SourceRepresentation(
            feature_3d=feature_3d.copy(), x_s=x_s.copy(), kp_canonical=info.kp.copy(), scale=info.scale.copy()
        )

    def transform(self, source_representation: Any, target_face_rgb: np.ndarray) -> np.ndarray:
        if self._warp is None or self._motion is None:
            raise RuntimeError("The LivePortrait model is not loaded.")
        representation: SourceRepresentation = source_representation
        d_info = self._extract_kp_info(target_face_rgb)
        x_d = _transform_keypoints(
            representation.kp_canonical,
            d_info.pitch,
            d_info.yaw,
            d_info.roll,
            d_info.exp,
            representation.scale,
            d_info.t,
        )
        inputs_by_role = {
            "feature": representation.feature_3d,
            "source": representation.x_s,
            "driving": x_d,
        }
        order = self._warp_input_order or [item.name for item in self._warp.get_inputs()]
        feed = {}
        role_sequence = ["feature", "source", "driving"]
        for name, role in zip(order, role_sequence):
            feed[name] = inputs_by_role[role]
        output = self._warp.run(None, feed)[0]
        image = np.asarray(output, dtype=np.float32)
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.transpose(1, 2, 0)
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"LivePortrait output must be an RGB image; got {image.shape}.")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0 or height != width:
            raise RuntimeError(f"LivePortrait output must be a non-empty square RGB image; got {image.shape}.")
        if self._warp_output_size is not None and (height, width) != self._warp_output_size:
            raise RuntimeError(
                "LivePortrait output did not match the warping/SPADE graph metadata: "
                f"expected {self._warp_output_size[0]}x{self._warp_output_size[1]}x3; got {image.shape}."
            )
        if image.max() <= 1.5:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    def unload_model(self) -> None:
        self._appearance = None
        self._motion = None
        self._warp = None
        self._warp_input_order = None
        self._warp_output_size = None
        self.device = "not loaded"
