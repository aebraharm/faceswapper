"""ONNX Runtime adapter for a two-graph identity-encoder + face-swapper model bundle.

The adapter intentionally defines an explicit ABI instead of guessing at incompatible
third-party swapper input conventions. See models/README.md before supplying weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.transformation.base import FaceTransformer, ModelNotConfiguredError


class OnnxIdentityTransformer(FaceTransformer):
    name = "ONNX identity encoder + target-conditioned transformer"

    def __init__(
        self,
        source_encoder_path: Path | None,
        transformer_path: Path | None,
        provider: str = "auto",
    ) -> None:
        self.source_encoder_path = source_encoder_path
        self.transformer_path = transformer_path
        self.requested_provider = provider
        self.device = "not loaded"
        self._ort = None
        self._encoder = None
        self._transformer = None
        self._provider = "CPUExecutionProvider"

    def load_model(self) -> None:
        if not self.source_encoder_path or not self.transformer_path:
            raise ModelNotConfiguredError(
                "No licensed model bundle is configured. Set FACE_SOURCE_ENCODER_MODEL and "
                "FACE_TRANSFORMER_MODEL to compatible ONNX files (see models/README.md)."
            )
        for path in (self.source_encoder_path, self.transformer_path):
            if not path.is_file():
                raise ModelNotConfiguredError(f"Model file does not exist: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelNotConfiguredError(
                "ONNX Runtime is not installed. Install backend/requirements.txt first."
            ) from exc
        available = ort.get_available_providers()
        if self.requested_provider == "auto":
            candidates = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
        elif self.requested_provider in available:
            candidates = [self.requested_provider]
        else:
            raise ModelNotConfiguredError(
                f"Provider {self.requested_provider} is unavailable. Available providers: {', '.join(available)}"
            )
        if not candidates:
            raise ModelNotConfiguredError("ONNX Runtime reports no usable execution providers.")

        self._ort = ort
        failures: list[str] = []
        for candidate in candidates:
            self._provider = candidate
            try:
                self._encoder = ort.InferenceSession(str(self.source_encoder_path), providers=[candidate])
                self._transformer = ort.InferenceSession(str(self.transformer_path), providers=[candidate])
                required = {item.name for item in self._transformer.get_inputs()}
                if not {"source_identity", "target_face"}.issubset(required):
                    raise ModelNotConfiguredError(
                        "The transformer graph must expose float inputs named 'source_identity' and 'target_face'. "
                        "See the model ABI in models/README.md."
                    )
                self.device = "CUDA" if "CUDA" in candidate else "CPU"
                self._warm_up()
                return
            except Exception as exc:
                failures.append(f"{candidate}: {exc}")
                self._encoder = None
                self._transformer = None
                # ``auto`` degrades to CPU when a CUDA provider is installed but unusable.
                if self.requested_provider != "auto":
                    break
        self.unload_model()
        raise ModelNotConfiguredError("Model load failed. " + " | ".join(failures))

    def _warm_up(self) -> None:
        assert self._encoder is not None and self._transformer is not None
        warm_face = np.zeros((1, 3, 256, 256), dtype=np.float32)
        encoder_input = self._encoder.get_inputs()[0]
        identity = self._encoder.run(None, {encoder_input.name: warm_face})[0]
        identity = np.asarray(identity, dtype=np.float32)
        if not np.isfinite(identity).all():
            raise ModelNotConfiguredError("The source encoder warm-up produced non-finite identity features.")
        try:
            self._transformer.run(
                None,
                {"source_identity": identity, "target_face": warm_face},
            )
        except Exception as exc:
            raise ModelNotConfiguredError(
                "The transformer graph does not accept the identity encoder output and 256×256 target ABI."
            ) from exc

    @staticmethod
    def _tensor(rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        return ((resized.astype(np.float32) / 127.5) - 1.0).transpose(2, 0, 1)[None, ...]

    def prepare_source(self, source_face_rgb: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("The ONNX source encoder is not loaded.")
        input_name = self._encoder.get_inputs()[0].name
        output = self._encoder.run(None, {input_name: self._tensor(source_face_rgb)})[0]
        # Return an owned, compact array: no source image is written to disk.
        return np.asarray(output, dtype=np.float32).copy()

    def transform(self, source_representation: Any, target_face_rgb: np.ndarray) -> np.ndarray:
        if self._transformer is None:
            raise RuntimeError("The ONNX face transformer is not loaded.")
        target = self._tensor(target_face_rgb)
        output = self._transformer.run(
            None,
            {"source_identity": np.asarray(source_representation, dtype=np.float32), "target_face": target},
        )[0]
        image = np.asarray(output, dtype=np.float32)
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.transpose(1, 2, 0)
        if image.shape != (256, 256, 3):
            raise RuntimeError(f"Transformer output must be 1×3×256×256 or 256×256×3; got {image.shape}.")
        if image.min() < -0.01:
            image = (image + 1.0) * 127.5
        elif image.max() <= 1.01:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    def unload_model(self) -> None:
        self._encoder = None
        self._transformer = None
        self._ort = None
        self.device = "not loaded"
