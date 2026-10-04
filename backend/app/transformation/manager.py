"""Lifecycle manager around the selected, replaceable FaceTransformer implementation."""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.config import Settings
from app.transformation.base import FaceTransformer, ModelNotConfiguredError
from app.transformation.onnx_transformer import OnnxIdentityTransformer


class TransformerManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: FaceTransformer | None = None
        self._last_error: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def load(self, provider: str = "auto") -> dict[str, Any]:
        with self._lock:
            if self._model is not None:
                return self.status()
            encoder_path = self.settings.resolved_model_path(self.settings.source_encoder_model)
            transformer_path = self.settings.resolved_model_path(self.settings.face_transformer_model)
            model = OnnxIdentityTransformer(encoder_path, transformer_path, provider)
            try:
                model.load_model()
            except (ModelNotConfiguredError, RuntimeError, ValueError) as exc:
                self._last_error = str(exc)
                raise
            self._model = model
            self._last_error = None
            return self.status()

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                self._model.unload_model()
            self._model = None

    def prepare_source(self, aligned_source: np.ndarray) -> Any | None:
        with self._lock:
            return self._model.prepare_source(aligned_source) if self._model is not None else None

    def transform(self, representation: Any, target_face: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No face-transformation model is loaded.")
            return self._model.transform(representation, target_face)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": self._model is not None,
                "name": self._model.name if self._model else "No model loaded",
                "device": self._model.device if self._model else self._runtime_device(),
                "provider": getattr(self._model, "_provider", None) if self._model else None,
                "error": self._last_error,
                "requires_model_files": self._model is None,
            }

    @staticmethod
    def _runtime_device() -> str:
        try:
            import onnxruntime as ort

            return "CUDA available" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPU"
        except ImportError:
            return "CPU (ONNX Runtime not installed)"
