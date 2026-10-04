"""Lifecycle manager around the selected, replaceable FaceTransformer implementation."""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.config import Settings
from app.transformation.base import FaceTransformer, ModelNotConfiguredError
from app.transformation.liveportrait_transformer import LivePortraitOnnxTransformer
from app.transformation.model_manager import ModelInstallManager
from app.transformation.onnx_transformer import OnnxIdentityTransformer

NO_MODEL_CONFIGURED_MESSAGE = (
    "No licensed model bundle is configured. Install the LivePortrait model from the "
    "Model Status panel, or set FACE_SOURCE_ENCODER_MODEL and FACE_TRANSFORMER_MODEL to "
    "your own compatible ONNX files (see models/README.md)."
)


class TransformerManager:
    def __init__(self, settings: Settings, model_store: ModelInstallManager | None = None) -> None:
        self.settings = settings
        self.model_store = model_store
        self._model: FaceTransformer | None = None
        self._last_error: str | None = None
        self._active_model_id: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def _build_model(self, provider: str) -> tuple[FaceTransformer, str | None]:
        """Prefer an explicit bring-your-own ONNX bundle; otherwise use an installed catalog model."""
        encoder_path = self.settings.resolved_model_path(self.settings.source_encoder_model)
        transformer_path = self.settings.resolved_model_path(self.settings.face_transformer_model)
        if encoder_path and transformer_path:
            return OnnxIdentityTransformer(encoder_path, transformer_path, provider), None

        if self.model_store is not None:
            model_id = self.settings.selected_model_id
            assets = self.model_store.installed_asset_paths(model_id)
            if assets is not None:
                try:
                    return (
                        LivePortraitOnnxTransformer(
                            appearance_feature_extractor_path=assets["appearance_feature_extractor.onnx"],
                            motion_extractor_path=assets["motion_extractor.onnx"],
                            warping_spade_path=assets["warping_spade.onnx"],
                            provider=provider,
                            intra_threads=self.settings.onnx_intra_threads,
                        ),
                        model_id,
                    )
                except KeyError as exc:
                    raise ModelNotConfiguredError(
                        f"The installed '{model_id}' model is missing an expected file: {exc}. "
                        "Remove and reinstall it from the Model Status panel."
                    ) from exc

        raise ModelNotConfiguredError(NO_MODEL_CONFIGURED_MESSAGE)

    def load(self, provider: str = "auto") -> dict[str, Any]:
        with self._lock:
            if self._model is not None:
                return self.status()
            try:
                model, model_id = self._build_model(provider)
                model.load_model()
            except (ModelNotConfiguredError, RuntimeError, ValueError, MemoryError) as exc:
                self._last_error = str(exc)
                raise ModelNotConfiguredError(str(exc)) from exc
            self._model = model
            self._active_model_id = model_id
            self._last_error = None
            return self.status()

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                self._model.unload_model()
            self._model = None
            self._active_model_id = None

    def prepare_source(self, aligned_source: np.ndarray) -> Any | None:
        with self._lock:
            return self._model.prepare_source(aligned_source) if self._model is not None else None

    def transform(self, representation: Any, target_face: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No face-transformation model is loaded.")
            try:
                return self._model.transform(representation, target_face)
            except MemoryError as exc:
                raise RuntimeError(
                    "Ran out of memory while transforming this frame. Try a lower processing "
                    "resolution or disable transformation to free RAM."
                ) from exc

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": self._model is not None,
                "name": self._model.name if self._model else "No model loaded",
                "device": self._model.device if self._model else self._runtime_device(),
                "provider": getattr(self._model, "_provider", None) if self._model else None,
                "model_id": self._active_model_id,
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
