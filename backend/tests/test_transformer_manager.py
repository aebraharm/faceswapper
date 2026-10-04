from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.transformation.base import ModelNotConfiguredError
from app.transformation.manager import NO_MODEL_CONFIGURED_MESSAGE, TransformerManager
from app.transformation.model_manager import ModelInstallManager
from fixtures.tiny_liveportrait import build_tiny_liveportrait_bundle


def test_no_model_configured_raises_actionable_error_mentioning_readme_and_panel(tmp_path):
    settings = Settings(source_encoder_model=None, face_transformer_model=None, models_dir=str(tmp_path))
    manager = TransformerManager(settings, ModelInstallManager(tmp_path))
    with pytest.raises(ModelNotConfiguredError) as excinfo:
        manager.load()
    assert "models/README.md" in str(excinfo.value)
    assert NO_MODEL_CONFIGURED_MESSAGE in str(excinfo.value)
    status = manager.status()
    assert status["loaded"] is False
    assert status["error"]


def test_installed_catalog_model_is_discovered_and_loaded(tmp_path):
    build_tiny_liveportrait_bundle(tmp_path / "liveportrait-v1")
    settings = Settings(source_encoder_model=None, face_transformer_model=None, models_dir=str(tmp_path))
    store = ModelInstallManager(tmp_path)
    manager = TransformerManager(settings, store)
    status = manager.load(provider="CPUExecutionProvider")
    try:
        assert status["loaded"] is True
        assert status["model_id"] == "liveportrait-v1"
        assert manager.loaded is True
    finally:
        manager.unload()


def test_transform_wraps_out_of_memory_into_a_recoverable_runtime_error(tmp_path, monkeypatch):
    build_tiny_liveportrait_bundle(tmp_path / "liveportrait-v1")
    settings = Settings(source_encoder_model=None, face_transformer_model=None, models_dir=str(tmp_path))
    manager = TransformerManager(settings, ModelInstallManager(tmp_path))
    manager.load(provider="CPUExecutionProvider")
    try:
        rep = manager.prepare_source(np.zeros((256, 256, 3), dtype=np.uint8))

        def boom(*args, **kwargs):
            raise MemoryError("arena exhausted")

        monkeypatch.setattr(manager._model._warp, "run", boom)
        # Per the brief: the websocket loop only catches (ValueError, RuntimeError) --
        # a bare MemoryError here would otherwise crash the camera stream.
        with pytest.raises(RuntimeError) as excinfo:
            manager.transform(rep, np.zeros((256, 256, 3), dtype=np.uint8))
        assert not isinstance(excinfo.value, MemoryError)
    finally:
        manager.unload()


def test_transform_without_a_loaded_model_raises_runtime_error_not_attribute_error(tmp_path):
    settings = Settings(source_encoder_model=None, face_transformer_model=None, models_dir=str(tmp_path))
    manager = TransformerManager(settings, ModelInstallManager(tmp_path))
    with pytest.raises(RuntimeError):
        manager.transform(None, np.zeros((256, 256, 3), dtype=np.uint8))


def test_bring_your_own_onnx_path_still_takes_precedence_over_the_catalog(tmp_path):
    # A BYO path that points nowhere should still surface the BYO-specific failure
    # rather than silently falling back to the catalog model.
    settings = Settings(
        source_encoder_model=str(tmp_path / "missing_encoder.onnx"),
        face_transformer_model=str(tmp_path / "missing_transformer.onnx"),
        models_dir=str(tmp_path),
    )
    build_tiny_liveportrait_bundle(tmp_path / "liveportrait-v1")
    manager = TransformerManager(settings, ModelInstallManager(tmp_path))
    with pytest.raises(ModelNotConfiguredError):
        manager.load()
    status = manager.status()
    assert status["model_id"] is None
