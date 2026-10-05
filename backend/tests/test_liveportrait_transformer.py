from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from app.transformation.base import ModelNotConfiguredError
from app.transformation.liveportrait_transformer import LivePortraitOnnxTransformer
from fixtures.tiny_liveportrait import build_tiny_liveportrait_bundle


@pytest.fixture()
def tiny_bundle(tmp_path: Path) -> dict[str, Path]:
    return build_tiny_liveportrait_bundle(tmp_path)


def _model(bundle: dict[str, Path]) -> LivePortraitOnnxTransformer:
    return LivePortraitOnnxTransformer(
        bundle["appearance_feature_extractor.onnx"],
        bundle["motion_extractor.onnx"],
        bundle["warping_spade.onnx"],
        provider="CPUExecutionProvider",
    )


def test_load_warms_up_and_reports_device(tiny_bundle):
    model = _model(tiny_bundle)
    model.load_model()
    try:
        assert model.device == "CPU"
        assert model._provider == "CPUExecutionProvider"
    finally:
        model.unload_model()


def test_prepare_source_caches_a_compact_representation_without_pixels(tiny_bundle):
    model = _model(tiny_bundle)
    model.load_model()
    try:
        source = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        rep = model.prepare_source(source)
        assert rep.feature_3d.shape == (1, 32, 16, 64, 64)
        assert rep.x_s.shape == (1, 21, 3)
        assert rep.kp_canonical.shape == (1, 21, 3)
        assert rep.scale.shape == (1, 1)
        # Only derived numeric arrays are retained -- no field is (or aliases) the raw pixels.
        for field in rep:
            assert isinstance(field, np.ndarray)
            assert field.dtype != np.uint8
            assert field.shape != source.shape
            assert not np.shares_memory(field, source)
    finally:
        model.unload_model()


def test_transform_output_is_a_valid_rgb_face_crop_and_tracks_the_target(tiny_bundle):
    model = _model(tiny_bundle)
    model.load_model()
    try:
        source = np.full((256, 256, 3), 90, dtype=np.uint8)
        rep = model.prepare_source(source)
        dark_target = np.zeros((256, 256, 3), dtype=np.uint8)
        bright_target = np.full((256, 256, 3), 255, dtype=np.uint8)
        out_dark = model.transform(rep, dark_target)
        out_bright = model.transform(rep, bright_target)
        assert out_dark.shape == (256, 256, 3)
        assert out_dark.dtype == np.uint8
        # Changing the live/driving frame changes the output -- pose/expression tracking.
        assert out_dark.mean() != out_bright.mean()
    finally:
        model.unload_model()


def test_transform_preserves_liveportrait_native_512_render_output(tmp_path: Path):
    bundle = build_tiny_liveportrait_bundle(tmp_path, warping_output_size=512)
    model = _model(bundle)
    model.load_model()
    try:
        assert model._warp.get_outputs()[0].shape == [1, 3, 512, 512]
        assert model._warp_output_size == (512, 512)
        source = np.full((256, 256, 3), 90, dtype=np.uint8)
        output = model.transform(model.prepare_source(source), source)
        assert output.shape == (512, 512, 3)
        assert output.dtype == np.uint8
    finally:
        model.unload_model()


def test_prepare_source_is_cached_and_not_recomputed_per_frame(tiny_bundle, monkeypatch):
    model = _model(tiny_bundle)
    model.load_model()
    try:
        call_count = {"n": 0}
        original_run = model._appearance.run

        def counting_run(*args, **kwargs):
            call_count["n"] += 1
            return original_run(*args, **kwargs)

        monkeypatch.setattr(model._appearance, "run", counting_run)
        source = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        rep = model.prepare_source(source)
        calls_after_prepare = call_count["n"]
        assert calls_after_prepare >= 1
        for _ in range(5):
            model.transform(rep, np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        # transform() must never call the (expensive) appearance extractor again.
        assert call_count["n"] == calls_after_prepare
    finally:
        model.unload_model()


def test_unload_releases_sessions_and_requires_reload_before_use(tiny_bundle):
    model = _model(tiny_bundle)
    model.load_model()
    source = np.zeros((256, 256, 3), dtype=np.uint8)
    rep = model.prepare_source(source)
    model.unload_model()
    assert model.device == "not loaded"
    with pytest.raises(RuntimeError):
        model.prepare_source(source)
    with pytest.raises(RuntimeError):
        model.transform(rep, source)


def test_missing_model_file_raises_actionable_configuration_error(tiny_bundle, tmp_path):
    missing = tmp_path / "does-not-exist.onnx"
    model = LivePortraitOnnxTransformer(
        tiny_bundle["appearance_feature_extractor.onnx"], missing, tiny_bundle["warping_spade.onnx"],
        provider="CPUExecutionProvider",
    )
    with pytest.raises(ModelNotConfiguredError, match="missing"):
        model.load_model()


def test_corrupt_model_file_raises_actionable_configuration_error(tiny_bundle):
    corrupt = tiny_bundle["motion_extractor.onnx"]
    corrupt.write_bytes(b"not a real onnx graph")
    model = _model(tiny_bundle)
    with pytest.raises(ModelNotConfiguredError):
        model.load_model()


def test_unsupported_provider_raises_actionable_configuration_error(tiny_bundle):
    model = LivePortraitOnnxTransformer(
        tiny_bundle["appearance_feature_extractor.onnx"],
        tiny_bundle["motion_extractor.onnx"],
        tiny_bundle["warping_spade.onnx"],
        provider="TotallyMadeUpExecutionProvider",
    )
    with pytest.raises(ModelNotConfiguredError, match="unavailable"):
        model.load_model()


def test_out_of_memory_during_load_is_reported_as_a_friendly_configuration_error(tiny_bundle, monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError("failed to allocate memory arena")

    monkeypatch.setattr(ort, "InferenceSession", boom)
    model = _model(tiny_bundle)
    with pytest.raises(ModelNotConfiguredError, match="(?i)memory"):
        model.load_model()


def test_out_of_memory_during_transform_propagates_so_the_manager_can_degrade_gracefully(tiny_bundle, monkeypatch):
    model = _model(tiny_bundle)
    model.load_model()
    try:
        rep = model.prepare_source(np.zeros((256, 256, 3), dtype=np.uint8))

        def boom(*args, **kwargs):
            raise MemoryError("arena exhausted")

        monkeypatch.setattr(model._warp, "run", boom)
        with pytest.raises(MemoryError):
            model.transform(rep, np.zeros((256, 256, 3), dtype=np.uint8))
    finally:
        model.unload_model()
