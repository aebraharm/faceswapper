from __future__ import annotations

import json

import numpy as np
import pytest

from app.alignment.face_aligner import FaceAligner
from app.detection.analyzer import FaceAnalyzer
from app.detection.face_detector import OpenCVFaceDetector
from app.detection.types import BoundingBox, FaceObservation
from app.source_faces import SourceFaceStore
from app.transformation.base import FaceTransformer, ModelNotConfiguredError
from app.transformation.manager import TransformerManager
from app.config import Settings
from conftest import FakeAnalyzer


class CountingTransformer(FaceTransformer):
    name = "test model"
    device = "CPU"

    def __init__(self):
        self.prepared = 0
        self.transformed = 0

    def load_model(self):
        pass

    def prepare_source(self, source_face_rgb):
        self.prepared += 1
        return np.mean(source_face_rgb, axis=(0, 1)).astype(np.float32)

    def transform(self, source_representation, target_face_rgb):
        self.transformed += 1
        return target_face_rgb.copy()

    def unload_model(self):
        pass


class ManagerStub:
    def __init__(self, transformer=None):
        self.model = transformer

    def prepare_source(self, image):
        return self.model.prepare_source(image) if self.model else None

    def transform(self, rep, target):
        return self.model.transform(rep, target)

    @property
    def loaded(self):
        return self.model is not None


class NoOpModel(FaceTransformer):
    def load_model(self): pass
    def prepare_source(self, source_face_rgb): return np.ones((1, 8), dtype=np.float32)
    def transform(self, source_representation, target_face_rgb): return target_face_rgb
    def unload_model(self): pass


class FallbackCascade:
    def detectMultiScale(self, image, **kwargs):
        return np.asarray([[20, 20, 100, 120]], dtype=np.int32)


class UnavailableLandmarks:
    available = False

    def analyze(self, image):
        return []


def test_source_face_upload_prepares_opencv_fallback_observation(sample_rgb):
    detector = object.__new__(OpenCVFaceDetector)
    detector._cascade = FallbackCascade()
    detector._scale_factor = 1.12
    detector._min_neighbors = 5
    model = CountingTransformer()
    store = SourceFaceStore(
        FaceAnalyzer(detector, UnavailableLandmarks()),
        FaceAligner(),
        ManagerStub(model),
    )

    status = store.upload(sample_rgb)

    assert status["selected_face_index"] == 0
    assert status["faces"] == [{"x": 20, "y": 20, "width": 100, "height": 120, "confidence": None}]
    assert status["model_ready"] is True
    assert model.prepared == 1


def test_source_upload_emits_one_opt_in_comparison_trace(sample_rgb, caplog):
    model = CountingTransformer()
    store = SourceFaceStore(FakeAnalyzer(), FaceAligner(), ManagerStub(model), diagnostics_enabled=True)
    with caplog.at_level("WARNING"):
        store.upload(sample_rgb)
    traces = [json.loads(record.message) for record in caplog.records if record.name == "app.camera.frame_diagnostics"]
    assert traces[-1]["type"] == "source_upload_diagnostics"
    assert traces[-1]["trace_kind"] == "source_upload"
    stages = {entry["stage"]: entry for entry in traces[-1]["stages"]}
    assert stages["source_upload.rgb"]["channel_order"] == "RGB"
    assert stages["source_upload.detection_result"]["face_count"] == 1
    assert stages["alignment"]["method"] == "bounding_box"


def test_source_face_multiple_selection_and_cached_representation(sample_rgb):
    faces = [
        FaceObservation(BoundingBox(15, 20, 50, 60)),
        FaceObservation(BoundingBox(95, 20, 55, 65)),
    ]
    analyzer = FakeAnalyzer(faces)
    model = CountingTransformer()
    manager = ManagerStub(model)
    store = SourceFaceStore(analyzer, FaceAligner(), manager)
    status = store.upload(sample_rgb)
    assert status["face_count"] == 2
    assert status["selected_face_index"] is None
    assert not status["ready"]

    selected = store.select_face(1)
    assert selected["selected_face_index"] == 1
    assert selected["model_ready"]
    identity = store.representation()
    assert model.prepared == 1
    assert np.array_equal(identity, store.representation())
    manager.transform(identity, np.zeros((256, 256, 3), np.uint8))
    assert model.prepared == 1  # source features are not recomputed per target frame
    assert model.transformed == 1


def test_single_face_is_selected_automatically_and_remove_clears_state(sample_rgb):
    model = CountingTransformer()
    store = SourceFaceStore(FakeAnalyzer(), FaceAligner(), ManagerStub(model))
    status = store.upload(sample_rgb)
    assert status["selected_face_index"] == 0
    store.clear()
    assert store.status()["uploaded"] is False
    assert store.representation() is None


def test_transformer_contract_and_missing_model_error():
    assert issubclass(FaceTransformer, object)
    manager = TransformerManager(Settings(source_encoder_model=None, face_transformer_model=None))
    with pytest.raises(ModelNotConfiguredError, match="No licensed model bundle"):
        manager.load()
    assert not manager.loaded
    assert manager.status()["loaded"] is False


def test_failed_replacement_keeps_previous_valid_source(sample_rgb):
    analyzer = FakeAnalyzer()
    store = SourceFaceStore(analyzer, FaceAligner(), ManagerStub())
    store.upload(sample_rgb)
    analyzer.faces = []
    with pytest.raises(ValueError, match="No face"):
        store.upload(np.zeros_like(sample_rgb))
    status = store.status()
    assert status["uploaded"] is True
    assert status["selected_face_index"] == 0
