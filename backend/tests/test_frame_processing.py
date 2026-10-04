from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.alignment.face_aligner import FaceAligner
from app.blending.compositor import FaceCompositor
from app.camera.frame_processor import FrameProcessor
from app.camera.session import CameraSession
from app.detection.analyzer import FaceAnalyzer
from conftest import FakeAnalyzer


class TransformerStub:
    loaded = False
    def status(self):
        return {"loaded": False, "name": "No model loaded", "device": "CPU", "provider": None}
    def transform(self, source, target):
        raise AssertionError("transform must not run without a loaded model")


class SourceStub:
    def status(self):
        return {"model_ready": False}
    def representation(self):
        return None


def make_jpeg():
    ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_frame_processor_returns_frame_and_telemetry_without_a_camera_or_model():
    camera = CameraSession()
    camera.start()
    detector = type("NoFaceDetector", (), {"detect": lambda self, image: []})()
    processor = FrameProcessor(
        FaceAnalyzer(detector=detector, landmark_provider=type("NoLandmarks", (), {"available": False, "analyze": lambda self, image: []})()),
        FaceAligner(), FaceCompositor(), TransformerStub(), SourceStub(), camera,
    )
    output, stats = processor.process_jpeg(make_jpeg())
    decoded = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert stats["type"] == "stats"
    assert stats["face_count"] == 0
    assert stats["transformation_active"] is False
    assert stats["camera_resolution"] == "320 × 240"


def test_frame_processor_handles_invalid_camera_bytes_without_crashing():
    camera = CameraSession()
    detector = type("NoFaceDetector", (), {"detect": lambda self, image: []})()
    processor = FrameProcessor(
        FaceAnalyzer(detector=detector, landmark_provider=type("NoLandmarks", (), {"available": False, "analyze": lambda self, image: []})()),
        FaceAligner(), FaceCompositor(), TransformerStub(), SourceStub(), camera,
    )
    with pytest.raises(ValueError, match="valid JPEG"):
        processor.process_jpeg(b"not a jpeg")


def test_frame_processor_applies_transform_and_blend_only_when_prerequisites_are_ready():
    from app.detection.types import BoundingBox, FaceObservation

    target = FaceObservation(BoundingBox(100, 50, 80, 110))

    class OneFaceAnalyzer:
        def analyze(self, image):
            return [target]

    class LoadedTransformer:
        loaded = True
        def status(self):
            return {"loaded": True, "name": "mock", "device": "CPU", "provider": "CPUExecutionProvider"}
        def transform(self, source, aligned_target):
            return np.full((256, 256, 3), (230, 75, 60), dtype=np.uint8)

    class ReadySource:
        def status(self):
            return {"model_ready": True}
        def representation(self):
            return np.ones((1, 8), dtype=np.float32)

    camera = CameraSession()
    camera.start()
    camera.configure(transform_enabled=True, intensity=1.0)
    processor = FrameProcessor(
        OneFaceAnalyzer(), FaceAligner(), FaceCompositor(), LoadedTransformer(), ReadySource(), camera,
    )
    output, stats = processor.process_jpeg(make_jpeg())
    decoded = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR)
    assert stats["transformation_active"] is True
    assert decoded[100, 140].mean() > 15
