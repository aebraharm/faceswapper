from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.alignment.face_aligner import FaceAligner
from app.blending.compositor import FaceCompositor
from app.camera.frame_processor import FrameProcessor
from app.camera.session import CameraSession
from app.detection.types import BoundingBox, FaceObservation


def make_jpeg(size=(240, 320)):
    ok, encoded = cv2.imencode(".jpg", np.zeros((*size, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class ReadySource:
    def status(self):
        return {"model_ready": True}

    def representation(self):
        return np.ones((1, 8), dtype=np.float32)


class CountingTransformer:
    loaded = True

    def __init__(self):
        self.calls = 0

    def status(self):
        return {"loaded": True, "name": "mock", "device": "CPU", "provider": "CPUExecutionProvider"}

    def transform(self, source, aligned_target):
        self.calls += 1
        return np.full((256, 256, 3), 220, dtype=np.uint8)


class OneFaceAnalyzer:
    def __init__(self, present=True):
        self.present = present

    def analyze(self, image):
        if not self.present:
            return []
        return [FaceObservation(BoundingBox(100, 50, 80, 110))]


def _make_processor(analyzer, transformer, camera):
    return FrameProcessor(analyzer, FaceAligner(), FaceCompositor(), transformer, ReadySource(), camera)


def test_performance_mode_skips_inference_on_alternating_frames_and_holds_last_frame():
    camera = CameraSession()
    camera.start()
    camera.configure(transform_enabled=True, intensity=1.0, performance_mode="performance")
    transformer = CountingTransformer()
    processor = _make_processor(OneFaceAnalyzer(), transformer, camera)

    for _ in range(6):
        processor.process_jpeg(make_jpeg())

    # Roughly half the frames should have triggered a real model call; the model
    # must never be invoked once per frame under sustained "performance" mode.
    assert 0 < transformer.calls < 6


def test_quality_mode_never_skips_inference():
    camera = CameraSession()
    camera.start()
    camera.configure(transform_enabled=True, intensity=1.0, performance_mode="quality")
    transformer = CountingTransformer()
    processor = _make_processor(OneFaceAnalyzer(), transformer, camera)

    for _ in range(6):
        processor.process_jpeg(make_jpeg())

    assert transformer.calls == 6


def test_auto_mode_starts_skipping_only_after_sustained_low_fps():
    camera = CameraSession()
    camera.start()
    camera.configure(performance_mode="auto")

    # Before enough samples exist, "auto" never skips -- even if fps looks low.
    camera.fps = 2.5
    camera.frames_processed = 1
    assert all(camera.should_run_inference() for _ in range(4))

    # Once there's a sustained low rolling FPS and enough samples, "auto" starts
    # skipping every other frame instead of falling further behind.
    camera.frames_processed = 50
    camera.fps = 2.5
    decisions = [camera.should_run_inference() for _ in range(8)]
    assert decisions.count(False) > 0
    assert decisions.count(True) > 0

    # A healthy fps goes back to running inference on every frame.
    camera.fps = 30.0
    assert all(camera.should_run_inference() for _ in range(4))


def test_disabling_transformation_returns_instantly_to_the_original_frame():
    camera = CameraSession()
    camera.start()
    camera.configure(transform_enabled=True, intensity=1.0, performance_mode="quality")
    transformer = CountingTransformer()
    processor = _make_processor(OneFaceAnalyzer(), transformer, camera)

    output, stats = processor.process_jpeg(make_jpeg())
    assert stats["transformation_active"] is True
    decoded = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR)
    assert decoded[100, 140].mean() > 10  # transformed color was blended in

    camera.configure(transform_enabled=False)
    output2, stats2 = processor.process_jpeg(make_jpeg())
    assert stats2["transformation_active"] is False
    decoded2 = cv2.imdecode(np.frombuffer(output2, np.uint8), cv2.IMREAD_COLOR)
    # Back to (near) the raw black source frame -- no stale transformed pixels linger.
    assert decoded2[100, 140].mean() < 10


def test_face_disappearing_then_reappearing_does_not_crash_and_resumes_cleanly():
    camera = CameraSession()
    camera.start()
    camera.configure(transform_enabled=True, intensity=1.0, performance_mode="quality")
    transformer = CountingTransformer()
    analyzer = OneFaceAnalyzer(present=True)
    processor = _make_processor(analyzer, transformer, camera)

    _, stats_present = processor.process_jpeg(make_jpeg())
    assert stats_present["transformation_active"] is True

    analyzer.present = False
    _, stats_gone = processor.process_jpeg(make_jpeg())
    assert stats_gone["transformation_active"] is False
    assert stats_gone["face_count"] == 0

    analyzer.present = True
    _, stats_back = processor.process_jpeg(make_jpeg())
    assert stats_back["transformation_active"] is True
