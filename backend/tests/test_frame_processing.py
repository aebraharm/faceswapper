from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.alignment.face_aligner import FaceAligner
from app.blending.compositor import FaceCompositor
from app.camera.frame_processor import FrameProcessor
from app.camera.session import CameraSession
from app.detection.analyzer import FaceAnalyzer
from app.transformation.liveportrait_transformer import LivePortraitOnnxTransformer
from fixtures.tiny_liveportrait import build_tiny_liveportrait_bundle


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
    # Deliberately non-black, channel-distinct camera content. It makes a black
    # backend JPEG or a BGR/RGB regression observable after a full encode/decode.
    y, x = np.indices((240, 320))
    frame_bgr = np.dstack(((x % 200) + 20, (y % 180) + 30, ((x + y) % 160) + 60)).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    assert ok
    return encoded.tobytes()


def _no_face_processor(camera: CameraSession, **kwargs) -> FrameProcessor:
    detector = type("NoFaceDetector", (), {"detect": lambda self, image: []})()
    return FrameProcessor(
        FaceAnalyzer(
            detector=detector,
            landmark_provider=type("NoLandmarks", (), {"available": False, "analyze": lambda self, image: []})(),
        ),
        FaceAligner(),
        FaceCompositor(),
        TransformerStub(),
        SourceStub(),
        camera,
        **kwargs,
    )


def test_frame_processor_returns_frame_and_telemetry_without_a_camera_or_model():
    camera = CameraSession()
    camera.start()
    processor = _no_face_processor(camera)
    output, stats = processor.process_jpeg(make_jpeg())
    decoded = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert stats["type"] == "stats"
    assert stats["face_count"] == 0
    assert stats["transformation_active"] is False
    assert stats["camera_resolution"] == "320 × 240"
    # Camera capture → RGB/BGR conversion → JPEG serialization preserves a
    # visible non-black frame even before a model or face is available.
    assert np.any(decoded != 0)
    assert decoded[0, 0].mean() > 10


def test_opt_in_frame_diagnostics_trace_capture_and_jpeg_decode_boundaries(caplog):
    camera = CameraSession()
    camera.start()
    processor = _no_face_processor(camera, diagnostics_enabled=True, diagnostics_interval_ms=60_000)
    with caplog.at_level("WARNING"):
        output, stats = processor.process_jpeg(make_jpeg())

    trace = stats["frame_diagnostics"]
    assert trace["first_invalid"] is None
    stages = {entry["stage"]: entry for entry in trace["stages"]}
    assert stages["camera_capture.bgr"]["channel_order"] == "BGR"
    assert stages["preprocessing.rgb"]["channel_order"] == "RGB"
    assert stages["face_analyzer.start"]["landmarks_available"] is False
    assert stages["target_tracker.selection"]["detected_faces"] == []
    assert stages["serialization.jpeg"]["decode_valid"] is True
    assert stages["serialization.jpeg"]["decoded_dimensions"] == {"height": 240, "width": 320, "channels": 3}
    assert stages["serialization.jpeg.decoded_bgr"]["all_zero"] is False
    assert output.startswith(b"\xff\xd8") and output.endswith(b"\xff\xd9")
    logged = [json.loads(record.message) for record in caplog.records if record.name == "app.camera.frame_diagnostics"]
    assert logged[-1]["type"] == "frame_diagnostics"


def test_real_liveportrait_512_pipeline_preserves_a_nonblack_decodable_camera_frame(tmp_path):
    """Regression: trace a real ONNX adapter output through every backend image boundary."""
    from app.detection.types import BoundingBox, FaceObservation

    bundle = build_tiny_liveportrait_bundle(tmp_path, warping_output_size=512)
    liveportrait = LivePortraitOnnxTransformer(
        bundle["appearance_feature_extractor.onnx"],
        bundle["motion_extractor.onnx"],
        bundle["warping_spade.onnx"],
        provider="CPUExecutionProvider",
    )
    liveportrait.load_model()
    try:
        source_representation = liveportrait.prepare_source(np.full((256, 256, 3), (80, 130, 210), dtype=np.uint8))

        class OneFaceAnalyzer:
            def analyze(self, image):
                return [FaceObservation(BoundingBox(96, 46, 100, 130))]

        class LivePortraitManager:
            def status(self):
                return {"loaded": True, "name": "LivePortrait test", "device": "CPU", "provider": "CPUExecutionProvider"}

            def transform(self, source, target):
                return liveportrait.transform(source, target)

        class ReadySource:
            def status(self):
                return {"model_ready": True}

            def representation(self):
                return source_representation

        camera = CameraSession()
        camera.start()
        camera.configure(transform_enabled=True, intensity=1.0)
        processor = FrameProcessor(
            OneFaceAnalyzer(),
            FaceAligner(),
            FaceCompositor(),
            LivePortraitManager(),
            ReadySource(),
            camera,
            diagnostics_enabled=True,
            diagnostics_interval_ms=60_000,
        )
        payload, stats = processor.process_jpeg(make_jpeg())
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)

        assert stats["transformation_active"] is True
        assert decoded is not None and decoded.shape == (240, 320, 3)
        assert np.any(decoded != 0)
        trace = stats["frame_diagnostics"]
        assert trace["first_invalid"] is None
        stages = {entry["stage"]: entry for entry in trace["stages"]}
        native = stages["liveportrait.warping_spade.native_output_nchw"]
        assert native["dimensions"] == {"batch": 1, "channels": 3, "height": 512, "width": 512}
        assert native["finite"] is True
        assert native["all_zero"] is False
        assert native["channels"]
        assert stages["compositor.resampled_rgb_256"]["dimensions"] == {"height": 256, "width": 256, "channels": 3}
        assert stages["serialization.jpeg"]["decode_valid"] is True
    finally:
        liveportrait.unload_model()


def test_frame_processor_handles_invalid_camera_bytes_without_crashing():
    camera = CameraSession()
    processor = _no_face_processor(camera)
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
            # Catalog LivePortrait renders 512×512 even though FRAME's aligned
            # model inputs and inverse-compositing coordinates are 256×256.
            return np.full((512, 512, 3), (230, 75, 60), dtype=np.uint8)

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
    # The inverse face blend cannot erase the non-black camera background.
    assert decoded[0, 0].mean() > 10
