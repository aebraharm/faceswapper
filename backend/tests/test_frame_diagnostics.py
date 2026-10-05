from __future__ import annotations

import json

import numpy as np

from app.camera.frame_diagnostics import FrameDiagnostics, configure_diagnostics_log, record_event
from app.config import Settings
from app.detection.face_detector import OpenCVFaceDetector


class Cascade:
    def detectMultiScale(self, image, **kwargs):
        return np.asarray([[12, 18, 36, 42]], dtype=np.int32)


def test_frame_diagnostics_file_is_json_lines_and_rotates_in_the_configured_location(tmp_path):
    path = tmp_path / "logs" / "trace.jsonl"
    configure_diagnostics_log(path)
    try:
        trace = FrameDiagnostics()
        with trace.activate():
            record_event("test.detector", {"detections": 1})
        trace.log()
        payload = json.loads(path.read_text(encoding="utf-8").strip())
        assert payload["type"] == "frame_diagnostics"
        assert isinstance(payload["trace_id"], str) and payload["trace_id"]
        assert payload["stages"] == [{"stage": "test.detector", "kind": "event", "detections": 1}]
    finally:
        configure_diagnostics_log(None)


def test_settings_resolves_packaged_user_data_diagnostics_log_path(tmp_path):
    settings = Settings(frame_diagnostics=True, models_dir=str(tmp_path / "FRAME" / "models"))
    assert settings.resolved_frame_diagnostics_log_file() == tmp_path / "FRAME" / "logs" / "frame-diagnostics.jsonl"


def test_opencv_haar_trace_reports_its_parameters_and_detected_box():
    detector = object.__new__(OpenCVFaceDetector)
    detector._cascade = Cascade()
    detector._cascade_path = "C:/FRAME/cv2/haarcascade_frontalface_default.xml"
    detector._scale_factor = 1.12
    detector._min_neighbors = 5
    trace = FrameDiagnostics()
    with trace.activate():
        faces = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
    stages = {stage["stage"]: stage for stage in trace.report()["stages"]}
    assert faces[0].bbox == (12, 18, 36, 42)
    assert stages["detector.opencv_haar"]["cascade_path"].endswith("haarcascade_frontalface_default.xml")
    assert stages["detector.opencv_haar"]["input_dimensions"] == {"height": 80, "width": 100, "channels": 3}
    assert stages["detector.opencv_haar"]["detections"] == [
        {"x": 12, "y": 18, "width": 36, "height": 42, "confidence": None}
    ]
    assert stages["detector.opencv_haar"]["confidence_available"] is False
