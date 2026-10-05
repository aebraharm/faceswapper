"""Choose landmark observations when available, otherwise fall back to Haar boxes."""
from __future__ import annotations

import numpy as np

from app.camera.frame_diagnostics import record_event
from app.detection.face_detector import FaceDetector, OpenCVFaceDetector
from app.detection.types import FaceObservation
from app.landmarks.landmark_detector import LandmarkProvider, MediaPipeLandmarkDetector


class FaceAnalyzer:
    def __init__(
        self,
        detector: FaceDetector | None = None,
        landmark_provider: LandmarkProvider | None = None,
    ) -> None:
        self.detector = detector or OpenCVFaceDetector()
        self.landmark_provider = landmark_provider or MediaPipeLandmarkDetector()

    @property
    def landmarks_available(self) -> bool:
        return self.landmark_provider.available

    def analyze(self, rgb: np.ndarray) -> list[FaceObservation]:
        record_event(
            "face_analyzer.start",
            {
                "landmarks_available": bool(self.landmark_provider.available),
                "landmark_provider": type(self.landmark_provider).__name__,
                "fallback_detector": type(self.detector).__name__,
            },
        )
        if self.landmark_provider.available:
            observations = self.landmark_provider.analyze(rgb)
            record_event(
                "face_analyzer.landmarks",
                {"detections": [_diagnostic_face(item) for item in observations]},
            )
            if observations:
                return observations
        observations = self.detector.detect(rgb)
        record_event(
            "face_analyzer.fallback",
            {"detections": [_diagnostic_face(item) for item in observations]},
        )
        return observations


def _diagnostic_face(face: FaceObservation) -> dict[str, object]:
    """Small opt-in record of target detection/landmark facts, never image pixels."""
    detail: dict[str, object] = dict(face.as_dict())
    landmarks = face.landmarks5
    detail["landmarks5_shape"] = list(np.asarray(landmarks).shape) if landmarks is not None else None
    detail["landmarks5_finite"] = bool(np.isfinite(landmarks).all()) if landmarks is not None else None
    detail["landmarks5"] = np.asarray(landmarks).round(3).tolist() if landmarks is not None else None
    return detail
