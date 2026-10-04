"""Choose landmark observations when available, otherwise fall back to Haar boxes."""
from __future__ import annotations

import numpy as np

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
        if self.landmark_provider.available:
            observations = self.landmark_provider.analyze(rgb)
            if observations:
                return observations
        return self.detector.detect(rgb)
