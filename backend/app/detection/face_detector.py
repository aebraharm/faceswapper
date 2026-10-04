"""CPU face detector fallback using OpenCV's bundled Haar cascade."""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .types import FaceObservation


class FaceDetector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[FaceObservation]: ...


class OpenCVFaceDetector:
    def __init__(self, scale_factor: float = 1.12, min_neighbors: int = 5) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Unable to load OpenCV face cascade: {cascade_path}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

    def detect(self, rgb: np.ndarray) -> list[FaceObservation]:
        if rgb is None or rgb.size == 0:
            return []
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(36, 36),
        )
        return [FaceObservation(tuple(int(v) for v in box)) for box in boxes]
