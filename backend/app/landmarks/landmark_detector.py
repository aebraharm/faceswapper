"""MediaPipe Face Mesh adapter. OpenCV detection remains available without MediaPipe."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from app.detection.types import BoundingBox, FaceObservation


class LandmarkProvider(Protocol):
    available: bool

    def analyze(self, rgb: np.ndarray) -> list[FaceObservation]: ...


class MediaPipeLandmarkDetector:
    # Canonical points, sorted left-to-right in the image by FaceAligner.
    _FIVE_POINT_INDICES = (33, 263, 1, 61, 291)

    def __init__(self, max_num_faces: int = 4, task_model_path: Path | None = None) -> None:
        self.available = False
        self._mesh = None
        self._tasks = False
        self._lock = threading.Lock()
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=0.55,
                )
                self.available = True
            elif task_model_path and task_model_path.is_file():
                # Newer MediaPipe releases expose Tasks but no legacy solutions API.
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                options = vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(task_model_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=max_num_faces,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._mesh = vision.FaceLandmarker.create_from_options(options)
                self._tasks = True
                self.available = True
        except (ImportError, AttributeError, RuntimeError, ValueError):
            # MediaPipe is optional at runtime; OpenCV detection is the fallback.
            self._mesh = None
            self.available = False

    def analyze(self, rgb: np.ndarray) -> list[FaceObservation]:
        if not self.available or self._mesh is None or rgb is None or rgb.size == 0:
            return []
        h, w = rgb.shape[:2]
        with self._lock:
            if self._tasks:
                import mediapipe as mp

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                results = self._mesh.detect(image)
                landmark_sets = results.face_landmarks or []
            else:
                results = self._mesh.process(np.ascontiguousarray(rgb))
                landmark_sets = [face.landmark for face in (results.multi_face_landmarks or [])]
        observations: list[FaceObservation] = []
        for landmarks in landmark_sets:
            points = np.asarray([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)
            if len(points) <= max(self._FIVE_POINT_INDICES):
                continue
            x1 = max(0, int(np.floor(points[:, 0].min())))
            y1 = max(0, int(np.floor(points[:, 1].min())))
            x2 = min(w, int(np.ceil(points[:, 0].max())))
            y2 = min(h, int(np.ceil(points[:, 1].max())))
            if x2 <= x1 or y2 <= y1:
                continue
            five = points[list(self._FIVE_POINT_INDICES)].copy()
            # Resolve left/right by image position so the alignment template remains stable.
            eye_order = np.argsort(five[[0, 1], 0])
            mouth_order = np.argsort(five[[3, 4], 0])
            five[[0, 1]] = five[[0, 1]][eye_order]
            five[[3, 4]] = five[[3, 4]][mouth_order]
            observations.append(
                FaceObservation(BoundingBox(x1, y1, x2 - x1, y2 - y1), five, confidence=1.0)
            )
        return observations

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None
            self.available = False
