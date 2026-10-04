"""Small temporal tracker that smooths detector jitter without changing identity."""
from __future__ import annotations

from dataclasses import replace

from app.detection.types import BoundingBox, FaceObservation


class TargetFaceTracker:
    def __init__(self, smoothing: float = 0.35) -> None:
        self.smoothing = smoothing
        self._last: FaceObservation | None = None
        self._requested_index: int | None = None

    def select(self, faces: list[FaceObservation], selected_index: int | None) -> tuple[int | None, FaceObservation | None]:
        if not faces:
            self._last = None
            self._requested_index = None
            return None, None
        if len(faces) == 1:
            index = 0
        elif selected_index is not None and 0 <= selected_index < len(faces):
            if self._last is None or selected_index != self._requested_index:
                index = selected_index
            else:
                previous = self._last.bbox
                prev_center = (previous.x + previous.width / 2, previous.y + previous.height / 2)
                index = min(
                    range(len(faces)),
                    key=lambda i: (
                        (faces[i].bbox.x + faces[i].bbox.width / 2 - prev_center[0]) ** 2
                        + (faces[i].bbox.y + faces[i].bbox.height / 2 - prev_center[1]) ** 2
                    ),
                )
        else:
            self._last = None
            self._requested_index = None
            return None, None
        self._requested_index = selected_index
        current = faces[index]
        if self._last is None:
            tracked = current
        else:
            blend = self.smoothing
            old = self._last.bbox
            new = current.bbox
            values = [int((1.0 - blend) * a + blend * b) for a, b in zip(old, new)]
            smoothed_landmarks = current.landmarks5
            if current.landmarks5 is not None and self._last.landmarks5 is not None:
                smoothed_landmarks = (1.0 - blend) * self._last.landmarks5 + blend * current.landmarks5
            tracked = replace(current, bbox=BoundingBox(*values), landmarks5=smoothed_landmarks)
        self._last = tracked
        return index, tracked
