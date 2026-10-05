"""Similarity alignment shared by source-face preparation and target-frame processing."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.camera.frame_diagnostics import record_event, record_image
from app.detection.types import FaceObservation


@dataclass(frozen=True)
class AlignedFace:
    image: np.ndarray
    # Matrices map original -> canonical and canonical -> original, respectively.
    forward_matrix: np.ndarray
    inverse_matrix: np.ndarray
    output_size: int


class FaceAligner:
    def __init__(self, output_size: int = 256, padding: float = 1.55) -> None:
        self.output_size = output_size
        self.padding = padding

    def align(self, rgb: np.ndarray, face: FaceObservation, output_size: int | None = None) -> AlignedFace:
        size = output_size or self.output_size
        if rgb is None or rgb.size == 0:
            raise ValueError("Cannot align an empty image.")
        matrix = self._landmark_matrix(face, size)
        method = "landmarks5" if matrix is not None else "bounding_box"
        if matrix is None:
            matrix = self._box_matrix(face, rgb.shape[1], rgb.shape[0], size)
        aligned = cv2.warpAffine(
            rgb,
            matrix,
            (size, size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        inverse = cv2.invertAffineTransform(matrix)
        record_image("alignment.output_rgb", aligned, channel_order="RGB")
        record_event(
            "alignment",
            {
                "method": method,
                "input_dimensions": {"height": int(rgb.shape[0]), "width": int(rgb.shape[1]), "channels": int(rgb.shape[2])},
                "bbox": face.as_dict(),
                "landmarks5_shape": list(np.asarray(face.landmarks5).shape) if face.landmarks5 is not None else None,
                "forward_matrix": matrix.astype(np.float32),
                "inverse_matrix": inverse.astype(np.float32),
                "output_size": int(size),
                "matrix_finite": bool(np.isfinite(matrix).all() and np.isfinite(inverse).all()),
            },
        )
        return AlignedFace(aligned, matrix.astype(np.float32), inverse.astype(np.float32), size)

    @staticmethod
    def _landmark_matrix(face: FaceObservation, size: int) -> np.ndarray | None:
        if face.landmarks5 is None or np.asarray(face.landmarks5).shape != (5, 2):
            return None
        source = np.asarray(face.landmarks5, dtype=np.float32)
        scale = float(size)
        destination = np.asarray(
            [
                (0.35 * scale, 0.39 * scale),
                (0.65 * scale, 0.39 * scale),
                (0.50 * scale, 0.57 * scale),
                (0.39 * scale, 0.73 * scale),
                (0.61 * scale, 0.73 * scale),
            ],
            dtype=np.float32,
        )
        matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            return None
        return matrix.astype(np.float32)

    def _box_matrix(self, face: FaceObservation, width: int, height: int, size: int) -> np.ndarray:
        x, y, box_w, box_h = face.bbox
        side = max(float(box_w), float(box_h), 1.0) * self.padding
        center_x = x + box_w / 2.0
        center_y = y + box_h / 2.0
        left = center_x - side / 2.0
        top = center_y - side / 2.0
        scale = size / side
        # Reflection padding in warpAffine handles crops which cross the image boundary.
        return np.asarray(
            [[scale, 0.0, -left * scale], [0.0, scale, -top * scale]], dtype=np.float32
        )
