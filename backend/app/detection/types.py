from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class BoundingBox(NamedTuple):
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class FaceObservation:
    """One face in image pixel coordinates; identity and target are kept separate."""

    bbox: BoundingBox
    landmarks5: np.ndarray | None = None
    confidence: float | None = None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "x": int(self.bbox.x),
            "y": int(self.bbox.y),
            "width": int(self.bbox.width),
            "height": int(self.bbox.height),
            "confidence": self.confidence,
        }
