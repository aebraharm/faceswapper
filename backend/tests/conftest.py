from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.detection.types import BoundingBox, FaceObservation


class FakeAnalyzer:
    landmarks_available = True

    def __init__(self, faces=None):
        self.faces = [FaceObservation(BoundingBox(50, 40, 100, 120))] if faces is None else faces
        self.calls = 0

    def analyze(self, image):
        self.calls += 1
        return list(self.faces)


def png_bytes(width=160, height=160, color=(180, 120, 90)):
    image = Image.new("RGB", (width, height), color)
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


@pytest.fixture
def sample_rgb():
    return np.full((180, 180, 3), 127, dtype=np.uint8)
