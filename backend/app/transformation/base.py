"""Stable model contract: cache source identity once, transform each target face."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ModelNotConfiguredError(RuntimeError):
    pass


class FaceTransformer(ABC):
    """Model boundary used by the camera pipeline.

    ``source_representation`` is prepared on upload (or once when a model is loaded),
    never recomputed for each webcam frame.
    """

    name = "face-transformer"
    device = "unknown"

    @abstractmethod
    def load_model(self) -> None:
        """Load weights and warm up inference resources."""

    @abstractmethod
    def prepare_source(self, source_face_rgb: np.ndarray) -> Any:
        """Extract/cache source identity features from one aligned source face."""

    @abstractmethod
    def transform(self, source_representation: Any, target_face_rgb: np.ndarray) -> np.ndarray:
        """Return a target-pose/expression-conditioned transformed face crop."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model sessions and device memory."""
