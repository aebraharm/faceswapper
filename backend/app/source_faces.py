"""Ephemeral source-face state: decoded pixels and identity representation stay in RAM."""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.alignment.face_aligner import FaceAligner
from app.detection.analyzer import FaceAnalyzer
from app.detection.types import FaceObservation
from app.transformation.manager import TransformerManager


class SourceFaceStore:
    def __init__(self, analyzer: FaceAnalyzer, aligner: FaceAligner, transformer: TransformerManager) -> None:
        self.analyzer = analyzer
        self.aligner = aligner
        self.transformer = transformer
        self._lock = threading.RLock()
        self._source_rgb: np.ndarray | None = None
        self._faces: list[FaceObservation] = []
        self._selected_face_index: int | None = None
        self._aligned_source: np.ndarray | None = None
        self._representation: Any | None = None
        self._width = 0
        self._height = 0

    def upload(self, source_rgb: np.ndarray) -> dict[str, Any]:
        faces = self.analyzer.analyze(source_rgb)
        if not faces:
            raise ValueError("No face was found. Choose a clear, front-facing photo and try again.")
        # Prepare before replacing the current source so model/alignment failures leave
        # the previous valid portrait untouched.
        aligned: np.ndarray | None = None
        representation: Any | None = None
        if len(faces) == 1:
            aligned = self.aligner.align(source_rgb, faces[0], output_size=256).image.copy()
            representation = self.transformer.prepare_source(aligned)
        with self._lock:
            self._clear_pixels()
            self._source_rgb = source_rgb.copy()
            self._faces = faces
            self._width = int(source_rgb.shape[1])
            self._height = int(source_rgb.shape[0])
            self._selected_face_index = 0 if len(faces) == 1 else None
            self._aligned_source = aligned
            self._representation = representation
            return self.status()

    def select_face(self, index: int) -> dict[str, Any]:
        with self._lock:
            if self._source_rgb is None:
                raise ValueError("Upload a source photo before selecting a face.")
            if index < 0 or index >= len(self._faces):
                raise ValueError(f"Face index must be between 0 and {len(self._faces) - 1}.")
            self._select_locked(index)
            return self.status()

    def _select_locked(self, index: int) -> None:
        assert self._source_rgb is not None
        aligned = self.aligner.align(self._source_rgb, self._faces[index], output_size=256).image.copy()
        representation = self.transformer.prepare_source(aligned)
        if isinstance(self._representation, np.ndarray):
            self._representation.fill(0)
        self._selected_face_index = index
        self._aligned_source = aligned
        self._representation = representation

    def prepare_for_loaded_model(self) -> None:
        with self._lock:
            if self._aligned_source is not None:
                representation = self.transformer.prepare_source(self._aligned_source)
                if isinstance(self._representation, np.ndarray):
                    self._representation.fill(0)
                self._representation = representation

    def clear_model_representation(self) -> None:
        """Discard identity embeddings when a model is unloaded, retaining the aligned crop."""
        with self._lock:
            if isinstance(self._representation, np.ndarray):
                self._representation.fill(0)
            self._representation = None

    def representation(self) -> Any | None:
        with self._lock:
            if isinstance(self._representation, np.ndarray):
                return self._representation
            return self._representation

    def status(self) -> dict[str, Any]:
        with self._lock:
            selected = self._selected_face_index
            aligned = self._aligned_source is not None
            return {
                "uploaded": self._source_rgb is not None,
                "face_count": len(self._faces),
                "faces": [face.as_dict() for face in self._faces],
                "selected_face_index": selected,
                "ready": aligned,
                "model_ready": aligned and self._representation is not None,
                "width": self._width,
                "height": self._height,
                "storage": "memory only",
            }

    def clear(self) -> None:
        with self._lock:
            self._clear_pixels()
            self._faces = []
            self._selected_face_index = None
            self._width = 0
            self._height = 0

    def _clear_pixels(self) -> None:
        for item in (self._source_rgb, self._aligned_source, self._representation):
            if isinstance(item, np.ndarray):
                item.fill(0)
        self._source_rgb = None
        self._aligned_source = None
        self._representation = None
