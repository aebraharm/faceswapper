"""Configuration shared by the API and processing pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    _repo_env = Path(__file__).resolve().parents[2] / ".env"
    if _repo_env.is_file():
        load_dotenv(_repo_env, override=False)
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = 8 * 1024 * 1024
    min_image_dimension: int = 64
    max_image_dimension: int = 6000
    max_image_pixels: int = 24_000_000
    default_processing_resolution: int = 640
    source_encoder_model: str | None = None
    face_transformer_model: str | None = None
    face_landmarker_task: str | None = None
    default_provider: str = "auto"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            max_upload_bytes=int(os.getenv("FACE_MAX_UPLOAD_BYTES", 8 * 1024 * 1024)),
            min_image_dimension=int(os.getenv("FACE_MIN_IMAGE_DIMENSION", 64)),
            max_image_dimension=int(os.getenv("FACE_MAX_IMAGE_DIMENSION", 6000)),
            max_image_pixels=int(os.getenv("FACE_MAX_IMAGE_PIXELS", 24_000_000)),
            default_processing_resolution=int(os.getenv("FACE_PROCESSING_RESOLUTION", 640)),
            source_encoder_model=os.getenv("FACE_SOURCE_ENCODER_MODEL") or None,
            face_transformer_model=os.getenv("FACE_TRANSFORMER_MODEL") or None,
            face_landmarker_task=os.getenv("FACE_LANDMARKER_TASK") or None,
            default_provider=os.getenv("FACE_ONNX_PROVIDER", "auto"),
        )

    def resolved_model_path(self, value: str | None) -> Path | None:
        return Path(value).expanduser().resolve() if value else None
