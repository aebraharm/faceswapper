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
    models_dir: str | None = None
    selected_model_id: str = "liveportrait-v1"
    onnx_intra_threads: int | None = None
    # Diagnostic tracing is deliberately opt-in. It is sampled so enabling it
    # while debugging a live camera does not produce a log line for every frame.
    frame_diagnostics: bool = False
    frame_diagnostics_interval_ms: int = 1_000

    @classmethod
    def from_environment(cls) -> "Settings":
        intra_threads = os.getenv("FACE_ONNX_INTRA_THREADS")
        diagnostics_value = os.getenv("FRAME_FRAME_DIAGNOSTICS", "").strip().lower()
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
            models_dir=os.getenv("FRAME_MODELS_DIR") or None,
            selected_model_id=os.getenv("FACE_MODEL_ID", "liveportrait-v1"),
            onnx_intra_threads=int(intra_threads) if intra_threads else None,
            frame_diagnostics=diagnostics_value in {"1", "true", "yes", "on"},
            frame_diagnostics_interval_ms=max(1, int(os.getenv("FRAME_FRAME_DIAGNOSTICS_INTERVAL_MS", 1_000))),
        )

    def resolved_model_path(self, value: str | None) -> Path | None:
        return Path(value).expanduser().resolve() if value else None

    def resolved_models_dir(self) -> Path:
        if self.models_dir:
            return Path(self.models_dir).expanduser().resolve()
        # Development/CLI fallback. The Electron shell always sets FRAME_MODELS_DIR
        # to a per-user application-data directory (see desktop-paths.ts).
        return Path.home() / ".frame" / "models"
