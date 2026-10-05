from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CameraStartRequest(BaseModel):
    device_id: str | None = Field(default=None, max_length=512)


class SourceFaceSelectionRequest(BaseModel):
    face_index: int = Field(ge=0, le=3)


class TargetSelectionRequest(BaseModel):
    face_index: int | None = Field(default=None, ge=0, le=3)


class CameraSettingsRequest(BaseModel):
    transform_enabled: bool | None = None
    intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    processing_resolution: Literal[320, 480, 640, 720] | None = None
    performance_mode: Literal["auto", "quality", "performance"] | None = None


class TransformerLoadRequest(BaseModel):
    provider: Literal["auto", "CPUExecutionProvider", "CUDAExecutionProvider"] | None = None
