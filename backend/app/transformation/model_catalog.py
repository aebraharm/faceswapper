"""Catalog of face-transformation models FRAME can install on request.

No weights live in this repository. Every entry below points at files that are
publicly hosted by their original authors (or a well-known mirror) under a
license that explicitly permits redistribution/commercial use, together with
the published SHA-256 checksum so a download can be verified before use.

Before adding a new catalog entry, verify independently that:
  * the exact checkpoint (not just the surrounding code) is permissively
    licensed for the intended use, and
  * the hosting terms allow an application to fetch the file programmatically.

See ``models/README.md`` for the licensing notes shown to the user.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAsset:
    """One file that belongs to a model bundle."""

    filename: str
    url: str
    sha256: str
    approx_bytes: int


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    display_name: str
    kind: str  # Selects the FaceTransformer implementation in TransformerManager.
    version: str
    license_name: str
    license_url: str
    source_url: str
    summary: str
    assets: tuple[ModelAsset, ...]

    @property
    def approx_total_bytes(self) -> int:
        return sum(asset.approx_bytes for asset in self.assets)


# LivePortrait (human) core reenactment networks: appearance feature extractor,
# motion extractor and the fused warping+SPADE generator. These three graphs are
# MIT-licensed by the original authors (Kuaishou/KwaiVGI) -- see
# https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE. The official LICENSE
# notes that LivePortrait's *reference pipeline* also uses InsightFace's face
# detector/landmarker, which is non-commercial-only, and recommends replacing it
# for MIT compliance. FRAME never uses the InsightFace assets: it reuses its own
# OpenCV + MediaPipe detector/landmarker/aligner, so only the MIT-licensed
# reenactment networks below are downloaded.
#
# The ONNX export of the same checkpoints is republished with public SHA-256
# checksums by the community project warmshao/FasterLivePortrait (MIT-licensed
# code; the weights remain the original MIT-licensed LivePortrait checkpoints):
# https://huggingface.co/warmshao/FasterLivePortrait/tree/main/liveportrait_onnx
_LIVE_PORTRAIT_BASE = "https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx"

LIVEPORTRAIT_V1 = ModelDefinition(
    id="liveportrait-v1",
    display_name="LivePortrait (identity-preserving reenactment)",
    kind="liveportrait",
    version="1.0",
    license_name="MIT",
    license_url="https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE",
    source_url="https://huggingface.co/warmshao/FasterLivePortrait/tree/main/liveportrait_onnx",
    summary=(
        "Drives the cached source identity with the live camera's pose and expression. "
        "MIT-licensed; FRAME uses its own OpenCV/MediaPipe face detector instead of the "
        "non-commercial InsightFace detector bundled with the reference implementation."
    ),
    assets=(
        ModelAsset(
            filename="appearance_feature_extractor.onnx",
            url=f"{_LIVE_PORTRAIT_BASE}/appearance_feature_extractor.onnx",
            sha256="d070afccca7f528ffb0ef5052b21588b42225a996661833f7bdede562d1ab921",
            approx_bytes=3_360_000,
        ),
        ModelAsset(
            filename="motion_extractor.onnx",
            url=f"{_LIVE_PORTRAIT_BASE}/motion_extractor.onnx",
            sha256="219a46174297b2b411bb3c5dce48d3a8c8e07a9d82120a0da21f49f58b67fca6",
            approx_bytes=113_000_000,
        ),
        ModelAsset(
            filename="warping_spade.onnx",
            url=f"{_LIVE_PORTRAIT_BASE}/warping_spade.onnx",
            sha256="b0e7a566db8fba690c23523bcd2faa4f0d13f05418db84a779397a851062ad69",
            approx_bytes=421_000_000,
        ),
    ),
)

CATALOG: dict[str, ModelDefinition] = {LIVEPORTRAIT_V1.id: LIVEPORTRAIT_V1}


def get_model(model_id: str) -> ModelDefinition | None:
    return CATALOG.get(model_id)


def list_models() -> list[ModelDefinition]:
    return list(CATALOG.values())
