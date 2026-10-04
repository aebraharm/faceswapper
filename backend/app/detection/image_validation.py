"""Strict in-memory validation/decoding for user-selected source portraits."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_MIME_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class ImageValidationError(ValueError):
    """An upload failed a user-facing image validation check."""


@dataclass(frozen=True)
class DecodedImage:
    rgb: np.ndarray
    width: int
    height: int
    format: str


def decode_source_image(
    content: bytes,
    content_type: str | None,
    *,
    max_bytes: int = 8 * 1024 * 1024,
    min_dimension: int = 64,
    max_dimension: int = 6000,
    max_pixels: int = 24_000_000,
) -> DecodedImage:
    """Validate actual file contents and MIME, then decode to RGB without writing to disk."""
    if not content:
        raise ImageValidationError("The uploaded image is empty.")
    if len(content) > max_bytes:
        raise ImageValidationError(f"Image is too large. Maximum upload size is {max_bytes // (1024 * 1024)} MB.")
    if content_type not in SUPPORTED_MIME_TYPES:
        raise ImageValidationError("Unsupported image type. Upload a JPG, PNG, or WEBP image.")

    try:
        with Image.open(BytesIO(content)) as image:
            actual_format = (image.format or "").upper()
            expected_format = SUPPORTED_MIME_TYPES[content_type]
            if actual_format != expected_format:
                raise ImageValidationError("The file contents do not match the selected image format.")
            width, height = image.size
            if width < min_dimension or height < min_dimension:
                raise ImageValidationError(f"Image must be at least {min_dimension} × {min_dimension} pixels.")
            if width > max_dimension or height > max_dimension or width * height > max_pixels:
                raise ImageValidationError(
                    f"Image dimensions are too large. Maximum is {max_dimension} px per side and {max_pixels:,} pixels total."
                )
            image.verify()
        with Image.open(BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = np.asarray(image, dtype=np.uint8).copy()
        return DecodedImage(rgb=pixels, width=pixels.shape[1], height=pixels.shape[0], format=actual_format)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The image could not be decoded. Try a valid JPG, PNG, or WEBP file.") from exc
