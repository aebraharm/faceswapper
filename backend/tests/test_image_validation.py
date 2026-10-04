from __future__ import annotations

from PIL import Image
import io
import numpy as np
import pytest

from app.detection.image_validation import ImageValidationError, decode_source_image
from conftest import png_bytes


def test_valid_png_decodes_to_rgb():
    result = decode_source_image(png_bytes(), "image/png")
    assert result.format == "PNG"
    assert result.rgb.shape == (160, 160, 3)
    assert result.rgb.dtype == np.uint8


def test_rejects_unsupported_mime():
    with pytest.raises(ImageValidationError, match="JPG, PNG, or WEBP"):
        decode_source_image(png_bytes(), "image/gif")


def test_rejects_mime_content_mismatch():
    with pytest.raises(ImageValidationError, match="do not match"):
        decode_source_image(png_bytes(), "image/jpeg")


def test_rejects_oversized_upload_before_decode():
    with pytest.raises(ImageValidationError, match="too large"):
        decode_source_image(b"x" * 100, "image/png", max_bytes=10)


def test_rejects_dimensions_below_minimum():
    result = io.BytesIO()
    Image.new("RGB", (32, 32)).save(result, format="PNG")
    with pytest.raises(ImageValidationError, match="at least"):
        decode_source_image(result.getvalue(), "image/png")
