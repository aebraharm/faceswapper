"""Feathered face mask and inverse alignment warp for natural integration."""
from __future__ import annotations

import cv2
import numpy as np

from app.alignment.face_aligner import AlignedFace
from app.camera.frame_diagnostics import record_image


def _require_finite_image(
    stage: str,
    image: np.ndarray,
    *,
    channel_order: str | None = None,
    check_finite: bool = False,
) -> np.ndarray:
    """Record every boundary; scan float data only where conversion could hide NaN.

    RGB inputs and intermediate OpenCV images are uint8 after the renderer's
    validation.  Re-scanning each 512px image would cost several full memory
    passes per camera frame, so debug traces perform the complete audit while
    production always checks the final float composite before uint8 conversion.
    """
    array = np.asarray(image)
    record_image(stage, array, channel_order=channel_order)
    if not np.issubdtype(array.dtype, np.number) or array.size == 0:
        raise RuntimeError(f"{stage} is empty or has a non-numeric dtype.")
    if check_finite and not np.isfinite(array).all():
        raise RuntimeError(f"{stage} contains NaN/Inf values.")
    return array


class FaceCompositor:
    def __init__(self, feather_fraction: float = 0.10) -> None:
        self.feather_fraction = feather_fraction

    def blend(
        self,
        frame_rgb: np.ndarray,
        transformed_rgb: np.ndarray,
        aligned: AlignedFace,
        intensity: float = 1.0,
    ) -> np.ndarray:
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("Transformation intensity must be between 0 and 1.")
        frame_rgb = _require_finite_image("compositor.camera_frame_rgb", frame_rgb, channel_order="RGB")
        transformed_rgb = _require_finite_image("compositor.native_input_rgb", transformed_rgb, channel_order="RGB")
        _require_finite_image("compositor.aligned_target_rgb", aligned.image, channel_order="RGB")
        height, width = frame_rgb.shape[:2]
        size = aligned.output_size
        # The LivePortrait renderer natively emits a 512×512 crop after taking
        # 256×256 feature/motion inputs. The inverse alignment matrix below is in
        # FRAME's 256×256 canonical crop coordinates, so resample only here at the
        # compositing boundary. INTER_AREA preserves the renderer's image semantics
        # when reducing its native high-resolution output.
        source_height, source_width = transformed_rgb.shape[:2]
        if (source_width, source_height) == (size, size):
            transformed = transformed_rgb
        else:
            interpolation = cv2.INTER_AREA if source_width >= size and source_height >= size else cv2.INTER_CUBIC
            transformed = cv2.resize(transformed_rgb, (size, size), interpolation=interpolation)
        _require_finite_image("compositor.resampled_rgb_256", transformed, channel_order="RGB")
        mask = np.zeros((size, size), dtype=np.float32)
        center = (size // 2, int(size * 0.52))
        axes = (int(size * 0.43), int(size * 0.48))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        record_image("compositor.mask_256", mask)
        transformed = self._match_target_lighting(transformed, aligned.image, mask)
        _require_finite_image("compositor.lighting_matched_rgb_256", transformed, channel_order="RGB")
        blur = max(3, int(size * self.feather_fraction) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        record_image("compositor.feathered_mask_256", mask)
        warped = cv2.warpAffine(
            transformed,
            aligned.inverse_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_mask = cv2.warpAffine(
            mask,
            aligned.inverse_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        _require_finite_image("compositor.inverse_warped_rgb", warped, channel_order="RGB")
        _require_finite_image("compositor.inverse_warped_mask", warped_mask)
        alpha = np.clip(warped_mask[..., None] * intensity, 0.0, 1.0)
        record_image("compositor.alpha", alpha)
        result = frame_rgb.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        _require_finite_image("compositor.float_result_rgb", result, channel_order="RGB", check_finite=True)
        converted = np.clip(result, 0, 255).astype(np.uint8)
        input_is_nonblack = bool(np.any(frame_rgb != 0))
        record_image(
            "compositor.uint8_result_rgb",
            converted,
            channel_order="RGB",
            require_nonzero=input_is_nonblack,
        )
        # A non-black camera frame must never silently become an all-black
        # processed frame.  Raise at this precise boundary instead of sending a
        # black JPEG that obscures whether the renderer or alpha composite failed.
        if input_is_nonblack and not np.any(converted != 0):
            raise RuntimeError("Face compositing produced an all-zero frame from a non-zero camera frame.")
        return converted

    @staticmethod
    def _match_target_lighting(transformed_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Gently match target luminance statistics while retaining source chroma/identity."""
        transformed_lab = cv2.cvtColor(transformed_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        region = mask > 0.55
        if not np.any(region):
            return transformed_rgb
        source_l = transformed_lab[..., 0]
        target_l = target_lab[..., 0]
        source_mean = float(source_l[region].mean())
        target_mean = float(target_l[region].mean())
        source_std = float(source_l[region].std())
        target_std = float(target_l[region].std())
        contrast = np.clip(target_std / max(source_std, 1.0), 0.75, 1.35)
        matched_l = (source_l - source_mean) * contrast + target_mean
        transformed_lab[..., 0] = source_l * 0.35 + matched_l * 0.65
        return cv2.cvtColor(np.clip(transformed_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
