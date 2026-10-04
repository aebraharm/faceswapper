"""Feathered face mask and inverse alignment warp for natural integration."""
from __future__ import annotations

import cv2
import numpy as np

from app.alignment.face_aligner import AlignedFace


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
        height, width = frame_rgb.shape[:2]
        size = aligned.output_size
        transformed = cv2.resize(transformed_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((size, size), dtype=np.float32)
        center = (size // 2, int(size * 0.52))
        axes = (int(size * 0.43), int(size * 0.48))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        transformed = self._match_target_lighting(transformed, aligned.image, mask)
        blur = max(3, int(size * self.feather_fraction) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
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
        alpha = np.clip(warped_mask[..., None] * intensity, 0.0, 1.0)
        result = frame_rgb.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

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
