"""Retinal Foveation Engine for Artificial Visual Cortex.

Mimics the biological primate retina:
1. Peripheral Vision: Full-field, downscaled stream capturing global layout,
   window boundaries, and coarse visual structure.
2. Foveal Vision: High-resolution crop centered at the current fixation point,
   preserving fine detail, UI icons, and small text.
3. Saccade Controller: Dynamically updates the fixation point based on:
   - Explicit gaze / cursor coordinates, OR
   - Peak temporal motion (visual change / animation), OR
   - Saliency hotspots (contrast / new elements).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RetinalView:
    """Multi-scale visual inputs delivered to the visual cortex."""
    peripheral: torch.Tensor       # (B, C, H_periph, W_periph) - Global layout
    foveal: torch.Tensor           # (B, C, H_fovea, W_fovea)   - High-res focus
    fixation: Tuple[float, float]  # Normalized (x, y) in [0, 1]
    fovea_bbox: Tuple[int, int, int, int] # (x0, y0, x1, y1) in original image coords


class RetinalFoveation:
    """Extracts peripheral and foveal views from high-res screen frames."""

    def __init__(
        self,
        peripheral_size: Tuple[int, int] = (256, 256),
        foveal_size: Tuple[int, int] = (256, 256),
        foveal_coverage_ratio: float = 0.35, # Fovea covers ~35% of the screen diameter
        saccade_smoothing: float = 0.7,      # EMA alpha for smooth saccadic movement
    ) -> None:
        self.peripheral_size = peripheral_size
        self.foveal_size = foveal_size
        self.foveal_coverage_ratio = foveal_coverage_ratio
        self.saccade_smoothing = saccade_smoothing
        self.current_fixation: Tuple[float, float] = (0.5, 0.5)

    def update_fixation(
        self,
        target_fixation: Optional[Tuple[float, float]] = None,
        motion_map: Optional[np.ndarray] = None,
        saliency_map: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Update the eye fixation point (gaze / saccade)."""
        if target_fixation is not None:
            tx, ty = target_fixation
        elif motion_map is not None and np.max(motion_map) > 0.05:
            # Saccade to peak motion
            y_idx, x_idx = np.unravel_index(np.argmax(motion_map), motion_map.shape)
            tx = float(x_idx) / max(motion_map.shape[1] - 1, 1)
            ty = float(y_idx) / max(motion_map.shape[0] - 1, 1)
        elif saliency_map is not None and np.max(saliency_map) > 0.1:
            # Saccade to peak saliency
            y_idx, x_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
            tx = float(x_idx) / max(saliency_map.shape[1] - 1, 1)
            ty = float(y_idx) / max(saliency_map.shape[0] - 1, 1)
        else:
            tx, ty = self.current_fixation

        # Smooth saccade (inertial saccadic tracking)
        cx = self.saccade_smoothing * self.current_fixation[0] + (1.0 - self.saccade_smoothing) * tx
        cy = self.saccade_smoothing * self.current_fixation[1] + (1.0 - self.saccade_smoothing) * ty
        self.current_fixation = (float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)))
        return self.current_fixation

    def extract_views(
        self,
        frame: torch.Tensor | np.ndarray,
        target_fixation: Optional[Tuple[float, float]] = None,
        motion_map: Optional[np.ndarray] = None,
        saliency_map: Optional[np.ndarray] = None,
    ) -> RetinalView:
        """Process raw screen image into dual-stream RetinalView.

        Parameters
        ----------
        frame: torch.Tensor of shape (B, C, H, W) or (C, H, W), or np.ndarray (H, W, C)
        """
        # Convert to Tensor (B, C, H, W) float in [0, 1]
        if isinstance(frame, np.ndarray):
            if frame.dtype == np.uint8:
                t = torch.from_numpy(frame).float() / 255.0
            else:
                t = torch.from_numpy(frame).float()
            if t.ndim == 3:
                # (H, W, C) -> (1, C, H, W)
                t = t.permute(2, 0, 1).unsqueeze(0)
        else:
            t = frame.float()
            if t.max() > 1.0:
                t = t / 255.0
            if t.ndim == 3:
                t = t.unsqueeze(0)

        _, _, H, W = t.shape

        # 1. Update Fixation Point
        fx, fy = self.update_fixation(target_fixation, motion_map, saliency_map)

        # 2. Extract Peripheral View (Global layout)
        peripheral = F.interpolate(
            t, size=self.peripheral_size, mode="bilinear", align_corners=False
        )

        # 3. Extract Foveal View (Crop around fixation)
        crop_w = int(max(W * self.foveal_coverage_ratio, 32))
        crop_h = int(max(H * self.foveal_coverage_ratio, 32))

        center_x = int(fx * W)
        center_y = int(fy * H)

        x0 = max(0, min(center_x - crop_w // 2, W - crop_w))
        y0 = max(0, min(center_y - crop_h // 2, H - crop_h))
        x1 = min(W, x0 + crop_w)
        y1 = min(H, y0 + crop_h)

        foveal_crop = t[:, :, y0:y1, x0:x1]
        foveal = F.interpolate(
            foveal_crop, size=self.foveal_size, mode="bilinear", align_corners=False
        )

        return RetinalView(
            peripheral=peripheral,
            foveal=foveal,
            fixation=(fx, fy),
            fovea_bbox=(x0, y0, x1, y1),
        )

