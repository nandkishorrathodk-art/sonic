"""Dorsal Stream ("WHERE / HOW") - Spatial Affordances & Action Fields.

In primate neurobiology, the dorsal stream (parietal pathway) specializes in
spatial location, motion, and motor affordance ("How can I interact with this?").

For a computer screen, this module transforms visual features into a dense
affordance field, predicting where actions are physically possible:
- Clickable (Buttons, Links, Icons, Tabs)
- Typeable (Input fields, Textareas, Search boxes)
- Scrollable (Feeds, Long panels, Code editors)
- Draggable (Sliders, Splitters, Window bars)
- Static (Non-interactive text, backgrounds)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

AFFORDANCE_CLASSES = [
    "static",      # 0: non-interactive content
    "clickable",   # 1: buttons, links, icons, tabs
    "typeable",    # 2: input fields, search bars
    "scrollable",  # 3: scrollable containers, lists
    "draggable",   # 4: sliders, window frames, splitters
]


class DorsalAffordanceStream(nn.Module):
    """Predicts a dense spatial affordance probability tensor."""

    def __init__(self, in_dim: int = 128, num_classes: int = len(AFFORDANCE_CLASSES)) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Spatial refinement head: (B, in_dim, 16, 16) -> (B, num_classes, 32, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # Upsample 2x
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, num_classes, H_affordance, W_affordance)."""
        return self.head(visual_features)

    @torch.no_grad()
    def get_affordance_map(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities of shape (B, num_classes, H, W)."""
        logits = self.forward(visual_features)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def find_action_hotspots(
        self,
        visual_features: torch.Tensor,
        action_type: str = "clickable",
        top_k: int = 3,
        threshold: float = 0.25,
        margin: float = 0.05,
    ) -> List[Tuple[float, float, float]]:
        """Finds normalized (x, y, confidence) for top interactive points using 2D spatial NMS.

        Eliminates cluster collapsing by suppressing adjacent pixels within a 3x3 window,
        and eliminates static background false-positives via competitive margin checks.
        """
        probs = self.get_affordance_map(visual_features)[0]  # (num_classes, H, W)
        if action_type not in AFFORDANCE_CLASSES:
            action_type = "clickable"
        cls_idx = AFFORDANCE_CLASSES.index(action_type)
        heat = probs[cls_idx]  # (H, W)
        static_heat = probs[0]  # (H, W)

        H, W = heat.shape

        # 1. 2D Spatial Non-Maximum Suppression (3x3 MaxPool)
        heat_4d = heat.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        max_pool = F.max_pool2d(heat_4d, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)

        # 2. Peak identification: local maxima exceeding threshold and static background margin
        is_peak = (heat == max_pool) & (heat >= threshold)
        if cls_idx != 0:
            is_peak = is_peak & ((heat - static_heat) >= margin)

        peak_indices = torch.nonzero(is_peak, as_tuple=False)  # (N, 2) -> (y, x)
        if len(peak_indices) == 0:
            # Fallback to threshold mask if no strict peak with margin
            fallback_mask = (heat >= threshold) & (heat == max_pool)
            peak_indices = torch.nonzero(fallback_mask, as_tuple=False)

        if len(peak_indices) == 0:
            # Absolute fallback
            fallback_mask = heat >= threshold
            peak_indices = torch.nonzero(fallback_mask, as_tuple=False)

        if len(peak_indices) == 0:
            return []

        peak_vals = heat[peak_indices[:, 0], peak_indices[:, 1]]
        k = min(top_k, len(peak_vals))
        vals, sort_idx = torch.topk(peak_vals, k=k)

        hotspots = []
        for i in range(k):
            idx = sort_idx[i]
            y_idx = int(peak_indices[idx, 0].item())
            x_idx = int(peak_indices[idx, 1].item())
            val = float(vals[i].item())
            norm_x = (x_idx + 0.5) / W
            norm_y = (y_idx + 0.5) / H
            hotspots.append((float(norm_x), float(norm_y), float(val)))
        return hotspots

    @torch.no_grad()
    def find_action_regions(
        self,
        visual_features: torch.Tensor,
        action_type: str = "clickable",
        top_k: int = 3,
        threshold: float = 0.25,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Finds normalized (center_x, center_y, width, height, confidence) for interactive regions."""
        hotspots = self.find_action_hotspots(
            visual_features, action_type=action_type, top_k=top_k, threshold=threshold
        )
        probs = self.get_affordance_map(visual_features)[0]
        H, W = probs.shape[1], probs.shape[2]

        regions = []
        for nx, ny, conf in hotspots:
            cx = int(nx * W)
            cy = int(ny * H)
            x_min, x_max = max(0, cx - 1), min(W - 1, cx + 1)
            y_min, y_max = max(0, cy - 1), min(H - 1, cy + 1)
            w_norm = max(0.04, (x_max - x_min + 1) / W)
            h_norm = max(0.03, (y_max - y_min + 1) / H)
            regions.append((nx, ny, float(w_norm), float(h_norm), conf))
        return regions

