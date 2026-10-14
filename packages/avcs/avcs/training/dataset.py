"""Procedural synthetic fixture dataset for Cortex-2M smoke experiments.

This dataset renders colored rectangles, not real desktop screenshots, and its
generated labels do not measure grounding on actual applications.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from avcs.cortex_2m import UNIVERSAL_AFFORDANCE_CLASSES, UNIVERSAL_APP_CATEGORIES


class UniversalScreenDataset(Dataset):
    """Synthetic colored-block dataset; unsuitable as real desktop training data."""

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: Tuple[int, int] = (128, 128),
        feature_grid: Tuple[int, int] = (8, 8),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.H, self.W = image_size
        self.gh, self.gw = feature_grid
        self.num_apps = len(UNIVERSAL_APP_CATEGORIES)
        self.num_affordances = len(UNIVERSAL_AFFORDANCE_CLASSES)
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _render_scene(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, str, Tuple[float, float]]:
        rng = random.Random((self.seed or 42) + idx)

        # 1. Select Application Category
        app_idx = rng.randint(0, self.num_apps - 1)
        app_name = UNIVERSAL_APP_CATEGORIES[app_idx]

        # 2. Base Theme Palette based on Application Domain
        if "code" in app_name or "terminal" in app_name:
            bg_color = [30 + rng.randint(-5, 5), 30 + rng.randint(-5, 5), 35 + rng.randint(-5, 5)]  # Dark IDE
        elif "canvas" in app_name or "cad" in app_name or "video" in app_name:
            bg_color = [45 + rng.randint(-5, 5), 45 + rng.randint(-5, 5), 50 + rng.randint(-5, 5)]  # Slate CAD
        elif "spreadsheet" in app_name or "document" in app_name or "browser" in app_name:
            bg_color = [240 + rng.randint(-10, 10), 242 + rng.randint(-10, 10), 245 + rng.randint(-10, 10)] # Light Office
        else:
            bg_color = [rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220)]

        frame = np.full((self.H, self.W, 3), bg_color, dtype=np.uint8)
        aff_mask = np.zeros((self.num_affordances, self.gh, self.gw), dtype=np.float32)
        aff_mask[0, :, :] = 1.0  # Background channel default

        # 3. Procedural UI Components Placement
        num_components = rng.randint(3, 8)
        chosen_target_coord = (0.5, 0.5)
        chosen_query = "click center"

        for comp_i in range(num_components):
            aff_class_idx = rng.randint(1, self.num_affordances - 1)
            aff_name = UNIVERSAL_AFFORDANCE_CLASSES[aff_class_idx]

            # Coordinate & Dimension
            cw = rng.randint(12, 40)
            ch = rng.randint(8, 24)
            cx = rng.randint(4, self.W - cw - 4)
            cy = rng.randint(4, self.H - ch - 4)

            # Component color
            if "primary" in aff_name:
                comp_color = [40, 120, 240]  # Accent Blue
            elif "button" in aff_name or "tab" in aff_name:
                comp_color = [180, 185, 195] if bg_color[0] > 128 else [70, 75, 85]
            elif "text" in aff_name or "code" in aff_name:
                comp_color = [255, 255, 255] if bg_color[0] < 128 else [20, 20, 25]
            elif "handle" in aff_name or "slider" in aff_name:
                comp_color = [250, 150, 40]   # Orange Handle
            else:
                comp_color = [120, 130, 140]

            frame[cy:cy + ch, cx:cx + cw] = comp_color

            # Project into feature grid (8x8)
            gx = min(self.gw - 1, max(0, int((cx + cw / 2.0) / self.W * self.gw)))
            gy = min(self.gh - 1, max(0, int((cy + ch / 2.0) / self.H * self.gh)))

            # Mark affordance heatmap peak
            aff_mask[aff_class_idx, gy, gx] = 1.0
            aff_mask[0, gy, gx] = 0.0  # Clear background at hotspot

            if comp_i == 0:
                nx = (cx + cw / 2.0) / float(self.W)
                ny = (cy + ch / 2.0) / float(self.H)
                chosen_target_coord = (nx, ny)
                chosen_query = f"{aff_name.replace('_', ' ')}"

        return frame, aff_mask, app_idx, chosen_query, chosen_target_coord

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame, aff_mask, app_idx, query, target_coord = self._render_scene(idx)

        # 1. Normalize image: (H, W, 3) -> (3, H, W) in [0, 1]
        img_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0

        # 2. Affordance heatmap: (24, gh, gw)
        aff_tensor = torch.from_numpy(aff_mask).float()

        # 3. Application Category target: scalar long
        cat_tensor = torch.tensor(app_idx, dtype=torch.long)

        # 4. Deterministic Character N-gram Text Embedding (128-d)
        text_emb = np.zeros(128, dtype=np.float32)
        for char_i, ch in enumerate(query[:32]):
            bucket = (ord(ch) * 31 + char_i * 17) % 128
            text_emb[bucket] += 1.0
        norm = np.linalg.norm(text_emb) + 1e-6
        text_emb_tensor = torch.from_numpy(text_emb / norm).float()

        # 5. Gaussian Grounding Target Heatmap (1, gh, gw)
        tx, ty = target_coord
        gx = tx * (self.gw - 1)
        gy = ty * (self.gh - 1)
        yy, xx = np.meshgrid(np.arange(self.gh), np.arange(self.gw), indexing="ij")
        dist_sq = (xx - gx) ** 2 + (yy - gy) ** 2
        ground_map = np.exp(-dist_sq / (2.0 * (1.2 ** 2))).astype(np.float32)
        ground_tensor = torch.from_numpy(ground_map).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "affordance_map": aff_tensor,
            "category": cat_tensor,
            "text_embedding": text_emb_tensor,
            "grounding_target": ground_tensor,
        }
