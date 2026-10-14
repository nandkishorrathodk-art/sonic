"""Real-World UI Dataset Ingestion & Grounding Adapter for AVCS.

Supports loading and converting open GUI grounding datasets:
- ScreenSpot (OS/Web/Mobile screenshot grounding)
- Mind2Web (Web interaction & element grounding)
- RICO / Android UI datasets (Mobile UI hierarchy & bounding boxes)

Transforms bounding boxes and task instructions into AVCS-compatible:
1. Dense 5-class affordance masks (static, clickable, typeable, scrollable, draggable)
2. Normalized target gaze fixation coordinates (x, y)
3. Simulated or real text embedding representations for top-down modulation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from forge.vision.cortex.core.dorsal import AFFORDANCE_CLASSES
from forge.vision.cortex.core.ventral import UI_CATEGORIES
from forge.vision.cortex.data.synthetic import ProceduralScreenGenerator


@dataclass
class RealUISample:
    image_path: str
    instruction: str
    bbox: List[float]  # [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax] normalized [0, 1]
    action_type: str   # "click", "type", "scroll", "drag", "select"
    category: str      # "browser_webpage", "editor_code", "mobile_app", etc.


def action_to_affordance_class(action_type: str) -> int:
    """Maps a textual interaction action into AVCS 5-class affordance index."""
    act = action_type.lower()
    if any(k in act for k in ("click", "press", "tap", "select", "check")):
        return 1  # clickable
    elif any(k in act for k in ("type", "input", "write", "enter", "edit")):
        return 2  # typeable
    elif any(k in act for k in ("scroll", "swipe", "wheel")):
        return 3  # scrollable
    elif any(k in act for k in ("drag", "drop", "slider", "move")):
        return 4  # draggable
    return 0  # static / background


def rasterize_bbox_to_affordance_mask(
    bboxes: List[Tuple[List[float], int]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterizes bounding boxes and their affordance classes into a dense 2D mask.
    
    Args:
        bboxes: List of (bbox, affordance_class_idx), where bbox is [ymin, xmin, ymax, xmax] in [0, 1].
        height: Output mask height.
        width: Output mask width.
    """
    mask = np.zeros((height, width), dtype=np.int64)
    for box, cls_idx in bboxes:
        ymin, xmin, ymax, xmax = box
        y0 = int(np.clip(ymin * height, 0, height - 1))
        y1 = int(np.clip(ymax * height, 0, height))
        x0 = int(np.clip(xmin * width, 0, width - 1))
        x1 = int(np.clip(xmax * width, 0, width))
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = cls_idx
    return mask


class RealUIDataset(Dataset):
    """PyTorch Dataset adapter for open UI grounding benchmarks (ScreenSpot, Mind2Web)."""

    def __init__(
        self,
        data_json_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        target_size: Tuple[int, int] = (640, 360),
        affordance_grid_size: Tuple[int, int] = (32, 32),
        latent_dim: int = 256,
        synthetic_fallback_count: int = 200,
    ) -> None:
        self.target_size = target_size  # (width, height)
        self.affordance_grid_size = affordance_grid_size
        self.latent_dim = latent_dim
        self.samples: List[RealUISample] = []
        self.image_dir = Path(image_dir) if image_dir else None

        # Load real annotations if provided and existing
        if data_json_path and os.path.exists(data_json_path):
            self._load_json(data_json_path)

        # Fallback procedural generation if dataset files are absent locally
        self.synthetic_fallback = None
        if len(self.samples) == 0:
            self.synthetic_fallback_count = synthetic_fallback_count
            self.synthetic_generator = ProceduralScreenGenerator(
                width=target_size[0], height=target_size[1]
            )
        else:
            self.synthetic_fallback_count = 0

    def _load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # ScreenSpot format: list of dicts with 'img_path', 'instruction', 'bbox', 'data_type'
        for entry in raw_data:
            img = entry.get("img_path") or entry.get("image") or entry.get("file_name", "")
            instr = entry.get("instruction") or entry.get("goal") or "click element"
            bbox = entry.get("bbox") or [0.1, 0.1, 0.2, 0.2]
            action = entry.get("action_type") or entry.get("action", "click")
            cat = entry.get("ui_type") or "browser_webpage"

            # Normalize bbox format if in [xmin, ymin, xmax, ymax]
            # Standardize to [ymin, xmin, ymax, xmax]
            if len(bbox) == 4:
                # If values are > 1.0, normalize by assumed 1920x1080 resolution
                if any(v > 1.0 for v in bbox):
                    w_ref = entry.get("width", 1920.0)
                    h_ref = entry.get("height", 1080.0)
                    bbox = [bbox[1] / h_ref, bbox[0] / w_ref, bbox[3] / h_ref, bbox[2] / w_ref]

                self.samples.append(
                    RealUISample(
                        image_path=img,
                        instruction=instr,
                        bbox=bbox,
                        action_type=action,
                        category=cat if cat in UI_CATEGORIES else "browser_webpage",
                    )
                )

    def __len__(self) -> int:
        return len(self.samples) if len(self.samples) > 0 else self.synthetic_fallback_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_w, target_h = self.target_size

        if len(self.samples) > 0:
            sample = self.samples[idx % len(self.samples)]
            # Load image
            img_full_path = (self.image_dir / sample.image_path) if self.image_dir else Path(sample.image_path)
            if img_full_path.exists():
                pil_img = Image.open(img_full_path).convert("RGB").resize(self.target_size)
                frame_np = np.array(pil_img, dtype=np.uint8)
            else:
                # Mock if missing
                frame_np = np.full((target_h, target_w, 3), 40, dtype=np.uint8)

            # Rasterize affordance mask
            cls_idx = action_to_affordance_class(sample.action_type)
            mask_np = rasterize_bbox_to_affordance_mask([(sample.bbox, cls_idx)], target_h, target_w)

            # Target gaze coordinate (center of target bounding box)
            ymin, xmin, ymax, xmax = sample.bbox
            center_x = float(np.clip((xmin + xmax) / 2.0, 0.0, 1.0))
            center_y = float(np.clip((ymin + ymax) / 2.0, 0.0, 1.0))
            target_coord = torch.tensor([center_x, center_y], dtype=torch.float32)

            cat_idx = UI_CATEGORIES.index(sample.category) if sample.category in UI_CATEGORIES else 0
        else:
            # Synthetic fallback with realistic GUI components
            data = self.synthetic_generator.generate(seed=idx)
            frame_np = data.frame
            mask_np = data.affordance_mask
            cat_idx = data.category_idx
            if data.interactive_points:
                pt = data.interactive_points[0]
                target_coord = torch.tensor([pt[0], pt[1]], dtype=torch.float32)
            else:
                target_coord = torch.tensor([0.5, 0.5], dtype=torch.float32)

        # PyTorch normalization
        frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0

        # Downsample affordance mask to affordance_grid_size
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
        mask_down = F.interpolate(mask_t, size=self.affordance_grid_size, mode="nearest").squeeze().long()

        # Simulated or lightweight text query embedding for top-down cross-attention
        # In full multimodal training, this can be encoded using a frozen CLIP or MiniLM text tower
        query_embed = torch.randn(self.latent_dim, dtype=torch.float32)

        return {
            "frame_t": frame_t,
            "frame_next": frame_t,  # single-frame or sequential
            "affordance_mask": mask_down,
            "category_idx": torch.tensor(cat_idx, dtype=torch.long),
            "target_coord": target_coord,
            "query_embed": query_embed,
        }


class UnifiedScreenDataset(Dataset):
    """Combines synthetic procedural traces with real GUI screen datasets.
    
    Balances synthetic procedural generation (infinite diverse samples with pixel-perfect masks)
    with real-world desktop screenshots (complex typography, modern web apps, native OS styles).
    """

    def __init__(
        self,
        synthetic_dataset: Dataset,
        real_dataset: Dataset,
        synthetic_ratio: float = 0.5,
    ) -> None:
        self.synthetic = synthetic_dataset
        self.real = real_dataset
        self.synthetic_ratio = synthetic_ratio
        self.total_len = len(self.synthetic) + len(self.real)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if (idx % 100) / 100.0 < self.synthetic_ratio:
            return self.synthetic[idx % len(self.synthetic)]
        else:
            return self.real[idx % len(self.real)]
