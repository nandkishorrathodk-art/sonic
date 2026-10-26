"""Cortex training dataset built from watched-video frames.

The cortex multi-task loss consumes five targets: ``frame_t``, ``frame_next``,
``affordance_mask``, ``category_idx``, and ``target_coord``. A raw video has
none of the three labels. Rather than invent them, this dataset derives only
what the pixels and cursor trajectory actually support:

* ``frame_t`` / ``frame_next`` - consecutive real frames (temporal predictive
  coding is fully self-supervised; the strongest, honest signal from video).
* ``target_coord`` - the cursor location when the video gave a confident point,
  else the frame centre as a *neutral* placeholder. A neutral prior teaches the
  grounding head "no specific target", which is honest; it is NOT a fabricated
  click label.
* ``affordance_mask`` - a single-class "static" mask, because we genuinely do
  not know which regions are interactive. This keeps the tensor shape valid
  without claiming pixel-level ground truth we do not have.
* ``category_idx`` - a dedicated ``unlabeled`` category index when one exists,
  else ``-1`` and the caller must mask the category loss.

Because three of five targets are weak, this dataset is only appropriate for
*pretraining* the self-supervised trunk. ``labeled_ratio`` reports how many
samples carried a real cursor point so a caller can judge how much of the batch
is genuinely supervised.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional  # noqa: UP035

import numpy as np
import torch
from torch.utils.data import Dataset

from forge.vision.cortex.core.dorsal import AFFORDANCE_CLASSES
from forge.vision.cortex.core.ventral import UI_CATEGORIES

# Class index 0 is "static" (non-interactive background) in AFFORDANCE_CLASSES.
_STATIC_CLASS = 0
_UNLABELED_CATEGORY = -1


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class VideoFrameDataset(Dataset):
    """Self-supervised cortex samples from a ``video_ingest.json`` manifest.

    ``manifest_path`` is the file written by
    ``avcs.training.video_apprenticeship.write_ingest_manifest``. Frames must
    still exist on disk next to it; a missing frame fails loudly.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int = 640,
        height: int = 360,
        affordance_grid_size: tuple[int, int] = (32, 32),
        latent_dim: int = 256,
        labeled_only: bool = False,
    ) -> None:
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"Video ingest manifest not found: {manifest}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        raw_frames = payload.get("frames") or []
        if len(raw_frames) < 2:
            raise ValueError(
                "A video frame dataset needs at least two frames for temporal "
                "prediction; ingest more frames first."
            )
        self.root = manifest.parent
        self.width = width
        self.height = height
        self.affordance_grid_size = affordance_grid_size
        self.latent_dim = latent_dim
        self.records = []
        for entry in raw_frames:
            point = entry.get("cursor_point_norm")
            if labeled_only and not point:
                continue
            self.records.append(entry)
        if len(self.records) < 2:
            raise ValueError(
                "Not enough frames remain after filtering; ingest more frames "
                "or disable labeled_only."
            )
        self.unlabeled_category = (
            UI_CATEGORIES.index("unlabeled") if "unlabeled" in UI_CATEGORIES else _UNLABELED_CATEGORY
        )

    def __len__(self) -> int:
        # Pairs of consecutive frames.
        return len(self.records) - 1

    @property
    def labeled_ratio(self) -> float:
        labeled = sum(1 for r in self.records if r.get("cursor_point_norm"))
        return labeled / max(len(self.records), 1)

    def _load(self, entry: dict) -> np.ndarray:
        frame_path = self.root / str(entry["path"])
        if not frame_path.is_file():
            raise FileNotFoundError(f"Video frame missing from manifest: {frame_path}")
        rgb = _read_rgb(frame_path)
        resized = _nearest_resize(rgb, self.width, self.height)
        return resized.astype(np.float32).transpose(2, 0, 1) / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cur = self.records[idx]
        nxt = self.records[idx + 1]
        frame_t = torch.from_numpy(self._load(cur))
        frame_next = torch.from_numpy(self._load(nxt))

        point = cur.get("cursor_point_norm")
        if point:
            coord = torch.tensor([float(point[0]), float(point[1])], dtype=torch.float32)
            grounding_valid = True
        else:
            # Neutral prior: "no specific target in this frame".
            coord = torch.tensor([0.5, 0.5], dtype=torch.float32)
            grounding_valid = False

        grid_h, grid_w = self.affordance_grid_size
        mask = torch.full((grid_h, grid_w), _STATIC_CLASS, dtype=torch.long)
        category_idx = torch.tensor(self.unlabeled_category, dtype=torch.long)
        # Watched video has no language label; a zero query embedding is the
        # honest "no query" signal, and the grounding term is masked so the head
        # is never trained on an invented coordinate.
        query_embed = torch.zeros(self.latent_dim, dtype=torch.float32)
        return {
            "frame_t": frame_t,
            "frame_next": frame_next,
            "affordance_mask": mask,
            "category_idx": category_idx,
            "target_coord": coord,
            "query_embed": query_embed,
            "grounding_valid": torch.tensor(grounding_valid, dtype=torch.bool),
        }


def _nearest_resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    if (src_h, src_w) == (height, width):
        return rgb
    y_idx = (np.arange(height) * src_h // max(height, 1)).clip(0, src_h - 1)
    x_idx = (np.arange(width) * src_w // max(width, 1)).clip(0, src_w - 1)
    return rgb[y_idx][:, x_idx]


__all__ = ["VideoFrameDataset", "AFFORDANCE_CLASSES", "UI_CATEGORIES"]


def video_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    labeled_only: bool = False,
) -> torch.utils.data.DataLoader:  # noqa: F821
    from torch.utils.data import DataLoader

    dataset = VideoFrameDataset(manifest_path, labeled_only=labeled_only)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def available_category_count() -> int:
    """Number of real UI categories the ventral head can actually predict."""
    return len(UI_CATEGORIES)


def _optional_import_guard() -> Optional[int]:  # noqa: UP045
    """Kept for API symmetry; returns the number of affordance classes."""
    return len(AFFORDANCE_CLASSES)
