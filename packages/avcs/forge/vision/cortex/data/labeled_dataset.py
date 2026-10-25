"""Dataset for hand-labeled real-desktop frames captured from the workstation.

Unlike :class:`VideoFrameDataset` (raw watched video, no labels), this dataset
consumes a capture manifest whose every frame carries a *real* scene label the
operator observed, so the cortex can be trained on genuine desktop imagery with
genuine category targets. Nothing is invented: a frame with an unknown label is
masked (category -1) rather than assigned a guess.

Affordance and click-coordinate targets are NOT fabricated here; they are
masked (``-100`` for the affordance map, and grounding is skipped) so those
heads keep whatever they learned from any synthetic/labeled run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from forge.vision.cortex.core.ventral import UI_CATEGORIES

_UNLABELED = -1
_MASKED_AFFORDANCE = -100
_AFFORDANCE_GRID = (32, 32)


class LabeledFrameDataset(Dataset):
    """Real captured desktop frames with honest scene-category labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int = 640,
        height: int = 360,
        latent_dim: int = 256,
    ) -> None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        root = Path(manifest_path).parent
        self.width = width
        self.height = height
        self.latent_dim = latent_dim
        self.records: list[dict] = []
        for item in manifest.get("frames", []):
            label = str(item.get("label", ""))
            category = UI_CATEGORIES.index(label) if label in UI_CATEGORIES else _UNLABELED
            self.records.append({
                "path": root / item["frame"],
                "category_idx": category,
                "label": label,
            })
        if not self.records:
            raise ValueError(f"no frames in manifest: {manifest_path}")
        # Consecutive frames of the same scene act as the temporal pair.
        self._by_label: dict[str, list[int]] = {}
        for idx, rec in enumerate(self.records):
            self._by_label.setdefault(rec["label"], []).append(idx)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labeled_ratio(self) -> float:
        labeled = sum(1 for r in self.records if r["category_idx"] != _UNLABELED)
        return labeled / max(len(self.records), 1)

    def _load(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height))
        return np.asarray(img, dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        frame = self._load(rec["path"])
        # Pair with another frame of the same scene class for the predictive task.
        siblings = self._by_label.get(rec["label"], [idx])
        other = siblings[(siblings.index(idx) + 1) % len(siblings)] if len(siblings) > 1 else idx
        frame_next = self._load(self.records[other]["path"])

        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_next_t = torch.from_numpy(frame_next).permute(2, 0, 1).float() / 255.0
        # Real frames carry no hand-labeled affordance mask or click target.
        affordance = torch.full(_AFFORDANCE_GRID, _MASKED_AFFORDANCE, dtype=torch.long)
        return {
            "frame_t": frame_t,
            "frame_next": frame_next_t,
            "affordance_mask": affordance,
            "category_idx": torch.tensor(rec["category_idx"], dtype=torch.long),
            "target_coord": torch.tensor([0.5, 0.5], dtype=torch.float32),
            "query_embed": torch.randn(self.latent_dim, dtype=torch.float32),
        }


def create_labeled_dataloader(
    manifest_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LabeledFrameDataset(manifest_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
