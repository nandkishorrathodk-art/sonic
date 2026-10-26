"""Outcome-labelled affordance supervision: learn where controls are by clicking.

The dorsal head predicts *where* actions are possible. Training it needs dense
ground-truth masks, which hand-labelling does not scale to and synthetic screens
only fake. But the being already produces exactly the signal that is missing,
every time it works: it clicks somewhere and then observes whether the screen
changed.

That is real supervision, and it is point-labelled rather than dense:

* the click changed the screen  -> the clicked point WAS an interactive control
  of the action's class (a positive example);
* the click did not change the screen -> the clicked point was NOT that class
  (a negative example).

Everything else in the frame is **unknown**, and this dataset says so. Only a
small disk around the clicked point is supervised; every other pixel is
``-100`` (ignored) so the loss never asserts that an unobserved region is
non-interactive. Asserting that would be fabricated supervision of exactly the
kind this module exists to avoid.

When a click carries the operator's description of what was clicked, the sample
also yields real grounding supervision: a genuine (query text -> click point)
pair. That is the other head which otherwise had no honest training signal.

Provenance: every sample records ``outcome`` and ``label_source`` so a reader
can tell a measured outcome from an assumed one. A negative (unchanged) click is
weaker evidence than a positive one -- an action can be real yet produce no
visible change -- so the manifest records the counts and the trainer reports
them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from forge.vision.cortex.core.dorsal import AFFORDANCE_CLASSES
from forge.vision.cortex.core.ventral import UI_CATEGORIES

IGNORE_INDEX = -100
AFFORDANCE_GRID = (32, 32)
_UNLABELED_CATEGORY = -1

# How many grid cells around the clicked point are supervised. The dorsal head
# emits a 32x32 map, so one cell is ~1/32 of the screen; a radius of 2 marks a
# small neighbourhood rather than a single pixel, matching how far a click can
# realistically land from a control's centre.
DEFAULT_LABEL_RADIUS = 2


def _grid_center(point: tuple[float, float], grid: tuple[int, int]) -> tuple[int, int]:
    """Map a normalized ``(x, y)`` in [0, 1] to the nearest grid cell."""
    gx = int(round(float(point[0]) * (grid[1] - 1)))
    gy = int(round(float(point[1]) * (grid[0] - 1)))
    return (
        min(max(gx, 0), grid[1] - 1),
        min(max(gy, 0), grid[0] - 1),
    )


def build_point_mask(
    point: tuple[float, float],
    class_index: int,
    *,
    grid: tuple[int, int] = AFFORDANCE_GRID,
    radius: int = DEFAULT_LABEL_RADIUS,
) -> np.ndarray:
    """A sparse supervision mask: the class on a disk, ``-100`` everywhere else.

    Only the clicked neighbourhood carries a label. The rest of the frame stays
    ``-100`` so the loss ignores it instead of asserting it is static.
    """
    if class_index < 0 or class_index >= len(AFFORDANCE_CLASSES):
        raise ValueError(f"affordance class index out of range: {class_index}")
    mask = np.full(grid, IGNORE_INDEX, dtype=np.int64)
    gx, gy = _grid_center(point, grid)
    r = max(0, int(radius))
    for y in range(max(0, gy - r), min(grid[0], gy + r + 1)):
        for x in range(max(0, gx - r), min(grid[1], gx + r + 1)):
            if (x - gx) ** 2 + (y - gy) ** 2 <= r * r:
                mask[y, x] = class_index
    return mask


class OutcomeAffordanceDataset(Dataset):
    """Clicks with a measured outcome, as point-labelled affordance supervision.

    Manifest shape (one JSON object per sample)::

        {"frame": "frames/frame_00001.png", "click": [0.31, 0.72],
         "action": "clickable", "changed": true, "group": "run-3",
         "query": "Save button", "category": "editor_code",
         "label_source": "click_outcome", "recorded_at": 1.7e9}

    ``changed`` is the measured outcome and is required. ``query`` and
    ``category`` are optional; when absent the corresponding head is masked
    rather than guessed.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int = 640,
        height: int = 360,
        latent_dim: int = 256,
        label_radius: int = DEFAULT_LABEL_RADIUS,
    ) -> None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        root = Path(manifest_path).parent
        self.width = width
        self.height = height
        self.latent_dim = latent_dim
        self.label_radius = label_radius
        self.records: list[dict] = []
        for item in manifest.get("samples", []):
            click = item.get("click")
            if not (isinstance(click, (list, tuple)) and len(click) == 2):
                continue
            if "changed" not in item:
                # No measured outcome -> no honest label. Skip, never assume.
                continue
            label = str(item.get("category", ""))
            self.records.append({
                "path": root / item["frame"],
                "click": (float(click[0]), float(click[1])),
                "action": str(item.get("action", "clickable")),
                "changed": bool(item["changed"]),
                "query": str(item.get("query", "")),
                "category_idx": (
                    UI_CATEGORIES.index(label) if label in UI_CATEGORIES else _UNLABELED_CATEGORY
                ),
                "group": str(item.get("group", "")),
            })
        if not self.records:
            raise ValueError(f"no outcome-labelled samples in manifest: {manifest_path}")
        self._by_group: dict[str, list[int]] = {}
        for idx, rec in enumerate(self.records):
            self._by_group.setdefault(rec["group"], []).append(idx)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def positive_ratio(self) -> float:
        positives = sum(1 for r in self.records if r["changed"])
        return positives / max(len(self.records), 1)

    def _load(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height))
        return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        frame = self._load(rec["path"])

        # Temporal pair for the predictive task: the next click in the same run,
        # falling back to this frame so a lone sample stays usable.
        group = self._by_group.get(rec["group"], [idx])
        nxt = group[(group.index(idx) + 1) % len(group)] if len(group) > 1 else idx
        frame_next = self._load(self.records[nxt]["path"])

        # Positive -> the clicked class. Negative -> static. Everything else in
        # the frame stays -100 because it was never observed.
        action = rec["action"] if rec["action"] in AFFORDANCE_CLASSES else "clickable"
        class_index = AFFORDANCE_CLASSES.index(action) if rec["changed"] else 0
        mask = build_point_mask(
            rec["click"], class_index, grid=AFFORDANCE_GRID, radius=self.label_radius
        )

        sample: dict[str, torch.Tensor] = {
            "frame_t": torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0,
            "frame_next": torch.from_numpy(frame_next).permute(2, 0, 1).float() / 255.0,
            "affordance_mask": torch.from_numpy(mask),
            "category_idx": torch.tensor(rec["category_idx"], dtype=torch.long),
            "target_coord": torch.tensor(
                [rec["click"][0], rec["click"][1]], dtype=torch.float32
            ),
        }
        # Grounding supervision exists only when the operator named the target.
        if rec["query"]:
            from forge.vision.cortex.core.modulation import encode_text_query

            sample["query_embed"] = encode_text_query(rec["query"], self.latent_dim)[0]
            sample["grounding_valid"] = torch.tensor(True, dtype=torch.bool)
        else:
            sample["query_embed"] = torch.zeros(self.latent_dim, dtype=torch.float32)
            sample["grounding_valid"] = torch.tensor(False, dtype=torch.bool)
        return sample


def create_outcome_dataloader(
    manifest_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    dataset = OutcomeAffordanceDataset(manifest_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
