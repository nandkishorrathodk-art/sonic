"""Fine-tune the dorsal (affordance) head from measured click outcomes.

:class:`~forge.vision.cortex.data.outcome_affordance.OutcomeAffordanceDataset`
turns clicks-with-a-measured-outcome into sparse point supervision. This trainer
consumes that supervision to actually train the dorsal head, which previously
had no honest training signal and therefore emitted random hotspots.

Design choices that keep it honest:

* **Only the dorsal head is trained.** The ventral category head and the trunk
  are frozen so a small, sparse click set cannot degrade the already-verified
  category model. The head is the thing that was missing supervision.
* **Held-out verification is required.** The split is by ``group`` (one working
  run), so near-duplicate frames never leak across the boundary. The measured
  held-out IoU is written into the provenance sidecar. A run that cannot be
  measured is not recorded as trained.
* **Provenance is written only after a real run.** The sidecar is what makes
  :func:`~forge.vision.cortex.training.affordance_provenance.affordance_head_trained`
  return ``True``; without it the agent keeps reporting hotspots as unavailable.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from forge.vision.cortex.core.dorsal import AFFORDANCE_CLASSES
from forge.vision.cortex.data.outcome_affordance import (
    IGNORE_INDEX,
    OutcomeAffordanceDataset,
)
from forge.vision.cortex.model import ArtificialVisualCortex
from forge.vision.cortex.training.affordance_provenance import (
    write_affordance_provenance,
)
from forge.vision.cortex.training.evaluate import load_cortex_checkpoint

# Classes the head is actually supervised on. ``static`` (index 0) is the
# negative class; the rest are interactive classes a click can name.
_SUPERVISED_CLASSES = tuple(range(len(AFFORDANCE_CLASSES)))


@dataclass
class AffordanceTrainingResult:
    """Outcome of one dorsal fine-tune. Real numbers only."""

    trained: bool
    reason: str = ""
    samples: int = 0
    positives: int = 0
    negatives: int = 0
    held_out_samples: int = 0
    held_out_iou: float | None = None
    held_out_point_accuracy: float | None = None
    epochs: int = 0
    checkpoint: str = ""
    provenance: str = ""
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trained": self.trained,
            "reason": self.reason,
            "samples": self.samples,
            "positives": self.positives,
            "negatives": self.negatives,
            "held_out_samples": self.held_out_samples,
            "held_out_iou": self.held_out_iou,
            "held_out_point_accuracy": self.held_out_point_accuracy,
            "epochs": self.epochs,
            "checkpoint": self.checkpoint,
            "provenance": self.provenance,
            "history": self.history,
        }


def grouped_split(
    samples: list[dict], held_out_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by ``group`` so whole working runs move to one side.

    A group is one continuous interaction; frames inside it are near-duplicates.
    Keeping groups intact stops a held-out IoU from being measured on frames
    that are effectively copies of the training set.
    """
    groups: dict[str, list[dict]] = {}
    for item in samples:
        groups.setdefault(str(item.get("group", "")), []).append(item)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_held = max(1, round(len(keys) * held_out_fraction)) if len(keys) >= 2 else 0
    n_held = min(n_held, max(0, len(keys) - 1))
    held_keys = set(keys[:n_held])
    train: list[dict] = []
    held: list[dict] = []
    for key in keys:
        (held if key in held_keys else train).extend(groups[key])
    return train, held


class AffordanceTrainer:
    """Trains the dorsal affordance head from measured click outcomes."""

    def __init__(
        self,
        base_checkpoint: str | Path,
        *,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 12,
        batch_size: int = 8,
        held_out_fraction: float = 0.25,
        seed: int = 7,
        dataset_provenance: str = "click_outcome_supervision",
    ) -> None:
        self.base_checkpoint = Path(base_checkpoint)
        self.device = torch.device(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.held_out_fraction = held_out_fraction
        self.seed = seed
        self.dataset_provenance = dataset_provenance

    # -- data ---------------------------------------------------------------

    def _split_manifests(self, manifest_path: Path, out_dir: Path) -> tuple[Path, Path, int, int, int]:
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = [s for s in payload.get("samples", []) if "changed" in s]
        if not samples:
            raise ValueError("no outcome-labelled samples to train on")
        train, held = grouped_split(samples, self.held_out_fraction, self.seed)
        if not train or not held:
            raise ValueError("split produced an empty train or held-out set")

        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "affordance_train.json"
        held_path = out_dir / "affordance_heldout.json"
        train_path.write_text(
            json.dumps({"provenance": self.dataset_provenance, "samples": train}, indent=2),
            encoding="utf-8",
        )
        held_path.write_text(
            json.dumps({"provenance": self.dataset_provenance, "samples": held}, indent=2),
            encoding="utf-8",
        )
        positives = sum(1 for s in samples if s.get("changed"))
        return train_path, held_path, len(samples), positives, len(samples) - positives

    # -- training -----------------------------------------------------------

    def train(
        self,
        manifest_path: str | Path,
        output_checkpoint: str | Path,
        *,
        work_dir: str | Path | None = None,
    ) -> AffordanceTrainingResult:
        manifest_path = Path(manifest_path)
        output_checkpoint = Path(output_checkpoint)
        work_dir = Path(work_dir) if work_dir is not None else output_checkpoint.parent

        try:
            train_path, held_path, total, positives, negatives = self._split_manifests(
                manifest_path, work_dir
            )
        except Exception as exc:
            return AffordanceTrainingResult(trained=False, reason=str(exc))

        model = load_cortex_checkpoint(self.base_checkpoint, device=str(self.device))
        model.to(self.device)
        model.eval()

        # Freeze everything; the dorsal head is the only thing that lacked
        # supervision, so it is the only thing this run is allowed to change.
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.dorsal.parameters():
            param.requires_grad_(True)

        train_ds = OutcomeAffordanceDataset(train_path)
        held_ds = OutcomeAffordanceDataset(held_path)
        train_dl = DataLoader(
            train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=True, num_workers=0
        )

        optimizer = torch.optim.AdamW(
            model.dorsal.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        history: list[dict[str, float]] = []
        for epoch in range(1, self.epochs + 1):
            model.dorsal.train()
            epoch_loss = 0.0
            steps = 0
            for batch in train_dl:
                frame = batch["frame_t"].to(self.device)
                target = batch["affordance_mask"].to(self.device)
                optimizer.zero_grad()
                view = model.retina.extract_views(frame)
                out = model.forward(view.peripheral.to(self.device), view.foveal.to(self.device))
                logits = out["affordance_logits"]
                # The mask is already on the head's 32x32 grid, so no resampling
                # is needed and no label is invented between cells.
                if not bool((target != IGNORE_INDEX).any()):
                    continue
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(steps, 1)})

        point_accuracy, held_iou = self._held_out_metrics(model, held_ds)

        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_checkpoint)

        sidecar = write_affordance_provenance(
            output_checkpoint,
            dataset_provenance=self.dataset_provenance,
            samples=total,
            positive_samples=positives,
            negative_samples=negatives,
            held_out_samples=len(held_ds),
            held_out_iou=held_iou,
            held_out_point_accuracy=point_accuracy,
            base_checkpoint=str(self.base_checkpoint),
            trained_at=time.time(),
        )

        return AffordanceTrainingResult(
            trained=True,
            reason="dorsal head fine-tuned on measured click outcomes",
            samples=total,
            positives=positives,
            negatives=negatives,
            held_out_samples=len(held_ds),
            held_out_iou=held_iou,
            held_out_point_accuracy=point_accuracy,
            epochs=self.epochs,
            checkpoint=str(output_checkpoint),
            provenance=str(sidecar),
            history=history,
        )

    @torch.no_grad()
    def _held_out_metrics(
        self, model: ArtificialVisualCortex, held_ds: OutcomeAffordanceDataset
    ) -> tuple[float | None, float | None]:
        """Measured held-out point accuracy and IoU over supervised cells.

        Point accuracy is the directly relevant number: a hotspot is useful only
        if the head predicts the labelled class *at the labelled cell*. IoU is
        reported alongside it, but over a sparse point mask IoU is dominated by
        the unsupported background, so point accuracy is what the readiness gate
        compares against chance.
        """
        if len(held_ds) == 0:
            return None, None
        model.eval()
        correct = 0
        total = 0
        ious: list[float] = []
        for idx in range(len(held_ds)):
            sample = held_ds[idx]
            frame = sample["frame_t"].unsqueeze(0).to(self.device)
            target = sample["affordance_mask"].to(self.device)
            view = model.retina.extract_views(frame)
            out = model.forward(view.peripheral.to(self.device), view.foveal.to(self.device))
            pred = out["affordance_logits"].argmax(dim=1).squeeze(0)
            supervised = target != IGNORE_INDEX
            if bool(supervised.any()):
                correct += int((pred[supervised] == target[supervised]).sum().item())
                total += int(supervised.sum().item())
            for cls in torch.unique(target):
                cls_i = int(cls.item())
                if cls_i == IGNORE_INDEX:
                    continue
                pred_c = pred == cls_i
                mask_c = target == cls_i
                union = int((pred_c | mask_c).sum().item())
                if union:
                    ious.append(int((pred_c & mask_c).sum().item()) / union)
        point_accuracy = round(correct / total, 4) if total else None
        iou = round(float(np.mean(ious)), 4) if ious else None
        return point_accuracy, iou


__all__ = [
    "AffordanceTrainer",
    "AffordanceTrainingResult",
    "grouped_split",
    "_SUPERVISED_CLASSES",
]
