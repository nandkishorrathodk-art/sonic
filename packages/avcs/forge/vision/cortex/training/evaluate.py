"""Held-out evaluation for the Artificial Visual Cortex.

The trainer reports *training* loss, which cannot tell you whether the model
generalizes. This module measures real held-out accuracy for the two heads that
make the cortex useful for computer-use (UI category and dense affordance) and
writes a provenance record next to the checkpoint.

Split discipline: the synthetic generator is seeded by sample index, so the
train and held-out sets use disjoint seed ranges. Nothing is shared.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from forge.vision.cortex.config import ModelConfig
from forge.vision.cortex.data.synthetic import ProceduralScreenGenerator
from forge.vision.cortex.model import ArtificialVisualCortex

# Seed ranges are disjoint so a held-out sample can never be a training sample.
TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 900_000


@dataclass
class CortexEvaluation:
    """Measured held-out metrics for a cortex checkpoint."""

    samples: int
    category_accuracy: float
    affordance_pixel_accuracy: float
    affordance_mean_iou: float
    latency_ms_mean: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def evaluate_cortex(
    model: ArtificialVisualCortex,
    samples: int = 64,
    device: str = "cpu",
    seed_base: int = EVAL_SEED_BASE,
) -> CortexEvaluation:
    """Measure held-out category + affordance accuracy. Real numbers only."""
    if samples <= 0:
        raise ValueError("Cortex evaluation requires a positive sample count")
    model = model.to(device)
    model.eval()
    model.reset_temporal_memory()

    generator = ProceduralScreenGenerator(width=640, height=360)
    target = device

    category_correct = 0
    afford_correct = 0
    afford_total = 0
    ious: list[float] = []
    latencies: list[float] = []

    for idx in range(samples):
        data = generator.generate(seed=seed_base + idx)
        frame = (
            torch.from_numpy(data.frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ).to(target)

        view = model.retina.extract_views(frame)
        _t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.forward(view.peripheral.to(target), view.foveal.to(target))
        latencies.append((time.perf_counter() - _t0) * 1000.0)

        pred_cat = int(out["category_logits"].argmax(dim=-1).item())
        if pred_cat == data.category_idx:
            category_correct += 1

        # Dorsal logits are (1, C, h, w); the ground-truth mask is downsampled to
        # the same grid with nearest-neighbour so every pixel is compared.
        logits = out["affordance_logits"]
        grid = logits.shape[-2:]
        mask = torch.from_numpy(data.affordance_mask).float()[None, None]
        mask = functional.interpolate(mask, size=grid, mode="nearest").squeeze().long().to(target)
        pred = logits.argmax(dim=1).squeeze(0)

        afford_correct += int((pred == mask).sum().item())
        afford_total += int(mask.numel())

        # Mean IoU over classes actually present in this sample.
        sample_ious: list[float] = []
        for cls in torch.unique(mask):
            cls_i = int(cls.item())
            pred_c = pred == cls_i
            mask_c = mask == cls_i
            union = int((pred_c | mask_c).sum().item())
            if union:
                sample_ious.append(int((pred_c & mask_c).sum().item()) / union)
        if sample_ious:
            ious.append(sum(sample_ious) / len(sample_ious))



    return CortexEvaluation(
        samples=samples,
        category_accuracy=round(category_correct / samples, 4),
        affordance_pixel_accuracy=round(afford_correct / max(afford_total, 1), 4),
        affordance_mean_iou=round(sum(ious) / max(len(ious), 1), 4),
        latency_ms_mean=round(sum(latencies) / max(len(latencies), 1), 2),
        device=device,
    )


def load_cortex_checkpoint(path: str | Path, device: str = "cpu") -> ArtificialVisualCortex:
    """Load a cortex checkpoint, failing closed on shape/format mismatch."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AVCS cortex checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    if not isinstance(state, dict):
        raise ValueError("AVCS cortex checkpoint must be a state_dict mapping")
    model = ArtificialVisualCortex(ModelConfig())
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def write_evaluation_sidecar(
    checkpoint_path: str | Path,
    evaluation: CortexEvaluation,
    dataset_provenance: str,
    *,
    dataset_note: str = "",
) -> Path:
    """Persist evaluation + provenance next to the checkpoint.

    ``dataset_provenance`` must say what the numbers were measured on. A metric
    without provenance is not evidence, so callers are forced to name the data.
    """
    checkpoint = Path(checkpoint_path)
    sidecar = checkpoint.with_suffix(".eval.json")
    payload = {
        "checkpoint": checkpoint.name,
        "dataset_provenance": dataset_provenance,
        "dataset_note": dataset_note,
        "evaluation": evaluation.to_dict(),
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out evaluation for AVCS cortex")
    parser.add_argument("--checkpoint", default="runs/cortex_model.pt")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--provenance",
        default="synthetic_procedural_screens",
        help="What the held-out data actually is (required for honesty).",
    )
    args = parser.parse_args()

    model = load_cortex_checkpoint(args.checkpoint, device=args.device)
    result = evaluate_cortex(model, samples=args.samples, device=args.device)
    print(json.dumps(result.to_dict(), indent=2))
    sidecar = write_evaluation_sidecar(
        args.checkpoint,
        result,
        dataset_provenance=args.provenance,
        dataset_note="Procedurally generated desktop screens; not real operator screens.",
    )
    print(f"Wrote evaluation sidecar: {sidecar}")


if __name__ == "__main__":
    main()
