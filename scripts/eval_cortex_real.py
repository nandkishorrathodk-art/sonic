"""Evaluate a cortex checkpoint on REAL held-out desktop frames.

Reports honest per-class accuracy and a confusion matrix for the UI-category
head on frames the model never trained on. Writes a provenance sidecar so a
metric cannot be recorded without naming the data it was measured on.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from forge.vision.cortex.config import ModelConfig
from forge.vision.cortex.core.ventral import UI_CATEGORIES
from forge.vision.cortex.model import ArtificialVisualCortex

W, H = 640, 360


def _load(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H))
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", default="", help="provenance sidecar path")
    args = ap.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = data["frames"]

    cfg = ModelConfig()
    model = ArtificialVisualCortex(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    model.reset_temporal_memory()

    correct = 0
    total = 0
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latencies: list[float] = []

    for item in frames:
        label = item["label"]
        if label not in UI_CATEGORIES:
            continue
        target_idx = UI_CATEGORIES.index(label)
        frame = _load(item["frame"])
        view = model.retina.extract_views(frame)
        t0 = time.perf_counter()
        out = model.forward(view.peripheral, view.foveal)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        logits = out["category_logits"]
        pred_idx = int(logits.argmax(dim=-1).item())
        pred_label = UI_CATEGORIES[pred_idx]

        total += 1
        confusion[label][pred_label] += 1
        per_class[label][1] += 1
        if pred_idx == target_idx:
            correct += 1
            per_class[label][0] += 1

    acc = correct / total if total else 0.0
    print(f"\nReal held-out frames: {total}")
    print(f"Category accuracy   : {acc:.3f}  (8-way chance = 0.125)")
    print(f"Latency             : {np.mean(latencies):.1f} ms/frame (cpu)")
    print("\nPer-class recall:")
    for label in sorted(per_class):
        c, t = per_class[label]
        print(f"  {label:18s} {c}/{t} = {c/t:.2f}")
    print("\nConfusion (true -> predicted):")
    for label in sorted(confusion):
        row = ", ".join(f"{k}:{v}" for k, v in sorted(confusion[label].items()))
        print(f"  {label:18s} -> {row}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "dataset_provenance": f"real desktop capture: {args.manifest}",
            "checkpoint": str(args.checkpoint),
            "samples": total,
            "category_accuracy": acc,
            "chance_accuracy": 0.125,
            "latency_ms_mean": float(np.mean(latencies)),
            "per_class": {k: {"correct": v[0], "total": v[1]} for k, v in per_class.items()},
            "confusion": {k: dict(v) for k, v in confusion.items()},
        }, indent=2))
        print(f"\nProvenance written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
