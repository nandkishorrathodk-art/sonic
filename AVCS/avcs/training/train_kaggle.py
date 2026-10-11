"""Synthetic-data smoke trainer for the experimental Cortex-2M architecture.

This script trains only on procedurally rendered colored blocks. Its metrics are
not evidence of real desktop grounding and its checkpoints are not production-ready.

The script supports GPU training on Kaggle, Google Colab, or local workstations:
- Multi-Task Optimization:
    1. Spatial Affordance Segmentation (24-class weighted BCE / Focal)
    2. Desktop Application Domain Classification (32-class Cross-Entropy)
    3. Natural Language Gaze Grounding (Spatial Gaussian MSE / DSNT)
- Accelerated Mixed Precision (torch.amp) with NVIDIA Tensor Core utilization
- Automatic Multi-GPU DataParallel scaling across Kaggle dual-T4 GPUs
- Exports experimental checkpoints with explicit synthetic-data provenance
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.cortex_2m import UnifiedCortex2M, UNIVERSAL_AFFORDANCE_CLASSES, UNIVERSAL_APP_CATEGORIES
from avcs.training.dataset import UniversalScreenDataset
from avcs.training.vicreg_safeguard import normalize_embedding, vicreg_variance_loss


def train_cortex_2m(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 5e-4,
    num_samples: int = 15000,
    val_samples: int = 2000,
    output_dir: str = "runs/synthetic_cortex_2m",
    use_amp: bool = True,
    device_str: Optional[str] = None,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Device Selection & Hardware Accelerators
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"

    print("=" * 78)
    print("      AVCS // EXPERIMENTAL SYNTHETIC-DATA SMOKE TRAINING                 ")
    print("=" * 78)
    print(f"Target Compute Device : {device.type.upper()} ({gpu_name})")
    print(f"Detected GPU Count    : {num_gpus}")
    print(f"Training Batch Size   : {batch_size}")
    print(f"Synthetic Train Samples: {num_samples:,} procedural colored-block scenes")
    print(f"Synthetic Val Samples  : {val_samples:,} procedural colored-block scenes")
    print(f"Total Epochs          : {epochs}")
    print(f"Mixed Precision (AMP) : {use_amp and device.type == 'cuda'}\n")

    # 2. Datasets & DataLoaders
    train_dataset = UniversalScreenDataset(num_samples=num_samples, seed=42)
    val_dataset = UniversalScreenDataset(num_samples=val_samples, seed=999)

    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. Model Architecture Initialization
    model = UnifiedCortex2M(feature_dim=256).to(device)
    print(f"Model Initialized     : UnifiedCortex2M with {model.total_parameters:,} parameters (~2.40M)")

    if num_gpus > 1:
        print(f"Enabling DataParallel across {num_gpus} GPUs...")
        model = nn.DataParallel(model)

    pos_weights = torch.tensor([3.0] * len(UNIVERSAL_AFFORDANCE_CLASSES), device=device).view(-1, 1, 1)
    affordance_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    category_loss_fn = nn.CrossEntropyLoss()
    grounding_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # 5. Training Loop
    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_aff_loss = 0.0
        total_cat_loss = 0.0
        total_ground_loss = 0.0
        correct_cats = 0
        total_items = 0

        t_epoch_start = time.perf_counter()

        for batch_i, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            aff_targets = batch["affordance_map"].to(device, non_blocking=True)
            cat_targets = batch["category"].to(device, non_blocking=True)
            text_embs = batch["text_embedding"].to(device, non_blocking=True)
            ground_targets = batch["grounding_target"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                out = model(imgs, text_emb=text_embs)
                loss_aff = affordance_loss_fn(out["affordance_logits"], aff_targets)
                loss_cat = category_loss_fn(out["category_logits"], cat_targets)
                loss_ground = grounding_loss_fn(out["grounding_map"], ground_targets)
                loss_var = vicreg_variance_loss(out["visual_latent_raw"], target_std=1.0)

                # Multi-task loss formulation with anti-collapse safeguard on pre-normalization projection
                loss = 1.0 * loss_aff + 0.5 * loss_cat + 2.0 * loss_ground + 1.0 * loss_var

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_aff_loss += loss_aff.item() * bs
            total_cat_loss += loss_cat.item() * bs
            total_ground_loss += loss_ground.item() * bs

            preds = torch.argmax(out["category_logits"], dim=1)
            correct_cats += int((preds == cat_targets).sum().item())
            total_items += bs

        scheduler.step()
        train_loss = total_loss / max(1, total_items)
        train_cat_acc = (correct_cats / max(1, total_items)) * 100.0

        # 6. Validation Evaluation Loop
        model.eval()
        val_loss = 0.0
        val_correct_cats = 0
        val_items = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                aff_targets = batch["affordance_map"].to(device, non_blocking=True)
                cat_targets = batch["category"].to(device, non_blocking=True)
                text_embs = batch["text_embedding"].to(device, non_blocking=True)
                ground_targets = batch["grounding_target"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    out = model(imgs, text_emb=text_embs)
                    loss_aff = affordance_loss_fn(out["affordance_logits"], aff_targets)
                    loss_cat = category_loss_fn(out["category_logits"], cat_targets)
                    loss_ground = grounding_loss_fn(out["grounding_map"], ground_targets)
                    loss = 1.0 * loss_aff + 0.5 * loss_cat + 2.0 * loss_ground

                bs = imgs.size(0)
                val_loss += loss.item() * bs
                preds = torch.argmax(out["category_logits"], dim=1)
                val_correct_cats += int((preds == cat_targets).sum().item())
                val_items += bs

        val_total_loss = val_loss / max(1, val_items)
        val_cat_acc = (val_correct_cats / max(1, val_items)) * 100.0
        epoch_sec = time.perf_counter() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_cat_acc": round(train_cat_acc, 2),
            "val_loss": round(val_total_loss, 4),
            "val_cat_acc": round(val_cat_acc, 2),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "epoch_sec": round(epoch_sec, 2),
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_cat_acc:4.1f}%) | "
            f"Val Loss: {val_total_loss:.4f} (Acc: {val_cat_acc:4.1f}%) | "
            f"LR: {epoch_record['lr']}"
        )

        # Save Best Checkpoint
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            raw_model = model.module if hasattr(model, "module") else model
            best_ckpt_path = out_path / "cortex_2m_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "val_loss": val_total_loss,
                "val_cat_acc": val_cat_acc,
                "total_parameters": raw_model.total_parameters,
                "universal_categories": UNIVERSAL_APP_CATEGORIES,
                "universal_affordances": UNIVERSAL_AFFORDANCE_CLASSES,
                "training_data_provenance": "procedural_synthetic_blocks",
                "is_production_ready": False,
                "metric_scope": "synthetic_validation_only",
            }, best_ckpt_path)
            print(f"  -> [SAVED BEST CHECKPOINT] Val Loss improved to {best_val_loss:.4f} -> {best_ckpt_path.name}")

    total_training_sec = time.perf_counter() - start_train_time
    print("\n" + "=" * 78)
    print(f"SYNTHETIC SMOKE TRAINING COMPLETED in {total_training_sec / 60.0:.2f} minutes.")
    print("This result does not certify real desktop grounding or GUI task completion.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Export history metadata JSON
    with open(out_path / "training_history.json", "w") as f:
        json.dump({
            "device": device.type,
            "gpu_name": gpu_name,
            "total_training_seconds": total_training_sec,
            "best_val_loss": best_val_loss,
            "training_data_provenance": "procedural_synthetic_blocks",
            "is_production_ready": False,
            "metric_scope": "synthetic_validation_only",
            "history": history,
        }, f, indent=2)

    print(f"Artifacts preserved in: {out_path.resolve()}")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental Cortex-2M synthetic-data smoke trainer"
    )
    parser.add_argument(
        "--synthetic-data-only",
        action="store_true",
        help="Acknowledge that this trains on synthetic colored blocks, not desktop demos",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--samples", type=int, default=15000, help="Training dataset sample count")
    parser.add_argument("--val-samples", type=int, default=2000, help="Validation sample count")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/synthetic_cortex_2m",
        help="Directory for experimental synthetic-data checkpoints",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.synthetic_data_only:
        parser.error(
            "this trainer uses synthetic colored blocks only; pass "
            "--synthetic-data-only to run a smoke experiment"
        )
    train_cortex_2m(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.samples,
        val_samples=args.val_samples,
        output_dir=args.output_dir,
        use_amp=True,
        device_str="cpu" if args.cpu else None,
    )


if __name__ == "__main__":
    main()
