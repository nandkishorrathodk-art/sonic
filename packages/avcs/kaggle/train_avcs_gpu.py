"""Self-Contained Cloud GPU Training Script for Artificial Visual Cortex (AVCS).

Optimized for Kaggle (Dual T4), Google Colab, RunPod, or AWS EC2 instances.
Trains the AVCS model end-to-end on synthetic and procedural screen streams.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# If running directly inside Kaggle without local forge directory
if not os.path.exists("forge") and os.path.exists("/kaggle"):
    print("Cloning latest forge-screen-vision branch inside Kaggle...")
    os.system("git clone -b forge-screen-vision https://github.com/nandkishorrathodk-art/primeASI.git /kaggle/working/repo")
    if os.path.exists("/kaggle/working/repo/forge"):
        import shutil
        shutil.copytree("/kaggle/working/repo/forge", "/kaggle/working/forge", dirs_exist_ok=True)
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")

from forge.vision.cortex import (
    ArtificialVisualCortex,
    CortexConfig,
    create_dataloader,
    export_onnx,
    export_torchscript,
)
from forge.vision.cortex.training.losses import AVCSMultiTaskLoss


def run_cloud_gpu_training(
    epochs: int = 15,
    batch_size: int = 32,
    train_samples: int = 3000,
    val_samples: int = 400,
    lr: float = 5e-4,
    save_dir: str = "runs",
):
    print("=" * 70)
    print("       AVCS CLOUD GPU PRE-TRAINING ENGINE (V-JEPA + AFFORDANCE)      ")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} (x{gpu_count}) | Total VRAM: {vram_gb:.1f} GB")
    else:
        print("WARNING: CUDA not detected. Running on CPU.")

    os.makedirs(save_dir, exist_ok=True)

    # 1. Config & Model
    cfg = CortexConfig()
    cfg.model.feature_dim = 128
    cfg.model.latent_dim = 256
    cfg.training.lr = lr
    cfg.training.epochs = epochs

    model = ArtificialVisualCortex(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    # 2. DataLoaders
    print(f"\nGenerating {train_samples} training samples & {val_samples} validation samples...")
    train_loader = create_dataloader(num_samples=train_samples, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(num_samples=val_samples, batch_size=batch_size, shuffle=False)

    # 3. Loss & Optimizer
    loss_fn = AVCSMultiTaskLoss(cfg.training)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\nStarting {epochs} epochs of multi-task training (AMP FP16: {use_amp})...\n")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        t0 = time.perf_counter()
        train_loss = 0.0
        train_pred = 0.0
        train_aff = 0.0
        train_cat = 0.0
        train_grd = 0.0
        steps = 0

        for batch in train_loader:
            f_t = batch["frame_t"].to(device)
            f_next = batch["frame_next"].to(device)
            aff_target = batch["affordance_mask"].to(device)
            cat_target = batch["category_idx"].to(device)
            coord_target = batch["target_coord"].to(device)
            q_embed = batch["query_embed"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Retinal Foveation
                view_t = model.retina.extract_views(f_t)
                p_t, fov_t = view_t.peripheral.to(device), view_t.foveal.to(device)

                view_next = model.retina.extract_views(f_next)
                p_next, fov_next = view_next.peripheral.to(device), view_next.foveal.to(device)

                # Forward pass on t
                out_t = model(p_t, fov_t, text_embedding=q_embed)

                # Forward pass on t+1 for self-supervised predictive target
                with torch.no_grad():
                    out_next = model(p_next, fov_next)
                    latent_next = out_next["scene_latent"]

                losses = loss_fn(out_t, latent_next, aff_target, cat_target, coord_target)
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pred += losses["loss_predictive"].item()
            train_aff += losses["loss_affordance"].item()
            train_cat += losses["loss_category"].item()
            train_grd += losses["loss_grounding"].item()
            steps += 1

        scheduler.step()
        train_time = time.perf_counter() - t0

        # --- EVALUATION ---
        model.eval()
        val_loss = 0.0
        correct_cat = 0
        total_cat = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                f_t = batch["frame_t"].to(device)
                f_next = batch["frame_next"].to(device)
                aff_target = batch["affordance_mask"].to(device)
                cat_target = batch["category_idx"].to(device)
                coord_target = batch["target_coord"].to(device)
                q_embed = batch["query_embed"].to(device)

                view_t = model.retina.extract_views(f_t)
                p_t, fov_t = view_t.peripheral.to(device), view_t.foveal.to(device)

                view_next = model.retina.extract_views(f_next)
                p_next, fov_next = view_next.peripheral.to(device), view_next.foveal.to(device)

                out_t = model(p_t, fov_t, text_embedding=q_embed)
                out_next = model(p_next, fov_next)

                losses = loss_fn(out_t, out_next["scene_latent"], aff_target, cat_target, coord_target)
                val_loss += losses["loss"].item()

                preds = out_t["category_logits"].argmax(dim=-1)
                correct_cat += (preds == cat_target).sum().item()
                total_cat += cat_target.size(0)
                val_steps += 1

        avg_train = train_loss / max(steps, 1)
        avg_val = val_loss / max(val_steps, 1)
        cat_acc = (correct_cat / max(total_cat, 1)) * 100.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train:.4f} (Pred: {train_pred/steps:.3f}, Aff: {train_aff/steps:.3f}) | "
            f"Val Loss: {avg_val:.4f} | "
            f"Cat Acc: {cat_acc:.1f}% | "
            f"Time: {train_time:.1f}s"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, "avcs_best.pt"))

    print("\n" + "=" * 70)
    print("TRAINING FINISHED! EXPORTING TRAINED CORTEX ARTIFACTS...")
    print("=" * 70)

    # Export to TorchScript and ONNX
    pt_path = os.path.join(save_dir, "avcs_cortex.pt")
    onnx_path = os.path.join(save_dir, "avcs_cortex.onnx")

    export_torchscript(model, save_path=pt_path)
    export_onnx(model, save_path=onnx_path)

    # Run Cloud GPU latency verification
    print("\nPROFILING TRAINED MODEL ON THIS HARDWARE:")
    model.eval()
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = model.perceive(dummy_frame, device=device)

    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        _ = model.perceive(dummy_frame, device=device)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    p50 = np.median(latencies)
    fps = 1000.0 / p50
    print(f"  • Final Hardware p50 Latency: {p50:.2f} ms")
    print(f"  • Sustained Throughput:       {fps:.1f} FPS")
    print("=" * 70)


if __name__ == "__main__":
    run_cloud_gpu_training(epochs=5, batch_size=16, train_samples=300, val_samples=50)

