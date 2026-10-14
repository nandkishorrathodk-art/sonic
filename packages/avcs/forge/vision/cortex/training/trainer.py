"""Cloud GPU Trainer for Artificial Visual Cortex (AVCS)."""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from forge.vision.cortex.config import CortexConfig, TrainingConfig
from forge.vision.cortex.data.dataset import ScreenDataset, create_dataloader
from forge.vision.cortex.model import ArtificialVisualCortex
from forge.vision.cortex.training.losses import AVCSMultiTaskLoss


class AVCSCloudTrainer:
    """Trains the Artificial Visual Cortex on Cloud GPU or local host."""

    def __init__(
        self,
        model: Optional[ArtificialVisualCortex] = None,
        config: Optional[CortexConfig] = None,
        dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config or CortexConfig()
        self.train_cfg = self.config.training

        # Select Device
        if torch.cuda.is_available() and self.train_cfg.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize Model & Loss
        self.model = (model or ArtificialVisualCortex(self.config)).to(self.device)
        self.loss_fn = AVCSMultiTaskLoss(self.train_cfg)

        # DataLoader
        self.dataloader = dataloader or create_dataloader(
            num_samples=200,
            batch_size=self.train_cfg.batch_size,
        )

        # Optimizer & LR Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.train_cfg.epochs, 1)
        )

        self.use_amp = self.train_cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_pred = 0.0
        total_aff = 0.0
        total_cat = 0.0
        total_grd = 0.0
        steps = 0

        t0 = time.perf_counter()

        for batch in self.dataloader:
            frame_t = batch["frame_t"].to(self.device)
            frame_next = batch["frame_next"].to(self.device)
            afford_target = batch["affordance_mask"].to(self.device)
            cat_target = batch["category_idx"].to(self.device)
            coord_target = batch["target_coord"].to(self.device)
            query_embed = batch["query_embed"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 1. Extract views for frame_t
                view_t = self.model.retina.extract_views(frame_t)
                p_t = view_t.peripheral.to(self.device)
                f_t = view_t.foveal.to(self.device)

                # 2. Extract views for frame_next (target for predictive coding)
                view_next = self.model.retina.extract_views(frame_next)
                p_next = view_next.peripheral.to(self.device)
                f_next = view_next.foveal.to(self.device)

                # 3. Model forward pass on frame_t
                out_t = self.model(p_t, f_t, text_embedding=query_embed)

                # 4. Model forward pass on frame_next (target latent)
                with torch.no_grad():
                    out_next = self.model(p_next, f_next)
                    latent_next = out_next["scene_latent"]

                # 5. Compute Multi-Task Loss
                loss_dict = self.loss_fn(
                    out_t, latent_next, afford_target, cat_target, coord_target
                )
                loss = loss_dict["loss"]

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_pred += loss_dict["loss_predictive"].item()
            total_aff += loss_dict["loss_affordance"].item()
            total_cat += loss_dict["loss_category"].item()
            total_grd += loss_dict["loss_grounding"].item()
            steps += 1

        self.scheduler.step()
        dt = time.perf_counter() - t0

        metrics = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "loss_predictive": total_pred / max(steps, 1),
            "loss_affordance": total_aff / max(steps, 1),
            "loss_category": total_cat / max(steps, 1),
            "loss_grounding": total_grd / max(steps, 1),
            "time_sec": dt,
        }
        return metrics

    def train(self, epochs: Optional[int] = None, save_path: str = "runs/cortex_model.pt") -> None:
        num_epochs = epochs or self.train_cfg.epochs
        print(f"Starting AVCS Training on {self.device} for {num_epochs} epochs...")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        for ep in range(1, num_epochs + 1):
            m = self.train_epoch(ep)
            print(
                f"[Epoch {ep:02d}/{num_epochs:02d}] "
                f"Loss: {m['loss']:.4f} | "
                f"Pred: {m['loss_predictive']:.4f} | "
                f"Aff: {m['loss_affordance']:.4f} | "
                f"Cat: {m['loss_category']:.4f} | "
                f"Grd: {m['loss_grounding']:.4f} | "
                f"Time: {m['time_sec']:.2f}s"
            )

        torch.save(self.model.state_dict(), save_path)
        print(f"Training Complete! Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()

    cfg = CortexConfig()
    cfg.training.batch_size = args.batch_size
    cfg.training.epochs = args.epochs

    dl = create_dataloader(num_samples=args.samples, batch_size=args.batch_size)
    trainer = AVCSCloudTrainer(config=cfg, dataloader=dl)
    trainer.train(epochs=args.epochs)

