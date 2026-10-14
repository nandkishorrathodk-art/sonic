"""Multi-Task Loss Functions for AVCS Training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.vision.cortex.config import TrainingConfig


class AVCSMultiTaskLoss(nn.Module):
    """Combines self-supervised temporal prediction, spatial affordance,

    semantic UI category, and language-guided coordinate grounding.
    """

    def __init__(self, cfg: TrainingConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TrainingConfig()

        self.ce_loss = nn.CrossEntropyLoss()
        self.coord_loss = nn.SmoothL1Loss()

    def forward(
        self,
        model_outputs_t: dict[str, torch.Tensor],
        latent_next: torch.Tensor,
        affordance_targets: torch.Tensor,
        category_targets: torch.Tensor,
        coord_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all task losses.

        Parameters
        ----------
        model_outputs_t: dict from model.forward(...) on frame_t
        latent_next: (B, latent_dim) observed latent of frame_{t+1}
        affordance_targets: (B, H, W) ground-truth affordance class indices
        category_targets: (B,) ground-truth UI category class indices
        coord_targets: (B, 2) ground-truth target coordinates
        """
        # 1. Self-Supervised Predictive World-Model Loss
        # Encourages temporal_memory to predict next frame's latent
        pred_memory = model_outputs_t["temporal_memory"]
        norm_pred = F.normalize(pred_memory, dim=-1)
        norm_next = F.normalize(latent_next.detach(), dim=-1)
        loss_predictive = (1.0 - (norm_pred * norm_next).sum(dim=-1)).mean()

        # 2. Dorsal Affordance Loss
        # Logits shape: (B, C, H, W), Targets: (B, H, W)
        affordance_logits = model_outputs_t["affordance_logits"]
        loss_affordance = self.ce_loss(affordance_logits, affordance_targets)

        # 3. Ventral UI Category Loss
        cat_logits = model_outputs_t["category_logits"]
        loss_category = self.ce_loss(cat_logits, category_targets)

        # 4. Language Grounding Coordinate Regression Loss
        query_coords = model_outputs_t["query_coords"]
        if query_coords is not None:
            loss_grounding = self.coord_loss(query_coords, coord_targets)
        else:
            loss_grounding = torch.tensor(0.0, device=affordance_logits.device)

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.cfg.predictive_loss_weight * loss_predictive
            + self.cfg.affordance_loss_weight * loss_affordance
            + self.cfg.category_loss_weight * loss_category
            + self.cfg.grounding_loss_weight * loss_grounding
        )

        return {
            "loss": total_loss,
            "loss_predictive": loss_predictive,
            "loss_affordance": loss_affordance,
            "loss_category": loss_category,
            "loss_grounding": loss_grounding,
        }

