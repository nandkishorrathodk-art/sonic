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

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        # Separate loss for the category head so unlabeled samples (index -1)
        # are ignored instead of trained as a fabricated class.
        self.category_loss = nn.CrossEntropyLoss(ignore_index=-1)
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
        # Logits shape: (B, C, H, W), Targets: (B, H, W).  ``ignore_index=-100``
        # lets real-desktop frames without a hand-labeled affordance mask
        # contribute nothing instead of training on invented click regions.
        # When a batch is entirely masked, CrossEntropyLoss returns nan;
        # substitute a real zero so the term is honestly absent.
        affordance_logits = model_outputs_t["affordance_logits"]
        if bool((affordance_targets != -100).any()):
            loss_affordance = self.ce_loss(affordance_logits, affordance_targets)
        else:
            loss_affordance = torch.zeros(
                (), device=affordance_logits.device, dtype=affordance_logits.dtype
            )

        # 3. Ventral UI Category Loss
        # ``ignore_index=-1`` lets unlabeled data (e.g. watched-video frames,
        # which carry no UI category) contribute nothing to this term instead of
        # training on a fabricated class. Existing labeled data never uses -1,
        # so this is a no-op for them. When a batch is *entirely* unlabeled,
        # CrossEntropyLoss returns nan; substitute a real zero so the category
        # term is honestly absent rather than poisoning the total loss.
        cat_logits = model_outputs_t["category_logits"]
        if bool((category_targets != -1).any()):
            loss_category = self.category_loss(cat_logits, category_targets)
        else:
            loss_category = torch.zeros((), device=cat_logits.device, dtype=cat_logits.dtype)

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

