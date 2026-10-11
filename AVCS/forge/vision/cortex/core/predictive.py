"""Temporal Predictive Coding & Latent World Model.

Implements the biological principle of predictive coding (Friston, 2005) and
self-supervised latent world modeling (V-JEPA):

1. Maintains a rolling visual latent state Z_t representing the screen across time.
2. Predicts future latent state Z_hat_{t+1} from current state and action history.
3. Computes Temporal Surprise: Error = ||Z_{t} - Z_hat_{t}||.
4. Categorizes screen stability:
   - STABLE: Screen settled, safe for agent actions.
   - ANIMATING: Minor ongoing transitions (cursor blinking, spinner).
   - TRANSITIONING: Major scene changes (page reload, app switch).
   - SURPRISE_EVENT: Unexpected visual emergence (modal popup, error toast).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STABILITY_STATES = [
    "stable",
    "animating",
    "transitioning",
    "surprise_event",
]


class PredictiveCodingEngine(nn.Module):
    """Predicts next visual latent and detects temporal surprises and stability."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Latent transition dynamics model (Mamba/GRU-style recurrent cell)
        self.transition = nn.GRUCell(latent_dim, latent_dim)

        # Predictor head: predicts residual change to latent state
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # Identity persistence prior: initialize residual weights to zero
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

        # Stability classifier based on prediction error and dynamics
        self.stability_head = nn.Linear(latent_dim + 1, len(STABILITY_STATES))

        self.prev_latent: Optional[torch.Tensor] = None
        self.predicted_latent: Optional[torch.Tensor] = None
        self.current_state: str = "stable"
        self.consecutive_surprises: int = 0

    def reset(self) -> None:
        """Reset internal temporal memory."""
        self.prev_latent = None
        self.predicted_latent = None
        self.current_state = "stable"
        self.consecutive_surprises = 0

    def forward(
        self,
        current_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process current screen latent and return temporal dynamics.

        Parameters
        ----------
        current_latent: (B, latent_dim)

        Returns
        -------
        updated_memory: (B, latent_dim)
        surprise_tensor: (B, 1)
        stability_logits: (B, 4)
        """
        B = current_latent.shape[0]

        # 1. Initialize memory if first frame
        if self.prev_latent is None or self.prev_latent.shape[0] != B:
            self.prev_latent = current_latent
            self.predicted_latent = current_latent
            self.consecutive_surprises = 0

        # 2. Compute Prediction Error (Surprise)
        norm_obs = F.normalize(current_latent, dim=-1)
        norm_pred = F.normalize(self.predicted_latent, dim=-1)
        surprise_tensor = 1.0 - (norm_obs * norm_pred).sum(dim=-1, keepdim=True)  # (B, 1)

        # 3. Anomaly-Gated Temporal Memory Update
        # Sudden acute spikes are dampened until confirmed across multiple ticks
        raw_memory = self.transition(current_latent, self.prev_latent)
        gate = torch.clamp(1.0 - 0.5 * surprise_tensor, 0.2, 1.0)
        updated_memory = gate * raw_memory + (1.0 - gate) * self.prev_latent

        # 4. Predict NEXT frame's latent state (residual around current memory)
        next_predicted = updated_memory + self.predictor(updated_memory)
        self.predicted_latent = next_predicted.detach()
        self.prev_latent = updated_memory.detach()

        # 5. Classify stability state
        combined = torch.cat([updated_memory, surprise_tensor], dim=-1)
        stability_logits = self.stability_head(combined)

        return updated_memory, surprise_tensor, stability_logits


def decode_stability(
    stability_logits: torch.Tensor,
    surprise_score: Optional[float] = None,
    prev_state: Optional[str] = None,
) -> str:
    """Decodes stability logits tensor into a human-readable state string.

    Uses Schmitt-trigger dual-threshold hysteresis when prev_state and surprise_score
    are provided to prevent high-frequency frame flicker.
    """
    raw_idx = int(stability_logits.argmax(dim=-1)[0].item())
    raw_state = STABILITY_STATES[raw_idx] if raw_idx < len(STABILITY_STATES) else "stable"

    if prev_state is None or surprise_score is None:
        return raw_state

    # Schmitt-Trigger Hysteresis:
    # High threshold to leave 'stable' (0.22), low threshold to return (0.08)
    if prev_state == "stable":
        if surprise_score > 0.55:
            return "surprise_event"
        if surprise_score > 0.22:
            return "animating" if raw_state in ["animating", "transitioning"] else raw_state
        return "stable"
    elif prev_state in ["animating", "transitioning"]:
        if surprise_score > 0.55:
            return "surprise_event"
        if surprise_score < 0.08:
            return "stable"
        return prev_state
    elif prev_state == "surprise_event":
        if surprise_score < 0.25:
            return "transitioning"
        return "surprise_event"

    return raw_state
