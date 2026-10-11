"""Ventral Stream ("WHAT") - Semantic UI & Object Recognition.

In primate neurobiology, the ventral stream (inferotemporal pathway) specializes
in object identification, text recognition, shapes, and category semantics
("What is on the screen?").

For a computer screen, this module transforms visual features into:
1. High-level UI category logits (Modal, Editor, Navigation, Form, Table, etc.)
2. Text-density & code-region indicators
3. Compact semantic latent vector summarizing the scene's conceptual identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

UI_CATEGORIES = [
    "desktop_canvas",
    "editor_code",
    "browser_webpage",
    "terminal_shell",
    "modal_dialog",
    "form_input",
    "media_playback",
    "settings_panel",
]


class VentralSemanticStream(nn.Module):
    """Processes visual features into semantic UI categories and compact latents."""

    def __init__(self, in_dim: int = 128, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Global semantic pooling & projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # UI Category Classification Head
        self.category_head = nn.Linear(latent_dim, len(UI_CATEGORIES))

        # Text Density & Visual Complexity Indicator
        self.text_density_head = nn.Linear(latent_dim, 1)

    def forward(self, visual_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """visual_features: (B, in_dim, H, W)

        Returns:
            - latent: (B, latent_dim) - global semantic identity
            - category_logits: (B, num_categories)
            - text_density: (B, 1) - sigmoid in [0, 1]
        """
        pooled = self.pool(visual_features).flatten(1) # (B, in_dim)
        latent = self.proj(pooled)                      # (B, latent_dim)

        cat_logits = self.category_head(latent)
        text_density = torch.sigmoid(self.text_density_head(latent))

        return {
            "latent": latent,
            "category_logits": cat_logits,
            "text_density": text_density,
        }

    @torch.no_grad()
    def get_primary_category(self, visual_features: torch.Tensor) -> tuple[str, float]:
        """Returns the most likely UI category and confidence."""
        out = self.forward(visual_features)
        probs = torch.softmax(out["category_logits"], dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        return UI_CATEGORIES[idx], float(probs[idx].item())

