"""Configuration for Artificial Visual Cortex for Screens (AVCS)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RetinaConfig:
    peripheral_size: Tuple[int, int] = (256, 256)
    foveal_size: Tuple[int, int] = (256, 256)
    foveal_coverage_ratio: float = 0.35
    saccade_smoothing: float = 0.7


@dataclass
class ModelConfig:
    feature_dim: int = 128
    latent_dim: int = 256
    in_channels: int = 3
    num_affordance_classes: int = 5
    num_ui_categories: int = 8
    retina: RetinaConfig = field(default_factory=RetinaConfig)


@dataclass
class TrainingConfig:
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    predictive_loss_weight: float = 1.0
    affordance_loss_weight: float = 1.0
    category_loss_weight: float = 0.5
    grounding_loss_weight: float = 1.0
    use_amp: bool = True
    device: str = "cuda" # Automatically falls back to cpu if unavailable


@dataclass
class CortexConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

