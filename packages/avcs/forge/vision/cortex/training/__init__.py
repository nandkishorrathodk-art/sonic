"""Training modules and utilities for AVCS."""
from forge.vision.cortex.training.losses import AVCSMultiTaskLoss
from forge.vision.cortex.training.trainer import AVCSCloudTrainer

__all__ = [
    "AVCSMultiTaskLoss",
    "AVCSCloudTrainer",
]

