"""Local, pixel-only focus verification for text entry.

Before SONIC types into a control it must be able to answer, from the current
workstation frame alone, "is this point a focused text-entry control?".  This
module provides that capability as a small, locally trained classifier and a
checkpoint loader with the same fail-closed discipline as the goal grounder:

* the model abstains until it carries a checkpoint trained on *recorded* focus
  demonstrations with a held-out precision evaluation;
* readiness is proven from checkpoint metadata, never inferred from a file's
  existence or from configuration.

The verifier never sees the LLM and never uploads pixels.  It is a local head
that either returns a calibrated focused/not-focused verdict or fails closed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F  # noqa: N812

from avcs.cortex_2m import CorticalBackbone2M
from avcs.goal_grounder import ByteQueryEncoder

FOCUS_MODEL_TYPE = "avcs_focus_verifier"
FOCUS_CHECKPOINT_SCHEMA = 1
FOCUS_DATASET_PROVENANCE = "recorded_ui_focus_demonstrations"
FOCUS_ACCEPTANCE_METRIC = "focus_verification_precision"
FOCUS_ACCEPTANCE_THRESHOLD = 0.8
FOCUS_MIN_EVALUATION_SAMPLES = 30
DEFAULT_PATCH_FRACTION = 0.28
DEFAULT_INPUT_SIZE = 96


@dataclass(frozen=True)
class FocusPrediction:
    """A single focused/not-focused decision for one point on one frame."""

    focused: bool
    score: float
    threshold: float
    patch_bounds: tuple[int, int, int, int]


def _validate_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    for name, value in (("x", x), ("y", y)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"focus point {name} must be an integer pixel coordinate")
    if width <= 0 or height <= 0:
        raise ValueError("focus frame dimensions must be positive")
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("focus point must lie inside the frame")
    return x, y


def focus_patch_bounds(
    x: int,
    y: int,
    width: int,
    height: int,
    patch_fraction: float = DEFAULT_PATCH_FRACTION,
) -> tuple[int, int, int, int]:
    """Return the clamped square patch (left, top, right, bottom) around a point."""
    x, y = _validate_point(x, y, width, height)
    if not (0.0 < patch_fraction <= 1.0):
        raise ValueError("patch_fraction must be in (0, 1]")
    side = max(8, int(round(min(width, height) * patch_fraction)))
    half = side // 2
    left = min(max(0, x - half), max(0, width - side))
    top = min(max(0, y - half), max(0, height - side))
    right = min(width, left + side)
    bottom = min(height, top + side)
    return left, top, right, bottom


def focus_patch_rgb(
    frame_rgb: np.ndarray,
    x: int,
    y: int,
    input_size: int = DEFAULT_INPUT_SIZE,
    patch_fraction: float = DEFAULT_PATCH_FRACTION,
) -> torch.Tensor:
    """Crop and resize a square patch around a point into a normalized tensor."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must have shape (height, width, 3)")
    if frame_rgb.dtype != np.uint8:
        raise ValueError("frame_rgb must contain uint8 RGB pixels")
    if input_size < 16 or input_size % 16:
        raise ValueError("input_size must be at least 16 and divisible by 16")
    height, width = frame_rgb.shape[:2]
    left, top, right, bottom = focus_patch_bounds(
        x, y, width, height, patch_fraction=patch_fraction
    )
    patch = frame_rgb[top:bottom, left:right]
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(patch, mode="RGB").resize(
        (input_size, input_size), resampling.BILINEAR
    )
    tensor = torch.from_numpy(np.asarray(resized).copy())
    return tensor.permute(2, 0, 1).float().div_(255.0)


class FocusVerifier(nn.Module):
    """Classify whether a screen point is a focused text-entry control."""

    def __init__(self, feature_dim: int = 96, input_size: int = DEFAULT_INPUT_SIZE) -> None:
        super().__init__()
        if input_size < 16 or input_size % 16:
            raise ValueError("input_size must be at least 16 and divisible by 16")
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.visual_encoder = CorticalBackbone2M(feature_dim=feature_dim)
        self.query_encoder = ByteQueryEncoder(output_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.is_trained = False
        self.decision_threshold = 0.5
        self.checkpoint_metadata: dict[str, object] = {}

    def forward(self, image: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (batch, 3, height, width)")
        if image.shape[0] != query_tokens.shape[0]:
            raise ValueError("image and query batch sizes must match")
        *_stages, visual = self.visual_encoder.forward_pyramid(image)
        pooled = F.adaptive_avg_pool2d(visual, 1).flatten(1)
        query = self.query_encoder(query_tokens)
        logits = self.head(torch.cat((pooled, query), dim=1))
        return logits.squeeze(1)

    @torch.inference_mode()
    def predict(
        self, frame_rgb: np.ndarray, x: int, y: int, target: str
    ) -> FocusPrediction:
        """Return a focused/not-focused decision; requires a trained checkpoint."""
        if not self.is_trained:
            raise RuntimeError(
                "FocusVerifier has no demonstration-trained checkpoint; train it on "
                "recorded focus demonstrations before inference."
            )
        from avcs.goal_grounder import encode_queries

        patch = focus_patch_rgb(frame_rgb, x, y, input_size=self.input_size)
        device = next(self.parameters()).device
        logits = self(
            patch.unsqueeze(0).to(device),
            encode_queries([target]).to(device),
        )
        score = float(torch.sigmoid(logits).item())
        height, width = frame_rgb.shape[:2]
        bounds = focus_patch_bounds(x, y, width, height)
        return FocusPrediction(
            focused=score >= self.decision_threshold,
            score=score,
            threshold=self.decision_threshold,
            patch_bounds=bounds,
        )


def load_focus_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"  # noqa: UP007
) -> FocusVerifier:
    """Load only focus checkpoints carrying real demonstration provenance."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("focus checkpoint must be a metadata dictionary")
    if checkpoint.get("schema_version") != FOCUS_CHECKPOINT_SCHEMA:
        raise ValueError("focus checkpoint has an invalid schema version")
    if checkpoint.get("model_type") != FOCUS_MODEL_TYPE:
        raise ValueError("focus checkpoint has an invalid model type")
    if checkpoint.get("dataset_provenance") != FOCUS_DATASET_PROVENANCE:
        raise ValueError("focus checkpoint lacks recorded focus demonstration provenance")
    for key in ("training_samples", "validation_samples", "held_out_test_samples"):
        count = checkpoint.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"focus checkpoint has invalid {key!r}")
    evaluation = checkpoint.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("focus checkpoint is missing its held-out evaluation")
    precision = evaluation.get("precision")
    samples = evaluation.get("samples")
    if (
        evaluation.get("metric") != FOCUS_ACCEPTANCE_METRIC
        or evaluation.get("passed") is not True
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < FOCUS_MIN_EVALUATION_SAMPLES
        or samples != checkpoint.get("held_out_test_samples")
        or isinstance(precision, bool)
        or not isinstance(precision, (int, float))
        or not math.isfinite(float(precision))
        or float(precision) < FOCUS_ACCEPTANCE_THRESHOLD
    ):
        raise ValueError("focus checkpoint does not meet the held-out precision contract")
    threshold = checkpoint.get("decision_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 < float(threshold) <= 1.0
    ):
        raise ValueError("focus checkpoint has an invalid decision threshold")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise TypeError("focus checkpoint is missing its model state dictionary")
    input_size = checkpoint.get("input_size", DEFAULT_INPUT_SIZE)
    if isinstance(input_size, bool) or not isinstance(input_size, int):
        raise TypeError("focus checkpoint has an invalid input_size")
    model = FocusVerifier(input_size=input_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.is_trained = True
    model.decision_threshold = float(threshold)
    model.checkpoint_metadata = {
        key: checkpoint[key]
        for key in (
            "dataset_provenance",
            "training_samples",
            "validation_samples",
            "held_out_test_samples",
            "evaluation",
            "decision_threshold",
        )
        if key in checkpoint
    }
    return model
