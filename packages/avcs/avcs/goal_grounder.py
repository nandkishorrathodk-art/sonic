"""Goal-conditioned, pixel-only target grounding for AVCS."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F  # noqa: N812

from avcs.cortex_2m import CorticalBackbone2M
from avcs.human_eye import ConvBNAct

GROUNDING_MODEL_TYPE = "avcs_goal_conditioned_grounder"
GROUNDING_CHECKPOINT_SCHEMA = 2
DEFAULT_INPUT_SIZE = 256
MAX_QUERY_BYTES = 96
GROUNDING_ACCEPTANCE_METRIC = "top1_point_accuracy_within_0.02_diagonal"
GROUNDING_ACCEPTANCE_THRESHOLD = 0.85
GROUNDING_QUERY_REQUIRED_PRECISION = 0.9
GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES = 10


def encode_queries(queries: Sequence[str], max_length: int = MAX_QUERY_BYTES) -> torch.Tensor:
    """Encode query UTF-8 bytes as token IDs; zero is reserved for padding."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    tokens = torch.zeros((len(queries), max_length), dtype=torch.long)
    for row, query in enumerate(queries):
        encoded = query.strip().encode("utf-8")
        if not encoded:
            raise ValueError("Grounding queries must not be empty")
        if len(encoded) > max_length:
            raise ValueError(f"Grounding queries must be at most {max_length} UTF-8 bytes")
        tokens[row, : len(encoded)] = torch.tensor(
            [byte + 1 for byte in encoded], dtype=torch.long
        )
    return tokens


class ByteQueryEncoder(nn.Module):
    """Learned query encoder, independent of OCR/accessibility text extraction."""

    def __init__(self, output_dim: int = 192, embedding_dim: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        self.convolutions = nn.Sequential(
            nn.Conv1d(embedding_dim, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("query tokens must have shape (batch, non-empty sequence)")
        if torch.any(tokens < 0) or torch.any(tokens > 256):
            raise ValueError("query token IDs must be in [0, 256]")
        mask = tokens.ne(0).unsqueeze(1)
        if torch.any(~mask.any(dim=2)):
            raise ValueError("each query must contain at least one non-padding token")
        encoded = self.embedding(tokens).transpose(1, 2)
        encoded = self.convolutions(encoded)
        mask_float = mask.to(encoded.dtype)
        mean = (encoded * mask_float).sum(dim=2) / mask_float.sum(dim=2).clamp_min(1)
        maximum = encoded.masked_fill(~mask, torch.finfo(encoded.dtype).min).amax(dim=2)
        return self.projection(torch.cat((mean, maximum), dim=1))


class GoalConditionedGrounder(nn.Module):
    """Predict a dense click-target heatmap from RGB pixels and a textual subgoal.

    The model learns a location-conditioned compatibility score. It does not
    select high-level task steps, type input, or establish that a click is safe.
    """

    def __init__(
        self, feature_dim: int = 192, input_size: int = DEFAULT_INPUT_SIZE
    ) -> None:
        super().__init__()
        if input_size < 64 or input_size % 16:
            raise ValueError("input_size must be at least 64 and divisible by 16")
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.visual_encoder = CorticalBackbone2M(feature_dim=feature_dim)
        self.query_encoder = ByteQueryEncoder(output_dim=feature_dim)
        self.query_modulation = nn.Linear(feature_dim, feature_dim * 2)
        self.lateral_8 = ConvBNAct(128, 64, kernel_size=1, padding=0)
        self.lateral_4 = ConvBNAct(64, 64, kernel_size=1, padding=0)
        self.lateral_2 = ConvBNAct(32, 32, kernel_size=1, padding=0)
        self.decode_8 = ConvBNAct(feature_dim + 64, 96, kernel_size=3, padding=1)
        self.decode_4 = ConvBNAct(96 + 64, 64, kernel_size=3, padding=1)
        self.decode_2 = ConvBNAct(64 + 32, 64, kernel_size=3, padding=1)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        self.is_trained = False
        self.checkpoint_metadata: dict[str, object] = {}

    def forward(self, image: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (batch, 3, height, width)")
        if image.shape[0] != query_tokens.shape[0]:
            raise ValueError("image and query batch sizes must match")
        stem, stage1, stage2, _stage3, visual = self.visual_encoder.forward_pyramid(image)
        query = self.query_encoder(query_tokens)
        scale, bias = self.query_modulation(query).chunk(2, dim=1)
        visual = visual * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        decoded = F.interpolate(visual, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_8(torch.cat((decoded, self.lateral_8(stage2)), dim=1))
        decoded = F.interpolate(decoded, size=stage1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_4(torch.cat((decoded, self.lateral_4(stage1)), dim=1))
        decoded = F.interpolate(decoded, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_2(torch.cat((decoded, self.lateral_2(stem)), dim=1))
        return self.output(decoded)

    @torch.inference_mode()
    def predict(self, frame_rgb: np.ndarray, query: str) -> GroundingPrediction:
        """Return a candidate point only for a checkpoint trained on real demos."""
        if not self.is_trained:
            raise RuntimeError(
                "GoalConditionedGrounder has no demonstration-trained checkpoint; "
                "train it on recorded UI demonstrations before inference."
            )
        image, transform = letterbox_rgb(frame_rgb, self.input_size)
        device = next(self.parameters()).device
        logits = self(
            image.unsqueeze(0).to(device),
            encode_queries([query]).to(device),
        )
        probabilities = logits.flatten(1).softmax(dim=1).view_as(logits)
        flat_index = int(probabilities.flatten().argmax())
        _, _, heat_h, heat_w = probabilities.shape
        row, column = divmod(flat_index, heat_w)
        canvas_point = ((column + 0.5) / heat_w, (row + 0.5) / heat_h)
        point = transform.canvas_to_original(canvas_point)
        if not (0.0 <= point[0] < 1.0 and 0.0 <= point[1] < 1.0):
            point_result = None
        else:
            point_result = point
        return GroundingPrediction(
            point_norm=point_result,
            peak_probability=float(probabilities.flatten()[flat_index]),
            heatmap=probabilities[0, 0].detach().cpu(),
        )


@dataclass(frozen=True)
class LetterboxTransform:
    left_norm: float
    top_norm: float
    width_norm: float
    height_norm: float

    def original_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            self.left_norm + point[0] * self.width_norm,
            self.top_norm + point[1] * self.height_norm,
        )

    def canvas_to_original(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            (point[0] - self.left_norm) / self.width_norm,
            (point[1] - self.top_norm) / self.height_norm,
        )


@dataclass(frozen=True)
class GroundingPrediction:
    point_norm: Optional[tuple[float, float]]  # noqa: UP045
    peak_probability: float
    heatmap: torch.Tensor


def letterbox_rgb(
    frame_rgb: np.ndarray, input_size: int = DEFAULT_INPUT_SIZE
) -> tuple[torch.Tensor, LetterboxTransform]:
    """Resize an RGB frame without distorting its aspect ratio."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must have shape (height, width, 3)")
    if frame_rgb.dtype != np.uint8:
        raise ValueError("frame_rgb must contain uint8 RGB pixels")
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame_rgb must not be empty")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, min(input_size, round(width * scale)))
    resized_height = max(1, min(input_size, round(height * scale)))
    left = (input_size - resized_width) // 2
    top = (input_size - resized_height) // 2
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(frame_rgb, mode="RGB").resize(
        (resized_width, resized_height), resampling.BILINEAR
    )
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(resized)
    tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float().div_(255.0)
    transform = LetterboxTransform(
        left_norm=left / input_size,
        top_norm=top / input_size,
        width_norm=resized_width / input_size,
        height_norm=resized_height / input_size,
    )
    return tensor, transform


def click_heatmap(
    point_norm: tuple[float, float],
    output_size: tuple[int, int],
    sigma_cells: float = 1.5,
) -> torch.Tensor:
    """Build a normalized Gaussian target centered at a normalized click label."""
    x, y = point_norm
    height, width = output_size
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < 1 and 0 <= y < 1):
        raise ValueError("point_norm values must be finite and in [0, 1)")
    if height <= 0 or width <= 0 or sigma_cells <= 0:
        raise ValueError("output dimensions and sigma_cells must be positive")
    yy, xx = torch.meshgrid(
        (torch.arange(height, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    dist = (xx - x).square() * (width**2) + (yy - y).square() * (height**2)
    target = torch.exp(-dist / (2.0 * sigma_cells**2))
    return target / target.sum().clamp_min(torch.finfo(target.dtype).eps)


def load_grounding_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"  # noqa: UP007
) -> GoalConditionedGrounder:
    """Load only checkpoints carrying real-demonstration training provenance."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("grounding checkpoint must be a metadata dictionary")
    required = {
        "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": "recorded_ui_click_demonstrations",
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"grounding checkpoint has invalid {key!r} provenance")
    if int(checkpoint.get("training_samples", 0)) <= 0:
        raise ValueError("grounding checkpoint reports no training demonstrations")
    if int(checkpoint.get("validation_samples", 0)) <= 0:
        raise ValueError("grounding checkpoint reports no validation demonstrations")
    evaluation = checkpoint.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("grounding checkpoint is missing held-out test evaluation")
    if (
        evaluation.get("metric") != GROUNDING_ACCEPTANCE_METRIC
        or evaluation.get("passed") is not True
        or isinstance(evaluation.get("samples"), bool)
        or not isinstance(evaluation.get("samples"), int)
        or evaluation["samples"] <= 0
        or evaluation["samples"] != checkpoint.get("held_out_test_samples")
    ):
        raise ValueError("grounding checkpoint does not meet the held-out evaluation contract")
    accuracy = evaluation.get("value")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or float(accuracy) < GROUNDING_ACCEPTANCE_THRESHOLD
    ):
        raise ValueError("grounding checkpoint is below the held-out point-accuracy threshold")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise TypeError("grounding checkpoint is missing its model state dictionary")
    if checkpoint.get("coordinate_frame") != "cropped_target_window":
        raise ValueError("grounding checkpoint has an unsupported coordinate frame")
    confidence_calibration = checkpoint.get("query_confidence_calibration")
    confidence_evaluation = (
        confidence_calibration.get("evaluation")
        if isinstance(confidence_calibration, dict)
        else None
    )
    if not isinstance(confidence_calibration, dict) or not isinstance(
        confidence_evaluation, dict
    ):
        raise ValueError("grounding checkpoint is missing query confidence calibration")
    confidence_threshold = confidence_evaluation.get("threshold")
    confidence_precision = confidence_evaluation.get("precision")
    required_precision = confidence_evaluation.get("required_precision")
    confidence_samples = confidence_evaluation.get("samples")
    accepted_samples = confidence_evaluation.get("accepted_samples")
    minimum_accepted_samples = confidence_evaluation.get("minimum_accepted_samples")
    if (
        not isinstance(confidence_calibration.get("method"), str)
        or not confidence_calibration["method"].strip()
        or confidence_evaluation.get("metric") != "empirical_precision_at_threshold"
        or confidence_evaluation.get("passed") is not True
        or isinstance(confidence_samples, bool)
        or not isinstance(confidence_samples, int)
        or confidence_samples != checkpoint.get("held_out_test_samples")
        or isinstance(accepted_samples, bool)
        or not isinstance(accepted_samples, int)
        or isinstance(minimum_accepted_samples, bool)
        or not isinstance(minimum_accepted_samples, int)
        or minimum_accepted_samples < GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES
        or accepted_samples < minimum_accepted_samples
        or accepted_samples > confidence_samples
        or isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, (int, float))
        or not math.isfinite(float(confidence_threshold))
        or not 0 < float(confidence_threshold) <= 1
        or isinstance(confidence_precision, bool)
        or not isinstance(confidence_precision, (int, float))
        or not math.isfinite(float(confidence_precision))
        or not 0 <= float(confidence_precision) <= 1
        or isinstance(required_precision, bool)
        or not isinstance(required_precision, (int, float))
        or not math.isfinite(float(required_precision))
        or not GROUNDING_QUERY_REQUIRED_PRECISION
        <= float(required_precision)
        <= 1
        or float(confidence_precision) < float(required_precision)
    ):
        raise ValueError("grounding checkpoint fails query confidence calibration")
    input_size = checkpoint.get("input_size", DEFAULT_INPUT_SIZE)
    if isinstance(input_size, bool) or not isinstance(input_size, int):
        raise TypeError("grounding checkpoint has invalid input_size")
    model = GoalConditionedGrounder(input_size=input_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.is_trained = True
    model.checkpoint_metadata = {
        key: checkpoint[key]
        for key in (
            "dataset_provenance",
            "training_samples",
            "validation_samples",
            "held_out_test_samples",
            "evaluation",
            "query_confidence_calibration",
            "coordinate_frame",
        )
        if key in checkpoint
    }
    return model
