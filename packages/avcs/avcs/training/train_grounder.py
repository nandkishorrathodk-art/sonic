"""Train AVCS goal-conditioned grounding on recorded UI click demonstrations."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    GROUNDING_ACCEPTANCE_METRIC,
    GROUNDING_ACCEPTANCE_THRESHOLD,
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_MODEL_TYPE,
    GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES,
    GROUNDING_QUERY_REQUIRED_PRECISION,
    GoalConditionedGrounder,
)
from avcs.training.demonstrations import (
    RECORDED_PROVENANCE,
    DemonstrationClickDataset,
    discover_demonstrations,
    split_sessions,
)

_MIN_SHM_FOR_WORKERS_BYTES = 512 * 1024 * 1024


def _resolve_loader_workers() -> int:
    """Pick a DataLoader worker count that cannot exhaust shared memory.

    ``num_workers=2`` crashes with "unable to allocate shared memory" inside a
    container whose ``/dev/shm`` is the Docker default of 64 MB, which is the
    common case -- so training silently failed to start in exactly the
    environment it is meant to run in. Fall back to in-process loading unless
    ``/dev/shm`` is comfortably large (or the caller overrides via
    ``SONIC_AVCS_DATALOADER_WORKERS``).
    """
    override = os.environ.get("SONIC_AVCS_DATALOADER_WORKERS", "").strip()
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            pass
    if os.name == "nt":
        return 0
    try:
        stats = os.statvfs("/dev/shm")
        available = stats.f_bavail * stats.f_frsize
    except (OSError, AttributeError):
        return 0
    return 2 if available >= _MIN_SHM_FOR_WORKERS_BYTES else 0


def _resolve_dataset_provenance(records: list[Any]) -> str:
    """Return the single provenance shared by every record, or ``"mixed"``."""
    provenances = {
        getattr(record, "provenance", RECORDED_PROVENANCE) for record in records
    }
    if len(provenances) == 1:
        return next(iter(provenances))
    return "mixed"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_canvas_points(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("grounder logits must have shape (batch, 1, height, width)")
    height, width = logits.shape[-2:]
    indices = logits[:, 0].flatten(1).argmax(dim=1)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices.remainder(width)
    x = (columns.to(torch.float32) + 0.5) / width
    y = (rows.to(torch.float32) + 0.5) / height
    return torch.stack((x, y), dim=1)


def _batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    original_points: torch.Tensor,
    letterbox: torch.Tensor,
    *,
    hit_radii: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> dict[str, torch.Tensor]:
    log_probabilities = logits[:, 0].flatten(1).log_softmax(dim=1)
    loss = -(targets.flatten(1) * log_probabilities).sum(dim=1).mean()
    canvas_points = _decode_canvas_points(logits)
    original_x = (canvas_points[:, 0] - letterbox[:, 0]) / letterbox[:, 2]
    original_y = (canvas_points[:, 1] - letterbox[:, 1]) / letterbox[:, 3]
    decoded = torch.stack((original_x, original_y), dim=1)
    delta = decoded - original_points
    aspect_ratio = letterbox[:, 2] / letterbox[:, 3]
    diagonal_distance = torch.sqrt(
        (delta[:, 0] * aspect_ratio).square() + delta[:, 1].square()
    ) / torch.sqrt(aspect_ratio.square() + 1.0)
    metrics = {
        "loss": loss,
        "mean_normalized_click_distance": diagonal_distance.mean(),
    }
    for radius in hit_radii:
        if not 0 < radius < 1:
            raise ValueError("hit_radii values must be in (0, 1)")
        metrics[f"top1_point_accuracy_within_{radius:g}_diagonal"] = (
            (diagonal_distance <= radius).to(torch.float32).mean()
        )
    return metrics


def _run_epoch(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],  # noqa: UP045
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_error = 0.0
    total_hits = {radius: 0.0 for radius in (0.01, 0.02, 0.05)}
    total_samples = 0
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch in loader:
            image = batch["image"].to(device)
            queries = batch["query_tokens"].to(device)
            target = batch["target_heatmap"].to(device)
            points = batch["point_norm"].to(device)
            letterbox = batch["letterbox"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image, queries)
            metrics = _batch_metrics(logits, target, points, letterbox)
            loss = metrics["loss"]
            if any(not torch.isfinite(value) for value in metrics.values()):
                raise RuntimeError("Grounding epoch produced non-finite metrics")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = image.shape[0]
            total_loss += float(metrics["loss"].detach()) * batch_size
            total_error += float(metrics["mean_normalized_click_distance"].detach()) * batch_size
            for radius in total_hits:
                metric_name = f"top1_point_accuracy_within_{radius:g}_diagonal"
                total_hits[radius] += float(metrics[metric_name].detach()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("A training or validation split contained no samples")
    result = {
        "loss": total_loss / total_samples,
        "mean_normalized_click_distance": total_error / total_samples,
        "samples": float(total_samples),
    }
    for radius, hits in total_hits.items():
        result[f"top1_point_accuracy_within_{radius:g}_diagonal"] = hits / total_samples
    return result


def _confidence_examples(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[bool]]:
    """Collect peak heatmap scores paired with held-out point-localization hits."""
    scores: list[float] = []
    hits: list[bool] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["query_tokens"].to(device))
            peak_scores = logits[:, 0].flatten(1).softmax(dim=1).amax(dim=1)
            canvas_points = _decode_canvas_points(logits)
            letterbox = batch["letterbox"].to(device)
            original_points = batch["point_norm"].to(device)
            original_x = (canvas_points[:, 0] - letterbox[:, 0]) / letterbox[:, 2]
            original_y = (canvas_points[:, 1] - letterbox[:, 1]) / letterbox[:, 3]
            delta = torch.stack((original_x, original_y), dim=1) - original_points
            aspect_ratio = letterbox[:, 2] / letterbox[:, 3]
            diagonal_distance = torch.sqrt(
                (delta[:, 0] * aspect_ratio).square() + delta[:, 1].square()
            ) / torch.sqrt(aspect_ratio.square() + 1.0)
            scores.extend(float(score) for score in peak_scores.cpu())
            hits.extend(bool(hit) for hit in (diagonal_distance <= 0.02).cpu())
    return scores, hits


def _select_confidence_threshold(
    scores: list[float],
    hits: list[bool],
    *,
    required_precision: float,
) -> tuple[float, int, float]:
    """Select maximum validation coverage subject to the requested precision."""
    if len(scores) != len(hits):
        raise ValueError("confidence scores and correctness labels must have equal length")
    best: tuple[float, int, float] | None = None
    for threshold in sorted(set(scores)):
        accepted = [hit for score, hit in zip(scores, hits) if score >= threshold]
        if len(accepted) < GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES:
            continue
        precision = sum(accepted) / len(accepted)
        if precision >= required_precision and (
            best is None or len(accepted) > best[1]
        ):
            best = (threshold, len(accepted), precision)
    return best if best is not None else (1.0, 0, 0.0)


def train_grounder(
    *,
    data_dir: str,
    output_dir: str = "runs/real_ui_grounder",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    input_size: int = DEFAULT_INPUT_SIZE,
    seed: int = 42,
    device_name: Optional[str] = None,  # noqa: UP045
    overwrite: bool = False,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if input_size < 64 or input_size % 16:
        raise ValueError("input_size must be at least 64 and divisible by 16")

    sessions = discover_demonstrations(data_dir)
    task_window_groups = {
        (records[0].task.casefold(), records[0].target_window_title.casefold())
        for records in sessions.values()
        if records
    }
    train_records, validation_records, test_records = split_sessions(sessions, seed=seed)
    train_data = DemonstrationClickDataset(train_records, input_size=input_size)
    validation_data = DemonstrationClickDataset(validation_records, input_size=input_size)
    loader_workers = _resolve_loader_workers()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
    )
    test_data = DemonstrationClickDataset(test_records, input_size=input_size)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
    )

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Provenance is derived from the data, never assumed. A checkpoint trained on
    # anything other than the real recorder's sessions must not claim the
    # real-demonstration provenance, because the loader gates on that string.
    dataset_provenance = _resolve_dataset_provenance(
        train_records + validation_records + test_records
    )
    if dataset_provenance != RECORDED_PROVENANCE:
        print(
            f"WARNING: training on non-recorder data (provenance="
            f"{dataset_provenance!r}). The resulting checkpoint will carry that "
            "provenance and is not a real-demonstration grounder."
        )

    _seed_everything(seed)
    model = GoalConditionedGrounder(input_size=input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "goal_grounder_best.pt"
    report_path = output_path / "training_report.json"
    if not overwrite and (checkpoint_path.exists() or report_path.exists()):
        raise FileExistsError(
            f"Training outputs already exist in {output_path}; use a new directory "
            "or pass --overwrite."
        )

    history: list[dict[str, float]] = []
    best_validation_loss = math.inf
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        validation_metrics = _run_epoch(model, validation_loader, None, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_mean_normalized_click_distance": train_metrics[
                    "mean_normalized_click_distance"
                ],
                "validation_loss": validation_metrics["loss"],
                "validation_mean_normalized_click_distance": validation_metrics[
                    "mean_normalized_click_distance"
                ],
                "validation_top1_point_accuracy_within_0.02_diagonal": (
                    validation_metrics["top1_point_accuracy_within_0.02_diagonal"]
                ),
            }
        )
        print(
            f"epoch {epoch:03d}/{epochs}: "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            "val_mean_normalized_click_distance="
            f"{validation_metrics['mean_normalized_click_distance']:.5f} "
            "val_top1_point_accuracy@2%="
            f"{validation_metrics['top1_point_accuracy_within_0.02_diagonal']:.3f}"
        )
        if math.isfinite(validation_metrics["loss"]) and (
            validation_metrics["loss"] < best_validation_loss
        ):
            best_validation_loss = validation_metrics["loss"]
            checkpoint = {
                "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
                "model_type": GROUNDING_MODEL_TYPE,
                "dataset_provenance": dataset_provenance,
                "training_samples": len(train_records),
                "validation_samples": len(validation_records),
                "held_out_test_samples": len(test_records),
                "input_size": input_size,
                "model_state_dict": model.state_dict(),
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)

    if not math.isfinite(best_validation_loss) or not checkpoint_path.is_file():
        raise RuntimeError("No finite validation checkpoint was produced")

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    test_metrics = _run_epoch(model, test_loader, None, device)
    held_out_test_metrics = {key: value for key, value in test_metrics.items() if key != "samples"}
    acceptance_value = held_out_test_metrics[GROUNDING_ACCEPTANCE_METRIC]
    evaluation = {
        "metric": GROUNDING_ACCEPTANCE_METRIC,
        "value": acceptance_value,
        "minimum": GROUNDING_ACCEPTANCE_THRESHOLD,
        "passed": acceptance_value >= GROUNDING_ACCEPTANCE_THRESHOLD,
        "samples": int(test_metrics["samples"]),
    }
    best_checkpoint["evaluation"] = evaluation
    validation_scores, validation_hits = _confidence_examples(
        model, validation_loader, device
    )
    confidence_threshold, validation_accepted, validation_precision = (
        _select_confidence_threshold(
            validation_scores,
            validation_hits,
            required_precision=GROUNDING_QUERY_REQUIRED_PRECISION,
        )
    )
    held_out_scores, held_out_hits = _confidence_examples(model, test_loader, device)
    held_out_accepted = [
        hit
        for score, hit in zip(held_out_scores, held_out_hits)
        if score >= confidence_threshold
    ]
    held_out_precision = (
        sum(held_out_accepted) / len(held_out_accepted)
        if held_out_accepted
        else 0.0
    )
    confidence_passed = (
        validation_accepted >= GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES
        and len(held_out_accepted) >= GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES
        and held_out_precision >= GROUNDING_QUERY_REQUIRED_PRECISION
    )
    confidence_calibration = {
        "method": "validation_selected_peak_score_threshold",
        "evaluation": {
            "metric": "empirical_precision_at_threshold",
            "samples": len(held_out_scores),
            "accepted_samples": len(held_out_accepted),
            "threshold": confidence_threshold,
            "precision": held_out_precision,
            "required_precision": GROUNDING_QUERY_REQUIRED_PRECISION,
            "minimum_accepted_samples": GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES,
            "passed": confidence_passed,
        },
        "validation": {
            "samples": len(validation_scores),
            "accepted_samples": validation_accepted,
            "precision": validation_precision,
        },
    }
    best_checkpoint["query_confidence_calibration"] = confidence_calibration
    best_checkpoint["coordinate_frame"] = "cropped_target_window"
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(best_checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    report = {
        "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": dataset_provenance,
        "session_split": "task_and_window_disjoint",
        "recording_sessions": len(sessions),
        "task_window_groups": len(task_window_groups),
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "held_out_test_samples": len(test_records),
        "input_size": input_size,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "best_validation_loss": best_validation_loss,
        "validation_metric": "loss and Top-1 point localization accuracy",
        "held_out_test_metrics": held_out_test_metrics,
        "query_confidence_calibration": confidence_calibration,
        "runtime_ready": bool(
            evaluation["passed"]
            and confidence_passed
            and dataset_provenance == RECORDED_PROVENANCE
        ),
        "held_out_test_point_samples": int(test_metrics["samples"]),
        "point_accuracy_definition": (
            "Top-1 argmax distance from the human-labeled click point, normalized "
            "by the screenshot diagonal; this is not target-box hit rate."
        ),
        "point_accuracy_target": {
            "metric": GROUNDING_ACCEPTANCE_METRIC,
            "minimum": GROUNDING_ACCEPTANCE_THRESHOLD,
            "met": evaluation["passed"],
            "source": "operator_requested_goal_not_a_measured_baseline",
        },
        "target_hit_rate": ("not_measured; recorder has click points, not target bounds"),
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the AVCS pixel-and-goal grounder from recorded UI clicks."
    )
    parser.add_argument("--data-dir", required=True, help="Root containing recorder sessions.")
    parser.add_argument("--output-dir", default="runs/real_ui_grounder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:  # noqa: UP045
    args = build_parser().parse_args(argv)
    report = train_grounder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        seed=args.seed,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(
        f"Saved demonstration-trained checkpoint to "
        f"{Path(args.output_dir) / 'goal_grounder_best.pt'}"
    )
    print(
        "Held-out test point accuracy @2% diagonal: "
        f"{report['held_out_test_metrics']['top1_point_accuracy_within_0.02_diagonal']:.3f} "
        f"({report['held_out_test_point_samples']} samples; "
        f"85% target {'met' if report['point_accuracy_target']['met'] else 'not met'})."
    )
    print(
        "Runtime grounding readiness: "
        f"{'ready' if report['runtime_ready'] else 'not ready'}."
    )
    return 0 if report["runtime_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
