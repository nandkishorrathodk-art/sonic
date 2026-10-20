"""Train the AVCS local focus-verification head on recorded focus demonstrations.

Mirrors the goal-grounder trainer's honesty discipline: the checkpoint is only
marked runtime-ready when a held-out evaluation over a minimum sample count
meets the precision threshold.  A checkpoint that cannot clear the gate is still
written with ``passed: false`` so the loader refuses it and SONIC keeps
abstaining from unverified text entry.
"""
from __future__ import annotations

import argparse
import json
import math
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

from avcs.focus_verifier import (
    DEFAULT_INPUT_SIZE,
    FOCUS_ACCEPTANCE_METRIC,
    FOCUS_ACCEPTANCE_THRESHOLD,
    FOCUS_CHECKPOINT_SCHEMA,
    FOCUS_DATASET_PROVENANCE,
    FOCUS_MIN_EVALUATION_SAMPLES,
    FOCUS_MODEL_TYPE,
    FocusVerifier,
)
from avcs.training.focus_dataset import (
    FocusDemoDataset,
    discover_focus_demonstrations,
    focus_collate,
    split_focus_sessions,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: FocusVerifier,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],  # noqa: UP045
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    correct = 0
    positives = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for patches, queries, labels in loader:
            patches = patches.to(device)
            queries = queries.to(device)
            labels = labels.to(device)
            logits = model(patches, queries)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            total += labels.numel()
            correct += int((predictions == labels).sum().item())
            positives += int(labels.sum().item())
            total_loss += float(loss.item()) * labels.numel()
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "samples": float(total),
        "positives": float(positives),
    }


def _score_examples(
    model: FocusVerifier, loader: DataLoader, device: torch.device
) -> tuple[list[float], list[int]]:
    model.eval()
    scores: list[float] = []
    hits: list[int] = []
    with torch.inference_mode():
        for patches, queries, labels in loader:
            logits = model(patches.to(device), queries.to(device))
            probabilities = torch.sigmoid(logits).cpu().tolist()
            scores.extend(float(p) for p in probabilities)
            hits.extend(int(v) for v in labels.tolist())
    return scores, hits


def _select_threshold(
    scores: list[float], hits: list[int], *, required_precision: float
) -> tuple[float, int, float]:
    """Pick the lowest threshold whose empirical precision clears the target.

    Candidate thresholds are midpoints between adjacent distinct scores (plus a
    value just below the minimum).  Using midpoints rather than the raw scores
    avoids a boundary-equality failure where a true positive whose score equals
    the threshold is excluded by ``score >= threshold``.
    """
    if len(scores) != len(hits):
        raise ValueError("focus scores and labels must have equal length")
    if not scores:
        return 0.5, 0, 0.0
    ordered = sorted(set(scores))
    candidates: list[float] = [max(0.0, ordered[0] - 1e-6)]
    for lower, upper in zip(ordered, ordered[1:]):
        candidates.append((lower + upper) / 2.0)
    best: tuple[float, int, float] | None = None
    for threshold in candidates:
        accepted = [h for s, h in zip(scores, hits) if s >= threshold]
        if len(accepted) < FOCUS_MIN_EVALUATION_SAMPLES:
            continue
        precision = sum(accepted) / len(accepted)
        if precision >= required_precision and (best is None or threshold < best[0]):
            best = (threshold, len(accepted), precision)
    if best is not None:
        return best
    # No candidate clears the coverage bar at the required precision: return the
    # highest-precision operating point so the caller still sees honest evidence.
    best_threshold = max(candidates)
    best_precision = 0.0
    best_count = 0
    for threshold in candidates:
        accepted = [h for s, h in zip(scores, hits) if s >= threshold]
        if not accepted:
            continue
        precision = sum(accepted) / len(accepted)
        if precision > best_precision:
            best_precision = precision
            best_threshold = threshold
            best_count = len(accepted)
    return best_threshold, best_count, best_precision


def train_focus_verifier(
    *,
    data_dir: str,
    output_dir: str = "runs/focus_verifier",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    input_size: int = DEFAULT_INPUT_SIZE,
    seed: int = 42,
    device_name: Optional[str] = None,  # noqa: UP045
    overwrite: bool = False,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if input_size < 16 or input_size % 16:
        raise ValueError("input_size must be at least 16 and divisible by 16")

    sessions = discover_focus_demonstrations(data_dir)
    train_records, validation_records, test_records = split_focus_sessions(
        sessions, seed=seed
    )
    train_loader = DataLoader(
        FocusDemoDataset(train_records, input_size=input_size),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=focus_collate,
        num_workers=0,
    )
    validation_loader = DataLoader(
        FocusDemoDataset(validation_records, input_size=input_size),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=focus_collate,
        num_workers=0,
    )
    test_loader = DataLoader(
        FocusDemoDataset(test_records, input_size=input_size),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=focus_collate,
        num_workers=0,
    )

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(seed)
    model = FocusVerifier(input_size=input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "focus_verifier_best.pt"
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
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        print(
            f"epoch {epoch:03d}/{epochs}: train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_acc={validation_metrics['accuracy']:.3f}"
        )
        if math.isfinite(validation_metrics["loss"]) and (
            validation_metrics["loss"] < best_validation_loss
        ):
            best_validation_loss = validation_metrics["loss"]
            checkpoint = {
                "schema_version": FOCUS_CHECKPOINT_SCHEMA,
                "model_type": FOCUS_MODEL_TYPE,
                "dataset_provenance": FOCUS_DATASET_PROVENANCE,
                "training_samples": len(train_records),
                "validation_samples": len(validation_records),
                "held_out_test_samples": len(test_records),
                "input_size": input_size,
                "model_state_dict": model.state_dict(),
            }
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)

    if not math.isfinite(best_validation_loss) or not checkpoint_path.is_file():
        raise RuntimeError("No finite validation checkpoint was produced")

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)

    validation_scores, validation_hits = _score_examples(model, validation_loader, device)
    threshold, validation_accepted, validation_precision = _select_threshold(
        validation_scores, validation_hits, required_precision=FOCUS_ACCEPTANCE_THRESHOLD
    )
    held_out_scores, held_out_hits = _score_examples(model, test_loader, device)
    accepted = [h for s, h in zip(held_out_scores, held_out_hits) if s >= threshold]
    precision = sum(accepted) / len(accepted) if accepted else 0.0
    passed = (
        len(accepted) >= FOCUS_MIN_EVALUATION_SAMPLES
        and precision >= FOCUS_ACCEPTANCE_THRESHOLD
    )
    evaluation = {
        "metric": FOCUS_ACCEPTANCE_METRIC,
        "precision": precision,
        "minimum": FOCUS_ACCEPTANCE_THRESHOLD,
        "samples": len(held_out_scores),
        "accepted_samples": len(accepted),
        "passed": passed,
    }
    best_checkpoint["evaluation"] = evaluation
    best_checkpoint["decision_threshold"] = float(threshold)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(best_checkpoint, temporary)
    temporary.replace(checkpoint_path)

    report = {
        "schema_version": FOCUS_CHECKPOINT_SCHEMA,
        "model_type": FOCUS_MODEL_TYPE,
        "dataset_provenance": FOCUS_DATASET_PROVENANCE,
        "session_split": "task_and_window_disjoint",
        "recording_sessions": len(sessions),
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "held_out_test_samples": len(test_records),
        "input_size": input_size,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "best_validation_loss": best_validation_loss,
        "decision_threshold": float(threshold),
        "validation_precision_at_threshold": validation_precision,
        "validation_accepted_samples": validation_accepted,
        "evaluation": evaluation,
        "runtime_ready": bool(passed),
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the AVCS local focus-verification head from recorded demos."
    )
    parser.add_argument("--data-dir", required=True, help="Root containing recorder sessions.")
    parser.add_argument("--output-dir", default="runs/focus_verifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:  # noqa: UP045
    args = build_parser().parse_args(argv)
    report = train_focus_verifier(
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
    print(json.dumps(report["evaluation"], indent=2))
    print(f"runtime_ready={report['runtime_ready']}")
    return 0 if report["runtime_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
