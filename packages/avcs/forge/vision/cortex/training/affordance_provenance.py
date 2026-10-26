"""Training provenance for the dorsal (affordance) head.

The dorsal head predicts *where* actions are physically possible -- the
``action_hotspots`` the agent reads to decide where to click. Those hotspots are
only meaningful if the head was actually trained on real supervision.

They were not. Every real-desktop frame arrived with the affordance mask masked
(``-100``), so the head kept its random initialisation and ``find_action_hotspots``
returned noise that looked exactly like a confident detection. The agent then
reported ``interactive_targets=4`` to the reasoning model, which is a fabricated
claim about the screen.

A checkpoint file cannot tell you whether a head inside it was trained. So this
module records that fact *next to* the checkpoint, in the same auditable
sidecar style as the held-out evaluation record. Readiness is then derived from
recorded evidence rather than assumed from the checkpoint's existence:

    affordance_head_trained(path) is True only when a sidecar records a real
    training run that named its data AND was measured on a non-empty held-out
    slice.

A checkpoint with no sidecar -- or one whose run was never verified -- reports
``False``, and callers must treat the hotspots as unavailable.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

AFFORDANCE_PROVENANCE_SUFFIX = ".affordance.json"

# Minimum held-out point accuracy for the head to be considered usable. Chance
# for 5 affordance classes is 0.20; a head must clear this by a clear margin
# before its hotspots are published as detected controls.
MIN_HELD_OUT_POINT_ACCURACY = 0.40


def affordance_provenance_path(checkpoint_path: str | Path) -> Path:
    """Sidecar path for a cortex checkpoint (``x.pt`` -> ``x.affordance.json``)."""
    return Path(checkpoint_path).with_suffix(AFFORDANCE_PROVENANCE_SUFFIX)


def write_affordance_provenance(
    checkpoint_path: str | Path,
    *,
    dataset_provenance: str,
    samples: int,
    positive_samples: int,
    negative_samples: int,
    held_out_samples: int,
    held_out_iou: float | None,
    held_out_point_accuracy: float | None = None,
    base_checkpoint: str = "",
    trained_at: float | None = None,
) -> Path:
    """Persist the affordance head's training provenance beside its checkpoint.

    ``dataset_provenance`` is required: a metric without provenance is not
    evidence, so callers are forced to name what the head was trained on.
    """
    if not str(dataset_provenance).strip():
        raise ValueError("affordance provenance requires a dataset_provenance string")
    if int(samples) <= 0:
        raise ValueError("affordance provenance requires a positive sample count")

    sidecar = affordance_provenance_path(checkpoint_path)
    payload: dict[str, Any] = {
        "checkpoint": Path(checkpoint_path).name,
        "trained": True,
        "dataset_provenance": str(dataset_provenance),
        "samples": int(samples),
        "positive_samples": int(positive_samples),
        "negative_samples": int(negative_samples),
        "held_out_samples": int(held_out_samples),
        "held_out_iou": None if held_out_iou is None else float(held_out_iou),
        "held_out_point_accuracy": (
            None if held_out_point_accuracy is None else float(held_out_point_accuracy)
        ),
        "base_checkpoint": str(base_checkpoint),
        "trained_at": float(trained_at if trained_at is not None else time.time()),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar


def read_affordance_provenance(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Return the recorded provenance, or ``None`` when absent/unreadable."""
    sidecar = affordance_provenance_path(checkpoint_path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def affordance_head_trained(checkpoint_path: str | Path) -> bool:
    """True only when a sidecar proves a real, *useful* affordance training run.

    Fails closed on every ambiguity: a missing sidecar, a run flagged untrained,
    an unnamed dataset, zero training samples, an unmeasured verification, or a
    held-out point accuracy no better than guessing. A head that was fine-tuned
    but still predicts the labelled cell at chance is not usable, so reporting
    its hotspots would be as misleading as reporting an untrained head's.
    """
    payload = read_affordance_provenance(checkpoint_path)
    if not payload:
        return False
    if not bool(payload.get("trained")):
        return False
    if not str(payload.get("dataset_provenance", "")).strip():
        return False
    try:
        if int(payload.get("samples", 0)) <= 0:
            return False
        if int(payload.get("held_out_samples", 0)) <= 0:
            return False
    except (TypeError, ValueError):
        return False

    # Require measured point accuracy above chance. With ``n`` affordance
    # classes, an uninformative head scores about ``1/n``; demand a clear margin
    # so a barely-trained head is not published as a control detector.
    accuracy = payload.get("held_out_point_accuracy")
    if accuracy is None:
        return False
    try:
        accuracy = float(accuracy)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(accuracy):
        return False
    return accuracy >= MIN_HELD_OUT_POINT_ACCURACY
