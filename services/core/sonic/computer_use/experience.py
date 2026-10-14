"""Online, state-grounded experience model for computer-use actions.

This module learns transitions from real observations only. It does not choose
actions, authorize them, or contain application-specific workflows. The LLM
proposes an action; this model records whether the observed state supported the
prediction and exposes that error to the next reasoning step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateSignature:
    screen_hash: str
    active_window: str
    windows: tuple[str, ...]
    processes: tuple[str, ...]
    visible_text_hash: str
    terminal_output_hash: str
    exit_code: int | None
    graph_hash: str = ""


@dataclass(frozen=True)
class ExperienceRecord:
    action: str
    predicted_outcome: str
    before: StateSignature
    after: StateSignature
    state_changed: bool
    prediction_error: float
    evidence: str
    graph_delta: str = ""


def _digest(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8"), digest_size=12).hexdigest() if value else ""


def state_signature(
    *,
    screen_hash: str = "",
    active_window: str = "",
    windows: Any = (),
    processes: Any = (),
    visible_text: str = "",
    terminal_output: str = "",
    exit_code: int | None = None,
    graph_hash: str = "",
) -> StateSignature:
    """Build a compact signature from live observation data."""
    return StateSignature(
        screen_hash=str(screen_hash or ""),
        active_window=str(active_window or ""),
        windows=tuple(sorted(str(item) for item in (windows or ()))),
        processes=tuple(sorted(str(item) for item in (processes or ()))),
        visible_text_hash=_digest(str(visible_text or "")),
        terminal_output_hash=_digest(str(terminal_output or "")),
        exit_code=exit_code,
        graph_hash=str(graph_hash or ""),
    )


def compare_prediction(
    *,
    action: str,
    predicted_outcome: str,
    before: StateSignature,
    after: StateSignature,
    evidence: str = "",
    graph_delta: str = "",
) -> ExperienceRecord:
    """Compare a predicted transition with the real observed transition.

    A changed state is not automatically success: the provider status and
    exit-code evidence remain authoritative. The score only measures how much
    the observation diverged from the predicted outcome for replanning.
    """
    changed = before != after
    prediction = (predicted_outcome or "").strip().lower()
    evidence_text = (evidence or "").strip().lower()
    has_positive_evidence = bool(evidence_text) and not any(
        marker in evidence_text
        for marker in ("error:", "failed", "blocked", "timed out", "traceback")
    )
    if not prediction:
        error = 1.0
    elif changed and has_positive_evidence:
        error = 0.0
    elif changed or has_positive_evidence:
        error = 0.5
    else:
        error = 1.0
    return ExperienceRecord(
        action=action,
        predicted_outcome=predicted_outcome,
        before=before,
        after=after,
        state_changed=changed,
        prediction_error=error,
        evidence=evidence,
        graph_delta=graph_delta,
    )
