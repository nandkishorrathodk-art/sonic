"""Offline evaluation of captured computer-use traces.

Replay is intentionally observational: it never dispatches an action or
reaches a provider. It measures whether a recorded run was evidence-backed and
whether the controller avoided repeating failed approaches.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def _value(item: Any, name: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


@dataclass(frozen=True)
class ReplayMetrics:
    total_actions: int
    verified_actions: int
    successful_actions: int
    failed_actions: int
    safety_blocks: int
    evidence_complete_actions: int
    prediction_accuracy: float
    verification_rate: float
    repeated_action_rate: float
    evidence_completeness: float
    composite_score: float


@dataclass(frozen=True)
class ReplayPromotionDecision:
    allowed: bool
    reason: str
    baseline_score: float
    candidate_score: float


def promotion_gate(
    baseline: ReplayMetrics,
    candidate: ReplayMetrics,
) -> ReplayPromotionDecision:
    """Require no safety/evidence regression before promoting a strategy."""
    if candidate.safety_blocks > baseline.safety_blocks:
        return ReplayPromotionDecision(
            False,
            "candidate introduced additional safety blocks",
            baseline.composite_score,
            candidate.composite_score,
        )
    if candidate.evidence_completeness < baseline.evidence_completeness:
        return ReplayPromotionDecision(
            False,
            "candidate evidence completeness regressed",
            baseline.composite_score,
            candidate.composite_score,
        )
    if candidate.verification_rate < baseline.verification_rate:
        return ReplayPromotionDecision(
            False,
            "candidate verification rate regressed",
            baseline.composite_score,
            candidate.composite_score,
        )
    if candidate.composite_score < baseline.composite_score:
        return ReplayPromotionDecision(
            False,
            "candidate replay quality regressed",
            baseline.composite_score,
            candidate.composite_score,
        )
    return ReplayPromotionDecision(
        True,
        "candidate passed replay quality and safety gates",
        baseline.composite_score,
        candidate.composite_score,
    )


def evaluate_traces(traces: Iterable[Any]) -> ReplayMetrics:
    """Compute honest quality metrics from persisted traces only."""
    items = list(traces)
    total = len(items)
    statuses = [str(_value(item, "status", "")) for item in items]
    successful = sum(status in {"SUCCESS", "SUCCEEDED", "COMPLETED"} for status in statuses)
    verified = sum(status == "VERIFIED" for status in statuses)
    failed = sum(status in {"FAILED", "TIMED_OUT", "CANCELLED"} for status in statuses)
    safety_blocks = sum(
        status == "BLOCKED"
        and str(_value(item, "actual_observation", "")).lower().startswith("safety blocked:")
        for item, status in zip(items, statuses)
    )
    evidence_complete = sum(
        bool(str(_value(item, "actual_observation", "")).strip())
        and bool(str(_value(item, "verification_evidence", "")).strip() or
                 _value(item, "state_changed", None) is not None)
        for item in items
    )
    errors = [
        max(0.0, min(1.0, float(_value(item, "prediction_error", 0.0) or 0.0)))
        for item in items
        if _value(item, "prediction_error", None) is not None
    ]
    prediction_accuracy = round(1.0 - (sum(errors) / len(errors)), 3) if errors else 0.0
    action_keys = [
        (str(_value(item, "action_type", "")), str(_value(item, "target_resource", "")))
        for item in items
    ]
    repeated = sum(
        index > 0 and key == action_keys[index - 1]
        for index, key in enumerate(action_keys)
    )
    verification_rate = round(verified / total, 3) if total else 0.0
    repeated_rate = round(repeated / total, 3) if total else 0.0
    evidence_rate = round(evidence_complete / total, 3) if total else 0.0
    composite = round(
        max(
            0.0,
            min(
                100.0,
                prediction_accuracy * 30
                + verification_rate * 30
                + evidence_rate * 25
                + (1.0 - repeated_rate) * 15,
            ),
        ),
        1,
    )
    return ReplayMetrics(
        total_actions=total,
        verified_actions=verified,
        successful_actions=successful,
        failed_actions=failed,
        safety_blocks=safety_blocks,
        evidence_complete_actions=evidence_complete,
        prediction_accuracy=prediction_accuracy,
        verification_rate=verification_rate,
        repeated_action_rate=repeated_rate,
        evidence_completeness=evidence_rate,
        composite_score=composite,
    )
