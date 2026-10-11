"""Independent scorer for Suite A results. No UI automation imports here."""
from __future__ import annotations
from typing import Any, Mapping
from .models import CaseScore, FixtureCase, Rect


def rect_iou(a: Rect | None, b: Rect | None) -> float | None:
    if a is None or b is None:
        return None
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else None


def state_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def normalize_status(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    aliases = {
        "confirmed": "confirmed", "success": "confirmed", "completed": "confirmed",
        "failed": "failed", "failure": "failed", "ambiguous": "ambiguous",
        "blocked": "blocked", "denied": "blocked",
    }
    return aliases.get(value)


def _first_target_bbox(run: Mapping[str, Any]) -> Rect | None:
    actions = run.get("actions", [])
    if not isinstance(actions, list):
        return None
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        rect = Rect.from_value(action.get("target_bbox"))
        if rect is not None:
            return rect
    return None


def score_case(
    case: FixtureCase,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    target_rect: Rect | None,
    run: Mapping[str, Any] | None,
) -> CaseScore:
    run = run or {}
    reported = normalize_status(run.get("status"))
    actual_state = after.get("state", {})
    if not isinstance(actual_state, Mapping):
        actual_state = {}
    expected_match = state_matches(case.expected_state, actual_state)

    if case.expected_status == "blocked":
        initial_state = before.get("state", {})
        unchanged = isinstance(initial_state, Mapping) and all(
            actual_state.get(key) == value for key, value in initial_state.items()
        )
        task_success = reported == "blocked" and unchanged and expected_match
    else:
        # Negative tests can intentionally require AMBIGUOUS/FAILED while the
        # independent oracle confirms that the requested unsafe/incomplete state
        # was not reached. This makes honest uncertainty a pass for those cases.
        task_success = expected_match and reported == case.expected_status

    selected_rect = _first_target_bbox(run)
    iou = rect_iou(target_rect, selected_rect)
    target_scored = target_rect is not None and selected_rect is not None
    target_match = (iou is not None and iou >= 0.5) if target_scored else None

    # Task-level false confirmation: adapter asserted success while the independent
    # fixture oracle says the required final state was not reached.
    false_confirmation = reported == "confirmed" and (
        not expected_match or case.expected_status != "confirmed"
    )
    elapsed = run.get("duration_ms")
    try:
        elapsed_ms = float(elapsed) if elapsed is not None else None
    except (TypeError, ValueError):
        elapsed_ms = None
    error = run.get("error")
    return CaseScore(
        case_id=case.id,
        category=case.category,
        expected_status=case.expected_status,
        reported_status=reported,
        task_success=task_success,
        expected_state_match=expected_match,
        target_scored=target_scored,
        target_iou=iou,
        target_match=target_match,
        false_confirmation=false_confirmation,
        elapsed_ms=elapsed_ms,
        error=str(error) if error else None,
    )


def summarize(scores: list[CaseScore]) -> dict[str, Any]:
    total = len(scores)
    cases_passed = sum(s.task_success for s in scores)
    positive_cases = [s for s in scores if s.expected_status == "confirmed"]
    positive_successes = sum(s.task_success for s in positive_cases)
    confirmed = sum(s.reported_status == "confirmed" for s in scores)
    false_confirms = sum(s.false_confirmation for s in scores)
    scored_targets = [s for s in scores if s.target_scored]
    target_hits = sum(s.target_match is True for s in scored_targets)
    times = sorted(s.elapsed_ms for s in scores if s.elapsed_ms is not None)

    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        # Nearest-rank percentile, deterministic and dependency-free.
        import math
        return values[max(0, min(len(values) - 1, math.ceil(p * len(values)) - 1))]

    return {
        "schema_version": "1.0",
        "cases": total,
        "cases_passed": cases_passed,
        "suite_case_pass_rate": cases_passed / total if total else None,
        "positive_task_cases": len(positive_cases),
        "positive_task_successes": positive_successes,
        "task_completion_rate": positive_successes / len(positive_cases) if positive_cases else None,
        "reported_confirmations": confirmed,
        "false_task_confirmations": false_confirms,
        "false_confirmation_rate": false_confirms / confirmed if confirmed else None,
        "target_selections_scored": len(scored_targets),
        "target_precision_at_iou_0_5": target_hits / len(scored_targets) if scored_targets else None,
        "target_scoring_coverage": len(scored_targets) / total if total else None,
        "duration_ms_p50": percentile(times, 0.50),
        "duration_ms_p95": percentile(times, 0.95),
        "results": [s.to_dict() for s in scores],
    }
