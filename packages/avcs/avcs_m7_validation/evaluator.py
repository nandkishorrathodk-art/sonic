from __future__ import annotations
import math
from collections import defaultdict
from typing import Iterable
from .contracts import ActionStatus, Mode, OracleResult, Scenario, TrialResult, ExecutionResult, OutcomeCategory


def score_trial(*, run_id: str, scenario: Scenario, mode: Mode, seed: int,
                execution: ExecutionResult, oracle: OracleResult) -> TrialResult:
    reported = execution.status == ActionStatus.CONFIRMED
    false_confirmation = reported and oracle.valid and not oracle.success
    verified_success = reported and oracle.valid and oracle.success
    if not oracle.valid:
        category = OutcomeCategory.EVALUATION_INVALID
    elif false_confirmation:
        category = OutcomeCategory.FALSE_CONFIRMATION
    elif execution.status == ActionStatus.BLOCKED:
        category = OutcomeCategory.BLOCKED
    elif execution.status in (ActionStatus.AMBIGUOUS, ActionStatus.CANCELLED, ActionStatus.TIMED_OUT):
        category = OutcomeCategory.ABSTENTION
    elif execution.status == ActionStatus.ERROR:
        category = OutcomeCategory.EXECUTION_ERROR
    elif oracle.success:
        category = OutcomeCategory.TASK_SUCCESS
    else:
        category = OutcomeCategory.TASK_FAILURE
    if execution.elapsed_ms < 0 or not math.isfinite(execution.elapsed_ms):
        raise ValueError("elapsed_ms must be finite and non-negative")
    if execution.action_count < 0:
        raise ValueError("action_count must be non-negative")
    return TrialResult(
        run_id=run_id, scenario_id=scenario.scenario_id, mode=mode, seed=seed,
        status=execution.status, oracle_success=oracle.success, oracle_valid=oracle.valid,
        reported_confirmed=reported, verified_success=verified_success,
        false_confirmation=false_confirmation, elapsed_ms=execution.elapsed_ms,
        action_count=execution.action_count, evidence_method=execution.evidence_method,
        category=category, oracle_id=oracle.oracle_id, assertion_id=oracle.assertion_id,
        detail_code=execution.detail_code,
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Require trials > 0 and 0 <= successes <= trials")
    p = successes / trials
    z2 = z * z
    denom = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    radius = z * math.sqrt((p * (1 - p) / trials) + (z2 / (4 * trials * trials))) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize(results: Iterable[TrialResult]) -> dict:
    rows = list(results)
    groups: dict[str, list[TrialResult]] = defaultdict(list)
    by_scenario: dict[str, list[TrialResult]] = defaultdict(list)
    for row in rows:
        groups[row.mode.value].append(row)
        by_scenario[f"{row.scenario_id}:{row.mode.value}"].append(row)

    def stats(items: list[TrialResult]) -> dict:
        n = len(items)
        valid_items = [r for r in items if r.oracle_valid]
        oracle_successes = sum(r.oracle_success and r.oracle_valid for r in items)
        verified = sum(r.verified_success for r in items)
        false = sum(r.false_confirmation for r in items)
        sorted_latency = sorted(r.elapsed_ms for r in items)
        def pct(p: float) -> float | None:
            if not sorted_latency:
                return None
            # Nearest-rank percentile, deliberately specified for reproducibility.
            idx = max(0, math.ceil(p * len(sorted_latency)) - 1)
            return round(sorted_latency[idx], 3)
        ci = wilson_interval(oracle_successes, len(valid_items)) if valid_items else (None, None)
        return {
            "trials": n,
            "oracle_valid_trials": len(valid_items),
            "oracle_invalid_trials": n - len(valid_items),
            "oracle_successes": oracle_successes,
            "oracle_task_success_rate": round(oracle_successes / len(valid_items), 6) if valid_items else None,
            "oracle_success_wilson_95": [round(ci[0], 6), round(ci[1], 6)] if valid_items else None,
            "agent_verified_successes": verified,
            "agent_verified_success_rate": round(verified / n, 6) if n else None,
            "false_confirmations": false,
            "false_confirmation_rate": round(false / len(valid_items), 6) if valid_items else None,
            "blocked": sum(r.status == ActionStatus.BLOCKED for r in items),
            "abstentions": sum(r.category.value == "abstention" for r in items),
            "errors": sum(r.status == ActionStatus.ERROR for r in items),
            "latency_ms": {"p50": pct(.50), "p95": pct(.95), "p99": pct(.99), "max": round(max(sorted_latency), 3) if sorted_latency else None},
            "total_actions": sum(r.action_count for r in items),
        }

    return {
        "schema_version": "1.0",
        "total_trials": len(rows),
        "by_mode": {mode: stats(items) for mode, items in sorted(groups.items())},
        "by_scenario_and_mode": {key: stats(items) for key, items in sorted(by_scenario.items())},
        "false_confirmation_run_ids": [r.run_id for r in rows if r.false_confirmation],
    }
