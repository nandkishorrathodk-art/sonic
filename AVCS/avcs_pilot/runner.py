from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import ActionStatus, AgentExecution, PilotScenario, PublicTask, VerificationResult, json_safe
from .evaluator import verify_exact_text
from .plugins import load_factory


SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def aggregate_records(records: list[dict[str, Any]], expected_runs: int, min_success_rate: float = 0.95) -> dict[str, Any]:
    total = len(records)
    verified_successes = sum(bool(r.get("verification", {}).get("passed")) for r in records)
    false_confirmations = sum(
        r.get("execution", {}).get("status") == ActionStatus.CONFIRMED.value
        and not bool(r.get("verification", {}).get("passed"))
        for r in records
    )
    confirmations = sum(r.get("execution", {}).get("status") == ActionStatus.CONFIRMED.value for r in records)
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("execution", {}).get("status", "UNKNOWN"))
        statuses[status] = statuses.get(status, 0) + 1
    # Latency is measured by the harness wall clock, never trusted from adapter telemetry.
    times = [float(r["harness_duration_ms"]) for r in records if r.get("harness_duration_ms") is not None]
    action_budget_violations = sum(bool(r.get("safety", {}).get("action_budget_exceeded")) for r in records)
    path_escape_events = sum(r.get("verification", {}).get("evidence_code") == "PATH_ESCAPE" for r in records)
    non_low_risk_trials = sum(str(r.get("risk_level", "LOW")).upper() != "LOW" for r in records)
    lower, upper = wilson_interval(verified_successes, total)
    by_workflow: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_workflow.setdefault(str(record.get("workflow_id", "unknown")), []).append(record)
    workflows: dict[str, Any] = {}
    for workflow_id, rows in sorted(by_workflow.items()):
        success = sum(bool(x.get("verification", {}).get("passed")) for x in rows)
        lo, hi = wilson_interval(success, len(rows))
        workflows[workflow_id] = {
            "runs": len(rows),
            "verified_successes": success,
            "success_rate": success / len(rows) if rows else None,
            "wilson_95_ci": [lo, hi],
            "false_confirmations": sum(x.get("execution", {}).get("status") == "CONFIRMED" and not x.get("verification", {}).get("passed") for x in rows),
        }
    success_rate = verified_successes / total if total else 0.0
    gates = {
        "minimum_run_count": {"pass": total >= expected_runs, "observed": total, "required": expected_runs},
        "verified_task_completion": {"pass": total > 0 and success_rate >= min_success_rate, "observed_rate": success_rate, "required_rate": min_success_rate},
        "zero_false_confirmations": {"pass": false_confirmations == 0, "observed": false_confirmations},
        "workflow_coverage": {"pass": all(x["runs"] > 0 for x in workflows.values()) and bool(workflows), "observed_workflows": len(workflows)},
        "action_budget_compliance": {"pass": action_budget_violations == 0, "violations": action_budget_violations},
        "no_path_escape": {"pass": path_escape_events == 0, "events": path_escape_events},
        "low_risk_scope_only": {"pass": non_low_risk_trials == 0, "non_low_risk_trials": non_low_risk_trials},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "trials": total,
        "verified_successes": verified_successes,
        "verified_task_completion_rate": success_rate,
        "verified_task_completion_wilson_95_ci": [lower, upper],
        "reported_confirmations": confirmations,
        "false_confirmations": false_confirmations,
        "false_confirmation_rate_per_trial": false_confirmations / total if total else None,
        "false_confirmation_rate_per_reported_confirmation": false_confirmations / confirmations if confirmations else None,
        "terminal_status_counts": statuses,
        "latency_ms": {"p50": percentile(times, .50), "p95": percentile(times, .95), "p99": percentile(times, .99), "max": max(times) if times else None},
        "workflows": workflows,
        "acceptance_gates": gates,
        "computational_go": all(g["pass"] for g in gates.values()),
        "release_note": "Computational gates do not replace human privacy, security, customer-approval, or Windows-environment sign-off.",
    }


async def _execute_one(adapter: Any, scenario: PilotScenario, run_dir: Path, run_id: str, run_index: int, seed: int) -> dict[str, Any]:
    # Validate paths before any fixture write or agent execution.
    public_task = scenario.public_task(str(run_dir), f"{scenario.task_id}-{run_index:04d}")
    if scenario.fixture_relative_path:
        fixture = (run_dir / scenario.fixture_relative_path).resolve()
        root_resolved = run_dir.resolve()
        if root_resolved not in fixture.parents:
            raise ValueError("fixture_relative_path must resolve inside the trial directory")
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(scenario.fixture_text or "", encoding="utf-8", newline="")
    task_id = f"{scenario.task_id}-{run_index:04d}"
    # Keep the runner-generated identifier consistent with the task boundary.
    public_task = PublicTask(
        task_id=task_id,
        workflow_id=public_task.workflow_id,
        prompt=public_task.prompt,
        application=public_task.application,
        timeout_s=public_task.timeout_s,
        max_actions=public_task.max_actions,
        risk_level=public_task.risk_level,
    )
    # Assert the agent-facing payload cannot accidentally contain the private expected output.
    if scenario.expected_text and scenario.expected_text in public_task.prompt:
        # This may be legitimate for text-entry tasks; the goal itself can name desired text.
        # We don't prohibit it, but the verifier's file path/state are still not passed separately.
        pass
    started_wall = utc_now()
    start = time.perf_counter()
    try:
        raw = await asyncio.wait_for(adapter.execute(public_task), timeout=scenario.timeout_s)
        execution = raw if isinstance(raw, AgentExecution) else AgentExecution.from_mapping(raw)
    except asyncio.TimeoutError:
        execution = AgentExecution(status=ActionStatus.TIMED_OUT, duration_ms=(time.perf_counter() - start) * 1000, message="Pilot harness timeout")
    except asyncio.CancelledError:
        execution = AgentExecution(status=ActionStatus.CANCELLED, duration_ms=(time.perf_counter() - start) * 1000, message="Execution cancelled")
    except Exception as exc:  # A runtime exception is recorded as a failed trial, not dropped.
        execution = AgentExecution(status=ActionStatus.FAILED, duration_ms=(time.perf_counter() - start) * 1000, message=f"Adapter error: {type(exc).__name__}")
    wall_duration_ms = (time.perf_counter() - start) * 1000
    adapter_reported_duration_ms = execution.duration_ms if execution.duration_ms > 0 else None
    # Canonical duration is measured by the harness; preserve adapter-reported time only as diagnostic metadata.
    execution = AgentExecution(
        status=execution.status,
        action_count=execution.action_count,
        duration_ms=wall_duration_ms,
        provider=execution.provider,
        model=execution.model,
        confirmation_method=execution.confirmation_method,
        message=execution.message,
        diagnostics={**dict(execution.diagnostics), "adapter_reported_duration_ms": adapter_reported_duration_ms},
    )
    # Evaluator receives private scenario/oracle locally. It is never passed to adapter.execute.
    verification = verify_exact_text(scenario, run_dir)
    finished_wall = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "trial_id": str(uuid.uuid4()),
        "seed": seed,
        "run_index": run_index,
        "task_id": task_id,
        "workflow_id": scenario.workflow_id,
        "application": scenario.application,
        "risk_level": scenario.risk_level,
        "max_actions": scenario.max_actions,
        "started_at": started_wall,
        "finished_at": finished_wall,
        "harness_duration_ms": wall_duration_ms,
        "host": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "avcs_commit": os.environ.get("AVCS_COMMIT", "UNSET"),
            "avcs_sdk_version": os.environ.get("AVCS_SDK_VERSION", "UNSET"),
            "avcs_adapter_version": os.environ.get("AVCS_ADAPTER_VERSION", "UNSET"),
            "windows_build": os.environ.get("AVCS_WINDOWS_BUILD", "UNSET"),
            "application_version": os.environ.get("AVCS_APP_VERSION", "UNSET"),
        },
        "execution": execution.to_record(),
        "verification": asdict(verification),
        "safety": {"action_budget_exceeded": execution.action_count > scenario.max_actions},
        "derived": {
            "task_verified_success": verification.passed,
            "false_confirmation": execution.status == ActionStatus.CONFIRMED and not verification.passed,
        },
        "run_dir_id": run_dir.name,
    }


async def run_pilot(scenarios: list[PilotScenario], adapter: Any, output_dir: Path, runs_per_workflow: int, seed: int) -> dict[str, Any]:
    if runs_per_workflow < 1:
        raise ValueError("runs_per_workflow must be >= 1")
    if not scenarios:
        raise ValueError("At least one scenario is required")
    workflows = {s.workflow_id for s in scenarios}
    if len(workflows) != len(scenarios):
        raise ValueError("Each scenario must have a unique workflow_id in this pilot configuration")
    if any(s.risk_level != "LOW" for s in scenarios):
        raise ValueError("Milestone 6 pilot harness only permits LOW-risk scenarios")
    run_id = datetime.now(timezone.utc).strftime("pilot-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    root = output_dir / run_id
    root.mkdir(parents=True, exist_ok=False)
    sandbox = root / "sandbox"
    sandbox.mkdir()
    records: list[dict[str, Any]] = []
    schedule = [s for s in scenarios for _ in range(runs_per_workflow)]
    random.Random(seed).shuffle(schedule)
    run_counters: dict[str, int] = {s.workflow_id: 0 for s in scenarios}
    traces_path = root / "trials.jsonl"
    with traces_path.open("w", encoding="utf-8", newline="\n") as trace:
        for global_index, scenario in enumerate(schedule, 1):
            run_counters[scenario.workflow_id] += 1
            run_dir = sandbox / scenario.workflow_id / f"trial-{run_counters[scenario.workflow_id]:04d}"
            run_dir.mkdir(parents=True, exist_ok=False)
            record = await _execute_one(adapter, scenario, run_dir, run_id, global_index, seed)
            records.append(record)
            trace.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            trace.flush()
    summary = aggregate_records(records, expected_runs=len(scenarios) * runs_per_workflow)
    summary.update({
        "run_id": run_id,
        "random_seed": seed,
        "runs_per_workflow": runs_per_workflow,
        "scenario_count": len(scenarios),
        "workflow_ids": sorted(workflows),
        "trace_file": str(traces_path),
        "environment": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "avcs_commit": os.environ.get("AVCS_COMMIT", "UNSET"),
            "avcs_sdk_version": os.environ.get("AVCS_SDK_VERSION", "UNSET"),
            "avcs_adapter_version": os.environ.get("AVCS_ADAPTER_VERSION", "UNSET"),
            "windows_build": os.environ.get("AVCS_WINDOWS_BUILD", "UNSET"),
            "application_version": os.environ.get("AVCS_APP_VERSION", "UNSET"),
            "versions_pinned": all(os.environ.get(name) for name in (
                "AVCS_COMMIT", "AVCS_SDK_VERSION", "AVCS_ADAPTER_VERSION", "AVCS_WINDOWS_BUILD", "AVCS_APP_VERSION"
            )),
        },
    })
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_scenarios(path: Path) -> list[PilotScenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported scenario schema version: {raw.get('schema_version')!r}")
    items = raw.get("scenarios")
    if not isinstance(items, list):
        raise ValueError("Scenario config must contain a scenarios array")
    return [PilotScenario.from_mapping(item) for item in items]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AVCS controlled Windows pilot harness")
    parser.add_argument("--scenarios", type=Path, default=Path("config/pilot_scenarios.json"))
    parser.add_argument("--adapter", default=os.environ.get("AVCS_PILOT_ADAPTER", ""), help="module.path:factory that returns an adapter implementing async execute(PublicTask)")
    parser.add_argument("--runs-per-workflow", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output-dir", type=Path, default=Path("pilot-results"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print public prompts; does not claim pilot results")
    args = parser.parse_args(argv)
    scenarios = load_scenarios(args.scenarios)
    expected = len(scenarios) * args.runs_per_workflow
    if args.dry_run:
        print(json.dumps({"dry_run": True, "workflows": [s.workflow_id for s in scenarios], "runs_per_workflow": args.runs_per_workflow, "planned_trials": expected, "adapter_loaded": False}, indent=2))
        return 0
    if os.name != "nt":
        parser.error("Live pilot scenarios target Windows Notepad and require an interactive Windows session. Use --dry-run for config validation.")
    if not args.adapter:
        parser.error("No AVCS pilot adapter configured. Supply --adapter module:factory or AVCS_PILOT_ADAPTER; harness refuses to simulate passing results.")
    adapter = load_factory(args.adapter)
    if not callable(getattr(adapter, "execute", None)):
        parser.error("Adapter must implement async execute(PublicTask) -> AgentExecution or mapping")
    summary = asyncio.run(run_pilot(scenarios, adapter, args.output_dir, args.runs_per_workflow, args.seed))
    print(json.dumps(summary, indent=2))
    return 0 if summary["computational_go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
