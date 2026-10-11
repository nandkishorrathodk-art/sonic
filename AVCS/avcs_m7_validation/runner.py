from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from . import __version__
from .catalog import SCENARIOS
from .contracts import AgentTask, Mode, TrialResult
from .evaluator import score_trial, summarize
from .plugins import load_live_plugins, MissingPlugin

DEFAULT_MODES = ",".join(m.value for m in Mode)

def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def build_plan(runs_per_scenario: int, seed: int, modes: list[Mode]) -> dict:
    trials = []
    rng = random.Random(seed)
    for scenario in SCENARIOS:
        available_modes = [mode for mode in modes if mode.value in scenario.modes]
        for repetition in range(runs_per_scenario):
            # Same seed per scenario/repetition across modes enables paired comparisons.
            trial_seed = rng.randrange(0, 2**63)
            for mode in available_modes:
                trials.append({"scenario_id": scenario.scenario_id, "mode": mode.value,
                               "repetition": repetition, "seed": trial_seed,
                               "category": scenario.category, "application": scenario.application,
                               "held_out": scenario.held_out})
    rng.shuffle(trials)
    return {"schema_version": "1.0", "kind": "plan_only_not_execution_evidence",
            "package_version": __version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "master_seed": seed, "runs_per_scenario": runs_per_scenario,
            "modes": [m.value for m in modes], "scenario_count": len(SCENARIOS),
            "planned_trial_count": len(trials), "trials": trials}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AVCS Milestone 7 cross-application validation")
    parser.add_argument("--dry-run", action="store_true", help="emit a plan only; no desktop actions are run")
    parser.add_argument("--runs-per-scenario", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--output-dir", default="m7-results")
    args = parser.parse_args(argv)
    if args.runs_per_scenario <= 0 or args.timeout_s <= 0:
        parser.error("runs-per-scenario and timeout-s must be positive")
    try:
        modes = [Mode(part.strip()) for part in args.modes.split(",") if part.strip()]
    except ValueError as exc:
        parser.error(f"invalid mode: {exc}")
    if not modes or len(modes) != len(set(modes)):
        parser.error("provide one or more unique modes")

    out = Path(args.output_dir)
    plan = build_plan(args.runs_per_scenario, args.seed, modes)
    _json_dump(out / "plan.json", plan)
    if args.dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k != "trials"}, indent=2))
        print(f"Plan only. No adapter loaded and no tasks executed. Full plan: {out / 'plan.json'}")
        return 0

    try:
        adapter, oracle = load_live_plugins()
    except Exception as exc:
        print(f"LIVE RUN BLOCKED: {exc}", file=sys.stderr)
        print("Plan saved; no desktop action was executed.", file=sys.stderr)
        return 2

    scenario_by_id = {s.scenario_id: s for s in SCENARIOS}
    rng = random.Random(args.seed)
    results: list[TrialResult] = []
    trace_path = out / "trials.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0", "package_version": __version__,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.seed, "runs_per_scenario": args.runs_per_scenario,
        "modes": [m.value for m in modes],
        "metadata": {k: os.getenv(k) for k in ("AVCS_COMMIT", "AVCS_SDK_VERSION", "AVCS_ADAPTER_VERSION", "AVCS_WINDOWS_BUILD", "AVCS_APP_VERSION")},
    }
    _json_dump(out / "manifest.json", manifest)
    trial_specs = list(plan["trials"])
    with trace_path.open("w", encoding="utf-8") as fp:
        for idx, spec in enumerate(trial_specs, 1):
            scenario = scenario_by_id[spec["scenario_id"]]
            mode = Mode(spec["mode"])
            seed = int(spec["seed"])
            run_id = f"m7-{idx:05d}-{uuid.uuid4().hex[:8]}"
            started = time.perf_counter()
            from .contracts import ActionStatus, ExecutionResult, OracleResult
            try:
                execution = adapter.execute(run_id=run_id, task=AgentTask(task_prompt=scenario.task_prompt), mode=mode,
                                            seed=seed, timeout_s=args.timeout_s)
                if not isinstance(execution, ExecutionResult):
                    raise TypeError("adapter returned an invalid ExecutionResult")
            except Exception:
                # Keep details out of traces to avoid accidental screen text/PII leakage.
                execution = ExecutionResult(ActionStatus.ERROR,
                    max(0.0, (time.perf_counter() - started) * 1000), 0, "none", "adapter_exception")
            try:
                # Oracle is separate from execution plugin and receives no execution result.
                observed = oracle.evaluate(run_id=run_id, scenario=scenario, seed=seed)
                if not isinstance(observed, OracleResult) or not observed.valid:
                    raise TypeError("oracle returned an invalid result")
            except Exception:
                # Mark this trial unevaluable, not a failure and not evidence of zero false confirmations.
                observed = OracleResult(False, "oracle-unavailable", "oracle-check-failed",
                                        "oracle_unavailable", valid=False)
            result = score_trial(run_id=run_id, scenario=scenario, mode=mode,
                                 seed=seed, execution=execution, oracle=observed)
            results.append(result)
            fp.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            fp.flush()
            if idx % 25 == 0 or idx == len(trial_specs):
                print(f"Progress: {idx}/{len(trial_specs)} trials")
    summary = summarize(results)
    summary["manifest"] = manifest
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    scenario_coverage = len({r.scenario_id for r in results})
    all_workflows_pass = all(
        values["oracle_invalid_trials"] == 0
        and values["oracle_task_success_rate"] is not None
        and values["oracle_task_success_rate"] >= 0.95
        for values in summary["by_scenario_and_mode"].values()
    )
    summary["release_decision"] = {
        "computational_go": bool(summary["total_trials"] >= 300
          and scenario_coverage >= 8
          and all(v["trials"] > 0 and v["oracle_invalid_trials"] == 0
                  and v["false_confirmations"] == 0 for v in summary["by_mode"].values())
          and all_workflows_pass),
        "scenario_coverage": scenario_coverage,
        "per_scenario_mode_success_gate": all_workflows_pass,
        "note": "Computational screening only. Human security/privacy approval and external application review remain required.",
    }
    _json_dump(out / "summary.json", summary)
    print(json.dumps({"total_trials": summary["total_trials"], "by_mode": summary["by_mode"],
                      "release_decision": summary["release_decision"]}, indent=2))
    return 0 if summary["release_decision"]["computational_go"] else 3

if __name__ == "__main__":
    raise SystemExit(main())
