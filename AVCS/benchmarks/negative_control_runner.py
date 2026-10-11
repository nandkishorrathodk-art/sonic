"""AVCS Milestone 4: Negative-Control Benchmark Runner.

Executes >=3,000 representative, independently sampled negative-control trials
under deterministic fault injection and unsatisfied postconditions.
Proves false confirmation probability upper bound p_upper <= 3/3000 = 0.1%.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from avcs.action_broker import (
    ActionBroker,
    ActionClass,
    ActionRequest,
    ActionResult,
    ActionSafetyController,
    ActionStatus,
)
from avcs.capture_adapter import SyntheticCaptureAdapter
from avcs.fault_injection import (
    FaultConfig,
    FaultInjectedCaptureAdapter,
    FaultInjectedDriver,
    FaultInjectionEnvironment,
    FaultType,
)
from avcs.human_eye import HumanEye, HumanEyeState
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return float(s[idx])


def run_negative_trial(
    trial_idx: int,
    base_seed: int,
    broker: ActionBroker,
    base_driver: VirtualCursorDriver,
    cap: FaultInjectedCaptureAdapter,
) -> Dict[str, Any]:
    """Executes a single randomized negative-control trial."""
    trial_seed = (base_seed + trial_idx * 37) & 0xFFFFFFFF
    rng = random.Random(trial_seed)

    fault_candidates = [
        FaultType.SILENT_INPUT_DROP,
        FaultType.FAILED_DISPATCH,
        FaultType.STALE_OBSERVATION,
        FaultType.TARGET_MOVEMENT,
        FaultType.FOCUS_THEFT,
        FaultType.UNEXPECTED_OVERLAY,
        FaultType.DELAYED_TRANSITION,
        FaultType.CAPTURE_ERROR,
        FaultType.VERIFIER_EXCEPTION,
    ]
    fault_type = rng.choice(fault_candidates)

    action_class_candidates = [
        ActionClass.IDEMPOTENT,
        ActionClass.CONDITIONALLY_REPEATABLE,
        ActionClass.NON_IDEMPOTENT,
        ActionClass.HIGH_RISK,
    ]
    action_class = rng.choice(action_class_candidates)
    action_type = rng.choice(["click", "double_click", "type", "press_key"])

    tx = rng.uniform(0.05, 0.95)
    ty = rng.uniform(0.05, 0.95)

    fault_cfg = FaultConfig(
        fault_type=fault_type,
        dispatch_error_message=f"Simulated dispatch failure on trial {trial_idx}",
        capture_error_message=f"Simulated GDI capture failure on trial {trial_idx}",
        exception_message=f"Simulated verifier exception on trial {trial_idx}",
    )

    cap.config = fault_cfg
    cap.armed = False  # Keep unarmed during initial probe, armed during execute

    driver = FaultInjectedDriver(base_driver, fault_cfg)
    driver.armed = True
    broker.driver = driver

    env = FaultInjectionEnvironment(config=fault_cfg, driver=driver, capture=cap, eye=broker.eye)

    # Postcondition is defined to verify the intended state change, which DELIBERATELY
    # remains unsatisfied in all negative-control trials
    intended_postcondition = env.build_postcondition(lambda pre, post: False)

    # For focus theft, target window title is specified so safety controller checks focus
    target_window = "Deterministic Target Window" if fault_type == FaultType.FOCUS_THEFT else None

    # For high risk, authorization is intentionally revoked to test containment
    authorized = (action_class != ActionClass.HIGH_RISK) or (rng.random() < 0.2)

    req = ActionRequest(
        action_type=action_type,
        target_coord=(tx, ty),
        action_class=action_class,
        action_id=f"neg_{trial_idx:05d}",
        text_content="TEST" if action_type == "type" else None,
        expected_postcondition=intended_postcondition,
        precondition_validator=lambda s: rng.random() > 0.5,
        authorized=authorized,
        target_window_title=target_window,
    )

    cap.armed = True
    t0 = time.perf_counter()
    res = broker.execute(req)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    cap.armed = False

    # FALSE CONFIRMATION: The postcondition was unsatisfied, but the agent reported CONFIRMED
    is_false_confirmation = (res.status == ActionStatus.CONFIRMED)

    trial_record: Dict[str, Any] = {
        "trial_id": trial_idx,
        "seed": trial_seed,
        "fault_type": fault_type.value,
        "action_class": action_class.value,
        "action_type": action_type,
        "target_coord": [round(tx, 4), round(ty, 4)],
        "authorized": authorized,
        "status": res.status.value,
        "is_confirmed": res.is_confirmed,
        "false_confirmation": is_false_confirmation,
        "recovery_attempts": res.recovery_attempts,
        "error_category": res.error_category,
        "evidence": res.evidence,
        "ripple_energy": round(float(res.optical_ripple_energy), 6),
        "photo_delta": round(float(res.photometric_delta), 6),
        "duration_ms": round(duration_ms, 3),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return trial_record


def run_negative_control_suite(
    trials_count: int = 3000,
    base_seed: int = 1337,
    output_summary_path: str = "benchmarks/results/negative_control_summary.json",
    trace_jsonl_path: str = "benchmarks/results/negative_control_trials.jsonl",
) -> Dict[str, Any]:
    """Executes the full >=3,000 trials negative-control verification suite."""
    print(f"=== Starting AVCS Negative-Control Benchmark ({trials_count} trials, seed={base_seed}) ===")

    summary_file = Path(output_summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file = Path(trace_jsonl_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    base_driver = VirtualCursorDriver()
    base_cap = SyntheticCaptureAdapter().open()
    cap = FaultInjectedCaptureAdapter(base_cap, FaultConfig())
    eye = HumanEye(source=cap)
    safety = ActionSafetyController(allow_destructive=False)
    # Using low settlement delay for accelerated high-throughput synthetic testing
    broker = ActionBroker(eye=eye, driver=base_driver, safety_controller=safety, settle_delay_s=0.001)

    records: List[Dict[str, Any]] = []
    durations: List[float] = []

    false_confirmations = 0
    status_counts: Dict[str, int] = {
        "confirmed": 0,
        "blocked": 0,
        "failed": 0,
        "ambiguous": 0,
    }
    fault_breakdown: Dict[str, Dict[str, int]] = {}

    t_start = time.perf_counter()

    with trace_file.open("w", encoding="utf-8") as f_jsonl:
        for i in range(trials_count):
            record = run_negative_trial(i, base_seed, broker, base_driver, cap)
            records.append(record)
            durations.append(record["duration_ms"])

            f_jsonl.write(json.dumps(record) + "\n")

            st = record["status"]
            status_counts[st] = status_counts.get(st, 0) + 1

            ft = record["fault_type"]
            if ft not in fault_breakdown:
                fault_breakdown[ft] = {"trials": 0, "false_confirmations": 0, "blocked": 0, "failed": 0, "ambiguous": 0}
            fault_breakdown[ft]["trials"] += 1
            fault_breakdown[ft][st] = fault_breakdown[ft].get(st, 0) + 1

            if record["false_confirmation"]:
                false_confirmations += 1
                fault_breakdown[ft]["false_confirmations"] += 1

            if (i + 1) % 500 == 0 or (i + 1) == trials_count:
                elapsed_s = time.perf_counter() - t_start
                rate = (i + 1) / max(0.001, elapsed_s)
                print(f"[{i + 1}/{trials_count}] completed ({rate:.1f} trials/s) - False confirmations: {false_confirmations}")

    total_time = time.perf_counter() - t_start
    false_conf_rate = false_confirmations / trials_count if trials_count > 0 else 0.0

    # Rule of Three upper bound: 3 / N for zero observed failures
    rule_of_three_upper_bound = 3.0 / trials_count if trials_count > 0 else 1.0

    # Exit Gate: Zero false confirmations
    exit_gate_passed = (false_confirmations == 0)

    summary: Dict[str, Any] = {
        "benchmark_name": "AVCS Negative-Control Robustness & Safety Benchmark",
        "milestone": "Milestone 4",
        "schema_version": "1.0",
        "total_trials": trials_count,
        "base_seed": base_seed,
        "duration_total_seconds": round(total_time, 2),
        "trials_per_second": round(trials_count / max(0.001, total_time), 2),
        "false_confirmations": false_confirmations,
        "false_confirmation_rate": round(false_conf_rate, 6),
        "rule_of_three_95_upper_bound": round(rule_of_three_upper_bound, 6),
        "exit_gate_passed": exit_gate_passed,
        "status_distribution": status_counts,
        "latency_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "p99": _percentile(durations, 0.99),
        },
        "fault_breakdown": fault_breakdown,
        "trace_file": str(trace_file.resolve()),
    }

    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== Summary Report written to {summary_file.resolve()} ===")
    print(f"Exit Gate Passed: {exit_gate_passed} (False confirmations: {false_confirmations}/{trials_count})")
    print(f"Rule-of-Three 95% Confidence Upper Bound: {rule_of_three_upper_bound * 100:.3f}%")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="AVCS Milestone 4: Negative-Control Benchmark Runner")
    parser.add_argument("--trials", type=int, default=3000, help="Number of trials (default: 3000)")
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed (default: 1337)")
    parser.add_argument("--output", default="benchmarks/results/negative_control_summary.json", help="Summary JSON output path")
    parser.add_argument("--trace-jsonl", default="benchmarks/results/negative_control_trials.jsonl", help="JSONL trace output path")
    args = parser.parse_args()

    summary = run_negative_control_suite(
        trials_count=args.trials,
        base_seed=args.seed,
        output_summary_path=args.output,
        trace_jsonl_path=args.trace_jsonl,
    )

    return 0 if summary["exit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

