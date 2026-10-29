"""AVCS Milestone 4: 8-Hour Reliability & Resource Soak Runner.

Monitors memory RSS, Windows OS handles (GDI/USER/Process), capture errors,
action outcomes, and latency percentiles under sustained continuous execution.
Performs linear regression slope trend analysis to detect long-tail resource leaks.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from avcs.action_broker import (
    ActionBroker,
    ActionClass,
    ActionRequest,
    ActionResult,
    ActionSafetyController,
    ActionStatus,
)
from avcs.capture_adapter import BaseCaptureAdapter, SyntheticCaptureAdapter
from avcs.human_eye import HumanEye, HumanEyeState
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return float(s[idx])


def _linear_regression_slope(xs: List[float], ys: List[float]) -> float:
    """Calculates ordinary least squares slope: dy/dx."""
    n = len(xs)
    if n < 2:
        return 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        return 0.0
    return float((n * sum_xy - sum_x * sum_y) / denom)


@dataclass
class TelemetrySnapshot:
    cycle: int
    elapsed_seconds: float
    rss_mb: float
    vms_mb: float
    process_handles: int
    gdi_objects: int
    user_objects: int
    cycle_duration_ms: float
    action_status: str
    capture_error_count: int
    timestamp_iso: str


class SystemResourceTracker:
    """Tracks OS-level process resources, memory, and handles."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.is_windows = platform.system() == "Windows"
        self._kernel32 = None
        self._user32 = None
        self._current_process_handle = None

        if self.is_windows:
            try:
                self._kernel32 = ctypes.windll.kernel32
                self._user32 = ctypes.windll.user32
                self._current_process_handle = self._kernel32.GetCurrentProcess()
            except Exception:
                self.is_windows = False

    def get_metrics(self) -> Tuple[float, float, int, int, int]:
        """Returns (rss_mb, vms_mb, os_handles, gdi_objects, user_objects)."""
        mem = self.process.memory_info()
        rss_mb = mem.rss / (1024.0 * 1024.0)
        vms_mb = mem.vms / (1024.0 * 1024.0)

        process_handles = 0
        try:
            process_handles = self.process.num_handles()
        except Exception:
            pass

        gdi_objects = 0
        user_objects = 0
        if self.is_windows and self._user32 and self._current_process_handle:
            try:
                # 0 = GR_GDIOBJECTS, 1 = GR_USEROBJECTS
                gdi_objects = int(self._user32.GetGuiResources(self._current_process_handle, 0))
                user_objects = int(self._user32.GetGuiResources(self._current_process_handle, 1))
            except Exception:
                pass

        return rss_mb, vms_mb, process_handles, gdi_objects, user_objects


class SoakTestRunner:
    """Orchestrates sustained execution and records resource telemetry."""

    def __init__(
        self,
        duration_seconds: float = 60.0,
        max_cycles: Optional[int] = None,
        telemetry_interval_cycles: int = 10,
        warmup_cycles: int = 30,
        base_seed: int = 42,
        settle_delay_s: float = 0.001,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.max_cycles = max_cycles
        self.telemetry_interval = telemetry_interval_cycles
        self.warmup_cycles = warmup_cycles
        self.base_seed = base_seed
        self.settle_delay_s = settle_delay_s

        self.tracker = SystemResourceTracker()
        self.rng = random.Random(base_seed)

    def run(
        self,
        summary_path: str = "benchmarks/results/soak_summary.json",
        trace_jsonl_path: str = "benchmarks/results/soak_telemetry.jsonl",
    ) -> Dict[str, Any]:
        print(f"=== Starting AVCS Reliability & Resource Soak Test ===")
        print(f"Duration Target: {self.duration_seconds:.1f}s ({self.duration_seconds / 3600.0:.2f} hours)")
        if self.max_cycles:
            print(f"Max Cycles: {self.max_cycles}")

        summary_file = Path(summary_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file = Path(trace_jsonl_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)

        driver = VirtualCursorDriver()
        cap = SyntheticCaptureAdapter().open()
        eye = HumanEye(source=cap)
        safety = ActionSafetyController(allow_destructive=False)
        broker = ActionBroker(
            eye=eye,
            driver=driver,
            safety_controller=safety,
            settle_delay_s=self.settle_delay_s,
        )

        # Warmup Phase to settle initial allocations and JIT warmups
        if self.warmup_cycles > 0:
            print(f"Executing {self.warmup_cycles} warmup cycles...")
            for w in range(self.warmup_cycles):
                warm_req = ActionRequest(
                    action_type="click",
                    target_coord=(0.5, 0.5),
                    action_class=ActionClass.IDEMPOTENT,
                    action_id=f"warmup_{w:04d}",
                    expected_postcondition=lambda pre, post: True,
                    authorized=True,
                )
                broker.execute(warm_req)
            import gc
            gc.collect()

        telemetry_history: List[TelemetrySnapshot] = []
        cycle_durations: List[float] = []
        status_counts = {"confirmed": 0, "blocked": 0, "failed": 0, "ambiguous": 0}
        capture_error_count = 0
        false_confirmations = 0

        start_time = time.perf_counter()
        cycle = 0

        # Post-warmup baseline snapshot
        init_rss, init_vms, init_handles, init_gdi, init_user = self.tracker.get_metrics()

        with trace_file.open("w", encoding="utf-8") as f_trace:
            while True:
                now = time.perf_counter()
                elapsed = now - start_time
                if elapsed >= self.duration_seconds:
                    break
                if self.max_cycles and cycle >= self.max_cycles:
                    break

                t_cycle_start = time.perf_counter()

                # Generate randomized action
                cx = self.rng.uniform(0.1, 0.9)
                cy = self.rng.uniform(0.1, 0.9)
                action_type = self.rng.choice(["click", "move", "key"])
                action_class = self.rng.choice([
                    ActionClass.IDEMPOTENT,
                    ActionClass.CONDITIONALLY_REPEATABLE,
                    ActionClass.NON_IDEMPOTENT,
                ])
                # In 10% of cases, test high-risk safety interception
                authorized = (self.rng.random() >= 0.10)

                # Simulated condition: true for clicks and moves
                sim_satisfied = self.rng.random() < 0.85

                def soak_condition(pre: HumanEyeState, post: HumanEyeState) -> bool:
                    return sim_satisfied

                req = ActionRequest(
                    action_type=action_type,
                    target_coord=(cx, cy),
                    action_class=action_class,
                    action_id=f"soak_{cycle:07d}",
                    expected_postcondition=soak_condition,
                    authorized=authorized,
                )

                try:
                    res = broker.execute(req)
                    status_str = res.status.value
                except Exception as exc:
                    status_str = "error"
                    capture_error_count += 1

                cycle_dur_ms = (time.perf_counter() - t_cycle_start) * 1000.0
                cycle_durations.append(cycle_dur_ms)

                if status_str in status_counts:
                    status_counts[status_str] += 1

                # Check false confirmation: confirmed but condition was unsatisfied
                if status_str == ActionStatus.CONFIRMED.value and not sim_satisfied:
                    false_confirmations += 1

                # Telemetry sampling
                if (cycle % self.telemetry_interval == 0) or (cycle == 0):
                    rss_mb, vms_mb, handles, gdi, user = self.tracker.get_metrics()
                    snap = TelemetrySnapshot(
                        cycle=cycle,
                        elapsed_seconds=round(elapsed, 2),
                        rss_mb=round(rss_mb, 3),
                        vms_mb=round(vms_mb, 3),
                        process_handles=handles,
                        gdi_objects=gdi,
                        user_objects=user,
                        cycle_duration_ms=round(cycle_dur_ms, 3),
                        action_status=status_str,
                        capture_error_count=capture_error_count,
                        timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                    telemetry_history.append(snap)
                    f_trace.write(json.dumps(asdict(snap)) + "\n")

                if (cycle + 1) % 500 == 0:
                    current_rss, _, current_h, current_gdi, current_usr = self.tracker.get_metrics()
                    print(
                        f"[Cycle {cycle + 1}] Elapsed: {elapsed:.1f}s | "
                        f"RSS: {current_rss:.2f} MB | Handles: {current_h} | "
                        f"GDI: {current_gdi} | Confirmed: {status_counts['confirmed']}"
                    )

                cycle += 1

        total_elapsed = time.perf_counter() - start_time
        final_rss, final_vms, final_handles, final_gdi, final_user = self.tracker.get_metrics()

        # Trend analysis: linear regression of RSS against elapsed seconds & cycles
        times_list = [s.elapsed_seconds for s in telemetry_history]
        cycles_list = [float(s.cycle) for s in telemetry_history]
        rss_list = [s.rss_mb for s in telemetry_history]
        handles_list = [float(s.process_handles) for s in telemetry_history]

        slope_mb_per_second = _linear_regression_slope(times_list, rss_list)
        slope_mb_per_hour = slope_mb_per_second * 3600.0
        slope_mb_per_cycle = _linear_regression_slope(cycles_list, rss_list)
        handle_slope_per_hour = _linear_regression_slope(times_list, handles_list) * 3600.0
        handle_delta = final_handles - init_handles

        # Memory leak gate:
        # If running short test run (< 50 cycles), verify RSS growth stays
        # bounded (delta <= 25MB). Growth during the run is the leak signal;
        # an absolute RSS ceiling is not meaningful here because the runner is
        # routinely embedded in a larger process (a pytest session with torch
        # loaded can already exceed any fixed cap), which previously produced a
        # false failure with a perfectly flat delta.
        # If running short CI soak (>= 50 cycles, < 300s), check slope per cycle <= 0.02 MB/cycle (20 KB/cycle).
        # If running sustained soak (>= 300s), check hourly slope <= 5.0 MB/hour.
        if cycle < 50:
            memory_stable = (final_rss - init_rss) <= 25.0
        elif total_elapsed < 300.0:
            memory_stable = (slope_mb_per_cycle <= 0.02)
        else:
            memory_stable = (slope_mb_per_hour <= 5.0)

        # Handle leak gate: handle delta <= 5 handles or handle slope <= 2.0/hr
        handles_stable = (handle_delta <= 5) or (handle_slope_per_hour <= 2.0)
        zero_false_confirmations = (false_confirmations == 0)
        zero_unhandled_crashes = (capture_error_count == 0)

        soak_passed = (
            memory_stable
            and handles_stable
            and zero_false_confirmations
            and zero_unhandled_crashes
            and cycle > 0
        )

        summary: Dict[str, Any] = {
            "benchmark_name": "AVCS Sustained Reliability & Resource Soak Test",
            "milestone": "Milestone 4",
            "schema_version": "1.0",
            "configuration": {
                "duration_target_seconds": self.duration_seconds,
                "duration_target_hours": round(self.duration_seconds / 3600.0, 3),
                "max_cycles": self.max_cycles,
                "base_seed": self.base_seed,
            },
            "execution": {
                "total_cycles_completed": cycle,
                "total_elapsed_seconds": round(total_elapsed, 2),
                "throughput_cycles_per_sec": round(cycle / max(0.001, total_elapsed), 2),
            },
            "resource_telemetry": {
                "initial": {
                    "rss_mb": round(init_rss, 3),
                    "vms_mb": round(init_vms, 3),
                    "process_handles": init_handles,
                    "gdi_objects": init_gdi,
                    "user_objects": init_user,
                },
                "final": {
                    "rss_mb": round(final_rss, 3),
                    "vms_mb": round(final_vms, 3),
                    "process_handles": final_handles,
                    "gdi_objects": final_gdi,
                    "user_objects": final_user,
                },
                "deltas": {
                    "delta_rss_mb": round(final_rss - init_rss, 3),
                    "delta_process_handles": final_handles - init_handles,
                    "delta_gdi_objects": final_gdi - init_gdi,
                    "delta_user_objects": final_user - init_user,
                },
                "trend_slopes": {
                    "rss_slope_mb_per_cycle": round(slope_mb_per_cycle, 6),
                    "rss_slope_mb_per_hour": round(slope_mb_per_hour, 4),
                    "handle_slope_per_hour": round(handle_slope_per_hour, 4),
                },
            },
            "reliability": {
                "action_status_distribution": status_counts,
                "capture_errors_encountered": capture_error_count,
                "false_confirmations": false_confirmations,
                "zero_false_confirmations": zero_false_confirmations,
            },
            "latency_ms": {
                "p50": _percentile(cycle_durations, 0.50),
                "p95": _percentile(cycle_durations, 0.95),
                "p99": _percentile(cycle_durations, 0.99),
            },
            "acceptance_gates": {
                "memory_stable": memory_stable,
                "handles_stable": handles_stable,
                "zero_false_confirmations": zero_false_confirmations,
                "zero_unhandled_crashes": zero_unhandled_crashes,
                "exit_gate_passed": soak_passed,
            },
            "trace_file": str(trace_file.resolve()),
        }

        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n=== Soak Summary written to {summary_file.resolve()} ===")
        print(
            f"Exit Gate Passed: {soak_passed} | Cycles: {cycle} | "
            f"Delta RSS: {final_rss - init_rss:+.2f} MB | RSS Slope: {slope_mb_per_hour:+.2f} MB/h"
        )

        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="AVCS Milestone 4 Soak Test Runner")
    parser.add_argument("--duration-hours", type=float, default=None, help="Duration in hours (e.g. 8.0)")
    parser.add_argument("--duration-seconds", type=float, default=60.0, help="Duration in seconds (default: 60.0)")
    parser.add_argument("--max-cycles", type=int, default=None, help="Max cycles to execute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="benchmarks/results/soak_summary.json", help="Summary JSON output path")
    parser.add_argument("--trace-jsonl", default="benchmarks/results/soak_telemetry.jsonl", help="Trace JSONL output path")
    args = parser.parse_args()

    duration_sec = args.duration_seconds
    if args.duration_hours is not None:
        duration_sec = args.duration_hours * 3600.0

    runner = SoakTestRunner(
        duration_seconds=duration_sec,
        max_cycles=args.max_cycles,
        base_seed=args.seed,
    )

    summary = runner.run(
        summary_path=args.output,
        trace_jsonl_path=args.trace_jsonl,
    )

    return 0 if summary["acceptance_gates"]["exit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
