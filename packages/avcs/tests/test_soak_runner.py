"""Tests for AVCS Milestone 4: Soak Runner and System Resource Tracker."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmarks.soak_runner import (
    SoakTestRunner,
    SystemResourceTracker,
    _linear_regression_slope,
    _percentile,
)


def test_linear_regression_slope_calculation():
    """Verify linear regression slope calculation for known trends."""
    assert _linear_regression_slope([], []) == 0.0
    assert _linear_regression_slope([1.0], [5.0]) == 0.0

    # Flat line: dy/dx = 0
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [10.0, 10.0, 10.0, 10.0]
    assert _linear_regression_slope(xs, ys) == pytest.approx(0.0)

    # Slope = 2.5
    ys_sloped = [0.0, 2.5, 5.0, 7.5]
    assert _linear_regression_slope(xs, ys_sloped) == pytest.approx(2.5)


def test_percentile_helper():
    """Verify percentile calculation edge cases."""
    assert _percentile([], 0.5) is None
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(vals, 0.5) == 30.0
    assert _percentile(vals, 0.95) == 50.0


def test_system_resource_tracker():
    """Verify system resource tracker produces non-negative valid metrics."""
    tracker = SystemResourceTracker()
    rss, vms, handles, gdi, user = tracker.get_metrics()

    assert rss > 0.0
    assert vms > 0.0
    assert handles >= 0
    assert gdi >= 0
    assert user >= 0


def test_soak_runner_short_run():
    """Verify that SoakTestRunner completes short run and verifies stability gates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "test_soak_summary.json"
        trace_path = Path(tmpdir) / "test_soak_trace.jsonl"

        runner = SoakTestRunner(
            duration_seconds=15.0,
            max_cycles=15,
            telemetry_interval_cycles=5,
            warmup_cycles=5,
            base_seed=999,
        )

        summary = runner.run(
            summary_path=str(summary_path),
            trace_jsonl_path=str(trace_path),
        )

        assert summary["execution"]["total_cycles_completed"] >= 10
        assert summary["acceptance_gates"]["zero_false_confirmations"] is True
        assert summary["acceptance_gates"]["zero_unhandled_crashes"] is True
        assert summary["acceptance_gates"]["exit_gate_passed"] is True

        assert summary_path.is_file()
        assert trace_path.is_file()


def test_short_run_memory_gate_ignores_absolute_rss_baseline():
    """A flat run must pass even when the host process already has a large RSS.

    The runner is normally embedded in a larger process (a pytest session with
    torch loaded already exceeds 1 GB), so an absolute RSS ceiling is not a
    valid leak signal. This reproduced a false failure in CI where delta RSS
    was 0.0 MB yet the gate reported memory_stable=False.
    """
    ballast = bytearray(700 * 1024 * 1024)
    for i in range(0, len(ballast), 4096):
        ballast[i] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = SoakTestRunner(
            duration_seconds=15.0,
            max_cycles=15,
            telemetry_interval_cycles=5,
            warmup_cycles=5,
            base_seed=999,
        )
        summary = runner.run(
            summary_path=f"{tmpdir}/s.json",
            trace_jsonl_path=f"{tmpdir}/t.jsonl",
        )

    final_rss = summary["resource_telemetry"]["final"]["rss_mb"]
    assert final_rss > 1000.0, "test premise: host RSS should exceed the old cap"
    assert summary["acceptance_gates"]["memory_stable"] is True
    assert summary["acceptance_gates"]["exit_gate_passed"] is True


def test_short_run_memory_gate_still_fails_on_real_growth(monkeypatch):
    """A genuine RSS leak must still fail the short-run gate."""
    grown = {"rss": 0.0}
    original = SystemResourceTracker.get_metrics

    def leaking_get_metrics(self):
        rss, vms, handles, gdi, user = original(self)
        grown["rss"] += 30.0  # 30 MB of growth > the 25 MB budget
        return grown["rss"], vms, handles, gdi, user

    monkeypatch.setattr(SystemResourceTracker, "get_metrics", leaking_get_metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = SoakTestRunner(
            duration_seconds=15.0,
            max_cycles=15,
            telemetry_interval_cycles=5,
            warmup_cycles=5,
            base_seed=999,
        )
        summary = runner.run(
            summary_path=f"{tmpdir}/s.json",
            trace_jsonl_path=f"{tmpdir}/t.jsonl",
        )

    assert summary["acceptance_gates"]["memory_stable"] is False
    assert summary["acceptance_gates"]["exit_gate_passed"] is False

