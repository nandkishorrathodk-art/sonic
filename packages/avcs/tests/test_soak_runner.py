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



@pytest.mark.parametrize(
    ("baseline_mb", "growth_mb", "expected_stable"),
    [(1500.0, 0.0, True), (1500.0, 26.0, False), (500.0, 26.0, False)],
)
def test_short_soak_measures_growth_from_warm_baseline(
    tmp_path, monkeypatch, baseline_mb, growth_mb, expected_stable
):
    """Other tests' model allocations must not masquerade as a soak memory leak."""
    runner = SoakTestRunner(max_cycles=1, warmup_cycles=0, base_seed=999)
    readings = iter([baseline_mb])
    monkeypatch.setattr(
        runner.tracker,
        "get_metrics",
        lambda: (next(readings, baseline_mb + growth_mb), 2000.0, 0, 0, 0),
    )

    summary = runner.run(
        summary_path=str(tmp_path / "summary.json"),
        trace_jsonl_path=str(tmp_path / "trace.jsonl"),
    )

    assert summary["acceptance_gates"]["memory_stable"] is expected_stable
    assert summary["acceptance_gates"]["exit_gate_passed"] is expected_stable
