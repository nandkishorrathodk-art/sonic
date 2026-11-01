"""
Tests for Phase 14: Human vs SONIC Multi-Trial Statistical Comparison.
"""

import pytest
from sonic.computer_use.benchmark import (
    BenchmarkTaskTrials,
    MultiTrialBenchmarkSuite,
    TrialMeasurement,
)


def test_measured_benchmark_models_are_public_computer_use_exports():
    from sonic.computer_use import BenchmarkTaskTrials as ExportedTask
    from sonic.computer_use import TrialMeasurement as ExportedTrial

    assert ExportedTask is BenchmarkTaskTrials
    assert ExportedTrial is TrialMeasurement


def test_human_vs_sonic_multi_trial_statistical_metrics():
    task = BenchmarkTaskTrials(
        task_family="desktop_workflow",
        task_name="save_document",
        human_trials=[
            TrialMeasurement(
                duration_seconds=duration,
                success=True,
                evidence_verified=True,
                action_count=20,
            )
            for duration in (100, 105, 110, 115, 120)
        ],
        sonic_trials=[
            TrialMeasurement(
                duration_seconds=duration,
                success=True,
                evidence_verified=True,
                action_count=8,
            )
            for duration in (25, 30, 40, 50, 80)
        ],
    )
    res = MultiTrialBenchmarkSuite.evaluate_task(task)

    assert res.trials == 5
    assert res.certification_eligible is True
    assert res.human_median_seconds > res.sonic_median_seconds
    assert res.human_p25_seconds > res.sonic_p75_seconds  # SONIC slowest trial faster than human fastest quartile
    assert res.sonic_variance >= 0.0
    assert res.human_variance >= 0.0
    assert res.median_speedup == 2.75
    assert res.time_reduction_pct == 175.0
    assert res.action_efficiency_pct == 60.0


def test_unverified_sonic_trial_contributes_zero_speedup():
    task = BenchmarkTaskTrials(
        task_family="desktop_workflow",
        task_name="save_document",
        human_trials=[
            TrialMeasurement(
                duration_seconds=100,
                success=True,
                evidence_verified=True,
            )
            for _ in range(5)
        ],
        sonic_trials=[
            TrialMeasurement(duration_seconds=20, success=True, evidence_verified=True),
            TrialMeasurement(duration_seconds=30, success=True, evidence_verified=True),
            TrialMeasurement(duration_seconds=10, success=True, evidence_verified=False),
            TrialMeasurement(duration_seconds=40, success=True, evidence_verified=True),
            TrialMeasurement(duration_seconds=50, success=True, evidence_verified=True),
        ],
    )

    result = MultiTrialBenchmarkSuite.evaluate_task(task)

    assert result.sonic_success_rate == pytest.approx(4 / 5)
    assert result.trial_speedups == [5.0, 3.3333, 0.0, 2.5, 2.0]
    assert result.median_speedup == 2.5
    assert result.certification_eligible is False
