"""
Tests for Phase 14: Computer-Use Evolution & Hold-out Task Evaluation.
"""

import pytest
from sonic.computer_use.benchmark import (
    BenchmarkTaskTrials,
    MultiTrialBenchmarkSuite,
    MultiTrialResult,
    TrialMeasurement,
)


def test_computer_use_evolution_and_holdout_evaluation():
    def task(name, *, holdout=False):
        return BenchmarkTaskTrials(
            task_family="gui_workflow",
            task_name=name,
            is_holdout=holdout,
            human_trials=[
                TrialMeasurement(duration_seconds=100, success=True, evidence_verified=True)
                for _ in range(5)
            ],
            sonic_trials=[
                TrialMeasurement(duration_seconds=25, success=True, evidence_verified=True)
                for _ in range(5)
            ],
        )

    results = MultiTrialBenchmarkSuite.run_full_suite(
        [
            task("workflow-a"),
            task("workflow-b"),
            task("workflow-c"),
            task("workflow-d", holdout=True),
            task("workflow-e", holdout=True),
            task("workflow-f", holdout=True),
        ]
    )

    assert len(results) == 6
    training_tasks = [r for r in results if not r.is_holdout]
    holdout_tasks = [r for r in results if r.is_holdout]

    assert len(training_tasks) == 3
    assert len(holdout_tasks) == 3

    for r in results:
        assert isinstance(r, MultiTrialResult)
        assert r.trials == 5
        assert r.sonic_success_rate == 1.0
        assert r.median_speedup == 4.0
        assert r.certification_eligible is True


def test_full_suite_rejects_duplicate_task_names():
    task = BenchmarkTaskTrials(
        task_family="gui_workflow",
        task_name="same-task",
        human_trials=[TrialMeasurement(duration_seconds=100, success=True, evidence_verified=True)],
        sonic_trials=[TrialMeasurement(duration_seconds=25, success=True, evidence_verified=True)],
    )

    with pytest.raises(ValueError, match="task names must be unique"):
        MultiTrialBenchmarkSuite.run_full_suite([task, task])
