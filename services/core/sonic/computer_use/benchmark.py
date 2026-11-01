"""Measured human-versus-SONIC benchmark aggregation.

This module deliberately does not create task names, timings, or success flags.
Callers must supply recorded human measurements and SONIC trials with
independent verification evidence. An unsuccessful or unverified SONIC trial
contributes zero speedup, regardless of how quickly it ended.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from pydantic import BaseModel, Field


class TrialMeasurement(BaseModel):
    """One measured task attempt; verification must be independently assessed."""

    duration_seconds: float = Field(gt=0)
    success: bool
    evidence_verified: bool
    action_count: int | None = Field(default=None, ge=0)


class BenchmarkTaskTrials(BaseModel):
    """Recorded human and SONIC attempts for the same task and conditions."""

    task_family: str
    task_name: str
    is_holdout: bool = False
    human_trials: list[TrialMeasurement]
    sonic_trials: list[TrialMeasurement]


class MultiTrialResult(BaseModel):
    """Statistical summary across repeated trials."""
    task_family: str
    task_name: str
    is_holdout: bool
    trials: int
    human_trials: int
    human_median_seconds: float | None
    human_p25_seconds: float | None
    human_p75_seconds: float | None
    human_variance: float | None
    human_success_rate: float
    sonic_median_seconds: float
    sonic_p25_seconds: float
    sonic_p75_seconds: float
    sonic_variance: float
    sonic_success_rate: float
    trial_speedups: list[float]
    median_speedup: float
    time_reduction_pct: float | None
    action_efficiency_pct: float | None
    certification_eligible: bool


class MultiTrialBenchmarkSuite:
    """Compute statistics only from explicitly supplied trial measurements."""

    MIN_CERTIFICATION_TRIALS = 5

    @classmethod
    def evaluate_task(
        cls,
        task: BenchmarkTaskTrials,
    ) -> MultiTrialResult:
        if not task.human_trials or not task.sonic_trials:
            raise ValueError("human and SONIC trial measurements are required")

        human_successes = [
            trial.duration_seconds
            for trial in task.human_trials
            if trial.success and trial.evidence_verified
        ]
        sonic_durations = [trial.duration_seconds for trial in task.sonic_trials]
        human_median = statistics.median(human_successes) if human_successes else None
        human_quartiles = cls._quartiles(human_successes)
        sonic_quartiles = cls._quartiles(sonic_durations)

        trial_speedups = [
            round(human_median / trial.duration_seconds, 4)
            if human_median is not None and trial.success and trial.evidence_verified
            else 0.0
            for trial in task.sonic_trials
        ]
        median_speedup = round(statistics.median(trial_speedups), 4)
        time_reduction = (
            round((median_speedup - 1.0) * 100, 1)
            if human_median is not None
            else None
        )

        human_actions = [
            trial.action_count
            for trial in task.human_trials
            if trial.success and trial.evidence_verified and trial.action_count is not None
        ]
        sonic_actions = [
            trial.action_count
            for trial in task.sonic_trials
            if trial.success and trial.evidence_verified and trial.action_count is not None
        ]
        action_efficiency = None
        if human_actions and sonic_actions:
            human_action_median = statistics.median(human_actions)
            sonic_action_median = statistics.median(sonic_actions)
            if human_action_median > 0:
                action_efficiency = round(
                    (human_action_median - sonic_action_median)
                    / human_action_median
                    * 100,
                    1,
                )

        return MultiTrialResult(
            task_family=task.task_family,
            task_name=task.task_name,
            is_holdout=task.is_holdout,
            trials=len(task.sonic_trials),
            human_trials=len(task.human_trials),
            human_median_seconds=(round(human_median, 2) if human_median is not None else None),
            human_p25_seconds=human_quartiles[0],
            human_p75_seconds=human_quartiles[1],
            human_variance=(
                round(statistics.variance(human_successes), 2)
                if len(human_successes) > 1
                else None
            ),
            human_success_rate=round(len(human_successes) / len(task.human_trials), 4),
            sonic_median_seconds=round(statistics.median(sonic_durations), 2),
            sonic_p25_seconds=sonic_quartiles[0],
            sonic_p75_seconds=sonic_quartiles[1],
            sonic_variance=(
                round(statistics.variance(sonic_durations), 2)
                if len(sonic_durations) > 1
                else 0.0
            ),
            sonic_success_rate=round(
                sum(trial.success and trial.evidence_verified for trial in task.sonic_trials)
                / len(task.sonic_trials),
                4,
            ),
            trial_speedups=trial_speedups,
            median_speedup=median_speedup,
            time_reduction_pct=time_reduction,
            action_efficiency_pct=action_efficiency,
            certification_eligible=(
                len(task.human_trials) >= cls.MIN_CERTIFICATION_TRIALS
                and len(task.sonic_trials) >= cls.MIN_CERTIFICATION_TRIALS
                and human_median is not None
                and all(
                    trial.success and trial.evidence_verified
                    for trial in task.sonic_trials
                )
            ),
        )

    @staticmethod
    def _quartiles(values: Sequence[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        if len(values) == 1:
            value = round(float(values[0]), 2)
            return value, value
        quartiles = statistics.quantiles(values, n=4)
        return round(quartiles[0], 2), round(quartiles[2], 2)

    @classmethod
    def run_full_suite(
        cls,
        tasks: Sequence[BenchmarkTaskTrials],
    ) -> list[MultiTrialResult]:
        """Evaluate caller-supplied measurements; no tasks or outcomes are fabricated."""
        task_names = [task.task_name for task in tasks]
        if len(task_names) != len(set(task_names)):
            raise ValueError("benchmark task names must be unique")
        return [cls.evaluate_task(task) for task in tasks]
