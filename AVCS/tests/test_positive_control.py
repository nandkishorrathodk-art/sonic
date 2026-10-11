"""Tests for AVCS Milestone 4: Positive-Control Runner and Task Completion."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from avcs.action_broker import ActionBroker, ActionSafetyController
from avcs.human_eye import HumanEye
from benchmarks.positive_control_runner import (
    InteractiveSyntheticCaptureAdapter,
    MockInteractiveSurface,
    run_positive_control_suite,
    run_positive_trial,
)
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def test_positive_control_trial_completion():
    """Verify that an individual positive-control trial confirms successfully."""
    base_driver = VirtualCursorDriver()
    base_cap = InteractiveSyntheticCaptureAdapter().open()
    surface = MockInteractiveSurface(base_cap)
    eye = HumanEye(source=base_cap)
    safety = ActionSafetyController(allow_destructive=False)
    broker = ActionBroker(eye=eye, driver=base_driver, safety_controller=safety, settle_delay_s=0.001)

    for i in range(10):
        record = run_positive_trial(i, 100, broker, surface)
        assert record["task_success"] is True
        assert record["is_confirmed"] is True
        assert record["status"] == "confirmed"


def test_positive_control_suite_execution():
    """Verify that positive-control suite completes and satisfies the >=99% completion gate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "pos_summary.json"
        trace_path = Path(tmpdir) / "pos_trace.jsonl"

        summary = run_positive_control_suite(
            trials_count=20,
            base_seed=777,
            output_summary_path=str(summary_path),
            trace_jsonl_path=str(trace_path),
        )

        assert summary["total_trials"] == 20
        assert summary["task_successes"] == 20
        assert summary["task_completion_rate"] == 1.0
        assert summary["exit_gate_passed"] is True

        assert summary_path.is_file()
        assert trace_path.is_file()

