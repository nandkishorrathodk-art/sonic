"""Tests for AVCS Milestone 4: Negative-Control Runner and Fault Invariance."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from avcs.action_broker import ActionBroker, ActionSafetyController
from avcs.capture_adapter import SyntheticCaptureAdapter
from avcs.fault_injection import (
    FaultConfig,
    FaultInjectedCaptureAdapter,
    FaultInjectedDriver,
    FaultInjectionEnvironment,
    FaultType,
)
from avcs.human_eye import HumanEye
from benchmarks.negative_control_runner import run_negative_control_suite, run_negative_trial
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def test_negative_control_trial_zero_false_confirmations():
    """Verify that a single negative-control trial never yields a false confirmation."""
    base_driver = VirtualCursorDriver()
    base_cap = SyntheticCaptureAdapter().open()
    cap = FaultInjectedCaptureAdapter(base_cap, FaultConfig())
    eye = HumanEye(source=cap)
    safety = ActionSafetyController(allow_destructive=False)
    broker = ActionBroker(eye=eye, driver=base_driver, safety_controller=safety, settle_delay_s=0.001)

    for i in range(25):
        record = run_negative_trial(i, 42, broker, base_driver, cap)
        assert record["is_confirmed"] is False, f"Trial {i} was falsely confirmed!"
        assert record["status"] in ("failed", "blocked", "ambiguous")


def test_negative_control_suite_execution():
    """Verify that negative-control suite completes and generates valid summary and trace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "neg_summary.json"
        trace_path = Path(tmpdir) / "neg_trace.jsonl"

        summary = run_negative_control_suite(
            trials_count=30,
            base_seed=123,
            output_summary_path=str(summary_path),
            trace_jsonl_path=str(trace_path),
        )

        assert summary["total_trials"] == 30
        assert summary["false_confirmations"] == 0
        assert summary["false_confirmation_rate"] == 0.0
        assert summary["exit_gate_passed"] is True
        assert summary["rule_of_three_95_upper_bound"] == pytest.approx(3.0 / 30.0)

        assert summary_path.is_file()
        assert trace_path.is_file()

        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 30
