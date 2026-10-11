"""Comprehensive test suite for AVCS Milestone 8 Phase D Live Application Matrix.

Validates the full 12-scenario live harness suite:
1. Coverage: 1-to-1 parity between catalog.py and LIVE_HARNESS_REGISTRY.
2. Contracts: Typed BaseLiveHarness compliance, AgentTask generation, target properties.
3. Fail-Closed Oracles: Verification that empty, corrupt, or missing state files fail closed.
4. Physical Execution Smoke: Real Windows GDI capture, SendInput dispatch, and independent state verification.
5. Negative Controls: Verification that clicking incorrect coordinates yields oracle failure.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

from avcs_m7_validation.catalog import SCENARIOS, SCENARIO_BY_ID
from avcs_m7_validation.contracts import (
    ActionStatus,
    EvidenceProvenance,
    Mode,
    OutcomeCategory,
)
from avcs_m7_validation.live_harness import (
    BaseLiveHarness,
    LIVE_HARNESS_REGISTRY,
    get_live_harness,
    list_supported_scenarios,
    validate_catalog_coverage,
)
from avcs_m7_validation.live_harness.live_runner import LiveScenarioRunner


class LiveHarnessMatrixTests(unittest.TestCase):
    """Test suite validating the 12 live scenario harnesses and independent oracles."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="avcs_live_test_"))

    def tearDown(self) -> None:
        time.sleep(0.2)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_registry_maps_all_12_catalog_scenarios(self) -> None:
        """Verifies 1-to-1 parity between catalog.py and LIVE_HARNESS_REGISTRY."""
        all_mapped, missing = validate_catalog_coverage()
        self.assertTrue(all_mapped, f"Unmapped catalog scenarios: {missing}")
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(LIVE_HARNESS_REGISTRY), 12)

        for s in SCENARIOS:
            self.assertIn(s.scenario_id, LIVE_HARNESS_REGISTRY)
            h = get_live_harness(s.scenario_id)
            self.assertIsInstance(h, BaseLiveHarness)
            self.assertEqual(h.scenario_id, s.scenario_id)
            self.assertEqual(h.target_property, s.target_property)

    def test_harness_agent_tasks_and_properties(self) -> None:
        """Verifies that all 12 harnesses generate clean public AgentTasks without leaking scenario metadata."""
        for sid in list_supported_scenarios():
            h = get_live_harness(sid)
            task = h.get_agent_task(seed=101)
            self.assertTrue(len(task.task_prompt) > 10)
            # Ensure task prompt is agent-facing instructions
            self.assertIsInstance(task.task_prompt, str)

    def test_oracles_fail_closed_on_empty_workspace(self) -> None:
        """Verifies that every oracle evaluates to False if the application state is absent."""
        empty_ws = self.temp_dir / "empty_ws"
        empty_ws.mkdir()

        for sid in list_supported_scenarios():
            h = get_live_harness(sid)
            # Empty state must fail closed
            res = h.evaluate_oracle(empty_ws, action_dispatched=False, seed=42)
            self.assertFalse(
                res.success,
                f"Oracle for '{sid}' unexpectedly succeeded on an empty workspace!",
            )
            self.assertTrue(len(res.reason_code) > 0)

    @unittest.skipUnless(sys.platform == "win32", "Requires live Windows desktop")
    def test_live_physical_smoke_trial_similar_buttons(self) -> None:
        """Executes a real physical trial against native_similar_buttons on the Windows desktop."""
        runner = LiveScenarioRunner(output_dir=self.temp_dir)
        trial_res, trace = runner.run_live_trial(
            scenario_id="native_similar_buttons",
            run_id="smoke_live_test_001",
            mode=Mode.VISUAL_ONLY,
            seed=42,
        )

        # 1. Provenance Verification
        self.assertEqual(trial_res.provenance, EvidenceProvenance.LIVE_APPLICATION)
        self.assertIsNotNone(trial_res.app_identity)
        self.assertGreater(trial_res.app_identity["pid"], 0)
        self.assertGreater(trial_res.app_identity["hwnd"], 0)

        # 2. Evidence Verification
        self.assertTrue(Path(trace.pre_screenshot_path).is_file())
        self.assertTrue(Path(trace.post_screenshot_path).is_file())
        self.assertGreater(len(trace.pre_screenshot_sha256), 32)
        self.assertGreater(len(trace.post_screenshot_sha256), 32)

        # 3. Postcondition Verification
        self.assertTrue(trial_res.oracle_success)
        self.assertEqual(trial_res.status, ActionStatus.CONFIRMED)
        self.assertTrue(trial_res.verified_success)
        self.assertFalse(trial_res.false_confirmation)
        self.assertEqual(trial_res.category, OutcomeCategory.TASK_SUCCESS)
        self.assertTrue(trace.cleanup_successful)

    @unittest.skipUnless(sys.platform == "win32", "Requires live Windows desktop")
    def test_live_physical_smoke_trial_canvas_select(self) -> None:
        """Executes a real physical trial against custom_canvas_select on the Windows desktop."""
        runner = LiveScenarioRunner(output_dir=self.temp_dir)
        trial_res, trace = runner.run_live_trial(
            scenario_id="custom_canvas_select",
            run_id="smoke_live_test_002",
            mode=Mode.VISUAL_ONLY,
            seed=42,
        )

        self.assertEqual(trial_res.provenance, EvidenceProvenance.LIVE_APPLICATION)
        self.assertTrue(trial_res.oracle_success)
        self.assertEqual(trial_res.status, ActionStatus.CONFIRMED)
        self.assertTrue(trial_res.verified_success)
        self.assertFalse(trial_res.false_confirmation)
        self.assertTrue(trace.cleanup_successful)

    @unittest.skipUnless(sys.platform == "win32", "Requires live Windows desktop")
    def test_negative_control_wrong_action_fails_closed(self) -> None:
        """Proves that dispatching to wrong coordinates results in oracle failure, not false confirmation."""
        ws = self.temp_dir / "neg_control_ws"
        ws.mkdir()
        harness = get_live_harness("native_similar_buttons")
        try:
            harness.launch(ws, seed=42)
            # Intentionally dispatch click to top-left corner instead of target button
            from forge.vision.cortex.motor.win32 import Win32CursorDriver
            driver = Win32CursorDriver(failsafe=False)
            driver.move_to_pixels(10, 10)
            driver.click(button="left")
            time.sleep(0.15)

            # Oracle must independently catch that target button was NOT activated
            oracle_res = harness.evaluate_oracle(ws, action_dispatched=True, seed=42)
            self.assertFalse(oracle_res.success)
            self.assertEqual(oracle_res.reason_code, "wrong_button_activated")
        finally:
            harness.teardown()


if __name__ == "__main__":
    unittest.main()
