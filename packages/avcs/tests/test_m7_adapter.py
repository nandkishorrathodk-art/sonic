"""Tests for Milestone 7 repository adapter and independent oracle."""
import os
from pathlib import Path
import tempfile
import unittest

from avcs_m7_validation.catalog import SCENARIOS
from avcs_m7_validation.contracts import ActionStatus, AgentTask, Mode, Scenario
from avcs_m7_validation.independent_oracle import FixtureIndependentOracle, create_oracle
from avcs_m7_validation.repository_adapter import RepositoryValidationAdapter, create_adapter
from avcs_m7_validation.runner import main as runner_main


class M7AdapterAndOracleTests(unittest.TestCase):
    def setUp(self):
        self.adapter = create_adapter()
        self.oracle = create_oracle()

    def test_adapter_fails_closed_visual_only(self):
        task = AgentTask(task_prompt="Click the Apply button")
        result = self.adapter.execute(
            run_id="test-run-001",
            task=task,
            mode=Mode.VISUAL_ONLY,
            seed=42,
            timeout_s=10.0,
        )
        self.assertEqual(result.status, ActionStatus.FAILED)
        self.assertEqual(result.action_count, 0)
        self.assertEqual(result.evidence_method, "none")
        self.assertEqual(result.detail_code, "autonomous_gui_worker_unavailable")
        self.assertFalse(result.metadata["certification_eligible"])
        self.assertEqual(result.metadata["provenance"], "simulation_fixture")

    def test_adapter_fails_closed_uia_only(self):
        task = AgentTask(task_prompt="Select menu item Rename")
        result = self.adapter.execute(
            run_id="test-run-002",
            task=task,
            mode=Mode.UIA_ONLY,
            seed=42,
            timeout_s=10.0,
        )
        self.assertEqual(result.status, ActionStatus.FAILED)
        self.assertEqual(result.action_count, 0)
        self.assertFalse(result.metadata["certification_eligible"])

    def test_adapter_fails_closed_hybrid(self):
        task = AgentTask(task_prompt="Activate the custom-drawn control")
        result = self.adapter.execute(
            run_id="test-run-003",
            task=task,
            mode=Mode.HYBRID,
            seed=42,
            timeout_s=10.0,
        )
        self.assertEqual(result.status, ActionStatus.FAILED)
        self.assertEqual(result.action_count, 0)
        self.assertFalse(result.metadata["certification_eligible"])

    def test_phase_d_runner_blocks_fixture_certification(self):
        from avcs_m7_validation.phase_d_certification import PhaseDCertificationRunner

        with tempfile.TemporaryDirectory() as tmp:
            runner = PhaseDCertificationRunner(runs_per_scenario=1, output_dir=tmp)
            with self.assertRaisesRegex(RuntimeError, "simulation fixture"):
                runner.run_certification()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_oracle_evaluates_all_12_scenarios(self):
        for scenario in SCENARIOS:
            res = self.oracle.evaluate(
                run_id=f"test-{scenario.scenario_id}",
                scenario=scenario,
                seed=100,
            )
            self.assertTrue(res.valid)
            self.assertTrue(res.success)
            self.assertEqual(res.assertion_id, f"assert-{scenario.scenario_id}")

    def test_oracle_handles_invalid_property(self):
        invalid_scenario = Scenario(
            scenario_id="invalid",
            category="test",
            application="test",
            task_prompt="test",
            target_property="non_existent_property_xyz",
        )
        res = self.oracle.evaluate(
            run_id="test-invalid",
            scenario=invalid_scenario,
            seed=1,
        )
        self.assertFalse(res.valid)
        self.assertEqual(res.reason_code, "unknown_target_property")

    def test_runner_execution_with_live_plugins(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_adapter = os.environ.get("AVCS_M7_ADAPTER")
            old_oracle = os.environ.get("AVCS_M7_ORACLE")
            try:
                os.environ["AVCS_M7_ADAPTER"] = "avcs_m7_validation.repository_adapter:create_adapter"
                os.environ["AVCS_M7_ORACLE"] = "avcs_m7_validation.independent_oracle:create_oracle"
                code = runner_main([
                    "--runs-per-scenario", "1",
                    "--seed", "20261009",
                    "--output-dir", tmp,
                ])
                # In 1-run mode, 36 trials < 300 required, so computational_go is false (exit code 3)
                self.assertEqual(code, 3)
                summary_file = Path(tmp) / "summary.json"
                self.assertTrue(summary_file.is_file())
            finally:
                if old_adapter is not None:
                    os.environ["AVCS_M7_ADAPTER"] = old_adapter
                else:
                    os.environ.pop("AVCS_M7_ADAPTER", None)
                if old_oracle is not None:
                    os.environ["AVCS_M7_ORACLE"] = old_oracle
                else:
                    os.environ.pop("AVCS_M7_ORACLE", None)


if __name__ == "__main__":
    unittest.main()
