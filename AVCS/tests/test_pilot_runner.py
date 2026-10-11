import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path

from avcs_pilot.contracts import ActionStatus, AgentExecution, PilotScenario, PublicTask
from avcs_pilot.evaluator import verify_exact_text
from avcs_pilot.runner import aggregate_records, load_scenarios, percentile, run_pilot, wilson_interval


class PilotContractTests(unittest.TestCase):
    def test_status_is_validated(self):
        with self.assertRaises(ValueError):
            AgentExecution.from_mapping({"status": "SUCCESS_BUT_UNVERIFIED"})

    def test_negative_action_count_is_rejected(self):
        with self.assertRaises(ValueError):
            AgentExecution.from_mapping({"status": "FAILED", "action_count": -1})

    def test_public_task_has_no_oracle_fields(self):
        scenario = PilotScenario(
            task_id="x", workflow_id="wf", application="Notepad", prompt_template="Save {output_path}",
            timeout_s=10, max_actions=2, risk_level="LOW", output_relative_path="out.txt", expected_text="secret oracle",
        )
        task = scenario.public_task("/tmp/run", "x-1")
        self.assertIsInstance(task, PublicTask)
        self.assertFalse(hasattr(task, "expected_text"))
        self.assertFalse(hasattr(task, "output_relative_path"))

    def test_exact_text_verifier_passes_and_fails(self):
        scenario = PilotScenario(
            task_id="x", workflow_id="wf", application="Notepad", prompt_template="Save {output_path}",
            timeout_s=10, max_actions=2, risk_level="LOW", output_relative_path="out.txt", expected_text="expected\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out.txt").write_text("expected\n", encoding="utf-8")
            self.assertTrue(verify_exact_text(scenario, root).passed)
            (root / "out.txt").write_text("wrong\n", encoding="utf-8")
            self.assertFalse(verify_exact_text(scenario, root).passed)

    def test_verifier_rejects_path_escape(self):
        scenario = PilotScenario(
            task_id="x", workflow_id="wf", application="Notepad", prompt_template="x",
            timeout_s=10, max_actions=2, risk_level="LOW", output_relative_path="../outside.txt", expected_text="x",
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_exact_text(scenario, Path(tmp))
            self.assertFalse(result.passed)
            self.assertEqual(result.evidence_code, "PATH_ESCAPE")

    def test_public_task_rejects_output_path_escape_before_execution(self):
        scenario = PilotScenario(
            task_id="x", workflow_id="wf", application="Notepad", prompt_template="Save {output_path}",
            timeout_s=10, max_actions=2, risk_level="LOW", output_relative_path="../outside.txt", expected_text="x",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                scenario.public_task(tmp, "x-1")

    def test_false_confirmation_is_counted(self):
        records = [{"workflow_id":"wf","execution":{"status":"CONFIRMED","duration_ms":5},"verification":{"passed":False}}]
        summary = aggregate_records(records, expected_runs=1)
        self.assertEqual(summary["false_confirmations"], 1)
        self.assertFalse(summary["acceptance_gates"]["zero_false_confirmations"]["pass"])

    def test_positive_completion_uses_independent_verifier(self):
        records = [{"workflow_id":"wf","execution":{"status":"AMBIGUOUS","duration_ms":8},"verification":{"passed":True}}]
        summary = aggregate_records(records, expected_runs=1)
        self.assertEqual(summary["verified_successes"], 1)
        self.assertEqual(summary["reported_confirmations"], 0)

    def test_action_budget_violation_fails_gate(self):
        records = [{"workflow_id":"wf","execution":{"status":"CONFIRMED","duration_ms":8},"verification":{"passed":True},"safety":{"action_budget_exceeded":True},"risk_level":"LOW"}]
        summary = aggregate_records(records, expected_runs=1)
        self.assertFalse(summary["acceptance_gates"]["action_budget_compliance"]["pass"])
        self.assertFalse(summary["computational_go"])

    def test_path_escape_fails_gate(self):
        records = [{"workflow_id":"wf","execution":{"status":"BLOCKED","duration_ms":8},"verification":{"passed":False,"evidence_code":"PATH_ESCAPE"},"safety":{"action_budget_exceeded":False},"risk_level":"LOW"}]
        summary = aggregate_records(records, expected_runs=1)
        self.assertFalse(summary["acceptance_gates"]["no_path_escape"]["pass"])

    def test_non_low_risk_record_fails_gate(self):
        records = [{"workflow_id":"wf","execution":{"status":"BLOCKED","duration_ms":8},"verification":{"passed":False},"safety":{"action_budget_exceeded":False},"risk_level":"HIGH"}]
        summary = aggregate_records(records, expected_runs=1)
        self.assertFalse(summary["acceptance_gates"]["low_risk_scope_only"]["pass"])

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([0, 10], 0.5), 5.0)

    def test_wilson_empty_sample(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_scenario_config_has_three_unique_workflows(self):
        path = Path(__file__).parents[1] / "config" / "pilot_scenarios.json"
        scenarios = load_scenarios(path)
        self.assertEqual(len(scenarios), 3)
        self.assertEqual(len({s.workflow_id for s in scenarios}), 3)

    def test_audit_record_redacts_free_form_desktop_text(self):
        execution = AgentExecution.from_mapping({
            "status": "FAILED",
            "message": "Customer secret: do not store",
            "diagnostics": {"error_code": "INPUT_LOST", "raw_screenshot_ocr": "secret"},
        }).to_record()
        serialized = json.dumps(execution)
        self.assertNotIn("Customer secret", serialized)
        self.assertNotIn("raw_screenshot_ocr", serialized)
        self.assertIn("INPUT_LOST", serialized)

    def test_mock_adapter_exercises_runner_without_claiming_live_pilot(self):
        class MockAdapter:
            async def execute(self, task: PublicTask):
                prompt = task.prompt
                if task.workflow_id == "notepad_create_save_exact_text":
                    match = re.search(r"Save it as this path: (.+)", prompt)
                    target = Path(match.group(1))
                    target.write_text("AVCS CONTROLLED PILOT\nWorkflow: create-save\nRun marker: AVCS-PILOT-ALPHA\n", encoding="utf-8")
                elif task.workflow_id == "notepad_append_single_line":
                    match = re.search(r"open this existing pilot fixture: (.+)", prompt)
                    target = Path(match.group(1))
                    target.write_text(target.read_text(encoding="utf-8") + "APPENDED-BY-AVCS-PILOT\n", encoding="utf-8")
                elif task.workflow_id == "notepad_find_replace_token":
                    match = re.search(r"open this pilot fixture: (.+)", prompt)
                    target = Path(match.group(1))
                    target.write_text(target.read_text(encoding="utf-8").replace("PILOT-OLD-7F31", "PILOT-NEW-7F31"), encoding="utf-8")
                return {"status": "CONFIRMED", "action_count": 2, "duration_ms": 0.5, "provider": "test-double", "model": "not-a-model"}

        config = Path(__file__).parents[1] / "config" / "pilot_scenarios.json"
        scenarios = load_scenarios(config)
        with tempfile.TemporaryDirectory() as tmp:
            summary = asyncio.run(run_pilot(scenarios, MockAdapter(), Path(tmp), runs_per_workflow=1, seed=3))
        self.assertEqual(summary["trials"], 3)
        self.assertEqual(summary["verified_successes"], 3)
        self.assertEqual(summary["false_confirmations"], 0)
        self.assertTrue(summary["computational_go"])


if __name__ == "__main__":
    unittest.main()
