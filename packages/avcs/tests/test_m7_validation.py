import json
import tempfile
import unittest
from pathlib import Path
from avcs_m7_validation.catalog import SCENARIOS
from avcs_m7_validation.contracts import ActionStatus, ExecutionResult, Mode, OracleResult
from avcs_m7_validation.evaluator import score_trial, summarize, wilson_interval
from avcs_m7_validation.runner import build_plan

class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.scenario = SCENARIOS[0]

    def test_confirmed_and_oracle_success_is_verified(self):
        row = score_trial(run_id="r1", scenario=self.scenario, mode=Mode.HYBRID, seed=4,
            execution=ExecutionResult(ActionStatus.CONFIRMED, 12.5, 1, "uia"),
            oracle=OracleResult(True, "oracle-test", "assertion-1"))
        self.assertTrue(row.verified_success)
        self.assertFalse(row.false_confirmation)

    def test_false_confirmation_is_separately_counted(self):
        row = score_trial(run_id="r2", scenario=self.scenario, mode=Mode.VISUAL_ONLY, seed=4,
            execution=ExecutionResult(ActionStatus.CONFIRMED, 12.5, 1, "visual"),
            oracle=OracleResult(False, "oracle-test", "assertion-1"))
        self.assertTrue(row.false_confirmation)
        self.assertEqual(row.category.value, "false_confirmation")

    def test_oracle_failure_is_invalid_not_counted_as_false_confirmation(self):
        row = score_trial(run_id="r2b", scenario=self.scenario, mode=Mode.HYBRID, seed=4,
            execution=ExecutionResult(ActionStatus.CONFIRMED, 12.5, 1, "visual"),
            oracle=OracleResult(False, "oracle-test", "assertion-1", "oracle_unavailable", valid=False))
        self.assertFalse(row.false_confirmation)
        self.assertFalse(row.verified_success)
        self.assertEqual(row.category.value, "evaluation_invalid")

    def test_ambiguous_is_abstention_not_success(self):
        row = score_trial(run_id="r3", scenario=self.scenario, mode=Mode.HYBRID, seed=4,
            execution=ExecutionResult(ActionStatus.AMBIGUOUS, 12.5, 1, "postcondition"),
            oracle=OracleResult(False, "oracle-test", "assertion-1"))
        self.assertFalse(row.verified_success)
        self.assertEqual(row.category.value, "abstention")

    def test_metrics_by_mode(self):
        rows = [score_trial(run_id=f"r{i}", scenario=self.scenario, mode=Mode.HYBRID, seed=i,
            execution=ExecutionResult(ActionStatus.CONFIRMED, i + 1, 1, "test"),
            oracle=OracleResult(i % 2 == 0, "oracle-test", f"assertion-{i}")) for i in range(10)]
        summary = summarize(rows)
        self.assertEqual(summary["total_trials"], 10)
        self.assertEqual(summary["by_mode"]["hybrid"]["false_confirmations"], 5)
        self.assertEqual(summary["by_mode"]["hybrid"]["oracle_valid_trials"], 10)
        self.assertEqual(summary["by_mode"]["hybrid"]["latency_ms"]["p50"], 5)

    def test_wilson_interval_edges(self):
        low, high = wilson_interval(100, 100)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)

class CatalogAndPlanTests(unittest.TestCase):
    def test_catalog_has_diverse_categories(self):
        self.assertGreaterEqual(len(SCENARIOS), 10)
        categories = {s.category for s in SCENARIOS}
        self.assertGreaterEqual(len(categories), 6)
        self.assertEqual(len({s.scenario_id for s in SCENARIOS}), len(SCENARIOS))

    def test_plan_count_and_reproducibility(self):
        a = build_plan(2, 123, list(Mode))
        b = build_plan(2, 123, list(Mode))
        self.assertEqual(a["planned_trial_count"], 2 * len(SCENARIOS) * 3)
        self.assertEqual(a["trials"], b["trials"])
        self.assertEqual(a["kind"], "plan_only_not_execution_evidence")
        paired = {}
        for trial in a["trials"]:
            key = (trial["scenario_id"], trial["repetition"])
            paired.setdefault(key, {})[trial["mode"]] = trial["seed"]
        self.assertTrue(all(len(set(seed_map.values())) == 1 for seed_map in paired.values()))

if __name__ == "__main__":
    unittest.main()
