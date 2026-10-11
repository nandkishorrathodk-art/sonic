"""Tests for AVCS Milestone 8 — Evidence Certification and Trial Auditor."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from avcs_m7_validation.contracts import (
    ActionStatus,
    EvidenceProvenance,
    Mode,
    OutcomeCategory,
    TrialResult,
)
from avcs_m7_validation.evidence_auditor import (
    AuditViolation,
    TrialEvidenceAuditor,
    ViolationSeverity,
    main as auditor_main,
)


class EvidenceAuditorTests(unittest.TestCase):
    """Test suite for Milestone 8 trial evidence auditor and certification engine."""

    def setUp(self) -> None:
        self.auditor = TrialEvidenceAuditor()
        self.repo_root = Path(__file__).resolve().parent.parent
        self.m7_results_dir = self.repo_root / "m7-results"

    def _create_mock_trial(
        self,
        idx: int,
        run_id: str,
        scenario_id: str = "native_context_menu",
        mode: str = "hybrid",
        status: str = "confirmed",
        oracle_success: bool = True,
        oracle_valid: bool = True,
        false_confirmation: bool = False,
        elapsed_ms: float = 18.5,
        provenance: str = "simulation_fixture",
        app_identity: dict | None = None,
    ) -> dict:
        return {
            "action_count": 2,
            "assertion_id": f"assert-{scenario_id}",
            "category": "false_confirmation" if false_confirmation else ("task_success" if oracle_success else "task_failure"),
            "detail_code": "verified_execution",
            "elapsed_ms": elapsed_ms,
            "evidence_method": "fused_visual_and_uia_verification",
            "false_confirmation": false_confirmation,
            "mode": mode,
            "oracle_id": "m7-fixture-oracle" if provenance == "simulation_fixture" else "m8-live-oracle",
            "oracle_success": oracle_success,
            "oracle_valid": oracle_valid,
            "reported_confirmed": status == "confirmed",
            "run_id": run_id,
            "scenario_id": scenario_id,
            "seed": 1000 + idx,
            "status": status,
            "verified_success": oracle_success and not false_confirmation,
            "provenance": provenance,
            "app_identity": app_identity,
        }

    def test_audit_existing_m7_results(self) -> None:
        """Audits the existing m7-results directory from Milestone 7."""
        if not self.m7_results_dir.is_dir():
            self.skipTest("m7-results directory not present in working tree")

        report = self.auditor.audit_directory(self.m7_results_dir, enforce_live=False)

        self.assertEqual(report.total_records, 360)
        self.assertEqual(report.unique_trial_ids, 360)
        self.assertEqual(report.scenarios_covered, 12)
        self.assertEqual(sorted(report.modes_covered), ["hybrid", "uia_only", "visual_only"])
        self.assertTrue(report.summary_matches_raw_recomputation)
        self.assertEqual(len(report.discrepancies), 0)
        self.assertEqual(report.detected_provenance, EvidenceProvenance.SIMULATION_FIXTURE)

        # Harness gate must pass; live certification gate must be blocked
        self.assertTrue(report.harness_verification_gate_passed)
        self.assertFalse(report.live_certification_gate_passed)

    def test_auditor_detects_duplicate_run_id(self) -> None:
        """Auditor must fail closed on encountering duplicated trial run_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            trials = [
                self._create_mock_trial(1, "trial-001"),
                self._create_mock_trial(2, "trial-001"),  # Duplicate ID
            ]
            trials_path = p / "trials.jsonl"
            with trials_path.open("w", encoding="utf-8") as fp:
                for t in trials:
                    fp.write(json.dumps(t) + "\n")

            report = self.auditor.audit_directory(p)
            self.assertFalse(report.harness_verification_gate_passed)
            self.assertTrue(any(v.code == "DUPLICATE_TRIAL_ID" and v.severity == ViolationSeverity.BLOCKING for v in report.violations))

    def test_auditor_detects_corrupt_json_line(self) -> None:
        """Auditor must fail closed on corrupted JSON lines in trials.jsonl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            trials_path = p / "trials.jsonl"
            trials_path.write_text('{"run_id": "ok-1", "action_count": 1}\n{corrupted json\n', encoding="utf-8")

            report = self.auditor.audit_directory(p)
            self.assertFalse(report.harness_verification_gate_passed)
            self.assertTrue(any(v.code == "CORRUPT_JSONL_LINE" and v.severity == ViolationSeverity.BLOCKING for v in report.violations))

    def test_auditor_detects_missing_mandatory_fields(self) -> None:
        """Auditor must fail closed when trial records omit required schema fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            trials_path = p / "trials.jsonl"
            # Missing status and oracle_id
            incomplete = {"run_id": "trial-001", "scenario_id": "test", "mode": "hybrid"}
            trials_path.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")

            report = self.auditor.audit_directory(p)
            self.assertFalse(report.harness_verification_gate_passed)
            self.assertTrue(any(v.code == "MISSING_RECORD_FIELDS" for v in report.violations))

    def test_auditor_detects_false_confirmation(self) -> None:
        """Auditor must flag false confirmations as blocking violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            trials = [
                self._create_mock_trial(1, "trial-001", false_confirmation=True),
            ]
            trials_path = p / "trials.jsonl"
            with trials_path.open("w", encoding="utf-8") as fp:
                for t in trials:
                    fp.write(json.dumps(t) + "\n")

            report = self.auditor.audit_directory(p)
            self.assertFalse(report.harness_verification_gate_passed)
            self.assertTrue(any(v.code == "FALSE_CONFIRMATION_DETECTED" and v.severity == ViolationSeverity.BLOCKING for v in report.violations))

    def test_auditor_detects_summary_metric_discrepancy(self) -> None:
        """Auditor must detect mismatches between pre-baked summary.json and raw trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            trials = [self._create_mock_trial(i, f"trial-{i:03d}") for i in range(10)]
            trials_path = p / "trials.jsonl"
            with trials_path.open("w", encoding="utf-8") as fp:
                for t in trials:
                    fp.write(json.dumps(t) + "\n")

            # Fake summary claiming 15 trials instead of 10
            fake_summary = {
                "total_trials": 15,
                "by_mode": {"hybrid": {"trials": 15, "oracle_successes": 15}},
            }
            (p / "summary.json").write_text(json.dumps(fake_summary), encoding="utf-8")

            report = self.auditor.audit_directory(p)
            self.assertFalse(report.summary_matches_raw_recomputation)
            self.assertTrue(any(v.code == "SUMMARY_METRICS_DISCREPANCY" for v in report.violations))
            self.assertGreater(len(report.discrepancies), 0)

    def test_auditor_live_application_provenance_validation(self) -> None:
        """Verifies that live application records satisfy live certification gate when valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            live_identity = {
                "process_name": "notepad.exe",
                "hwnd": 45678,
                "window_title": "Document - Notepad",
            }
            from avcs_m7_validation.catalog import SCENARIOS

            # Generate 300+ live trials covering all 12 scenarios and all 3 modes
            trials = []
            idx = 1
            for sc in SCENARIOS:
                for m in ["visual_only", "uia_only", "hybrid"]:
                    for rep in range(10):
                        trials.append(
                            self._create_mock_trial(
                                idx=idx,
                                run_id=f"live-trial-{idx:05d}",
                                scenario_id=sc.scenario_id,
                                mode=m,
                                provenance="live_application",
                                app_identity=live_identity,
                            )
                        )
                        idx += 1

            trials_path = p / "trials.jsonl"
            with trials_path.open("w", encoding="utf-8") as fp:
                for t in trials:
                    fp.write(json.dumps(t) + "\n")

            # Run auditor with live enforcement
            report = self.auditor.audit_directory(p, enforce_live=True)
            self.assertEqual(report.detected_provenance, EvidenceProvenance.LIVE_APPLICATION)
            self.assertTrue(report.harness_verification_gate_passed)
            self.assertTrue(report.live_certification_gate_passed)

    def test_auditor_cli_execution_and_reports(self) -> None:
        """Tests the auditor CLI runner entrypoint."""
        if not self.m7_results_dir.is_dir():
            self.skipTest("m7-results directory not present in working tree")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_out = Path(tmpdir) / "audit_report.md"
            exit_code = auditor_main([
                "--results-dir", str(self.m7_results_dir),
                "--output-report", str(report_out),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(report_out.is_file())
            content = report_out.read_text(encoding="utf-8")
            self.assertIn("Harness & Statistical Verification", content)
            self.assertIn("SIMULATION_FIXTURE", content)


if __name__ == "__main__":
    unittest.main()
