"""Unit tests for AVCS Milestone 8 Phase D Preflight Engine."""
from __future__ import annotations

import os
import unittest

from avcs_m7_validation.phase_d_preflight import (
    CANONICAL_ARCH_SHA256,
    CANONICAL_MODEL_PARAMS,
    CANONICAL_SHAPE_SHA256,
    PhaseDPreflightEngine,
    PreflightStatus,
)


class PhaseDPreflightTests(unittest.TestCase):
    """Test suite validating Phase D preflight checks and fail-closed gates."""

    def setUp(self) -> None:
        self.engine = PhaseDPreflightEngine()

    @unittest.skipUnless(
        os.environ.get("AVCS_RUN_CANONICAL_PREFLIGHT") == "1",
        "Canonical frozen-candidate certification is an explicit release gate.",
    )
    def test_candidate_freeze_checks(self) -> None:
        """Verifies that parameter count and tensor shapes match canonical frozen candidate."""
        res = self.engine.check_candidate_freeze()
        self.assertEqual(res.status, PreflightStatus.PASSED)
        self.assertEqual(res.details["total_parameters"], CANONICAL_MODEL_PARAMS)
        self.assertEqual(res.details["trainable_parameters"], CANONICAL_MODEL_PARAMS)
        self.assertEqual(res.details["non_trainable_parameters"], 0)
        self.assertEqual(res.details["arch_sha256"], CANONICAL_ARCH_SHA256)
        self.assertEqual(res.details["shape_sha256"], CANONICAL_SHAPE_SHA256)

    @unittest.skipUnless(
        os.environ.get("AVCS_RUN_LIVE_PREFLIGHT") == "1",
        "Physical desktop provenance certification requires an explicit live run.",
    )
    def test_live_provenance_classes(self) -> None:
        """Verifies physical capture and input driver operations on Windows."""
        res = self.engine.check_live_provenance_classes()
        self.assertEqual(res.status, PreflightStatus.PASSED)
        self.assertTrue(res.details["physical_capture_operational"])
        self.assertTrue(res.details["physical_driver_operational"])
        self.assertTrue(res.details["synthetic_adapters_rejected"])

    def test_negative_controls(self) -> None:
        """Verifies that synthetic provenance and duplicate trial IDs fail closed."""
        res = self.engine.check_negative_controls()
        self.assertEqual(res.status, PreflightStatus.PASSED)
        self.assertTrue(res.details["synthetic_provenance_blocked_under_live_enforcement"])
        self.assertTrue(res.details["duplicate_trial_id_blocked"])

    def test_12_scenario_readiness_audit(self) -> None:
        """Verifies that all 12 catalog scenarios have live application harnesses registered."""
        res = self.engine.audit_12_scenario_live_readiness()
        # All 12 scenarios are now mapped
        self.assertEqual(res.status, PreflightStatus.PASSED)
        self.assertEqual(res.details["unmapped_count"], 0)
        self.assertEqual(len(res.details["live_mapped_scenarios"]), 12)
        self.assertIn("native_file_save_as", res.details["live_mapped_scenarios"])
        self.assertIn("overlay_stale_target", res.details["live_mapped_scenarios"])

    def test_unmapped_scenario_fails_closed(self) -> None:
        """Verifies that missing or unmapped scenario harness immediately fails closed to BLOCKED."""
        from avcs_m7_validation.live_harness.registry import LIVE_HARNESS_REGISTRY
        # Temporarily remove a scenario to simulate missing harness
        saved = LIVE_HARNESS_REGISTRY.pop("native_file_save_as", None)
        try:
            res = self.engine.audit_12_scenario_live_readiness()
            self.assertEqual(res.status, PreflightStatus.BLOCKED)
            self.assertGreater(res.details["unmapped_count"], 0)
            self.assertIn("native_file_save_as", res.details["unmapped_scenarios"])
        finally:
            if saved is not None:
                LIVE_HARNESS_REGISTRY["native_file_save_as"] = saved


if __name__ == "__main__":
    unittest.main()
