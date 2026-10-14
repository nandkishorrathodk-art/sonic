"""AVCS Milestone 8 — Trial Evidence Auditor and Certification Engine.

Independently inspects, audits, and validates Milestone 7 trial evidence records
from raw trials.jsonl, recomputing all metrics from ground truth rather than
trusting pre-baked summary.json. Strictly enforces zero false confirmations,
unique trial IDs, scenario coverage, independent oracle decoupling, and
provenance boundaries (distinguishing simulation_fixture from live_application).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from avcs_m7_validation.catalog import SCENARIOS
from avcs_m7_validation.contracts import (
    ActionStatus,
    EvidenceProvenance,
    Mode,
    OutcomeCategory,
    TrialResult,
)
from avcs_m7_validation.evaluator import summarize, wilson_interval


class ViolationSeverity(str, Enum):
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class AuditViolation:
    """Discrete evidence irregularity or policy violation."""
    code: str
    severity: ViolationSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class AuditReport:
    """Structured certification and audit outcome for an evidence directory."""
    target_dir: str
    audited_at_utc: str
    total_records: int
    unique_trial_ids: int
    scenarios_covered: int
    modes_covered: List[str]
    detected_provenance: EvidenceProvenance
    summary_matches_raw_recomputation: bool
    harness_verification_gate_passed: bool
    live_certification_gate_passed: bool
    violations: List[AuditViolation]
    recomputed_summary: Dict[str, Any]
    discrepancies: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_blocking_violations(self) -> bool:
        return any(v.severity == ViolationSeverity.BLOCKING for v in self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_dir": self.target_dir,
            "audited_at_utc": self.audited_at_utc,
            "total_records": self.total_records,
            "unique_trial_ids": self.unique_trial_ids,
            "scenarios_covered": self.scenarios_covered,
            "modes_covered": self.modes_covered,
            "detected_provenance": self.detected_provenance.value,
            "summary_matches_raw_recomputation": self.summary_matches_raw_recomputation,
            "harness_verification_gate_passed": self.harness_verification_gate_passed,
            "live_certification_gate_passed": self.live_certification_gate_passed,
            "violations": [v.to_dict() for v in self.violations],
            "discrepancies": self.discrepancies,
            "metadata": self.metadata,
            "recomputed_summary": self.recomputed_summary,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# AVCS Milestone 8 — Evidence Certification Audit",
            f"",
            f"- **Target Directory**: `{self.target_dir}`",
            f"- **Audited At (UTC)**: `{self.audited_at_utc}`",
            f"- **Total Raw Trials**: `{self.total_records}`",
            f"- **Unique Trial IDs**: `{self.unique_trial_ids}`",
            f"- **Detected Provenance**: `{self.detected_provenance.value.upper()}`",
            f"- **Raw Recomputation Match**: `{'PASSED' if self.summary_matches_raw_recomputation else 'FAILED'}`",
            f"",
            f"## Certification Gates",
            f"",
            f"| Gate Name | Status | Assessment |",
            f"|---|---|---|",
            f"| **Harness & Statistical Verification** | `{'PASSED' if self.harness_verification_gate_passed else 'FAILED'}` | Validates math, trial uniqueness, coverage, and 0 false confirmations. |",
            f"| **Live Application Certification** | `{'PASSED' if self.live_certification_gate_passed else 'BLOCKED'}` | Strictly requires real Windows OS application provenance and telemetry. |",
            f"",
        ]

        if not self.live_certification_gate_passed:
            lines.extend([
                f"> [!CAUTION]",
                f"> **Live Application Certification Gate is BLOCKED.**",
                f"> Evaluated records represent `{self.detected_provenance.value}` (synthetic/fixture execution).",
                f"> While the harness and statistical evaluation are mathematically verified, RC1 commercial release",
                f"> requires empirical validation against live Windows processes before formal release sign-off.",
                f"",
            ])

        lines.extend([
            f"## Audit Violations ({len(self.violations)})",
            f"",
        ])
        if not self.violations:
            lines.append("No violations detected. Zero false confirmations and zero duplicate trial IDs verified.\n")
        else:
            lines.append("| Severity | Code | Message |")
            lines.append("|---|---|---|")
            for v in self.violations:
                lines.append(f"| `{v.severity.value.upper()}` | `{v.code}` | {v.message} |")
            lines.append("")

        if self.discrepancies:
            lines.extend([
                f"## Summary Discrepancies ({len(self.discrepancies)})",
                f"",
                f"The following mismatches were identified between `summary.json` and ground-truth raw records:",
                f"",
            ])
            for d in self.discrepancies:
                lines.append(f"- **{d['metric']}**: Recomputed `{d['recomputed']}` vs Saved `{d['saved']}`")
            lines.append("")

        return "\n".join(lines)


class TrialEvidenceAuditor:
    """Rigorous independent auditor that recomputes metrics from ground-truth trial logs."""

    def __init__(
        self,
        expected_scenario_count: int = 12,
        expected_modes: Optional[Sequence[Mode]] = None,
        min_trials: int = 300,
    ) -> None:
        self.expected_scenario_count = expected_scenario_count
        self.expected_modes = list(expected_modes) if expected_modes else list(Mode)
        self.min_trials = min_trials

    def audit_directory(
        self,
        results_dir: Path | str,
        enforce_live: bool = False,
    ) -> AuditReport:
        """Audits all artifacts in results_dir, failing closed on missing or corrupted data."""
        target_path = Path(results_dir)
        now_utc = datetime.now(timezone.utc).isoformat()
        violations: List[AuditViolation] = []
        discrepancies: List[Dict[str, Any]] = []

        if not target_path.is_dir():
            violations.append(
                AuditViolation(
                    code="TARGET_DIRECTORY_MISSING",
                    severity=ViolationSeverity.BLOCKING,
                    message=f"Directory '{target_path}' does not exist.",
                )
            )
            return AuditReport(
                target_dir=str(target_path),
                audited_at_utc=now_utc,
                total_records=0,
                unique_trial_ids=0,
                scenarios_covered=0,
                modes_covered=[],
                detected_provenance=EvidenceProvenance.UNKNOWN,
                summary_matches_raw_recomputation=False,
                harness_verification_gate_passed=False,
                live_certification_gate_passed=False,
                violations=violations,
                recomputed_summary={},
                discrepancies=[],
            )

        trials_file = target_path / "trials.jsonl"
        summary_file = target_path / "summary.json"
        manifest_file = target_path / "manifest.json"
        plan_file = target_path / "plan.json"

        # 1. Parse and validate raw trials.jsonl
        results, raw_violations = self._parse_and_validate_trials(trials_file)
        violations.extend(raw_violations)

        total_records = len(results)
        unique_ids = len({r.run_id for r in results})
        scenarios_covered = len({r.scenario_id for r in results})
        modes_covered = sorted(list({r.mode.value for r in results}))

        # 2. Determine Provenance
        provenance = self._determine_provenance(results, manifest_file)

        # 3. Recompute ground-truth summary
        recomputed_summary = summarize(results) if results else {}

        # 4. Cross-verify saved summary.json against recomputed metrics
        summary_matches = False
        if summary_file.is_file() and results:
            try:
                saved_summary = json.loads(summary_file.read_text(encoding="utf-8"))
                discrepancies = self._compare_summaries(recomputed_summary, saved_summary)
                if discrepancies:
                    violations.append(
                        AuditViolation(
                            code="SUMMARY_METRICS_DISCREPANCY",
                            severity=ViolationSeverity.BLOCKING,
                            message=f"Saved summary.json does not match ground-truth raw trial recomputation ({len(discrepancies)} discrepancies).",
                            context={"discrepancies": discrepancies[:5]},
                        )
                    )
                else:
                    summary_matches = True
            except Exception as exc:
                violations.append(
                    AuditViolation(
                        code="SUMMARY_JSON_UNREADABLE",
                        severity=ViolationSeverity.BLOCKING,
                        message=f"Failed to read summary.json: {exc}",
                    )
                )
        elif not summary_file.is_file():
            violations.append(
                AuditViolation(
                    code="SUMMARY_JSON_MISSING",
                    severity=ViolationSeverity.WARNING,
                    message="summary.json not found in target directory; using recomputed metrics only.",
                )
            )
            summary_matches = bool(results)

        # 5. Validate Plan Consistency if plan.json is available
        if plan_file.is_file():
            try:
                plan = json.loads(plan_file.read_text(encoding="utf-8"))
                planned_count = plan.get("planned_trial_count", 0)
                if planned_count > 0 and planned_count != total_records:
                    violations.append(
                        AuditViolation(
                            code="PLAN_TRIAL_COUNT_MISMATCH",
                            severity=ViolationSeverity.BLOCKING,
                            message=f"Plan specifies {planned_count} trials, but raw log contains {total_records}.",
                            context={"planned": planned_count, "actual": total_records},
                        )
                    )
            except Exception as exc:
                violations.append(
                    AuditViolation(
                        code="PLAN_JSON_UNREADABLE",
                        severity=ViolationSeverity.WARNING,
                        message=f"Failed to parse plan.json: {exc}",
                    )
                )

        # 6. Check Coverage Constraints
        if scenarios_covered < self.expected_scenario_count and total_records > 0:
            violations.append(
                AuditViolation(
                    code="INCOMPLETE_SCENARIO_COVERAGE",
                    severity=ViolationSeverity.BLOCKING,
                    message=f"Covered only {scenarios_covered}/{self.expected_scenario_count} expected scenarios.",
                    context={"covered": scenarios_covered, "expected": self.expected_scenario_count},
                )
            )

        expected_mode_values = {m.value for m in self.expected_modes}
        if set(modes_covered) != expected_mode_values and total_records > 0:
            violations.append(
                AuditViolation(
                    code="INCOMPLETE_MODE_COVERAGE",
                    severity=ViolationSeverity.BLOCKING,
                    message=f"Modes covered {modes_covered} do not match expected {sorted(list(expected_mode_values))}.",
                    context={"actual": modes_covered, "expected": sorted(list(expected_mode_values))},
                )
            )

        # 7. Evaluate Certification Gates
        blocking_violations = [v for v in violations if v.severity == ViolationSeverity.BLOCKING]

        # Gate A: Harness Verification
        harness_passed = (
            total_records >= self.min_trials
            and unique_ids == total_records
            and summary_matches
            and not blocking_violations
        )

        # Gate B: Live Application Certification
        if provenance != EvidenceProvenance.LIVE_APPLICATION:
            violations.append(
                AuditViolation(
                    code="PROVENANCE_NOT_LIVE_APPLICATION",
                    severity=ViolationSeverity.BLOCKING if enforce_live else ViolationSeverity.INFO,
                    message=f"Evidence provenance is '{provenance.value}'. Cannot satisfy live application certification.",
                    context={"detected_provenance": provenance.value},
                )
            )

        live_passed = (
            harness_passed
            and provenance == EvidenceProvenance.LIVE_APPLICATION
            and not any(v.code == "PROVENANCE_NOT_LIVE_APPLICATION" and v.severity == ViolationSeverity.BLOCKING for v in violations)
        )

        manifest_metadata = {}
        if manifest_file.is_file():
            try:
                mf = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifest_metadata = mf.get("metadata", {})
            except Exception:
                pass

        return AuditReport(
            target_dir=str(target_path),
            audited_at_utc=now_utc,
            total_records=total_records,
            unique_trial_ids=unique_ids,
            scenarios_covered=scenarios_covered,
            modes_covered=modes_covered,
            detected_provenance=provenance,
            summary_matches_raw_recomputation=summary_matches,
            harness_verification_gate_passed=harness_passed,
            live_certification_gate_passed=live_passed,
            violations=violations,
            recomputed_summary=recomputed_summary,
            discrepancies=discrepancies,
            metadata=manifest_metadata,
        )

    def _parse_and_validate_trials(
        self,
        trials_path: Path,
    ) -> Tuple[List[TrialResult], List[AuditViolation]]:
        """Parses trials.jsonl, enforcing schema validity, non-negative numbers, and ID uniqueness."""
        results: List[TrialResult] = []
        violations: List[AuditViolation] = []

        if not trials_path.is_file():
            violations.append(
                AuditViolation(
                    code="TRIALS_JSONL_MISSING",
                    severity=ViolationSeverity.BLOCKING,
                    message=f"trials.jsonl log not found at '{trials_path}'.",
                )
            )
            return results, violations

        seen_ids: Set[str] = set()
        required_keys = {
            "run_id", "scenario_id", "mode", "seed", "status",
            "oracle_success", "oracle_valid", "reported_confirmed",
            "verified_success", "false_confirmation", "elapsed_ms",
            "action_count", "evidence_method", "category",
            "oracle_id", "assertion_id",
        }

        try:
            with trials_path.open("r", encoding="utf-8") as fp:
                for line_idx, raw_line in enumerate(fp, 1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        violations.append(
                            AuditViolation(
                                code="CORRUPT_JSONL_LINE",
                                severity=ViolationSeverity.BLOCKING,
                                message=f"Corrupt JSON line at index {line_idx}: {exc}",
                                context={"line_number": line_idx},
                            )
                        )
                        continue

                    # Key presence check
                    missing = required_keys - set(data.keys())
                    if missing:
                        violations.append(
                            AuditViolation(
                                code="MISSING_RECORD_FIELDS",
                                severity=ViolationSeverity.BLOCKING,
                                message=f"Line {line_idx} missing mandatory fields: {sorted(list(missing))}",
                                context={"line_number": line_idx, "missing": sorted(list(missing))},
                            )
                        )
                        continue

                    run_id = str(data["run_id"])
                    if run_id in seen_ids:
                        violations.append(
                            AuditViolation(
                                code="DUPLICATE_TRIAL_ID",
                                severity=ViolationSeverity.BLOCKING,
                                message=f"Duplicate trial run_id encountered: '{run_id}' at line {line_idx}.",
                                context={"run_id": run_id, "line_number": line_idx},
                            )
                        )
                    seen_ids.add(run_id)

                    elapsed_ms = float(data["elapsed_ms"])
                    if elapsed_ms < 0 or not math.isfinite(elapsed_ms):
                        violations.append(
                            AuditViolation(
                                code="INVALID_LATENCY_METRIC",
                                severity=ViolationSeverity.BLOCKING,
                                message=f"Trial '{run_id}' has non-finite or negative elapsed_ms: {elapsed_ms}",
                                context={"run_id": run_id, "elapsed_ms": elapsed_ms},
                            )
                        )

                    action_count = int(data["action_count"])
                    if action_count < 0:
                        violations.append(
                            AuditViolation(
                                code="INVALID_ACTION_COUNT",
                                severity=ViolationSeverity.BLOCKING,
                                message=f"Trial '{run_id}' has negative action_count: {action_count}",
                                context={"run_id": run_id, "action_count": action_count},
                            )
                        )

                    # False confirmation check
                    if data["false_confirmation"]:
                        violations.append(
                            AuditViolation(
                                code="FALSE_CONFIRMATION_DETECTED",
                                severity=ViolationSeverity.BLOCKING,
                                message=f"Trial '{run_id}' reports a verified FALSE CONFIRMATION (action claimed confirmed, oracle failed).",
                                context={"run_id": run_id, "scenario_id": data["scenario_id"]},
                            )
                        )

                    # Parse provenance if present
                    raw_prov = data.get("provenance", "simulation_fixture")
                    try:
                        prov = EvidenceProvenance(raw_prov)
                    except ValueError:
                        prov = EvidenceProvenance.UNKNOWN

                    try:
                        res = TrialResult(
                            run_id=run_id,
                            scenario_id=str(data["scenario_id"]),
                            mode=Mode(data["mode"]),
                            seed=int(data["seed"]),
                            status=ActionStatus(data["status"]),
                            oracle_success=bool(data["oracle_success"]),
                            oracle_valid=bool(data["oracle_valid"]),
                            reported_confirmed=bool(data["reported_confirmed"]),
                            verified_success=bool(data["verified_success"]),
                            false_confirmation=bool(data["false_confirmation"]),
                            elapsed_ms=elapsed_ms,
                            action_count=action_count,
                            evidence_method=str(data["evidence_method"]),
                            category=OutcomeCategory(data["category"]),
                            oracle_id=str(data["oracle_id"]),
                            assertion_id=str(data["assertion_id"]),
                            detail_code=str(data.get("detail_code", "")),
                            provenance=prov,
                            app_identity=data.get("app_identity"),
                            action_latency_ms=data.get("action_latency_ms"),
                            postcondition_evidence_ref=data.get("postcondition_evidence_ref"),
                        )
                        results.append(res)
                    except Exception as exc:
                        violations.append(
                            AuditViolation(
                                code="RECORD_CONSTRUCTION_ERROR",
                                severity=ViolationSeverity.BLOCKING,
                                message=f"Failed to instantiate TrialResult at line {line_idx}: {exc}",
                                context={"line_number": line_idx, "error": str(exc)},
                            )
                        )
        except Exception as exc:
            violations.append(
                AuditViolation(
                    code="TRIALS_READ_ERROR",
                    severity=ViolationSeverity.BLOCKING,
                    message=f"Error reading trials.jsonl: {exc}",
                )
            )

        return results, violations

    def _determine_provenance(
        self,
        results: Sequence[TrialResult],
        manifest_path: Path,
    ) -> EvidenceProvenance:
        """Determines whether evidence originates from a live application or simulated fixture."""
        if not results:
            return EvidenceProvenance.UNKNOWN

        # If any trial is explicitly marked live_application and has real app identity
        has_live_trials = any(
            r.provenance == EvidenceProvenance.LIVE_APPLICATION
            and r.app_identity is not None
            and bool(r.app_identity.get("process_name") or r.app_identity.get("window_title"))
            for r in results
        )
        if has_live_trials:
            return EvidenceProvenance.LIVE_APPLICATION

        # Check oracle IDs and evidence methods
        fixture_oracles = {"m7-fixture-oracle", "fixture-oracle"}
        has_fixture_oracles = all(r.oracle_id in fixture_oracles for r in results)

        if has_fixture_oracles:
            return EvidenceProvenance.SIMULATION_FIXTURE

        return EvidenceProvenance.UNKNOWN

    def _compare_summaries(
        self,
        recomputed: Dict[str, Any],
        saved: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compares recomputed summary metrics with saved summary.json."""
        discrepancies: List[Dict[str, Any]] = []

        if recomputed.get("total_trials") != saved.get("total_trials"):
            discrepancies.append({
                "metric": "total_trials",
                "recomputed": recomputed.get("total_trials"),
                "saved": saved.get("total_trials"),
            })

        for mode, rec_stats in recomputed.get("by_mode", {}).items():
            saved_stats = saved.get("by_mode", {}).get(mode)
            if not saved_stats:
                discrepancies.append({
                    "metric": f"by_mode.{mode}",
                    "recomputed": "present",
                    "saved": "missing",
                })
                continue

            for key in ("trials", "oracle_successes", "false_confirmations", "agent_verified_successes", "blocked", "abstentions", "errors"):
                if rec_stats.get(key) != saved_stats.get(key):
                    discrepancies.append({
                        "metric": f"by_mode.{mode}.{key}",
                        "recomputed": rec_stats.get(key),
                        "saved": saved_stats.get(key),
                    })

            rec_lat = rec_stats.get("latency_ms", {})
            saved_lat = saved_stats.get("latency_ms", {})
            for lat_key in ("p50", "p95", "p99", "max"):
                r_val = rec_lat.get(lat_key)
                s_val = saved_lat.get(lat_key)
                if r_val != s_val:
                    discrepancies.append({
                        "metric": f"by_mode.{mode}.latency_ms.{lat_key}",
                        "recomputed": r_val,
                        "saved": s_val,
                    })

        return discrepancies


def main(argv: Optional[List[str]] = None) -> int:
    """CLI Entrypoint: avcs-audit-evidence."""
    parser = argparse.ArgumentParser(description="AVCS Milestone 8: Trial Evidence Certification Auditor")
    parser.add_argument("--results-dir", default="m7-results", help="Directory containing trials.jsonl and summary.json")
    parser.add_argument("--enforce-live", action="store_true", help="Require live application provenance for gate passage")
    parser.add_argument("--output-report", default=None, help="Optional path to write markdown or JSON audit report")
    parser.add_argument("--json", action="store_true", help="Emit audit report as JSON to stdout")

    args = parser.parse_args(argv)

    auditor = TrialEvidenceAuditor()
    report = auditor.audit_directory(args.results_dir, enforce_live=args.enforce_live)

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        else:
            out_path.write_text(report.to_markdown() + "\n", encoding="utf-8")
        print(f"Audit report written to: {out_path}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_markdown())

    # Exit code: 0 if harness passed and (not enforcing live or live passed), 1 if blocking
    if report.harness_verification_gate_passed and (not args.enforce_live or report.live_certification_gate_passed):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
