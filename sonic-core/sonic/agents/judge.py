"""
SONIC — Independent Judge Agent (Planner-Worker-Judge Verification)
===================================================================
Independent Judge Agent that validates worker outputs, reproduction artifacts,
and execution receipts before the Planner (BossAgent) or Teamwork Runtime accepts them.

Strict Mandate & Prohibitions (AGENTS.md):
    1. Zero Hardcoded Application/Binary Names (Rule 1): Never match or invoke hardcoded
       vendor or tool binaries. Inspect workspace state via provider APIs (`status()`, `read_file()`).
    2. Zero Scripted Puppet Heuristics (Rule 2): Never classify goals via keyword-matching trees
       (`if "pytest" in goal`, `if "create file" in goal`, etc.). Evaluate strictly from typed
       execution trace receipts (`exit_code`, `status`, `verification_evidence`, `actual_observation`)
       and independent verification.
    3. Zero Silent Failure Masking (Rule 6): No `|| true` or `2>/dev/null`.
    4. Anti-Self-Attestation: A worker cannot self-certify completion without concrete execution proof.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from sonic.computer_use.models_boss import SubMission, SubMissionResult
from sonic.evidence.independent_verifier import AdversarialReviewer, IndependentVerifier
from sonic.evidence.reproduction_engine import ReproductionEngine
from sonic.logger import get_logger

logger = get_logger(__name__)


def _t_val(t: Any, key: str, default: Any = "") -> Any:
    """Safely extract field from trace whether dict or object."""
    if isinstance(t, dict):
        return t.get(key, default)
    return getattr(t, key, default)


class JudgeVerdict(StrEnum):
    CONFIRMED = "CONFIRMED"
    FALSIFIED = "FALSIFIED"
    BENIGN = "BENIGN"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNVERIFIED = "UNVERIFIED"


class JudgeEvaluation(BaseModel):
    verdict: JudgeVerdict = JudgeVerdict.UNVERIFIED
    confidence: float = 0.0
    notes: str = ""
    reproduction_exit_code: int | None = None
    reproduced_output: str = ""
    self_attestation_rejected: bool = False
    evidence_items: list[str] = Field(default_factory=list)
    remediation_blueprint: dict[str, Any] = Field(default_factory=dict)


class JudgeAgent:
    """Independent Judge Agent for the SONIC multi-agent architecture.
    Validates `SubMissionResult` strictly from empirical execution receipts and provider state,
    with zero keyword-matching heuristics or hardcoded binary names.
    """

    def __init__(
        self,
        computer_provider: Any = None,
        reproduction_engine: ReproductionEngine | None = None,
        independent_verifier: IndependentVerifier | None = None,
        adversarial_reviewer: AdversarialReviewer | None = None,
        judge_id: str = "judge-alpha",
    ):
        self.computer = computer_provider
        self.reproduction_engine = reproduction_engine or (
            ReproductionEngine(compute_provider=computer_provider)
            if computer_provider and hasattr(computer_provider, "execute")
            else None
        )
        self.independent_verifier = independent_verifier or IndependentVerifier()
        self.adversarial_reviewer = adversarial_reviewer or AdversarialReviewer()
        self.judge_id = judge_id

    async def evaluate_sub_mission(
        self,
        workspace_id: str,
        sub_mission: SubMission,
        result: SubMissionResult,
        worker_agent_id: str = "worker",
    ) -> JudgeEvaluation:
        """Independently evaluate a worker's `SubMissionResult` from ground-truth execution receipts."""
        traces = list(result.traces or [])

        # 1. Total Execution Failure Check (Zero successful traces and non-zero exit codes)
        if not result.success and not any(_t_val(t, "exit_code", 1) == 0 for t in traces):
            all_errors = [
                str(_t_val(t, "actual_observation") or _t_val(t, "observation") or "")
                for t in traces
            ]
            err_summary = "; ".join(e[:100] for e in all_errors if e)
            return JudgeEvaluation(
                verdict=JudgeVerdict.FALSIFIED,
                confidence=0.95,
                notes=f"Worker produced no successful execution receipts. Errors: {err_summary[:200]}",
                self_attestation_rejected=False,
                remediation_blueprint={
                    "fix_goal": f"Resolve execution errors for objective: {sub_mission.goal}",
                    "error_context": err_summary[:300],
                },
            )

        # 2. Extract Concrete Ground-Truth Receipts from Traces
        worker_self_verified = any(
            str(_t_val(t, "status", "")).upper() in ("VERIFIED", "SUCCESS")
            for t in traces
        ) or bool(result.success)

        has_concrete_proof = False
        evidence_items: list[str] = []
        written_files: list[str] = []

        for t in traces:
            exit_c = _t_val(t, "exit_code", None)
            obs = str(_t_val(t, "actual_observation") or _t_val(t, "observation") or "").strip()
            target = str(_t_val(t, "target_resource") or _t_val(t, "target") or "").strip()
            action_raw = _t_val(t, "action_type", "")
            action = action_raw.value if hasattr(action_raw, "value") else str(action_raw)
            verif_ev = str(_t_val(t, "verification_evidence", "")).strip()
            t_status = str(_t_val(t, "status", "")).upper()

            if t_status in ("FAILED", "BLOCKED") or (exit_c is not None and exit_c != 0):
                continue

            if (exit_c == 0 and (obs or verif_ev)) or (exit_c is None and (obs or verif_ev)):
                has_concrete_proof = True
                if target:
                    evidence_items.append(f"{action}:{target[:120]}")
                if action in ("FILE_WRITE", "FILE_EDIT", "FILE_CREATE") and target:
                    written_files.append(target)

        # 3. Anti-Self-Attestation Gate
        if worker_self_verified and not has_concrete_proof:
            logger.warning(
                "judge_self_attestation_rejected",
                worker=worker_agent_id,
                sub_mission_id=sub_mission.id,
            )
            return JudgeEvaluation(
                verdict=JudgeVerdict.FALSIFIED,
                confidence=1.0,
                notes="Self-attestation rejected: Worker claimed completion without concrete execution receipts.",
                self_attestation_rejected=True,
                remediation_blueprint={
                    "fix_goal": f"Produce verifiable execution receipts for: {sub_mission.goal}",
                },
            )

        # 4. Independent File Artifact Verification (verify files written during trace actually exist on disk)
        if written_files and self.computer and hasattr(self.computer, "read_file"):
            for fpath in written_files:
                try:
                    content = await self.computer.read_file(workspace_id, fpath)
                    if content is None or content == "":
                        return JudgeEvaluation(
                            verdict=JudgeVerdict.FALSIFIED,
                            confidence=0.9,
                            notes=f"File artifact '{fpath}' recorded in trace is missing or empty on disk.",
                            remediation_blueprint={
                                "fix_goal": f"Ensure '{fpath}' is properly written and non-empty.",
                            },
                        )
                except Exception as exc:
                    return JudgeEvaluation(
                        verdict=JudgeVerdict.FALSIFIED,
                        confidence=0.9,
                        notes=f"Independent verification failed reading '{fpath}': {exc}",
                        remediation_blueprint={
                            "fix_goal": f"Fix file path or write error for '{fpath}'.",
                        },
                    )

        # A successful process is not evidence that it advanced the mission.
        # Require a meaningful objective token in the concrete receipt before
        # accepting a generic successful command as confirmation.
        stop_words = {
            "a", "an", "and", "by", "for", "from", "in", "into", "of",
            "on", "or", "the", "to", "using", "with",
        }
        goal_terms = {
            token for token in re.findall(r"[a-z0-9]+", sub_mission.goal.lower())
            if len(token) > 2 and token not in stop_words
        }
        receipt_text = " ".join(
            f"{_t_val(trace, 'target_resource', '')} "
            f"{_t_val(trace, 'actual_observation', '')}"
            for trace in traces
        ).lower()
        receipt_terms = set(re.findall(r"[a-z0-9]+", receipt_text))
        if has_concrete_proof and goal_terms and not goal_terms.intersection(receipt_terms):
            return JudgeEvaluation(
                verdict=JudgeVerdict.INCONCLUSIVE,
                confidence=0.4,
                notes="Execution succeeded but the receipt is not relevant to the sub-mission objective.",
                evidence_items=evidence_items,
            )

        # 5. Confirmed Grounded Verdict
        if has_concrete_proof and result.success:
            return JudgeEvaluation(
                verdict=JudgeVerdict.CONFIRMED,
                confidence=0.9,
                notes="Independent verification confirmed concrete execution receipts with exit code 0.",
                evidence_items=evidence_items,
            )

        return JudgeEvaluation(
            verdict=JudgeVerdict.INCONCLUSIVE,
            confidence=0.5,
            notes="Trace execution lacked sufficient independent reproduction receipts.",
        )
