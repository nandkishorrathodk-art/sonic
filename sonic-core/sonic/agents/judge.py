"""
SONIC-REDA — Judge Agent (Planner-Worker-Judge Swarm)
======================================================
Independent Judge Agent that validates worker outputs, reproduction artifacts,
and claimed findings before the Planner (BossAgent) accepts them.

Mandate & Prohibitions:
    1. Zero Success-by-Decree (Rule 15): No claim is accepted without empirical verification.
    2. Anti-Self-Attestation (Rule 1): The discovering worker agent cannot judge its own work.
    3. Independent Falsification: Actively attempts to falsify or reproduce candidate claims
       in the sandbox environment.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from sonic.computer_use.models_boss import SubMission, SubMissionResult
from sonic.evidence.independent_verifier import AdversarialReviewer, IndependentVerifier
from sonic.evidence.models import (
    ProvenancedFinding,
)
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


class JudgeAgent:
    """
    Dedicated Judge Agent for the Swarm architecture.
    Validates SubMissionResult from workers before acceptance by BossAgent.
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
        """
        Independently evaluate a worker's SubMissionResult.
        Enforces empirical verification and the anti-self-attestation rule.
        """
        traces = result.traces or []

        # 1. Total Failure Detection (All traces failed or exit 126/127/1)
        if not result.success and not any(_t_val(t, "exit_code", 1) == 0 for t in traces):
            all_errors = [
                str(_t_val(t, "actual_observation") or _t_val(t, "observation") or "")
                for t in traces
            ]
            err_summary = "; ".join(e[:100] for e in all_errors if e)
            return JudgeEvaluation(
                verdict=JudgeVerdict.FALSIFIED,
                confidence=0.95,
                notes=f"Worker produced no successful execution traces. Errors: {err_summary[:200]}",
                self_attestation_rejected=False,
            )

        # 2. Extract Concrete Execution Evidence
        worker_self_verified = any(
            str(_t_val(t, "status", "")).upper() in ("VERIFIED", "SUCCESS")
            for t in traces
        ) or bool(result.success)
        has_concrete_proof = False
        valid_commands = []
        for t in traces:
            exit_c = _t_val(t, "exit_code", None)
            obs = str(_t_val(t, "actual_observation") or _t_val(t, "observation") or "").strip()
            target = str(_t_val(t, "target_resource") or _t_val(t, "target") or "")
            action = str(_t_val(t, "action_type") or "")
            verif_ev = str(_t_val(t, "verification_evidence", "")).strip()
            t_status = str(_t_val(t, "status", "")).upper()

            # Exclude Fail-Closed errors, provider crashes, and failed executions from proof
            if any(marker in obs for marker in ("FAIL-CLOSED", "Failed to launch GUI", "Exit 126", "Error 126", "ConnectionRefused")):
                continue
            if t_status == "FAILED" or (exit_c is not None and exit_c != 0):
                continue

            if (exit_c == 0 and obs) or (exit_c is None and obs):
                has_concrete_proof = True
                if "TERMINAL" in action or "EXEC" in action:
                    valid_commands.append(target)

        # Anti-Self-Attestation Check:
        if worker_self_verified and not has_concrete_proof:
            logger.warning(
                "judge_self_attestation_rejected",
                worker=worker_agent_id,
                sub_mission_id=sub_mission.id,
            )
            return JudgeEvaluation(
                verdict=JudgeVerdict.FALSIFIED,
                confidence=1.0,
                notes="Self-attestation rejected: Worker declared status='VERIFIED' without concrete, reproducible sandbox proof.",
                self_attestation_rejected=True,
            )

        # 3. Security Vulnerability Reproduction Check:
        # If candidate discoveries mention CVE, SQLi, XSS, RCE, IDOR, open ports, etc.
        vuln_markers = ("sqli", "sql injection", "xss", "rce", "cve-", "idor", "open port", "flag{", "bypass", "vulnerability")
        combined_findings = (result.findings_summary + " " + " ".join(result.key_discoveries)).lower()
        if any(m in combined_findings for m in vuln_markers):
            from sonic.evidence.models import FindingSeverity
            finding = ProvenancedFinding(
                tenant_id="default",
                engagement_id="swarm-eval",
                target=workspace_id,
                vulnerability_class="Candidate Swarm Finding",
                title=result.findings_summary[:120],
                description=result.findings_summary or "Candidate finding discovered by worker",
                severity=FindingSeverity.MEDIUM,
                poc="; ".join(valid_commands[:3]) if valid_commands else "",
                created_by_agent=worker_agent_id,
            )

            # If there's an executable PoC command, test re-execution via computer provider
            if valid_commands and self.computer and hasattr(self.computer, "terminal"):
                try:
                    reprod_cmd = valid_commands[-1]
                    re_exec = await self.computer.terminal(workspace_id, reprod_cmd)
                    re_exit = getattr(re_exec, "exit_code", 1)
                    re_obs = str(getattr(re_exec, "stdout", "") or getattr(re_exec, "stderr", ""))
                    if re_exit == 0:
                        v_res = self.independent_verifier.process_verification_result(
                            finding=finding,
                            verifier_agent_id=self.judge_id,
                            is_reproduced=True,
                            notes=f"Re-execution of PoC '{reprod_cmd[:60]}' confirmed exit 0.",
                        )
                        return JudgeEvaluation(
                            verdict=JudgeVerdict.CONFIRMED,
                            confidence=0.95,
                            notes=v_res.notes,
                            reproduction_exit_code=0,
                            reproduced_output=re_obs[:300],
                            evidence_items=[reprod_cmd],
                        )
                    else:
                        v_res = self.independent_verifier.process_verification_result(
                            finding=finding,
                            verifier_agent_id=self.judge_id,
                            is_reproduced=False,
                            notes=f"Independent re-execution of PoC failed (exit {re_exit}).",
                        )
                        return JudgeEvaluation(
                            verdict=JudgeVerdict.FALSIFIED,
                            confidence=0.85,
                            notes=v_res.notes,
                            reproduction_exit_code=re_exit,
                            reproduced_output=re_obs[:300],
                        )
                except Exception as ex:
                    logger.warning("judge_poc_reproduction_error", error=str(ex))

        # 4. Unit / Integration Test Suite Check:
        # If the goal explicitly specifies running unit tests (e.g. pytest, unit test)
        goal_lower = sub_mission.goal.lower()
        if any(w in goal_lower for w in ("pytest", "unit test", "run tests", "test suite")):
            test_success = False
            for t in traces:
                obs = str(_t_val(t, "actual_observation") or _t_val(t, "observation") or "").lower()
                exit_c = _t_val(t, "exit_code", None)
                if exit_c == 0 and bool(re.search(r"\b(passed|ok|success|\d+\s+passed)\b", obs)):
                    if not any(f in obs for f in ("failed", "error", "traceback", "assertionerror")):
                        test_success = True
                        break
                elif any(f in obs for f in ("failed", "error", "traceback", "assertionerror")):
                    test_success = False

            if not test_success:
                return JudgeEvaluation(
                    verdict=JudgeVerdict.FALSIFIED,
                    confidence=0.9,
                    notes="Testing objective unverified: No passing test output (passed/ok) detected in trace observations.",
                )

        # 5. File Creation / Modification Check:
        if any(w in goal_lower for w in ("create file", "create folder", "write file", "author", "evidence file")):
            file_names = re.findall(r'[\w\-./]+\.(?:py|md|txt|json|sh|yaml|yml)', sub_mission.goal)
            if file_names and self.computer and hasattr(self.computer, "read_file"):
                for fname in file_names:
                    try:
                        content = await self.computer.read_file(workspace_id, fname)
                        if not content:
                            return JudgeEvaluation(
                                verdict=JudgeVerdict.FALSIFIED,
                                confidence=0.9,
                                notes=f"Claimed file '{fname}' was not found or is empty in workspace (Rule 15 violation).",
                            )
                    except Exception:
                        return JudgeEvaluation(
                            verdict=JudgeVerdict.FALSIFIED,
                            confidence=0.9,
                            notes=f"Error reading claimed file '{fname}' from workspace.",
                        )

        # 6. Workstation Application & Desktop Window Check (Rule 5 & Rule 7):
        is_gui_mission = getattr(sub_mission, "plane", "") == "gui" or any(
            w in goal_lower for w in ("desktop", "browser", "window", "gui", "open application", "launch application")
        )
        if is_gui_mission:
            has_gui_action = any(
                str(_t_val(t, "action_type", "")).startswith("GUI_")
                or str(_t_val(t, "action_type", "")) == "APP_LAUNCH"
                for t in traces
            )
            gui_verified = has_gui_action
            if self.computer and hasattr(self.computer, "terminal"):
                try:
                    wm_res = await self.computer.terminal(workspace_id, "DISPLAY=:99 wmctrl -l 2>/dev/null | grep -v ' -1 '")
                    open_wins = [line for line in (getattr(wm_res, "stdout", "") or "").splitlines() if line.strip()]
                    if open_wins:
                        gui_verified = True
                except Exception:
                    pass

            if not gui_verified:
                return JudgeEvaluation(
                    verdict=JudgeVerdict.FALSIFIED,
                    confidence=0.85,
                    notes="GUI desktop interaction unverified: No application window or visual action was detected on Display :99.",
                )

        # 7. Default Grounded Verdict
        if has_concrete_proof and result.success:
            return JudgeEvaluation(
                verdict=JudgeVerdict.CONFIRMED,
                confidence=0.85,
                notes="Empirical execution verified: Non-trivial command executed with exit 0 and observations.",
                evidence_items=valid_commands,
            )

        return JudgeEvaluation(
            verdict=JudgeVerdict.INCONCLUSIVE,
            confidence=0.5,
            notes="Trace execution lacked sufficient independent reproduction evidence.",
        )
