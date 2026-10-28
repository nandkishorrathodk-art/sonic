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

import base64
import re
import time
from collections.abc import Mapping
from enum import StrEnum
from io import BytesIO
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from sonic.computer_use.models_boss import SubMission, SubMissionResult
from sonic.evidence.independent_verifier import AdversarialReviewer, IndependentVerifier
from sonic.evidence.reproduction_engine import ReproductionEngine
from sonic.logger import get_logger

logger = get_logger(__name__)
_GUI_JUDGE_MAX_FRAME_AGE_MS = 2_000
_SUCCESSFUL_GUI_RECEIPT_STATUSES = {
    "COMPLETED",
    "SUCCESS",
    "SUCCEEDED",
    "RECOVERED",
    "VERIFIED",
}


def _is_current_avcs_state(state: Any, screen: Any) -> bool:
    """Accept only a well-formed AVCS state attached to this fresh frame."""
    if not isinstance(state, Mapping) or state.get("engine") != "avcs_reflex":
        return False

    frame_index = state.get("frame_index")
    scene_hash = state.get("scene_hash")
    if (
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or frame_index <= 0
        or not isinstance(scene_hash, str)
        or not scene_hash.strip()
    ):
        return False

    width = state.get("width")
    height = state.get("height")
    screen_width = getattr(screen, "width", None)
    screen_height = getattr(screen, "height", None)
    return (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
        and width == screen_width
        and height == screen_height
    )


def _t_val(t: Any, key: str, default: Any = "") -> Any:
    """Safely extract field from trace whether dict or object."""
    if isinstance(t, dict):
        return t.get(key, default)
    return getattr(t, key, default)


# Trace statuses that mean the action actually did its job. GUI actions reach
# one of these without ever producing a subprocess exit code.
_SUCCESS_TRACE_STATUSES = frozenset(
    {"SUCCESS", "SUCCEEDED", "COMPLETED", "VERIFIED", "RECOVERED"}
)


def _trace_succeeded(t: Any) -> bool:
    """Whether one trace is real evidence that its action succeeded.

    Terminal/tool actions prove success with exit code 0. GUI actions are not
    subprocesses and legitimately have ``exit_code is None``, so they prove
    success through their status instead. Requiring an exit code would falsify
    every GUI worker that only launched or observed an application.
    """
    exit_c = _t_val(t, "exit_code", None)
    if isinstance(exit_c, int) and not isinstance(exit_c, bool) and exit_c == 0:
        return True
    status = str(_t_val(t, "status", "")).upper()
    return status in _SUCCESS_TRACE_STATUSES


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
        llm_router: Any = None,
    ):
        self.computer = computer_provider
        self.llm_router = llm_router
        self.reproduction_engine = reproduction_engine or (
            ReproductionEngine(compute_provider=computer_provider)
            if computer_provider and hasattr(computer_provider, "execute")
            else None
        )
        self.independent_verifier = independent_verifier or IndependentVerifier()
        self.adversarial_reviewer = adversarial_reviewer or AdversarialReviewer()
        self.judge_id = judge_id

    async def _evaluate_gui_result(
        self,
        workspace_id: str,
        sub_mission: SubMission,
        result: SubMissionResult,
        traces: list[Any],
    ) -> JudgeEvaluation:
        """Inspect GUI evidence locally and refuse unproven semantic completion."""
        if self.computer is None:
            return JudgeEvaluation(
                verdict=JudgeVerdict.INCONCLUSIVE,
                confidence=0.0,
                notes="GUI outcome cannot be checked: desktop provider is unavailable.",
                self_attestation_rejected=True,
            )

        gui_actions = []
        for trace in traces:
            action = _t_val(trace, "action_type", "")
            action = action.value if hasattr(action, "value") else str(action)
            if (
                action.startswith("GUI_")
                or action.startswith("BROWSER_")
                or action in ("APP_LAUNCH", "APP_FOCUS", "APP_CLOSE")
            ):
                receipt_status = _t_val(trace, "status", "")
                receipt_status = (
                    receipt_status.value
                    if hasattr(receipt_status, "value")
                    else str(receipt_status)
                ).upper()
                if receipt_status not in _SUCCESSFUL_GUI_RECEIPT_STATUSES:
                    continue
                gui_actions.append(
                    {
                        "action": action,
                        "target": str(_t_val(trace, "target_resource", ""))[:240],
                        "status": receipt_status[:40],
                        "observation": str(
                            _t_val(trace, "actual_observation", "")
                            or _t_val(trace, "observation", "")
                        )[:500],
                    }
                )
        if not gui_actions:
            return JudgeEvaluation(
                verdict=JudgeVerdict.INCONCLUSIVE,
                confidence=0.0,
                notes=(
                    "GUI outcome cannot be confirmed: worker supplied no successful "
                    "GUI interaction receipts."
                ),
                self_attestation_rejected=True,
            )

        avcs_evidence: str | None = None
        try:
            screen = await self.computer.screenshot(workspace_id)
            screen_state = str(getattr(screen, "desktop_state", "") or "UNKNOWN")
            if screen_state.upper() == "NO_DISPLAY":
                return JudgeEvaluation(
                    verdict=JudgeVerdict.INCONCLUSIVE,
                    confidence=0.0,
                    notes="GUI outcome cannot be confirmed: workstation reports no active display.",
                    self_attestation_rejected=True,
                )
            capture_age_ms = getattr(screen, "capture_age_ms", None)
            if (
                not isinstance(capture_age_ms, (int, float))
                or isinstance(capture_age_ms, bool)
            ):
                capture_timestamp_ns = getattr(screen, "capture_timestamp_ns", None)
                if (
                    isinstance(capture_timestamp_ns, int)
                    and not isinstance(capture_timestamp_ns, bool)
                    and capture_timestamp_ns > 0
                ):
                    age_ns = time.time_ns() - capture_timestamp_ns
                    capture_age_ms = age_ns / 1_000_000 if age_ns >= 0 else None
                else:
                    capture_age_ms = None
            if (
                capture_age_ms is None
                or capture_age_ms < 0
                or capture_age_ms > _GUI_JUDGE_MAX_FRAME_AGE_MS
            ):
                return JudgeEvaluation(
                    verdict=JudgeVerdict.INCONCLUSIVE,
                    confidence=0.0,
                    notes=(
                        "GUI outcome cannot be confirmed: screenshot freshness "
                        "is missing or outside the allowed age."
                    ),
                    self_attestation_rejected=True,
                )

            from sonic.computer_use.avcs_perception import AVCSPerception

            perception = AVCSPerception()
            raw_rgb = bytes(getattr(screen, "raw_rgb", b"") or b"")
            if raw_rgb:
                width = getattr(screen, "width", None)
                height = getattr(screen, "height", None)
                frame_id = getattr(screen, "frame_id", None)
                if (
                    not isinstance(frame_id, int)
                    or isinstance(frame_id, bool)
                    or frame_id <= 0
                ):
                    raise ValueError("Raw Judge frame is missing its source frame ID")
                avcs_state = perception.perceive_rgb_bytes(
                    raw_rgb, width, height, frame_id
                )
                if avcs_state.get("source_frame_id") != frame_id:
                    raise ValueError("AVCS Judge evidence does not match the raw source frame")
            else:
                image = (
                    getattr(screen, "screenshot_base64", "")
                    or getattr(screen, "image_base64", "")
                )
                if not image:
                    return JudgeEvaluation(
                        verdict=JudgeVerdict.INCONCLUSIVE,
                        confidence=0.0,
                        notes="GUI outcome cannot be confirmed: fresh desktop pixels are unavailable.",
                        self_attestation_rejected=True,
                    )
                encoded_image = image.split(",", 1)[-1]
                image_bytes = base64.b64decode(encoded_image, validate=True)
                with Image.open(BytesIO(image_bytes)) as frame:
                    frame.load()
                    image_format = frame.format
                    image_dimensions = frame.size
                    if image_format not in {"PNG", "JPEG"}:
                        raise ValueError("Judge screenshot must be PNG or JPEG")
                    if image_dimensions != (
                        getattr(screen, "width", None),
                        getattr(screen, "height", None),
                    ):
                        raise ValueError("Judge screenshot dimensions do not match frame metadata")
                    if image_format == "PNG":
                        avcs_image_bytes = image_bytes
                    else:
                        avcs_image = BytesIO()
                        frame.convert("RGB").save(avcs_image, format="PNG")
                        avcs_image_bytes = avcs_image.getvalue()
                avcs_state = perception.perceive_png(avcs_image_bytes)
            if not _is_current_avcs_state(avcs_state, screen):
                raise ValueError("AVCS did not return valid evidence for the current frame")

            scene_hash = str(avcs_state["scene_hash"])
            avcs_evidence = (
                f"AVCS frame {avcs_state['frame_index']}: scene {scene_hash[:16]}, "
                f"changed={bool(avcs_state.get('scene_changed', False))}, "
                f"regions={len(avcs_state.get('regions', []))}, "
                f"events={len(avcs_state.get('events', []))}."
            )
            logger.info(
                "judge_gui_avcs_observation_recorded",
                workspace_id=workspace_id,
                sub_mission_id=sub_mission.id,
                frame_index=avcs_state.get("frame_index"),
                scene_changed=avcs_state.get("scene_changed"),
                scene_hash=scene_hash,
            )

            return JudgeEvaluation(
                verdict=JudgeVerdict.INCONCLUSIVE,
                confidence=0.0,
                notes=(
                    "Fresh desktop pixels were processed locally by AVCS, but its "
                    "reflex state cannot establish this task's semantic postcondition. "
                    "GUI completion remains unconfirmed until a validated local "
                    "task-verification head exists; a worker's own claim is not "
                    "independent proof."
                ),
                self_attestation_rejected=True,
                evidence_items=[avcs_evidence],
            )
        except Exception as exc:
            logger.warning(
                "judge_gui_outcome_evaluation_failed",
                workspace_id=workspace_id,
                sub_mission_id=sub_mission.id,
                error=str(exc),
            )
            return JudgeEvaluation(
                verdict=JudgeVerdict.INCONCLUSIVE,
                confidence=0.0,
                notes=f"GUI outcome verification failed; no completion claim is allowed ({exc}).",
                self_attestation_rejected=True,
                evidence_items=[avcs_evidence] if avcs_evidence is not None else [],
            )

    async def evaluate_sub_mission(
        self,
        workspace_id: str,
        sub_mission: SubMission,
        result: SubMissionResult,
        worker_agent_id: str = "worker",
    ) -> JudgeEvaluation:
        """Independently evaluate a worker's `SubMissionResult` from ground-truth execution receipts."""
        traces = list(result.traces or [])
        is_gui_mission = str(getattr(sub_mission, "plane", "")).lower() == "gui"
        if not is_gui_mission:
            for trace in traces:
                action_raw = _t_val(trace, "action_type", "")
                action = action_raw.value if hasattr(action_raw, "value") else str(action_raw)
                if action.startswith(("GUI_", "APP_", "WINDOW_", "BROWSER_")):
                    is_gui_mission = True
                    break

        # 1. Total Execution Failure Check (no successful trace of any kind).
        #
        # A trace succeeds either by returning a zero exit code (terminal/tool
        # actions) or by reaching a success status. GUI actions (APP_LAUNCH,
        # APP_FOCUS, GUI_SCREENSHOT, ...) legitimately carry exit_code=None --
        # they are not subprocesses -- so keying this gate on `exit_code == 0`
        # alone falsified genuinely successful GUI workers whose only evidence
        # was a completed GUI action.
        if not result.success and not any(_trace_succeeded(t) for t in traces):
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
        failed_traces: list[str] = []
        syntax_errors: list[str] = []
        stop_words = {
            "a", "an", "and", "by", "for", "from", "in", "into", "of",
            "on", "or", "the", "to", "using", "with", "run", "check",
        }
        goal_terms = {
            token for token in re.findall(r"[a-z0-9]+", sub_mission.goal.lower())
            if len(token) > 2 and token not in stop_words
        }

        # Check for unsaved editor buffers or window titles with unsaved asterisks (*untitled)
        unsaved_match = None
        for t in traces:
            t_obs = str(_t_val(t, "actual_observation") or _t_val(t, "observation") or "")
            t_ev = str(_t_val(t, "verification_evidence", ""))
            t_text = f"{t_obs} {t_ev}"
            if re.search(r'\*untitled|\buntitled\*|\bunsaved\b|\[modified\]', t_text, re.IGNORECASE):
                unsaved_match = f"Trace observation indicated unsaved editor buffer: {t_text[:120]}"
                break

        if not unsaved_match and self.computer:
            try:
                if hasattr(self.computer, "status"):
                    st = await self.computer.status(workspace_id)
                    wins = list(getattr(st, "windows", []))
                    act = str(getattr(st, "active_application", ""))
                    if act:
                        wins.append(act)
                    for w in wins:
                        if re.search(r'\*untitled|\buntitled\*|\bunsaved\b|\[modified\]|^\s*[*•]\s*[\w\.\-]+|[*•]\s*[\w\.\-]+\s*[-–—]', str(w), re.IGNORECASE):
                            unsaved_match = f"Open editor window contains unsaved marker: '{w}'"
                            break
            except Exception:
                pass

        if unsaved_match:
            return JudgeEvaluation(
                verdict=JudgeVerdict.FALSIFIED,
                confidence=0.95,
                notes=f"Rejection: File or editor window is unsaved ({unsaved_match}). Changes must be saved to disk.",
                self_attestation_rejected=False,
                remediation_blueprint={
                    "fix_goal": f"Save unsaved editor buffers to disk for objective: {sub_mission.goal}",
                    "error_context": unsaved_match,
                },
            )

        for t in traces:
            exit_c = _t_val(t, "exit_code", None)
            obs = str(_t_val(t, "actual_observation") or _t_val(t, "observation") or "").strip()
            target = str(_t_val(t, "target_resource") or _t_val(t, "target") or "").strip()
            action_raw = _t_val(t, "action_type", "")
            action = action_raw.value if hasattr(action_raw, "value") else str(action_raw)
            verif_ev = str(_t_val(t, "verification_evidence", "")).strip()
            t_status = str(_t_val(t, "status", "")).upper()

            # Check for terminal syntax errors
            combined_t = f"{obs}\n{verif_ev}".lower()
            is_syntax_err = any(
                se in combined_t
                for se in (
                    "syntax error", "syntaxerror", "indentationerror", "taberror",
                    "invalid syntax", "parse error", "unexpected token", "unexpected eof",
                )
            )
            if is_syntax_err:
                syntax_errors.append(f"{action}:{target}: {obs[:120]}")

            # Failed traces and terminal syntax errors must NOT be silently skipped with continue
            is_failed = t_status in ("FAILED", "BLOCKED") or (exit_c is not None and exit_c != 0) or is_syntax_err
            if is_failed:
                failed_traces.append(f"{action}:{target}: status={t_status}, exit={exit_c}, err={obs[:100]}")
                continue

            # Concrete proof evaluation:
            # GUI actions with exit_code is None must NOT be treated as has_concrete_proof = True
            # without verifying the target application state or artifacts.
            is_gui_action = (
                action.startswith(("GUI_", "APP_", "WINDOW_", "BROWSER_"))
            )

            if is_gui_action:
                # GUI actions with exit_code is None must NOT be treated as has_concrete_proof = True
                # without verifying the target application state or artifacts.
                app_state_verified = False
                if obs and any(
                    term in obs.lower()
                    for term in (
                        "[app_state_verified]",
                        "launched and focused",
                        "window state verified",
                        "application state confirmed",
                    )
                ):
                    app_state_verified = True
                elif self.computer and hasattr(self.computer, "status"):
                    try:
                        comp_status = await self.computer.status(workspace_id)
                        active_app = str(getattr(comp_status, "active_application", "")).lower()
                        open_apps = [str(a).lower() for a in getattr(comp_status, "open_applications", [])]
                        if target and (target.lower() in active_app or any(target.lower() in app for app in open_apps)):
                            app_state_verified = True
                    except Exception:
                        pass

                if app_state_verified:
                    has_concrete_proof = True
                    if target:
                        evidence_items.append(f"{action}:{target[:120]}")
                    if action in ("FILE_WRITE", "FILE_EDIT", "FILE_CREATE") and target:
                        written_files.append(target)
            else:
                if (
                    exit_c == 0
                    and obs.strip()
                    and (
                        verif_ev.strip()
                        or (
                            action == "TERMINAL_EXEC"
                            and any(
                                term in f"{target} {obs} {verif_ev}".lower()
                                for term in goal_terms
                            )
                        )
                    )
                ):
                    has_concrete_proof = True
                    if target:
                        evidence_items.append(f"{action}:{target[:120]}")
                    if action in ("FILE_WRITE", "FILE_EDIT", "FILE_CREATE") and target:
                        written_files.append(target)

        successful_receipt_text = " ".join(
            f"{_t_val(trace, 'target_resource', '')} "
            f"{_t_val(trace, 'target', '')} "
            f"{_t_val(trace, 'actual_observation', '')} "
            f"{_t_val(trace, 'verification_evidence', '')}"
            for trace in traces
            if (
                str(_t_val(trace, "status", "")).upper()
                not in ("FAILED", "BLOCKED")
                and (
                    _t_val(trace, "exit_code", None) is None
                    or _t_val(trace, "exit_code", None) == 0
                )
            )
        ).lower()
        successful_receipt_terms = set(re.findall(r"[a-z0-9]+", successful_receipt_text))
        has_relevant_successful_receipt = (
            bool(goal_terms.intersection(successful_receipt_terms))
            or any(term in successful_receipt_text for term in goal_terms)
        )

        # A later successful, relevant command can recover from an earlier failed attempt.
        res_text = f"{result.findings_summary} {' '.join(result.key_discoveries)}".lower()
        if any(se in res_text for se in ("syntax error", "syntaxerror", "indentationerror", "taberror", "invalid syntax")):
            syntax_errors.append(f"Result summary reported syntax error: {result.findings_summary[:120]}")

        if syntax_errors and not (
            not is_gui_mission
            and has_concrete_proof
            and has_relevant_successful_receipt
        ):
            err_details = "; ".join(syntax_errors[:3])
            return JudgeEvaluation(
                verdict=JudgeVerdict.FALSIFIED,
                confidence=0.95,
                notes=f"Worker trace contained terminal syntax errors: {err_details}",
                self_attestation_rejected=False,
                remediation_blueprint={
                    "fix_goal": f"Fix terminal syntax errors for objective: {sub_mission.goal}",
                    "error_context": err_details,
                },
            )

        # 3.5 Security Vulnerability Reproduction Check:
        vuln_markers = ("sqli", "sql injection", "xss", "rce", "cve-", "idor", "flag{", "vulnerability")
        combined_findings = (result.findings_summary + " " + " ".join(result.key_discoveries)).lower()
        if any(m in combined_findings for m in vuln_markers):
            return JudgeEvaluation(
                verdict=JudgeVerdict.INCONCLUSIVE,
                confidence=0.0,
                notes=(
                    "A worker-supplied command is not an independent reproduction plan. "
                    "Vulnerability claims remain inconclusive until a scoped isolated "
                    "reproduction produces its own evidence."
                ),
                self_attestation_rejected=True,
                evidence_items=evidence_items,
            )

        # 4. Independent File Artifact Verification
        # For file-writing/editor goals, verify the file actually exists and is non-empty on disk before confirming.
        goal_lower = sub_mission.goal.lower()
        is_file_goal = (
            any(
                w in goal_lower
                for w in (
                    "create file", "create folder", "write file", "edit file",
                    "save file", "author", "evidence file", "generate report",
                    "touch", "file", "write", "editor", "script",
                )
            )
            or bool(written_files)
            or any(
                str(_t_val(t, "action_type", "")).upper() in ("FILE_WRITE", "FILE_EDIT", "FILE_CREATE")
                for t in traces
            )
        )
        candidate_files = list(written_files)
        candidate_files.extend(getattr(sub_mission, "owned_paths", []))
        file_matches_in_goal = re.findall(
            r'[\w\-./\\]+\.(?:py|md|txt|json|sh|yaml|yml|csv|html|toml|rs|js|ts)',
            sub_mission.goal,
            re.IGNORECASE,
        )
        candidate_files.extend(file_matches_in_goal)

        for t in traces:
            t_cmd = str(_t_val(t, "target_resource") or _t_val(t, "target") or "")
            t_matches = re.findall(
                r'(?:touch|cat\s*>\s*|echo\s*.*?>\s*|nano\s+|vim\s+|tee\s+)?([a-zA-Z0-9_\-./\\]+\.(?:py|md|txt|json|sh|yaml|yml|csv|html|toml|rs|js|ts))',
                t_cmd,
                re.IGNORECASE,
            )
            candidate_files.extend(t_matches)

        candidate_files = list(dict.fromkeys(f.strip() for f in candidate_files if f and f.strip()))

        if is_file_goal and candidate_files and self.computer and hasattr(self.computer, "read_file"):
            for fpath in candidate_files:
                try:
                    content = await self.computer.read_file(workspace_id, fpath)
                    if not content or content.strip() == "":
                        return JudgeEvaluation(
                            verdict=JudgeVerdict.FALSIFIED,
                            confidence=0.9,
                            notes=f"Claimed file '{fpath}' was not found or is empty in workspace (Rule 15 violation).",
                            self_attestation_rejected=False,
                            remediation_blueprint={
                                "fix_goal": f"Ensure '{fpath}' is properly written and non-empty.",
                            },
                        )
                    has_concrete_proof = True
                    evidence_items.append(f"Independently read non-empty file: {fpath[:120]}")
                except Exception as exc:
                    return JudgeEvaluation(
                        verdict=JudgeVerdict.FALSIFIED,
                        confidence=0.9,
                        notes=f"Error reading claimed file '{fpath}' from workspace: {exc}",
                        self_attestation_rejected=False,
                        remediation_blueprint={
                            "fix_goal": f"Fix file path or write error for '{fpath}'.",
                        },
                    )

        # 4.5 Anti-Self-Attestation Gate, after independent artifact checks.
        if worker_self_verified and not has_concrete_proof and not is_gui_mission:
            has_failed_execution = any(
                str(_t_val(trace, "status", "")).upper() in ("FAILED", "BLOCKED")
                or (
                    _t_val(trace, "exit_code", None) is not None
                    and _t_val(trace, "exit_code", None) != 0
                )
                for trace in traces
            )
            has_successful_execution = any(
                str(_t_val(trace, "status", "")).upper()
                not in ("FAILED", "BLOCKED")
                and (
                    _t_val(trace, "exit_code", None) is None
                    or _t_val(trace, "exit_code", None) == 0
                )
                for trace in traces
            )
            if has_failed_execution and has_successful_execution:
                return JudgeEvaluation(
                    verdict=JudgeVerdict.INCONCLUSIVE,
                    confidence=0.85,
                    notes=(
                        "Execution succeeded, but the successful receipt is not relevant "
                        "to the objective; failed attempts cannot establish relevance."
                    ),
                    self_attestation_rejected=True,
                )
            logger.warning(
                "judge_self_attestation_rejected",
                worker=worker_agent_id,
                sub_mission_id=sub_mission.id,
            )
            return JudgeEvaluation(
                verdict=(
                    JudgeVerdict.FALSIFIED
                    if has_failed_execution
                    else JudgeVerdict.INCONCLUSIVE
                ),
                confidence=0.9 if has_failed_execution else 0.4,
                notes=(
                    "Self-attestation rejected: execution evidence contradicts the "
                    "completion claim."
                    if has_failed_execution
                    else "Completion is unverified: successful execution lacks independent task evidence."
                ),
                self_attestation_rejected=True,
                remediation_blueprint={
                    "fix_goal": f"Produce verifiable execution receipts for: {sub_mission.goal}",
                },
            )

        gui_evaluation: JudgeEvaluation | None = None
        if is_gui_mission:
            gui_evaluation = await self._evaluate_gui_result(
                workspace_id,
                sub_mission,
                result,
                traces,
            )
            if gui_evaluation.verdict != JudgeVerdict.CONFIRMED:
                return gui_evaluation
            has_concrete_proof = True
            evidence_items.extend(gui_evaluation.evidence_items)

        # A successful process is not evidence that it advanced the mission.
        # Require a meaningful objective token in the concrete receipt before
        # accepting a generic successful command as confirmation.
        has_overlap = has_relevant_successful_receipt
        if gui_evaluation is None and has_concrete_proof and goal_terms and not has_overlap:
            return JudgeEvaluation(
                verdict=JudgeVerdict.INCONCLUSIVE,
                confidence=0.85,
                notes="Execution succeeded but the receipt is not relevant to the sub-mission objective.",
                evidence_items=evidence_items,
            )

        # 5. Confirmed Grounded Verdict
        if has_concrete_proof and result.success:
            failure_note = (
                f" Earlier failed attempts remain in the trace ({len(failed_traces)}); "
                "the final relevant successful receipt was evaluated separately."
                if failed_traces
                else ""
            )
            return JudgeEvaluation(
                verdict=JudgeVerdict.CONFIRMED,
                confidence=0.9,
                notes=(
                    "Independent verification confirmed concrete execution receipts "
                    f"with exit code 0.{failure_note}"
                ),
                evidence_items=evidence_items,
            )

        return JudgeEvaluation(
            verdict=JudgeVerdict.INCONCLUSIVE,
            confidence=0.5,
            notes="Trace execution lacked sufficient independent reproduction receipts.",
        )
