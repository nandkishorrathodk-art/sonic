"""
SONIC Autonomous Teamwork & Dynamic Subagent Runtime (`/teamwork-preview` Protocol)
===================================================================================
Implements Antigravity-grade, 100% unscripted multi-agent teamwork for SONIC:

1. Dynamic Subagent Definition & Invocation (`define_subagent`, `invoke_subagent`, `send_message`, `manage_subagents`):
   - SONIC's LLM decides at runtime when to define specialized subagent types or invoke 1..N parallel subagents.
   - Zero hardcoded role checklists or puppet scripts: every subagent receives ONLY **What** (`requirements`) and **Objective Acceptance Criteria** (`acceptance_criteria`), leaving the **How** to the subagent's own autonomous reasoning.

2. Workspace Isolation (`inherit` | `branch` | `share`) + Display Lease (`:99`):
   - `inherit`: operates directly on the parent workspace.
   - `branch` / `share`: provisions an isolated sub-workspace (git worktree or `.sonic_subagents/<conv_id>`) so parallel subagents never cause file/merge conflicts, and merges verified outputs back into the parent workspace ONLY after Judge verification passes.
   - `DisplayLeaseManager`: serializes physical X11 `:99` mouse/keyboard/window actions via an async lease while allowing terminal, file, git, and tool actions to execute 100% in parallel.

3. Dual Verification Forcing Function + Self-Correcting `send_message` Loop:
   - When a subagent finishes and claims completion, an independent Judge (`JudgeAgent` + Acceptance Criteria rubric) verifies the real workspace artifacts/commands.
   - If any criterion fails, the runtime automatically sends the failure evidence back to the SAME subagent via `send_message(conversation_id, ...)` so it debugs and fixes its work before returning to the parent Being.

4. Silent Background Execution:
   - Subagents and Judge loops execute silently in the background without polluting the operator's main Workstation UI feed; only the Parent Being's main desktop and final merged result are surfaced.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from sonic.agents.judge import JudgeAgent, JudgeVerdict
from sonic.computer_use.models_boss import SubMission, SubMissionResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Display Lease Manager (Shared :99 GUI Focus Serialization)
# ---------------------------------------------------------------------------

class DisplayLeaseManager:
    """Ensures parallel subagents never fight over the physical X11 mouse/keyboard cursor on :99."""

    _locks: dict[str, asyncio.Lock] = {}
    _lock_metadata: dict[str, dict[str, Any]] = {}

    class Interrupted(RuntimeError):
        """Raised when a waiting display lease is interrupted."""

    @classmethod
    def get_lock(cls, workspace_id: str, display: str | None = None) -> asyncio.Lock:
        # Workspace IDs are logical; several tenant workspaces can still share
        # the same physical X11 display. Serialize by display to protect the
        # real cursor and keyboard, while retaining a workspace fallback for
        # providers that expose an isolated display.
        key = f"display:{display or os.environ.get('DISPLAY') or ':99'}"
        if key not in cls._locks:
            cls._locks[key] = asyncio.Lock()
            cls._lock_metadata[key] = {"display": display or os.environ.get("DISPLAY") or ":99"}
        return cls._locks[key]

    @classmethod
    async def acquire(
        cls,
        workspace_id: str,
        display: str | None = None,
        *,
        timeout: float = 30.0,
        owner: str | None = None,
        interrupt_check: Any | None = None,
    ) -> asyncio.Lock:
        """Acquire exclusive ownership; elapsed time never transfers a live lease."""
        key = f"display:{display or os.environ.get('DISPLAY') or ':99'}"
        lock = cls.get_lock(workspace_id, display=display)
        deadline = time.monotonic() + timeout
        while True:
            if callable(interrupt_check) and interrupt_check():
                raise cls.Interrupted("display lease acquisition interrupted")
            try:
                await asyncio.wait_for(lock.acquire(), timeout=min(0.05, max(0.001, deadline - time.monotonic())))
                break
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    raise

        cls._lock_metadata[key].update(
            {
                "owner": owner or workspace_id,
                "owner_task": asyncio.current_task(),
                "acquired_at": time.time(),
            }
        )
        return lock

    @classmethod
    def release(cls, lock: asyncio.Lock, display: str | None = None) -> None:
        """Release a lease and clear ownership telemetry."""
        key = f"display:{display or os.environ.get('DISPLAY') or ':99'}"
        metadata = cls._lock_metadata.get(key)
        is_owner = (
            metadata is not None
            and metadata.get("owner_task") is asyncio.current_task()
        )
        if is_owner and cls._locks.get(key) is lock and lock.locked():
            metadata.pop("owner", None)
            metadata.pop("owner_task", None)
            metadata.pop("acquired_at", None)
            try:
                lock.release()
            except RuntimeError:
                pass


# ---------------------------------------------------------------------------
# Data Models for Dynamic Subagents & Teamwork
# ---------------------------------------------------------------------------

@dataclass
class DefinedSubagentType:
    """A dynamically defined subagent template registered via `DEFINE_SUBAGENT`."""

    name: str
    description: str
    system_prompt: str
    enable_write_tools: bool = True
    enable_subagent_tools: bool = False
    created_at: float = field(default_factory=time.time)


def _norm_str_list(v: Any) -> list[str]:
    """Normalize string, list, or None into a clean list of non-empty strings.
    Prevents Python's list(str) from splitting string character-by-character."""
    if not v:
        return []
    if isinstance(v, str):
        lines = [ln.strip() for ln in v.splitlines() if ln.strip()]
        return lines if lines else [v.strip()]
    if isinstance(v, (list, tuple, set)):
        return [str(it).strip() for it in v if it is not None and str(it).strip()]
    return [str(v).strip()]


def _get_field(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Case-insensitive and alias-aware dictionary field lookup."""
    if not isinstance(d, dict) or not d:
        return default

    def _is_valid(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False
        return True

    for target_key in keys:
        if not target_key:
            continue

        # 1. Exact match
        if target_key in d:
            val = d[target_key]
            if _is_valid(val):
                return val

        # 2. Case-insensitive match
        t_lower = target_key.lower()
        for k, v in d.items():
            if str(k).lower() == t_lower and _is_valid(v):
                return v

        # 3. Normalized alphanumeric match (ignores underscores, hyphens, casing)
        t_norm = "".join(c for c in t_lower if c.isalnum())
        if t_norm:
            for k, v in d.items():
                k_norm = "".join(c for c in str(k).lower() if c.isalnum())
                if k_norm == t_norm and _is_valid(v):
                    return v

    return default


@dataclass
class SubagentTaskSpec:
    """A single subagent invocation entry inside `INVOKE_SUBAGENT`."""

    type_name: str = "self"
    role: str = "Autonomous Subagent"
    prompt: str = ""
    requirements: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    verification_command: str = ""
    workspace_mode: str = "branch"   # "branch" (default isolated) | "inherit" | "share"
    execution_plane: str = "dual"
    model_tier: str = "inherit"      # "inherit" | "flash" | "pro"
    max_steps: int = 12
    max_correction_rounds: int = 2
    employee_id: str | None = None
    employee_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.requirements = _norm_str_list(self.requirements)
        self.acceptance_criteria = _norm_str_list(self.acceptance_criteria)


@dataclass
class SubagentConversation:
    """Live state of an invoked background subagent conversation."""

    conversation_id: str
    type_name: str
    role: str
    workspace_mode: str
    parent_workspace_id: str
    effective_workspace_id: str
    employee_id: str | None = None
    employee_context: dict[str, Any] = field(default_factory=dict)
    isolated_dir: Path | None = None
    git_branch: str | None = None
    state: str = "running"  # "running" | "verifying" | "idle" | "completed" | "errored" | "canceled"
    state_detail: str = ""
    agent_instance: Any = None
    messages: list[dict[str, str]] = field(default_factory=list)
    traces: list[Any] = field(default_factory=list)
    last_summary: str = ""
    verified: bool = False
    judge_verdict: str = "UNVERIFIED"
    judge_notes: str = ""
    correction_rounds_used: int = 0
    merged_files: list[str] = field(default_factory=list)
    current_task: Any = None
    last_spec: Any = None


# ---------------------------------------------------------------------------
# Subagent Teamwork Runtime
# ---------------------------------------------------------------------------

class SubagentTeamworkManager:
    """Manages dynamic subagent definitions, parallel invocations, `send_message` continuation,
    isolated workspace branching/merging, and independent Judge forcing functions.
    """

    def __init__(
        self,
        parent_agent: Any,
        computer_provider: Any,
        llm_router: Any,
        safety: Any,
        tenant_id: str = "default",
    ) -> None:
        self.parent = parent_agent
        self.computer = computer_provider
        self.llm_router = llm_router
        self.safety = safety
        self.tenant_id = tenant_id
        self.defined_types: dict[str, DefinedSubagentType] = {
            "self": DefinedSubagentType(
                name="self",
                description="General-purpose AI-Human subagent inheriting full workstation capabilities.",
                system_prompt="",
                enable_write_tools=True,
                enable_subagent_tools=True,
            ),
            "research": DefinedSubagentType(
                name="research",
                description="Read-focused research and verification subagent.",
                system_prompt="Focus on fast, accurate investigation, reading files/pages, and returning verified facts.",
                enable_write_tools=False,
                enable_subagent_tools=False,
            ),
        }
        self.conversations: dict[str, SubagentConversation] = {}
        self.judge = JudgeAgent(
            computer_provider=self.computer,
            judge_id=f"teamwork-judge-{tenant_id}",
            llm_router=self.llm_router,
        )
        self._baseline_snapshots: dict[str, dict[str, tuple[int, str]]] = {}

    # ------------------------------------------------------------------
    # 1. DEFINE_SUBAGENT
    # ------------------------------------------------------------------

    def define_subagent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        enable_write_tools: bool = True,
        enable_subagent_tools: bool = False,
    ) -> DefinedSubagentType:
        clean_name = (name or "").strip().lower().replace(" ", "_")
        if not clean_name:
            raise ValueError("Subagent type `name` must be non-empty.")
        spec = DefinedSubagentType(
            name=clean_name,
            description=(description or "").strip(),
            system_prompt=(system_prompt or "").strip(),
            enable_write_tools=bool(enable_write_tools),
            enable_subagent_tools=bool(enable_subagent_tools),
        )
        self.defined_types[clean_name] = spec
        logger.info("subagent_type_defined", name=clean_name, write_tools=enable_write_tools)
        return spec

    # ------------------------------------------------------------------
    # 2. INVOKE_SUBAGENT (1..N Concurrent Silent Background Subagents)
    # ------------------------------------------------------------------

    async def invoke_subagents(
        self,
        workspace_id: str,
        subagents: list[SubagentTaskSpec | dict[str, Any]],
    ) -> list[SubagentConversation]:
        specs: list[SubagentTaskSpec] = []
        for raw in subagents:
            if isinstance(raw, SubagentTaskSpec):
                if raw.prompt.strip():
                    specs.append(raw)
            elif isinstance(raw, dict):
                p = str(
                    _get_field(raw, "prompt", "goal", "task", "instruction", "objective", default="") or ""
                ).strip()
                if not p:
                    continue

                type_name = str(_get_field(raw, "type_name", "type", default="self") or "self").strip()
                role = str(_get_field(raw, "role", "title", "name", default="Autonomous Subagent") or "Autonomous Subagent").strip()
                requirements = _norm_str_list(_get_field(raw, "requirements", default=None))
                acceptance_criteria = _norm_str_list(
                    _get_field(raw, "acceptance_criteria", "criteria", default=None)
                )
                verification_command = str(
                    _get_field(raw, "verification_command", "verify_cmd", default="") or ""
                ).strip()
                workspace_mode = str(
                    _get_field(raw, "workspace_mode", "workspace", default="branch") or "branch"
                ).lower().strip()
                execution_plane = str(
                    _get_field(raw, "execution_plane", "plane", default="dual") or "dual"
                ).lower().strip()
                model_tier = str(
                    _get_field(raw, "model_tier", "model", default="inherit") or "inherit"
                ).lower().strip()

                raw_max_steps = _get_field(raw, "max_steps", default=12)
                try:
                    max_steps = int(str(raw_max_steps if raw_max_steps is not None else 12))
                except (ValueError, TypeError):
                    max_steps = 12

                raw_max_cr = _get_field(raw, "max_correction_rounds", default=2)
                try:
                    max_correction_rounds = int(str(raw_max_cr if raw_max_cr is not None else 2))
                except (ValueError, TypeError):
                    max_correction_rounds = 2

                raw_emp_id = _get_field(raw, "employee_id", default=None)
                employee_id = str(raw_emp_id).strip() or None if raw_emp_id is not None else None
                if employee_id in ("None", "null", ""):
                    employee_id = None

                specs.append(
                    SubagentTaskSpec(
                        type_name=type_name,
                        role=role,
                        prompt=p,
                        requirements=requirements,
                        acceptance_criteria=acceptance_criteria,
                        verification_command=verification_command,
                        workspace_mode=workspace_mode,
                        execution_plane=execution_plane,
                        model_tier=model_tier,
                        max_steps=max_steps,
                        max_correction_rounds=max_correction_rounds,
                        employee_id=employee_id,
                    )
                )

        if not specs:
            return []

        async def _launch_one(spec: SubagentTaskSpec) -> SubagentConversation:
            conv_id = f"subconv-{uuid.uuid4().hex[:10]}"
            w_mode = spec.workspace_mode if spec.workspace_mode in ("inherit", "branch", "share") else "branch"
            eff_ws_id, isolated_path = await self._prepare_subagent_workspace(workspace_id, conv_id, w_mode)

            employee_context: dict[str, Any] = {}
            if spec.employee_id:
                from sonic.organization.store import get_organization_store
                employee = await get_organization_store().get_employee(
                    self.tenant_id, spec.employee_id
                )
                if employee is None:
                    raise ValueError(
                        f"Unknown employee '{spec.employee_id}' for tenant '{self.tenant_id}'"
                    )
                employee_context = employee
            conv = SubagentConversation(
                conversation_id=conv_id,
                type_name=spec.type_name,
                role=spec.role,
                employee_id=spec.employee_id,
                employee_context=employee_context,
                workspace_mode=w_mode,
                parent_workspace_id=workspace_id,
                effective_workspace_id=eff_ws_id,
                isolated_dir=isolated_path,
                git_branch=f"subagent/{conv_id}" if w_mode in ("branch", "share") else None,
                state="running",
                state_detail=f"Executing: {spec.role}",
                last_spec=spec,
            )
            self.conversations[conv_id] = conv

            # Format the prompt using `/teamwork-preview` Principles (What + Acceptance Criteria, NOT How)
            formatted_goal = self._format_teamwork_prompt(spec, isolated_path)
            conv.messages.append({"role": "user", "content": formatted_goal})

            # Run the subagent + Independent Judge Forcing Function loop silently in background
            await self._execute_and_verify_loop(conv, spec, formatted_goal)
            return conv

        raw_results = await asyncio.gather(*(_launch_one(s) for s in specs), return_exceptions=True)
        results: list[SubagentConversation] = []
        for spec, r in zip(specs, raw_results):
            if isinstance(r, (Exception, BaseException)):
                logger.error("subagent_launch_exception", role=spec.role, error=str(r))
                err_conv = None
                for c in list(self.conversations.values()):
                    if getattr(c, "last_spec", None) is spec:
                        c.state = "errored"
                        c.state_detail = f"Launch failed: {r}"
                        c.last_summary = f"Launch failed: {r}"
                        err_conv = c
                        break
                if err_conv is None:
                    conv_id = f"subconv-{uuid.uuid4().hex[:10]}"
                    w_mode = spec.workspace_mode if spec.workspace_mode in ("inherit", "branch", "share") else "branch"
                    err_conv = SubagentConversation(
                        conversation_id=conv_id,
                        type_name=spec.type_name,
                        role=spec.role,
                        workspace_mode=w_mode,
                        parent_workspace_id=workspace_id,
                        effective_workspace_id=workspace_id,
                        employee_id=spec.employee_id,
                        employee_context=getattr(spec, "employee_context", {}) or {},
                        state="errored",
                        state_detail=f"Launch failed: {r}",
                        last_summary=f"Launch failed: {r}",
                        last_spec=spec,
                    )
                    self.conversations[conv_id] = err_conv
                results.append(err_conv)
            elif isinstance(r, SubagentConversation):
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # 3. SEND_MESSAGE (Continue / Redirect / Fix an Existing Subagent)
    # ------------------------------------------------------------------

    async def send_message(
        self,
        conversation_id: str,
        message: str,
        acceptance_criteria: list[str] | None = None,
        verification_command: str = "",
        max_steps: int = 8,
    ) -> SubagentConversation:
        conv = self.conversations.get(conversation_id)
        if conv is None:
            raise KeyError(f"Unknown subagent conversation_id: {conversation_id}")

        conv.state = "running"
        conv.state_detail = f"Continuing conversation: {message[:80]}"
        conv.messages.append({"role": "user", "content": message})

        prev_spec = getattr(conv, "last_spec", None)
        effective_criteria = (
            _norm_str_list(acceptance_criteria)
            if acceptance_criteria
            else (list(prev_spec.acceptance_criteria) if prev_spec else [])
        )
        effective_command = (
            verification_command.strip()
            if verification_command and verification_command.strip()
            else (prev_spec.verification_command if prev_spec else "")
        )
        effective_requirements = list(prev_spec.requirements) if prev_spec else []

        continuation_spec = SubagentTaskSpec(
            type_name=conv.type_name,
            role=conv.role,
            prompt=message,
            requirements=effective_requirements,
            acceptance_criteria=effective_criteria,
            verification_command=effective_command,
            workspace_mode=conv.workspace_mode,
            execution_plane=getattr(conv.agent_instance, "execution_plane", "dual") if conv.agent_instance else "dual",
            employee_id=conv.employee_id,
            employee_context=dict(conv.employee_context),
            max_steps=max_steps,
            max_correction_rounds=1,
        )
        conv.last_spec = continuation_spec
        formatted_message = self._format_teamwork_prompt(continuation_spec, conv.isolated_dir)
        await self._execute_and_verify_loop(conv, continuation_spec, formatted_message)
        return conv

    # ------------------------------------------------------------------
    # 4. MANAGE_SUBAGENTS (list / kill / kill_all)
    # ------------------------------------------------------------------

    def manage_subagents(
        self,
        action: str = "list",
        conversation_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        def _terminate_conversation(c: SubagentConversation, detail: str) -> None:
            c.state = "canceled"
            c.state_detail = detail
            cur = getattr(c, "current_task", None)
            if cur is not None and cur is not asyncio.current_task():
                try:
                    if not cur.done():
                        cur.cancel()
                except Exception:
                    pass
            if getattr(c, "agent_instance", None) is not None:
                try:
                    if hasattr(c.agent_instance, "interrupt"):
                        c.agent_instance.interrupt()
                except Exception:
                    pass

        act = (action or "list").strip().lower()
        if act == "kill_all":
            for c in self.conversations.values():
                if c.state in ("running", "verifying"):
                    _terminate_conversation(c, "Terminated via kill_all")
        elif act == "kill" and conversation_ids:
            for cid in conversation_ids:
                if cid in self.conversations:
                    _terminate_conversation(self.conversations[cid], "Terminated via kill")

        return [
            {
                "conversationId": c.conversation_id,
                "role": c.role,
                "type": c.type_name,
                "workspaceMode": c.workspace_mode,
                "gitBranch": c.git_branch,
                "state": c.state,
                "stateDetail": c.state_detail,
                "verified": c.verified,
                "judgeVerdict": c.judge_verdict,
                "judgeNotes": c.judge_notes,
                "correctionRounds": c.correction_rounds_used,
                "mergedFiles": c.merged_files,
                "summary": c.last_summary[:300],
            }
            for c in self.conversations.values()
        ]

    # ------------------------------------------------------------------
    # Internal Helpers: Prompt Formatting, Isolation, & Judge Loop
    # ------------------------------------------------------------------

    def _format_teamwork_prompt(self, spec: SubagentTaskSpec, isolated_dir: Path | None) -> str:
        """Apply Antigravity `/teamwork-preview` structured envelope principles:
        1. Role Mandate & Specialization
        2. Employee Ownership Context (if specified)
        3. Primary Objective (Specify What, Not How)
        4. Operational Execution Plane (gui, headless, dual) with clear capability rules
        5. Isolated Workspace mode (branch, share, inherit)
        6. Functional Requirements (- **R1**: ...)
        7. Objective Acceptance Criteria (Independent Judge Rubric, - [ ] ...)
        8. Programmatic Verification Forcing Function (Command that must exit 0)
        9. Operational Invariants & Guidelines
        """
        type_def = self.defined_types.get(spec.type_name.lower())
        sections: list[str] = ["=== SUBAGENT TASK ENVELOPE ==="]

        # 1. Role Mandate & Specialization
        role_mandate_text = type_def.system_prompt if (type_def and type_def.system_prompt) else spec.role
        sections.append(
            f"## Role Mandate & Specialization\n"
            f"[ROLE MANDATE ({spec.role})]: {role_mandate_text}\n"
            f"- Assigned Role: {spec.role}\n"
            f"- Subagent Type: {spec.type_name}"
        )

        # 2. Employee Ownership Context (if specified)
        if spec.employee_id:
            ctx = spec.employee_context
            sections.append(
                "## Employee Ownership Context\n"
                f"- Employee: {ctx.get('name', spec.employee_id)} ({spec.employee_id})\n"
                f"- Role: {ctx.get('role', spec.role)}\n"
                f"- Department: {ctx.get('department_id', 'unknown')}\n"
                f"- Team: {ctx.get('team_id', 'unknown')}\n"
                f"- Owned subsystems: {', '.join(ctx.get('subsystem_ids', []))}"
            )

        # 3. Primary Objective (Specify What, Not How)
        sections.append(
            f"## Primary Objective (Specify What, Not How)\n"
            f"{spec.prompt.strip()}"
        )

        # 4. Operational Execution Plane
        execution_plane = str(spec.execution_plane or "dual").lower().strip()
        if execution_plane not in {"gui", "headless", "dual"}:
            execution_plane = "dual"
        if execution_plane == "gui":
            plane_rules = "Operates strictly on visible desktop (DISPLAY=:99). Must interact with graphical applications, windows, and UI controls."
        elif execution_plane == "headless":
            plane_rules = "Direct headless execution plane (TERMINAL_EXEC, files, git). Do not poll screenshots or attempt desktop GUI actions."
        else:
            plane_rules = "Full dual-plane workstation agency. Freely operate graphical desktop applications (:99) and direct headless terminal/file actions as appropriate."
        sections.append(
            f"## Operational Execution Plane\n"
            f"- Assigned Plane: `{execution_plane}`\n"
            f"- Capability Rules: {plane_rules}"
        )

        # 5. Isolated Workspace mode
        ws_mode = spec.workspace_mode.lower().strip() if spec.workspace_mode else "branch"
        if ws_mode not in ("inherit", "branch", "share"):
            ws_mode = "branch"
        iso_info = f"`{isolated_dir.as_posix()}`" if isolated_dir is not None else "(shared workspace)"
        if ws_mode == "branch":
            mode_desc = "Isolated branch copy. Verified changes will be cleanly merged back to parent upon passing independent verification."
        elif ws_mode == "share":
            mode_desc = "Shared git worktree. Operates on a dedicated branch with zero collision."
        else:
            mode_desc = "Directly inherits parent workspace root."
        sections.append(
            f"## Isolated Workspace Mode\n"
            f"- Mode: `{ws_mode}`\n"
            f"- Working Directory: {iso_info}\n"
            f"- Isolation Policy: {mode_desc}"
        )

        # 6. Functional Requirements
        if spec.requirements:
            req_lines = "\n".join(f"- **R{i+1}**: {r}" for i, r in enumerate(spec.requirements))
        else:
            req_lines = f"- **R1**: Deliver complete outcome satisfying: {spec.prompt.strip()}"
        sections.append(f"## Functional Requirements\n{req_lines}")

        # 7. Objective Acceptance Criteria (Independent Judge Rubric)
        if spec.acceptance_criteria:
            ac_lines = "\n".join(f"- [ ] {ac}" for ac in spec.acceptance_criteria)
        else:
            ac_lines = f"- [ ] Tangible output/artifact created for: {spec.prompt.strip()}"
        sections.append(
            f"## Objective Acceptance Criteria (Independent Judge Rubric)\n"
            f"Your work will be verified by an independent Judge against these exact criteria:\n"
            f"{ac_lines}"
        )

        # 8. Programmatic Verification Forcing Function
        if spec.verification_command and spec.verification_command.strip():
            sections.append(
                f"## Programmatic Verification Forcing Function\n"
                f"Command that must exit 0: `{spec.verification_command.strip()}`"
            )
        else:
            sections.append(
                "## Programmatic Verification Forcing Function\n"
                "None specified. Verification relies on Independent Judge rubric evaluation."
            )

        # 9. Operational Invariants & Guidelines
        sections.append(
            "## Operational Invariants & Guidelines\n"
            "- Pursue the entire objective holistically; do not stop until all requirements and criteria are satisfied.\n"
            "- Ground every action in live observation receipts rather than assuming state.\n"
            "- Zero self-attestation: Do NOT declare GOAL_COMPLETE without concrete verifiable files or traces.\n"
            "- All executions, scripts, and commands must terminate cleanly with exit code 0."
        )

        return "\n\n".join(sections)

    @staticmethod
    def _snapshot_dir(dir_path: Path | None) -> dict[str, tuple[int, str]]:
        """Takes a content digest snapshot (size, sha256) of all files in directory."""
        snapshot: dict[str, tuple[int, str]] = {}
        if dir_path is None or not dir_path.exists():
            return snapshot
        for f in dir_path.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(dir_path)
            if any(part in (".sonic_subagents", "__pycache__", ".pytest_cache", "node_modules", ".git") for part in rel.parts):
                continue
            try:
                content = f.read_bytes()
                h = hashlib.sha256(content).hexdigest()
                snapshot[rel.as_posix()] = (len(content), h)
            except Exception:
                pass
        return snapshot

    async def _prepare_subagent_workspace(
        self,
        parent_workspace_id: str,
        conversation_id: str,
        workspace_mode: str,
    ) -> tuple[str, Path | None]:
        """Prepare `inherit`, `branch`, or `share` workspace isolation."""
        if workspace_mode == "inherit":
            return parent_workspace_id, None

        # Resolve host path if using HeadlessComputeProvider / local workspace
        parent_dir: Path | None = None
        if hasattr(self.computer, "_get_workspace_dir"):
            try:
                parent_dir = Path(self.computer._get_workspace_dir(parent_workspace_id))
            except Exception:
                parent_dir = None

        git_branch_name = f"subagent/{conversation_id}"
        is_git_repo = parent_dir is not None and (parent_dir / ".git").exists()

        # If parent workspace is a git repo, create a dedicated branch without switching parent HEAD or disturbing uncommitted files
        if is_git_repo and parent_dir is not None:
            try:
                subprocess.run(
                    ["git", "branch", git_branch_name, "HEAD"],
                    cwd=parent_dir,
                    check=False,
                    capture_output=True,
                )
            except Exception:
                pass

        eff_ws_id = f"{parent_workspace_id}--branch-{conversation_id}"
        iso_root: Path | None = None

        if hasattr(self.computer, "create"):
            try:
                sub_ws = await self.computer.create(
                    tenant_id=getattr(self.parent, "tenant_id", "default"),
                    engagement_id=f"branch-{conversation_id}",
                )
                eff_ws_id = sub_ws.id
                if hasattr(self.computer, "_get_workspace_dir"):
                    iso_root = Path(self.computer._get_workspace_dir(eff_ws_id))
            except Exception:
                pass

        if iso_root is None and parent_dir is not None and parent_dir.exists():
            iso_root = parent_dir / ".sonic_subagents" / conversation_id
            iso_root.mkdir(parents=True, exist_ok=True)

        # 1. Workspace mode "share": Share underlying repository directory using git worktree (if git available)
        if workspace_mode == "share" and is_git_repo and parent_dir is not None and iso_root is not None:
            try:
                res = subprocess.run(
                    ["git", "worktree", "add", str(iso_root), git_branch_name],
                    cwd=parent_dir,
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    self._baseline_snapshots[conversation_id] = self._snapshot_dir(iso_root)
                    return eff_ws_id, iso_root
            except Exception:
                pass

        # 2. Workspace mode "branch" (or fallback for "share"): Copy baseline files to isolated sub-workspace
        if iso_root is not None and parent_dir is not None and parent_dir.exists():
            self._baseline_snapshots[conversation_id] = self._snapshot_dir(parent_dir)
            for item in parent_dir.iterdir():
                if item.name in (".sonic_subagents", "__pycache__", ".pytest_cache", "node_modules", ".git"):
                    continue
                dest = iso_root / item.name
                try:
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    elif item.is_file():
                        shutil.copy2(item, dest)
                except Exception:
                    pass
            return eff_ws_id, iso_root

        # Inside remote Linux/Docker container: create isolated directory under /tmp or workspace and copy files
        try:
            iso_remote = f"/tmp/sonic_subagents/{conversation_id}"
            copy_cmd = (
                f"mkdir -p '{iso_remote}' && "
                f"if [ -d '/home/sonic/workspace' ] && [ '/home/sonic/workspace' != '{iso_remote}' ]; then "
                f"  cp -a /home/sonic/workspace/. '{iso_remote}/'; "
                f"elif [ -d '.' ] && [ \"$(pwd)\" != '{iso_remote}' ]; then "
                f"  cp -a ./. '{iso_remote}/'; "
                f"fi"
            )
            await self.computer.terminal(parent_workspace_id, copy_cmd)
            return eff_ws_id, Path(iso_remote)
        except Exception:
            return eff_ws_id, None

    async def _merge_isolated_workspace(self, conv: SubagentConversation) -> list[str]:
        """Merge verified files from `branch`/`share` subagent workspace back into parent workspace."""
        if conv.isolated_dir is None or conv.workspace_mode == "inherit":
            return []

        parent_dir: Path | None = None
        if hasattr(self.computer, "_get_workspace_dir"):
            try:
                parent_dir = Path(self.computer._get_workspace_dir(conv.parent_workspace_id))
            except Exception:
                parent_dir = None

        merged: list[str] = []
        is_git_repo = parent_dir is not None and (parent_dir / ".git").exists()

        # 1. Clean Git Merge if a git branch exists and has commits ahead of parent
        if is_git_repo and parent_dir is not None and conv.git_branch:
            try:
                r_diff = subprocess.run(
                    ["git", "rev-list", "--count", f"HEAD..{conv.git_branch}"],
                    cwd=parent_dir,
                    capture_output=True,
                    text=True,
                )
                if r_diff.returncode == 0 and int(r_diff.stdout.strip() or 0) > 0:
                    # Clean merge with --autostash so uncommitted parent files are never clobbered or blocked
                    m_res = subprocess.run(
                        ["git", "merge", "--autostash", conv.git_branch],
                        cwd=parent_dir,
                        capture_output=True,
                        text=True,
                    )
                    if m_res.returncode == 0:
                        r_files = subprocess.run(
                            ["git", "diff", "--name-only", "HEAD~1..HEAD"],
                            cwd=parent_dir,
                            capture_output=True,
                            text=True,
                        )
                        for line in r_files.stdout.splitlines():
                            if line.strip():
                                merged.append(line.strip().replace("\\", "/"))
            except Exception as e:
                logger.warning("subagent_git_merge_failed", error=str(e), branch=conv.git_branch)

        # 2. File-level delta merge: ONLY merge new or genuinely modified files, preserving parent uncommitted files
        baseline = self._baseline_snapshots.get(conv.conversation_id, {})
        current_snapshot = self._snapshot_dir(conv.isolated_dir) if conv.isolated_dir.exists() else {}

        if parent_dir is not None and conv.isolated_dir.exists():
            for rel_str, (size, h) in current_snapshot.items():
                # If file did not exist in baseline OR its content/hash changed, it was created/modified by subagent!
                if rel_str not in baseline or baseline[rel_str][1] != h:
                    src_file = conv.isolated_dir / rel_str
                    dst_file = parent_dir / rel_str
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(src_file, dst_file)
                        if rel_str not in merged:
                            merged.append(rel_str)
                    except Exception as e:
                        logger.warning("subagent_file_merge_error", src=str(src_file), error=str(e))

            # Clean up git worktree if share mode was used
            if conv.workspace_mode == "share" and is_git_repo and conv.git_branch:
                try:
                    subprocess.run(["git", "worktree", "remove", "--force", str(conv.isolated_dir)], cwd=parent_dir, capture_output=True)
                    subprocess.run(["git", "branch", "-D", conv.git_branch], cwd=parent_dir, capture_output=True)
                except Exception:
                    pass

        # 3. Remote container fallback merge
        elif parent_dir is None and hasattr(self.computer, "terminal"):
            try:
                iso_remote = str(conv.isolated_dir).replace("\\", "/")
                merge_cmd = (
                    f"if [ -d '{iso_remote}' ]; then "
                    f"  DEST='.'; "
                    f"  if [ -d '/home/sonic/workspace' ]; then DEST='/home/sonic/workspace'; fi; "
                    f"  cp -r -u '{iso_remote}'/. \"$DEST\"/ && "
                    f"  (cd '{iso_remote}' && find . -maxdepth 2 -type f | sed 's|^\\./||'); "
                    f"fi"
                )
                res = await self.computer.terminal(conv.parent_workspace_id, merge_cmd)
                out = getattr(res, "stdout", "") or ""
                for line in out.splitlines():
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith("cp:") and not cleaned.startswith("find:") and not cleaned.startswith("sed:"):
                        merged.append(cleaned)
            except Exception:
                pass

        return merged

    async def _execute_and_verify_loop(
        self,
        conv: SubagentConversation,
        spec: SubagentTaskSpec,
        initial_goal: str,
    ) -> None:
        """Run the subagent silently in the background, then enforce the Dual Forcing Function
        (Programmatic Verification + Independent JudgeAgent). If verification fails, use
        self-correcting continuation (`send_message` feedback loop) until criteria pass or budget ends.
        """
        from sonic.computer_use.agent import ComputerUseAgent
        from sonic.computer_use.models import ComputerAutonomyLevel, EngineeringMissionMode
        from sonic.safety.action_policy import ActionPolicy

        conv.current_task = asyncio.current_task()
        try:
            type_def = self.defined_types.get(spec.type_name.lower())
            safety_clone = (
                self.safety.clone_for_agent(conv.conversation_id)
                if isinstance(self.safety, ActionPolicy)
                else self.safety
            )
            if isinstance(safety_clone, ActionPolicy):
                if conv.isolated_dir is not None:
                    safety_clone.workspace_root = conv.isolated_dir.resolve()
                if conv.employee_context.get("read_only"):
                    read_only_types = {
                        "FILE_READ", "GIT_STATUS", "GIT_DIFF",
                        "GUI_SCREENSHOT", "GUI_WAIT",
                        "BROWSER_NAVIGATE", "BROWSER_SCREENSHOT", "BROWSER_WAIT",
                        "WAIT_FOR_HUMAN", "GOAL_COMPLETE",
                    }
                    safety_clone.allowed_types = frozenset(
                        t for t in safety_clone.allowed_types if t in read_only_types
                    )
                elif type_def and not type_def.enable_write_tools:
                    write_tools = {"FILE_WRITE", "GIT_COMMIT", "GIT_PUSH", "TOOL_AUTHOR"}
                    safety_clone.allowed_types = frozenset(
                        t for t in safety_clone.allowed_types if t not in write_tools
                    )
                if type_def and not type_def.enable_subagent_tools:
                    sub_tools = {"DEFINE_SUBAGENT", "INVOKE_SUBAGENT", "SEND_MESSAGE", "MANAGE_SUBAGENTS"}
                    safety_clone.allowed_types = frozenset(t for t in safety_clone.allowed_types if t not in sub_tools)

            execution_plane = str(spec.execution_plane or "dual").lower().strip()
            if execution_plane not in {"gui", "headless", "dual"}:
                execution_plane = "dual"

            # Reuse existing ComputerUseAgent on continuation (`send_message`) so full history is preserved!
            if conv.agent_instance is None:
                scoped_ctx: dict[str, Any] = {
                    "subagent_conversation_id": conv.conversation_id,
                    "subagent_role": conv.role,
                    "workspace_mode": conv.workspace_mode,
                    "silent_background": True,
                    "employee_id": conv.employee_id or "",
                    "employee_department": conv.employee_context.get("department_id", ""),
                    "employee_team": conv.employee_context.get("team_id", ""),
                    "employee_subsystems": list(conv.employee_context.get("subsystem_ids", [])),
                }
                if conv.isolated_dir is not None:
                    scoped_ctx["isolated_working_dir"] = conv.isolated_dir.as_posix()

                compact_mode = True if spec.model_tier == "flash" else (False if spec.model_tier == "pro" else None)
                overlay = type_def.system_prompt if (type_def and type_def.system_prompt) else ""

                conv.agent_instance = ComputerUseAgent(
                    computer_provider=self.computer,
                    autonomy_level=ComputerAutonomyLevel.L3_AUTONOMOUS,
                    mode=EngineeringMissionMode.GENERAL_ENGINEERING_MODE,
                    max_actions=max(1, min(int(spec.max_steps), 30)),
                    llm_router=self.llm_router,
                    security_tools=getattr(self.parent, "security_tools", {}),
                    safety=safety_clone,
                    self_host=bool(safety_clone),
                    tenant_id=self.tenant_id,
                    agent_id=conv.conversation_id,
                    being=getattr(self.parent, "being", None),
                    being_mind=getattr(self.parent, "being_mind", None),
                    enable_llm_decomposition=False,
                    compact_mode=compact_mode,
                    system_prompt_overlay=overlay,
                    is_subagent_worker=True,
                    toolsmith=getattr(self.parent, "toolsmith", None),
                    method_lab=getattr(self.parent, "method_lab", None),
                    lessons_ledger=getattr(self.parent, "lessons_ledger", None),
                    observe_desktop=execution_plane in ("gui", "dual"),
                    execution_plane=execution_plane,
                    initial_context=scoped_ctx,
                )

            agent = conv.agent_instance
            current_goal = initial_goal
            max_rounds = max(0, int(spec.max_correction_rounds))

            for round_idx in range(max_rounds + 1):
                conv.state = "running"
                conv.state_detail = f"Round {round_idx + 1}/{max_rounds + 1}"

                # Replenish action budget for this round
                agent.max_actions = getattr(agent, "action_counter", 0) + max(1, min(int(spec.max_steps), 30))

                # Silent background execution: step_callback=None so main UI is never spammed!
                traces = await agent.run_mission(
                    workspace_id=conv.effective_workspace_id,
                    goal=current_goal,
                    steps=max(1, min(int(spec.max_steps), 30)),
                    step_callback=None,
                )
                conv.traces.extend(traces or [])
                summary = getattr(agent, "mission_summary", "") or self._summarize_traces(traces)
                conv.last_summary = summary
                conv.messages.append({"role": "assistant", "content": summary})

                # Independent Forcing Function Verification
                conv.state = "verifying"
                passed, verdict_str, notes = await self._verify_with_independent_judge(conv, spec, traces, summary)
                conv.judge_verdict = verdict_str
                conv.judge_notes = notes
                conv.verified = passed

                if passed:
                    if conv.workspace_mode in ("branch", "share"):
                        conv.merged_files = await self._merge_isolated_workspace(conv)
                    conv.state = "idle"
                    conv.state_detail = "Verified & Ready"
                    return

                # Verification failed — if we have correction rounds left, feed Judge feedback back to the same subagent!
                if round_idx < max_rounds:
                    conv.correction_rounds_used += 1
                    feedback_prompt = (
                        f"{initial_goal}\n\n"
                        f"=== INDEPENDENT JUDGE REJECTION — SELF-CORRECTION ROUND {round_idx + 1} ===\n"
                        f"Verdict: {verdict_str}\n"
                        f"Reason: {notes}\n\n"
                        "Do NOT self-certify. Inspect the actual workspace state, fix the root cause so that "
                        "every Acceptance Criterion and Verification Command genuinely passes, then declare GOAL_COMPLETE."
                    )
                    conv.messages.append({"role": "user", "content": feedback_prompt})
                    current_goal = feedback_prompt
                else:
                    conv.state = "errored"
                    conv.state_detail = f"Verification failed after {round_idx + 1} rounds: {notes[:120]}"
        finally:
            conv.current_task = None

    async def _verify_with_independent_judge(
        self,
        conv: SubagentConversation,
        spec: SubagentTaskSpec,
        traces: list[Any],
        summary: str,
    ) -> tuple[bool, str, str]:
        """Run programmatic verification_command (if specified) + JudgeAgent anti-self-attestation check."""
        # 1. Programmatic Verification Command (Hard Forcing Function)
        if spec.verification_command.strip():
            cmd = spec.verification_command.strip()
            try:
                cwd = str(conv.isolated_dir) if conv.isolated_dir is not None else None
                if hasattr(self.computer, "execute"):
                    res = await self.computer.execute(
                        workspace_id=conv.effective_workspace_id,
                        command=cmd,
                        cwd=cwd,
                    )
                elif hasattr(self.computer, "terminal"):
                    cmd_run = f'cd "{conv.isolated_dir.as_posix()}" && {cmd}' if conv.isolated_dir is not None else cmd
                    res = await self.computer.terminal(conv.effective_workspace_id, cmd_run)
                else:
                    res = None
                exit_code = getattr(res, "exit_code", 1)
                out = str(getattr(res, "stdout", "") or "") + "\n" + str(getattr(res, "stderr", "") or "")
                if exit_code != 0:
                    return (
                        False,
                        JudgeVerdict.FALSIFIED.value,
                        f"Programmatic verification command `{spec.verification_command}` failed with exit {exit_code}: {out[:300].strip()}",
                    )
            except Exception as e:
                return False, JudgeVerdict.FALSIFIED.value, f"Programmatic verification error: {e}"

        # 2. Independent JudgeAgent Evaluation
        has_file_or_terminal = any(
            str(getattr(t, "action_type", "") if not isinstance(t, dict) else t.get("action_type", "")).upper()
            in (
                "FILE_WRITE", "FILE_EDIT", "FILE_CREATE", "FILE_READ",
                "TERMINAL_EXEC", "GIT_COMMIT", "GIT_PUSH", "GIT_BRANCH",
            )
            for t in (traces or [])
        )
        has_gui = any(
            str(getattr(t, "action_type", "") if not isinstance(t, dict) else t.get("action_type", "")).upper().startswith("GUI_")
            or str(getattr(t, "action_type", "") if not isinstance(t, dict) else t.get("action_type", "")).upper() in ("APP_LAUNCH", "APP_FOCUS", "APP_CLOSE")
            or str(getattr(t, "action_type", "") if not isinstance(t, dict) else t.get("action_type", "")).upper().startswith("BROWSER_")
            for t in (traces or [])
        )

        if spec.execution_plane == "gui":
            eval_plane = "gui"
        elif spec.execution_plane == "headless":
            eval_plane = "headless"
        else:
            # "dual" mode: if file/terminal artifacts were produced, verify headless receipts;
            # if only GUI actions were taken, verify GUI receipts.
            eval_plane = "gui" if (has_gui and not has_file_or_terminal) else "headless"

        sub_m = SubMission(
            id=conv.conversation_id,
            goal=spec.prompt or conv.role,
            role=conv.role,
            max_steps=spec.max_steps,
            plane=eval_plane,
        )
        has_ok_trace = any(
            str(getattr(t, "status", "") if not isinstance(t, dict) else t.get("status", "")).upper()
            in ("COMPLETED", "SUCCESS", "VERIFIED")
            for t in (traces or [])
        )
        sub_res = SubMissionResult(
            sub_mission_id=conv.conversation_id,
            goal=sub_m.goal,
            role=conv.role,
            traces=list(traces or []),
            findings_summary=summary,
            success=has_ok_trace,
            actions_taken=len(traces or []),
        )
        if self.judge.llm_router is None and self.llm_router is not None:
            self.judge.llm_router = self.llm_router
        eval_res = await self.judge.evaluate_sub_mission(
            workspace_id=conv.effective_workspace_id,
            sub_mission=sub_m,
            result=sub_res,
            worker_agent_id=conv.conversation_id,
        )

        # 3. Acceptance Criteria Rubric Check (if explicit criteria provided)
        if eval_res.verdict == JudgeVerdict.FALSIFIED:
            return False, eval_res.verdict.value, eval_res.notes

        if eval_res.verdict == JudgeVerdict.INCONCLUSIVE:
            return False, eval_res.verdict.value, eval_res.notes or "Judge evaluation was inconclusive; failing closed."

        if spec.acceptance_criteria and self.llm_router is not None:
            try:
                from sonic.llm.schemas import LLMRequest, Message, MessageRole

                criteria_list = "\n".join(f"- {c}" for c in spec.acceptance_criteria)
                trace_digest = self._summarize_traces(traces)
                req = LLMRequest(
                    messages=[
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                "You are an Independent Acceptance Criteria Judge (`teamwork_preview` rubric evaluator). "
                                "Evaluate whether the subagent's execution evidence satisfies all acceptance criteria. "
                                "Never accept unverified claims. Output ONLY valid JSON: "
                                '{"passed": true/false, "notes": "specific criterion feedback"}'
                            ),
                        ),
                        Message(
                            role=MessageRole.USER,
                            content=(
                                f"Acceptance Criteria:\n{criteria_list}\n\n"
                                f"Subagent Execution Evidence:\n{trace_digest}\n\n"
                                f"Subagent Summary:\n{summary}"
                            ),
                        ),
                    ],
                    task_type="verification",
                    max_tokens=250,
                    temperature=0.1,
                )
                resp = await asyncio.wait_for(self.llm_router.complete(req), timeout=30.0)
                raw = (resp.content or "").strip()
                m = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) if "{" in raw and "}" in raw else {}
                if not m.get("passed", False):
                    return False, JudgeVerdict.FALSIFIED.value, str(m.get("notes") or "Acceptance criteria not met.")
            except Exception as judge_err:
                logger.warning("judge_acceptance_criteria_eval_failed", error=str(judge_err))
                return False, JudgeVerdict.FALSIFIED.value, f"Acceptance criteria verification error: {judge_err}"

        if eval_res.verdict == JudgeVerdict.CONFIRMED and has_ok_trace:
            return True, JudgeVerdict.CONFIRMED.value, eval_res.notes or "All acceptance criteria verified."
        return False, eval_res.verdict.value, eval_res.notes or "No successful execution evidence recorded."

    @staticmethod
    def _summarize_traces(traces: list[Any]) -> str:
        if not traces:
            return "No actions executed."
        lines: list[str] = []
        for t in traces[-8:]:
            act = getattr(t, "action_type", "") if not isinstance(t, dict) else t.get("action_type", "")
            act_str = act.value if hasattr(act, "value") else str(act)
            tgt = getattr(t, "target_resource", "") if not isinstance(t, dict) else t.get("target_resource", "")
            obs = getattr(t, "actual_observation", "") if not isinstance(t, dict) else t.get("actual_observation", "")
            status = getattr(t, "status", "") if not isinstance(t, dict) else t.get("status", "")
            lines.append(f"[{status}] {act_str} ({tgt}): {str(obs)[:200]}")
        return "\n".join(lines)
