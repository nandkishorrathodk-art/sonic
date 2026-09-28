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
    ) -> asyncio.Lock:
        """Acquire a physical-display lease without waiting forever."""
        lock = cls.get_lock(workspace_id, display=display)
        await asyncio.wait_for(lock.acquire(), timeout=timeout)
        key = f"display:{display or os.environ.get('DISPLAY') or ':99'}"
        cls._lock_metadata[key].update(
            {"owner": owner or workspace_id, "acquired_at": time.time()}
        )
        return lock

    @classmethod
    def release(cls, lock: asyncio.Lock, display: str | None = None) -> None:
        """Release a lease and clear ownership telemetry."""
        key = f"display:{display or os.environ.get('DISPLAY') or ':99'}"
        metadata = cls._lock_metadata.get(key)
        if metadata is not None:
            metadata.pop("owner", None)
            metadata.pop("acquired_at", None)
        if lock.locked():
            lock.release()


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
    model_tier: str = "inherit"      # "inherit" | "flash" | "pro"
    max_steps: int = 12
    max_correction_rounds: int = 2
    employee_id: str | None = None
    employee_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentConversation:
    """Live state of an invoked background subagent conversation."""

    conversation_id: str
    type_name: str
    role: str
    employee_id: str | None = None
    employee_context: dict[str, Any] = field(default_factory=dict)
    workspace_mode: str
    parent_workspace_id: str
    effective_workspace_id: str
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
        self.judge = JudgeAgent(computer_provider=self.computer, judge_id=f"teamwork-judge-{tenant_id}")
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
                specs.append(raw)
            elif isinstance(raw, dict):
                specs.append(
                    SubagentTaskSpec(
                        type_name=str(raw.get("type_name") or raw.get("TypeName") or "self"),
                        role=str(raw.get("role") or raw.get("Role") or "Autonomous Subagent"),
                        prompt=str(raw.get("prompt") or raw.get("Prompt") or raw.get("goal") or ""),
                        requirements=list(raw.get("requirements") or []),
                        acceptance_criteria=list(raw.get("acceptance_criteria") or []),
                        verification_command=str(raw.get("verification_command") or ""),
                        workspace_mode=str(raw.get("workspace_mode") or raw.get("Workspace") or "branch").lower().strip(),
                        model_tier=str(raw.get("model_tier") or raw.get("Model") or "inherit").lower().strip(),
                        max_steps=int(raw.get("max_steps") or 12),
                        max_correction_rounds=int(raw.get("max_correction_rounds") if raw.get("max_correction_rounds") is not None else 2),
                        employee_id=str(raw.get("employee_id") or "").strip() or None,
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
            )
            self.conversations[conv_id] = conv

            # Format the prompt using `/teamwork-preview` Principles (What + Acceptance Criteria, NOT How)
            formatted_goal = self._format_teamwork_prompt(spec, isolated_path)
            conv.messages.append({"role": "user", "content": formatted_goal})

            # Run the subagent + Independent Judge Forcing Function loop silently in background
            await self._execute_and_verify_loop(conv, spec, formatted_goal)
            return conv

        if len(specs) == 1:
            return [await _launch_one(specs[0])]
        raw_results = await asyncio.gather(*(_launch_one(s) for s in specs), return_exceptions=True)
        results: list[SubagentConversation] = []
        for r in raw_results:
            if isinstance(r, Exception):
                logger.error("subagent_launch_exception", error=str(r))
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

        continuation_spec = SubagentTaskSpec(
            type_name=conv.type_name,
            role=conv.role,
            prompt=message,
            acceptance_criteria=list(acceptance_criteria or []),
            verification_command=verification_command,
            workspace_mode=conv.workspace_mode,
            employee_id=conv.employee_id,
            employee_context=dict(conv.employee_context),
            max_steps=max_steps,
            max_correction_rounds=1,
        )
        await self._execute_and_verify_loop(conv, continuation_spec, message)
        return conv

    # ------------------------------------------------------------------
    # 4. MANAGE_SUBAGENTS (list / kill / kill_all)
    # ------------------------------------------------------------------

    def manage_subagents(
        self,
        action: str = "list",
        conversation_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        act = (action or "list").strip().lower()
        if act == "kill_all":
            for c in self.conversations.values():
                if c.state in ("running", "verifying"):
                    c.state = "canceled"
                    c.state_detail = "Terminated via kill_all"
        elif act == "kill" and conversation_ids:
            for cid in conversation_ids:
                if cid in self.conversations:
                    self.conversations[cid].state = "canceled"
                    self.conversations[cid].state_detail = "Terminated via kill"

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
        """Apply `/teamwork-preview` core principles:
        1. Specify What, Not How (requirements + acceptance criteria, zero puppet steps).
        2. Objective Verification Forcing Function.
        """
        type_def = self.defined_types.get(spec.type_name.lower())
        parts: list[str] = []
        if type_def and type_def.system_prompt:
            parts.append(f"[ROLE MANDATE ({spec.role})]: {type_def.system_prompt}")
        else:
            parts.append(f"[ROLE]: {spec.role}")

        if spec.employee_id:
            ctx = spec.employee_context
            parts.append(
                "\n## Employee Ownership Context\n"
                f"- Employee: {ctx.get('name', spec.employee_id)} ({spec.employee_id})\n"
                f"- Role: {ctx.get('role', spec.role)}\n"
                f"- Department: {ctx.get('department_id', 'unknown')}\n"
                f"- Team: {ctx.get('team_id', 'unknown')}\n"
                f"- Owned subsystems: {', '.join(ctx.get('subsystem_ids', []))}\n"
            )
        parts.append(f"\n## Objective (Specify What, Not How)\n{spec.prompt.strip()}")

        if isolated_dir is not None:
            parts.append(f"\nWorking directory (isolated {spec.workspace_mode} workspace): `{isolated_dir.as_posix()}`")

        if spec.requirements:
            req_lines = "\n".join(f"- **R{i+1}**: {r}" for i, r in enumerate(spec.requirements))
            parts.append(f"\n## Requirements\n{req_lines}")

        if spec.acceptance_criteria:
            ac_lines = "\n".join(f"- [ ] {ac}" for ac in spec.acceptance_criteria)
            parts.append(
                "\n## Objective Acceptance Criteria (Independent Judge Rubric)\n"
                "Your work will be verified by an independent Judge against these exact criteria:\n"
                f"{ac_lines}"
            )
        if spec.verification_command:
            parts.append(f"\n## Programmatic Verification Forcing Function\nCommand that must exit 0: `{spec.verification_command}`")

        return "\n".join(parts)

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

        # Inside remote Linux/Docker container: create isolated directory under /tmp or workspace
        try:
            iso_remote = f"/tmp/sonic_subagents/{conversation_id}"
            await self.computer.terminal(parent_workspace_id, f"mkdir -p {iso_remote}")
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
                merge_cmd = f"if [ -d '{iso_remote}' ]; then cp -r -u '{iso_remote}'/. . 2>&1 && ls -1 '{iso_remote}'; fi"
                res = await self.computer.terminal(conv.parent_workspace_id, merge_cmd)
                out = getattr(res, "stdout", "") or ""
                for line in out.splitlines():
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith("cp:"):
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
                sub_tools = {"DEFINE_SUBAGENT", "INVOKE_SUBAGENT"}
                safety_clone.allowed_types = frozenset(t for t in safety_clone.allowed_types if t not in sub_tools)

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
                browser=getattr(self.parent, "browser", None),
                toolsmith=getattr(self.parent, "toolsmith", None),
                method_lab=getattr(self.parent, "method_lab", None),
                lessons_ledger=getattr(self.parent, "lessons_ledger", None),
                observe_desktop=True,
                execution_plane="gui",
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
                    f"[INDEPENDENT JUDGE REJECTION — SELF-CORRECTION ROUND {round_idx + 1}]\n"
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
        sub_m = SubMission(
            id=conv.conversation_id,
            goal=spec.prompt or conv.role,
            role=conv.role,
            max_steps=spec.max_steps,
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
        eval_res = await self.judge.evaluate_sub_mission(
            workspace_id=conv.effective_workspace_id,
            sub_mission=sub_m,
            result=sub_res,
            worker_agent_id=conv.conversation_id,
        )

        # 3. Acceptance Criteria Rubric Check (if explicit criteria provided)
        if eval_res.verdict == JudgeVerdict.FALSIFIED:
            return False, eval_res.verdict.value, eval_res.notes

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
                if m.get("passed") is False:
                    return False, JudgeVerdict.FALSIFIED.value, str(m.get("notes") or "Acceptance criteria not met.")
            except Exception as judge_err:
                logger.warning("judge_acceptance_criteria_eval_failed", error=str(judge_err))
                return False, JudgeVerdict.FALSIFIED.value, f"Acceptance criteria verification error: {judge_err}"

        if has_ok_trace:
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
