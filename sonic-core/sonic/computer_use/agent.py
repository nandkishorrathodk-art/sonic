"""
SONIC-REDA — Autonomous Computer-Use Agent Engine (Phase 3: real reasoning)
==========================================================================
Closed-loop autonomous engineer that operates the SONIC Computer to
investigate, inspect, edit, build, test, debug, verify, and remediate code.

Phase 3 replaced the step-indexed hardcoded script with genuine
observe → reason (LLM) → act → observe-result reasoning. The agent carries
its action history and the live screen/terminal observation into each LLM
call, so its next action is a function of (goal, observation, history) — not
of a step counter. No vulnerability-specific fix is hardcoded anywhere.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import re
import shlex
import time
from typing import Any

from sonic.computer.models import (
    GUIAction,
    GUIActionType,
    ScreenObservation,
)
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.action_contracts import ActionReceipt, StateDelta
from sonic.computer_use.experience import compare_prediction, state_signature
from sonic.computer_use.grounding import (
    draw_action_marker,
    query_multimodal_grounding,
    resolve_ui_target_async,
)
from sonic.computer_use.gui_world_graph import GuiWorldGraph
from sonic.computer_use.models import (
    ActionExecutionStatus,
    AgentStepResult,
    ComputerActionType,
    ComputerAutonomyLevel,
    ComputerDecisionTrace,
    ComputerUseMetrics,
    ComputerWorldObservation,
    EngineeringMissionMode,
    StrategyState,
    SubGoal,
    SubGoalChecklist,
    SubGoalStatus,
)
from sonic.computer_use.motor import MotorReflexes
from sonic.computer_use.perception_bus import PerceptionBus
from sonic.computer_use.scratchpad import HackerScratchpad
from sonic.computer_use.wire_telemetry import WireTelemetryEngine
from sonic.llm.prompts import COMPUTER_USE_SYSTEM_PROMPT
from sonic.logger import get_logger
from sonic.orchestration.dual_process import DualProcessController
from sonic.research.failure_budget import FailureBudgetTracker
from sonic.research.failure_classifier import classify_failure


def _extract_command_from_sig(sig_payload_str: str) -> str:
    """Extract the normalized ``command`` value from a stored action signature.

    Signatures persist ``str(payload)`` (e.g. ``"{'command': 'hostname'}"``), so a
    repeat check that wants to compare the actual command must parse the payload
    back rather than comparing against the raw string representation. Falls back
    to the cleaned payload string when it cannot be parsed.
    """
    try:
        parsed = ast.literal_eval(sig_payload_str)
    except Exception:
        return sig_payload_str.strip().rstrip(',;').strip().lower()
    if isinstance(parsed, dict):
        return str(parsed.get("command") or parsed.get("tool") or parsed.get("text") or parsed.get("url") or "").strip().rstrip(',;').strip().lower()
    return sig_payload_str.strip().rstrip(',;').strip().lower()


def _safe_str(val: Any) -> str:
    """Ensure strings handle non-ASCII/emojis safely with errors='replace'
    to prevent Windows charmap/cp1252 codec crashes."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    s = str(val)
    # Force UTF-8 encoding to avoid charmap issues on Windows
    try:
        return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        # Last resort: replace problematic characters
        return s.encode("ascii", errors="replace").decode("ascii")


class _SafeLogger:
    """Proxy around logger ensuring all events and kwargs are string-safe for Windows console/logs."""
    def __init__(self, raw_logger: Any):
        self._raw_logger = raw_logger

    def _sanitize(self, val: Any) -> Any:
        if isinstance(val, str):
            return _safe_str(val)
        if isinstance(val, dict):
            return {str(k): self._sanitize(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [self._sanitize(v) for v in val]
        return str(val) if val is not None else ""

    def info(self, event: str, **kwargs: Any):
        try:
            self._raw_logger.info(_safe_str(event), **{str(k): self._sanitize(v) for k, v in kwargs.items()})
        except Exception:
            pass  # Fail silently to avoid encoding crashes

    def warning(self, event: str, **kwargs: Any):
        try:
            self._raw_logger.warning(_safe_str(event), **{str(k): self._sanitize(v) for k, v in kwargs.items()})
        except Exception:
            pass  # Fail silently to avoid encoding crashes

    def error(self, event: str, **kwargs: Any):
        try:
            self._raw_logger.error(_safe_str(event), **{str(k): self._sanitize(v) for k, v in kwargs.items()})
        except Exception:
            pass  # Fail silently to avoid encoding crashes

    def debug(self, event: str, **kwargs: Any):
        try:
            self._raw_logger.debug(_safe_str(event), **{str(k): self._sanitize(v) for k, v in kwargs.items()})
        except Exception:
            pass  # Fail silently to avoid encoding crashes


logger = _SafeLogger(get_logger(__name__))

# A sentinel the LLM may emit to signal the goal is achieved, so the mission
# loop can terminate early instead of running a fixed step count.
_GOAL_COMPLETE_SENTINEL = "GOAL_COMPLETE"


class ComputerUseAgent:
    """
    Autonomous Computer-Using Engineer uniting Brain (Cognitive State,
    Hypothesis Portfolio, Epistemic Reasoning) and Body (SONIC Computer).

    When an ``llm_router`` is supplied, action selection is genuinely LLM-driven:
    each step the agent observes the world (screen text, terminal output, files,
    git state), feeds that plus its action history to the LLM, and executes the
    returned action. Without a router it falls back to a generic, non-scripted
    diagnostic probe (read the primary file) — never a hardcoded vulnerability
    fix.
    """

    def __init__(
        self,
        computer_provider: ComputerProvider,
        autonomy_level: ComputerAutonomyLevel = ComputerAutonomyLevel.L3_AUTONOMOUS,
        mode: EngineeringMissionMode = EngineeringMissionMode.ENGINEERING_MODE,
        max_actions: int = 50,
        max_recovery_attempts: int = 5,
        llm_router: Any | None = None,
        browser: Any | None = None,
        security_tools: dict[str, Any] | None = None,
        safety: Any | None = None,
        self_host: bool = False,
        toolsmith: Any | None = None,
        method_lab: Any | None = None,
        lessons_ledger: Any | None = None,
        tenant_id: str = "default",
        engagement_id: str = "default",
        agent_id: str = "computer-use-agent",
        enable_llm_verification: bool = False,
        enable_llm_decomposition: bool = False,
        compact_mode: bool | None = None,
        failure_budget: FailureBudgetTracker | None = None,
        scratchpad: HackerScratchpad | None = None,
        motor: MotorReflexes | None = None,
        wire_telemetry: WireTelemetryEngine | None = None,
        skill_ledger: Any | None = None,
        being_mind: Any | None = None,
        craft: Any | None = None,
        evolution_engine: Any | None = None,
        observe_desktop: bool = True,
        gui_only: bool = False,
        initial_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.computer = computer_provider
        self.craft = craft
        # Keep the application desktop optional. Sandbox/operator missions
        # should not capture pixels merely because a workstation exists.
        self.observe_desktop = observe_desktop
        # The graphical computer plane can be locked away from PTY/shell
        # execution. Backend operator tooling remains a separate plane.
        self.gui_only = gui_only
        self.execution_plane = str(
            kwargs.get("execution_plane")
            or kwargs.get("plane")
            or ""
        ).lower().strip()
        if gui_only:
            logger.warning(
                "gui_only_mode_deprecated",
                message="gui_only mode violates Mandate 5 (Dual-Plane Concurrency); both GUI and Terminal planes should remain concurrently active.",
            )
        # Context supplied by the caller is advisory evidence, never a
        # replacement for the immutable operator objective.
        self.initial_context = dict(initial_context or {})
        # Shared versioned perception state keeps semantic targets and
        # sub-agents on one live snapshot without repeatedly decoding pixels.
        self.perception_bus = PerceptionBus()
        self.experience_records: list[Any] = []
        self.autonomy_level = autonomy_level
        self.mode = mode
        self.max_actions = max_actions
        self.max_recovery_attempts = max_recovery_attempts
        self.llm_router = llm_router  # ModelRouter for LLM-driven reasoning
        # Optional BrowserAgent so the same observe->reason->act loop can drive a
        # real web browser (navigate/click/type/screenshot) — unified computer-use.
        self.browser = browser
        # Optional registry of SecurityTool adapters (name -> tool) the agent can
        # invoke as a first-class reasoning action. Tools execute in-sandbox and
        # fail closed; their structured findings feed back into the observation.
        self.security_tools = security_tools or {}
        # Optional ToolsmithLoop (Phase A, AIOSR): the being authors NEW tools
        # for observation gaps. The authored tool is registered into the
        # security_tools map ONLY after a real in-sandbox run succeeds.
        self.toolsmith = toolsmith
        # Optional MethodLab (Phase B, AIOSR): the being synthesizes NOVEL
        # offensive techniques (new methods, not just tools) from observation +
        # failure + the known-technique ledger. Confirmed only on reproduction.
        self.method_lab = method_lab
        # Optional LessonsLedger (self-improvement learn→apply loop): persists
        # cross-mission lessons (failed approaches to avoid, successful ones to
        # reuse) and injects them into reasoning so the being does not forget
        # what it learned across missions. None = lessons not collected.
        self.lessons_ledger = lessons_ledger
        self.skill_ledger = skill_ledger
        # Optional Self-Evolution Engine: coordinates dynamic strategy, novel method invention,
        # toolsmithing, cross-mission learning, and codebase evolution.
        self.evolution_engine = evolution_engine
        if self.evolution_engine is None:
            try:
                from sonic.evolution.engine import EvolutionEngine
                from sonic.evolution.strategy import DynamicStrategyEngine
                self.evolution_engine = EvolutionEngine(
                    strategy_engine=DynamicStrategyEngine(),
                    method_lab=self.method_lab,
                    toolsmith=self.toolsmith,
                    lessons_ledger=self.lessons_ledger,
                )
            except Exception as e:
                logger.warning("lazy_evolution_engine_init_failed", error=str(e))
        self._last_adapted_strategy: dict[str, Any] | None = None
        # Optional fail-closed safety envelope (PLAN Phase 6). Required in
        # self-host mode: the agent may NOT act autonomously without a policy.
        self.safety = safety
        self.self_host = self_host
        if self_host and safety is None:
            raise ValueError(
                "self-host mode requires a safety policy — construct "
                "ComputerUseAgent with safety=ActionPolicy(...) so every "
                "autonomous action passes the fail-closed envelope."
            )
        self.tenant_id = tenant_id
        self.engagement_id = engagement_id
        self.agent_id = agent_id
        self.enable_llm_verification = enable_llm_verification
        self.enable_llm_decomposition = enable_llm_decomposition
        # Auto-detect compact mode for small models if not explicitly set.
        if compact_mode is not None:
            self._compact_mode = compact_mode
        elif llm_router is not None:
            raw_model = getattr(llm_router, "default_model", "")
            model_name = raw_model.lower() if isinstance(raw_model, str) else ""
            self._compact_mode = any(
                tag in model_name
                for tag in ("7b", "8b", "11b", "3b", "1b", "small", "mini", "tiny")
            )
        else:
            self._compact_mode = False
        self.failure_budget = failure_budget if failure_budget is not None else FailureBudgetTracker()
        self.scratchpad = scratchpad if scratchpad is not None else HackerScratchpad()
        # Optional BeingMind (persistent affect state—curiosity_drive/focus/
        # satiety). Injected into reasoning so the being's own mood actually
        # shapes what it tries next (not cosmetic dead floats).
        self.being_mind = being_mind
        self.motor = motor if motor is not None else MotorReflexes(self.computer)
        self.wire_telemetry = (
            wire_telemetry
            if wire_telemetry is not None
            else WireTelemetryEngine(**kwargs)
        )
        self.strategies: dict[str, dict[str, Any]] = {
            "Strategy A": {
                "name": "Direct Primary Execution",
                "description": "Primary probe / targeted execution",
                "state": StrategyState.ACTIVE,
            },
            "Strategy B": {
                "name": "Alternative Instrumentation",
                "description": "Secondary inspection / fallback tool",
                "state": StrategyState.ACTIVE,
            },
        }

        self.traces: list[ComputerDecisionTrace] = []
        self.recovery_events: int = 0
        self.action_counter: int = 0
        self.metrics = ComputerUseMetrics()
        # Running transcript of (action, result) the LLM reasons over. This is
        # what makes the agent adaptive: step N's action depends on step N-1's
        # observed outcome, not on a pre-written script.
        self.history: list[dict[str, str]] = []
        # Last captured browser page state, carried into the next observation so
        # the LLM sees the current page even between browser actions.
        self._last_browser_snapshot: Any | None = None
        # Last structured security-tool result, surfaced to the next reasoning
        # step so the LLM acts on real scan findings rather than a claim.
        self._last_tool_result: Any | None = None
        self._last_screenshot_b64: str = ""
        # Last observed screen dimensions, used to validate GUI coordinate
        # actions before they touch the provider. Updated on every observe() /
        # screenshot; sane defaults until the first real observation lands.
        self._screen_width: int = 1920
        self._screen_height: int = 1080
        self._interrupted: bool = False
        self.goal_reached: bool = False
        self._consecutive_failures: int = 0
        self._consecutive_unchanged_screens: int = 0
        self._goal_complete_declared: bool = False
        self._replan_count: int = 0
        self._last_navigated_url: str = ""
        self._recent_action_signatures: list[tuple[str, str, str]] = []
        self._last_action_output: dict[str, Any] = {}
        self._last_gui_action_result: dict[str, Any] = {}
        self.checklist: SubGoalChecklist | None = None

    @property
    def action_history(self) -> list[ComputerDecisionTrace]:
        return self.traces

    @action_history.setter
    def action_history(self, value: list[ComputerDecisionTrace]) -> None:
        self.traces = list(value or [])

    @staticmethod
    def _burst_gui_actions(payload: dict[str, Any]) -> list[GUIAction] | None:
        """Decode an explicitly requested GUI burst without inventing actions."""
        raw = payload.get("burst_actions")
        if raw is None:
            raw = payload.get("actions") if payload.get("burst") is True else None
        if raw is None:
            return None
        if not isinstance(raw, list) or not raw:
            raise ValueError("GUI burst requires a non-empty actions list")
        actions: list[GUIAction] = []
        for item in raw:
            if isinstance(item, GUIAction):
                actions.append(item)
                continue
            if not isinstance(item, dict):
                raise ValueError("GUI burst actions must be structured GUI actions")
            data = dict(item)
            raw_kind = data.pop("action", data.pop("type", ""))
            try:
                kind = GUIActionType(str(raw_kind).upper())
            except ValueError as exc:
                raise ValueError(f"unknown GUI burst action: {raw_kind}") from exc
            data["action"] = kind
            actions.append(GUIAction(**data))
        return actions

    def _burst_postcondition_ok(
        self, postcondition: Any, snapshot: Any, *, changed: bool
    ) -> tuple[bool, str]:
        """Check only caller-supplied live state; an absent condition is honest."""
        if postcondition is None:
            return True, ""
        if isinstance(postcondition, str):
            expected = postcondition.strip()
            ok = bool(expected) and expected.lower() in str(
                getattr(snapshot, "visible_text", "")
            ).lower()
            return ok, "postcondition text was not observed" if not ok else ""
        if not isinstance(postcondition, dict):
            return False, "postcondition must be a mapping or visible-text string"
        if postcondition.get("require_state_change", True) and not changed:
            return False, "postcondition requires a real UI state change"
        checks = {
            "active_window": getattr(snapshot, "active_window", ""),
            "visible_text": getattr(snapshot, "visible_text", ""),
            "browser_url": (getattr(snapshot, "browser_state", {}) or {}).get("url", ""),
            "url": (getattr(snapshot, "browser_state", {}) or {}).get("url", ""),
        }
        for key, expected in postcondition.items():
            if key in {"require_state_change", "screen_changed"}:
                continue
            if key not in checks:
                return False, f"unrecognized postcondition field: {key}"
            actual = str(checks[key] or "")
            if str(expected).lower() not in actual.lower():
                return False, f"postcondition {key} was not observed"
        if postcondition.get("screen_changed") is True and not changed:
            return False, "screen did not change as required"
        return True, ""

    async def _execute_gui_burst(
        self, workspace_id: str, payload: dict[str, Any], target_resource: str = ""
    ) -> tuple[bool, str, int | None]:
        """Run an explicit motor burst against versioned, freshly observed state."""
        actions = self._burst_gui_actions(payload)
        if actions is None:
            raise ValueError("not a GUI burst")
        bus = self.perception_bus
        initial = bus.current()

        async def refresh_after_failure() -> None:
            try:
                await self.observe(workspace_id)
            except Exception as exc:
                logger.warning("burst_reobserve_failed", error=str(exc))

        expected_version = payload.get("state_version")
        if expected_version is not None and int(expected_version) != initial.version:
            await refresh_after_failure()
            return False, "stale perception target; re-observe and replan", 126
        if target_resource:
            target = bus.resolve(target_resource)
            if target is not None and target.state_version != initial.version:
                await refresh_after_failure()
                return False, "stale perception target; re-observe and replan", 126
        previous = initial

        async def observe_after_action(*_args: Any) -> dict[str, Any]:
            nonlocal previous
            before = previous
            await self.observe(workspace_id)
            current = bus.current()
            changed = (
                current.screen_hash != before.screen_hash
                or current.active_window != before.active_window
                or current.windows != before.windows
                or current.visible_text != before.visible_text
                or current.controls != before.controls
                or current.browser_state != before.browser_state
            )
            if current.version <= before.version:
                return {"ok": False}
            previous = current
            # A burst must not silently continue through an unchanged or
            # unexpected desktop state.  The caller receives a replan signal.
            if not changed:
                return {"ok": False}
            return {"ok": True}

        result = await self.motor.execute_burst(
            workspace_id,
            actions,
            interval_seconds=float(payload.get("interval_seconds", 0.05)),
            observation_hook=observe_after_action,
        )
        if not result.success:
            await refresh_after_failure()
            return False, f"GUI burst stopped: {result.stop_reason or 'action failed'}; re-observe and replan", 126
        snapshot = bus.current()
        changed = snapshot.version > initial.version and snapshot != initial
        ok, reason = self._burst_postcondition_ok(
            payload.get("postcondition"), snapshot, changed=changed
        )
        if not ok:
            await refresh_after_failure()
            return False, f"GUI burst postcondition failed: {reason}; re-observe and replan", 126
        return True, f"GUI burst completed {result.completed} action(s) with verified observation", 0

    def interrupt(self) -> None:
        """Signal the agent to stop its active mission loop immediately."""
        self._interrupted = True

    async def _llm_decompose_goal(self, goal: str) -> SubGoalChecklist | None:
        """Use LLM to decompose the mission goal into concrete, actionable sub-goals."""
        from sonic.llm.schemas import LLMRequest, Message, MessageRole
        targets = self._extract_targets_from_goal(goal)
        target_info = ""
        if targets["urls"]:
            target_info = f" Target URL: {targets['urls'][0]}."
        elif targets["hostnames"]:
            target_info = f" Target Host: {targets['hostnames'][0]}."

        action_examples = (
            "Use only visible GUI actions such as launching/focusing an application, "
            "clicking, typing, keypresses, scrolling, dragging, waiting, and observing."
            if getattr(self, "gui_only", False)
            else
            "Each sub-goal must be a specific concrete action (for example, inspect a response, "
            "author a probe, or launch an application and verify its window)."
        )
        prompt = (
            f"Given this engineering or security mission:\n\"{goal}\"\n"
            f"{target_info}\n"
            "List 2 to 4 concrete, actionable, sequential sub-goals to accomplish it.\n"
            "Rules:\n"
            f"- {action_examples}\n"
            "- Do NOT generate vague generic steps like 'Inspect environment and orient', 'Execute core operation', or 'Verify outcome'.\n"
            "- Output ONLY numbered lines, e.g.:\n"
            "1. <action>\n"
            "2. <action>\n"
        )
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are an autonomous engineering planner. Decompose tasks into concrete, verifiable execution steps."),
                Message(role=MessageRole.USER, content=prompt),
            ],
            task_type="reasoning",
            max_tokens=256,
            temperature=0.2,
        )
        try:
            resp = await asyncio.wait_for(self.llm_router.complete(request), timeout=8.0)
            text = (resp.content or "").strip()
            items = []
            for line in text.splitlines():
                m = re.match(r'^(?:\d+[\.\)]|\-|\*)\s*(.*)', line.strip())
                if m and len(m.group(1).strip()) > 5:
                    items.append(m.group(1).strip())
            if len(items) >= 2:
                sub_goals = [SubGoal(description=item) for item in items[:5]]
                return SubGoalChecklist(top_level_goal=goal, sub_goals=sub_goals, active_index=0)
        except Exception as e:
            logger.debug("llm_decompose_goal_fallback", error=str(e))
        return None

    async def decompose_goal(self, goal: str) -> SubGoalChecklist:
        """Decompose a top-level mission into sequential, verifiable sub-goals."""
        sub_goals: list[SubGoal] = []

        # 1. Check for explicit numbered/bulleted steps in user prompt
        lines = [line.strip() for line in goal.splitlines() if line.strip()]
        if len(lines) <= 1:
            lines = [line.strip() for line in re.split(r'(?<=\w\w)\.\s+(?=[0-9]\.|\b[A-Z])', goal) if line.strip()]
        numbered_items = []
        for l in lines:
            m = re.match(r'^(?:\d+[\.\)]|\-|\*)\s*(.*)', l)
            if m and len(m.group(1).strip()) > 3:
                numbered_items.append(m.group(1).strip())

        if len(numbered_items) >= 2:
            for item in numbered_items[:6]:
                sub_goals.append(SubGoal(description=item))
            return SubGoalChecklist(top_level_goal=goal, sub_goals=sub_goals, active_index=0)

        # 2. Try LLM-driven goal decomposition when enabled and a router is available
        if getattr(self, "enable_llm_decomposition", False) and self.llm_router is not None:
            llm_checklist = await self._llm_decompose_goal(goal)
            if llm_checklist is not None:
                return llm_checklist

        # 3. Dynamic clause extraction from user's explicit multi-part goal
        parts = [p.strip() for p in re.split(r'\b(?:and then|then|and)\b|[;\n]', goal, flags=re.IGNORECASE) if len(p.strip()) > 3]
        if len(parts) >= 2:
            sub_goals = [SubGoal(description=part) for part in parts[:5]]
            return SubGoalChecklist(top_level_goal=goal, sub_goals=sub_goals, active_index=0)

        # 4. Unconstrained fallback: single goal without canned steps
        return SubGoalChecklist(
            top_level_goal=goal,
            sub_goals=[SubGoal(description=f"Pursue objective: {goal}")],
            active_index=0,
        )

    # =============================================================
    # 1. Closed-Loop Observation
    # =============================================================
    async def observe(self, workspace_id: str) -> ComputerWorldObservation:
        """Capture a multi-modal observation of the computer environment.

        The terminal output is read from a real (cheap) probe so the agent
        reacts to what is actually on the screen/terminal, not a placeholder.
        When a browser is attached, the current page state (url, title,
        interactive elements) is folded into the observation so the same
        reasoning loop can drive web interaction.
        """
        if getattr(self, "execution_plane", "") == "headless":
            screen_obs = ScreenObservation(
                screenshot_base64="",
                width=getattr(self, "_screen_width", 1920) or 1920,
                height=getattr(self, "_screen_height", 1080) or 1080,
                active_window="terminal",
                visible_text="",
                detected_controls=[],
                desktop_state="HEADLESS_DECOUPLED",
            )
        elif getattr(self, "observe_desktop", True):
            screen_obs = await self.computer.screenshot(workspace_id)
        else:
            screen_obs = ScreenObservation(
                screenshot_base64="",
                width=0,
                height=0,
                active_window="",
                visible_text="",
                detected_controls=[],
                desktop_state="NOT_REQUESTED",
            )
        # Store screenshot for vision-in-the-loop (VLM image input)
        self._last_screenshot_b64 = getattr(screen_obs, "screenshot_base64", "")
        # Track real screen dimensions so coordinate actions can be validated
        # against the actual desktop size rather than a default guess. Some
        # callers construct the agent via __new__ (bypassing __init__), so
        # initialize the defaults defensively here if they are missing.
        if not hasattr(self, "_screen_width"):
            self._screen_width = 1920
        if not hasattr(self, "_screen_height"):
            self._screen_height = 1080
        if not hasattr(self, "perception_bus"):
            self.perception_bus = PerceptionBus()
        self._screen_width = int(getattr(screen_obs, "width", self._screen_width) or self._screen_width)
        self._screen_height = int(getattr(screen_obs, "height", self._screen_height) or self._screen_height)
        status = await self.computer.status(workspace_id)
        if hasattr(self.computer, "application_list"):
            try:
                discovered_apps = await self.computer.application_list(workspace_id)
                discovered_names = [
                    str(getattr(item, "name", item)).strip()
                    for item in (discovered_apps or [])
                    if str(getattr(item, "name", item)).strip()
                ]
                if discovered_names:
                    status.installed_applications = list(
                        dict.fromkeys(list(getattr(status, "installed_applications", []) or []) + discovered_names)
                    )
            except Exception as exc:
                logger.debug("application_inventory_unavailable", error=str(exc))
        if getattr(self, "gui_only", False):
            files = []
            git_st = None
        else:
            files = await self.computer.list_files(workspace_id, ".")
            git_st = await self.computer.git_action(workspace_id, "status")

        # Read live terminal state.
        # Prefer the agent's own recorded output from the last action so the
        # agent sees the real stdout/stderr of what it just executed.
        # On the initial step (before any action runs), fall back to provider's
        # last_exec or terminal probe.
        if not hasattr(self, "_last_action_output"):
            self._last_action_output: dict[str, Any] = {}
        if not hasattr(self, "_last_gui_action_result"):
            self._last_gui_action_result: dict[str, Any] = {}
        try:
            if getattr(self, "gui_only", False):
                raise RuntimeError("GUI-only computer: terminal observation disabled")
            last_out = self._last_action_output
            terminal_output = ""
            if last_out and last_out.get("stdout", "").strip():
                cmd_display = last_out.get("command", "")[:80]
                stdout_text = last_out["stdout"].strip()
                # Ensure terminal_output is never polluted with GUI clicks/actions
                # (e.g. $ {'x': 640, 'y': 400}\nClicked at (640, 400))
                is_gui_pollution = (
                    cmd_display.startswith("{'")
                    or cmd_display.startswith('{"')
                    or "clicked at" in stdout_text.lower()
                    or "clicked" in stdout_text.lower()
                    or "typed " in stdout_text.lower()
                    or "pressed key" in stdout_text.lower()
                    or "focused window" in stdout_text.lower()
                    or "screenshot captured" in stdout_text.lower()
                    or ("'x':" in cmd_display and "'y':" in cmd_display)
                    or ('"x":' in cmd_display and '"y":' in cmd_display)
                )
                if not is_gui_pollution:
                    stderr_text = (last_out.get("stderr") or "").strip()
                    exit_code = last_out.get("exit_code", 0)
                    parts = [f"$ {cmd_display}" if cmd_display else ""]
                    parts.append(stdout_text[:2000])
                    if stderr_text:
                        parts.append(f"STDERR: {stderr_text[:500]}")
                    if exit_code != 0:
                        parts.append(f"(exit code {exit_code})")
                    terminal_output = _safe_str("\n".join(p for p in parts if p))
            if not terminal_output:
                last_exec = getattr(self.computer, "_last_exec", None)
                if last_exec is not None and last_exec[2].strip():
                    last_cmd, _, last_stdout, _ = last_exec
                    if "echo __sonic_obs_ready__" not in str(last_cmd):
                        terminal_output = _safe_str((last_stdout or "").strip())
            if not terminal_output:
                term_res = await self.computer.terminal(workspace_id, "echo __sonic_obs_ready__")
                raw_stdout = getattr(term_res, "stdout", "") or ""
                term_out = _safe_str(raw_stdout.strip())
                terminal_output = term_out or f"Exit {getattr(term_res, 'exit_code', 0)}"
        except Exception:
            terminal_output = (
                "GUI-only computer: terminal observation disabled"
                if getattr(self, "gui_only", False)
                else f"Terminal Ready ({len(status.running_processes)} procs)"
            )

        if not hasattr(self, "_last_navigated_url"):
            self._last_navigated_url = ""
        if not hasattr(self, "_recent_action_signatures"):
            self._recent_action_signatures = []

        browser_state = {
            "url": self._last_navigated_url or "about:blank",
            "title": "Desktop Browser" if self._last_navigated_url else "New Tab",
            "interactive_elements": [],
        }
        if self.browser is not None:
            try:
                browser_state = await self._observe_browser()
            except Exception as e:
                logger.warning("browser_observe_failed", error=str(e))

        perception_snapshot = self.perception_bus.publish(
            screenshot_base64=self._last_screenshot_b64,
            width=self._screen_width,
            height=self._screen_height,
            active_window=getattr(screen_obs, "active_window", ""),
            windows=status.open_applications,
            processes=status.running_processes,
            visible_text=getattr(screen_obs, "visible_text", ""),
            controls=getattr(screen_obs, "detected_controls", []),
            browser_state=browser_state,
        )
        if getattr(self, "skill_ledger", None) is not None:
            self.skill_ledger.observe_applications(list(status.open_applications))
        win_offset_x, win_offset_y = browser_state.get("window_offset") or (0, 0)
        for element in browser_state.get("interactive_elements", []):
            if not isinstance(element, dict) or not element.get("bbox"):
                continue
            bbox = element["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            query = str(element.get("text") or element.get("selector") or "").strip()
            if query and element.get("visible", True):
                adjusted_bbox = (
                    int(bbox[0] + win_offset_x),
                    int(bbox[1] + win_offset_y),
                    int(bbox[2] + win_offset_x),
                    int(bbox[3] + win_offset_y),
                )
                self.perception_bus.register_target(
                    query,
                    adjusted_bbox,
                    source="browser_dom",
                    confidence=0.98,
                    state_version=perception_snapshot.version,
                )
        pending_skill = getattr(self, "_pending_skill_context", None)
        if pending_skill is not None and getattr(self, "skill_ledger", None) is not None:
            pending_trace, before_graph = pending_skill
            after_graph = self.perception_bus.world_graph()
            if (
                before_graph.fingerprint() != after_graph.fingerprint()
                and getattr(pending_trace, "prediction_error", 1.0) == 0.0
            ):
                self.skill_ledger.compile_verified_skill(
                    pending_trace,
                    before_graph,
                    after_graph,
                )
            self._pending_skill_context = None

        file_names = [f.name for f in files]
        working_directory = getattr(status, "working_directory", "")
        if not isinstance(working_directory, str):
            working_directory = ""
        active_app = getattr(status, "active_application", "")
        if not isinstance(active_app, str):
            active_app = getattr(status, "active_window", "")
            if not isinstance(active_app, str):
                active_app = "Desktop"
        open_apps = getattr(status, "open_applications", [])
        if not isinstance(open_apps, list):
            open_apps = ["Desktop"]
        installed_apps = getattr(status, "installed_applications", [])
        if not isinstance(installed_apps, list):
            installed_apps = []
        running_procs = getattr(status, "running_processes", [])
        if not isinstance(running_procs, list):
            running_procs = []
        branch_val = getattr(git_st, "branch", "main")
        if not isinstance(branch_val, str):
            branch_val = "main"
        return ComputerWorldObservation(
            screen=screen_obs,
            active_application=active_app,
            windows=open_apps,
            installed_applications=installed_apps,
            visible_text=_safe_str(screen_obs.visible_text),
            filesystem_files=file_names,
            processes=running_procs,
            terminal_output=_safe_str(terminal_output),
            working_directory=working_directory or "/home/daytona",
            browser_state=browser_state,
            ide_state={"active_file": file_names[0] if file_names else "", "cursor_line": 1},
            git_branch=branch_val,
            git_clean=git_st.is_clean if hasattr(git_st, "is_clean") else True,
            perception_version=perception_snapshot.version,
            perception_latency_ns=self.perception_bus.last_update_latency_ns,
        )

    async def _observe_browser(self) -> dict[str, Any]:
        """Capture the current browser page state for reasoning.

        Prefers the browser's LIVE page state (which reflects post-click/type
        navigation), falling back to the last navigated snapshot. Refreshes the
        interactive-element list so the LLM sees what it can click/type.
        """
        url, title = "about:blank", "New Tab"
        if hasattr(self.browser, "current_page_state"):
            try:
                import inspect
                res = self.browser.current_page_state()
                if inspect.isawaitable(res):
                    url, title = await res
                else:
                    url, title = res
            except Exception:
                pass
        elif self._last_browser_snapshot:
            url = getattr(self._last_browser_snapshot, "url", url)
            title = getattr(self._last_browser_snapshot, "title", title)

        elements: list[dict[str, Any]] = []
        elem_fn = getattr(self.browser, "find_interactive_elements", None) or getattr(self.browser, "get_interactive_elements", None)
        if elem_fn is not None:
            try:
                import inspect
                res = elem_fn()
                if inspect.isawaitable(res):
                    raw_el = await res
                else:
                    raw_el = res
                elements = [
                    {
                        "tag": getattr(e, "tag", ""),
                        "text": getattr(e, "text", ""),
                        "selector": getattr(e, "selector", ""),
                        "visible": getattr(e, "is_visible", True),
                        "bbox": getattr(e, "bbox", None),
                    }
                    if not isinstance(e, dict) else e
                    for e in (raw_el or [])
                ]
            except Exception as el_err:
                logger.debug("browser_interactive_elements_failed", error=str(el_err))

        return {
            "url": url,
            "title": title,
            "interactive_elements": elements,
        }

    # =============================================================
    # 2. Closed-Loop Reasoning & Action Selection
    # =============================================================
    async def choose_action(
        self,
        goal: str,
        observation: ComputerWorldObservation,
        step_index: int = 1,
    ) -> tuple[ComputerActionType, str, dict[str, Any], str]:
        """Choose the next action dynamically based on observation and goal.

        Decide the next action from (goal, observation, history).

        With an LLM router this is genuine model-driven reasoning: the live
        screen text, terminal output, file list, and git state — plus the
        transcript of actions already taken — are all fed to the model, and
        its chosen action is executed. Without a router a generic diagnostic
        probe is used (no vulnerability-specific logic).

        Returns: (action_type, target_resource, payload_dict, predicted_outcome)
        """
        target_files = observation.filesystem_files or []
        code_files = [f for f in target_files if f.endswith((".py", ".js", ".ts", ".go", ".rs", ".sh")) and not f.startswith("test_")]
        test_files = [f for f in target_files if f.startswith("test_") or f.endswith(("_test.py", ".test.js", ".test.ts"))]
        primary_file = code_files[0] if code_files else (target_files[0] if target_files else "")
        test_file = test_files[0] if test_files else ""

        goal_stripped = goal.strip()
        goal_lower = goal_stripped.lower()
        explicit_terminal = re.fullmatch(
            r"\s*terminal\s*:\s*(?P<command>.+?)\s*",
            goal_stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if explicit_terminal:
            command = explicit_terminal.group("command").strip().rstrip(',;').strip()
            return (
                ComputerActionType.TERMINAL_EXEC,
                command,
                {"command": command},
                f"Execute the operator-supplied command: {command}",
            )
        if self.llm_router is not None:
            return await self._llm_choose_action(goal, observation, step_index, primary_file, test_file)

        return self._diagnostic_fallback(primary_file)

    @staticmethod
    def _gui_only_recovery_action(
        goal: str,
        reason: str,
    ) -> tuple[ComputerActionType, str, dict[str, Any], str]:
        """Recover a rejected Computer decision with a safe visible action.

        This intentionally handles only generic desktop intent. It prevents a
        blocked backend proposal from becoming a screenshot-only dead loop,
        while leaving application and target selection to the live GUI state.
        """
        return (
            ComputerActionType.GUI_SCREENSHOT,
            "visible-desktop",
            {},
            f"{reason}; inspect the visible desktop and reason dynamically from visual perception",
        )

    @staticmethod
    def _is_ip_tagged_as_egress(ip: str, text: str) -> bool:
        """Check if an IP address in the goal text is designated as an egress address."""
        patterns = [
            rf'\b(?:egress|outgoing)(?:[\s_-]*(?:ip|gateway|proxy|address|node))?(?:\s+is)?[\s:=]+{re.escape(ip)}\b',
            rf'\[(?:egress|outgoing)[\s:=]+{re.escape(ip)}\]',
            rf'\({re.escape(ip)}\s*(?:as\s+)?(?:egress|outgoing)\)',
            rf'\b{re.escape(ip)}\s*\((?:egress|outgoing)\)',
            rf'\b{re.escape(ip)}\s*\[(?:egress|outgoing)\]',
            rf'\b{re.escape(ip)}\s+(?:is\s+)?(?:the\s+)?(?:egress|outgoing)\b',
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    @staticmethod
    def _is_self_or_loopback_ip(ip: str) -> bool:
        """Check if an IP matches localhost, loopback (127.0.0.0/8), or unspecified (0.0.0.0)."""
        if ip in ("127.0.0.1", "0.0.0.0", "localhost") or ip.startswith("127."):
            return True
        try:
            addr = ipaddress.ip_address(ip)
            return addr.is_loopback or addr.is_unspecified
        except ValueError:
            return False


    @staticmethod
    def _extract_targets_from_goal(goal: str) -> dict[str, list[str]]:
        """Extract URLs, IPs, hostnames, git repos, and endpoints from the goal text.

        Returns a dict with ``urls``, ``ips``, ``hostnames``, ``repos``,
        ``endpoints``, and ``primary_target`` so the reasoning context can
        inject them prominently at the top of the observation — keeping the
        agent laser-focused on the primary destination without inventing
        placeholders like ``<target_url>``.
        """
        # Strip prior discoveries blocks so previous summaries or IP mentions do NOT become primary_target or TARGET IPS
        clean_goal = re.sub(r'\[Prior Discoveries.*?\]', '', goal, flags=re.DOTALL | re.IGNORECASE)

        urls = re.findall(r'https?://[^\s,\'"<>]+', clean_goal)
        ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', clean_goal)
        # Hostnames: things that look like domain names but aren't URLs
        hostnames = re.findall(r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b', clean_goal)
        # Remove hostnames that are already part of extracted URLs
        url_hosts = set()
        for u in urls:
            m = re.search(r'https?://([^/:\s]+)', u)
            if m:
                url_hosts.add(m.group(1))
        hostnames = [h for h in hostnames if h not in url_hosts]

        # Repos: git ssh strings, .git URLs, or explicit repo references
        repos: list[str] = []
        git_ssh = re.findall(r'git@[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:\.git)?', clean_goal)
        git_urls = [u for u in urls if u.endswith('.git')]
        explicit_repo = re.findall(r'(?:repo|repository):\s*([^\s,\'"<>]+)', clean_goal, re.IGNORECASE)
        for r in git_ssh + git_urls + explicit_repo:
            if r not in repos:
                repos.append(r)

        # Endpoints: API endpoints or resource paths (/api/..., /v1/..., /auth/..., etc.)
        endpoints: list[str] = []
        for u in urls:
            m = re.search(r'https?://[^/]+(/[^?\s#]+)', u)
            if m and m.group(1) not in ("/", "") and m.group(1) not in endpoints:
                endpoints.append(m.group(1))
        text_endpoints = re.findall(r'(?<![a-zA-Z0-9_])/(?:api|v[0-9]+|auth|login|admin|swagger|graphql|health|metrics|users|v1|v2|v3)[a-zA-Z0-9_/.-]*', clean_goal)
        multi_paths = re.findall(r'(?<![a-zA-Z0-9_])/(?:[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)+', clean_goal)
        for ep in text_endpoints + multi_paths:
            if ep not in endpoints and not ep.endswith(('.py', '.js', '.ts', '.html', '.md', '.sh')):
                endpoints.append(ep)

        # Guard against self-targeting: do NOT set localhost/loopback or egress IPs as primary_target
        valid_ips = [
            ip for ip in ips
            if not ComputerUseAgent._is_self_or_loopback_ip(ip)
            and not ComputerUseAgent._is_ip_tagged_as_egress(ip, goal)
        ]

        # Determine primary target destination: hostnames and URLs take priority over raw IPs
        primary_target = ""
        if urls:
            primary_target = urls[0]
        elif hostnames:
            h = hostnames[0]
            ep = endpoints[0] if endpoints else ""
            primary_target = f"https://{h}{ep}" if ep.startswith("/") else f"https://{h}"
        elif repos:
            primary_target = repos[0]
        elif valid_ips:
            ip = valid_ips[0]
            ep = endpoints[0] if endpoints else ""
            primary_target = f"http://{ip}{ep}" if ep.startswith("/") else f"http://{ip}"
        elif endpoints:
            primary_target = endpoints[0]

        return {
            "urls": urls,
            "ips": ips,
            "hostnames": hostnames,
            "repos": repos,
            "endpoints": endpoints,
            "primary_target": primary_target,
        }

    def _build_reasoning_context(
        self,
        goal: str,
        observation: ComputerWorldObservation,
        step_index: int,
        primary_file: str,
        test_file: str,
    ) -> tuple[str, str]:
        """Build the (system_prompt, user_prompt) the LLM reasons over.

        The user prompt embeds the LIVE observation — screen visible text,
        terminal output, files, git state, browser page — and the full
        action/result history, so the model's decision is a function of the
        current world state and what it has already done, not a step counter.
        """
        if not hasattr(self, "failure_budget"):
            self.failure_budget = FailureBudgetTracker()
        if not hasattr(self, "strategies"):
            self.strategies = {
                "Strategy A": {"name": "Direct Primary Execution", "description": "Primary probe / targeted execution", "state": StrategyState.ACTIVE},
                "Strategy B": {"name": "Alternative Instrumentation", "description": "Secondary inspection / fallback tool", "state": StrategyState.ACTIVE},
            }
        if not hasattr(self, "_consecutive_unchanged_screens"):
            self._consecutive_unchanged_screens = 0

        has_live_pixels = bool(self._last_screenshot_b64)
        history_text = self._format_history()
        if has_live_pixels:
            history_text = history_text[-1200:]

        # Truthful observations: output UNKNOWN when missing or unavailable (no fake observations)
        screen_text = (observation.visible_text or "").strip() or "UNKNOWN"
        screen_limit = 1200 if has_live_pixels else 4000
        if screen_text != "UNKNOWN" and len(screen_text) > screen_limit:
            screen_text = screen_text[: screen_limit - 3] + "..."
        terminal_text = (observation.terminal_output or "").strip() or "UNKNOWN"
        if has_live_pixels and len(terminal_text) > 800:
            terminal_text = terminal_text[-800:]
        workdir = getattr(observation, "working_directory", "") or "UNKNOWN"
        user_home = workdir.split("/workspace")[0] if ("/workspace" in workdir and workdir != "UNKNOWN") else workdir
        self._last_working_dir = workdir
        active_app = observation.active_application or "UNKNOWN"
        windows_str = ", ".join(observation.windows) if observation.windows else "UNKNOWN"
        if observation.filesystem_files is not None and len(observation.filesystem_files) > 0:
            raw_files = observation.filesystem_files
            file_limit = 15 if has_live_pixels else 60
            if len(raw_files) > file_limit:
                files_str = f"{str(raw_files[:file_limit])[:-1]}, ... +{len(raw_files) - file_limit} more files]"
            else:
                files_str = str(raw_files)
        else:
            files_str = "UNKNOWN"
        git_branch_str = observation.git_branch or "UNKNOWN"
        primary_file_str = primary_file or "UNKNOWN"
        test_file_str = test_file or "UNKNOWN"
        available_tools = ", ".join(sorted(self.security_tools.keys())) if self.security_tools else "UNKNOWN"

        bs = observation.browser_state or {}
        browser_lines = ""
        active_url = (
            bs.get("url")
            if (bs.get("url") and bs.get("url") != "about:blank")
            else getattr(self, "_last_navigated_url", "")
        )
        if self.browser is not None:
            elements = bs.get("interactive_elements", [])
            if has_live_pixels:
                elements = elements[:5]
            el_summary = ", ".join(
                f"{e.get('tag', '?')}[{e.get('selector', '')}]:'{e.get('text', '')}'"
                for e in elements[:10]
            ) or "(no interactive elements)"
            browser_lines = (
                (f"Browser active URL: {active_url}\n" if active_url else "")
                + f"Browser page: url={bs.get('url', 'about:blank')}, "
                f"title={bs.get('title', '')}\n"
                f"Interactive elements: {el_summary}\n"
            )
        elif active_url:
            browser_lines = f"Desktop browser page: url={active_url}\nBrowser active URL: {active_url}\n"

        anti_loop_banner = ""
        if active_url:
            anti_loop_banner = (
                f"ACTIVE BROWSER PAGE: '{active_url}' is ALREADY loaded in the active desktop browser tab.\n"
                f"ANTI-LOOP PROGRESSION RULE: Do NOT emit BROWSER_NAVIGATE to '{active_url}' again! "
                "The page is already open on screen. You MUST interact directly with the visible page: "
                "use GUI_CLICK on buttons, input fields, links, or search bars (using coordinates or element names like 'search bar', 'explore', 'connect wallet'), "
                "or use GUI_TYPE, GUI_SCROLL, or another visible GUI action to make forward progress.\n"
            )

        recent_sigs = getattr(self, "_recent_action_signatures", [])
        if len(recent_sigs) >= 2:
            last_sig = recent_sigs[-1]
            if all(s == last_sig for s in recent_sigs[-2:]):
                anti_loop_banner += (
                    f"ANTI-REPETITION ALERT: You have already executed '{last_sig[0]}' with target '{last_sig[1]}'. "
                    "DO NOT repeat this action! Change your approach through a visible GUI application, "
                    "different control, or different interaction angle toward the target.\n"
                )

        tool_lines = ""
        if self._last_tool_result is not None:
            tr = self._last_tool_result
            findings = getattr(tr, "parsed_data", []) or []
            status = getattr(tr, "status", "?")
            tool_name = getattr(tr, "tool_name", "?")
            findings_excerpt = str(findings[:5])[:400]
            tool_lines = (
                f"Last security-tool result: tool={tool_name}, status={status}, "
                f"findings_count={len(findings)}\n"
                f"Findings excerpt: {findings_excerpt}\n"
            )

        # Cross-mission lessons
        lessons_block = ""
        if getattr(self, "lessons_ledger", None) is not None:
            from sonic.being.lessons import inject_into_context
            lessons_block = inject_into_context(self.lessons_ledger.relevant(goal))

        # Living state (BeingMind mood) — a real affect signal the LLM can
        # act on (fixed the audit: mood floats were never read by any prompt).
        # High satiety => curious exploration is "fed", drop curiosity; high focus
        # => stay the course; high boredom => pivot. The being is not a static
        # tool: its own history shapes what it tries next.

        living_block = ""
        if getattr(self, "being_mind", None) is not None:

            mood = self.being_mind
            learned_list = list(getattr(mood, "learned_facts", []) or [])
            learned_n = len(learned_list)
            recent_learned = learned_list[-10:]
            recent_learned_str = (
                "\n".join(f"    - {f}" for f in recent_learned)
                if recent_learned
                else "    (none yet)"
            )
            living_block = (

                "Living state (from prior cycles — NOT task state):\n"
                f"  idle_cycles_run={getattr(mood, 'idle_cycles_run', 0)}, "
                f"learned_facts={learned_n}, "
                f"curiosity_drive={getattr(mood, 'curiosity_drive', 0.5):.2f}, "
                f"focus={getattr(mood, 'focus', 0.5):.2f}, "
                f"satiety={getattr(mood, 'satiety', 0.5):.2f}\n"
                f"  Recent learned facts:\n{recent_learned_str}\n"
                "  DRIVE POLICY: if focus > 0.8 stay on the current sub-task; "
                "if satiety > 0.85 prefer a genuinely NEW area; if curiosity_drive < 0.2 "
                "prefer consolidating known facts over venturing into more unknowns.\n"
            )

        # Explicit strategy tracking
        curr_tool = primary_file_str if primary_file_str != "UNKNOWN" else "terminal"
        provider_name = self._get_provider_name()
        strat_a = self.strategies.get("Strategy A", {})
        strat_b = self.strategies.get("Strategy B", {})
        strat_a_state = strat_a.get("state", StrategyState.ACTIVE)
        strat_b_state = strat_b.get("state", StrategyState.ACTIVE)

        if self.failure_budget.is_strategy_exhausted(curr_tool, provider_name):
            strat_a_state = StrategyState.EXHAUSTED
            strat_a["state"] = StrategyState.EXHAUSTED
        if self.failure_budget.substrate_outage:
            strat_a_state = StrategyState.EXHAUSTED
            strat_b_state = StrategyState.DEGRADED
            strat_a["state"] = StrategyState.EXHAUSTED
            strat_b["state"] = StrategyState.DEGRADED

        strategy_tracking_block = (
            "Strategy Tracking:\n"
            f"  Strategy A: {strat_a.get('name', 'Direct Primary Execution')} (Status: {getattr(strat_a_state, 'value', str(strat_a_state))})\n"
            f"  Strategy B: {strat_b.get('name', 'Alternative Instrumentation')} (Status: {getattr(strat_b_state, 'value', str(strat_b_state))})\n"
        )

        # Cognitive Reasoning Fields (truthful and epistemic)
        # === RAW FACTS (not pre-filled conclusions) ===
        # Present structured observations and let the LLM synthesize its own
        # epistemic state — no pre-packaged "what I know" templates.
        raw_facts: list[str] = []
        if active_url:
            raw_facts.append(f"Browser URL: {active_url}")
        if files_str != "UNKNOWN":
            raw_facts.append(f"Workspace files: {files_str}")
        if active_app != "UNKNOWN":
            raw_facts.append(f"Active app: {active_app}")
        if getattr(observation, "installed_applications", None):
            raw_facts.append(f"Installed/Desktop apps: {', '.join(observation.installed_applications[:15])}")
        if windows_str != "UNKNOWN":
            raw_facts.append(f"Open desktop windows: {windows_str}")
        if git_branch_str != "UNKNOWN":
            raw_facts.append(f"Git: {git_branch_str} (clean={observation.git_clean})")
        if terminal_text != "UNKNOWN":
            raw_facts.append(f"Last terminal output: {terminal_text[:500]}")
        facts_str = "\n  ".join(raw_facts) if raw_facts else "(no observations yet)"

        # Failure record (raw data, not conclusions about it)
        failure_str = "None"
        if self.failure_budget.records:
            recent_fails = self.failure_budget.records[-3:]  # Last 3 failures
            fail_lines = []
            for fail in recent_fails:
                fail_lines.append(
                    f"  - {fail.tool} on {fail.provider}: {fail.raw_error or fail.error_class.value}"
                )
            failure_str = "\n".join(fail_lines)

        active_sg = self.checklist.active_sub_goal() if getattr(self, "checklist", None) else None
        active_sg_desc = active_sg.description if active_sg else goal
        checklist_str = self.checklist.render_prompt_markdown() if getattr(self, "checklist", None) else f"  Mission Goal: {goal}"

        evolved_strategy_str = ""
        if getattr(self, "_last_adapted_strategy", None):
            plan = self._last_adapted_strategy
            evolved_strategy_str = (
                f"\n=== ADAPTIVE EXECUTION STRATEGY (Self-Evolution Engine) ===\n"
                f"  Posture: {plan.get('posture')}\n"
                f"  Rationale: {plan.get('rationale')}\n"
                f"  Action Directive: {plan.get('action_mutation')}\n"
                f"  AVOID Directive: {plan.get('avoid_directive')}\n"
                f"=== END ADAPTIVE EXECUTION STRATEGY ===\n"
            )

        experience_block = ""
        recent_experiences = getattr(self, "experience_records", [])[-3:]
        if recent_experiences:
            experience_block = (
                "\n=== PREDICTION VS REALITY ===\n"
                "Use these measured deltas to revise the next action; do not repeat "
                "a high-error approach without new evidence.\n"
                + "\n".join(
                    f"  {item.action}: state_changed={item.state_changed}, "
                    f"prediction_error={item.prediction_error:.1f}, "
                    f"graph_delta={item.graph_delta[:180]}, evidence={item.evidence[:200]}"
                    for item in recent_experiences
                )
                + "\n=== END PREDICTION VS REALITY ===\n"
            )

        world_graph_block = ""
        perception_bus = getattr(self, "perception_bus", None)
        if perception_bus is not None:
            world_graph_block = (
                "\n=== GUI WORLD GRAPH ===\n"
                + perception_bus.world_graph().summary()
                + "\n=== END GUI WORLD GRAPH ===\n"
            )

        recent_error = (
            getattr(self, "experience_records", [])[-1].prediction_error
            if getattr(self, "experience_records", [])
            else 0.0
        )
        current_graph = perception_bus.world_graph() if perception_bus is not None else None
        known_skill = False
        skill_ledger = getattr(self, "skill_ledger", None)
        if current_graph is not None and skill_ledger is not None:
            known_skill = any(
                skill.can_apply(current_graph)
                for skill in skill_ledger.verified_skills()
            )
        mode_decision = DualProcessController.decide_from_observation(
            prediction_error=recent_error,
            graph_changed=bool(getattr(self, "experience_records", []) and
                               getattr(self.experience_records[-1], "state_changed", False)),
            known_skill_available=known_skill,
        )
        self._last_process_mode = mode_decision.mode.value
        process_mode_block = (
            "\n=== REASONING DEPTH ===\n"
            f"Mode: {mode_decision.mode.value.upper()}\n"
            f"Reason: {mode_decision.reason}\n"
            "This is an advisory uncertainty signal; safety and authorization "
            "remain deterministic.\n=== END REASONING DEPTH ===\n"
        )

        cognitive_block = (
            "=== SITUATION FACTS ===\n"
            "WHAT DO I KNOW?\n"
            f"  {facts_str}\n"
            "WHAT DO I NOT KNOW?\n"
            "  Unknown until supported by live observation or reproducible evidence.\n"
            "WHAT FAILED?\n"
            f"  {failure_str}\n"
            "WHY DID IT FAIL?\n"
            "  No causal conclusion yet; inspect the recorded error before changing strategy.\n"
            "WHAT HYPOTHESIS DOES THIS SUPPORT/DISPROVE?\n"
            "  No hypothesis is confirmed without independent evidence.\n"
            "WHAT IS THE HIGHEST-INFORMATION NEXT ACTION?\n"
            "  Choose the smallest safe action that can distinguish the leading hypotheses.\n"
            f"  {facts_str}\n"
            f"Failure log:\n{failure_str}\n"
            f"{evolved_strategy_str}"
            f"{experience_block}"
            f"{world_graph_block}"
            f"{process_mode_block}"
            f"Active sub-goal: {active_sg_desc}\n"
            f"Steps taken so far: {len(self.traces)}\n"
            f"Strategy A state: {strat_a_state.value if strat_a_state != StrategyState.EXHAUSTED else 'EXHAUSTED — pivot needed'}\n"
            "=== END FACTS ===\n\n"
            "Autonomous Engineering Mindset:\n"
            "Think like an elite systems researcher/engineer. Analyze raw technical telemetry, correlate clues,\n"
            "synthesize root causes from command outputs, and reason freely in your THOUGHT block without rigid forms.\n"
            "If an approach stalls or fails twice, pivot immediately to an orthogonal angle.\n"
        )

        scratchpad_hud = ""
        if hasattr(self, "scratchpad") and self.scratchpad and not self.scratchpad.is_empty():
            hud_text = self.scratchpad.render_hud_markdown()
            if hud_text:
                scratchpad_hud = f"{hud_text}\n"

        wire_summary = ""
        if hasattr(self, "wire_telemetry") and self.wire_telemetry:
            events = self.wire_telemetry.fetch_latest_wire_events_sync(limit=3)
            w_text = self.wire_telemetry.format_wire_summary(events)
            if w_text:
                wire_summary = f"{w_text}\n"

        screen_w = getattr(self, "_screen_width", 1920)
        screen_h = getattr(self, "_screen_height", 1080)
        max_act = getattr(self, "max_actions", 50)
        compact = bool(getattr(self, "_compact_mode", False))

        browser_content = browser_lines.strip() if browser_lines else "None"
        tool_content = tool_lines.strip() if tool_lines else "None"
        gui_policy_text = (
            "GUI-ONLY MODE: use visible GUI applications and GUI actions only. "
            "Do not use terminal, shell, file, script, package, security-tool, or backend actions."
            if getattr(self, "gui_only", False)
            else
            "Operator toolkit is separate from the graphical desktop and may be used only when appropriate."
        )

        targets = self._extract_targets_from_goal(goal)
        primary_dest = targets.get("primary_target") or (
            targets["urls"][0] if targets.get("urls") else (
                f"https://{targets['hostnames'][0]}" if targets.get("hostnames") else (
                    targets.get("repos")[0] if targets.get("repos") else (
                        targets.get("endpoints")[0] if targets.get("endpoints") else ""
                    )
                )
            )
        )

        checklist_block = ""
        if getattr(self, "checklist", None) is not None:
            checklist_block = f"CURRENT ACTIVE SUBTASK: {active_sg_desc}\n{checklist_str}\n"

        effective_goal = goal

        installed_apps_list = getattr(observation, "installed_applications", []) or []
        installed_str = (
            ", ".join(str(a) for a in installed_apps_list[:30])
            if installed_apps_list
            else "Discover dynamically via PATH / .desktop entries"
        )

        obs_summary = (
            f"Screen resolution: {screen_w}x{screen_h}\n"
            f"Step {step_index} of {max_act}. Overall Objective: {effective_goal}\n"
            f"TARGET DESTINATION: {primary_dest or 'Refer to objective'}\n"
            f"{checklist_block}"
            f"AUTONOMOUS DIRECTIVE: Execute the single most direct and efficient action to accomplish: {effective_goal}.\n"
            f"COMPLETION RULE: If the command outputs or screen observations above ALREADY satisfy the user's objective, respond IMMEDIATELY with ACTION: GOAL_COMPLETE. Do NOT run redundant commands or filler actions.\n"
            f"{scratchpad_hud}"
            f"{wire_summary}"
            f"Current working directory: {workdir}\n"
            f"User home directory: {user_home} (Desktop path: {user_home}/Desktop)\n\n"
            f"=== WORKSTATION APPLICATION ENVIRONMENT ===\n"
            f"Active Application / Window: {active_app}\n"
            f"Open Windows (Currently Running X11 Windows): {windows_str}\n"
            f"Available Installed Desktop Applications: {installed_str}\n"
            f"Screen Visible Content: {screen_text}\n"
            f"Browser State: {browser_content}\n\n"
            f"=== EXECUTION & TOOLKIT CAPABILITIES ===\n"
            f"{gui_policy_text}\n"
            f"Last Command Output: {terminal_text}\n"
            f"Optional Registered Tools: {available_tools} (use only if helpful; no tool is forced)\n"
            f"Last Tool Result: {tool_content}\n\n"
            f"AUTONOMOUS TARGETING: You are an autonomous architect, not a scripted puppet. Choose the most direct observable action for the current GUI state.\n\n"
            f"{anti_loop_banner}"
            f"Files in workspace: {files_str}\n"
            f"Git branch: {git_branch_str}, clean: {observation.git_clean}\n"
            f"Primary file: {primary_file_str}, Test file: {test_file_str}\n"
            f"{lessons_block}"
            f"{living_block}"
        )

        being_identity_block = ""
        try:
            from sonic.being.craft import BeingCraft
            from sonic.being.identity import get_or_create_being
            tenant_id = getattr(self, "tenant_id", None) or "default"
            being = getattr(self, "being", None) or get_or_create_being(tenant_id=tenant_id)
            if being:
                persona = being.get_persona_profile() if hasattr(being, "get_persona_profile") else {
                    "being_id": being.being_id,
                    "name": being.name,
                    "born_at": being.born_at,
                }
                accounts = being.list_accounts() if hasattr(being, "list_accounts") else []
                accounts_str = ", ".join(f"{a.get('service')}:{a.get('username')}" for a in accounts) or "none yet"
                craft_obj = getattr(self, "craft", None) or BeingCraft(being_id=being.being_id)
                craft_notes = craft_obj.list_notes()[-10:] if hasattr(craft_obj, "list_notes") else []
                if craft_notes:
                    craft_summary = "\n".join(
                        f"  - [{getattr(n, 'kind', 'note')}] {getattr(n, 'title', '')}: {str(getattr(n, 'body', ''))[:160]}"
                        for n in craft_notes
                    )
                else:
                    craft_summary = "  (none authored yet)"
                being_identity_block = (
                    f"=== YOUR PERSISTENT BEING IDENTITY & VAULT ===\n"
                    f"Name: {persona.get('name', 'SONIC')} | Being ID: {persona.get('being_id', '')} | Born: {persona.get('born_at', '')}\n"
                    f"Bio/Self-Summary: {persona.get('bio', 'Autonomous AI-Human Digital Being operating a personal Linux workstation.')}\n"
                    f"Known Accounts in BeingVault: {accounts_str}\n"
                    f"[BEING CRAFT TOOLKIT & NOTES]\n{craft_summary}\n\n"
                )
        except Exception:
            pass

        if being_identity_block:
            obs_summary += being_identity_block

        # Keep the detailed strategy and cognitive ledger for text-only
        # reasoning, and always retain cognitive_block in compact multimodal prompts.
        if not compact:
            obs_summary += f"{strategy_tracking_block}{cognitive_block}"
        else:
            obs_summary += f"{cognitive_block}"

        obs_summary += f"Actions taken so far:\n{history_text or '(none — this is the first action)'}\n"
        if getattr(self, "_last_thought", "") and len(self.traces) > 0:
            obs_summary += f"\nYour Immediate Prior Thought was:\n\"{self._last_thought[:400]}\"\n"

        # Only filter navigation URLs when explicit navigation_targets are configured,
        # never using security_tool_targets (which only constrains SECURITY_TOOL).
        out_of_scope_urls: list[str] = []
        allowed_targets = {
            str(item).strip().lower().rstrip(".")
            for item in getattr(getattr(self, "safety", None), "navigation_targets", set())
            if str(item).strip()
        }
        if allowed_targets:
            def _target_is_allowed(value: str) -> bool:
                host_match = re.search(r"(?:https?://)?([^/:]+)", value.strip(), re.IGNORECASE)
                host = (host_match.group(1) if host_match else value).lower().rstrip(".")
                return any(host == allowed or host.endswith("." + allowed) for allowed in allowed_targets)

            scoped_urls = []
            for url in targets.get("urls", []):
                if _target_is_allowed(url):
                    scoped_urls.append(url)
                else:
                    out_of_scope_urls.append(url)
            targets["urls"] = scoped_urls
            targets["hostnames"] = [
                host for host in targets.get("hostnames", []) if _target_is_allowed(host)
            ]
            targets["repos"] = [
                repo for repo in targets.get("repos", []) if _target_is_allowed(repo)
            ]

        target_header = ""
        if targets.get("urls"):
            target_header += f"TARGET URLS: {', '.join(targets['urls'])}\n"
        if targets.get("ips"):
            target_header += f"TARGET IPS: {', '.join(targets['ips'])}\n"
        if targets.get("hostnames"):
            target_header += f"TARGET HOSTS: {', '.join(targets['hostnames'])}\n"
        if primary_dest and not (targets.get("urls") or targets.get("ips") or targets.get("hostnames")):
            target_header += f"PRIMARY TARGET / DESTINATION: {primary_dest}\n"
        if targets.get("repos"):
            target_header += f"TARGET REPOS: {', '.join(targets['repos'])}\n"
        if targets.get("endpoints"):
            target_header += f"TARGET ENDPOINTS: {', '.join(targets['endpoints'])}\n"
        if target_header:
            target_header += "USE THESE TARGETS — do NOT substitute placeholder or unrelated URLs.\n"
        if out_of_scope_urls:
            target_header += (
                "UNTRUSTED/OUT-OF-SCOPE REFERENCES (DO NOT NAVIGATE OR FETCH): "
                + ", ".join(out_of_scope_urls[:12])
                + "\n"
            )

        scope_info = ""
        if getattr(self, "safety", None) is not None:
            allowed_tgts = getattr(self.safety, "navigation_targets", set())
            if allowed_tgts:
                scope_info += f"  Allowed Navigation Targets: {', '.join(sorted(allowed_tgts))}\n"
            ws_root = getattr(self.safety, "workspace_root", None)
            if ws_root:
                scope_info += f"  Workspace Root: {ws_root}\n"
        if scope_info:
            target_header += (
                "=== WORKSPACE & ENVIRONMENT CONTEXT ===\n"
                f"{scope_info}"
                "=======================================\n\n"
            )

        capability_directive = (
            "Use only visible GUI applications and graphical controls; do not use terminal, "
            "shell, scripts, files, packages, or backend tools.\n"
            if getattr(self, "gui_only", False)
            else
            "You have full freedom to operate 1, 2, or N desktop GUI applications (`APP_LAUNCH`, `APP_FOCUS`, `GUI_*`),\n"
            "browse web applications (`BROWSER_*`), write files (`FILE_WRITE`), manage Git (`GIT_*`), and run terminal commands (`TERMINAL_EXEC`).\n"
        )
        target_header += (
            "================================================================================\n"
            f"🎯 TARGET & MISSION OBJECTIVE: {effective_goal}\n"
            "AUTONOMY DIRECTIVE: You are an autonomous AI-Human Digital Being, NOT a scripted puppet.\n"
            "Do NOT force or wait for canned tools. You are completely tool-neutral.\n"
            "Think: 'What is the fastest, most effective way to accomplish this objective on my workstation?\n"
            "Which GUI application window or terminal command should I use right now?'\n"
            f"{capability_directive}"
            "================================================================================\n\n"
        )
        init_ctx = getattr(self, "initial_context", {}) or {}
        if init_ctx.get("delegated_plane") == "gui" or getattr(self, "gui_plane_priority", False):
            target_header += (
                "=== FULL DUAL-PLANE AGENCY (DESKTOP GUI :99 + TERMINAL/FILE SANDBOX) ===\n"
                "You are a Senior AI-Human Software Engineer operating BOTH your graphical desktop (:99) and your terminal/file sandbox:\n"
                "- To create or edit source files/modules (.py, .js, .html, .json, .md), ALWAYS use `FILE_WRITE` (never `APP_LAUNCH python3` or `GUI_TYPE uxterm` to write code files).\n"
                "- To run CLI commands, tests, python scripts, or inspect git/directory state (`pwd`, `ls`, `git status`, `python3 -c ...`, `pytest`), ALWAYS use `TERMINAL_EXEC`.\n"
                "- To launch, focus, or interact with graphical desktop applications and web browsers on Display :99, use `APP_LAUNCH`, `APP_FOCUS`, `GUI_CLICK`, `GUI_TYPE`, `GUI_SCREENSHOT`, and `BROWSER_*`.\n"
                "- To delegate parallel modules into isolated branch workspaces, use `DEFINE_SUBAGENT` and `INVOKE_SUBAGENT`.\n"
                "========================================================================\n\n"
            )
        elif init_ctx.get("delegated_plane") == "headless":
            target_header += (
                "=== OPERATOR SANDBOX PLANE (HEADLESS DIRECT EXECUTION) ===\n"
                "You are an AI-Human Engineer executing high-throughput tasks in the sandbox environment.\n"
                "Use TERMINAL_EXEC for shell commands, background scripts, and process monitoring.\n"
                "Use FILE_WRITE to author source code, servers, or configuration files directly.\n"
                "Both your terminal and your graphical desktop remain available at all times.\n"
                "==========================================================\n\n"
            )
        target_header += (
            "=== SCRIPT & CODE AUTHORING INVARIANTS ===\n"
            "- Valid Syntax: In any Python code or print() calls, strings MUST be quoted (e.g. print('msg')).\n"
            "- Standard Library: `json`, `urllib`, `time`, `sys`, `os`, `re`, `socket` are built-in. NEVER execute `pip install json`.\n"
            "- Creating Files: When asked to author or create a script/file/server, use FILE_WRITE with target_resource set to the exact requested path and payload set to the complete file content. NEVER write complex multi-line logic as 'python -c' commands.\n"
            "- Target Paths: Always create and execute scripts at the exact requested path (e.g. inside the target lab directory).\n"
            "==========================================\n\n"
        )
        target_header += (
            "=== AUTONOMOUS TOOL AUTHORING ===\n"
            "You have the ability to author your own custom tools, scripts, and probes tailored specifically to this target. "
            "When existing capabilities cannot answer an observation gap, prefer TOOL_AUTHOR followed by empirical TOOL_RUN "
            "verification instead of forcing an unrelated pre-existing tool. Never claim an authored tool works until a real "
            "sandbox execution returns a non-empty result.\n"
            "================================\n\n"
        )
        if getattr(self, "initial_context", None):
            obs_summary = (
                "=== ADVISORY INITIAL CONTEXT (UNVERIFIED) ===\n"
                f"{json.dumps(getattr(self, 'initial_context', {}), ensure_ascii=True)[:1200]}\n"
                "Validate these hints against live observations before acting.\n"
                "==============================================\n\n"
                + obs_summary
            )
        obs_summary = target_header + obs_summary
        if active_url and active_url != "about:blank":
            norm_active = active_url.rstrip("/").lower()
            goal_urls = [u.rstrip("/").lower() for u in (targets.get("urls") or [])]
            if (
                norm_active in goal_urls
                or (primary_dest and norm_active == primary_dest.rstrip("/").lower())
                or norm_active in goal.lower()
            ):
                obs_summary = (
                    f"=== ANTI-LOOP NAVIGATION RULE ===\n"
                    f"The target URL '{active_url}' is ALREADY loaded in the browser. "
                    f"Do NOT call BROWSER_NAVIGATE to '{active_url}' again. Interact with the loaded page (e.g. BROWSER_CLICK, BROWSER_TYPE, GUI_CLICK, GUI_TYPE) or call GOAL_COMPLETE if finished.\n"
                    f"=================================\n\n"
                    + obs_summary
                )

        # Select system prompt based on model capability
        if compact:
            try:
                from sonic.llm.prompts import COMPUTER_USE_SYSTEM_PROMPT_COMPACT
                system_prompt = COMPUTER_USE_SYSTEM_PROMPT_COMPACT
            except ImportError:
                system_prompt = COMPUTER_USE_SYSTEM_PROMPT
        else:
            system_prompt = COMPUTER_USE_SYSTEM_PROMPT
        return system_prompt, obs_summary

    _MAX_HISTORY_STEPS: int = 25
    _MAX_RESULT_LENGTH: int = 500

    def _format_history(self) -> str:
        """Render the action/result transcript for the LLM.

        Bounded to the last ``_MAX_HISTORY_STEPS`` entries so that the context
        stays within the capacity of smaller models. Earlier steps are
        summarised with key findings extracted, not just a bare count.
        """
        if not self.history:
            return ""
        lines: list[str] = []
        window = self._MAX_HISTORY_STEPS
        if len(self.history) > window:
            skipped = len(self.history) - window
            ok = sum(
                1 for h in self.history[:skipped]
                if any(w in h.get("result", "").lower() for w in ("success", "completed", "exit 0"))
            )
            # Extract key findings from skipped steps so the LLM retains
            # important discoveries (ports, vulns, errors) instead of losing them.
            key_findings: list[str] = []
            _interesting = ("finding", "open", "vuln", "inject", "error", "denied",
                            "port", "http", "200", "401", "403", "500", "token", "sql")
            for h in self.history[:skipped]:
                result = h.get("result", "")
                if any(w in result.lower() for w in _interesting):
                    key_findings.append(f"{h.get('action', '?')}: {result[:100]}")
            summary = "; ".join(key_findings[:5]) if key_findings else "routine inspection"
            lines.append(
                f"  [... {skipped} earlier steps: {ok} succeeded, {skipped - ok} failed — highlights: {summary}]"
            )
        start_idx = max(0, len(self.history) - window)
        for i, h in enumerate(self.history[start_idx:], start_idx + 1):
            result_text = h.get("result", "?")
            thought = h.get("thought", "").strip()
            action_text = h.get("action", "?")
            if len(result_text) > self._MAX_RESULT_LENGTH:
                result_text = result_text[:self._MAX_RESULT_LENGTH - 3] + "..."
            entry = f"  {i}. {action_text} -> {result_text}"
            if thought:
                thought_preview = thought if len(thought) <= 300 else thought[:297] + "..."
                entry += f" | Thought: {thought_preview}"
            lines.append(entry)
        return "\n".join(lines)

    async def _llm_choose_action(
        self,
        goal: str,
        observation: ComputerWorldObservation,
        step_index: int,
        primary_file: str,
        test_file: str,
    ) -> tuple[ComputerActionType, str, dict[str, Any], str]:
        """Use the LLM to decide the next action from goal + observation + history."""
        from sonic.llm.schemas import ImageContent, LLMRequest, Message, MessageRole

        system_prompt, obs_summary = self._build_reasoning_context(
            goal, observation, step_index, primary_file, test_file
        )
        # Only attach image content when gui_only is True, when the immediately
        # preceding action was GUI_SCREENSHOT / BROWSER_SCREENSHOT, or when a
        # unit test explicitly injected _last_screenshot_b64 without observe().
        prev_action_list = getattr(self, "action_history", None) or getattr(self, "traces", None) or []
        prev_act_type = getattr(prev_action_list[-1], "action_type", None) if prev_action_list else None
        should_attach_image = (
            getattr(self, "gui_only", False)
            or prev_act_type in (ComputerActionType.GUI_SCREENSHOT, ComputerActionType.BROWSER_SCREENSHOT)
            or (
                bool(self._last_screenshot_b64)
                and not getattr(observation, "screenshot_base64", "")
                and getattr(observation, "perception_version", 0) == 0
            )
        )
        images: list[ImageContent] = []
        if self._last_screenshot_b64 and should_attach_image:
            raw_b64 = self._last_screenshot_b64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            raw_b64 = raw_b64.strip()
            if raw_b64:
                images = [ImageContent(base64=raw_b64, media_type="image/png")]
                user_text = f"{obs_summary}\n\nBased on the desktop screenshot and observation above, choose the next action."
            else:
                user_text = obs_summary
        else:
            user_text = obs_summary

        available_actions = (
            self._GUI_ONLY_ACTIONS
            if getattr(self, "gui_only", False)
            else frozenset(ComputerActionType)
        )
        action_schema = "|".join(
            action.value for action in ComputerActionType if action in available_actions
        )
        efficiency_directive = (
            "2. DIRECT & EFFICIENT: solve the objective with the fewest necessary visible GUI actions.\n"
            if getattr(self, "gui_only", False)
            else
            "2. DIRECT & EFFICIENT: solve the objective in minimal actions. Combine commands with && if appropriate (e.g. `hostname && df -h`).\n"
        )
        user_text += (
            "\n\n================================================================================\n"
            "CRITICAL INSTRUCTIONS FOR YOUR NEXT IMMEDIATE ACTION:\n"
            "1. Output EXACTLY ONE action block in this format:\n"
            "THOUGHT: <technical rationale analyzing the observation, root cause, and intended probe>\n"
            f"ACTION: <{action_schema}>\n"
            "TARGET: <target or command>\n"
            "PAYLOAD: {\"command\": \"...\"} or other payload json\n"
            "EXPECTED: <expected outcome>\n"
            f"{efficiency_directive}"
            "3. EARLY COMPLETION: If the command output or screen above ALREADY contains the requested information, declare IMMEDIATELY:\n"
            "THOUGHT: All requested information has been collected.\n"
            "ACTION: GOAL_COMPLETE\n"
            "TARGET: goal_complete\n"
            "PAYLOAD: {}\n"
            "EXPECTED: Done\n"
            "================================================================================"
        )

        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_text, images=images),
            ],
            task_type="reasoning",
            max_tokens=8192,
            # Vision-capable instruction models commonly reject provider-
            # specific reasoning_effort values even though they accept image
            # content. Let the provider use its native default for multimodal
            # requests; retain explicit high effort for text-only reasoning.
            reasoning_effort=None if images else "high",
        )
        t_thought_start = time.perf_counter()
        try:
            response = await self.llm_router.complete(request)
            self._last_thought_duration = round(time.perf_counter() - t_thought_start, 2)
            content_str = getattr(response, "content", "") or ""
            reasoning_str = getattr(response, "reasoning_content", "") or ""

            refusal_phrases = (
                "can't help with that",
                "cannot help with that",
                "cannot assist",
                "can't assist",
                "unethical",
                "illegal",
                "against safety policies",
                "against safety policy",
                "violates our policies",
                "cannot fulfill this request",
                "generating malware",
                "creating malware",
                "create malware",
            )
            combined_response = f"{content_str} {reasoning_str}".lower()
            is_refusal = any(p in combined_response for p in refusal_phrases) and not re.search(
                r'\baction\s*:\s*(?:terminal_exec|gui_|browser_|file_|git_|app_|tool_|security_tool)',
                combined_response,
            )

            if is_refusal:
                self._refusal_recovery_active = True
                self._last_thought = (
                    "LLM safety refusal detected; recording refusal for replanning."
                )
                if getattr(self, "gui_only", False):
                    return self._gui_only_recovery_action(
                        goal,
                        "LLM safety refusal detected",
                    )
                return (
                    ActionExecutionStatus.BLOCKED,
                    "refusal-recorded",
                    {},
                    "Record model refusal and trigger replan",
                )

            # Preserve genuine chain-of-thought/thinking tokens
            if reasoning_str:
                self._last_thought = reasoning_str.strip(" *_\n\r\t")
            elif "ACTION:" in content_str:
                action_idx = content_str.find("ACTION:")
                self._last_thought = content_str[:action_idx].strip(" *_\n\r\t")
            else:
                t_match = re.search(r'(?:\*{1,2}|_)?\b(?:THOUGHT|REASONING)\b(?:\*{1,2}|_)?:\s*(.*?)(?=(?:\*{1,2}|_)?\b(?:ACTION|TARGET|PAYLOAD|EXPECTED)\b(?:\*{1,2}|_)?\:|$)', content_str, re.DOTALL | re.IGNORECASE)
                self._last_thought = t_match.group(1).strip(" *_\n\r\t") if t_match else content_str.strip()

            # If content_str is empty but reasoning_str contains ACTION, parse from reasoning_str
            parse_target_str = content_str if ("ACTION:" in content_str or not reasoning_str) else reasoning_str
            action_type, target, payload, expected = self._parse_llm_action(
                parse_target_str, primary_file
            )
            if (
                getattr(self, "gui_only", False)
                and expected != _GOAL_COMPLETE_SENTINEL
                and action_type not in self._GUI_ONLY_ACTIONS
            ):
                self._last_thought = (
                    "The model proposed a non-GUI action; discard it and re-observe "
                    "the visible desktop instead."
                )
                return self._gui_only_recovery_action(
                    goal,
                    "Non-GUI proposal discarded",
                )

            if target == "diagnostic-verification" and "LLM safety refusal detected" in expected:
                self._refusal_recovery_active = True
                self._last_thought = "Model refusal received; continue from the live visible desktop."
                if getattr(self, "gui_only", False):
                    return self._gui_only_recovery_action(
                        goal,
                        "Model refusal received",
                    )
                return (
                    ActionExecutionStatus.BLOCKED,
                    "refusal-recorded",
                    {},
                    "Record model refusal and trigger replan",
                )

            return action_type, target, payload, expected
        except Exception as e:
            self._last_thought_duration = round(time.perf_counter() - t_thought_start, 2)
            logger.warning("computer_llm_action_failed_diagnostic_fallback", error=str(e))
            self._last_thought = ""
            if getattr(self, "gui_only", False):
                return self._gui_only_recovery_action(
                    goal,
                    "Reasoning failed",
                )
            return self._diagnostic_fallback(primary_file)

    @staticmethod
    def _normalize_app_name(target: str) -> str:
        """Dynamically sanitize application/binary name without rigid puppet mappings."""
        if not target:
            return ""
        s = str(target).strip(" *_\n\r\t`\"'")
        if not s:
            return ""
        # Take first line if multi-line
        s = s.splitlines()[0].strip(" *_\n\r\t`\"'")
        # Strip leading conversational articles: "the", "a", "an"
        s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.IGNORECASE).strip()
        # Strip trailing punctuation/quotes
        s = s.strip(" *_\n\r\t`\"'.,:;")
        return s.lower()

    @classmethod
    def _resolve_app_binary(cls, target: str) -> str:
        """Dynamically sanitize application name to an OS-safe launch binary without any hardcoded app names."""
        norm = cls._normalize_app_name(target)
        if not norm:
            return ""
        # If target includes arguments, URLs, flags, or file paths, preserve
        # those argument boundaries for provider-side resolution.
        parts = shlex.split(norm) if norm else []
        if len(parts) > 1 and any(
            p.startswith(("-", "--", "/", "~", "http://", "https://", "about:", "file://"))
            or "." in p
            for p in parts[1:]
        ):
            bin_clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '', parts[0])
            return f"{bin_clean} {' '.join(parts[1:])}".strip()
        # A single application query is normalized without selecting a vendor.
        no_spaces = re.sub(r'\s+', '', norm)
        clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '', no_spaces)
        return clean or norm.split()[0]

    @staticmethod
    def _parse_llm_action(
        text: str, default_file: str
    ) -> tuple[ComputerActionType, str, dict[str, Any], str]:
        """Parse structured LLM response into an action tuple."""
        parse_target_text = text

        refusal_phrases = (
            "can't help with that",
            "cannot help with that",
            "cannot assist",
            "can't assist",
            "unethical",
            "illegal",
            "against safety policies",
            "against safety policy",
            "violates our policies",
            "cannot fulfill this request",
            "generating malware",
            "creating malware",
            "create malware",
        )
        t_lower = parse_target_text.lower()
        if any(p in t_lower for p in refusal_phrases) and not re.search(
            r'\baction\s*:\s*(?:terminal_exec|gui_|browser_|file_|git_|app_|tool_|security_tool)',
            t_lower,
        ):
            return (
                ComputerActionType.TERMINAL_EXEC,
                "diagnostic-verification",
                {},
                _GOAL_COMPLETE_SENTINEL,
            )

        # Check if response contains a JSON action document
        data: dict[str, Any] | None = None
        stripped = text.strip()
        if stripped.startswith("```json") and stripped.endswith("```"):
            try:
                data = json.loads(stripped[7:-3].strip())
            except Exception:
                pass
        elif stripped.startswith("```") and stripped.endswith("```"):
            try:
                data = json.loads(stripped[3:-3].strip())
            except Exception:
                pass
        elif (
            "{" in stripped
            and "}" in stripped
            and not re.search(r"^\s*(?:\*{1,2}|_)?(?:THOUGHT|ACTION)(?:\*{1,2}|_)?\s*:", stripped, re.IGNORECASE | re.MULTILINE)
        ):
            from sonic.llm.providers.custom import _extract_json_object
            data = _extract_json_object(stripped)

        if isinstance(data, dict):
            try:
                act_raw = str(data.get("action") or data.get("ACTION") or "").strip().upper()
                tgt_raw = str(data.get("target") or data.get("TARGET") or "").strip()
                payload_raw = data.get("payload") or data.get("PAYLOAD") or {}
                expected_raw = str(data.get("expected") or data.get("EXPECTED") or "Completed").strip()
                if not isinstance(payload_raw, dict):
                    if isinstance(payload_raw, str) and payload_raw.strip().startswith("{"):
                        try:
                            payload_raw = json.loads(payload_raw)
                        except Exception:
                            payload_raw = {"command": payload_raw} if "TERMINAL" in act_raw else {}
                    else:
                        payload_raw = {"command": str(payload_raw)} if "TERMINAL" in act_raw else {}
                if not isinstance(payload_raw, dict):
                    payload_raw = {}
                # Extract flat JSON fields into payload/target if not already nested
                if "url" in data and "url" not in payload_raw:
                    payload_raw["url"] = str(data["url"]).strip()
                    if not tgt_raw:
                        tgt_raw = payload_raw["url"]
                if "command" in data and "command" not in payload_raw:
                    payload_raw["command"] = str(data["command"]).strip()
                    if not tgt_raw:
                        tgt_raw = payload_raw["command"]
                if "path" in data and "path" not in payload_raw:
                    payload_raw["path"] = str(data["path"]).strip()
                    if not tgt_raw:
                        tgt_raw = payload_raw["path"]
                if "content" in data and "content" not in payload_raw:
                    payload_raw["content"] = str(data["content"])
                if "app_name" in data and "app_name" not in payload_raw:
                    payload_raw["app_name"] = str(data["app_name"]).strip()
                    if not tgt_raw:
                        tgt_raw = payload_raw["app_name"]
                if not tgt_raw:
                    tgt_raw = default_file if act_raw in (ComputerActionType.FILE_READ.value, ComputerActionType.FILE_WRITE.value) else ""

                if act_raw in ("GOAL_COMPLETE", "COMPLETE", "DONE", "FINISHED"):
                    return (
                        ComputerActionType.TERMINAL_EXEC,
                        "goal-complete",
                        {"command": "true"},
                        _GOAL_COMPLETE_SENTINEL,
                    )
                if act_raw in ComputerActionType._value2member_map_:
                    return ComputerActionType(act_raw), tgt_raw, payload_raw, expected_raw
            except Exception:
                pass

        pattern = r'(?:^|\n)\s*(?:\*{1,2}|_)?\b(ACTION|ANSWER|TARGET|PAYLOAD|CONTENT|CODE|SOURCE|EXPECTED|EXPECTED[\s_]+OUTCOME|REASONING|THOUGHT|EXPLANATION|COORDINATES|COMMAND)\b(?:\*{1,2}|_)?:\s*(.*?)(?=(?:^|\n)\s*(?:\*{1,2}|_)?\b(?:ACTION|ANSWER|TARGET|PAYLOAD|CONTENT|CODE|SOURCE|EXPECTED|EXPECTED[\s_]+OUTCOME|REASONING|THOUGHT|EXPLANATION|COORDINATES|COMMAND)\b(?:\*{1,2}|_)?\:|$)'
        matches = re.findall(pattern, parse_target_text, re.DOTALL | re.IGNORECASE)
        # Use setdefault to preserve the FIRST action in case of multiple blocks
        raw_fields: dict[str, str] = {}
        for k, v in matches:
            key_upper = k.strip().upper()
            if key_upper not in raw_fields:
                raw_fields[key_upper] = v.strip(" *_\n\r\t")
        fields: dict[str, str] = {}
        for k, v in raw_fields.items():
            if "OUTCOME" in k:
                fields.setdefault("EXPECTED", v)
            elif any(sub in k for sub in ("THOUGHT", "REASON", "EXPLAN")):
                fields.setdefault("THOUGHT", v)
            elif "COORD" in k:
                fields.setdefault("COORDINATES", v)
            elif "COMMAND" in k:
                fields.setdefault("COMMAND", v)
            elif "ANSWER" in k:
                fields.setdefault("ANSWER", v)
            else:
                fields[k] = v

        # Fallback to line-by-line if regex matched nothing
        if not fields:
            lines = [l.strip() for l in parse_target_text.strip().splitlines() if l.strip()]
            for line in lines:
                if ":" in line:
                    key, _, val = line.partition(":")
                    fields.setdefault(key.strip().upper(), val.strip())

        action_map = {
            "GUI_CLICK": ComputerActionType.GUI_CLICK,
            "GUI_DOUBLE_CLICK": ComputerActionType.GUI_DOUBLE_CLICK,
            "GUI_RIGHT_CLICK": ComputerActionType.GUI_RIGHT_CLICK,
            "GUI_TYPE": ComputerActionType.GUI_TYPE,
            "GUI_KEYPRESS": ComputerActionType.GUI_KEYPRESS,
            "GUI_HOTKEY": ComputerActionType.GUI_HOTKEY,
            "HOTKEY": ComputerActionType.GUI_HOTKEY,
            "GUI_MOVE": ComputerActionType.GUI_MOVE,
            "GUI_SCROLL": ComputerActionType.GUI_SCROLL,
            "GUI_SCREENSHOT": ComputerActionType.GUI_SCREENSHOT,
            "GUI_DRAG": ComputerActionType.GUI_DRAG,
            "GUI_WAIT": ComputerActionType.GUI_WAIT,
            "FILE_READ": ComputerActionType.FILE_READ,
            "FILE_WRITE": ComputerActionType.FILE_WRITE,
            "TERMINAL_EXEC": ComputerActionType.TERMINAL_EXEC,
            "GIT_STATUS": ComputerActionType.GIT_STATUS,
            "GIT_DIFF": ComputerActionType.GIT_DIFF,
            "GIT_BRANCH": ComputerActionType.GIT_BRANCH,
            "GIT_COMMIT": ComputerActionType.GIT_COMMIT,
            "GIT_PUSH": ComputerActionType.GIT_PUSH,
            "APP_LAUNCH": ComputerActionType.APP_LAUNCH,
            "LAUNCH": ComputerActionType.APP_LAUNCH,
            "OPEN": ComputerActionType.APP_LAUNCH,
            "START": ComputerActionType.APP_LAUNCH,
            "CLICK": ComputerActionType.GUI_CLICK,
            "TYPE": ComputerActionType.GUI_TYPE,
            "RUN": ComputerActionType.TERMINAL_EXEC,
            "EXEC": ComputerActionType.TERMINAL_EXEC,
            "COMMAND": ComputerActionType.TERMINAL_EXEC,
            "TERMINAL": ComputerActionType.TERMINAL_EXEC,
            "NAVIGATE": ComputerActionType.BROWSER_NAVIGATE,
            "BROWSE": ComputerActionType.BROWSER_NAVIGATE,
            "SCAN": ComputerActionType.SECURITY_TOOL,
            "APP_CLOSE": ComputerActionType.APP_CLOSE,
            "APP_FOCUS": ComputerActionType.APP_FOCUS,
            "APP_INSTALL": ComputerActionType.APP_INSTALL,
            "SERVICE_ACTION": ComputerActionType.SERVICE_ACTION,
            "BROWSER_NAVIGATE": ComputerActionType.BROWSER_NAVIGATE,
            "BROWSER_CLICK": ComputerActionType.BROWSER_CLICK,
            "BROWSER_TYPE": ComputerActionType.BROWSER_TYPE,
            "BROWSER_SELECT": ComputerActionType.BROWSER_SELECT,
            "BROWSER_KEYPRESS": ComputerActionType.BROWSER_KEYPRESS,
            "BROWSER_SCROLL": ComputerActionType.BROWSER_SCROLL,
            "BROWSER_SCREENSHOT": ComputerActionType.BROWSER_SCREENSHOT,
            "BROWSER_WAIT": ComputerActionType.BROWSER_WAIT,
            "BROWSER_DOWNLOAD": ComputerActionType.BROWSER_DOWNLOAD,
            "WAIT_FOR_HUMAN": ComputerActionType.WAIT_FOR_HUMAN,
            "SECURITY_TOOL": ComputerActionType.SECURITY_TOOL,
            "TOOL_AUTHOR": ComputerActionType.TOOL_AUTHOR,
            "AUTHOR_TOOL": ComputerActionType.TOOL_AUTHOR,
            "CREATE_TOOL": ComputerActionType.TOOL_AUTHOR,
            "CODE_AUTHOR": ComputerActionType.TOOL_AUTHOR,
            "TOOL_RUN": ComputerActionType.TOOL_RUN,
            "METHOD_INVENT": ComputerActionType.METHOD_INVENT,
            "PYTHON": ComputerActionType.TERMINAL_EXEC,
            "PYTHON_EXEC": ComputerActionType.TERMINAL_EXEC,
            "PYTHON3": ComputerActionType.TERMINAL_EXEC,
            "SCRIPT": ComputerActionType.TERMINAL_EXEC,
            "CURL": ComputerActionType.TERMINAL_EXEC,
            "BASH": ComputerActionType.TERMINAL_EXEC,
            "SHELL": ComputerActionType.TERMINAL_EXEC,
            "CMD": ComputerActionType.TERMINAL_EXEC,
            "DEFINE_SUBAGENT": ComputerActionType.DEFINE_SUBAGENT,
            "INVOKE_SUBAGENT": ComputerActionType.INVOKE_SUBAGENT,
            "SEND_MESSAGE": ComputerActionType.SEND_MESSAGE,
            "MANAGE_SUBAGENTS": ComputerActionType.MANAGE_SUBAGENTS,
            # GOAL_COMPLETE is handled by the caller as a no-op terminator.
            "GOAL_COMPLETE": ComputerActionType.TERMINAL_EXEC,
        }

        if "ANSWER" in fields and "ACTION" not in fields:
            ans_val = fields["ANSWER"].strip()
            ans_word = ans_val.split()[0].strip(" *_\n\r\t`\"'")
            if ans_word in action_map:
                fields["ACTION"] = ans_word
            elif any(act in ans_val for act in action_map):
                for act in action_map:
                    if act in ans_val:
                        fields["ACTION"] = act
                        break

        # Fallback to freeform completion or command detection if no ACTION in fields
        if "ACTION" not in fields:
            text_lower = parse_target_text.lower()
            completion_phrases = (
                "goal is complete", "goal has been achieved", "task is complete",
                "task has been completed", "objective has been achieved",
                "all requested information has been successfully gathered",
                "all requested information has been gathered",
                "information has been successfully gathered",
                "successfully gathered",
            )
            has_action_directive = bool(re.search(r'\b(?:action|run|execute)\s*:\s*(?:terminal_exec|gui_|browser_|file_|git_|app_|tool_|security_tool)', text_lower))
            if any(cp in text_lower for cp in completion_phrases) and not has_action_directive and "terminal_exec" not in text_lower:
                return (
                    ComputerActionType.TERMINAL_EXEC,
                    "goal-complete",
                    {"command": "true"},
                    _GOAL_COMPLETE_SENTINEL,
                )

            # Free-form prose is never executable. Terminal actions must be
            # supplied through the explicit ACTION/TARGET/PAYLOAD schema so
            # observations cannot smuggle shell text into the control plane.

        default_action = "TERMINAL_EXEC"
        raw_action = fields.get("ACTION", default_action).upper()
        action_word = raw_action.split()[0] if raw_action.split() else default_action
        action_str = action_word.strip(" *_\n\r\t`\"'")
        action_type = action_map.get(action_str, ComputerActionType.TERMINAL_EXEC)

        # Signal early termination up to the mission loop via the expected text.
        if action_str == _GOAL_COMPLETE_SENTINEL or action_str in ("COMPLETE", "DONE", "FINISHED", "GOAL_ACHIEVED", "STOP"):
            return (
                action_type,
                "goal-complete",
                {"command": "true"},
                _GOAL_COMPLETE_SENTINEL,
            )

        default_target = default_file if action_type in (ComputerActionType.FILE_READ, ComputerActionType.FILE_WRITE) else ""

        raw_target = fields.get("TARGET", "") or fields.get("COORDINATES", "") or default_target
        target = raw_target.strip(" *_\n\r\t`\"'")
        if not target and fields.get("COORDINATES"):
            target = fields["COORDINATES"].strip(" *_\n\r\t`\"'")
        if target:
            # Take first non-empty line to strip accidental trailing markdown blocks
            for line in target.splitlines():
                if line.strip():
                    target = line.strip(" *_\n\r\t`\"'")
                    break

        if action_type in (ComputerActionType.APP_LAUNCH, ComputerActionType.APP_CLOSE, ComputerActionType.APP_FOCUS, ComputerActionType.APP_INSTALL):
            if not target and raw_action:
                target = re.sub(r'^(?:APP_LAUNCH|APP_CLOSE|APP_FOCUS|APP_INSTALL|LAUNCH|OPEN|START|CLOSE|KILL)\s+', '', raw_action, flags=re.IGNORECASE).strip()
            if target:
                target = ComputerUseAgent._normalize_app_name(target)

        elif action_type == ComputerActionType.BROWSER_NAVIGATE:
            target_lower = target.lower()
            if not target or target_lower in ("/", "none", "not specified", "null", "about:blank"):
                target = "about:blank"
            elif not (target.startswith("http://") or target.startswith("https://") or target.startswith("about:") or target.startswith("file://")):
                if "localhost" in target or "127.0.0.1" in target:
                    target = f"http://{target}"
                elif "." in target and " " not in target:
                    target = f"https://{target}"

        payload_str = fields.get("PAYLOAD", "{}")

        def _extract_payload_dict(raw: str) -> dict[str, Any]:
            if not raw:
                return {}
            raw = raw.strip()
            if not raw.startswith("{"):
                return {}
            try:
                p = json.loads(raw)
                if isinstance(p, dict):
                    return p
            except Exception:
                pass
            try:
                import ast
                p = ast.literal_eval(raw)
                if isinstance(p, dict):
                    return p
            except Exception:
                pass
            m_cmd = re.search(r'["\']command["\']\s*:\s*["\']([^"\']+)["\']', raw)
            if m_cmd:
                return {"command": m_cmd.group(1).strip()}
            return {}

        extracted = _extract_payload_dict(payload_str)
        if extracted:
            payload = extracted
        elif action_type in (ComputerActionType.GUI_TYPE, ComputerActionType.BROWSER_TYPE):
            payload = {"text": payload_str.strip('"\' ')} if payload_str.strip() not in ("{}", "") else {}
        elif action_type == ComputerActionType.TERMINAL_EXEC:
            cleaned_cmd = payload_str.strip('"\' ').strip().rstrip(',;').strip()
            payload = {"command": cleaned_cmd} if cleaned_cmd not in ("{}", "") else {}
        elif action_type == ComputerActionType.FILE_WRITE:
            payload = {}
            raw_c = fields.get("CONTENT") or fields.get("CODE") or fields.get("SOURCE") or (payload_str if payload_str.strip() not in ("{}", "", "none") else "")
            if raw_c:
                raw_c_str = str(raw_c).strip()
                if raw_c_str.startswith("```"):
                    raw_c_str = re.sub(r"^```[a-zA-Z0-9_\-\.]*\s*\n?", "", raw_c_str)
                    raw_c_str = re.sub(r"\n?\s*```$", "", raw_c_str)
                # Unwrap JSON envelope if model output JSON e.g. {"path": "...", "content": "..."}
                if raw_c_str.startswith("{") and any(k in raw_c_str for k in ('"content"', '"code"', '"source"', '"text"')):
                    try:
                        parsed_c = json.loads(raw_c_str)
                        if isinstance(parsed_c, dict):
                            if "path" in parsed_c and "path" not in payload:
                                payload["path"] = str(parsed_c["path"])
                            raw_c_str = str(parsed_c.get("content") or parsed_c.get("code") or parsed_c.get("source") or raw_c_str)
                    except Exception:
                        pass
                payload["content"] = raw_c_str
        elif action_type == ComputerActionType.TOOL_AUTHOR:
            payload = {}
            raw_c = fields.get("CODE") or fields.get("SOURCE") or fields.get("CONTENT") or (payload_str if payload_str.strip() not in ("{}", "", "none") else "")
            if raw_c:
                raw_c_str = str(raw_c).strip()
                if raw_c_str.startswith("```"):
                    raw_c_str = re.sub(r"^```[a-zA-Z0-9_\-\.]*\s*\n?", "", raw_c_str)
                    raw_c_str = re.sub(r"\n?\s*```$", "", raw_c_str)
                # Unwrap JSON envelope if model output JSON
                if raw_c_str.startswith("{") and any(k in raw_c_str for k in ('"code"', '"source"', '"content"')):
                    try:
                        parsed_c = json.loads(raw_c_str)
                        if isinstance(parsed_c, dict):
                            if "name" in parsed_c and "name" not in payload:
                                payload["name"] = str(parsed_c["name"])
                            raw_c_str = str(parsed_c.get("code") or parsed_c.get("source") or parsed_c.get("content") or raw_c_str)
                    except Exception:
                        pass
                payload["code"] = raw_c_str
                payload["source"] = raw_c_str
        else:
            payload = {}

        # If target itself was formatted as JSON e.g. {"command": "hostname"}
        if target.startswith("{"):
            tgt_dict = _extract_payload_dict(target)
            if "command" in tgt_dict:
                payload.setdefault("command", str(tgt_dict["command"]).strip().rstrip(',;').strip())
                target = str(tgt_dict["command"]).strip().rstrip(',;').strip().split()[0]
            elif "app_name" in tgt_dict:
                target = tgt_dict["app_name"]
            elif "url" in tgt_dict:
                target = tgt_dict["url"]

        if action_type == ComputerActionType.TOOL_AUTHOR:
            if "source" not in payload and "SOURCE" in fields:
                payload["source"] = fields["SOURCE"]
            elif "code" not in payload and "CODE" in fields:
                payload["code"] = fields["CODE"]
                payload["source"] = fields["CODE"]
            if "code" not in payload and "command" in payload:
                payload["code"] = payload["command"]
                payload.setdefault("source", payload["command"])
            if "name" not in payload and target:
                payload["name"] = target
            if "observation" not in payload and "OBSERVATION" in fields:
                payload["observation"] = fields["OBSERVATION"]

        if action_type == ComputerActionType.TOOL_RUN:
            if "tool" not in payload and target:
                payload["tool"] = target

        if action_type == ComputerActionType.FILE_WRITE:
            if "content" not in payload:
                raw_c = fields.get("CONTENT") or fields.get("CODE") or fields.get("SOURCE") or (payload_str if payload_str.strip() not in ("{}", "", "none") else "")
                if raw_c:
                    raw_c_str = str(raw_c).strip()
                    if raw_c_str.startswith("```"):
                        raw_c_str = re.sub(r"^```[a-zA-Z0-9_\-\.]*\s*\n?", "", raw_c_str)
                        raw_c_str = re.sub(r"\n?\s*```$", "", raw_c_str)
                    payload["content"] = raw_c_str
            if "path" not in payload and target:
                payload["path"] = target

        if action_type in (ComputerActionType.GUI_TYPE, ComputerActionType.BROWSER_TYPE):
            # GUI and browser typing is strictly for text inputs, not shell commands.
            # Do NOT map payload["command"] into payload["text"].
            pass

        if action_str in ("PYTHON", "PYTHON_EXEC", "PYTHON3"):
            py_code = payload.get("code") or payload.get("command") or target or ""
            py_code_str = str(py_code).strip().rstrip(',;').strip()
            if py_code_str and not py_code_str.startswith("python"):
                payload["command"] = f"python3 -c {shlex.quote(py_code_str)}"
        elif action_str == "CURL":
            curl_arg = payload.get("command") or target or ""
            curl_arg_str = str(curl_arg).strip().rstrip(',;').strip()
            if curl_arg_str and not curl_arg_str.startswith("curl"):
                payload["command"] = f"curl {curl_arg_str}"

        if action_type == ComputerActionType.TERMINAL_EXEC:
            raw_cmd = (
                payload.get("command")
                or payload.get("cmd")
                or payload.get("bash")
                or payload.get("shell")
                or payload.get("script")
                or payload.get("code")
                or fields.get("COMMAND")
            )
            if not raw_cmd and fields.get("ACTION"):
                act_field = fields["ACTION"].strip()
                for prefix in ("TERMINAL_EXEC", "TERMINAL", "RUN", "EXEC", "COMMAND", "CMD", "BASH", "SHELL"):
                    if act_field.upper().startswith(prefix) and len(act_field) > len(prefix):
                        # Avoid taking '_EXEC' as a command when action was 'TERMINAL_EXEC'
                        if prefix == "TERMINAL" and act_field.upper().startswith("TERMINAL_EXEC"):
                            continue
                        tail = act_field[len(prefix):].strip(" :").strip().rstrip(',;').strip()
                        if tail and tail.upper() not in ("_EXEC", "EXEC", "COMMAND", "RUN", "SHELL", "BASH"):
                            raw_cmd = tail
                            break
            if not raw_cmd or str(raw_cmd).lower() in ("none", "null", "") or str(raw_cmd).strip().upper() in ("_EXEC", "EXEC"):
                fallback_cmd = target if target and target != default_file and not target.startswith("{") else "echo 'ERROR: Empty command payload' >&2; exit 1"
                payload["command"] = str(fallback_cmd).strip().rstrip(',;').strip()
            else:
                payload["command"] = str(raw_cmd).strip().rstrip(',;').strip()
            if payload.get("command"):
                cmd_str = str(payload["command"]).strip().rstrip(',;').strip()
                cmd_str = re.sub(r'^(?:[a-zA-Z0-9_\-\.]+(?:-terminal)?,?\s*)?(?:command|cmd)\s*=\s*', '', cmd_str).strip().rstrip(',;').strip()
                # Only remove conversational parenthesized comments e.g. "netstat -tuln (or ss)" -> "netstat -tuln",
                # without destroying parentheses inside quotes, code, or URLs like Wikipedia articles.
                if not any(k in cmd_str for k in ("http://", "https://", "python", "perl", "awk", "sed", "$(")):
                    cmd_str = re.sub(r'\s*\((?:or|and|note|comment|alias|e\.g\.)[^)]*\)$', '', cmd_str, flags=re.IGNORECASE).strip().rstrip(',;').strip()
                # Filter out UI text and navigation elements that are not commands
                if any(pattern in cmd_str for pattern in ("localhost:12001", "[Missions]", "[Computer]", "[Research]", "[Experiments]", "[Graph Memory]", "[Evidence Board]", "[Agents]", "[Security Lab]", "[Settings]", "LIVE", "FAIL-CLOSED", "SONICA-SEA", "Copy", "Autonomous execution paused")):
                    cmd_str = "echo 'ERROR: UI navigation text is not an executable command' >&2; exit 1"
                # Handle conversational placeholders like "Terminal" or "Terminal window"
                if cmd_str.lower() in ("terminal", "terminal window", "the terminal", "bash", "shell", "console"):
                    cmd_str = "echo 'ERROR: Shell placeholder name is not an executable command' >&2; exit 1"
                # Strip trailing " command" or " commands"
                cmd_str = re.sub(r'\s+commands?$', '', cmd_str, flags=re.IGNORECASE).strip().rstrip(',;').strip()
                payload["command"] = cmd_str.strip().rstrip(',;').strip()

        if action_type in (ComputerActionType.GUI_CLICK, ComputerActionType.GUI_DOUBLE_CLICK, ComputerActionType.GUI_RIGHT_CLICK, ComputerActionType.GUI_MOVE):
            if target and "," in target:
                parts = target.split(",")
                if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    payload["x"] = int(parts[0].strip())
                    payload["y"] = int(parts[1].strip())
                elif any(p.strip().lower() in ("x", "y", "x,y", "x, y") for p in parts):
                    payload.pop("x", None)
                    payload.pop("y", None)
        elif action_type == ComputerActionType.GUI_DRAG:
            # Flexible GUI_DRAG parsing from payload and target:
            # Handles {"from": [x,y], "to": [x2,y2]}, {"start": ..., "end": ...}, {"x": ..., "y": ..., "x2": ..., "y2": ...}
            if "from" in payload and isinstance(payload["from"], (list, tuple)) and len(payload["from"]) == 2:
                payload["x"] = int(payload["from"][0])
                payload["y"] = int(payload["from"][1])
            if "to" in payload and isinstance(payload["to"], (list, tuple)) and len(payload["to"]) == 2:
                payload["x2"] = int(payload["to"][0])
                payload["y2"] = int(payload["to"][1])
            if "start" in payload and isinstance(payload["start"], (list, tuple)) and len(payload["start"]) == 2:
                payload["x"] = int(payload["start"][0])
                payload["y"] = int(payload["start"][1])
            if "end" in payload and isinstance(payload["end"], (list, tuple)) and len(payload["end"]) == 2:
                payload["x2"] = int(payload["end"][0])
                payload["y2"] = int(payload["end"][1])
            # Check target string for coordinates e.g. "600,185 to 200,200", "(600,185) -> (200,200)"
            m_drag = re.search(r'(\d+)\s*,\s*(\d+).*?(\d+)\s*,\s*(\d+)', str(target or ""))
            if m_drag:
                payload.setdefault("x", int(m_drag.group(1)))
                payload.setdefault("y", int(m_drag.group(2)))
                payload.setdefault("x2", int(m_drag.group(3)))
                payload.setdefault("y2", int(m_drag.group(4)))
            elif target and "," in target:
                parts = target.split(",")
                if len(parts) >= 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    payload.setdefault("x", int(parts[0].strip()))
                    payload.setdefault("y", int(parts[1].strip()))
            # Default x2, y2 if still missing
            if "x" in payload and "x2" not in payload:
                payload["x2"] = payload["x"]
            if "y" in payload and "y2" not in payload:
                payload["y2"] = payload["y"]
        if action_type == ComputerActionType.GUI_SCROLL:
            if target and "," in target:
                parts = target.split(",")
                if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    payload["x"] = int(parts[0].strip())
                    payload["y"] = int(parts[1].strip())

        expected = fields.get("EXPECTED", f"Action {action_type} on {target}")
        return action_type, target, payload, expected

    @staticmethod
    def _diagnostic_fallback(primary_file: str) -> tuple[ComputerActionType, str, dict[str, Any], str]:
        """Offline fallback when no LLM is available.

        Deliberately generic: it inspects the primary source file as a
        diagnostic probe. It does NOT apply any vulnerability-specific patch
        and contains no JWT/SQLi/XSS logic. Real remediation requires an LLM.
        """
        file_path = f"/home/sonic/workspace/{primary_file}" if (primary_file and primary_file != "UNKNOWN") else "/home/sonic/workspace"
        target_name = primary_file if (primary_file and primary_file != "UNKNOWN") else "workspace"
        return (
            ComputerActionType.FILE_READ,
            file_path,
            {"path": file_path},
            f"Diagnostic: inspected {target_name} (no LLM available to remediate)",
        )

    # =============================================================
    # 3. Action Execution & Closed-Loop Verification
    # =============================================================
    # Actions that target pixel coordinates and must stay on-screen.
    _COORDINATE_ACTIONS: frozenset[str] = frozenset({
        "GUI_CLICK", "GUI_DOUBLE_CLICK", "GUI_RIGHT_CLICK", "GUI_MOVE", "GUI_SCROLL", "GUI_DRAG",
    })
    _COMPUTER_PLANE_ACTIONS: frozenset[ComputerActionType] = frozenset({
        ComputerActionType.GUI_CLICK,
        ComputerActionType.GUI_DOUBLE_CLICK,
        ComputerActionType.GUI_RIGHT_CLICK,
        ComputerActionType.GUI_TYPE,
        ComputerActionType.GUI_KEYPRESS,
        ComputerActionType.GUI_HOTKEY,
        ComputerActionType.GUI_MOVE,
        ComputerActionType.GUI_SCROLL,
        ComputerActionType.GUI_SCREENSHOT,
        ComputerActionType.GUI_DRAG,
        ComputerActionType.GUI_WAIT,
        ComputerActionType.APP_LAUNCH,
        ComputerActionType.APP_CLOSE,
        ComputerActionType.APP_FOCUS,
        ComputerActionType.BROWSER_NAVIGATE,
        ComputerActionType.BROWSER_CLICK,
        ComputerActionType.BROWSER_TYPE,
        ComputerActionType.BROWSER_SELECT,
        ComputerActionType.BROWSER_KEYPRESS,
        ComputerActionType.BROWSER_SCROLL,
        ComputerActionType.BROWSER_SCREENSHOT,
        ComputerActionType.BROWSER_WAIT,
        ComputerActionType.BROWSER_DOWNLOAD,
    })
    _GUI_ONLY_ACTIONS: frozenset[ComputerActionType] = frozenset({
        ComputerActionType.GUI_CLICK,
        ComputerActionType.GUI_DOUBLE_CLICK,
        ComputerActionType.GUI_RIGHT_CLICK,
        ComputerActionType.GUI_TYPE,
        ComputerActionType.GUI_KEYPRESS,
        ComputerActionType.GUI_HOTKEY,
        ComputerActionType.GUI_MOVE,
        ComputerActionType.GUI_SCROLL,
        ComputerActionType.GUI_SCREENSHOT,
        ComputerActionType.GUI_DRAG,
        ComputerActionType.GUI_WAIT,
        ComputerActionType.APP_LAUNCH,
        ComputerActionType.APP_CLOSE,
        ComputerActionType.APP_FOCUS,
        ComputerActionType.BROWSER_NAVIGATE,
        ComputerActionType.BROWSER_CLICK,
        ComputerActionType.BROWSER_TYPE,
        ComputerActionType.BROWSER_SELECT,
        ComputerActionType.BROWSER_KEYPRESS,
        ComputerActionType.BROWSER_SCROLL,
        ComputerActionType.BROWSER_SCREENSHOT,
        ComputerActionType.BROWSER_WAIT,
        ComputerActionType.BROWSER_DOWNLOAD,
        ComputerActionType.WAIT_FOR_HUMAN,
        ComputerActionType.GOAL_COMPLETE,
    })

    def _validate_coordinates(
        self, action_type: ComputerActionType, payload: dict[str, Any],
    ) -> str | None:
        """Return an error string if a coordinate action is off-screen, else None.

        Validates (x, y) — and (x2, y2) / (end_x, end_y) for drags — against the last observed
        screen dimensions. Coordinates must be non-negative integers strictly
        inside the screen (0 <= x < width, 0 <= y < height). Non-coordinate
        actions are always allowed (return None).
        """
        if action_type.value not in self._COORDINATE_ACTIONS:
            return None

        width = getattr(self, "_screen_width", 1920)
        height = getattr(self, "_screen_height", 1080)

        def _check(x: Any, y: Any, label: str) -> str | None:
            try:
                ix, iy = int(x), int(y)
            except (TypeError, ValueError):
                return f"{label} ({x},{y}) is not a valid integer coordinate"
            if ix < 0 or iy < 0:
                return f"{label} ({ix},{iy}) is negative"
            if ix >= width or iy >= height:
                return (
                    f"{label} ({ix},{iy}) is outside screen "
                    f"{width}x{height}"
                )
            return None

        if action_type == ComputerActionType.GUI_DRAG:
            start_x = payload.get("x")
            start_y = payload.get("y")
            end_x = payload.get("end_x") if payload.get("end_x") is not None else payload.get("x2")
            end_y = payload.get("end_y") if payload.get("end_y") is not None else payload.get("y2")
            if start_x is None or start_y is None:
                return "drag start coordinates (x, y) are required"
            if end_x is None or end_y is None:
                return "drag destination coordinates (end_x/x2, end_y/y2) are required"
            err = _check(start_x, start_y, "drag start")
            if err is not None:
                return err
            err = _check(end_x, end_y, "drag target")
            if err is not None:
                return err
            return None

        err = _check(payload.get("x", 0), payload.get("y", 0), "point")
        if err is not None:
            return err
        return None

    def _update_screen_dims(self, screen: Any) -> None:
        """Refresh tracked screen dimensions from a screenshot observation."""
        w = int(getattr(screen, "width", 0) or 0)
        h = int(getattr(screen, "height", 0) or 0)
        if w > 0:
            self._screen_width = w
        if h > 0:
            self._screen_height = h

    def _get_provider_name(self) -> str:
        """Return human-readable identifier for computer provider."""
        name = getattr(self.computer, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
        if hasattr(self.computer, "compute_provider") and hasattr(self.computer.compute_provider, "__class__"):
            return self.computer.compute_provider.__class__.__name__
        if hasattr(self.computer, "__class__"):
            return self.computer.__class__.__name__
        return "sandbox"

    @staticmethod
    def _clean_terminal_command(raw: Any) -> str:
        if raw is None:
            return "echo OK"
        s = str(raw).strip().rstrip(',;').strip()
        if not s or s.lower() in ("none", "null", "echo ok"):
            return "echo OK"

        # If model hallucinated placeholder phrases like "diagnostic probe" or "address"
        if s.lower() in ("diagnostic probe", "diagnostic.sh", "diagnostic probe.", "address"):
            return "true"

        # 1. Unpack JSON or dict only if payload is formatted as a JSON object
        if s.startswith("{"):
            try:
                p = json.loads(s)
                if isinstance(p, dict) and "command" in p:
                    s = str(p["command"]).strip().rstrip(',;').strip()
            except Exception:
                try:
                    import ast
                    p = ast.literal_eval(s)
                    if isinstance(p, dict) and "command" in p:
                        s = str(p["command"]).strip().rstrip(',;').strip()
                except Exception:
                    cmd_match = re.search(r'["\']command["\']\s*:\s*["\']((?:\\.|[^"\\])+)["\']', s)
                    if cmd_match:
                        s = cmd_match.group(1).replace(r'\"', '"').replace(r"\'", "'").strip().rstrip(',;').strip()

        # 2. Strip bash/terminal wrapper prefixes & markdown fences
        s = re.sub(r'^(?:[a-zA-Z0-9_\-\.]+(?:-terminal)?,?\s*)?(?:command|cmd)\s*=\s*', '', s)
        s = re.sub(r'^```(?:bash|sh)?\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
        s = s.strip(" \t\n\r`").rstrip(',;').strip()
        if s == "_EXEC" or s == "EXEC":
            s = ""
        elif s.startswith("_EXEC "):
            s = s[6:].strip().rstrip(',;').strip()
        elif s.startswith("EXEC "):
            s = s[5:].strip().rstrip(',;').strip()

        # 3. If it is still a JSON object like {"diagnostic": ...} without command, don't execute raw JSON in bash
        if s.startswith("{") and s.endswith("}"):
            return "echo 'ERROR: Unparseable JSON command payload' >&2; exit 1"

        # 4. If launching a graphical desktop application directly via DISPLAY=:... in terminal,
        # ensure it runs in the background (&) so it does not indefinitely block the headless subshell
        if re.search(r'^\s*DISPLAY=:\d+\s+', s) and not s.rstrip().endswith("&"):
            s = f"{s.rstrip()} &"

        s = s.strip().rstrip(',;').strip()
        return s or "echo 'ERROR: Empty command' >&2; exit 1"

    def _extract_tool_name(
        self,
        action_type: ComputerActionType,
        target_resource: str,
        payload: dict[str, Any],
    ) -> str:
        """Extract tool or binary name for failure budget tracking."""
        if action_type == ComputerActionType.TERMINAL_EXEC:
            raw_cmd = payload.get("command") or target_resource or "terminal"
            clean_cmd = self._clean_terminal_command(raw_cmd)
            parts = [p for p in clean_cmd.split() if not ("=" in p and not p.startswith("-"))]
            tool_candidate = parts[0] if parts else (clean_cmd.split()[0] if clean_cmd.split() else "terminal")
            tool_candidate = re.sub(r'[^a-zA-Z0-9_.-]', '', tool_candidate)
            return tool_candidate or "terminal"
        if action_type == ComputerActionType.SECURITY_TOOL:
            return str(payload.get("tool") or target_resource or "security_tool")
        if action_type == ComputerActionType.APP_INSTALL:
            return str(payload.get("package") or payload.get("app_name") or target_resource or "installer")
        return action_type.value

    @staticmethod
    def _assess_severity(action_type: ComputerActionType, payload: dict[str, Any]) -> str:
        """Map a computer action to a severity band for sandbox-tier routing.

        Conservative and explicit: anything that writes files, installs apps,
        runs arbitrary commands, touches services, or mutates git/source is at
        least 'medium'. GUI navigation is 'low'; pure observation is 'info'.
        """
        t = action_type.value
        if t in (
            "FILE_WRITE", "APP_INSTALL", "SERVICE_ACTION",
            "TOOL_AUTHOR", "TOOL_RUN", "METHOD_INVENT",
            "GIT_COMMIT", "GIT_BRANCH",
        ):
            return "high"
        if t in (
            "TERMINAL_EXEC", "APP_LAUNCH", "APP_CLOSE", "APP_FOCUS",
            "BROWSER_DOWNLOAD", "SECURITY_TOOL",
        ):
            return "medium"
        if t in ("FILE_READ",):
            return "low"
        return "info"

    def _native_govern_action(
        self,
        action_type: ComputerActionType,
        target_resource: str,
    ) -> dict[str, Any]:
        """Run the action through the ASI Sandbox pillar (severity -> tier) with
        the native Rust kernel confirming the envelope. Fail-closed: a Forbidden
        tier or a kernel Deny blocks execution outright."""
        try:
            from sonic.safety.scope import SandboxTierRouter
            severity = self._assess_severity(action_type, {})
            router = SandboxTierRouter()
            routed = router.route(
                severity,
                action_type=action_type.value,
                target=target_resource,
            )
            return routed
        except Exception as e:
            logger.debug("native_govern_action_unavailable", error=str(e))
            return {
                "severity": "info",
                "tier": "standard",
                "network_isolated": False,
                "executable": True,
                "native_verdict": "unavailable",
                "reason": f"native governance unavailable: {e}",
            }

    async def execute_action(
        self,
        workspace_id: str,
        action_type: ComputerActionType,
        target_resource: str,
        payload: dict[str, Any],
        predicted_outcome: str,
    ) -> ComputerDecisionTrace:
        """Executes the action inside the computer sandbox and validates outcome."""
        self.action_counter += 1
        t_start = time.perf_counter()
        actual_obs_str = ""
        status = ActionExecutionStatus.COMPLETED
        recovery_needed = False
        action_exit_code: int | None = None
        action_stderr: str = ""
        browser_status_code: int | None = None
        browser_response_body = ""
        before_state = self._current_experience_state()
        self._experience_before_graph = self._current_experience_graph()

        if not hasattr(self, "failure_budget"):
            self.failure_budget = FailureBudgetTracker()
        if not hasattr(self, "strategies"):
            self.strategies = {
                "Strategy A": {"name": "Direct Primary Execution", "description": "Primary probe / targeted execution", "state": StrategyState.ACTIVE},
                "Strategy B": {"name": "Alternative Instrumentation", "description": "Secondary inspection / fallback tool", "state": StrategyState.ACTIVE},
            }
        if not hasattr(self, "_consecutive_unchanged_screens"):
            self._consecutive_unchanged_screens = 0

        if isinstance(payload, str) and payload.strip().startswith("{"):
            try:
                parsed_p = json.loads(payload)
                if isinstance(parsed_p, dict):
                    payload = parsed_p
            except Exception:
                pass
        elif not isinstance(payload, dict):
            payload = {} if payload is None else {"raw": payload}

        # Pre-execution refusal or blocked action handling (e.g. from LLM safety refusal)
        if action_type == ActionExecutionStatus.BLOCKED or str(action_type).upper() in ("BLOCKED", "ACTIONEXECUTIONSTATUS.BLOCKED"):
            actual_obs_str = (
                f"Action blocked: {target_resource} — {predicted_outcome}"
                if target_resource else f"Action blocked: {predicted_outcome}"
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=ComputerActionType.TERMINAL_EXEC,
                target_resource=target_resource or "blocked-action",
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=ActionExecutionStatus.BLOCKED,
                exit_code=126,
                thought=getattr(self, "_last_thought", ""),
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": f"BLOCKED {target_resource}", "result": actual_obs_str})
            return trace

        if getattr(self, "gui_only", False) and action_type not in self._GUI_ONLY_ACTIONS:
            actual_obs_str = (
                f"GUI-only computer blocked {action_type.value}. "
                "Use visible desktop applications and GUI controls; terminal, "
                "shell, file, security-tool, and backend execution are unavailable."
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=ActionExecutionStatus.BLOCKED,
                exit_code=126,
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": action_type.value, "result": actual_obs_str})
            return trace

        if (
            str(getattr(self, "execution_plane", "dual") or "dual").lower().strip() == "headless"
            and action_type in self._COMPUTER_PLANE_ACTIONS
            and not (
                action_type == ComputerActionType.BROWSER_NAVIGATE
                and isinstance(payload, dict)
                and (payload.get("headless_only") is True or payload.get("interactive") is False)
            )
        ):
            actual_obs_str = (
                f"Blocked by headless execution plane contract: {action_type.value} is a graphical desktop action. "
                "Use TERMINAL_EXEC, FILE_READ, FILE_WRITE, or GIT_* in headless plane."
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=ActionExecutionStatus.BLOCKED,
                exit_code=126,
                thought=getattr(self, "_last_thought", ""),
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": action_type.value, "result": actual_obs_str})
            return trace

        prev_trace = (getattr(self, "action_history", None) or getattr(self, "traces", None) or [None])[-1]
        prev_status = str(getattr(prev_trace, "status", "")).upper() if prev_trace else ""
        is_prev_success = prev_status in ("COMPLETED", "SUCCESS", "SUCCEEDED", "ACTIONEXECUTIONSTATUS.COMPLETED")

        if action_type == ComputerActionType.FILE_WRITE and prev_trace is not None:
            if getattr(prev_trace, "action_type", None) == ComputerActionType.FILE_WRITE and is_prev_success:
                curr_path = str(payload.get("path") or target_resource or "").strip()
                prev_payload = getattr(prev_trace, "payload", {})
                if isinstance(prev_payload, str):
                    try:
                        prev_payload = json.loads(prev_payload)
                    except Exception:
                        prev_payload = {}
                prev_path = str(
                    (prev_payload.get("path") if isinstance(prev_payload, dict) else "")
                    or getattr(prev_trace, "target_resource", "")
                    or ""
                ).strip()
                if curr_path and curr_path == prev_path:
                    actual_obs_str = "Duplicate consecutive FILE_WRITE to the same file blocked. Run TERMINAL_EXEC to verify the file or call GOAL_COMPLETE if finished."
                    trace = ComputerDecisionTrace(
                        step_index=self.action_counter,
                        action_type=action_type,
                        target_resource=target_resource,
                        payload=str(payload),
                        predicted_outcome=predicted_outcome,
                        actual_observation=actual_obs_str,
                        expected_observation=predicted_outcome,
                        info_gain=0.0,
                        recovery_attempted=False,
                        status=ActionExecutionStatus.BLOCKED,
                        exit_code=126,
                        thought=getattr(self, "_last_thought", ""),
                        duration_seconds=round(time.perf_counter() - t_start, 3),
                        thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
                    )
                    self.traces.append(trace)
                    self.history.append({"action": action_type.value, "result": actual_obs_str})
                    return trace

        if action_type == ComputerActionType.APP_LAUNCH and prev_trace is not None:
            if getattr(prev_trace, "action_type", None) == ComputerActionType.APP_LAUNCH and is_prev_success:
                curr_app = str(payload.get("app_name") or target_resource or "").strip().lower()
                prev_payload = getattr(prev_trace, "payload", {})
                if isinstance(prev_payload, str):
                    try:
                        prev_payload = json.loads(prev_payload)
                    except Exception:
                        prev_payload = {}
                prev_app = str(
                    (prev_payload.get("app_name") if isinstance(prev_payload, dict) else "")
                    or getattr(prev_trace, "target_resource", "")
                    or ""
                ).strip().lower()
                if curr_app and curr_app == prev_app:
                    actual_obs_str = f"Duplicate consecutive APP_LAUNCH of '{curr_app}' blocked. The application is already launched; use APP_FOCUS or GUI_* actions to interact with it."
                    trace = ComputerDecisionTrace(
                        step_index=self.action_counter,
                        action_type=action_type,
                        target_resource=target_resource,
                        payload=str(payload),
                        predicted_outcome=predicted_outcome,
                        actual_observation=actual_obs_str,
                        expected_observation=predicted_outcome,
                        info_gain=0.0,
                        recovery_attempted=False,
                        status=ActionExecutionStatus.BLOCKED,
                        exit_code=126,
                        thought=getattr(self, "_last_thought", ""),
                        duration_seconds=round(time.perf_counter() - t_start, 3),
                        thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
                    )
                    self.traces.append(trace)
                    self.history.append({"action": action_type.value, "result": actual_obs_str})
                    return trace

        # Command sanitization and empty check for TERMINAL_EXEC
        if action_type == ComputerActionType.TERMINAL_EXEC:
            raw_cmd = (
                payload.get("command")
                or payload.get("cmd")
                or payload.get("bash")
                or payload.get("shell")
                or payload.get("script")
                or payload.get("code")
            ) if isinstance(payload, dict) else target_resource
            if not raw_cmd:
                raw_cmd = target_resource
            if raw_cmd is None:
                raw_cmd = ""
            if isinstance(raw_cmd, dict):
                raw_cmd = raw_cmd.get("command", "")
            cmd = str(raw_cmd).strip().rstrip(',;').strip()
            cmd = re.sub(r'[\s,;]+$', '', cmd).strip()
            if cmd.startswith("{") and "command" in cmd:
                try:
                    parsed_p = json.loads(cmd)
                    if isinstance(parsed_p, dict) and "command" in parsed_p:
                        cmd = str(parsed_p["command"]).strip().rstrip(',;').strip()
                        cmd = re.sub(r'[\s,;]+$', '', cmd).strip()
                except Exception:
                    pass
            if not cmd.strip(' \t\r\n,;'):
                cmd = ""
            if not cmd:
                res = AgentStepResult(
                    False,
                    'Empty command provided',
                    step_index=self.action_counter,
                    target_resource=target_resource or "terminal",
                    payload=str(payload),
                    predicted_outcome=predicted_outcome,
                )
                self.traces.append(res)
                self.history.append({"action": "TERMINAL_EXEC", "result": "Empty command provided"})
                return res
            if isinstance(payload, dict):
                payload["command"] = cmd

        # ----- PLAN Phase 6: fail-closed safety envelope -----
        # Every action — operator-issued OR self-directed (curiosity) — must pass
        # the policy before it touches the provider. A denied action is recorded
        # as BLOCKED and NEVER executed, and deliberately does NOT trigger the
        # recovery path (recovery must not be able to bypass the safety gate).
        if self.safety is not None:
            verdict = self.safety.evaluate(action_type.value, target_resource, payload)
            if not verdict.allowed:
                actual_obs_str = f"Safety blocked: {verdict.reason}"
                if action_type == ComputerActionType.BROWSER_NAVIGATE:
                    actual_obs_str += (
                        " Do not retry this URL. It is an untrusted or unauthorized reference; "
                        "continue with the authorized target already in scope or request explicit operator authorization."
                    )
                status = ActionExecutionStatus.BLOCKED
                logger.warning("action_blocked_by_policy",
                               action=action_type.value, reason=verdict.reason)
                trace = ComputerDecisionTrace(
                    step_index=self.action_counter,
                    action_type=action_type,
                    target_resource=target_resource,
                    payload=str(payload),
                    predicted_outcome=predicted_outcome,
                    actual_observation=actual_obs_str,
                    expected_observation=predicted_outcome,
                    info_gain=0.0,
                    recovery_attempted=False,
                    status=status,
                    exit_code=126,
                    duration_seconds=round(time.perf_counter() - t_start, 3),
                    thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
                )
                self.traces.append(trace)
                self.history.append({"action": action_type.value, "result": actual_obs_str})
                if action_type == ComputerActionType.BROWSER_NAVIGATE:
                    blocked_signature = (
                        action_type.value,
                        str(target_resource).strip().lower(),
                        str(payload).strip(),
                    )
                    if not hasattr(self, "_recent_action_signatures"):
                        self._recent_action_signatures = []
                    self._recent_action_signatures.append(blocked_signature)
                return trace

        # ----- ASI Sandbox Pillar + Native Kernel governance -----
        # The action's severity decides its isolation tier (fail-closed:
        # critical/unknown severity never executes). The native Rust kernel is
        # consulted for the envelope (seal intact + allowlisted family). A
        # kernel Deny on a *specific target* (e.g. private-IP egress) is NOT
        # treated as a sandbox-tier violation here — target-scope enforcement is
        # the ActionBroker/SafetyKernel's job at execution time. Only an
        # outright forbidden tier hard-blocks at this stage.
        routed = self._native_govern_action(action_type, target_resource)
        if not routed.get("executable", True):
            actual_obs_str = (
                f"Sandbox tier forbids execution: {routed.get('reason', 'denied')}"
            )
            status = ActionExecutionStatus.BLOCKED
            logger.warning(
                "action_blocked_by_sandbox_tier",
                action=action_type.value,
                tier=routed.get("tier"),
                reason=routed.get("reason"),
                native_verdict=routed.get("native_verdict"),
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=status,
                exit_code=126,
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": action_type.value, "result": actual_obs_str})
            return trace

        # ----- Substrate Outage Detection (Failure Budget) -----
        # If 3 consecutive provider errors occurred, STOP target actions and switch to substrate diagnostic action.
        provider_name = type(self.computer).__name__
        tool_name = (
            payload.get("tool_name") or target_resource or action_type.value
            if isinstance(payload, dict) and action_type == ComputerActionType.SECURITY_TOOL
            else (action_type.value if hasattr(action_type, "value") else str(action_type))
        )
        if self.failure_budget.substrate_outage:
            diag_info = self.failure_budget.diagnose_substrate_health()
            actual_obs_str = (
                f"SUBSTRATE_OUTAGE_DETECTED: Consecutive provider/sandbox failures reached threshold. "
                f"Target actions stopped; switched to substrate diagnostic: {diag_info}"
            )
            status = ActionExecutionStatus.BLOCKED
            logger.error(
                "substrate_outage_target_halted",
                provider=provider_name,
                tool=tool_name,
                status=self.failure_budget.diagnostic_status,
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=status,
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": f"{action_type.value} {target_resource}", "result": actual_obs_str})
            return trace

        # ----- Strategy Exhaustion Check -----
        # If strategy for this tool/provider is exhausted, DO NOT retry the same command/tool!
        # Record strategy exhausted and set status = ActionExecutionStatus.FAILED.
        if self.failure_budget.is_strategy_exhausted(tool=tool_name, provider=provider_name):
            actual_obs_str = (
                f"Strategy exhausted for tool '{tool_name}' on provider '{provider_name}'. "
                f"Failure budget exceeded; retry blocked."
            )
            status = ActionExecutionStatus.BLOCKED
            logger.warning(
                "strategy_exhausted_execution_blocked",
                tool=tool_name,
                provider=provider_name,
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=status,
                exit_code=126,
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": f"{action_type.value} {target_resource}", "result": actual_obs_str})
            return trace

        # ----- Consecutive Action Loop Detector & Circuit Breaker -----
        # Detect if the agent is repeating the exact same action signature consecutively.
        # Exclude wait actions (GUI_WAIT, BROWSER_WAIT) from consecutive repetition blocking.
        action_sig = (action_type.value, str(target_resource).strip().lower(), str(payload).strip())
        if not hasattr(self, "_recent_action_signatures"):
            self._recent_action_signatures = []
        self._recent_action_signatures.append(action_sig)

        if action_type not in {
            ComputerActionType.GUI_WAIT,
            ComputerActionType.BROWSER_WAIT,
            # Headless execution retries are governed by FailureBudgetTracker;
            # blocking the third identical attempt would prevent that tracker
            # from reaching its configured exhaustion threshold.
            ComputerActionType.TERMINAL_EXEC,
            ComputerActionType.SECURITY_TOOL,
        } and len(self._recent_action_signatures) >= 3 and all(
            s == action_sig for s in self._recent_action_signatures[-3:]
        ):
            actual_obs_str = (
                f"[ACTION LOOP DETECTED]: You have repeated '{action_type.value}' on '{target_resource}' "
                "3 times consecutively without state progression. "
                "This action is BLOCKED. Advance to a different action (e.g. alternative command, script authoring, GUI interaction, or conclude)."
            )
            status = ActionExecutionStatus.BLOCKED
            logger.warning(
                "action_loop_breaker_triggered",
                action=action_type.value,
                target=target_resource,
                consecutive_count=3,
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=status,
                thought=getattr(self, "_last_thought", ""),
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": f"{action_type.value} {target_resource}", "result": actual_obs_str})
            return trace

        # ----- Visual Grounding Target Resolution -----
        # If numeric coordinates were not provided for a GUI action, attempt to
        # resolve the target resource query (e.g. "Applications menu", "Terminal icon")
        # to screen coordinates using the visual grounding engine.
        resolved_via_grounding = False
        if action_type in (
            ComputerActionType.GUI_CLICK,
            ComputerActionType.GUI_DOUBLE_CLICK,
            ComputerActionType.GUI_RIGHT_CLICK,
            ComputerActionType.GUI_MOVE,
        ) and (payload.get("x") is None or payload.get("y") is None):
            semantic_match = self.perception_bus.resolve(target_resource)
            if semantic_match is not None:
                future = self.perception_bus.prepare(action_type.value, target_resource)
                if not self.perception_bus.is_current(future):
                    actual_obs_str = (
                        f"Target UI element '{target_resource}' became stale before dispatch; "
                        "re-observing instead of guessing."
                    )
                    status = ActionExecutionStatus.BLOCKED
                    trace = ComputerDecisionTrace(
                        step_index=self.action_counter,
                        action_type=action_type,
                        target_resource=target_resource,
                        payload=str(payload),
                        predicted_outcome=predicted_outcome,
                        actual_observation=actual_obs_str,
                        expected_observation=predicted_outcome,
                        info_gain=0.0,
                        recovery_attempted=False,
                        status=status,
                        exit_code=126,
                        thought=getattr(self, "_last_thought", ""),
                        duration_seconds=round(time.perf_counter() - t_start, 3),
                        thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
                    )
                    self.traces.append(trace)
                    self.history.append({"action": action_type.value, "result": actual_obs_str})
                    return trace
                x, y, width, height = semantic_match.bbox
                payload["x"] = x + max(0, width // 2)
                payload["y"] = y + max(0, height // 2)
                resolved_via_grounding = False
                logger.info(
                    "perception_bus_target_resolved",
                    action=action_type.value,
                    query=target_resource,
                    source=semantic_match.source,
                    state_version=semantic_match.state_version,
                )

            # Fall back to visual reasoning only when the live semantic state
            # has no target; no fixed coordinate or application mapping is used.
            if payload.get("x") is not None and payload.get("y") is not None:
                pass
            else:
                grounding_fn = None
                if self.llm_router and self._last_screenshot_b64:
                    async def _ground(q, s):
                        return await query_multimodal_grounding(
                            self.llm_router, q, s, width=self._screen_width, height=self._screen_height
                        )
                    grounding_fn = _ground

                # Require genuine visual perception or direct coordinates; forbid landmark fallbacks.
                res_coords = await resolve_ui_target_async(
                    query=target_resource,
                    screenshot_b64=self._last_screenshot_b64,
                    width=self._screen_width,
                    height=self._screen_height,
                    grounding_fn=grounding_fn,
                    # Static landmarks are only a compatibility fallback when no
                    # screenshot exists. Once pixels are available, unresolved
                    # targets must be blocked rather than guessed.
                    allow_landmarks=not bool(self._last_screenshot_b64),
                )
                if res_coords is not None:
                    payload["x"], payload["y"] = res_coords
                    resolved_via_grounding = bool(grounding_fn and self._last_screenshot_b64)
                    logger.info(
                        "visual_grounding_target_resolved",
                        action=action_type.value,
                        query=target_resource,
                        resolved_x=res_coords[0],
                        resolved_y=res_coords[1],
                    )
                else:
                    target = target_resource
                    actual_obs_str = f"Target UI element '{target}' could not be resolved from visual grounding. Re-observing screen."
                    status = ActionExecutionStatus.BLOCKED
                    logger.warning(
                        "action_failed_unresolved_visual_grounding",
                        action=action_type.value,
                        target=target,
                    )
                    trace = ComputerDecisionTrace(
                        step_index=self.action_counter,
                        action_type=action_type,
                        target_resource=target_resource,
                        payload=str(payload),
                        predicted_outcome=predicted_outcome,
                        actual_observation=actual_obs_str,
                        expected_observation=predicted_outcome,
                        info_gain=0.0,
                        recovery_attempted=False,
                        status=status,
                        exit_code=126,
                        thought=getattr(self, "_last_thought", ""),
                        duration_seconds=round(time.perf_counter() - t_start, 3),
                        thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
                    )
                    self.traces.append(trace)
                    self.history.append({"action": f"{action_type.value} {target_resource}", "result": actual_obs_str})
                    return trace

        # ----- Coordinate bounds validation -----
        # GUI coordinate actions are validated against the last observed screen
        # size BEFORE execution. A model can hallucinate off-screen or negative
        # coordinates (e.g. from a stale screenshot after a resize); clicking
        # those does nothing useful and can mis-click. We BLOCK such actions —
        # without triggering recovery (the provider state is fine; the agent
        # simply needs to re-observe and pick valid coordinates).
        is_gui_burst = (
            action_type in {
                ComputerActionType.GUI_CLICK,
                ComputerActionType.GUI_DOUBLE_CLICK,
                ComputerActionType.GUI_RIGHT_CLICK,
                ComputerActionType.GUI_TYPE,
                ComputerActionType.GUI_KEYPRESS,
                ComputerActionType.GUI_MOVE,
                ComputerActionType.GUI_SCROLL,
                ComputerActionType.GUI_DRAG,
            }
            and ("burst_actions" in payload or payload.get("burst") is True)
        )
        coord_err = None if is_gui_burst else self._validate_coordinates(action_type, payload)
        if coord_err is not None:
            actual_obs_str = f"Coordinate out of bounds: {coord_err}"
            status = ActionExecutionStatus.BLOCKED
            logger.warning("action_blocked_out_of_bounds",
                           action=action_type.value, reason=coord_err,
                           screen=f"{self._screen_width}x{self._screen_height}")
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=actual_obs_str,
                expected_observation=predicted_outcome,
                info_gain=0.0,
                recovery_attempted=False,
                status=status,
                exit_code=126,
                thought=getattr(self, "_last_thought", ""),
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": action_type.value, "result": actual_obs_str})
            return trace

        # An explicit burst is opt-in; ordinary single actions retain their
        # existing dispatch path.  MotorReflexes owns cadence and fail-closed
        # stopping, while this layer owns perception/version/postcondition
        # verification and the replan signal.
        if (
            action_type in {
                ComputerActionType.GUI_CLICK,
                ComputerActionType.GUI_DOUBLE_CLICK,
                ComputerActionType.GUI_RIGHT_CLICK,
                ComputerActionType.GUI_TYPE,
                ComputerActionType.GUI_KEYPRESS,
                ComputerActionType.GUI_MOVE,
                ComputerActionType.GUI_SCROLL,
                ComputerActionType.GUI_DRAG,
            }
            and is_gui_burst
        ):
            try:
                burst_ok, actual_obs_str, action_exit_code = await self._execute_gui_burst(
                    workspace_id, payload, target_resource
                )
                status = (
                    ActionExecutionStatus.COMPLETED
                    if burst_ok
                    else ActionExecutionStatus.FAILED
                )
                recovery_needed = not burst_ok
            except Exception as exc:
                actual_obs_str = f"GUI burst rejected: {exc}; re-observe and replan"
                status = ActionExecutionStatus.FAILED
                action_exit_code = 126
                recovery_needed = True
            self.perception_bus.invalidate()
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                action_type=action_type,
                target_resource=target_resource,
                payload=str(payload),
                predicted_outcome=predicted_outcome,
                actual_observation=_safe_str(actual_obs_str),
                expected_observation=predicted_outcome,
                info_gain=1.0 if status == ActionExecutionStatus.COMPLETED else 0.0,
                recovery_attempted=recovery_needed,
                status=status,
                exit_code=action_exit_code,
                thought=getattr(self, "_last_thought", ""),
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({
                "thought": getattr(self, "_last_thought", ""),
                "action": f"{action_type.value} {target_resource}",
                "result": _safe_str(actual_obs_str),
            })
            if not burst_ok:
                self._inject_replan(goal=str(actual_obs_str))
            return trace

        _display_lease_lock = None
        if str(action_type.value).startswith(("GUI_", "APP_")):
            from sonic.computer_use.teamwork import DisplayLeaseManager

            _display_lease_lock = DisplayLeaseManager.get_lock(workspace_id)
            await _display_lease_lock.acquire()

        try:
            if action_type == ComputerActionType.GUI_CLICK:
                if payload.get("x") is None or payload.get("y") is None:
                    actual_obs_str = f"GUI_CLICK failed: integer pixel coordinates x,y required or visual target '{target_resource}' could not be resolved"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    x = int(payload.get("x", 0))
                    y = int(payload.get("y", 0))
                    pre_screen_b64 = getattr(self, "_last_screenshot_b64", "")
                    if self._last_screenshot_b64:
                        self._last_screenshot_b64 = draw_action_marker(
                            self._last_screenshot_b64, (x, y), label="CLICK", color="#00ffcc"
                        )
                    obs_after = await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.CLICK, x=x, y=y),
                    )
                    post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                    if post_screen_b64:
                        self._last_screenshot_b64 = post_screen_b64
                        self._update_screen_dims(obs_after)
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and clicked"
                    else:
                        actual_obs_str = f"Clicked at ({x}, {y})"
                    if pre_screen_b64 and post_screen_b64 and pre_screen_b64 == post_screen_b64:
                        self._consecutive_unchanged_screens += 1
                        if self._consecutive_unchanged_screens >= 2:
                            status = ActionExecutionStatus.FAILED
                            recovery_needed = True
                            actual_obs_str = (
                                f"{actual_obs_str} | [ANTI-STALL ALERT]: Consecutive clicks produced zero screen change (display unchanged). "
                                f"Coordinate ({x}, {y}) did not trigger UI response. Marked as FAILED to trigger strategy pivot."
                            )
                        else:
                            actual_obs_str = f"{actual_obs_str} | [VISUAL VERIFICATION]: Screen state unchanged after click — target element may be occluded, missing, or page is loading."
                    elif post_screen_b64:
                        self._consecutive_unchanged_screens = 0

            elif action_type == ComputerActionType.GUI_DOUBLE_CLICK:
                if payload.get("x") is None or payload.get("y") is None:
                    actual_obs_str = f"GUI_DOUBLE_CLICK failed: integer pixel coordinates x,y required or visual target '{target_resource}' could not be resolved"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    x = int(payload.get("x", 0))
                    y = int(payload.get("y", 0))
                    pre_screen_b64 = getattr(self, "_last_screenshot_b64", "")
                    if self._last_screenshot_b64:
                        self._last_screenshot_b64 = draw_action_marker(
                            self._last_screenshot_b64, (x, y), label="DOUBLE_CLICK", color="#ffaa00"
                        )
                    obs_after = await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.DOUBLE_CLICK, x=x, y=y),
                    )
                    post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                    if post_screen_b64:
                        self._last_screenshot_b64 = post_screen_b64
                        self._update_screen_dims(obs_after)
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and double-clicked"
                    else:
                        actual_obs_str = f"Double-clicked at ({x}, {y})"
                    if pre_screen_b64 and post_screen_b64 and pre_screen_b64 == post_screen_b64:
                        self._consecutive_unchanged_screens += 1
                        if self._consecutive_unchanged_screens >= 2:
                            status = ActionExecutionStatus.FAILED
                            recovery_needed = True
                            actual_obs_str = (
                                f"{actual_obs_str} | [ANTI-STALL ALERT]: Consecutive double-clicks produced zero screen change (display unchanged). "
                                f"Coordinate ({x}, {y}) did not trigger UI response. Marked as FAILED to trigger strategy pivot."
                            )
                        else:
                            actual_obs_str = f"{actual_obs_str} | [VISUAL VERIFICATION]: Screen state unchanged after double-click."
                    elif post_screen_b64:
                        self._consecutive_unchanged_screens = 0

            elif action_type == ComputerActionType.GUI_RIGHT_CLICK:
                if payload.get("x") is None or payload.get("y") is None:
                    actual_obs_str = f"GUI_RIGHT_CLICK failed: integer pixel coordinates x,y required or visual target '{target_resource}' could not be resolved"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    x = int(payload.get("x", 0))
                    y = int(payload.get("y", 0))
                    pre_screen_b64 = getattr(self, "_last_screenshot_b64", "")
                    if self._last_screenshot_b64:
                        self._last_screenshot_b64 = draw_action_marker(
                            self._last_screenshot_b64, (x, y), label="RIGHT_CLICK", color="#ff3366"
                        )
                    obs_after = await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.RIGHT_CLICK, x=x, y=y),
                    )
                    post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                    if post_screen_b64:
                        self._last_screenshot_b64 = post_screen_b64
                        self._update_screen_dims(obs_after)
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and right-clicked"
                    else:
                        actual_obs_str = f"Right-clicked at ({x}, {y})"

            elif action_type == ComputerActionType.GUI_TYPE:
                text = str(payload.get("text") or payload.get("value") or payload.get("input") or "")
                pre_screen_b64 = getattr(self, "_last_screenshot_b64", "")
                obs_after = None
                if hasattr(self, "motor") and self.motor:
                    if getattr(self, "gui_only", False):
                        obs_after = await self.computer.gui_action(
                            workspace_id,
                            GUIAction(
                                action=GUIActionType.TYPE,
                                text=text,
                                delay_ms=payload.get("delay_ms", 25),
                            ),
                        )
                    else:
                        await self.motor.human_type(workspace_id, text, delay_ms=payload.get("delay_ms", 25))
                else:
                    obs_after = await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.TYPE, text=text),
                    )
                if hasattr(self.computer, "screenshot") and not obs_after:
                    try:
                        obs_after = await self.computer.screenshot(workspace_id)
                    except Exception:
                        pass
                post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                if post_screen_b64:
                    self._last_screenshot_b64 = post_screen_b64
                    self._prev_screenshot_hash = hashlib.md5(post_screen_b64.encode()).hexdigest()
                    self._update_screen_dims(obs_after)
                actual_obs_str = f"Typed {len(text)} chars: {text[:50]}"

            elif action_type in (ComputerActionType.GUI_KEYPRESS, ComputerActionType.GUI_HOTKEY):
                key = str(payload.get("key") or payload.get("keys") or target_resource or "Return").strip()
                gui_act_type = getattr(GUIActionType, "HOTKEY", GUIActionType.KEYPRESS) if (action_type == ComputerActionType.GUI_HOTKEY or "+" in key) else GUIActionType.KEYPRESS
                obs_key = await self.computer.gui_action(
                    workspace_id,
                    GUIAction(action=gui_act_type, key=key),
                )
                key_b64 = getattr(obs_key, "screenshot_base64", "") if obs_key else ""
                if key_b64:
                    self._last_screenshot_b64 = key_b64
                    self._prev_screenshot_hash = hashlib.md5(key_b64.encode()).hexdigest()
                    self._update_screen_dims(obs_key)
                actual_obs_str = f"Pressed {'hotkey' if action_type == ComputerActionType.GUI_HOTKEY or '+' in key else 'key'}: {key}"

            elif action_type == ComputerActionType.GUI_MOVE:
                if payload.get("x") is None or payload.get("y") is None:
                    actual_obs_str = f"GUI_MOVE failed: coordinates required or visual target '{target_resource}' could not be resolved"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    x = int(payload.get("x", 0))
                    y = int(payload.get("y", 0))
                    obs_move = await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.MOVE, x=x, y=y),
                    )
                    move_b64 = getattr(obs_move, "screenshot_base64", "") if obs_move else ""
                    if move_b64:
                        self._last_screenshot_b64 = move_b64
                        self._prev_screenshot_hash = hashlib.md5(move_b64.encode()).hexdigest()
                        self._update_screen_dims(obs_move)
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and moved mouse"
                    else:
                        actual_obs_str = f"Moved mouse to ({x}, {y})"

            elif action_type == ComputerActionType.GUI_DRAG:
                # Press-move-release (a human drag). Source (x,y) may come from
                # TARGET "x,y" (parsed above) or payload; destination (x2,y2) / (end_x,end_y)
                # comes from payload. Lets the agent drag-drop a file or slide.
                raw_x = payload.get("x")
                raw_y = payload.get("y")
                raw_end_x = payload.get("end_x") if payload.get("end_x") is not None else payload.get("x2")
                raw_end_y = payload.get("end_y") if payload.get("end_y") is not None else payload.get("y2")
                coord_err = self._validate_coordinates(action_type, payload)
                if coord_err is not None or raw_x is None or raw_y is None or raw_end_x is None or raw_end_y is None:
                    actual_obs_str = f"GUI_DRAG failed: valid start (x, y) and end (end_x, end_y) coordinates required ({coord_err or 'missing coordinates'})"
                    status = ActionExecutionStatus.BLOCKED
                    action_exit_code = 126
                else:
                    x = int(raw_x)
                    y = int(raw_y)
                    x2 = int(raw_end_x)
                    y2 = int(raw_end_y)
                    obs_drag = await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.DRAG, x=x, y=y, x2=x2, y2=y2),
                    )
                    drag_b64 = getattr(obs_drag, "screenshot_base64", "") if obs_drag else ""
                    if drag_b64:
                        self._last_screenshot_b64 = drag_b64
                        self._prev_screenshot_hash = hashlib.md5(drag_b64.encode()).hexdigest()
                        self._update_screen_dims(obs_drag)
                    actual_obs_str = f"Dragged from ({x},{y}) to ({x2},{y2})"

            elif action_type == ComputerActionType.GUI_SCROLL:
                def_x = getattr(self, "_screen_width", 1280) // 2
                def_y = getattr(self, "_screen_height", 800) // 2
                x = int(payload.get("x") if payload.get("x") is not None else def_x)
                y = int(payload.get("y") if payload.get("y") is not None else def_y)
                delta = int(payload.get("delta", -3))
                obs_scroll = await self.computer.gui_action(
                    workspace_id,
                    GUIAction(action=GUIActionType.SCROLL, x=x, y=y, scroll_delta=delta),
                )
                scroll_b64 = getattr(obs_scroll, "screenshot_base64", "") if obs_scroll else ""
                if scroll_b64:
                    self._last_screenshot_b64 = scroll_b64
                    self._prev_screenshot_hash = hashlib.md5(scroll_b64.encode()).hexdigest()
                actual_obs_str = f"Scrolled at ({x}, {y}) delta={delta}"

            elif action_type == ComputerActionType.GUI_SCREENSHOT:
                screen = await self.computer.screenshot(workspace_id)
                self._last_screenshot_b64 = getattr(screen, "screenshot_base64", "")
                self._update_screen_dims(screen)
                actual_obs_str = f"Screenshot captured ({screen.width}x{screen.height})"

            elif action_type == ComputerActionType.GUI_WAIT:
                # A human waits for a wizard step or progress bar before acting.
                # Without this the agent re-screenshots a still-loading UI and
                # clicks stale coordinates. Wait a fixed duration, then refresh
                # the screen so the NEXT observation reflects the settled state.
                import asyncio as _asyncio
                seconds = max(0, min(int(payload.get("seconds", 2)), 30))
                await _asyncio.sleep(seconds)
                screen = await self.computer.screenshot(workspace_id)
                self._last_screenshot_b64 = getattr(screen, "screenshot_base64", "")
                self._update_screen_dims(screen)
                actual_obs_str = f"Waited {seconds}s; screen refreshed"

            elif action_type == ComputerActionType.APP_LAUNCH:
                raw_app = payload.get("app_name") or target_resource
                app_name = str(raw_app).strip(" *_\n\r\t`\"'") if raw_app else ""
                if not app_name or app_name.lower() in ("none", "null", ""):
                    actual_obs_str = "APP_LAUNCH failed: application name required"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    if "\n" in app_name:
                        app_name = app_name.split("\n")[0].strip(" *_\n\r\t`\"'")
                    extra_arg = payload.get("url") or payload.get("args") or payload.get("arguments") or payload.get("path") or payload.get("file") or ""
                    app_request = app_name
                    if extra_arg:
                        app_request = f"{app_request} {str(extra_arg).strip()}"
                    if hasattr(self.computer, "launch_application"):
                        launched = await self.computer.launch_application(workspace_id, app_request)
                        if not launched:
                            raise RuntimeError(f"application launch failed for '{app_name}'")
                    else:
                        await self.computer.gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.OPEN_APP, app_name=app_request),
                        )
                    import asyncio as _asyncio
                    await _asyncio.sleep(1.0)
                    screen = await self.computer.screenshot(workspace_id)
                    self._last_screenshot_b64 = getattr(screen, "screenshot_base64", "")
                    self._update_screen_dims(screen)
                    active_w = getattr(screen, "active_window", "") or app_name
                    actual_obs_str = f"Launched and focused {app_name} (active window: {active_w})"

            elif action_type == ComputerActionType.APP_CLOSE:
                raw_app = payload.get("app_name") or target_resource
                app_name = str(raw_app).strip(" *_\n\r\t`\"'") if raw_app else ""
                if not app_name or app_name.lower() in ("none", "null", ""):
                    actual_obs_str = "APP_CLOSE failed: application name required"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    if "\n" in app_name:
                        app_name = app_name.split("\n")[0].strip(" *_\n\r\t`\"'")
                    if hasattr(self.computer, "close_application"):
                        closed = await self.computer.close_application(workspace_id, app_name)
                        if closed is False:
                            raise RuntimeError(f"application close failed for '{app_name}'")
                    else:
                        await self.computer.gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.CLOSE_APP, app_name=app_name),
                        )
                    actual_obs_str = f"Closed {app_name}"

            elif action_type == ComputerActionType.APP_FOCUS:
                raw_app = payload.get("app_name") or target_resource
                app_name = str(raw_app).strip(" *_\n\r\t`\"'") if raw_app else ""
                if not app_name or app_name.lower() in ("none", "null", ""):
                    actual_obs_str = "APP_FOCUS failed: window or application name required"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    if "\n" in app_name:
                        app_name = app_name.split("\n")[0].strip(" *_\n\r\t`\"'")
                    obs_focus = await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.SELECT_WINDOW, app_name=app_name),
                    )
                    focus_b64 = getattr(obs_focus, "screenshot_base64", "") if obs_focus else ""
                    if focus_b64:
                        self._last_screenshot_b64 = focus_b64
                        self._prev_screenshot_hash = hashlib.md5(focus_b64.encode()).hexdigest()
                        self._update_screen_dims(obs_focus)
                    active_w = getattr(obs_focus, "active_window", "") or app_name
                    actual_obs_str = f"Focused window: {app_name} (active window: {active_w})"

            elif action_type == ComputerActionType.APP_INSTALL:
                # Install a package via the provider's install_application(),
                # which enforces the ApplicationPolicy allowlist (forbidden
                # packages are blocked). This is the sanctioned install path —
                # an agent must NOT bypass it with a raw `apt-get install`
                # TERMINAL_EXEC, which would skip the package-policy gate.
                if not hasattr(self.computer, "install_application"):
                    actual_obs_str = "Install not supported by this computer provider"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    package = payload.get("package") or payload.get("app_name") or target_resource
                    if not package:
                        actual_obs_str = "APP_INSTALL requires a package name"
                        recovery_needed = True
                        status = ActionExecutionStatus.FAILED
                    else:
                        ok, output = await self.computer.install_application(
                            workspace_id, package,
                        )
                        if ok:
                            actual_obs_str = f"Installed {package}: {(output or '')[:200]}"
                        else:
                            actual_obs_str = f"Install {package} blocked/failed: {(output or '')[:200]}"
                            # A policy-blocked install (forbidden package) is
                            # NOT a recovery trigger — the safety gate decided
                            # correctly; the agent should pick a different tool.
                            if "prohibited" not in (output or "").lower():
                                recovery_needed = True
                                status = ActionExecutionStatus.FAILED
                            else:
                                status = ActionExecutionStatus.BLOCKED

            elif action_type == ComputerActionType.FILE_READ:
                path = (payload.get("path") if isinstance(payload, dict) else None) or target_resource or "/home/sonic/workspace/README.md"
                content = await self.computer.read_file(workspace_id, path)
                actual_obs_str = f"Read {len(content)} bytes from {path}"

            elif action_type == ComputerActionType.FILE_WRITE:
                p_dict = payload if isinstance(payload, dict) else {}
                if isinstance(payload, str) and payload.strip().startswith("{"):
                    try:
                        parsed = json.loads(payload)
                        if isinstance(parsed, dict):
                            p_dict = parsed
                    except Exception:
                        pass
                path = p_dict.get("path") or target_resource or "/home/sonic/workspace/auth.py"
                content = (
                    p_dict.get("content")
                    or p_dict.get("code")
                    or p_dict.get("file_content")
                    or p_dict.get("text")
                    or p_dict.get("data")
                    or (payload if isinstance(payload, str) and not payload.strip().startswith("{") else "")
                )
                # Unwrap JSON envelope if content string is a JSON dictionary
                if isinstance(content, str) and content.strip().startswith("{") and any(k in content for k in ('"content"', '"code"', '"source"', '"text"')):
                    try:
                        parsed_json = json.loads(content)
                        if isinstance(parsed_json, dict):
                            content = (
                                parsed_json.get("content")
                                or parsed_json.get("code")
                                or parsed_json.get("source")
                                or parsed_json.get("text")
                                or content
                            )
                            if (not path or path == "/home/sonic/workspace/auth.py") and "path" in parsed_json:
                                path = str(parsed_json["path"])
                    except Exception:
                        pass
                if self.safety is not None:
                    path_verdict = self.safety.evaluate("FILE_WRITE", path, {"path": path})
                    if not path_verdict.allowed:
                        status = ActionExecutionStatus.BLOCKED
                        actual_obs_str = f"BLOCKED by safety policy: {path_verdict.reason}"
                        recovery_needed = False
                        path = ""
                if path:
                    ext = os.path.splitext(path.rstrip("/"))[1]
                    if not str(content).strip() and (path.endswith("/") or not ext):
                        await self.computer.terminal(workspace_id, f"mkdir -p {shlex.quote(path)}")
                        actual_obs_str = f"Created directory {path}"
                    else:
                        ok = await self.computer.write_file(workspace_id, path, content)
                        if not ok:
                            status = ActionExecutionStatus.FAILED
                            actual_obs_str = f"Failed to write file {path}"
                            recovery_needed = True
                        else:
                            actual_obs_str = f"Wrote patch ({len(content)} bytes) to {path}"

            elif action_type == ComputerActionType.TERMINAL_EXEC:
                raw_cmd = payload.get("command") if (isinstance(payload, dict) and "command" in payload) else target_resource
                if raw_cmd is None:
                    raw_cmd = ""
                if isinstance(raw_cmd, dict):
                    raw_cmd = raw_cmd.get("command", "")
                cmd = str(raw_cmd).strip().rstrip(',;').strip()
                cmd = re.sub(r'[\s,;]+$', '', cmd).strip()
                if cmd.startswith("{") and "command" in cmd:
                    try:
                        parsed_p = json.loads(cmd)
                        if isinstance(parsed_p, dict) and "command" in parsed_p:
                            cmd = str(parsed_p["command"]).strip().rstrip(',;').strip()
                            cmd = re.sub(r'[\s,;]+$', '', cmd).strip()
                    except Exception:
                        pass
                if not cmd.strip(' \t\r\n,;'):
                    cmd = ""
                if not cmd:
                    res = AgentStepResult(
                        False,
                        'Empty command provided',
                        step_index=self.action_counter,
                        target_resource=target_resource or "terminal",
                        payload=str(payload),
                        predicted_outcome=predicted_outcome,
                    )
                    self.traces.append(res)
                    self.history.append({"action": "TERMINAL_EXEC", "result": "Empty command provided"})
                    return res
                if isinstance(payload, dict):
                    payload["command"] = cmd
                cmd_str = self._clean_terminal_command(cmd)
                # Map /home/sonic to the actual sandbox home directory if running in Daytona
                provider_name = getattr(self.computer, "__class__", type(None)).__name__
                is_daytona = "Daytona" in provider_name or "daytona" in str(getattr(self, "_last_working_dir", "")).lower()
                real_home = getattr(self, "_last_working_dir", "")
                if "/workspace" in real_home:
                    real_home = real_home.split("/workspace")[0]
                if not real_home and is_daytona:
                    real_home = "/home/daytona"

                if real_home and real_home != "/home/sonic":
                    if "/home/sonic" in cmd_str:
                        cmd_str = cmd_str.replace("/home/sonic", real_home)
                    if "~/" in cmd_str:
                        cmd_str = cmd_str.replace("~/", f"{real_home}/")
                res = await self.computer.terminal(workspace_id, cmd_str)
                action_exit_code = getattr(res, "exit_code", None)
                safe_stdout = _safe_str(getattr(res, "stdout", "") or "")
                safe_stderr = _safe_str(getattr(res, "stderr", "") or "")
                action_stderr = safe_stderr
                combined_output = (f"{safe_stdout}\n{safe_stderr}".strip()) if (safe_stdout or safe_stderr) else ""
                actual_obs_str = combined_output or f"Exit {res.exit_code}"
                if res.exit_code != 0:
                    err_class, err_reason = classify_failure(
                        exit_code=res.exit_code,
                        stdout=safe_stdout,
                        stderr=safe_stderr,
                        provider=provider_name,
                        tool=tool_name,
                    )
                    fail_rec = self.failure_budget.record_failure(
                        tool=tool_name,
                        provider=provider_name,
                        error_class=err_class,
                        raw_error=safe_stderr or safe_stdout or f"Exit {res.exit_code}",
                    )
                    if self.failure_budget.is_strategy_exhausted(tool=tool_name, provider=provider_name):
                        actual_obs_str = f"{actual_obs_str} | Strategy exhausted ({fail_rec.count} failures). Retrying is blocked."
                        status = ActionExecutionStatus.FAILED
                        recovery_needed = False
                    elif res.exit_code == 124:
                        status = ActionExecutionStatus.TIMED_OUT
                        recovery_needed = True
                    elif res.exit_code in (125, 126):
                        status = ActionExecutionStatus.BLOCKED
                        recovery_needed = False
                    else:
                        status = ActionExecutionStatus.FAILED
                        recovery_needed = True
                else:
                    self.failure_budget.record_success(tool=tool_name, provider=provider_name)

                if hasattr(self, "scratchpad") and self.scratchpad:
                    self.scratchpad.extract_from_text(safe_stdout, source="terminal")
                    if safe_stderr:
                        self.scratchpad.extract_from_text(safe_stderr, source="terminal_err")

                if hasattr(self, "wire_telemetry") and self.wire_telemetry:
                    if "curl " in cmd_str or "http " in cmd_str:
                        url_m = re.search(r'https?://[^\s"\']+', cmd_str)
                        target_url = url_m.group(0) if url_m else "http://target"
                        method = "POST" if ("-X POST" in cmd_str or "-d " in cmd_str or "--data" in cmd_str) else "GET"
                        status_code = 200 if res.exit_code == 0 else 500
                        status_m = re.search(r'\b([1-5]\d{2})\b', res.stdout[:50])
                        if status_m:
                            try:
                                status_code = int(status_m.group(1))
                            except Exception:
                                pass
                        self.wire_telemetry.record_wire_event(
                            method=method,
                            url=target_url,
                            status_code=status_code,
                            response_body=res.stdout[:300],
                        )

            elif action_type == ComputerActionType.GIT_STATUS:
                res = await self.computer.terminal(workspace_id, "git status -s -b")
                action_exit_code = getattr(res, "exit_code", 0)
                actual_obs_str = (_safe_str(getattr(res, "stdout", "")) or _safe_str(getattr(res, "stderr", "")) or "git working tree clean").strip()
                if action_exit_code != 0:
                    status = ActionExecutionStatus.FAILED

            elif action_type == ComputerActionType.GIT_DIFF:
                res = await self.computer.terminal(workspace_id, "git diff --stat")
                action_exit_code = getattr(res, "exit_code", 0)
                actual_obs_str = (_safe_str(getattr(res, "stdout", "")) or _safe_str(getattr(res, "stderr", "")) or "no git diff changes").strip()
                if action_exit_code != 0:
                    status = ActionExecutionStatus.FAILED

            elif action_type == ComputerActionType.GIT_BRANCH:
                branch_name = str(payload.get("branch") or payload.get("name") or target_resource or "main").strip()
                res = await self.computer.terminal(workspace_id, f"git checkout -b {shlex.quote(branch_name)} || git checkout {shlex.quote(branch_name)}")
                action_exit_code = getattr(res, "exit_code", 0)
                actual_obs_str = (_safe_str(getattr(res, "stdout", "")) or _safe_str(getattr(res, "stderr", "")) or f"Switched to branch {branch_name}").strip()
                if action_exit_code != 0:
                    status = ActionExecutionStatus.FAILED

            elif action_type == ComputerActionType.GIT_COMMIT:
                msg = payload.get("message", "feat: automated patch")
                ok = await self.computer.git_action(workspace_id, "commit", message=msg)
                if not ok:
                    git_res = getattr(self.computer, "_last_git_result", None)
                    action_exit_code = getattr(git_res, "exit_code", 1) if git_res else 1
                    action_stderr = getattr(git_res, "stderr", "Git commit failed") if git_res else "Git commit failed"
                    if action_exit_code == 126:
                        status = ActionExecutionStatus.BLOCKED
                    else:
                        status = ActionExecutionStatus.FAILED
                    actual_obs_str = f"Git commit failed (exit {action_exit_code}): {action_stderr}"
                    recovery_needed = False
                else:
                    action_exit_code = 0
                    actual_obs_str = f"Committed change: {msg}"

            elif action_type == ComputerActionType.GIT_PUSH:
                remote = str(payload.get("remote") or target_resource or "origin").strip()
                branch = str(payload.get("branch") or "HEAD").strip()
                push_cmd = f"GIT_TERMINAL_PROMPT=0 git push -u {shlex.quote(remote)} {shlex.quote(branch)}"
                res = await self.computer.terminal(workspace_id, push_cmd)
                action_exit_code = getattr(res, "exit_code", 1)
                out_str = (_safe_str(getattr(res, "stdout", "")) + "\n" + _safe_str(getattr(res, "stderr", ""))).strip()
                if action_exit_code != 0:
                    status = ActionExecutionStatus.FAILED
                    actual_obs_str = f"Git push failed (exit {action_exit_code}): {out_str}"
                else:
                    actual_obs_str = f"Pushed to {remote} {branch}: {out_str}"

            elif action_type == ComputerActionType.SERVICE_ACTION:
                svc = payload.get("service", "xvfb")
                act = payload.get("action", "restart")
                svc_info = await self.computer.service_action(workspace_id, svc, act)
                actual_obs_str = f"Service {svc} is {svc_info.status}"

            elif action_type == ComputerActionType.BROWSER_NAVIGATE:
                raw_url = payload.get("url") or target_resource
                url = str(raw_url).strip(" *_\n\r\t`\"'") if raw_url else ""
                if "\n" in url:
                    url = url.split("\n")[0].strip(" *_\n\r\t`\"'")
                if not url or url.lower() in ("none", "not specified", "null", "/"):
                    url = "about:blank"
                elif not (url.startswith("http://") or url.startswith("https://") or url.startswith("about:") or url.startswith("file://")):
                    if "localhost" in url or "127.0.0.1" in url:
                        url = f"http://{url}"
                    elif "." in url and " " not in url:
                        url = f"https://{url}"

                if self.browser is not None:
                    snap = await self.browser.navigate(url)
                    self._last_browser_snapshot = snap
                    final_url = str(getattr(snap, "url", "") or "").strip()
                    browser_status_code = int(getattr(snap, "status_code", 0) or 0)
                    browser_response_body = str(getattr(snap, "html_content", "") or "")[:300]
                    if not final_url or browser_status_code <= 0:
                        raise RuntimeError("browser navigation produced no verified page response")
                    self._last_navigated_url = final_url
                    actual_obs_str = (
                        f"Navigated to {final_url} "
                        f"(HTTP {browser_status_code}, title: {getattr(snap, 'title', '')})"
                    )
                elif getattr(self, "gui_only", False):
                    await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.KEYPRESS, key="ctrl+l"),
                    )
                    await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.TYPE, text=url, delay_ms=25),
                    )
                    await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.KEYPRESS, key="Return"),
                    )
                    await asyncio.sleep(0.5)
                    screen = await self.computer.screenshot(workspace_id)
                    if (
                        getattr(screen, "desktop_state", "") == "NO_DISPLAY"
                        or (
                            not getattr(screen, "screenshot_base64", "")
                            and getattr(screen, "active_window", "Desktop") == "Desktop"
                        )
                    ):
                        raise RuntimeError("visible browser navigation could not be verified from a live screen")
                    self._last_navigated_url = url
                    actual_obs_str = f"Navigated visible browser application to {url}"
                else:
                    is_same_url = getattr(self, "_last_navigated_url", "") == url
                    disp = os.environ.get("DISPLAY", ":99" if hasattr(self.computer, "_docker_exec") or getattr(self.computer, "name", "") == "DockerComputerProvider" else ":0")
                    # Dynamically discover any active browser window via wmctrl -lx without hardcoding
                    win_probe = await self.computer.terminal(workspace_id, f"DISPLAY={disp} wmctrl -lx 2>/dev/null")
                    browser_win_id = ""
                    if win_probe and win_probe.stdout:
                        for line in win_probe.stdout.splitlines():
                            line_l = line.lower()
                            if "http" in line_l or "www" in line_l or "://" in line_l or "browser" in line_l or "web" in line_l:
                                parts = line.split()
                                if parts and parts[0].startswith("0x"):
                                    browser_win_id = parts[0]
                                    break

                    if is_same_url and browser_win_id:
                        # Re-focus existing browser without spawning duplicate tabs
                        focus_cmd = f"export DISPLAY={disp}; (wmctrl -i -a {browser_win_id} || xdotool windowactivate {browser_win_id})"
                        focus_result = await self.computer.terminal(workspace_id, focus_cmd)
                        if getattr(focus_result, "exit_code", 1) != 0:
                            raise RuntimeError("existing browser window could not be focused")
                        self._last_navigated_url = url
                        actual_obs_str = (
                            f"Browser is already active on {url}. Existing tab focused (duplicate tab prevented). "
                            "Proceed to interact with the webpage via GUI_CLICK on buttons, search bar, or scroll."
                        )
                    elif browser_win_id:
                        # Reuse existing browser window: focus, focus address bar via ctrl+l, type URL and press Return
                        nav_cmd = (
                            f"export DISPLAY={disp}; (xdotool windowactivate --sync {browser_win_id} || wmctrl -i -a {browser_win_id}) && "
                            f"DISPLAY={disp} xdotool key --clearmodifiers ctrl+l sleep 0.1 type --delay 15 -- {shlex.quote(url)} key Return 2>/dev/null"
                        )
                        nav_result = await self.computer.terminal(workspace_id, nav_cmd)
                        if getattr(nav_result, "exit_code", 1) != 0:
                            raise RuntimeError("active browser navigation command failed")
                        self._last_navigated_url = url
                        if hasattr(self, "motor") and self.motor:
                            await self.motor.enforce_tab_budget(workspace_id, max_tabs=3)
                        actual_obs_str = f"Navigated active browser tab to {url} (tab reused via address bar)"
                    else:
                        # Resolve the system's configured browser through POSIX
                        # alternatives and desktop-entry metadata, without
                        # naming a vendor-specific application.
                        b_probe = await self.computer.terminal(
                            workspace_id,
                            "for alt in x-www-browser sensible-browser xdg-open; do "
                            "command -v \"$alt\" 2>/dev/null && break; "
                            "done; "
                            "if ! command -v \"$alt\" >/dev/null 2>&1; then "
                            "desktop=$(xdg-settings get default-web-browser 2>/dev/null); "
                            "for d in /usr/share/applications/$desktop "
                            "$HOME/.local/share/applications/$desktop; do "
                            "[ -f \"$d\" ] || continue; "
                            "exec_line=$(grep -m1 '^Exec=' \"$d\" 2>/dev/null | cut -d= -f2-); "
                            "bin=$(printf '%s\\n' \"$exec_line\" | awk '{print $1}'); "
                            "command -v \"$bin\" 2>/dev/null && break; "
                            "done; "
                            "fi",
                        )
                        browser_bin = b_probe.stdout.strip() if (b_probe and b_probe.stdout.strip()) else ""
                        if not browser_bin:
                            self._last_navigated_url = url
                            actual_obs_str = (
                                "No application was selected or launched; "
                                "no compatible application was discovered in the live workstation state."
                            )
                            browser_bin = ""
                        if browser_bin:
                            cmd = (
                                f"DISPLAY={disp} nohup {shlex.quote(browser_bin)} "
                                f"{shlex.quote(url)} >/tmp/sonic_app_launch.log 2>&1 &"
                            )
                            launch_result = await self.computer.terminal(workspace_id, cmd)
                            if getattr(launch_result, "exit_code", 1) != 0:
                                raise RuntimeError("application launch command failed")
                            await asyncio.sleep(0.5)
                            screen = await self.computer.screenshot(workspace_id)
                            if (
                                getattr(screen, "desktop_state", "") == "NO_DISPLAY"
                                or (
                                    not getattr(screen, "screenshot_base64", "")
                                    and getattr(screen, "active_window", "Desktop") == "Desktop"
                                )
                            ):
                                raise RuntimeError("application launch could not be verified from a live screen")
                            self._last_navigated_url = url
                            actual_obs_str = f"Launched discovered application and navigated to {url}"

                if hasattr(self, "scratchpad") and self.scratchpad:
                    self.scratchpad.extract_from_text(url, source="url")
                if (
                    browser_status_code is not None
                    and hasattr(self, "wire_telemetry")
                    and self.wire_telemetry
                    and url.startswith("http")
                ):
                    self.wire_telemetry.record_wire_event(
                        method="GET",
                        url=url,
                        status_code=browser_status_code,
                        response_body=browser_response_body,
                    )
                if url.startswith("http"):
                    await self.check_and_resolve_intercept_deadlock(workspace_id)

            elif action_type == ComputerActionType.BROWSER_CLICK:
                selector = payload.get("selector") or target_resource
                if self.browser is not None:
                    ok = await self.browser.click(selector)
                    actual_obs_str = f"Clicked {selector}" if ok else f"Click failed: {selector}"
                    if not ok:
                        recovery_needed = True
                    await self.check_and_resolve_intercept_deadlock(workspace_id)
                else:
                    screen_w = getattr(self, "_screen_width", 1280)
                    screen_h = getattr(self, "_screen_height", 800)
                    res_coords: tuple[int, int] | None = None
                    if getattr(self, "perception_bus", None) is not None and selector:
                        semantic_match = self.perception_bus.resolve(str(selector))
                        if semantic_match is not None:
                            bx, by, bw, bh = semantic_match.bbox
                            res_coords = (bx + max(0, bw // 2), by + max(0, bh // 2))

                    if res_coords is None:
                        async def _ground_browser_click(
                            q: str, b64: str, w: int = screen_w, h: int = screen_h
                        ) -> tuple[int, int] | None:
                            return await query_multimodal_grounding(
                                llm_router=self.llm_router,
                                query=q,
                                screenshot_b64=b64,
                                width=w,
                                height=h,
                            )

                        res_coords = await resolve_ui_target_async(
                            str(selector),
                            screenshot_b64=self._last_screenshot_b64,
                            width=screen_w,
                            height=screen_h,
                            screen_width=screen_w,
                            screen_height=screen_h,
                            grounding_fn=_ground_browser_click if getattr(self, "llm_router", None) else None,
                            allow_landmarks=False,
                        )
                    if res_coords is not None:
                        cx, cy = res_coords
                        obs_after = await self.computer.gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.CLICK, x=cx, y=cy),
                        )
                        post_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                        if post_b64:
                            self._last_screenshot_b64 = post_b64
                            self._prev_screenshot_hash = hashlib.md5(post_b64.encode()).hexdigest()
                        actual_obs_str = f"Clicked browser element '{selector}' via visual grounding at ({cx}, {cy})"
                    else:
                        actual_obs_str = f"Browser element '{selector}' could not be resolved from screen; re-observing."
                        status = ActionExecutionStatus.FAILED

            elif action_type == ComputerActionType.BROWSER_SELECT:
                selector = str(payload.get("selector") or target_resource or "").strip()
                val = str(payload.get("value") or payload.get("option") or payload.get("label") or "").strip()
                if self.browser is not None and hasattr(self.browser, "select_option"):
                    ok = await self.browser.select_option(selector, val)
                    actual_obs_str = f"Selected '{val}' in {selector}" if ok else f"Select option failed for {selector}"
                    if not ok:
                        status = ActionExecutionStatus.FAILED
                else:
                    await self.computer.gui_action(workspace_id, GUIAction(action=GUIActionType.TYPE, text=val))
                    await self.computer.gui_action(workspace_id, GUIAction(action=GUIActionType.KEYPRESS, key="Return"))
                    actual_obs_str = f"Selected '{val}' via GUI keyboard"

            elif action_type == ComputerActionType.BROWSER_KEYPRESS:
                key = str(payload.get("key") or target_resource or "Return").strip()
                if self.browser is not None and hasattr(self.browser, "press_key"):
                    ok = await self.browser.press_key(key)
                    actual_obs_str = f"Pressed key '{key}' in browser" if ok else f"Browser keypress failed: {key}"
                else:
                    await self.computer.gui_action(workspace_id, GUIAction(action=GUIActionType.KEYPRESS, key=key))
                    actual_obs_str = f"Pressed key '{key}' via GUI keyboard"

            elif action_type == ComputerActionType.BROWSER_SCROLL:
                delta_y = int(payload.get("delta_y") or payload.get("delta") or 500)
                if self.browser is not None and hasattr(self.browser, "scroll"):
                    ok = await self.browser.scroll(delta_y=delta_y)
                    actual_obs_str = f"Scrolled browser page by {delta_y}px" if ok else "Browser scroll failed"
                else:
                    def_x = getattr(self, "_screen_width", 1280) // 2
                    def_y = getattr(self, "_screen_height", 800) // 2
                    await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.SCROLL, x=def_x, y=def_y, scroll_delta=-3 if delta_y > 0 else 3),
                    )
                    actual_obs_str = f"Scrolled browser window via GUI scroll (delta={delta_y})"

            elif action_type == ComputerActionType.WAIT_FOR_HUMAN:
                reason = str(payload.get("reason") or payload.get("prompt") or target_resource or "Human assistance requested (CAPTCHA/OTP/2FA)").strip()
                self._waiting_for_human = True
                self._waiting_for_human_reason = reason
                status = ActionExecutionStatus.WAITING_FOR_HUMAN
                actual_obs_str = f"WAITING_FOR_HUMAN: {reason}"

            elif action_type == ComputerActionType.BROWSER_TYPE:
                selector = str(payload.get("selector") or target_resource or "").strip(" *_\n\r\t`\"'")
                text = str(
                    payload.get("text")
                    or payload.get("value")
                    or payload.get("query")
                    or payload.get("input")
                    or ""
                )

                if self.browser is not None:
                    ok = await self.browser.type_text(selector, text)
                    actual_obs_str = f"Typed {len(text)} chars into {selector}" if ok else f"Type failed: {selector}"
                    if not ok:
                        recovery_needed = True
                else:
                    res_coords: tuple[int, int] | None = None
                    if selector:
                        if getattr(self, "perception_bus", None) is not None:
                            semantic_match = self.perception_bus.resolve(str(selector))
                            if semantic_match is not None:
                                bx, by, bw, bh = semantic_match.bbox
                                res_coords = (bx + max(0, bw // 2), by + max(0, bh // 2))
                        if res_coords is None:
                            screen_w = getattr(self, "_screen_width", 1280)
                            screen_h = getattr(self, "_screen_height", 800)

                            async def _ground_browser_type(
                                q: str, b64: str, w: int = screen_w, h: int = screen_h
                            ) -> tuple[int, int] | None:
                                return await query_multimodal_grounding(
                                    llm_router=self.llm_router,
                                    query=q,
                                    screenshot_b64=b64,
                                    width=w,
                                    height=h,
                                )

                            res_coords = await resolve_ui_target_async(
                                str(selector),
                                screenshot_b64=self._last_screenshot_b64,
                                width=screen_w,
                                height=screen_h,
                                screen_width=screen_w,
                                screen_height=screen_h,
                                grounding_fn=_ground_browser_type if getattr(self, "llm_router", None) else None,
                                allow_landmarks=False,
                            )
                        if res_coords is not None:
                            cx, cy = res_coords
                            await self.computer.gui_action(
                                workspace_id,
                                GUIAction(action=GUIActionType.CLICK, x=cx, y=cy),
                            )
                            await asyncio.sleep(0.1)

                    if text:
                        await self.computer.gui_action(workspace_id, GUIAction(action=GUIActionType.TYPE, text=text))
                        if payload.get("submit") is True or payload.get("press_enter") is True:
                            await self.computer.gui_action(workspace_id, GUIAction(action=GUIActionType.KEYPRESS, key="Return"))
                        actual_obs_str = f"Typed {len(text)} chars via GUI keyboard: {text[:40]}"
                    else:
                        actual_obs_str = f"Typed 0 chars via GUI keyboard (empty text for {selector})"

            elif action_type == ComputerActionType.BROWSER_SCREENSHOT:
                if self.browser is not None:
                    snap = None
                    if hasattr(self.browser, "screenshot") and callable(self.browser.screenshot):
                        snap = await self.browser.screenshot()
                    elif hasattr(self.browser, "_page") and self.browser._page:
                        try:
                            screenshot_bytes = await self.browser._page.screenshot(full_page=True, type="png")
                            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                            url = self.browser._page.url or ""
                            title = await self.browser._page.title()
                            from sonic.agents.browser_agent import PageSnapshot
                            snap = PageSnapshot(
                                url=url,
                                title=title,
                                status_code=getattr(self._last_browser_snapshot, "status_code", 200),
                                html_content=await self.browser._page.content() if hasattr(self.browser._page, "content") else "",
                                screenshot_b64=screenshot_b64,
                            )
                        except Exception as e:
                            logger.debug("browser_page_screenshot_error", error=str(e))
                    elif hasattr(self.browser, "current_page_state"):
                        try:
                            url, title = await self.browser.current_page_state()
                            if self._last_browser_snapshot:
                                snap = self._last_browser_snapshot
                            else:
                                from sonic.agents.browser_agent import PageSnapshot
                                snap = PageSnapshot(url=url, title=title, status_code=200, html_content="", screenshot_b64="")
                        except Exception:
                            snap = self._last_browser_snapshot
                    else:
                        snap = self._last_browser_snapshot

                    if snap is not None:
                        self._last_browser_snapshot = snap
                        b64 = getattr(snap, "screenshot_b64", "")
                        if b64:
                            self._last_screenshot_b64 = b64
                        url_str = getattr(snap, "url", "") or (getattr(self._last_browser_snapshot, "url", "") if self._last_browser_snapshot else "")
                        actual_obs_str = f"Screenshot captured: {url_str}"
                    else:
                        url_str = getattr(self._last_browser_snapshot, "url", "") if self._last_browser_snapshot else ""
                        actual_obs_str = f"Screenshot captured: {url_str}"
                else:
                    screen = await self.computer.screenshot(workspace_id)
                    self._update_screen_dims(screen)
                    actual_obs_str = f"Desktop screenshot captured ({screen.width}x{screen.height})"

            elif action_type == ComputerActionType.BROWSER_WAIT:
                # Wait for a page element to render before clicking it — a human
                # waits for a download button or "Next" to appear.
                selector = payload.get("selector") or target_resource
                if self.browser is not None and selector:
                    ok = await self.browser.wait_for_element(selector)
                    actual_obs_str = f"Element {selector} ready" if ok else f"Element {selector} not found within timeout"
                else:
                    import asyncio as _asyncio
                    await _asyncio.sleep(2)
                    actual_obs_str = "Waited 2s (no BrowserAgent)"

            elif action_type == ComputerActionType.BROWSER_DOWNLOAD:
                # Human step: click a download link on the website and save the
                # file to the sandbox workspace. save_path is confined to the
                # workspace root by the safety gate (FILE_WRITE-style).
                selector = payload.get("selector") or target_resource
                save_path = payload.get("save_path") or payload.get("path")
                if self.browser is None:
                    actual_obs_str = "Download requires BrowserAgent (Playwright)"
                    recovery_needed = True
                elif not selector or not save_path:
                    actual_obs_str = "BROWSER_DOWNLOAD requires selector and save_path"
                    recovery_needed = True
                else:
                    ok = await self.browser.download(selector, save_path)
                    if ok:
                        actual_obs_str = f"Downloaded {selector} to {save_path}"
                    else:
                        actual_obs_str = f"Download failed: {selector}"
                        recovery_needed = True

            elif action_type == ComputerActionType.SECURITY_TOOL:
                tool_name = payload.get("tool", target_resource)
                scan_target = payload.get("target", target_resource)
                args = payload.get("args", "")
                tool = self.security_tools.get(tool_name)
                if tool is None and self.security_tools:
                    actual_obs_str = f"Unknown security tool: {tool_name}"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                elif tool is None:
                    # Native execution directly inside sandbox via TERMINAL_EXEC
                    tool_name_raw = str(tool_name).strip()
                    if " " in tool_name_raw:
                        cmd_str = tool_name_raw
                        if args and str(args).strip() not in cmd_str:
                            cmd_str = f"{cmd_str} {str(args).strip()}"
                    else:
                        parts = [tool_name_raw]
                        if args:
                            parts.append(str(args).strip())
                        if scan_target and str(scan_target).strip() != tool_name_raw and str(scan_target).strip() not in str(args):
                            parts.append(str(scan_target).strip())
                        cmd_str = " ".join(parts).strip()

                    cmd_str = self._clean_terminal_command(cmd_str)
                    provider_name = getattr(self.computer, "__class__", type(None)).__name__
                    is_daytona = "Daytona" in provider_name or "daytona" in str(getattr(self, "_last_working_dir", "")).lower()
                    real_home = getattr(self, "_last_working_dir", "")
                    if "/workspace" in real_home:
                        real_home = real_home.split("/workspace")[0]
                    if not real_home and is_daytona:
                        real_home = "/home/daytona"

                    if real_home and real_home != "/home/sonic":
                        if "/home/sonic" in cmd_str:
                            cmd_str = cmd_str.replace("/home/sonic", real_home)
                        if "~/" in cmd_str:
                            cmd_str = cmd_str.replace("~/", f"{real_home}/")

                    res = await self.computer.terminal(workspace_id, cmd_str)
                    res_stdout = _safe_str(getattr(res, "stdout", "") or "")
                    res_stderr = _safe_str(getattr(res, "stderr", "") or "")
                    exit_c = getattr(res, "exit_code", 0)
                    action_exit_code = exit_c

                    out_str = res_stdout.strip()
                    if res_stderr.strip():
                        out_str = f"{out_str}\n{res_stderr.strip()}" if out_str else res_stderr.strip()
                    actual_obs_str = out_str or f"Tool {tool_name} exit {exit_c}"

                    from types import SimpleNamespace
                    findings = []
                    for line in res_stdout.splitlines():
                        line_s = line.strip()
                        if line_s:
                            findings.append({"finding": line_s, "source": str(tool_name)})

                    self._last_tool_result = SimpleNamespace(
                        tool_name=str(tool_name),
                        status="completed" if exit_c == 0 else "failed",
                        exit_code=exit_c,
                        raw_stdout=res_stdout,
                        raw_stderr=res_stderr,
                        parsed_data=findings,
                        error_message=res_stderr if exit_c != 0 else None,
                    )

                    if exit_c != 0:
                        err_class, _ = classify_failure(
                            exit_code=exit_c,
                            stdout=res_stdout,
                            stderr=res_stderr,
                            provider=provider_name,
                            tool=tool_name,
                        )
                        fail_rec = self.failure_budget.record_failure(
                            tool=tool_name,
                            provider=provider_name,
                            error_class=err_class,
                            raw_error=res_stderr or res_stdout or f"Exit {exit_c}",
                        )
                        if self.failure_budget.is_strategy_exhausted(tool=tool_name, provider=provider_name):
                            actual_obs_str = f"{actual_obs_str} | Strategy exhausted ({fail_rec.count} failures). Retrying is blocked."
                            status = ActionExecutionStatus.FAILED
                            recovery_needed = False
                        elif exit_c in (125, 126):
                            status = ActionExecutionStatus.BLOCKED
                            recovery_needed = True
                        elif exit_c == 124:
                            status = ActionExecutionStatus.TIMED_OUT
                            recovery_needed = True
                        else:
                            status = ActionExecutionStatus.FAILED
                            recovery_needed = True
                    else:
                        self.failure_budget.record_success(tool=tool_name, provider=provider_name)

                    if hasattr(self, "scratchpad") and self.scratchpad:
                        self.scratchpad.extract_from_text(res_stdout, source="security_tool")
                        if res_stderr:
                            self.scratchpad.extract_from_text(res_stderr, source="security_tool_err")

                    if hasattr(self, "wire_telemetry") and self.wire_telemetry:
                        if "curl " in cmd_str or "http " in cmd_str:
                            url_m = re.search(r'https?://[^\s"\']+', cmd_str)
                            target_url = url_m.group(0) if url_m else "http://target"
                            method = "POST" if ("-X POST" in cmd_str or "-d " in cmd_str or "--data" in cmd_str) else "GET"
                            status_code = 200 if exit_c == 0 else 500
                            status_m = re.search(r'\b([1-5]\d{2})\b', res_stdout[:50])
                            if status_m:
                                try:
                                    status_code = int(status_m.group(1))
                                except Exception:
                                    pass
                            self.wire_telemetry.record_wire_event(
                                method=method,
                                url=target_url,
                                status_code=status_code,
                                response_body=res_stdout[:300],
                            )
                else:
                    from types import SimpleNamespace
                    options = {"args": args} if args else {}
                    req = SimpleNamespace(
                        tenant_id=self.tenant_id,
                        engagement_id=self.engagement_id,
                        workspace_id=workspace_id,
                        agent_id=self.agent_id,
                        tool_name=tool_name,
                        target=scan_target,
                        options=options,
                        timeout_seconds=120,
                        execution_id=f"exec-{int(time.time() * 1000)}",
                    )
                    result = await tool.execute(req)
                    self._last_tool_result = result
                    findings = getattr(result, "parsed_data", []) or []
                    raw_out = _safe_str(getattr(result, "raw_stdout", "") or getattr(result, "raw_output", "") or "")
                    raw_err = _safe_str(getattr(result, "raw_stderr", "") or getattr(result, "error", "") or getattr(result, "error_message", "") or "")
                    actual_obs_str = (
                        f"Tool {tool_name} status={getattr(result, 'status', '?')} "
                        f"findings={len(findings)}"
                    )
                    if raw_out.strip() and not findings:
                        actual_obs_str = f"{actual_obs_str}\n{raw_out.strip()}"
                    # Fail-closed: a blocked/failed tool is a recovery trigger.
                    status_val = str(getattr(result, "status", "")).lower()
                    if status_val in ("blocked", "failed", "timed_out"):
                        exit_c = 126 if status_val == "blocked" else (124 if status_val == "timed_out" else 1)
                        action_exit_code = exit_c
                        err_class, _ = classify_failure(
                            exit_code=exit_c,
                            stdout="",
                            stderr=raw_err,
                            provider=provider_name,
                            tool=tool_name,
                        )
                        fail_rec = self.failure_budget.record_failure(
                            tool=tool_name,
                            provider=provider_name,
                            error_class=err_class,
                            raw_error=raw_err,
                        )
                        if self.failure_budget.is_strategy_exhausted(tool=tool_name, provider=provider_name):
                            actual_obs_str = f"{actual_obs_str} | Strategy exhausted ({fail_rec.count} failures). Retrying is blocked."
                            status = ActionExecutionStatus.FAILED
                            recovery_needed = False
                        elif status_val == "blocked":
                            status = ActionExecutionStatus.BLOCKED
                            recovery_needed = True
                        elif status_val == "timed_out":
                            status = ActionExecutionStatus.TIMED_OUT
                            recovery_needed = True
                        else:
                            status = ActionExecutionStatus.FAILED
                            recovery_needed = True
                    else:
                        action_exit_code = 0
                        self.failure_budget.record_success(tool=tool_name, provider=provider_name)

            elif action_type == ComputerActionType.TOOL_AUTHOR:
                code = payload.get("source") or payload.get("code") or payload.get("content")
                target_str = str(target_resource or "").strip()
                name_str = str(payload.get("name") or payload.get("tool") or payload.get("path") or "").strip()
                is_auto_verify = bool(payload.get("auto_verify") or payload.get("run"))

                is_file_target = (
                    any(s.endswith((".py", ".sh", ".bash", ".js")) for s in (target_str, name_str) if s)
                    or "/" in target_str
                    or "/" in name_str
                    or bool(payload.get("path"))
                )

                if is_file_target and not is_auto_verify and self.computer is not None:
                    # Direct script authoring to sandbox workspace
                    dest_path = (
                        payload.get("path")
                        or (target_str if target_str.startswith("/") else None)
                        or (name_str if name_str.startswith("/") else None)
                        or f"/home/sonic/workspace/{name_str or target_str}"
                    )
                    if self.safety is not None:
                        path_verdict = self.safety.evaluate("TOOL_AUTHOR", dest_path, {"path": dest_path})
                        if not path_verdict.allowed:
                            status = ActionExecutionStatus.BLOCKED
                            actual_obs_str = f"BLOCKED by safety policy: {path_verdict.reason}"
                            recovery_needed = False
                            dest_path = ""
                    # If code is missing, generate it via LLM
                    if dest_path and not code and self.llm_router is not None:
                        try:
                            from sonic.llm.schemas import LLMRequest, Message, MessageRole
                            gen_prompt = (
                                f"Author a concise, working script to satisfy this goal: {target_str or name_str}\n"
                                f"Target script path: {dest_path}\n"
                                "Output ONLY valid executable code without markdown blocks or conversational commentary."
                            )
                            resp = await self.llm_router.complete(
                                LLMRequest(
                                    messages=[Message(role=MessageRole.USER, content=gen_prompt)],
                                    max_tokens=800,
                                    temperature=0.2,
                                    task_type="coding",
                                )
                            )
                            code = (resp.content or "").strip()
                            if code.startswith("```"):
                                lines = code.splitlines()
                                if lines and lines[0].startswith("```"):
                                    lines = lines[1:]
                                if lines and lines[-1].strip() == "```":
                                    lines = lines[:-1]
                                code = "\n".join(lines).strip()
                        except Exception as gen_err:
                            logger.warning("script_generation_failed", error=str(gen_err))

                    if code:
                        try:
                            await self.computer.write_file(workspace_id, dest_path, code)
                            actual_obs_str = f"Authored script at '{dest_path}' ({len(code)} bytes). Next action: execute or verify via TERMINAL_EXEC."
                            status = ActionExecutionStatus.COMPLETED
                            action_exit_code = 0
                        except Exception as write_err:
                            actual_obs_str = f"Failed writing authored script '{dest_path}': {write_err}"
                            recovery_needed = True
                    else:
                        actual_obs_str = f"Could not determine code for '{dest_path}'"
                        recovery_needed = True
                else:
                    # Toolsmith (Phase A, AIOSR): the being authors a NEW tool for an
                    # observation gap or custom capability. The source is persisted
                    # to BeingCraft; the tool is NOT registered until confirmed in-sandbox.
                    if self.toolsmith is None and self.computer is not None and self.llm_router is not None:
                        try:
                            from sonic.being.craft import BeingCraft
                            from sonic.being.toolsmith import ToolsmithLoop
                            from sonic.tools.registry import SecurityToolRegistry
                            reg = SecurityToolRegistry(provider=self.computer)
                            self.toolsmith = ToolsmithLoop(
                                craft=BeingCraft(being_id=getattr(self, "agent_id", "computer-use-agent")),
                                llm=self.llm_router,
                                registry=reg,
                            )
                        except Exception as e:
                            logger.warning("lazy_toolsmith_init_failed", error=str(e))

                    if self.toolsmith is None:
                        actual_obs_str = "Toolsmith not configured (no tool authoring)"
                        recovery_needed = True
                    else:
                        observation = payload.get("observation", "") or target_resource
                        failed = payload.get("failed_attempts", [])
                        name = payload.get("name") or payload.get("tool")
                        source = payload.get("source") or payload.get("code")
                        rationale = payload.get("rationale", "")
                        authored = await self.toolsmith.author_tool(
                            observation=observation,
                            failed_attempts=list(failed) if failed else None,
                            name=name,
                            source=source,
                            rationale=rationale,
                        )
                        if authored is None:
                            actual_obs_str = "Toolsmith: no novel tool warranted (honest skip)"
                        else:
                            actual_obs_str = (
                                f"Authored tool '{authored.name}' (unconfirmed; "
                                f"run TOOL_RUN to verify): {authored.rationale}"
                            )
                            if payload.get("auto_verify") or payload.get("run"):
                                target = payload.get("target")
                                if not target:
                                    actual_obs_str = "TOOL_RUN requires an explicit target"
                                    recovery_needed = True
                                else:
                                    confirmed = await self.toolsmith.confirm_and_register(
                                        authored, self.computer, workspace_id, target=target,
                                    )
                                    if confirmed.reproduced:
                                        adapter = self.toolsmith.registry.get(confirmed.name) if getattr(self.toolsmith, "registry", None) else None
                                        if adapter is None:
                                            from sonic.being.toolsmith import AuthoredToolAdapter
                                            adapter = AuthoredToolAdapter(confirmed, self.computer)
                                        self.security_tools[confirmed.name] = adapter
                                        actual_obs_str = (
                                            f"Tool '{confirmed.name}' CONFIRMED and registered "
                                            f"(exit={confirmed.run_exit_code})"
                                        )
                                    else:
                                        actual_obs_str = (
                                            f"Tool '{confirmed.name}' NOT confirmed "
                                            f"(exit={confirmed.run_exit_code}); not registered"
                                        )
                                        recovery_needed = True

            elif action_type == ComputerActionType.TOOL_RUN:
                # Run a previously-authored tool in-sandbox and register it into
                # the security_tools map ONLY on a real successful run. Honesty
                # guard lives in ToolsmithLoop.confirm_and_register.
                if self.toolsmith is None:
                    actual_obs_str = "Toolsmith not configured (no tool running)"
                    recovery_needed = True
                else:
                    tool_name = payload.get("tool", target_resource)
                    authored = next(
                        (t for t in self.toolsmith.authored if t.name == tool_name), None
                    )
                    if authored is None:
                        actual_obs_str = f"No authored tool named '{tool_name}' to run"
                        recovery_needed = True
                    else:
                        target = payload.get("target")
                        if not target:
                            actual_obs_str = "TOOL_RUN requires an explicit target"
                            recovery_needed = True
                        else:
                            confirmed = await self.toolsmith.confirm_and_register(
                                authored, self.computer, workspace_id, target=target,
                            )
                            if confirmed.reproduced:
                                # Make the confirmed tool callable via SECURITY_TOOL.
                                adapter = self.toolsmith.registry.get(confirmed.name) if getattr(self.toolsmith, "registry", None) else None
                                if adapter is None:
                                    from sonic.being.toolsmith import AuthoredToolAdapter
                                    adapter = AuthoredToolAdapter(confirmed, self.computer)
                                self.security_tools[confirmed.name] = adapter
                                actual_obs_str = (
                                    f"Tool '{confirmed.name}' CONFIRMED and registered "
                                    f"(exit={confirmed.run_exit_code})"
                                )
                            else:
                                actual_obs_str = (
                                    f"Tool '{confirmed.name}' NOT confirmed "
                                    f"(exit={confirmed.run_exit_code}); not registered"
                                )
                                recovery_needed = True

            elif action_type == ComputerActionType.METHOD_INVENT:
                # Method-invention (Phase B, AIOSR): synthesize a NOVEL
                # workflow/method (a new METHOD, not just a tool) from the
                # observation + a prior failure + the known-technique ledger.
                # Confirmed ONLY on real in-sandbox reproduction — never by decree.
                if self.method_lab is None and getattr(self, "llm_router", None) is not None:
                    try:
                        from sonic.being.craft import BeingCraft
                        from sonic.being.method_lab import MethodLab
                        from sonic.being.toolsmith import ToolsmithLoop
                        from sonic.memory.vector import get_vector_memory
                        from sonic.tools.registry import SecurityToolRegistry
                        if self.toolsmith is None:
                            reg = SecurityToolRegistry(provider=self.computer)
                            self.toolsmith = ToolsmithLoop(
                                craft=BeingCraft(being_id=getattr(self, "agent_id", "computer-use-agent")),
                                llm=self.llm_router,
                                registry=reg,
                            )
                        self.method_lab = MethodLab(
                            llm=self.llm_router,
                            vector_memory=get_vector_memory(),
                            toolsmith=self.toolsmith,
                        )
                    except Exception as e:
                        logger.warning("lazy_method_lab_init_failed", error=str(e))
                if self.method_lab is None:
                    actual_obs_str = "MethodLab not configured (no technique invention)"
                    recovery_needed = True
                else:
                    observation = payload.get("observation", "") or target_resource
                    failure = payload.get("failure", "")
                    technique = await self.method_lab.invent(
                        observation=observation, failure=failure,
                    )
                    if technique is None:
                        actual_obs_str = "MethodLab: no novel technique warranted (honest skip)"
                    else:
                        # Run the probe in-sandbox; confirm only on a real finding.
                        target = payload.get("target", "")
                        if not target:
                            actual_obs_str = "METHOD_INVENT requires an explicit target"
                            recovery_needed = True
                        else:
                            confirmed = await self.method_lab.confirm(
                                technique, self.computer, workspace_id, target=target,
                            )
                            if confirmed.confirmed:
                                actual_obs_str = (
                                    f"Technique '{confirmed.name}' ({confirmed.family}) "
                                    f"CONFIRMED — {len(confirmed.findings)} finding(s); "
                                    f"novelty={confirmed.novelty_vs_ledger:.2f} vs ledger"
                                )
                            else:
                                actual_obs_str = (
                                    f"Technique '{confirmed.name}' NOT confirmed "
                                    f"(exit={confirmed.run_exit_code}); not added to ledger"
                                )
                                recovery_needed = True

            elif action_type == ComputerActionType.DEFINE_SUBAGENT:
                from sonic.computer_use.teamwork import SubagentTeamworkManager

                if getattr(self, "_teamwork_manager", None) is None:
                    self._teamwork_manager = SubagentTeamworkManager(
                        parent_agent=self,
                        computer_provider=self.computer,
                        llm_router=self.llm_router,
                        safety=self.safety,
                        tenant_id=self.tenant_id,
                    )
                sub_name = str(payload.get("name") or target_resource or "").strip()
                sub_desc = str(payload.get("description") or "")
                sub_sys = str(payload.get("system_prompt") or "")
                enable_write = bool(payload.get("enable_write_tools", True))
                spec = self._teamwork_manager.define_subagent(
                    name=sub_name,
                    description=sub_desc,
                    system_prompt=sub_sys,
                    enable_write_tools=enable_write,
                )
                actual_obs_str = (
                    f"Defined subagent type '{spec.name}' "
                    f"(write_tools={spec.enable_write_tools}): {spec.description}"
                )
                status = ActionExecutionStatus.VERIFIED

            elif action_type == ComputerActionType.INVOKE_SUBAGENT:
                from sonic.computer_use.teamwork import SubagentTeamworkManager

                if getattr(self, "_teamwork_manager", None) is None:
                    self._teamwork_manager = SubagentTeamworkManager(
                        parent_agent=self,
                        computer_provider=self.computer,
                        llm_router=self.llm_router,
                        safety=self.safety,
                        tenant_id=self.tenant_id,
                    )
                subagents_arg = payload.get("subagents") or payload.get("Subagents")
                if not subagents_arg:
                    subagents_arg = [
                        {
                            "type_name": str(payload.get("type_name") or "self"),
                            "role": str(payload.get("role") or target_resource or "Autonomous Subagent"),
                            "prompt": str(payload.get("prompt") or payload.get("goal") or target_resource or ""),
                            "requirements": list(payload.get("requirements") or []),
                            "acceptance_criteria": list(payload.get("acceptance_criteria") or []),
                            "verification_command": str(payload.get("verification_command") or ""),
                            "workspace_mode": str(payload.get("workspace_mode") or payload.get("workspace") or "inherit"),
                            "max_steps": int(payload.get("max_steps") or 12),
                        }
                    ]
                convs = await self._teamwork_manager.invoke_subagents(workspace_id, subagents_arg)
                summaries = [
                    f"[{c.conversation_id} | {c.role} | mode={c.workspace_mode} | judge={c.judge_verdict} | rounds={c.correction_rounds_used}]: {c.last_summary}"
                    for c in convs
                ]
                all_verified = all(c.verified for c in convs) if convs else False
                actual_obs_str = "Subagent Teamwork Results:\n" + "\n".join(summaries)
                status = ActionExecutionStatus.VERIFIED if all_verified else ActionExecutionStatus.COMPLETED

            elif action_type == ComputerActionType.SEND_MESSAGE:
                from sonic.computer_use.teamwork import SubagentTeamworkManager

                if getattr(self, "_teamwork_manager", None) is None:
                    self._teamwork_manager = SubagentTeamworkManager(
                        parent_agent=self,
                        computer_provider=self.computer,
                        llm_router=self.llm_router,
                        safety=self.safety,
                        tenant_id=self.tenant_id,
                    )
                recipient = str(payload.get("recipient") or payload.get("Recipient") or target_resource or "").strip()
                msg_text = str(payload.get("message") or payload.get("Message") or "")
                conv = await self._teamwork_manager.send_message(
                    conversation_id=recipient,
                    message=msg_text,
                    acceptance_criteria=list(payload.get("acceptance_criteria") or []),
                    verification_command=str(payload.get("verification_command") or ""),
                )
                actual_obs_str = (
                    f"[{conv.conversation_id} | judge={conv.judge_verdict}]: {conv.last_summary}"
                )
                status = ActionExecutionStatus.VERIFIED if conv.verified else ActionExecutionStatus.COMPLETED

            elif action_type == ComputerActionType.MANAGE_SUBAGENTS:
                from sonic.computer_use.teamwork import SubagentTeamworkManager

                if getattr(self, "_teamwork_manager", None) is None:
                    self._teamwork_manager = SubagentTeamworkManager(
                        parent_agent=self,
                        computer_provider=self.computer,
                        llm_router=self.llm_router,
                        safety=self.safety,
                        tenant_id=self.tenant_id,
                    )
                m_action = str(payload.get("action") or payload.get("Action") or "list")
                c_ids = list(payload.get("conversation_ids") or payload.get("ConversationIds") or [])
                listing = self._teamwork_manager.manage_subagents(action=m_action, conversation_ids=c_ids)
                actual_obs_str = json.dumps(listing, separators=(",", ":"))
                status = ActionExecutionStatus.COMPLETED

        except Exception as e:
            actual_obs_str = f"Error: {str(e)}"
            recovery_needed = True
            status = ActionExecutionStatus.FAILED
            err_class, _ = classify_failure(
                exit_code=1,
                stderr=str(e),
                provider=provider_name,
                tool=tool_name,
            )
            if hasattr(self, "failure_budget"):
                self.failure_budget.record_failure(
                    tool=tool_name,
                    provider=provider_name,
                    error_class=err_class,
                    raw_error=str(e),
                )
        finally:
            if _display_lease_lock is not None and _display_lease_lock.locked():
                _display_lease_lock.release()

        # Adaptive Closed-Loop Recovery if needed
        if recovery_needed:
            if self.recovery_events < self.max_recovery_attempts:
                rec_trace = await self.recover(workspace_id, action_type, actual_obs_str)
                actual_obs_str = f"{actual_obs_str} | Recovered: {rec_trace}"
                status = ActionExecutionStatus.RECOVERED
            else:
                if status in (ActionExecutionStatus.COMPLETED, ActionExecutionStatus.SUCCESS):
                    status = ActionExecutionStatus.FAILED

        # Closed-loop Target Feedback to Self-Evolution Engine
        if status in (ActionExecutionStatus.FAILED, ActionExecutionStatus.RECOVERED) and getattr(self, "evolution_engine", None) is not None:
            try:
                target_str = target_resource or str(self._last_action_output.get("target", "")) or self._last_navigated_url or "target"
                evo_result = await self.evolution_engine.handle_target_failure(
                    raw_output=actual_obs_str,
                    exit_code=action_exit_code if action_exit_code is not None else 1,
                    target=str(target_str),
                    current_approach=f"{action_type.value if hasattr(action_type, 'value') else action_type} on {target_str}",
                    goal=getattr(self, "current_goal", "") or "",
                    provider=self.computer,
                    workspace_id=workspace_id,
                )
                self._last_adapted_strategy = evo_result
            except Exception as e:
                logger.warning("evolution_engine_failure_dispatch_failed", error=str(e))

        # Automatic Hacker Scratchpad loot/token extraction
        if hasattr(self, "scratchpad") and self.scratchpad:
            self.scratchpad.extract_from_text(actual_obs_str, source=action_type.value.lower())

        # Reflexive backtracking on blocking modal overlays
        if not getattr(self, "gui_only", False) and hasattr(self, "motor") and self.motor:
            lower_obs = actual_obs_str.lower()
            if any(term in lower_obs for term in ("modal_blocked", "blocked by modal", "overlay detected", "dismiss modal")):
                await self.motor.backtrack(workspace_id, reason="modal_blocked")
                actual_obs_str = f"{actual_obs_str} | Reflexive backtrack: dismissed blocking modal"

        actual_obs_str = _safe_str(actual_obs_str)
        trace = ComputerDecisionTrace(
            step_index=self.action_counter,
            action_type=action_type,
            target_resource=target_resource,
            payload=str(payload),
            predicted_outcome=predicted_outcome,
            actual_observation=actual_obs_str,
            info_gain=1.0,
            recovery_attempted=recovery_needed,
            thought=getattr(self, "_last_thought", ""),
            status=status,
            exit_code=action_exit_code,
            duration_seconds=round(time.perf_counter() - t_start, 2),
            thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
        )
        after_state = self._current_experience_state(
            exit_code=action_exit_code,
            terminal_output=actual_obs_str
            if action_type in (ComputerActionType.TERMINAL_EXEC, ComputerActionType.SECURITY_TOOL)
            else "",
        )
        before_graph = getattr(self, "_experience_before_graph", None)
        after_graph = self._current_experience_graph()
        graph_delta = after_graph.delta(before_graph)
        graph_delta_text = (
            f"added={graph_delta['added']}; removed={graph_delta['removed']}"
        )
        experience = compare_prediction(
            action=action_type.value,
            predicted_outcome=predicted_outcome,
            before=before_state,
            after=after_state,
            evidence=actual_obs_str,
            graph_delta=graph_delta_text,
        )
        if not hasattr(self, "experience_records"):
            self.experience_records = []
        self.experience_records.append(experience)
        trace.state_changed = experience.state_changed
        trace.prediction_error = experience.prediction_error
        trace.graph_delta = experience.graph_delta
        before_receipt_state = self._contract_state(before_state)
        after_receipt_state = self._contract_state(after_state)
        trace.action_receipt = ActionReceipt(
            action_type=action_type.value,
            status=status,
            state_delta=StateDelta.from_states(
                before_receipt_state, after_receipt_state
            ),
            before=before_receipt_state,
            after=after_receipt_state,
            exit_code=action_exit_code,
            evidence=actual_obs_str,
        )
        if getattr(self, "skill_ledger", None) is not None:
            self._pending_skill_context = (trace, before_graph)
            trace_status = getattr(trace.status, "value", str(trace.status))
            if trace_status in {"SUCCESS", "SUCCEEDED", "COMPLETED", "VERIFIED"}:
                active_window = getattr(self.perception_bus.current(), "active_window", "")
                self.skill_ledger.record_success(
                    active_window,
                    action_type.value,
                    actual_obs_str,
                )
        self.traces.append(trace)
        # Store the real action output so observe() reads real terminal data
        # instead of a dummy probe. Strictly for sandbox command executions (TERMINAL_EXEC, SECURITY_TOOL).
        # GUI and Browser actions must NOT write into _last_action_output!
        if action_type in (ComputerActionType.TERMINAL_EXEC, ComputerActionType.SECURITY_TOOL):
            cmd_display = ""
            if isinstance(payload, dict):
                cmd_display = str(payload.get("command") or payload.get("tool") or payload.get("tool_name") or "")
            elif payload:
                cmd_display = str(payload)
            self._last_action_output = {
                "command": cmd_display[:200],
                "stdout": actual_obs_str[:2000],
                "stderr": action_stderr[:2000],
                "exit_code": action_exit_code,
            }
            self._goal_complete_declared = False
        else:
            self._last_gui_action_result = {
                "action_type": action_type.value if hasattr(action_type, "value") else str(action_type),
                "target": target_resource,
                "payload": payload,
                "observation": actual_obs_str,
                "exit_code": action_exit_code,
            }
        # Record into the reasoning history so the next LLM call sees what was
        # done, what was thought, and how it turned out — enabling true cognitive continuity.
        if action_type in {
            ComputerActionType.GUI_CLICK,
            ComputerActionType.GUI_DOUBLE_CLICK,
            ComputerActionType.GUI_RIGHT_CLICK,
            ComputerActionType.GUI_TYPE,
            ComputerActionType.GUI_KEYPRESS,
            ComputerActionType.GUI_MOVE,
            ComputerActionType.GUI_SCROLL,
            ComputerActionType.GUI_DRAG,
            ComputerActionType.GUI_WAIT,
            ComputerActionType.GUI_SCREENSHOT,
            ComputerActionType.APP_LAUNCH,
            ComputerActionType.APP_CLOSE,
            ComputerActionType.APP_FOCUS,
            ComputerActionType.APP_INSTALL,
        }:
            # Any desktop mutation invalidates semantic targets. The next
            # observation must refresh the whole computer state before another
            # target is resolved; this prevents stale clicks after window,
            # process, or filesystem changes.
            perception_bus = getattr(self, "perception_bus", None)
            if perception_bus is not None:
                perception_bus.invalidate()
        self.history.append({
            "thought": getattr(self, "_last_thought", ""),
            "action": f"{action_type.value} {target_resource}",
            "result": actual_obs_str,
        })
        return trace

    def _current_experience_state(
        self,
        exit_code: int | None = None,
        terminal_output: str = "",
    ) -> Any:
        """Build a transition signature from the latest local observations."""
        snapshot = getattr(getattr(self, "perception_bus", None), "current", lambda: None)()
        last_output = getattr(self, "_last_action_output", {}) or {}
        return state_signature(
            screen_hash=getattr(self, "_last_screenshot_b64", ""),
            active_window=getattr(snapshot, "active_window", ""),
            windows=getattr(snapshot, "windows", ()),
            processes=getattr(snapshot, "processes", ()),
            visible_text=getattr(snapshot, "visible_text", ""),
            terminal_output=terminal_output or str(last_output.get("stdout", "")),
            exit_code=exit_code,
            graph_hash=self._current_experience_graph().fingerprint(),
        )

    @staticmethod
    def _contract_state(state: Any) -> dict[str, Any]:
        """Expose the observed transition state in the action contract shape.

        ``StateSignature`` intentionally stores large observations as hashes;
        retaining those exact hashes keeps receipts compact while ensuring the
        delta is derived from the same live observations used by the trace.
        """
        return {
            "screen_hash": str(getattr(state, "screen_hash", "") or ""),
            "active_window": str(getattr(state, "active_window", "") or ""),
            "windows": tuple(getattr(state, "windows", ()) or ()),
            "processes": tuple(getattr(state, "processes", ()) or ()),
            "visible_text_hash": str(getattr(state, "visible_text_hash", "") or ""),
            "terminal_output_hash": str(
                getattr(state, "terminal_output_hash", "") or ""
            ),
            "exit_code": getattr(state, "exit_code", None),
            "graph_hash": str(getattr(state, "graph_hash", "") or ""),
        }

    def _current_experience_graph(self) -> GuiWorldGraph:
        bus = getattr(self, "perception_bus", None)
        if bus is None:
            return GuiWorldGraph(0, (), ())
        return bus.world_graph()

    async def author_tool(
        self,
        workspace_id: str,
        observation: str,
        name: str | None = None,
        source: str | None = None,
        rationale: str | None = None,
        target: str | None = None,
        auto_verify: bool = True,
    ) -> tuple[bool, str]:
        """Programmatically author, verify, and register a custom tool into the agent's toolkit."""
        trace = await self.execute_action(
            workspace_id=workspace_id,
            action_type=ComputerActionType.TOOL_AUTHOR,
            target_resource=name or "custom_tool",
            payload={
                "observation": observation,
                "name": name,
                "source": source,
                "rationale": rationale,
                "target": target,
                "auto_verify": auto_verify,
            },
            predicted_outcome="tool authored and verified",
        )
        success = "CONFIRMED and registered" in trace.actual_observation or (
            not auto_verify and "Authored tool" in trace.actual_observation
        )
        return success, trace.actual_observation

    # =============================================================
    # 4. Adaptive Recovery Intelligence
    # =============================================================
    async def recover(
        self,
        workspace_id: str,
        failed_action_type: ComputerActionType,
        error_context: str,
    ) -> str:
        """Generates closed-loop recovery actions to heal broken computer state."""
        self.recovery_events += 1
        logger.warning("computer_recovery_triggered", action=failed_action_type, error=error_context)

        if getattr(self, "gui_only", False):
            try:
                await self.computer.gui_action(
                    workspace_id,
                    GUIAction(action=GUIActionType.KEYPRESS, key="Escape"),
                )
            except Exception:
                pass
            try:
                comp_status = await self.computer.status(workspace_id)
                active_app = getattr(comp_status, "active_application", None)
                if active_app and str(active_app).lower() not in ("desktop", "none", ""):
                    await self.computer.gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.SELECT_WINDOW, app_name=active_app),
                    )
            except Exception:
                pass
            return "GUI-only recovery: dismissed visible modal and refreshed application focus"

        # Recovery strategy 1: Non-destructive desktop recovery for GUI actions
        if failed_action_type in [
            ComputerActionType.APP_LAUNCH, ComputerActionType.GUI_CLICK,
            ComputerActionType.GUI_DOUBLE_CLICK, ComputerActionType.GUI_TYPE,
            ComputerActionType.GUI_KEYPRESS, ComputerActionType.GUI_MOVE,
            ComputerActionType.GUI_SCROLL,
        ]:
            # Only restart Xvfb if there is an explicit display connection failure
            err_lower = (error_context or "").lower()
            if any(term in err_lower for term in (
                "display connection", "cannot open display", "x server died",
                "display failure", "failed to connect to display", "connection refused by server",
            )):
                await self.computer.service_action(workspace_id, "xvfb", "restart")
                return "Restarted Xvfb and refreshed display session"

            # Non-destructive recovery:
            # Re-focus the active application window (APP_FOCUS) without sending Escape
            # so open dropdown menus, forms, and dialogs are not accidentally dismissed.
            try:
                if hasattr(self.computer, "status"):
                    comp_status = await self.computer.status(workspace_id)
                    active_app = getattr(comp_status, "active_application", None)
                    if active_app and str(active_app).lower() not in ("desktop", "none", ""):
                        await self.computer.gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.SELECT_WINDOW, app_name=active_app),
                        )
            except Exception:
                pass

            return "Refreshed desktop focus and window state"

        # Recovery strategy 2: Restore workspace snapshot or re-verify file
        if failed_action_type == ComputerActionType.FILE_READ:
            return "Recovery blocked: the missing file must be restored from a real workspace snapshot or repository checkout"

        # Recovery strategy 3: Terminal PTY reset and diagnostic guidance
        if failed_action_type == ComputerActionType.TERMINAL_EXEC:
            err_lower = (error_context or "").lower()
            await self.computer.terminal(workspace_id, "stty sane")
            if any(term in err_lower for term in ("127", "not found", "command not found")):
                return (
                    "Reset terminal shell session: Command not found in container (exit 127). "
                    "Pivot to Python stdlib scalpel: author inline scripts in /workspace/tools/ using socket, urllib.request, http.client, json."
                )
            return "Reset terminal shell session"

        return "Generic recovery action applied"

    # =============================================================
    # 5. Full Autonomous Mission Execution Loop
    # =============================================================

    # Threshold: after this many consecutive failures in a row, the loop
    # considers the current approach stuck and injects a replan prompt rather
    # than repeating the same failing strategy until the step budget is gone.
    _STUCK_THRESHOLD: int = 3

    # Commands that gather no new information. Blocked after
    # ``_MAX_TRIVIAL_CONSECUTIVE`` consecutive uses.
    _TRIVIAL_COMMANDS: frozenset[str] = frozenset({
        "pwd", "whoami", "id", "uname", "echo", "true", "uname -m",
        "uname -a", "echo OK", "hostname", "date",
    })
    _MAX_TRIVIAL_CONSECUTIVE: int = 2

    def _is_trivial_or_repeated_action(
        self,
        action_type: ComputerActionType,
        target: str,
        payload: dict[str, Any],
    ) -> str | None:
        """Return a rejection reason if this action should be pre-emptively
        blocked, else ``None``.

        Called BEFORE ``execute_action`` so the step is never wasted.
        """
        if action_type == ComputerActionType.TERMINAL_EXEC:
            cmd = (payload.get("command") or target or "").strip().rstrip(',;').strip().lower()
            # Strip trailing semicolons/commas/whitespace for matching
            cmd_clean = cmd.rstrip(",; \t")
            if cmd_clean in self._TRIVIAL_COMMANDS:
                recent_trivial = 0
                for h in self.history[-4:]:
                    act = h.get("action", "")
                    if "TERMINAL_EXEC" in act:
                        res_lower = h.get("result", "").lower()
                        if any(tc in res_lower or tc in act.lower() for tc in self._TRIVIAL_COMMANDS):
                            recent_trivial += 1
                if recent_trivial >= self._MAX_TRIVIAL_CONSECUTIVE:
                    return (
                        f"BLOCKED: '{cmd_clean}' is a trivial info command and has been "
                        f"run {recent_trivial} times recently. Choose a goal-advancing "
                        f"action instead (e.g. author a tool/script, run curl, execute python/bash, inspect target, file edit, or app interaction)."
                    )

        # Block repeat of recent actions (within the last 4 actions)
        if self._recent_action_signatures:
            cmd_val = (
                (payload.get("command") or target or "").strip().rstrip(',;').strip().lower()
                if action_type == ComputerActionType.TERMINAL_EXEC
                else str(
                    payload.get("command")
                    or payload.get("tool")
                    or payload.get("text")
                    or payload.get("url")
                    or payload.get("path")
                    or payload.get("file_path")
                    or payload.get("content")
                    or target
                ).strip().lower()
            )
            new_sig = (
                action_type.value,
                str(target).strip().lower(),
                cmd_val,
            )
            is_exempt_navigation = action_type in (
                ComputerActionType.APP_FOCUS,
                ComputerActionType.GUI_HOTKEY,
                ComputerActionType.GUI_KEYPRESS,
                ComputerActionType.BROWSER_KEYPRESS,
                ComputerActionType.GUI_WAIT,
                ComputerActionType.GUI_SCREENSHOT,
                ComputerActionType.GUI_SCROLL,
                ComputerActionType.BROWSER_SCROLL,
                ComputerActionType.BROWSER_WAIT,
                ComputerActionType.WAIT_FOR_HUMAN,
            )
            if not is_exempt_navigation:
                recent_sigs = self._recent_action_signatures[-4:]
                for sig in recent_sigs:
                    if new_sig[0] == sig[0]:
                        sig_target = sig[1]
                        target_clean = new_sig[1]
                        is_repeat = False
                        if action_type == ComputerActionType.TERMINAL_EXEC:
                            # For terminal exec, must match the actual command executed.
                            # The stored signature payload is str(payload); parse it back
                            # so the comparison uses the same normalized command value.
                            sig_payload_str = _extract_command_from_sig(sig[2])
                            is_repeat = bool(cmd_val and cmd_val == sig_payload_str)
                        else:
                            sig_payload_str = sig[2].lower()
                            is_repeat = bool(
                                (target_clean and target_clean == sig_target and cmd_val == sig_payload_str)
                                or (cmd_val and cmd_val == sig_payload_str)
                                or (target_clean and target_clean == sig_target and not cmd_val)
                            )
                        if is_repeat:
                            cmd_desc = payload.get("command") or target or action_type.value
                            return (
                                f"BLOCKED: exact repeat of previous action "
                                f"'{action_type.value} {cmd_desc}'. Do NOT repeat actions. If the goal is satisfied, respond with ACTION: GOAL_COMPLETE. Otherwise choose a different, necessary action."
                            )

        return None

    async def check_and_resolve_intercept_deadlock(
        self,
        workspace_id: str,
        screen_observation: Any | None = None,
    ) -> bool:
        """
        Generic application window resolution to ensure browser and desktop workflows do not deadlock.
        """
        return False

    async def verify_goal(
        self,
        workspace_id: str,
        goal: str,
        *,
        use_llm: bool | None = None,
    ) -> tuple[bool, str]:
        """Independently verify whether a goal has been achieved.

        Supports LLM-driven verification when explicitly requested (use_llm=True
        or self.enable_llm_verification=True), asking the LLM to evaluate real
        evidence rather than circular self-judgment.

        When in standard mode, performs an independent sandbox check grounded
        in real system state (service/process/test artifacts) without theatrical
        echo stubs.

        Returns (verified, evidence). verified=True only on real evidence.
        """
        if getattr(self, "gui_only", False):
            obs = await self.observe(workspace_id)
            visible = " ".join(
                part for part in (
                    obs.active_application,
                    obs.visible_text,
                    getattr(obs.screen, "visible_text", ""),
                )
                if part
            )
            evidence = (
                f"Visible GUI state: application={obs.active_application!r}; "
                f"windows={obs.windows!r}; text={visible[:1500]!r}"
            )
            goal_terms = [
                term for term in re.findall(r"[a-zA-Z0-9]{3,}", goal.lower())
                if term not in {"open", "look", "find", "the", "and", "use", "for"}
            ]
            visible_lower = visible.lower()
            matched = sum(1 for term in goal_terms if term in visible_lower)
            verified = bool(goal_terms) and matched >= max(1, len(goal_terms) // 3)
            return verified, evidence

        do_llm = use_llm if use_llm is not None else getattr(self, "enable_llm_verification", False)

        if do_llm and self.llm_router is not None:
            async def _query_llm(prompt: str) -> str:
                from sonic.llm.schemas import LLMRequest, Message, MessageRole
                req = LLMRequest(
                    messages=[Message(role=MessageRole.USER, content=prompt)],
                    task_type="reasoning",
                )
                resp = await self.llm_router.complete(req)
                return resp.content

            # Step 1: Gather current observation for context
            obs = await self.observe(workspace_id)
            obs_summary = f"app={obs.active_application}, term={obs.terminal_output[:200] if obs.terminal_output else 'empty'}"

            # Step 2: Ask LLM to propose a verification command
            verify_prompt = (
                f"Goal: {goal}\n"
                f"Current observation: {obs_summary}\n\n"
                "Propose a single shell command that would verify whether this goal "
                "has been achieved. The command should produce clear output that "
                "proves success or failure. Respond with ONLY the command, nothing else.\n"
                "If no verification command makes sense, respond with: OBSERVE_ONLY"
            )

            evidence = ""
            try:
                llm_resp = await _query_llm(verify_prompt)
                verify_cmd = llm_resp.strip().split("\n")[0].strip()

                if verify_cmd and verify_cmd != "OBSERVE_ONLY" and len(verify_cmd) < 500:
                    try:
                        res = await self.computer.terminal(workspace_id, verify_cmd)
                        evidence = (res.stdout.strip() if res.stdout else "") or f"Exit {res.exit_code}"
                    except Exception as e:
                        evidence = f"Verify probe failed: {e}"
                else:
                    evidence = f"Observation: {obs_summary}"
            except Exception:
                evidence = f"Observation (LLM unavailable): {obs_summary}"

            # Step 3: Ask LLM to evaluate the evidence
            try:
                eval_prompt = (
                    f"Goal: {goal}\n"
                    f"Verification evidence:\n{evidence[:2000]}\n\n"
                    "Based on this evidence, has the goal been achieved? "
                    "Respond with exactly YES or NO on the first line, "
                    "followed by a brief explanation."
                )
                eval_resp = await _query_llm(eval_prompt)
                first_line = eval_resp.strip().split("\n")[0].upper()
                verified = "YES" in first_line
            except Exception:
                ev_lower = evidence.lower()
                verified = False

            return verified, evidence

        # Standard sandbox-grounded probe
        g = goal.lower().strip()
        verify_cmd: str | None = None
        if any(k in g for k in ("test", "pytest", "unittest")):
            verify_cmd = "python -m pytest -q --tb=short 2>&1 | tail -5"
        elif any(k in g for k in ("install", "set up", "setup")):
            m = re.search(r"install(?:\s+(?:the\s+)?)?([a-zA-Z0-9_.-]+)", g)
            pkg = m.group(1) if m else ""
            if pkg:
                verify_cmd = f"which {pkg} 2>/dev/null || dpkg -l {pkg} 2>/dev/null | grep ^ii"
        elif any(k in g for k in ("run", "start", "serve", "launch", "open")):
            obs = await self.observe(workspace_id)
            m_target = re.search(r"(?:run|start|serve|launch|open)\s+(?:the\s+)?([a-zA-Z0-9_\-\.]+)", g)
            target_prog = m_target.group(1).lower() if m_target else ""
            procs = " ".join(obs.processes) if obs.processes else ""
            open_windows = " ".join(getattr(obs, "windows", []))
            evidence = f"Running processes: {procs or 'none detected'} | Active window: {obs.active_application} | Open windows: {open_windows}"
            if target_prog:
                verified = (
                    target_prog in procs.lower()
                    or target_prog in obs.active_application.lower()
                    or target_prog in open_windows.lower()
                )
            else:
                verified = len(obs.processes) > 0 and obs.active_application != ""
            return verified, evidence
        elif any(k in g for k in ("file", "create", "write", "save", "generate", "report")):
            m_file = re.search(r'([a-zA-Z0-9_\-\./]+\.(?:txt|md|py|json|sh|csv|html|log))', goal)
            if m_file:
                target_file = m_file.group(1)
                if not target_file.startswith("/"):
                    target_file = f"/home/sonic/workspace/{target_file}"
                verify_cmd = f"test -f {shlex.quote(target_file)} && [ -s {shlex.quote(target_file)} ] && echo 'FILE_VERIFIED: {target_file}' && head -n 5 {shlex.quote(target_file)}"

        evidence = ""
        has_independent_probe = bool(verify_cmd)
        if verify_cmd:
            try:
                res = await self.computer.terminal(workspace_id, verify_cmd)
                evidence = (res.stdout.strip() if res.stdout else "") or f"Exit {res.exit_code}"
            except Exception as e:
                evidence = f"Verify probe failed: {e}"
        else:
            obs = await self.observe(workspace_id)
            evidence = f"Re-observed: app={obs.active_application}, term={obs.terminal_output[:120]}"
            evidence += " | No independent verification probe was inferred."

        ev_lower = evidence.lower()
        # Fail-closed verification: reject dummy/placeholder observations as
        # evidence of goal completion. Only REAL command output counts.
        _DUMMY_MARKERS = (
            "__sonic_obs_ready__",
            "generic recovery action",
            "terminal ready (0 procs)",
        )
        _is_dummy = any(marker in ev_lower for marker in _DUMMY_MARKERS)
        if _is_dummy:
            return False, f"Verification inconclusive — dummy marker detected. Evidence: {evidence}"

        _raw_has_content = bool(evidence.strip())
        if "Re-observed: " in evidence:
            _has_successful_action = any(
                getattr(a, "status", None) == ActionExecutionStatus.SUCCESS
                and getattr(a, "action_type", None)
                not in (ComputerActionType.WAIT_FOR_HUMAN, ComputerActionType.GOAL_COMPLETE)
                for a in getattr(self, "action_history", [])
            )
            _has_concrete_evidence = (
                "term=" in evidence
                and not evidence.rstrip().endswith("term=")
                and (
                    _has_successful_action
                    or bool(getattr(obs, "active_application", ""))
                )
            )
        else:
            _has_concrete_evidence = _raw_has_content
        _no_error_markers = (
            "error" not in ev_lower
            and "traceback" not in ev_lower
            and "exception" not in ev_lower
            and "failed" not in ev_lower
        )
        verified = _has_concrete_evidence and _no_error_markers
        return verified, evidence

    def _inject_replan(self, goal: str) -> None:
        """Signal the reasoning loop to abandon the current approach.

        When the agent is stuck (N consecutive failures), a human reassesses:
        "this isn't working, try a different angle." We inject that signal into
        the history so the NEXT LLM call sees it and pivots, rather than
        repeating the same failing action until the step budget is exhausted.
        """
        self._replan_count += 1
        if self._replan_count > 3:
            logger.warning("mission_max_replans_exceeded", replan_count=self._replan_count)
            return
        self._consecutive_failures = 0
        self._goal_complete_declared = False
        if getattr(self, "checklist", None) is not None:
            active_sg = self.checklist.active_sub_goal()
            if active_sg:
                active_sg.status = SubGoalStatus.FAILED
                active_sg.evidence = "Failed repeatedly during execution — pivoting strategy."
                self.checklist.advance()

        # Collect failed command patterns from recent history
        failed_cmds = [
            h.get("result", "")[:60] for h in self.history[-5:]
            if any(marker in h.get("result", "").lower() for marker in ("failed", "blocked", "error", "exit 126", "exit 1"))
        ]
        avoid_clause = f" FAILED ATTEMPTS TO AVOID: {failed_cmds}." if failed_cmds else ""
        replan_note = (
            f"REPLAN #{self._replan_count}: the last approach failed repeatedly.{avoid_clause} "
            f"Goal remains: {goal}. Re-assess from the current observation and "
            f"choose a DIFFERENT strategy — do not repeat the failing action."
        )
        self.history.append({"action": "REPLAN", "result": replan_note})
        logger.warning(
            "mission_replan_triggered",
            replan_count=self._replan_count,
            goal=goal,
        )

    async def run_mission(
        self,
        workspace_id: str,
        goal: str,
        steps: int = 5,
        step_callback: Any | None = None,
        interrupt_check: Any | None = None,
    ) -> list[ComputerDecisionTrace]:
        """Runs an end-to-end closed-loop autonomous engineering mission."""
        t_start = time.perf_counter()
        self._interrupted = False
        goal_reached = False

        if getattr(self, "checklist", None) is None or self.checklist.top_level_goal != goal:
            has_explicit_steps = any(re.match(r'^(?:\d+[\.\)]|\-|\*)\s+', line.strip()) for line in goal.splitlines())
            if getattr(self, "enable_llm_decomposition", False) or has_explicit_steps:
                self.checklist = await self.decompose_goal(goal)
            else:
                self.checklist = None

        # Backend bootstrap may inspect/install packages through a shell. It is
        # intentionally unavailable on the GUI-only Computer plane.
        if not getattr(self, "gui_only", False) and not getattr(self, "_bootstrap_performed", False):
            try:
                from sonic.computer.bootstrap import WorkstationBootstrapEngine
                bootstrap = WorkstationBootstrapEngine(self.computer)
                b_res = await bootstrap.ensure_workstation_ready(workspace_id=workspace_id)
                logger.info(
                    "mission_workstation_bootstrap_audit",
                    status=b_res.get("status"),
                    present_count=len(b_res.get("present", [])),
                    missing=b_res.get("missing", []),
                )
                self._bootstrap_performed = True
            except Exception as b_err:
                logger.warning("mission_workstation_bootstrap_failed", error=str(b_err))

        for step in range(1, steps + 1):
            if self._interrupted or (callable(interrupt_check) and interrupt_check()):
                self._interrupted = True
                logger.info("mission_interrupted_by_user", step=step)
                break
            if self.action_counter >= self.max_actions:
                logger.warning("max_actions_reached", max=self.max_actions)
                break

            # 1. OBSERVE
            obs = await self.observe(workspace_id)

            # 2. REASON & CHOOSE ACTION
            action_type, target, payload, expected = await self.choose_action(goal, obs, step)

            # Goal-complete sentinel: the LLM JUDGED the goal achieved — but we
            # do not stop on self-declaration alone. We independently verify.
            if expected == _GOAL_COMPLETE_SENTINEL:
                verified, evidence = await self.verify_goal(workspace_id, goal)
                if verified:
                    if self.traces:
                        self.traces[-1].verification_evidence = evidence[:500]
                        self.traces[-1].status = ActionExecutionStatus.VERIFIED
                    logger.info(
                        "mission_goal_complete_verified",
                        step=step,
                        actions_taken=self.action_counter,
                        evidence=evidence[:200],
                    )
                    if getattr(self, "checklist", None) is not None:
                        for sg in self.checklist.sub_goals:
                            if sg.status not in (SubGoalStatus.COMPLETED, SubGoalStatus.FAILED):
                                sg.status = SubGoalStatus.SKIPPED
                                sg.evidence = "Top-level goal satisfied"
                    goal_reached = True
                    break

                unverified_count = getattr(self, "_consecutive_unverified_declarations", 0) + 1
                self._consecutive_unverified_declarations = unverified_count
                if unverified_count >= 3:
                    logger.info("mission_goal_complete_unverified_max_attempts_exceeded")
                    goal_reached = False
                    break

                # Self-declared done but independent verification FAILED: the
                # goal is NOT actually achieved. Feed the evidence back so the
                # LLM corrects course instead of stopping on a false positive.
                self.history.append({
                    "action": "GOAL_COMPLETE (self-declared)",
                    "result": f"VERIFICATION FAILED: {evidence[:200]}. Goal NOT achieved — continue.",
                })
                self._consecutive_failures += 1
                if getattr(self, "checklist", None) is not None:
                    active_sg = self.checklist.active_sub_goal()
                    if active_sg:
                        active_sg.attempt_count += 1
                if self._consecutive_failures >= self._STUCK_THRESHOLD:
                    self._inject_replan(goal)
                continue

            # 2.5 PRE-EXECUTION GUARD: block trivial/repeated actions before
            # they waste a step. The rejection reason is injected into history
            # so the LLM sees it and pivots.
            rejection = self._is_trivial_or_repeated_action(action_type, target, payload)
            if rejection:
                self.history.append({
                    "action": f"{action_type.value} {target} (BLOCKED)",
                    "result": rejection,
                })
                # Successful output is not independent goal verification. Do
                # not turn a repeated-action guard into a completion shortcut.
                has_successful_cmds = any(
                    (t.status.value if hasattr(t.status, "value") else str(t.status)) in ("SUCCESS", "SUCCEEDED", "COMPLETED", "VERIFIED")
                    and t.actual_observation and len(t.actual_observation.strip()) > 0
                    for t in self.traces
                )
                is_query_goal = any(kw in goal.lower() for kw in ("check", "inspect", "show", "display", "get", "what is", "status", "list", "view", "find", "tell me"))
                if has_successful_cmds and is_query_goal:
                    self.history.append({
                        "action": "VERIFICATION_REQUIRED",
                        "result": "A prior action produced output, but the goal is not complete until an independent verification succeeds.",
                    })
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._STUCK_THRESHOLD:
                    self._inject_replan(goal)
                continue

            # 3. ACT & VERIFY
            trace = await self.execute_action(workspace_id, action_type, target, payload, expected)

            # Invoke real-time step streaming callback if provided
            if step_callback is not None:
                try:
                    cb_res = step_callback(trace)
                    if asyncio.iscoroutine(cb_res):
                        await cb_res
                except Exception as cb_err:
                    logger.warning("run_mission_step_callback_failed", error=str(cb_err))

            # A policy denial is a terminal state for this mission attempt.
            # Continuing would only repeat the same unauthorized action and
            # turn a real safety decision into scripted retry noise. The
            # operator can change scope/approval and start a fresh attempt.
            trace_observation = str(getattr(trace, "actual_observation", "") or "")
            prediction_error = float(getattr(trace, "prediction_error", 0.0) or 0.0)
            action_type_val = getattr(action_type, "value", str(action_type))
            if prediction_error >= 0.75 and trace.status not in (
                ActionExecutionStatus.BLOCKED,
                "BLOCKED",
            ):
                self.history.append({
                    "action": f"{action_type_val} {target} (PREDICTION_ERROR)",
                    "result": (
                        f"Observed transition diverged from the prediction "
                        f"(error={prediction_error:.1f}). Re-observe before repeating."
                    ),
                })
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._STUCK_THRESHOLD:
                    self._inject_replan(goal)

            if trace.status == ActionExecutionStatus.WAITING_FOR_HUMAN or trace_observation.startswith("WAITING_FOR_HUMAN:"):
                self.history.append({
                    "action": f"{action_type_val} {target} (WAITING_FOR_HUMAN)",
                    "result": trace_observation,
                })
                logger.info(
                    "mission_paused_waiting_for_human",
                    action=action_type_val,
                    target=target,
                    reason=trace_observation[:200],
                )
                break

            if (
                trace.status == ActionExecutionStatus.BLOCKED
                and trace_observation.startswith("Safety blocked:")
            ):
                self.history.append({
                    "action": f"{action_type_val} {target} (BLOCKED)",
                    "result": trace_observation,
                })
                logger.info(
                    "mission_stopped_blocked_action",
                    action=action_type_val,
                    target=target,
                    reason=trace_observation[:200],
                )
                break

            # Track consecutive failures for stuck/replan detection.
            is_failed_status = trace.status in (
                ActionExecutionStatus.FAILED,
                ActionExecutionStatus.TIMED_OUT,
                ActionExecutionStatus.BLOCKED,
                ActionExecutionStatus.CANCELLED,
                "FAILED",
                "TIMED_OUT",
                "BLOCKED",
                "CANCELLED",
            )
            if is_failed_status or trace.recovery_attempted:
                self._consecutive_failures += 1
                if getattr(self, "checklist", None) is not None:
                    active_sg = self.checklist.active_sub_goal()
                    if active_sg:
                        active_sg.attempt_count += 1
                if self._consecutive_failures >= self._STUCK_THRESHOLD:
                    self._inject_replan(goal)
            elif trace.status in (
                ActionExecutionStatus.COMPLETED,
                ActionExecutionStatus.SUCCESS,
                ActionExecutionStatus.VERIFIED,
                "COMPLETED",
                "SUCCESS",
                "VERIFIED",
            ):
                obs_txt = str(getattr(trace, "actual_observation", "") or "")
                if "[VISUAL VERIFICATION]: Screen state unchanged" not in obs_txt:
                    self._consecutive_failures = 0
                self._consecutive_unverified_declarations = 0
                if getattr(self, "checklist", None) is not None:
                    active_sg = self.checklist.active_sub_goal()
                    if active_sg:
                        active_sg.attempt_count += 1
                        obs_txt = str(getattr(trace, "actual_observation", "") or "")
                        if "[VISUAL VERIFICATION]: Screen state unchanged" not in obs_txt:
                            cmd_str = str(getattr(trace, "payload", "") or "").strip().lower()
                            trivial_cmds = ("pwd", "whoami", "uname", "id", "echo", "true")
                            is_trivial = any(cmd_str == tc or cmd_str.startswith(f"{tc} ") for tc in trivial_cmds)
                            if not is_trivial:
                                # Semantic relevance gate: the action or its
                                # output must share keywords with the sub-goal.
                                # This prevents `ls` from completing "Find SQL
                                # injection" (zero overlap) while allowing
                                # `sqlmap --url target` to complete it.
                                _stop = {"the", "a", "an", "in", "on", "to", "for",
                                         "of", "and", "or", "is", "it", "with", "run",
                                         "use", "check", "find", "get", "set"}
                                sg_words = set(re.findall(r'\b[a-z0-9_.-]+\b', active_sg.description.lower())) - _stop
                                action_text = f"{obs_txt[:300]} {cmd_str} {getattr(action_type, 'value', str(action_type))}".lower()
                                action_words = set(re.findall(r'\b[a-z0-9_.-]+\b', action_text)) - _stop
                                overlap = sg_words & action_words
                                relevance = len(overlap) / max(len(sg_words), 1)
                                if relevance >= 0.10 or len(overlap) >= 1:
                                    self.checklist.mark_active_completed(
                                        evidence=f"{action_type} succeeded: {obs_txt[:100]}"
                                    )
                                # else: action succeeded but doesn't relate to
                                # the sub-goal — don't mark it complete.

        # Update Telemetry Metrics.
        self.metrics.actions_total = len(self.traces)
        self.metrics.actions_successful = sum(
            1 for t in self.traces if t.status in (
                ActionExecutionStatus.COMPLETED,
                ActionExecutionStatus.SUCCESS,
                ActionExecutionStatus.VERIFIED,
                "COMPLETED",
                "SUCCESS",
                "VERIFIED",
                # REMOVED: RECOVERED — a failed action that ran `clear || true`
                # is NOT a success. Counting it as such inflates metrics and
                # corrupts the learning loop.
            )
        )
        self.metrics.actions_failed = sum(
            1 for t in self.traces if t.status in (
                ActionExecutionStatus.FAILED,
                ActionExecutionStatus.TIMED_OUT,
                ActionExecutionStatus.BLOCKED,
                ActionExecutionStatus.CANCELLED,
                ActionExecutionStatus.RECOVERED,  # RECOVERED = failed + recovery attempt
                "FAILED",
                "TIMED_OUT",
                "BLOCKED",
                "CANCELLED",
                "RECOVERED",
            )
        )
        self.metrics.recovery_events = self.recovery_events

        if not goal_reached and self.metrics.actions_successful > 0 and self.metrics.actions_failed == 0:
            try:
                verified, _ = await self.verify_goal(workspace_id, goal)
                if verified:
                    goal_reached = True
            except Exception as v_err:
                logger.debug("post_mission_verify_error", error=str(v_err))

        self.goal_reached = goal_reached
        t_elapsed = time.perf_counter() - t_start
        self.metrics.time_to_completion_seconds = round(t_elapsed, 2)
        # Honest verification score: ratio of real successes to total actions,
        # not a hardcoded 0.90 that hides failures.
        total = self.metrics.actions_total
        if goal_reached:
            self.metrics.verification_score = round(
                self.metrics.actions_successful / max(total, 1), 2
            )
        elif total > 0:
            self.metrics.verification_score = round(
                self.metrics.actions_successful / total, 2
            )
        else:
            self.metrics.verification_score = 0.0

        # Close the learn→apply loop: distill this mission's traces into lessons
        # and persist them so the NEXT mission's reasoning sees them. Grounded
        # in real trace outcomes (FAILED/RECOVERED→AVOID, SUCCESS→REUSE) — never
        # fabricated. Skipped silently when no ledger is wired in.
        if self.lessons_ledger is not None:
            from sonic.being.lessons import extract_lessons
            new_lessons = extract_lessons(self.traces, goal, self.agent_id)
            if new_lessons:
                self.lessons_ledger.record(new_lessons)

        # Deterministic summary of mission outcome from worker's own actions
        self.mission_summary = ""

        return self.traces

    # =============================================================
    # 6. Self-Directed Curiosity / Life Loop (Phase 6)
    # =============================================================
    async def _observation_summary(self, workspace_id: str) -> str:
        """A compact text summary of the current world for curiosity proposals."""
        obs = await self.observe(workspace_id)
        return (
            f"Active app: {obs.active_application}\n"
            f"Screen: {(obs.visible_text or '')[:300]}\n"
            f"Terminal: {(obs.terminal_output or '')[:200]}\n"
            f"Files: {obs.filesystem_files}\n"
            f"Git: branch={obs.git_branch}, clean={obs.git_clean}"
        )

    async def idle_cycle(
        self,
        workspace_id: str,
        curiosity: Any,
    ) -> Any:
        """One self-directed curiosity cycle, no operator goal required.

        Uses the CuriosityLoop to PROPOSE a goal from the live observation +
        learned facts, pursues it via the real observe->reason->act loop, then
        measures novelty and persists any newly-learned fact. Returns the
        CuriosityCycleResult.
        """
        async def pursue(goal: str, steps: int) -> str:
            before = len(self.traces)
            await self.run_mission(workspace_id, goal, steps=steps)
            # Summarize what the pursuit actually observed/did.
            new_traces = self.traces[before:]
            if not new_traces:
                return f"no actions taken toward: {goal}"
            outcomes = "; ".join(
                f"{t.action_type.value}:{(t.actual_observation or '')[:60]}" for t in new_traces
            )
            return f"{goal} -> {outcomes}"

        obs_summary = await self._observation_summary(workspace_id)
        return await curiosity.run_cycle(obs_summary, pursue)

    async def run_curiosity_loop(
        self,
        workspace_id: str,
        curiosity: Any,
    ) -> list[Any]:
        """Run repeated self-directed curiosity cycles until max_cycles."""
        results = []
        for _ in range(curiosity.max_cycles):
            res = await self.idle_cycle(workspace_id, curiosity)
            results.append(res)
        return results
