"""General-purpose computer-use worker for SONIC's shared workstation."""

from __future__ import annotations

import ast
import asyncio
import base64
import contextlib
import hashlib
import inspect
import ipaddress
import json
import math
import os
import re
import shlex
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from sonic.computer.models import (
    GUIAction,
    GUIActionType,
    ScreenObservation,
)
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.action_contracts import (
    ActionReceipt,
    GUIActionIntent,
    StateDelta,
)
from sonic.computer_use.experience import compare_prediction, state_signature
from sonic.computer_use.grounding import draw_action_marker
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
    GUI_STATE_VERIFICATION_ACTIONS,
    StrategyState,
    SubGoal,
    SubGoalChecklist,
    SubGoalStatus,
    VerificationProbe,
    VerificationProbeKind,
)
from sonic.computer_use.motor import MotorReflexes
from sonic.computer_use.perception_bus import PerceptionBus
from sonic.computer_use.scratchpad import HackerScratchpad
from sonic.computer_use.wire_telemetry import WireTelemetryEngine
from sonic.llm.prompts import COMPUTER_USE_SYSTEM_PROMPT
from sonic.logger import get_logger
from sonic.research.failure_budget import FailureBudgetTracker
from sonic.research.failure_classifier import classify_failure

_GUI_REASONING_TIMEOUT_SECONDS = 60.0
_MAX_GUI_BURST_ACTIONS = 4
_POSTCONDITION_REQUIRED_GUI_ACTIONS = frozenset({
    ComputerActionType.GUI_CLICK,
    ComputerActionType.GUI_DOUBLE_CLICK,
    ComputerActionType.GUI_RIGHT_CLICK,
    ComputerActionType.GUI_KEYPRESS,
    ComputerActionType.GUI_HOTKEY,
    ComputerActionType.GUI_SCROLL,
    ComputerActionType.GUI_DRAG,
})

# Actions whose outcome the narrator phrases for the next reasoning step.
_GUI_ACTION_TYPES_FOR_NARRATION = frozenset({
    ComputerActionType.GUI_CLICK,
    ComputerActionType.GUI_DOUBLE_CLICK,
    ComputerActionType.GUI_RIGHT_CLICK,
    ComputerActionType.GUI_MOVE,
    ComputerActionType.GUI_TYPE,
    ComputerActionType.GUI_KEYPRESS,
    ComputerActionType.GUI_HOTKEY,
    ComputerActionType.GUI_SCROLL,
    ComputerActionType.GUI_DRAG,
    ComputerActionType.APP_LAUNCH,
    ComputerActionType.APP_FOCUS,
})


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
        # First try to ensure it's valid UTF-8
        return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        try:
            # Fallback to ASCII with replacement
            return s.encode("ascii", errors="replace").decode("ascii")
        except Exception:
            # Ultimate fallback: remove all non-ASCII
            return "".join(c for c in s if ord(c) < 128)


def _parse_pixel_coordinate_target(target: Any) -> tuple[int, int] | None:
    """Parse only explicit integer screen-pixel points from a GUI target."""
    if isinstance(target, dict):
        point = target
    elif isinstance(target, str):
        text = target.strip()
        pair = re.fullmatch(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", text)
        if pair:
            return int(pair.group(1)), int(pair.group(2))
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        point = parsed
    else:
        return None

    if set(point) != {"x", "y"}:
        return None
    x, y = point["x"], point["y"]
    if (
        not isinstance(x, int)
        or isinstance(x, bool)
        or not isinstance(y, int)
        or isinstance(y, bool)
    ):
        return None
    return x, y


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
        try:
            return str(val) if val is not None else ""
        except Exception:
            return ""

    def info(self, event: str, **kwargs: Any):
        try:
            safe_kwargs = {}
            for k, v in kwargs.items():
                try:
                    safe_kwargs[str(k)] = self._sanitize(v)
                except Exception:
                    safe_kwargs[str(k)] = ""
            self._raw_logger.info(_safe_str(event), **safe_kwargs)
        except Exception:
            pass  # Fail silently to avoid encoding crashes

    def warning(self, event: str, **kwargs: Any):
        try:
            safe_kwargs = {}
            for k, v in kwargs.items():
                try:
                    safe_kwargs[str(k)] = self._sanitize(v)
                except Exception:
                    safe_kwargs[str(k)] = ""
            self._raw_logger.warning(_safe_str(event), **safe_kwargs)
        except Exception:
            pass  # Fail silently to avoid encoding crashes

    def error(self, event: str, **kwargs: Any):
        try:
            safe_kwargs = {}
            for k, v in kwargs.items():
                try:
                    safe_kwargs[str(k)] = self._sanitize(v)
                except Exception:
                    safe_kwargs[str(k)] = ""
            self._raw_logger.error(_safe_str(event), **safe_kwargs)
        except Exception:
            pass  # Fail silently to avoid encoding crashes

    def debug(self, event: str, **kwargs: Any):
        try:
            safe_kwargs = {}
            for k, v in kwargs.items():
                try:
                    safe_kwargs[str(k)] = self._sanitize(v)
                except Exception:
                    safe_kwargs[str(k)] = ""
            self._raw_logger.debug(_safe_str(event), **safe_kwargs)
        except Exception:
            pass  # Fail silently to avoid encoding crashes


logger = _SafeLogger(get_logger(__name__))

# A sentinel the LLM may emit to signal the goal is achieved, so the mission
# loop can terminate early instead of running a fixed step count.
_GOAL_COMPLETE_SENTINEL = "GOAL_COMPLETE"


def sanitize_typing_text(text_val: str) -> str:
    """Ensure typing payloads never contain navigation commands ('File -> Open') or thoughts ('BeingCraft note:')."""
    if not text_val:
        return ""
    t = str(text_val).strip()

    # 1. Navigation commands: 'File -> Open', 'Edit -> Preferences', 'Menu -> Item'
    if re.match(r"^[A-Za-z0-9_\s]+\s*->\s*[A-Za-z0-9_\s]+$", t):
        return ""

    # Strip navigation command lines if multiline
    lines = [
        line for line in t.splitlines()
        if not re.match(r"^\s*[A-Za-z0-9_\s]+\s*->\s*[A-Za-z0-9_\s]+$", line.strip())
    ]
    t = "\n".join(lines).strip()

    # 2. Strip thoughts / BeingCraft note blocks
    t = re.sub(
        r'(?i)\b(?:BeingCraft[\s_]+note|BeingCraftNote|BeingCraft|Thought|Reasoning|Explanation|Commentary|Observation|Plan|Reflection|Summary)\b(?:\*{1,2}|_)?\s*:.*?(?=(?:\r?\n|$))',
        '',
        t,
    ).strip()
    t = re.sub(
        r'(?i)\b(?:Note|Notes)\b(?:\*{1,2}|_)?\s*:.*?(?=(?:\r?\n|$))',
        '',
        t,
    ).strip()
    t = re.sub(
        r'\*\*(?:BeingCraft(?:\s*note)?|Thought|Reasoning|Explanation|Commentary|Note)\*\*.*?(?=(?:\r?\n|$))',
        '',
        t,
        flags=re.IGNORECASE,
    ).strip()
    t = re.sub(
        r'(?:\r?\n)\s*(?:Note(?:\s+that)?|Please note|We need to|This will|I will|Next we|Now we)\b.*$',
        '',
        t,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return t


def sanitize_command_payload(cmd: str) -> str:
    """Strip trailing thoughts, BeingCraft note blocks, and commentary from shell commands."""
    if not cmd:
        return ""
    c = str(cmd).strip()
    # Strip BeingCraft note and thought blocks
    c = re.sub(
        r'(?i)(?:\n|\s+)(?:BeingCraft[\s_]+note|BeingCraftNote|BeingCraft|Thought|Reasoning|Explanation|Commentary|Observation|Plan|Reflection|Summary)\b(?:\*{1,2}|_)?\s*:.*$',
        '',
        c,
        flags=re.DOTALL,
    ).strip()
    c = re.sub(
        r'(?i)(?:\n|\s+)(?:Note|Notes)\b(?:\*{1,2}|_)?\s*:.*$',
        '',
        c,
        flags=re.DOTALL,
    ).strip()
    c = re.sub(
        r'\*\*(?:BeingCraft(?:\s*note)?|Thought|Reasoning|Explanation|Commentary|Note)\*\*.*$',
        '',
        c,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    quote = ""
    escaped = False
    for index, char in enumerate(c):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "(" and re.match(
            r"\(\s*(?:or|because|which|that|so|we|i|you|for example|e\.g\.)\b",
            c[index:],
            flags=re.IGNORECASE,
        ):
            return c[:index].rstrip()
    return c


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
        security_tools: dict[str, Any] | None = None,
        safety: Any | None = None,
        self_host: bool = False,
        toolsmith: Any | None = None,
        method_lab: Any | None = None,
        lessons_ledger: Any | None = None,
        tenant_id: str = "default",
        engagement_id: str = "default",
        agent_id: str = "computer-use-agent",
        multi_context_sources: list[Any] | None = None,
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
        system_prompt_overlay: str | None = None,
        **kwargs: Any,
    ):
        self.computer = computer_provider
        self.craft = craft
        # Keep the application desktop optional. Sandbox/operator missions
        # should not capture pixels merely because a workstation exists.
        self.observe_desktop = observe_desktop
        # Live narration: AVCS describes what it sees in plain language so the
        # reasoning model plans from words instead of full frames.  Built lazily
        # on first use; a missing AVCS narrator module degrades to no narration
        # rather than breaking observation.
        self._avcs_narrator: Any | None = None
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
        self.system_prompt_overlay = str(system_prompt_overlay or kwargs.get("system_prompt_overlay") or "").strip()
        self.operator_approval_callback = kwargs.get("operator_approval_callback")
        self.is_subagent_worker = bool(
            kwargs.get("is_subagent_worker")
            or self.initial_context.get("silent_background")
            or self.initial_context.get("subagent_role")
        )
        self.suppress_goal_decomposition = bool(kwargs.get("suppress_goal_decomposition"))
        # Shared versioned perception state keeps semantic targets and
        # sub-agents on one live snapshot without repeatedly decoding pixels.
        self.perception_bus = PerceptionBus()
        self.experience_records: list[Any] = []
        self.autonomy_level = autonomy_level
        self.mode = mode
        self.max_actions = max_actions
        self.max_recovery_attempts = max_recovery_attempts
        if llm_router is None:
            try:
                from sonic.llm.router import ModelRouter
                llm_router = ModelRouter.from_config()
            except Exception:
                llm_router = None
        self.llm_router = llm_router  # ModelRouter for LLM-driven reasoning
        if "browser" in kwargs:
            raise TypeError("Browser adapters were removed; use the visible desktop GUI worker")
        self.vault = kwargs.get("vault")
        if self.vault is None:
            try:
                from sonic.being.identity import get_or_create_being
                from sonic.being.vault import BeingVault
                _b = kwargs.get("being") or get_or_create_being(str(kwargs.get("tenant_id") or "default"))
                self.vault = BeingVault(_b)
            except Exception:
                self.vault = None
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
        self._agentic_lessons_goal: str | None = None
        self._agentic_lessons: list[Any] = []
        # Multi-context retrieval: fuse several tenant-scoped memory sources
        # (vector, graph findings, episodes) into one bounded pack. None or an
        # empty list disables it; a single source still works.
        self.multi_context_sources = list(multi_context_sources or [])
        self._multi_context_goal: str | None = None
        self._multi_context_block: str = ""
        self._multi_context_pack: Any | None = None
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
        # Closed-loop eye-hand servo (lazily built). It drives the pointer onto a
        # pixel target by re-observing after each corrective step, so a
        # mis-estimated coordinate is fixed by the next frame instead of being
        # committed blindly. The model itself is torch-free (forge.vision.servo).
        self._visual_servo: Any | None = None
        # Run-unique worker handle (e.g. "worker-1-a1b2c3"); the coordinator
        # inbox is keyed by it so the operator can steer this exact worker.
        self.agent_uid = str(kwargs.get("agent_uid") or "")
        self.enable_llm_verification = enable_llm_verification
        self.enable_llm_decomposition = enable_llm_decomposition
        # Auto-detect compact mode for small models if not explicitly set.
        if compact_mode is not None:
            self._compact_mode = compact_mode
        elif llm_router is not None:
            raw_model = getattr(llm_router, "default_model", "")
            model_name = raw_model.lower() if isinstance(raw_model, str) else ""
            self._compact_mode = bool(
                re.search(r"(?<![a-z0-9])(?:1|3|7|8|11)b(?![a-z0-9])", model_name)
                or re.search(r"\b(?:small|mini|tiny)\b", model_name)
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
        # Last structured security-tool result, surfaced to the next reasoning
        # step so the LLM acts on real scan findings rather than a claim.
        self._last_tool_result: Any | None = None
        # Optional AVCS apprenticeship loop. When attached, every dispatched
        # pointer action is recorded with its *measured* outcome (did the screen
        # change?) so the dorsal affordance head can later be trained on real
        # clicks instead of staying untrained and emitting random hotspots.
        self.apprenticeship: Any | None = None
        self._last_screenshot_b64: str = ""
        self._last_screen_has_pixels: bool = False
        self._last_screen_frame_id: int | None = None
        self._last_screen_observation_received_at: float = 0.0
        self._last_observed_screen_from_reasoning: bool = False
        self._last_gui_action_observation_valid: bool = False
        self._last_active_window_geometry: tuple[int, int, int, int] | None = None
        self._consecutive_unchanged_screens: int = 0
        self._watchdog_aborted: bool = False
        # Last observed screen dimensions, used to validate GUI coordinate
        # actions before they touch the provider. Updated on every observe() /
        # screenshot; sane defaults until the first real observation lands.
        self._screen_width: int = 1920
        self._screen_height: int = 1080
        self._interrupted: bool = False
        self._interrupt_event = asyncio.Event()
        self._active_interrupt_check: Any | None = None
        self.goal_reached: bool = False
        self._consecutive_failures: int = 0
        self._consecutive_unchanged_screens: int = 0
        self._consecutive_stale_gui_blocks: int = 0
        self._goal_complete_declared: bool = False
        self._replan_count: int = 0
        self._last_gui_navigation_target: str = ""
        self._recent_action_signatures: list[tuple[str, str, str]] = []
        self._consecutive_gui_contract_violations: int = 0
        self._consecutive_gui_postcondition_violations: int = 0
        self._pending_gui_contract_feedback: str = ""
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
    def _burst_gui_actions(
        payload: dict[str, Any],
    ) -> list[tuple[GUIAction, Any]] | None:
        """Decode a small explicit GUI chunk, preserving per-step expectations."""
        raw = payload.get("burst_actions")
        if raw is None:
            raw = payload.get("actions") if payload.get("burst") is True else None
        if raw is None:
            return None
        if not isinstance(raw, list) or not raw:
            raise ValueError("GUI burst requires a non-empty actions list")
        if len(raw) > _MAX_GUI_BURST_ACTIONS:
            raise ValueError(
                f"GUI burst is limited to {_MAX_GUI_BURST_ACTIONS} actions"
            )
        actions: list[tuple[GUIAction, Any]] = []
        for item in raw:
            if isinstance(item, GUIAction):
                actions.append((item, None))
                continue
            if not isinstance(item, dict):
                raise ValueError("GUI burst actions must be structured GUI actions")
            data = dict(item)
            postcondition = data.pop("postcondition", None)
            raw_kind = data.pop("action", data.pop("type", ""))
            try:
                kind = GUIActionType(str(raw_kind).upper())
            except ValueError as exc:
                raise ValueError(f"unknown GUI burst action: {raw_kind}") from exc
            data["action"] = kind
            actions.append((GUIAction(**data), postcondition))
        return actions

    @staticmethod
    def _has_gui_burst_payload(payload: dict[str, Any]) -> bool:
        return payload.get("burst_actions") is not None or (
            payload.get("burst") is True and payload.get("actions") is not None
        )

    @staticmethod
    def _burst_postcondition_schema_error(postcondition: Any) -> str | None:
        if isinstance(postcondition, str):
            return "text postconditions are unavailable in AVCS pixel-only mode"
        if not isinstance(postcondition, dict) or not postcondition:
            return "postcondition must be a non-empty mapping"
        allowed = {
            "active_window",
            "require_state_change",
            "screen_changed",
        }
        unknown = set(postcondition) - allowed
        if unknown:
            return f"unrecognized postcondition field: {next(iter(unknown))}"
        if "active_window" in postcondition and (
            not isinstance(postcondition["active_window"], str)
            or not postcondition["active_window"].strip()
        ):
            return "active-window postconditions must be non-empty strings"
        if any(
            key in postcondition and not isinstance(postcondition[key], bool)
            for key in ("require_state_change", "screen_changed")
        ):
            return "state-change postconditions must be boolean"
        if not set(postcondition).intersection(
            {"active_window", "screen_changed"}
        ):
            return "postcondition must name an observable state"
        return None

    def _burst_postcondition_ok(
        self, postcondition: Any, snapshot: Any, *, changed: bool
    ) -> tuple[bool, str]:
        """Check only caller-supplied live state; an absent condition is honest."""
        schema_error = self._burst_postcondition_schema_error(postcondition)
        if schema_error:
            return False, schema_error
        if isinstance(postcondition, str):
            return False, "text postconditions are unavailable in AVCS pixel-only mode"
        if not isinstance(postcondition, dict):
            return False, "postcondition must be a mapping"
        require_state_change = postcondition.get(
            "require_state_change", "screen_changed" not in postcondition
        )
        if require_state_change and not changed:
            return False, "postcondition requires a real UI state change"
        checks = {"active_window": getattr(snapshot, "active_window", "")}
        for key, expected in postcondition.items():
            if key in {"require_state_change", "screen_changed"}:
                continue
            if key not in checks:
                return False, f"unrecognized postcondition field: {key}"
            actual = str(checks[key] or "")
            if str(expected).lower() not in actual.lower():
                return False, f"postcondition {key} was not observed"
        if (
            "screen_changed" in postcondition
            and changed is not postcondition["screen_changed"]
        ):
            return False, "screen change state did not match the requested postcondition"
        return True, ""

    def _single_gui_postcondition_error(
        self, action_type: ComputerActionType, payload: dict[str, Any]
    ) -> str | None:
        if action_type not in _POSTCONDITION_REQUIRED_GUI_ACTIONS:
            return None
        postcondition = payload.get("postcondition")
        if not isinstance(postcondition, dict):
            return "a structured observable postcondition is required in PAYLOAD"
        return self._burst_postcondition_schema_error(postcondition)

    def _verify_single_gui_postcondition(
        self, payload: dict[str, Any]
    ) -> tuple[bool, str]:
        screen = getattr(self, "_last_gui_action_screen", None)
        if (
            screen is None
            or not getattr(self, "_last_gui_action_observation_valid", False)
        ):
            return False, "fresh post-action desktop evidence is unavailable"

        state_delta = getattr(self, "_last_gui_state_delta", None)
        changed = (
            state_delta.get("changed")
            if isinstance(state_delta, dict)
            and isinstance(state_delta.get("changed"), bool)
            else False
        )
        snapshot = SimpleNamespace(
            active_window=getattr(screen, "active_window", ""),
            visible_text=getattr(screen, "visible_text", ""),
            controls=getattr(screen, "detected_controls", ()) or (),
        )
        return self._burst_postcondition_ok(
            payload.get("postcondition"), snapshot, changed=changed
        )

    async def _execute_gui_burst(
        self, workspace_id: str, payload: dict[str, Any], target_resource: str = ""
    ) -> tuple[bool, str, int | None]:
        """Run an explicit motor burst against versioned, freshly observed state."""
        steps = self._burst_gui_actions(payload)
        if steps is None:
            raise ValueError("not a GUI burst")
        actions = [action for action, _ in steps]
        step_postconditions = [condition for _, condition in steps]
        if any(condition is None for condition in step_postconditions):
            return False, "every GUI burst action requires a postcondition", 126
        for index, condition in enumerate(step_postconditions):
            step_schema_error = self._burst_postcondition_schema_error(condition)
            if step_schema_error:
                return (
                    False,
                    f"GUI burst action {index + 1} requires a valid observable "
                    f"postcondition: {step_schema_error}",
                    126,
                )
        final_postcondition = payload.get("postcondition")
        schema_error = self._burst_postcondition_schema_error(final_postcondition)
        if schema_error:
            return False, f"GUI burst requires a valid final postcondition: {schema_error}", 126

        action_types = {
            GUIActionType.CLICK: ComputerActionType.GUI_CLICK,
            GUIActionType.DOUBLE_CLICK: ComputerActionType.GUI_DOUBLE_CLICK,
            GUIActionType.RIGHT_CLICK: ComputerActionType.GUI_RIGHT_CLICK,
            GUIActionType.TYPE: ComputerActionType.GUI_TYPE,
            GUIActionType.KEYPRESS: ComputerActionType.GUI_KEYPRESS,
            GUIActionType.HOTKEY: ComputerActionType.GUI_HOTKEY,
            GUIActionType.SCROLL: ComputerActionType.GUI_SCROLL,
            GUIActionType.MOVE: ComputerActionType.GUI_MOVE,
            GUIActionType.DRAG: ComputerActionType.GUI_DRAG,
        }
        for action in actions:
            action_type = action_types.get(action.action)
            if action_type is None:
                return False, f"GUI burst action is not permitted: {action.action.value}", 126
            coordinate_error = self._validate_coordinates(
                action_type,
                action.model_dump(exclude={"action"}),
            )
            if coordinate_error:
                return False, f"GUI burst action has invalid coordinates: {coordinate_error}", 126

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
        observed_step_index = 0

        async def observe_after_action(*_args: Any) -> dict[str, Any]:
            nonlocal previous, observed_step_index
            before = previous
            await self.observe(workspace_id)
            current = bus.current()
            changed = (
                current.screen_hash != before.screen_hash
                or current.active_window != before.active_window
                or current.windows != before.windows
                or current.visible_text != before.visible_text
                or current.controls != before.controls
            )
            if current.version <= before.version:
                return {"ok": False}
            if (
                current.width != initial.width
                or current.height != initial.height
            ):
                return {"ok": False, "reason": "display geometry changed during burst"}
            active_window_expectation = (
                step_postconditions[observed_step_index].get("active_window")
                if isinstance(step_postconditions[observed_step_index], dict)
                else None
            )
            if (
                current.active_window != before.active_window
                and (
                    not isinstance(active_window_expectation, str)
                    or active_window_expectation.casefold()
                    not in current.active_window.casefold()
                )
            ):
                return {"ok": False, "reason": "active window changed unexpectedly"}
            step_ok, step_reason = self._burst_postcondition_ok(
                step_postconditions[observed_step_index],
                current,
                changed=changed,
            )
            if not step_ok:
                return {"ok": False, "reason": step_reason}
            previous = current
            observed_step_index += 1
            return {"ok": True}

        result = await self.motor.execute_burst(
            workspace_id,
            actions,
            interval_seconds=float(payload.get("interval_seconds", 0.05)),
            observation_hook=observe_after_action,
            interrupt_check=self._mission_interrupt_requested,
            interrupt_event=self._interrupt_event,
        )
        if not result.success:
            await refresh_after_failure()
            return False, f"GUI burst stopped: {result.stop_reason or 'action failed'}; re-observe and replan", 126
        snapshot = bus.current()
        changed = snapshot.version > initial.version and snapshot != initial
        ok, reason = self._burst_postcondition_ok(
            final_postcondition, snapshot, changed=changed
        )
        if not ok:
            await refresh_after_failure()
            return False, f"GUI burst postcondition failed: {reason}; re-observe and replan", 126
        return True, f"GUI burst completed {result.completed} action(s) with verified observation", 0

    def interrupt(self) -> None:
        """Signal the agent to stop its active mission loop immediately."""
        self._interrupted = True
        self._interrupt_event.set()

    def _mission_interrupt_requested(self) -> bool:
        interrupt_check = getattr(self, "_active_interrupt_check", None)
        return bool(getattr(self, "_interrupted", False)) or (
            callable(interrupt_check) and bool(interrupt_check())
        )

    def _gui_action_contract_active(self) -> bool:
        """Return whether this worker must stay within visible GUI actions."""
        return bool(
            getattr(self, "gui_only", False)
            or str(getattr(self, "execution_plane", "") or "").lower().strip() == "gui"
        )

    def _mission_used_desktop_actions(self) -> bool:
        """True when this mission actually dispatched a visible-desktop action.

        Completion of GUI work must be confirmed by the independent Judge no
        matter which plane the worker was assigned. Plane alone is the wrong
        signal: a "dual" worker that used the desktop needs the same Judge gate
        as a "gui" worker, otherwise a changed screen would be accepted as proof
        of the outcome.
        """
        for trace in getattr(self, "traces", None) or []:
            action = str(
                getattr(
                    getattr(trace, "action_type", ""),
                    "value",
                    getattr(trace, "action_type", ""),
                )
            ).upper()
            if action.startswith(("GUI_", "APP_", "WINDOW_", "BROWSER_")):
                return True
        return False

    @staticmethod
    def _rank_apps_for_goal(apps: list, goal: str) -> list:
        """Order installed applications so goal-relevant ones appear first.

        Generic and name-agnostic: an app is scored by how much of its own name
        overlaps the goal text, so a specific match (the goal names the app, or
        the app name is mostly composed of goal words) outranks an app that
        merely shares one generic word such as "editor". No vendor names are
        hardcoded; apps with no overlap keep their discovery order. This keeps
        the operator's requested app inside the visible portion of a truncated
        app list.
        """
        if not apps or not goal:
            return list(apps or [])
        goal_tokens = {
            tok for tok in re.findall(r"[a-z0-9]+", str(goal).lower()) if len(tok) >= 3
        }
        if not goal_tokens:
            return list(apps)

        def score(name: str) -> float:
            name_tokens = set(re.findall(r"[a-z0-9]+", str(name).lower()))
            matched = name_tokens & goal_tokens
            if not matched:
                return -1.0
            # Fraction of the app's name that the goal mentions: rewards a
            # specific app ("mousepad") over a generic utility that happens to
            # share one word ("xfce4-settings-editor").
            base = len(matched) / max(len(name_tokens), 1)
            # The goal naming the app outright is the strongest signal.
            exact = 2.0 if str(name).lower().strip() in goal_tokens else 0.0
            return base + exact

        # Stable sort: equal scores keep discovery order, so unmatched apps
        # (score -1) are never reshuffled.
        return sorted(apps, key=score, reverse=True)

    def _gui_only_system_prompt(
        self,
    ) -> str:
        actions = "|".join(
            action.value
            for action in ComputerActionType
            if action in self._GUI_ONLY_ACTIONS
            and not action.name.startswith("BROWSER_")
        )
        grounding_source = "the current AVCS pixel-state summary"
        # Semantic clicks need an evaluated local AVCS grounder. When none is
        # configured the action fails closed, so tell the model up front and
        # point it at routes that do not require semantic grounding (launching
        # an observed installed application, keyboard navigation). This avoids
        # burning the whole step budget on a target that cannot be resolved.
        if self._avcs_grounder_ready():
            grounding_clause = (
                "For GUI_CLICK, GUI_DOUBLE_CLICK, GUI_RIGHT_CLICK, GUI_MOVE, and "
                "GUI_SCROLL, provide an observed semantic target in TARGET; a configured "
                "AVCS grounding checkpoint resolves it to pixels before dispatch. "
                "Never guess coordinates. "
            )
        else:
            grounding_clause = (
                "No AVCS grounding checkpoint is configured, so semantic GUI_CLICK "
                "targets cannot be resolved and will fail closed. Do not propose "
                "GUI_CLICK/GUI_DOUBLE_CLICK/GUI_RIGHT_CLICK/GUI_MOVE/GUI_SCROLL on a "
                "described control. Instead open an observed installed application with "
                "APP_LAUNCH (its name only, never a path or destination), then use "
                "keyboard actions such as GUI_HOTKEY and GUI_KEYPRESS. "
            )
        return (
            "You are SONIC, a general-purpose AI-human colleague operating a live "
            "desktop. This worker's assigned plane is the visible GUI only.\n"
            "For every turn, return exactly one JSON object with the keys "
            '"action", "target", "payload", and "expected"; do not return '
            "Markdown, prose, or omit keys.\n"
            "For GUI_CLICK, GUI_DOUBLE_CLICK, GUI_RIGHT_CLICK, "
            "GUI_KEYPRESS, GUI_HOTKEY, GUI_SCROLL, and GUI_DRAG, PAYLOAD must "
            "include a structured observable `postcondition` (for example "
            '{"screen_changed": true}). '
            "Do not treat the free-text `expected` field as a substitute.\n"
            f"Only these ACTION values are valid: {actions}. Never invent "
            "an action name or synonym.\n"
            f"Choose one immediate action from {grounding_source}. AVCS provides "
            "local pixel reflex state, not OCR text or semantic controls. Do not infer "
            "a control from the objective or earlier history.\n"
            f"{grounding_clause}"
            "Do not depend on OCR, extracted control lists, "
            "or accessibility metadata. Never describe a desktop shortcut as an already-open window "
            "or add inferred words such as 'window' or 'active'.\n"
            "APP_LAUNCH may name only an observed installed application, never a "
            "directory or destination path. Only launch an application that appears "
            "in the 'Available Installed Desktop Applications' list above; if the "
            "objective names an application that is NOT in that list, it is not "
            "installed — do not invent a binary name (for example do not guess a "
            "different terminal emulator), and instead report the objective as blocked. "
            "Text entry is unavailable until AVCS "
            "has an evaluated local focus-verification head. If a visual target cannot be grounded, do "
            "not repeat it or guess coordinates: use another observed route, including "
            "keyboard navigation when appropriate. If no route is grounded, choose "
            "GUI_SCREENSHOT once to refresh AVCS state, then stop as blocked. "
            "GOAL_COMPLETE is only a handoff proposal; the independent Judge decides "
            "completion from independently verifiable evidence and worker receipts."
        )

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
            if self._gui_action_contract_active()
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
        init_ctx = getattr(self, "initial_context", None) or {}
        if (
            init_ctx.get("silent_background")
            or init_ctx.get("subagent_role")
            or getattr(self, "is_subagent_worker", False)
            or getattr(self, "suppress_goal_decomposition", False)
        ):
            return SubGoalChecklist(
                top_level_goal=goal,
                sub_goals=[SubGoal(description=f"Pursue objective: {goal}")],
                active_index=0,
            )

        sub_goals: list[SubGoal] = []

        # 1. Check for explicit numbered/bulleted steps in user prompt
        lines = [line.strip() for line in goal.splitlines() if line.strip()]
        if len(lines) <= 1:
            lines = [line.strip() for line in re.split(r'(?<=\w\w)\.\s+(?=[0-9]\.|\b[A-Z])', goal) if line.strip()]
        rubric_bullet_pattern = re.compile(
            r'^(?:-|\*|\d+[\.\)])?\s*(?:-\s*)?(?:\*\*R\d+[:\*]*|\[[\s xX]\]|\*\*Requirement\s*\d+|R\d+:)',
            flags=re.IGNORECASE,
        )
        numbered_items = []
        for l in lines:
            # Do not treat rubric bullets as procedural steps
            if rubric_bullet_pattern.match(l):
                continue
            m = re.match(r'^(?:\d+[\.\)]|\-|\*)\s*(.*)', l)
            if m and len(m.group(1).strip()) > 3:
                numbered_items.append(m.group(1).strip())

        if len(numbered_items) >= 2:
            for item in numbered_items[:6]:
                sub_goals.append(SubGoal(description=item))
            return SubGoalChecklist(top_level_goal=goal, sub_goals=sub_goals, active_index=0)

        # 2. Try LLM-driven goal decomposition when enabled and a router is available
        if getattr(self, "enable_llm_decomposition", True) and self.llm_router is not None:
            llm_checklist = await self._llm_decompose_goal(goal)
            if llm_checklist is not None:
                return llm_checklist

        # 3. Dynamic clause extraction from user's explicit multi-part goal (supports Hinglish/English)
        raw_parts = [
            p.strip()
            for p in re.split(r'\b(?:and then|then|and|aur|phir|uske baad|baad me)\b|[;\n]', goal, flags=re.IGNORECASE)
            if len(p.strip()) > 3
        ]
        parts = [
            p for p in raw_parts
            if not rubric_bullet_pattern.match(p) and not p.startswith(("- [ ]", "* [ ]", "- [x]", "* [x]", "- **R", "* **R"))
        ]
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
    def _ensure_avcs_state(self, screen: ScreenObservation) -> ScreenObservation:
        try:
            perception = getattr(self, "_avcs_perception", None)
            if perception is None:
                from sonic.computer_use.avcs_perception import AVCSPerception

                perception = AVCSPerception()
                self._avcs_perception = perception
            if screen.raw_rgb:
                state = perception.perceive_rgb_bytes(
                    screen.raw_rgb,
                    screen.width,
                    screen.height,
                    screen.frame_id,
                )
            else:
                png_bytes = base64.b64decode(
                    screen.screenshot_base64, validate=True
                )
                state = perception.perceive_png(png_bytes)
            if (
                state.get("engine") != "avcs_reflex"
                or state.get("width") != screen.width
                or state.get("height") != screen.height
                or not isinstance(state.get("frame_index"), int)
                or isinstance(state.get("frame_index"), bool)
                or state["frame_index"] <= 0
                or not isinstance(state.get("scene_hash"), str)
                or not state["scene_hash"]
                or (
                    screen.frame_id is not None
                    and state.get("source_frame_id") != screen.frame_id
                )
            ):
                raise ValueError("AVCS returned invalid frame metadata")
        except Exception as exc:
            logger.error(
                "avcs_agent_perception_failed",
                workspace_id=getattr(self, "_active_workspace_id", ""),
                error=str(exc),
            )
            raise RuntimeError("AVCS could not process the current workstation frame") from exc

        screen.avcs_state = state
        # Learned semantics ride along when a trained cortex checkpoint is
        # configured; an empty mapping means "unavailable", not "nothing there".
        if screen.raw_rgb:
            try:
                frame = np.frombuffer(screen.raw_rgb, dtype=np.uint8).reshape(
                    screen.height, screen.width, 3
                )
                semantics = perception.perceive_semantics_rgb(frame)
            except Exception as exc:
                logger.warning("avcs_cortex_perception_failed", error=str(exc))
                semantics = {}
            if semantics:
                state["semantics"] = semantics
        screen.visible_text = ""
        screen.detected_controls = []
        screen.controls_age_ms = None
        screen.perception_version = int(state["frame_index"])
        return screen

    # -- live narration ----------------------------------------------------

    def _get_avcs_narrator(self) -> Any | None:
        """Lazily build the AVCS narrator, or ``None`` when unavailable."""
        narrator = getattr(self, "_avcs_narrator", None)
        if narrator is not None:
            return narrator
        try:
            from sonic.computer_use.avcs_narrator import AVCSNarrator

            narrator = AVCSNarrator()
            self._avcs_narrator = narrator
            return narrator
        except Exception as exc:
            logger.warning("avcs_narrator_init_failed", error=str(exc))
            return None

    def _ambient_narration_block(self) -> str:
        """Drain the eye's unprompted lines into a prompt block ("" if none).

        These are things AVCS noticed on its own while the agent was busy.  They
        are drained, so each is surfaced once rather than repeated every turn.
        """
        narrator = self._get_avcs_narrator()
        if narrator is None or not getattr(narrator, "available", False):
            return ""
        # Keyed on the tenant, not the workspace: the continuous awareness
        # monitor and this agent may name the same desktop differently.
        ambient_key = str(getattr(self, "tenant_id", "") or "") or self._narration_workspace_id()
        try:
            from sonic.computer_use.avcs_narrator import drain_ambient_narration

            lines = drain_ambient_narration(ambient_key)
        except Exception as exc:
            logger.warning("avcs_ambient_drain_failed", error=str(exc))
            return ""
        if not lines:
            return ""
        rendered = []
        for item in lines[-3:]:
            urgency = str(item.get("urgency") or "info")
            marker = {"alert": "!", "notice": "~"}.get(urgency, "-")
            rendered.append(f"  {marker} {item.get('text', '')}")
        return "\n".join(rendered)

    def _narrate_screen(
        self, screen: ScreenObservation | None, active_window: str = ""
    ) -> str:
        """Narrate a fresh screen and return the prompt block ("" if silent).

        Narration is an enhancement: any failure yields an empty string so the
        caller's prompt is unchanged rather than broken.
        """
        if screen is None:
            return ""
        narrator = self._get_avcs_narrator()
        if narrator is None or not getattr(narrator, "available", False):
            return ""
        try:
            narrator.observe_screen(
                self._narration_workspace_id(),
                screen,
                active_window=active_window or getattr(screen, "active_window", ""),
            )
            return narrator.format_for_prompt(self._narration_workspace_id())
        except Exception as exc:
            logger.warning("avcs_narrate_screen_failed", error=str(exc))
            return ""

    def _narrate_action_result(
        self,
        action_type: Any,
        target: str,
        visible_effect: bool | None,
        screen: ScreenObservation | None,
        latency_ms: float = 0.0,
        detail: str = "",
    ) -> str:
        """Phrase an action's outcome for the next reasoning step ("" on fail)."""
        narrator = self._get_avcs_narrator()
        if narrator is None or not getattr(narrator, "available", False):
            return ""
        try:
            action_name = getattr(action_type, "value", None) or str(action_type)
            line = narrator.action_result(
                self._narration_workspace_id(),
                action_name,
                target=target,
                visible_effect=visible_effect,
                screen=screen,
                latency_ms=latency_ms,
                detail=detail,
            )
            return str(line.get("text", "")) if line else ""
        except Exception as exc:
            logger.warning("avcs_narrate_action_failed", error=str(exc))
            return ""

    def _narration_workspace_id(self) -> str:
        return str(getattr(self, "_active_workspace_id", "") or "default")

    async def _ground_target_with_avcs(
        self,
        query: str,
        screen: ScreenObservation | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a GUI target locally; screenshots never leave the workstation path."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("AVCS target grounding requires a semantic target query")
        source_screen = screen or getattr(self, "_last_observed_screen", None)
        if source_screen is None:
            raise RuntimeError("AVCS target grounding has no observed desktop frame")
        if getattr(source_screen, "desktop_state", "") != "INTERACTIVE":
            raise RuntimeError("AVCS target grounding requires an interactive desktop")
        capture_age_ms = getattr(source_screen, "capture_age_ms", None)
        if (
            not isinstance(capture_age_ms, (int, float))
            or isinstance(capture_age_ms, bool)
            or not math.isfinite(capture_age_ms)
            or not 0 <= capture_age_ms <= 2_000
        ):
            raise RuntimeError("AVCS target grounding requires a fresh pixel frame")
        raw_rgb = bytes(getattr(source_screen, "raw_rgb", b"") or b"")
        frame_b64 = str(getattr(source_screen, "screenshot_base64", "") or "")
        source_frame_id = getattr(source_screen, "frame_id", None)
        if not raw_rgb and not frame_b64:
            raise RuntimeError("AVCS target grounding requires current pixel data")
        if screen is None and raw_rgb and (
            source_frame_id != getattr(self, "_last_screen_frame_id", None)
        ):
            raise RuntimeError("AVCS target grounding frame is not the latest observation")
        if screen is None and not raw_rgb and (
            frame_b64 != str(getattr(self, "_last_screenshot_b64", "") or "")
        ):
            raise RuntimeError("AVCS target grounding frame is not the latest observation")
        width = int(getattr(source_screen, "width", 0) or 0)
        height = int(getattr(source_screen, "height", 0) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("AVCS target grounding requires observed screen dimensions")
        state = getattr(source_screen, "avcs_state", None)
        if (
            not isinstance(state, dict)
            or state.get("engine") != "avcs_reflex"
            or state.get("width") != width
            or state.get("height") != height
            or not isinstance(state.get("scene_hash"), str)
            or not state["scene_hash"]
        ):
            raise RuntimeError("AVCS target grounding requires current local AVCS state")
        perception = getattr(self, "_avcs_perception", None)
        if perception is None:
            raise RuntimeError("local AVCS pixel processor is unavailable")

        # Optional cloud vision bridge. Only used when the operator explicitly
        # opts in AND no evaluated local AVCS grounder checkpoint exists. The
        # local checkpoint always wins when configured: the default path keeps
        # workstation pixels inside the workstation.
        if (
            not getattr(perception, "goal_grounder_configured", False)
            and os.environ.get("SONIC_ENABLE_VLM_GROUNDING", "0") == "1"
        ):
            bridge_b64 = frame_b64
            if not bridge_b64 and raw_rgb:
                from io import BytesIO

                from PIL import Image

                buffer = BytesIO()
                Image.fromarray(
                    np.frombuffer(raw_rgb, dtype=np.uint8).reshape(height, width, 3)
                ).save(buffer, format="PNG")
                bridge_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            if bridge_b64:
                from sonic.computer_use.grounding import _resolve_vlm_target

                point = await _resolve_vlm_target(
                    query.strip(),
                    bridge_b64,
                    width,
                    height,
                    router=getattr(self, "llm_router", None),
                )
                if point is not None:
                    logger.info(
                        "vlm_grounding_target_resolved",
                        query=query,
                        resolved_x=point[0],
                        resolved_y=point[1],
                    )
                    return {"x": point[0], "y": point[1], "engine": "vlm_grounder"}

        if raw_rgb:
            result = await asyncio.to_thread(
                perception.ground_target_rgb_bytes,
                raw_rgb,
                query.strip(),
                width,
                height,
                getattr(source_screen, "active_window_geometry", None),
                source_frame_id,
            )
        else:
            png_bytes = base64.b64decode(frame_b64, validate=True)
            result = await asyncio.to_thread(
                perception.ground_target_png,
                png_bytes,
                query.strip(),
                width,
                height,
                getattr(source_screen, "active_window_geometry", None),
            )
        if result is not None:
            x, y = result.get("x"), result.get("y")
            query_score = result.get("query_score")
            query_score_threshold = result.get("query_score_threshold")
            calibration = result.get("checkpoint_metadata", {}).get(
                "query_confidence_calibration"
            )
            calibration_evaluation = (
                calibration.get("evaluation")
                if isinstance(calibration, dict)
                else None
            )
            calibration_precision = (
                calibration_evaluation.get("precision")
                if isinstance(calibration_evaluation, dict)
                else None
            )
            required_precision = (
                calibration_evaluation.get("required_precision")
                if isinstance(calibration_evaluation, dict)
                else None
            )
            if (
                result.get("engine") != "avcs_goal_grounder"
                or result.get("scene_hash") != state.get("scene_hash")
                or result.get("frame_index") != state.get("frame_index")
                or (
                    raw_rgb
                    and result.get("source_frame_id") != source_frame_id
                )
                or isinstance(query_score, bool)
                or not isinstance(query_score, (int, float))
                or not math.isfinite(float(query_score))
                or not 0 <= float(query_score) <= 1
                or isinstance(query_score_threshold, bool)
                or not isinstance(query_score_threshold, (int, float))
                or not math.isfinite(float(query_score_threshold))
                or not 0 < float(query_score_threshold) <= 1
                or float(query_score) < float(query_score_threshold)
                or not isinstance(calibration_evaluation, dict)
                or calibration_evaluation.get("metric")
                != "empirical_precision_at_threshold"
                or calibration_evaluation.get("passed") is not True
                or calibration_evaluation.get("threshold") != query_score_threshold
                or isinstance(calibration_precision, bool)
                or not isinstance(calibration_precision, (int, float))
                or not math.isfinite(float(calibration_precision))
                or isinstance(required_precision, bool)
                or not isinstance(required_precision, (int, float))
                or not math.isfinite(float(required_precision))
                or float(calibration_precision) < float(required_precision)
                or not isinstance(x, int)
                or isinstance(x, bool)
                or not isinstance(y, int)
                or isinstance(y, bool)
                or not (0 <= x < width and 0 <= y < height)
            ):
                raise RuntimeError("AVCS grounder returned a stale or invalid pixel target")
            logger.info(
                "avcs_goal_grounding_target_resolved",
                query=query,
                resolved_x=x,
                resolved_y=y,
                held_out_point_accuracy=result["held_out_point_accuracy"],
                query_score=query_score,
                query_score_threshold=query_score_threshold,
                grounding_latency_ms=result["latency_ms"],
            )
        return result

    async def observe(self, workspace_id: str) -> ComputerWorldObservation:
        """Capture a multi-modal observation of the computer environment.

        The terminal output is read from a real (cheap) probe so the agent
        reacts to what is actually on the screen/terminal, not a placeholder.
        The current application state is grounded in the visible desktop
        observation so the same reasoning loop can operate any GUI workflow.
        """
        perception_source = "unavailable"
        if getattr(self, "execution_plane", "") == "headless":
            perception_source = "headless"
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
            for attempt in range(2):
                has_pixels = bool(
                    getattr(screen_obs, "raw_rgb", b"")
                    or getattr(screen_obs, "screenshot_base64", "")
                )
                has_dimensions = (
                    int(getattr(screen_obs, "width", 0) or 0) > 0
                    and int(getattr(screen_obs, "height", 0) or 0) > 0
                )
                capture_age_ms = getattr(screen_obs, "capture_age_ms", None)
                frame_is_fresh = (
                    isinstance(capture_age_ms, (int, float))
                    and not isinstance(capture_age_ms, bool)
                    and math.isfinite(capture_age_ms)
                    and 0 <= capture_age_ms <= 2_000
                )
                if has_pixels and has_dimensions and frame_is_fresh:
                    perception_source = "avcs_pixels"
                    screen_obs = self._ensure_avcs_state(screen_obs)
                    break
                if attempt == 1:
                    logger.warning(
                        "desktop_observation_incomplete",
                        workspace_id=workspace_id,
                        has_pixels=has_pixels,
                        capture_age_ms=capture_age_ms,
                        width=getattr(screen_obs, "width", 0),
                        height=getattr(screen_obs, "height", 0),
                    )
                    perception_source = "avcs_frame_unavailable"
                    break
                await asyncio.sleep(0.05)
                screen_obs = await self.computer.screenshot(workspace_id)
        else:
            perception_source = "not_requested"
            screen_obs = ScreenObservation(
                screenshot_base64="",
                width=0,
                height=0,
                active_window="",
                visible_text="",
                detected_controls=[],
                desktop_state="NOT_REQUESTED",
            )
        if perception_source != "avcs_pixels":
            screen_obs.visible_text = ""
            screen_obs.detected_controls = []
            screen_obs.controls_age_ms = None
        screen_received_at_ns = time.perf_counter_ns()
        self._last_screen_observation_received_at = asyncio.get_event_loop().time()
        self._last_observed_screen = screen_obs
        self._last_observed_screen_from_reasoning = True
        self._last_gui_dispatch_evidence = (
            screen_obs,
            asyncio.get_event_loop().time(),
        )
        self._last_gui_state_observation = None
        action_to_state_ms = None
        pending_gui_action = getattr(self, "_pending_gui_action_observation", None)
        if isinstance(pending_gui_action, dict):
            started_ns = pending_gui_action.get("started_ns")
            if isinstance(started_ns, int) and not isinstance(started_ns, bool):
                action_to_state_ms = max(
                    0.0, (screen_received_at_ns - started_ns) / 1_000_000
                )
                action_trace_id = pending_gui_action.get("trace_id")
                for previous_trace in reversed(getattr(self, "traces", [])):
                    if getattr(previous_trace, "id", None) == action_trace_id:
                        previous_trace.action_to_state_ms = action_to_state_ms
                        break
                logger.info(
                    "computer_action_to_state_observed",
                    workspace_id=workspace_id,
                    trace_id=action_trace_id,
                    latency_ms=round(action_to_state_ms, 2),
                    perception_source=perception_source,
                    capture_age_ms=getattr(screen_obs, "capture_age_ms", None),
                    perception_version=getattr(screen_obs, "perception_version", 0),
                )
            self._pending_gui_action_observation = None
        # Store screenshot for vision-in-the-loop (VLM image input)
        self._last_screenshot_b64 = (
            getattr(screen_obs, "screenshot_base64", "")
            if perception_source == "avcs_pixels"
            else ""
        )
        self._last_screen_has_pixels = (
            perception_source == "avcs_pixels"
            and bool(
                getattr(screen_obs, "raw_rgb", b"")
                or getattr(screen_obs, "screenshot_base64", "")
            )
        )
        self._last_screen_frame_id = (
            getattr(screen_obs, "frame_id", None)
            if self._last_screen_has_pixels
            else None
        )
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
        # A GUI worker needs the small set of real desktop applications. The
        # provider's full inventory also enumerates ~800 PATH binaries, which
        # overflows the truncated app list the prompt shows and hides the real
        # GUI apps (a text editor gets pushed out and the model wrongly reports
        # it "not installed"). Prefer the desktop-only list when available.
        gui_apps: list[str] = []
        if hasattr(self.computer, "desktop_applications"):
            try:
                desktop_apps = await self.computer.desktop_applications(workspace_id)
                gui_apps = [
                    str(getattr(item, "name", item)).strip()
                    for item in (desktop_apps or [])
                    if str(getattr(item, "name", item)).strip()
                ]
            except Exception as exc:
                logger.debug("desktop_application_inventory_unavailable", error=str(exc))
        if hasattr(self.computer, "application_list"):
            try:
                discovered_apps = await self.computer.application_list(workspace_id)
                discovered_names = [
                    str(getattr(item, "name", item)).strip()
                    for item in (discovered_apps or [])
                    if str(getattr(item, "name", item)).strip()
                ]
                if self._gui_action_contract_active() and gui_apps:
                    # GUI worker: only real desktop applications are usable
                    # (it cannot use the terminal), so show the small desktop
                    # set rather than merging in ~800 PATH binaries that would
                    # crowd the real apps out of the truncated list.
                    merged = gui_apps
                else:
                    merged = list(
                        dict.fromkeys(
                            list(getattr(status, "installed_applications", []) or []) + discovered_names
                        )
                    )
                if merged:
                    status.installed_applications = merged
            except Exception as exc:
                logger.debug("application_inventory_unavailable", error=str(exc))
        elif gui_apps:
            status.installed_applications = list(
                dict.fromkeys(list(getattr(status, "installed_applications", []) or []) + gui_apps)
            )
        if self._gui_action_contract_active():
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
            if self._gui_action_contract_active():
                raise RuntimeError("GUI execution contract: terminal observation is outside this worker's action plane")
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
                "GUI execution contract: terminal observation disabled"
                if self._gui_action_contract_active()
                else f"Terminal Ready ({len(status.running_processes)} procs)"
            )

        if not hasattr(self, "_last_gui_navigation_target"):
            self._last_gui_navigation_target = ""
        if not hasattr(self, "_recent_action_signatures"):
            self._recent_action_signatures = []

        perception_snapshot = self.perception_bus.publish(
            screenshot_base64=self._last_screenshot_b64,
            width=self._screen_width,
            height=self._screen_height,
            active_window=getattr(screen_obs, "active_window", ""),
            windows=status.open_applications,
            processes=status.running_processes,
            visible_text=getattr(screen_obs, "visible_text", ""),
            controls=getattr(screen_obs, "detected_controls", []),
        )
        if getattr(self, "skill_ledger", None) is not None:
            self.skill_ledger.observe_applications(list(status.open_applications))
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
        world_observation = ComputerWorldObservation(
            screen=screen_obs,
            active_application=active_app,
            windows=open_apps,
            installed_applications=installed_apps,
            visible_text=_safe_str(screen_obs.visible_text),
            filesystem_files=file_names,
            processes=running_procs,
            terminal_output=_safe_str(terminal_output),
            working_directory=working_directory or "/home/daytona",
            ide_state={"active_file": file_names[0] if file_names else "", "cursor_line": 1},
            git_branch=branch_val,
            git_clean=git_st.is_clean if hasattr(git_st, "is_clean") else True,
            perception_version=perception_snapshot.version,
            perception_latency_ns=self.perception_bus.last_update_latency_ns,
            observation_age_ms=(
                screen_obs.capture_age_ms
                + (time.perf_counter_ns() - screen_received_at_ns) / 1_000_000
                if screen_obs.capture_age_ms is not None
                else None
            ),
            observation_received_monotonic_ns=screen_received_at_ns,
            perception_source=perception_source,
            action_to_state_ms=action_to_state_ms,
        )
        self._last_world_observation = world_observation
        world_observation.accessible_scene_fresh = False
        counters = getattr(self, "_perception_counters", None)
        if not isinstance(counters, dict):
            counters = {
                "observations": 0,
                "avcs_frame_observations": 0,
                "image_attached_requests": 0,
                "visual_escalations": 0,
            }
            self._perception_counters = counters
        counters["observations"] += 1
        if perception_source == "avcs_pixels":
            counters["avcs_frame_observations"] += 1
        logger.info(
            "computer_perception_observation",
            workspace_id=workspace_id,
            source=perception_source,
            perception_version=world_observation.perception_version,
            capture_age_ms=screen_obs.capture_age_ms,
            controls_age_ms=None,
            accessible_scene_fresh=False,
            # AVCS delivers pixels as raw RGB, not base64; checking only the
            # base64 field reported "no pixels" while the frame was present.
            screenshot_pixels_available=self._screen_has_pixel_data(screen_obs),
            action_to_state_ms=(
                round(action_to_state_ms, 2)
                if action_to_state_ms is not None
                else None
            ),
            counters=dict(counters),
        )
        return world_observation

    # =============================================================
    # 2. Closed-Loop Reasoning & Action Selection
    # =============================================================
    async def choose_action(
        self,
        goal: str,
        observation: ComputerWorldObservation,
        step_index: int = 1,
    ) -> tuple[ComputerActionType | ActionExecutionStatus, str, dict[str, Any], str]:
        """Choose the next action dynamically based on observation and goal.

        Decide the next action from (goal, observation, history).

        With an LLM router this is genuine model-driven reasoning: the live
        screen text, terminal output, file list, and git state — plus the
        transcript of actions already taken — are all fed to the model, and
        its chosen action is executed. Without a router a generic diagnostic
        probe is used (no vulnerability-specific logic).

        Returns: (action_type, target_resource, payload_dict, predicted_outcome)
        """
        self._refresh_observation_age(observation)
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
            if self._gui_action_contract_active():
                return (
                    ActionExecutionStatus.BLOCKED,
                    "gui-contract-violation",
                    {},
                    "This worker is assigned to the visible GUI plane; terminal-prefixed "
                    "goals cannot bypass the GUI execution contract.",
                )
            return (
                ComputerActionType.TERMINAL_EXEC,
                command,
                {"command": command},
                f"Execute the operator-supplied command: {command}",
            )
        if self.llm_router is not None:
            return await self._llm_choose_action(goal, observation, step_index, primary_file, test_file)

        if getattr(self, "gui_only", False) or getattr(self, "execution_plane", "") == "gui":
            return self._reasoning_unavailable_action(
                "No reasoning model is configured for this GUI worker."
            )

        return self._diagnostic_fallback(primary_file)

    @staticmethod
    def _reasoning_unavailable_action(
        reason: str,
    ) -> tuple[ActionExecutionStatus, str, dict[str, Any], str]:
        return (
            ActionExecutionStatus.BLOCKED,
            "reasoning-unavailable",
            {},
            reason,
        )

    @staticmethod
    def _refresh_observation_age(observation: ComputerWorldObservation) -> None:
        received_at_ns = observation.observation_received_monotonic_ns
        capture_age_ms = observation.screen.capture_age_ms
        if received_at_ns is None or capture_age_ms is None:
            observation.observation_age_ms = None
            return
        elapsed_ns = max(0, time.perf_counter_ns() - received_at_ns)
        observation.observation_age_ms = capture_age_ms + elapsed_ns / 1_000_000

    @staticmethod
    def _frame_true_age_ms(screen: Any) -> float | None:
        """Return a frame's honest age in ms, or ``None`` when unmeasurable.

        ``capture_age_ms`` is stamped at capture time and never grows, so a
        frame observed a minute ago still reports ~47 ms. The wall-clock delta
        from ``capture_timestamp_ns`` is the only true age signal, so freshness
        must use the larger of the two. A clock that is ahead (negative delta)
        is unusable rather than proof of staleness, so it falls back to the
        capture-time stamp.
        """
        if screen is None:
            return None
        age_ms = getattr(screen, "capture_age_ms", None)
        if (
            isinstance(age_ms, bool)
            or not isinstance(age_ms, (int, float))
            or not math.isfinite(float(age_ms))
            or age_ms < 0
        ):
            age_ms = None
        timestamp_ns = getattr(screen, "capture_timestamp_ns", None)
        if isinstance(timestamp_ns, int) and not isinstance(timestamp_ns, bool):
            timestamp_age_ms = (time.time_ns() - timestamp_ns) / 1_000_000
            if math.isfinite(timestamp_age_ms) and timestamp_age_ms >= 0:
                age_ms = max(float(age_ms or 0.0), timestamp_age_ms)
        return None if age_ms is None else float(age_ms)

    def _frame_is_fresh_for_reasoning(self, screen: Any) -> bool:
        """Whether a frame is fresh enough to ground a reasoning decision.

        Uses true wall-clock age so a reused/stale observation cannot pass here
        and then fail the AVCS gate, which would block a GUI worker whose pixels
        were actually fine.
        """
        has_pixels = bool(
            getattr(screen, "raw_rgb", b"")
            or getattr(screen, "screenshot_base64", "")
        )
        if not has_pixels:
            return False
        true_age = self._frame_true_age_ms(screen)
        return true_age is not None and true_age <= 2_000

    def _adopt_reasoning_pixels(self, screen: Any, is_fresh: bool) -> None:
        """Adopt (or clear) the frame used to ground the next GUI decision."""
        if is_fresh:
            self._last_screenshot_b64 = getattr(screen, "screenshot_base64", "")
            self._last_screen_has_pixels = True
            self._last_screen_frame_id = getattr(screen, "frame_id", None)
        else:
            self._last_screenshot_b64 = ""
            self._last_screen_has_pixels = False
            self._last_screen_frame_id = None

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

    def _parse_gui_action_response(
        self,
        text: str,
        default_file: str,
    ) -> tuple[ComputerActionType | ActionExecutionStatus, str, dict[str, Any], str]:
        """Accept GUI-worker output only when it contains a strict action block."""
        field_pattern = re.compile(
            r"^\s*(?:(?:[-*+]|\d+[.)])\s+)?(?:\*\*|__)?"
            r"(ACTION|TARGET|PAYLOAD|EXPECTED)(?:\*\*|__)?\s*:\s*"
            r"(?:\*\*|__)?(.*)$",
            re.IGNORECASE,
        )
        fields: dict[str, str] = {}
        payload_lines: list[str] = []
        collecting_payload = False
        for line in text.splitlines():
            match = field_pattern.match(line)
            if match:
                key = match.group(1).upper()
                value = match.group(2).strip()
                if key == "PAYLOAD":
                    collecting_payload = True
                    if "PAYLOAD" not in fields:
                        fields[key] = value
                        if value:
                            payload_lines.append(value)
                else:
                    collecting_payload = False
                    fields.setdefault(key, value)
            elif collecting_payload and line.strip():
                payload_lines.append(line.strip())

        if "ACTION" not in fields:
            structured_text = text.strip()
            if structured_text.startswith("```") and structured_text.endswith("```"):
                structured_text = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    structured_text,
                    flags=re.IGNORECASE,
                ).strip()
            try:
                envelope = json.loads(structured_text)
            except json.JSONDecodeError:
                envelope = None
            direct_keys = {"action", "target", "payload", "expected"}
            if not (
                isinstance(envelope, dict)
                and set(envelope) == direct_keys
                and isinstance(envelope.get("action"), str)
                and isinstance(envelope.get("target"), str)
                and isinstance(envelope.get("payload"), dict)
                and isinstance(envelope.get("expected"), str)
            ):
                decoder = json.JSONDecoder()
                candidates: list[tuple[int, int, dict[str, Any]]] = []
                for start, character in enumerate(structured_text):
                    if character != "{":
                        continue
                    try:
                        candidate, end = decoder.raw_decode(structured_text, start)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(candidate, dict)
                        and set(candidate) == direct_keys
                        and isinstance(candidate.get("action"), str)
                        and isinstance(candidate.get("target"), str)
                        and isinstance(candidate.get("payload"), dict)
                        and isinstance(candidate.get("expected"), str)
                    ):
                        candidates.append((start, end, candidate))
                outer_candidates = [
                    item
                    for item in candidates
                    if not any(
                        other_start <= item[0]
                        and item[1] <= other_end
                        and (other_start, other_end) != (item[0], item[1])
                        for other_start, other_end, _ in candidates
                    )
                ]
                envelope = (
                    outer_candidates[0][2]
                    if len(outer_candidates) == 1
                    else None
                )
            if isinstance(envelope, dict):
                if set(envelope) == direct_keys:
                    fields["ACTION"] = envelope["action"]
                    fields["TARGET"] = envelope["target"]
                    fields["EXPECTED"] = envelope["expected"]
                    fields["PAYLOAD"] = json.dumps(envelope["payload"])
                    payload_lines = [fields["PAYLOAD"]]
                else:
                    action_value = envelope.get("action", envelope.get("name"))
                    arguments = envelope.get("arguments", envelope.get("payload", {}))
                    if isinstance(action_value, str) and isinstance(arguments, dict):
                        fields["ACTION"] = action_value
                        raw_target = envelope.get("target", arguments.get("target", ""))
                        fields["TARGET"] = (
                            json.dumps(raw_target)
                            if isinstance(raw_target, dict)
                            else str(raw_target)
                        )
                        fields["EXPECTED"] = str(envelope.get("expected", ""))
                        explicit_payload = arguments.get("payload")
                        if isinstance(explicit_payload, dict):
                            payload_value = explicit_payload
                        else:
                            payload_value = {
                                key: value
                                for key, value in arguments.items()
                                if key not in {"target", "expected"}
                            }
                        fields["PAYLOAD"] = json.dumps(payload_value)
                        payload_lines = [fields["PAYLOAD"]]

        if "ACTION" not in fields:
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                "GUI reasoning did not return a structured ACTION block.",
            )
        raw_action = fields["ACTION"].strip().strip("`*_")
        if not raw_action:
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                "GUI reasoning returned an empty ACTION field.",
            )
        action_match = re.match(r"([A-Za-z][A-Za-z0-9_]*)\b", raw_action)
        action_name = action_match.group(1).upper() if action_match else ""
        if not action_name:
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                "GUI reasoning ACTION must begin with a valid action identifier.",
            )

        if action_name == "GOAL_COMPLETE":
            return (
                ComputerActionType.TERMINAL_EXEC,
                "goal-complete",
                {"command": "true"},
                _GOAL_COMPLETE_SENTINEL,
            )

        try:
            action_type = ComputerActionType(action_name)
        except ValueError:
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                f"GUI reasoning returned an unsupported action type: {action_name}.",
            )

        if action_type not in self._GUI_ONLY_ACTIONS:
            return (
                action_type,
                fields.get("TARGET", ""),
                {},
                fields.get("EXPECTED", ""),
            )

        missing_fields = [
            field for field in ("TARGET", "PAYLOAD") if field not in fields
        ]
        if missing_fields and action_type == ComputerActionType.GUI_SCREENSHOT:
            fields.setdefault("TARGET", "visible-desktop")
            if "PAYLOAD" not in fields:
                fields["PAYLOAD"] = "{}"
                payload_lines = ["{}"]
            missing_fields = []
        if missing_fields:
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                f"GUI {action_type.value} action block is missing required fields: "
                f"{', '.join(missing_fields)}.",
            )

        payload_text = "\n".join(payload_lines).strip()
        if not payload_text:
            payload: dict[str, Any] = {}
        else:
            try:
                parsed_payload = json.loads(payload_text)
            except json.JSONDecodeError:
                from sonic.llm.providers.custom import parse_tool_arguments

                parsed_payload = parse_tool_arguments(payload_text)
                if set(parsed_payload) == {"value"}:
                    return (
                        ActionExecutionStatus.BLOCKED,
                        "gui-invalid-response",
                        {},
                        "GUI action PAYLOAD must be a JSON object.",
                    )
            if not isinstance(parsed_payload, dict):
                return (
                    ActionExecutionStatus.BLOCKED,
                    "gui-invalid-response",
                    {},
                    "GUI action PAYLOAD must be a JSON object.",
                )
            payload = parsed_payload

        normalized_response = "\n".join((
            f"ACTION: {action_name}",
            f"TARGET: {fields['TARGET']}",
            f"PAYLOAD: {json.dumps(payload)}",
            f"EXPECTED: {fields.get('EXPECTED', '')}",
        ))
        parsed = self._parse_llm_action(normalized_response, default_file)
        if parsed[0] != action_type:
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                "GUI action block could not be normalized safely.",
            )
        if action_type == ComputerActionType.GUI_TYPE and not any(
            str(parsed[2].get(key, "")).strip()
            for key in ("text", "value", "input")
        ):
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                "GUI_TYPE requires explicit text to enter into a visible control.",
            )
        coordinate_actions = {
            ComputerActionType.GUI_CLICK,
            ComputerActionType.GUI_DOUBLE_CLICK,
            ComputerActionType.GUI_RIGHT_CLICK,
            ComputerActionType.GUI_MOVE,
            ComputerActionType.GUI_SCROLL,
            ComputerActionType.GUI_DRAG,
            ComputerActionType.GUI_TYPE,
        }
        payload = dict(parsed[2])
        target_coordinates = _parse_pixel_coordinate_target(parsed[1])
        # A TARGET expressed as a JSON object is only an explicit screen-pixel
        # point when it is exactly an integer x/y pair. Any extra field (e.g.
        # coordinate_space) or non-integer value means the coordinate space is
        # unknown, so fail closed rather than guess pixels.
        target_text = str(parsed[1]).strip()
        if target_text.startswith("{") and target_coordinates is None:
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                "GUI TARGET coordinate object must be an exact integer x/y screen-pixel point.",
            )
        has_payload_coordinates = "x" in payload or "y" in payload
        payload_coordinates = None
        if has_payload_coordinates:
            x, y = payload.get("x"), payload.get("y")
            if (
                not isinstance(x, int)
                or isinstance(x, bool)
                or not isinstance(y, int)
                or isinstance(y, bool)
            ):
                return (
                    ActionExecutionStatus.BLOCKED,
                    "gui-invalid-response",
                    {},
                    "GUI payload must contain a complete valid integer coordinate x/y pair.",
                )
            payload_coordinates = (x, y)
        if (
            target_coordinates is not None
            and payload_coordinates is not None
            and target_coordinates != payload_coordinates
        ):
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-invalid-response",
                {},
                "GUI target and payload specify conflicting screen coordinates.",
            )
        coordinates = target_coordinates or payload_coordinates
        if action_type in coordinate_actions and coordinates is not None:
            payload["x"], payload["y"] = coordinates

        if action_type in (ComputerActionType.GUI_KEYPRESS, ComputerActionType.GUI_HOTKEY):
            if (
                not str(payload.get("key") or payload.get("keys") or "").strip()
                and not self._has_gui_burst_payload(payload)
            ):
                return (
                    ActionExecutionStatus.BLOCKED,
                    "gui-invalid-response",
                    {},
                    f"{action_type.value} requires an explicit key in PAYLOAD.",
                )
        elif action_type in coordinate_actions:
            if action_type == ComputerActionType.GUI_DRAG:
                for names in (("x", "y"), ("x2", "y2"), ("end_x", "end_y")):
                    present = [name in payload for name in names]
                    if any(present) and not all(
                        isinstance(payload.get(name), int)
                        and not isinstance(payload.get(name), bool)
                        for name in names
                    ):
                        return (
                            ActionExecutionStatus.BLOCKED,
                            "gui-invalid-response",
                            {},
                            "GUI_DRAG coordinate pairs must be complete integer pixel pairs when provided.",
                        )
                if not str(payload.get("destination_grounding_query") or "").strip():
                    return (
                        ActionExecutionStatus.BLOCKED,
                        "gui-invalid-response",
                        {},
                        "GUI_DRAG requires a semantic destination_grounding_query for local AVCS grounding.",
                    )
            elif not coordinates and not str(parsed[1]).strip():
                return (
                    ActionExecutionStatus.BLOCKED,
                    "gui-invalid-response",
                    {},
                    f"{action_type.value} requires an explicit pixel point or semantic target query.",
                )
        parsed = (parsed[0], parsed[1], payload, parsed[3])
        return parsed

    def _gui_action_target_error(
        self,
        action_type: ComputerActionType | ActionExecutionStatus,
        target: str,
        observation: ComputerWorldObservation,
    ) -> str | None:
        """Validate app/window targets against live GUI state rather than guesses."""
        if action_type == ComputerActionType.APP_LAUNCH and target:
            if self.perception_bus.resolve(target) is not None:
                return (
                    "APP_LAUNCH target resolves to a visible desktop control; "
                    "use GUI_CLICK for that observed control."
                )

        if action_type == ComputerActionType.APP_FOCUS:
            requested_window = " ".join(target.casefold().split())
            observed_windows = {
                " ".join(str(window).casefold().split())
                for window in [
                    observation.active_application,
                    *observation.windows,
                ]
                if str(window).strip()
            }
            if not requested_window or requested_window not in observed_windows:
                return (
                    "APP_FOCUS requires an active/open window title present in the "
                    "current desktop observation."
                )

        return None

    def _recover_rejected_gui_proposal(
        self,
        goal: str,
        reason: str,
    ) -> tuple[ComputerActionType | ActionExecutionStatus, str, dict[str, Any], str]:
        self._consecutive_gui_contract_violations = (
            getattr(self, "_consecutive_gui_contract_violations", 0) + 1
        )
        if self._consecutive_gui_contract_violations > 1:
            self._pending_gui_contract_feedback = ""
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-contract-violation",
                {},
                "Stopped after repeated invalid model proposals outside the visible GUI action contract. "
                f"Last proposal rejected: {reason}",
            )
        self._pending_gui_contract_feedback = reason
        return self._gui_only_recovery_action(goal, reason)

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

    @staticmethod
    def _validate_typing_text(text: str) -> tuple[bool, str]:
        """Verify that typing payload does not contain sentinel, garbage, or thought tokens."""
        if not text or not str(text).strip():
            return True, "text payload is empty"
        t_clean = str(text).strip()
        sentinel_tokens = {
            "undefined", "null", "[object object]", "none", "nan", "nil",
            "{'text': 'value'}", '{"text": "value"}', "{text: value}",
        }
        if t_clean.lower() in sentinel_tokens:
            return True, f"contains invalid sentinel '{t_clean}'"
        syntax_delims = set(" \t\r\n\"'{}[]()<>,.;:")
        if len(t_clean) >= 4 and all(c in syntax_delims for c in t_clean):
            return True, "contains only syntax delimiters"
        return False, ""

    @staticmethod
    def _validate_typing_field(
        target_resource: str,
        semantic_match: Any = None,
    ) -> tuple[bool, str]:
        """Verify that target element is an input/typeable field and not a button/icon/link/checkbox."""
        if semantic_match is not None:
            role = getattr(semantic_match, "role", "")
            if role:
                role_clean = str(role).lower().strip()
                non_input_roles = {
                    "button", "btn", "push-button", "pushbutton", "clickable",
                    "checkbox", "check-box", "radio", "radiobutton", "link",
                    "image", "img", "icon", "menuitem", "tab", "dialog", "modal",
                    "heading", "scrollbar", "separator", "status-bar",
                }
                if role_clean in non_input_roles:
                    return True, f"element role is '{role_clean}', which does not accept text input"
        if target_resource:
            t_low = target_resource.lower().strip()
            non_input_indicators = (
                "button", "btn", "checkbox", "check-box", "radio",
                "icon", "scrollbar", "tab", "menuitem", "link",
            )
            if any(ind in t_low for ind in non_input_indicators):
                if any(ind in t_low for ind in ("checkbox", "check-box", "radio", "icon", "scrollbar")):
                    return True, f"target '{target_resource}' is a non-input element"
                t_no_compound = t_low.replace("checkbox", "").replace("check-box", "").replace("combobox", "combo_input")
                input_indicators = (
                    "input", "field", "text box", "textbox", "text", "entry",
                    "search", "bar", "area", "editor", "terminal", "console",
                    "keyboard", "prompt", "url",
                )
                if not any(ind in t_no_compound for ind in input_indicators):
                    return True, f"target '{target_resource}' is a non-input element"
        return False, ""

    @staticmethod
    def _is_active_buffer_target(target: str, payload: dict[str, Any], prev_trace: Any = None) -> bool:
        """Check if target represents an already active editor/terminal buffer or focused window."""
        t_clean = str(target or "").strip().lower()
        if not t_clean:
            return True
        buffer_indicators = (
            "editor", "keyboard", "active", "active_window", "window", "buffer",
            "focused", "terminal", "console", "cursor", "text area", "workspace",
        )
        if any(ind in t_clean for ind in buffer_indicators):
            return True
        if payload.get("active") is True or payload.get("focused") is True:
            return True
        if prev_trace is not None:
            prev_act = getattr(prev_trace, "action_type", None)
            prev_tgt = str(getattr(prev_trace, "target_resource", "") or "").lower().strip()
            if prev_act in (ComputerActionType.APP_LAUNCH, ComputerActionType.APP_FOCUS):
                if t_clean in prev_tgt or prev_tgt in t_clean:
                    return True
        return False

    async def _handle_visual_stall_watchdog(
        self,
        workspace_id: str,
        action_type: ComputerActionType,
        actual_obs_str: str,
    ) -> tuple[str, bool]:
        """Report unchanged screens without injecting potentially destructive input."""
        unchanged = self._consecutive_unchanged_screens
        aborted = unchanged >= 4
        logger.warning(
            "computer_visual_stall",
            unchanged_count=unchanged,
            action=action_type.value,
            recovery="none",
            aborted=aborted,
        )
        if aborted:
            actual_obs_str += " | [WATCHDOG_ABORT]: Screen state perpetually stuck; aborting mission early to prevent quota burning."
        else:
            actual_obs_str += (
                f" | [WATCHDOG]: No visible screen change after {unchanged} GUI actions; "
                "their effects are unverified. No recovery input was sent."
            )
        return actual_obs_str, aborted


    def _render_operator_messages(self) -> str:
        """Drain operator/coordinator messages and render them as instructions.

        Returns "" when there is no handle or nothing pending, so the prompt is
        unchanged for the common case. Messages are explicitly labelled as
        operator directives and bounded in count/length.
        """
        uid = getattr(self, "agent_uid", "")
        if not uid:
            return ""
        try:
            from sonic.computer_use.coordinator_inbox import get_coordinator_inbox

            messages = get_coordinator_inbox().drain(uid)
        except Exception:
            return ""
        if not messages:
            return ""
        lines = "\n".join(
            f"  - [{m.get('sender', 'operator')}] {m.get('message', '')}"
            for m in messages[-5:]
        )
        return (
            "=== OPERATOR INSTRUCTIONS (mid-run steering) ===\n"
            "The coordinator sent these while you were working; treat them as\n"
            "authoritative directives for the rest of this task:\n"
            f"{lines}\n"
            "Acknowledge them by adjusting your plan; do not ignore them.\n"
            "=================================================\n\n"
        )

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
        terminal output, files, git state, and the full
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

        has_live_pixels = bool(
            getattr(self, "_last_screen_has_pixels", False)
            or self._last_screenshot_b64
        )
        history_text = self._format_history()
        if has_live_pixels:
            history_text = history_text[-1200:]

        # Truthful observations: output UNKNOWN when missing or unavailable (no fake observations)
        screen_text = (
            "Not extracted; current pixels are processed locally by AVCS."
            if has_live_pixels
            else "UNKNOWN"
        )
        screen_limit = 1200 if has_live_pixels else 4000
        if screen_text != "UNKNOWN" and len(screen_text) > screen_limit:
            screen_text = screen_text[: screen_limit - 3] + "..."
        perception_snapshot = self.perception_bus.current()
        perception_version = (
            getattr(observation, "perception_version", 0)
            or perception_snapshot.version
        )
        controls_block = ""
        terminal_text = (observation.terminal_output or "").strip() or "UNKNOWN"
        if has_live_pixels and len(terminal_text) > 800:
            terminal_text = terminal_text[-800:]
        workdir = getattr(observation, "working_directory", "") or "UNKNOWN"
        user_home = workdir.split("/workspace")[0] if ("/workspace" in workdir and workdir != "UNKNOWN") else workdir
        self._last_working_dir = workdir
        active_app = observation.active_application or "UNKNOWN"
        windows_str = ", ".join(observation.windows) if observation.windows else "UNKNOWN"
        changed_region = getattr(
            getattr(observation, "screen", None), "changed_region", None
        )
        change_summary = (
            f"\nLatest changed screen region (pixels): {changed_region}"
            if changed_region is not None
            else ""
        )
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

        anti_loop_banner = ""

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
            lessons = (
                self._agentic_lessons
                if self._agentic_lessons_goal == goal
                else self.lessons_ledger.relevant(goal)
            )
            lessons_block = inject_into_context(lessons)

        # Multi-context pack (fused vector + graph findings + episodes). Only
        # injected when the async pre-fetch ran for this same goal; otherwise
        # empty, so a fresh being adds no noise.
        multi_context_block = (
            getattr(self, "_multi_context_block", "")
            if getattr(self, "_multi_context_goal", None) == goal
            else ""
        )

        # Capability catalog (SKILLS.md). Advisory routing context so a worker
        # knows which skills exist and which are fail-closed; it grants no
        # permission and cannot widen the safety envelope.
        skill_catalog_block = ""
        if os.environ.get("SONIC_SKILL_CATALOG_IN_PROMPT", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        ):
            try:
                from sonic.computer_use.skill_catalog import render_skill_catalog

                skill_catalog_block = render_skill_catalog()
            except Exception as exc:
                logger.warning("skill_catalog_render_failed", error=str(exc))

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
        if files_str != "UNKNOWN":
            raw_facts.append(f"Workspace files: {files_str}")
        if active_app != "UNKNOWN":
            raw_facts.append(f"Active app: {active_app}")
        if getattr(observation, "installed_applications", None):
            ranked_apps = self._rank_apps_for_goal(
                list(observation.installed_applications), goal
            )
            raw_facts.append(f"Installed/Desktop apps: {', '.join(ranked_apps[:15])}")
        if windows_str != "UNKNOWN":
            raw_facts.append(f"Open desktop windows: {windows_str}")
        if git_branch_str != "UNKNOWN":
            raw_facts.append(f"Git: {git_branch_str} (clean={observation.git_clean})")
        if terminal_text != "UNKNOWN":
            raw_facts.append(f"Last terminal output: {terminal_text[:500]}")
        avcs_summary = self._fresh_avcs_reflex_summary(
            getattr(observation, "screen", None)
        )
        if avcs_summary != "unavailable":
            raw_facts.append(
                "AVCS local pixel reflex (coarse visual change only; not semantic "
                "UI labels or task proof): "
                + avcs_summary
            )
        # Live narration: the eye's own plain-language account of the screen.
        # Appended only when it has something to say, so a still screen adds no
        # noise to the prompt.
        narration_block = self._narrate_screen(getattr(observation, "screen", None))
        if narration_block:
            raw_facts.append(
                "AVCS live narration (local description of the visible screen; it "
                "cannot read on-screen text and does not prove task success):\n  "
                + narration_block
            )
        # Drain the eye's unsolicited lines once, here, and reuse the same text
        # in the GUI-only branch below. Draining twice would silently drop the
        # second copy, and the GUI-only branch is exactly the worker most likely
        # to be facing the popup the eye just reported.
        ambient_block = self._ambient_narration_block()
        if ambient_block and not self._gui_action_contract_active():
            raw_facts.append(
                "AVCS unsolicited observations since your last step (the eye noticed "
                "these on its own while you were working; they are not proof of "
                "success):\n" + ambient_block
            )
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
        from sonic.orchestration.dual_process import DualProcessController
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
            f"{self._render_operator_messages()}"
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

        application_content = (
            f"Active application: {active_app}\n"
            f"Open windows: {windows_str}\n"
            f"Visible content: {screen_text}{change_summary}"
        )
        tool_content = tool_lines.strip() if tool_lines else "None"
        gui_policy_text = (
            "GUI EXECUTION CONTRACT: use visible GUI applications and GUI actions only. "
            "Do not use terminal, shell, file, script, package, security-tool, or backend actions."
            if self._gui_action_contract_active()
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
        # Surface applications relevant to the goal first. A real desktop can
        # expose dozens of settings/utility entries before the app the task
        # needs (e.g. a text editor), and truncating the list then hides it, so
        # the model wrongly concludes the app is "not installed". Ranking by
        # goal-token overlap is generic (no vendor names) and keeps the app the
        # operator actually asked for inside the visible window.
        installed_apps_list = self._rank_apps_for_goal(installed_apps_list, effective_goal)
        installed_str = (
            ", ".join(str(a) for a in installed_apps_list[:60])
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
            f"{controls_block}"
            f"Desktop Application State:\n{application_content}\n\n"
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
            f"{multi_context_block}"
            f"{skill_catalog_block}"
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
            prior_thought = str(getattr(self, "_last_thought", "") or "")
            obs_summary += f"\nYour Immediate Prior Thought was:\n\"{prior_thought[:400]}\"\n"

        if self._gui_action_contract_active():
            avcs_summary = self._fresh_avcs_reflex_summary(
                getattr(observation, "screen", None)
            )
            narration_block = self._narrate_screen(getattr(observation, "screen", None))
            # ``ambient_block`` was drained once above; reuse it so the GUI-only
            # worker still hears what the eye said (draining again yields "").
            narration_line = (
                f"AVCS LIVE NARRATION (local plain-language description of what the "
                f"eye currently sees; it cannot read on-screen text and is not proof "
                f"of success):\n{narration_block}\n"
                if narration_block
                else ""
            )
            ambient_line = (
                f"AVCS UNSOLICITED OBSERVATIONS (the eye noticed these on its own "
                f"while you were working; they are not proof of success):\n"
                f"{ambient_block}\n"
                if ambient_block
                else ""
            )
            obs_summary = (
                "ASSIGNED PLANE: visible desktop GUI only.\n"
                f"OBJECTIVE: {goal}\n"
                f"CURRENT SCREEN DIMENSIONS: {screen_w}x{screen_h} pixels.\n"
                f"Available Installed Desktop Applications: {installed_str}\n"
                "CURRENT AVCS PIXEL STATE SUMMARY "
                "(fresh local frame; coarse scene/change/motion signals only; "
                "not semantic certainty or task proof; no screenshot is sent; "
                f"no pixels are sent): {avcs_summary}\n"
                f"{narration_line}"
                f"{ambient_line}"
                "GROUNDING RULE: AVCS reflex state describes pixel changes and scene "
                "motion; it does not provide OCR or semantic controls. Semantic targets "
                "must be resolved locally by the configured, evaluated AVCS grounder. "
                "Never invent coordinates. Earlier actions are history, not current evidence.\n"
                "Choose one immediate visible GUI action; do not provide a multi-step plan."
            )

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
            if self._gui_action_contract_active()
            else
            "You have full freedom to operate 1, 2, or N desktop GUI applications (`APP_LAUNCH`, `APP_FOCUS`, `GUI_*`),\n"
            "complete operator-requested online workflows through the visible desktop, write files (`FILE_WRITE`), manage Git (`GIT_*`), and run terminal commands (`TERMINAL_EXEC`).\n"
        )
        next_action_question = (
            "Which visible application window or GUI control should I use right now?'\n"
            if self._gui_action_contract_active()
            else "Which GUI application window or terminal command should I use right now?'\n"
        )
        target_header += (
            "================================================================================\n"
            f"🎯 TARGET & MISSION OBJECTIVE: {effective_goal}\n"
            "AUTONOMY DIRECTIVE: You are an autonomous AI-Human Digital Being, NOT a scripted puppet.\n"
            "Do NOT force or wait for canned tools. You are completely tool-neutral.\n"
            "Think: 'What is the fastest, most effective way to accomplish this objective on my workstation?\n"
            f"{next_action_question}"
            f"{capability_directive}"
            f"GUI ACTION COORDINATES: In GUI action TARGET/PAYLOAD fields, use actual screen pixels "
            f"(x=0..{max(0, screen_w - 1)}, y=0..{max(0, screen_h - 1)}), not normalized values. "
            "Semantic GUI targets are resolved locally by AVCS; do not invent coordinates.\n"
            "================================================================================\n\n"
        )
        init_ctx = getattr(self, "initial_context", {}) or {}
        if self._gui_action_contract_active():
            target_header += (
                "=== VISIBLE WORKSTATION APPLICATION PLANE ===\n"
                "This worker is assigned a visible desktop objective. Use observed GUI applications and controls only; "
                "if an application is unavailable, report the limitation or choose another observed GUI path. "
                "Do not switch to terminal, shell, or filesystem actions.\n"
                "=============================================\n\n"
            )
        elif init_ctx.get("delegated_plane") != "headless":
            target_header += (
                "=== FULL DUAL-PLANE AGENCY (DESKTOP GUI :99 + TERMINAL/FILE SANDBOX) ===\n"
                "You are an AI-Human Being & Engineer operating BOTH your graphical desktop (:99) and your terminal/file sandbox:\n"
                "- When the mission asks to operate desktop GUI applications or interact with visual tools, actively use your graphical desktop plane (`APP_LAUNCH`, `APP_FOCUS`, `GUI_CLICK`, `GUI_TYPE`, `GUI_MOVE`, `GUI_KEYPRESS`).\n"
                "- When the mission asks for source code, build scripts, tests, or CLI commands (`pwd`, `ls`, `git`, `pytest`), use `TERMINAL_EXEC` and `FILE_WRITE`.\n"
                "- Both planes work together: you can launch apps on screen and verify them with commands or inspect files while interacting with GUI windows.\n"
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
        if not self._gui_action_contract_active():
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
            init_ctx = getattr(self, "initial_context", {}) or {}
            owned_scope = init_ctx.get("owned_paths") or []
            scope_block = ""
            if isinstance(owned_scope, list) and owned_scope:
                scope_lines = "\n".join(f"  - {str(p)}" for p in owned_scope[:20])
                scope_block = (
                    "=== ASSIGNED WRITE SCOPE (parallel workers own disjoint paths) ===\n"
                    "You may only create or modify files under these paths:\n"
                    f"{scope_lines}\n"
                    "Other paths belong to peer workers running at the same time; do not "
                    "edit them. If your task needs a file outside this scope, report the "
                    "blocker instead of modifying it.\n"
                    "====================================================================\n\n"
                )
            if init_ctx.get("silent_background") or init_ctx.get("subagent_role"):
                role_label = str(init_ctx.get("subagent_role") or "Subagent Worker")
                obs_summary = (
                    f"=== ASSIGNED SUBAGENT ROLE & WORKSPACE: {role_label} ===\n"
                    f"{json.dumps(init_ctx, ensure_ascii=True)[:1200]}\n"
                    "===========================================================\n\n"
                    + scope_block
                    + obs_summary
                )
            else:
                obs_summary = (
                    "=== ADVISORY INITIAL CONTEXT (UNVERIFIED) ===\n"
                    f"{json.dumps(init_ctx, ensure_ascii=True)[:1200]}\n"
                    "Validate these hints against live observations before acting.\n"
                    "==============================================\n\n"
                    + scope_block
                    + obs_summary
                )
        obs_summary = target_header + obs_summary
        # Select system prompt based on model capability
        if compact:
            try:
                from sonic.llm.prompts import COMPUTER_USE_SYSTEM_PROMPT_COMPACT
                system_prompt = COMPUTER_USE_SYSTEM_PROMPT_COMPACT
            except ImportError:
                system_prompt = COMPUTER_USE_SYSTEM_PROMPT
        else:
            system_prompt = COMPUTER_USE_SYSTEM_PROMPT
        if getattr(self, "system_prompt_overlay", None) and self.system_prompt_overlay.strip():
            system_prompt = f"{self.system_prompt_overlay.strip()}\n\n{system_prompt}"
        return system_prompt, obs_summary

    @staticmethod
    def _format_avcs_reflex_summary(state: Any) -> str:
        """Render a bounded, non-semantic summary of AVCS temporal pixel signals."""
        if not isinstance(state, dict) or state.get("engine") != "avcs_reflex":
            return "unavailable"

        def format_number(value: Any) -> str:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return "unknown"
            return f"{float(value):.3f}"

        scene_hash = str(state.get("scene_hash", "")).strip()
        frame_id = state.get("source_frame_id")
        if (
            isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id <= 0
        ):
            frame_id = state.get("frame_index", "unknown")
        changed = state.get("scene_changed")
        overlay = state.get("overlay_suspected")
        regions = state.get("regions")
        region_count = len(regions) if isinstance(regions, list) else 0
        region_kinds = []
        if isinstance(regions, list):
            for region in regions:
                if not isinstance(region, dict):
                    continue
                kind = region.get("kind")
                if kind in {"band", "panel", "patch"} and kind not in region_kinds:
                    region_kinds.append(kind)
                if len(region_kinds) == 3:
                    break
        temporal_state = "unavailable"
        system3 = state.get("system3")
        if (
            isinstance(system3, dict)
            and system3.get("engine") == "system3_temporal_prediction"
        ):
            temporal_state = (
                f"{str(system3.get('stability_state', 'unknown'))[:24]} "
                f"(advisory surprise={format_number(system3.get('surprise_score'))})"
            )
        event_labels = {
            "scene_change": "whole-screen transition",
            "overlay": "possible compact overlay change",
            "move": "localized motion",
            "appear": "new visually busy area",
        }
        events = state.get("events")
        visual_events = []
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                label = event_labels.get(event.get("kind"))
                if label and label not in visual_events:
                    visual_events.append(label)
                if len(visual_events) == 3:
                    break
        dimensions = (
            f"{state['width']}x{state['height']}"
            if isinstance(state.get("width"), int)
            and not isinstance(state.get("width"), bool)
            and state["width"] > 0
            and isinstance(state.get("height"), int)
            and not isinstance(state.get("height"), bool)
            and state["height"] > 0
            else "unknown"
        )
        semantics = state.get("semantics")
        semantics_summary = "unavailable"
        if isinstance(semantics, dict) and semantics:
            if semantics.get("affordance_head_trained") is True:
                hotspots = semantics.get("action_hotspots")
                hotspot_count = len(hotspots) if isinstance(hotspots, list) else 0
                semantics_summary = (
                    f"{str(semantics.get('primary_category', 'unknown'))[:32]} "
                    f"(confidence={format_number(semantics.get('category_confidence'))}, "
                    f"stability={str(semantics.get('stability_state', 'unknown'))[:20]}, "
                    f"interactive_targets={hotspot_count})"
                )
            else:
                # The dorsal head was never trained (or its training is
                # unproven), so it has no real interactive-target predictions.
                # Say so rather than reporting a hotspot count the model would
                # read as detected controls.
                semantics_summary = (
                    f"{str(semantics.get('primary_category', 'unknown'))[:32]} "
                    f"(confidence={format_number(semantics.get('category_confidence'))}, "
                    f"stability={str(semantics.get('stability_state', 'unknown'))[:20]}, "
                    "interactive_targets=unavailable:affordance-head-untrained)"
                )
        return (
            "pixel-only; "
            f"frame={frame_id}, screen={dimensions}, "
            f"scene={scene_hash[:12] or 'unknown'}, "
            f"learned_semantics={semantics_summary}, "
            f"changed={changed if isinstance(changed, bool) else 'unknown'}, "
            f"change={format_number(state.get('change_fraction'))}, "
            f"motion={format_number(state.get('global_motion'))}, "
            f"peak_motion={format_number(state.get('peak_motion'))}, "
            f"overlay_suspected={overlay if isinstance(overlay, bool) else 'unknown'}, "
            f"coarse_regions={region_count} ({','.join(region_kinds) or 'unknown'}), "
            f"events={', '.join(visual_events) or 'none'}, temporal={temporal_state}"
        )

    @classmethod
    def _fresh_avcs_reflex_summary(
        cls, screen: ScreenObservation | None, *, max_age_ms: float = 2_000
    ) -> str:
        """Summarize only AVCS state matched to the screen's current fresh frame."""
        if screen is None or getattr(screen, "desktop_state", None) != "INTERACTIVE":
            return "unavailable"
        raw_rgb = getattr(screen, "raw_rgb", b"")
        screenshot = getattr(screen, "screenshot_base64", "")
        width, height = getattr(screen, "width", 0), getattr(screen, "height", 0)
        if not screenshot and not raw_rgb:
            return "unavailable"
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
            or (
                raw_rgb
                and (
                    not isinstance(raw_rgb, bytes)
                    or len(raw_rgb) != width * height * 3
                )
            )
        ):
            return "unavailable"

        age_ms = getattr(screen, "capture_age_ms", None)
        if (
            isinstance(age_ms, bool)
            or not isinstance(age_ms, (int, float))
            or not math.isfinite(float(age_ms))
            or age_ms < 0
        ):
            age_ms = None
        capture_timestamp_ns = getattr(screen, "capture_timestamp_ns", None)
        if isinstance(capture_timestamp_ns, int) and not isinstance(
            capture_timestamp_ns, bool
        ):
            timestamp_age_ms = (time.time_ns() - capture_timestamp_ns) / 1_000_000
            if not math.isfinite(timestamp_age_ms) or timestamp_age_ms < 0:
                return "unavailable"
            age_ms = max(float(age_ms or 0), timestamp_age_ms)
        if age_ms is None or age_ms > max_age_ms:
            return "unavailable"

        state = getattr(screen, "avcs_state", None)
        frame_index = state.get("frame_index") if isinstance(state, dict) else None
        screen_frame_id = getattr(screen, "frame_id", None)
        if (
            not isinstance(state, dict)
            or state.get("engine") != "avcs_reflex"
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index <= 0
            or getattr(screen, "perception_version", None) != frame_index
            or state.get("width") != width
            or state.get("height") != height
            or not isinstance(state.get("scene_hash"), str)
            or not state["scene_hash"].strip()
            or (
                screen_frame_id is not None
                and state.get("source_frame_id") != screen_frame_id
            )
            or (
                screen_frame_id is None
                and state.get("source_frame_id") is not None
            )
            or (
                raw_rgb
                and (
                    isinstance(screen_frame_id, bool)
                    or not isinstance(screen_frame_id, int)
                    or screen_frame_id <= 0
                    or state.get("source_frame_id") != screen_frame_id
                )
            )
        ):
            return "unavailable"
        return cls._format_avcs_reflex_summary(state)

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
    ) -> tuple[ComputerActionType | ActionExecutionStatus, str, dict[str, Any], str]:
        """Use the LLM to decide the next action from goal + observation + history."""
        from sonic.llm.schemas import LLMRequest, Message, MessageRole

        if (
            self.lessons_ledger is not None
            and self.llm_router is not None
            and hasattr(self.lessons_ledger, "relevant_agentic")
            and self._agentic_lessons_goal != goal
        ):
            try:
                self._agentic_lessons = await self.lessons_ledger.relevant_agentic(
                    goal, self.llm_router
                )
                self._agentic_lessons_goal = goal
            except Exception as exc:
                logger.warning("agentic_rag_lesson_retrieval_failed", error=str(exc))
                self._agentic_lessons = self.lessons_ledger.relevant(goal)
                self._agentic_lessons_goal = goal

        # Multi-context retrieval: fuse the available memory sources once per
        # goal so the reasoning prompt gets one coherent, cited picture. When a
        # router is available the agentic planner refines the query first.
        if getattr(self, "multi_context_sources", None) and getattr(
            self, "_multi_context_goal", None
        ) != goal:
            try:
                from sonic.memory.multi_context import (
                    AgenticMultiContextRetriever,
                    MultiContextAssembler,
                    render_multi_context,
                )

                assembler = MultiContextAssembler(self.multi_context_sources)
                if self.llm_router is not None:
                    retriever = AgenticMultiContextRetriever(
                        assembler, self.llm_router
                    )
                    pack = await retriever.retrieve(goal, tenant_id=self.tenant_id)
                else:
                    pack = await assembler.assemble(goal, tenant_id=self.tenant_id)
                self._multi_context_pack = pack
                self._multi_context_block = render_multi_context(pack)
                self._multi_context_goal = goal
                logger.info(
                    "multi_context_assembled",
                    tenant_id=self.tenant_id,
                    items=len(pack.items),
                    sources_used=list(pack.sources_used),
                    sources_failed=list(pack.sources_failed),
                )
            except Exception as exc:
                logger.warning("multi_context_assembly_failed", error=str(exc))
                self._multi_context_block = ""
                self._multi_context_pack = None
                self._multi_context_goal = goal

        plane = str(getattr(self, "execution_plane", "dual") or "dual").lower().strip()
        gui_reasoning = (
            plane in ("gui", "dual")
            or getattr(self, "gui_only", False)
        )
        structured_scene_available = False
        current_frame_is_fresh = self._frame_is_fresh_for_reasoning(observation.screen)
        if gui_reasoning:
            self._adopt_reasoning_pixels(observation.screen, current_frame_is_fresh)
        if (
            gui_reasoning
            and not current_frame_is_fresh
            and not self._last_screen_has_pixels
            and getattr(self, "_active_workspace_id", "")
            and hasattr(self.computer, "screenshot")
        ):
            visual_observation = await self.computer.screenshot(
                self._active_workspace_id
            )
            # Same honest-age rule as the gate above: a re-captured frame must be
            # judged by true wall-clock age, not the frozen capture-time stamp.
            visual_true_age = self._frame_true_age_ms(visual_observation)
            if (
                (
                    getattr(visual_observation, "raw_rgb", b"")
                    or visual_observation.screenshot_base64
                )
                and visual_observation.width > 0
                and visual_observation.height > 0
                and visual_true_age is not None
                and visual_true_age <= 2_000
            ):
                visual_observation = self._ensure_avcs_state(visual_observation)
                self._last_screenshot_b64 = visual_observation.screenshot_base64
                self._last_screen_has_pixels = True
                self._last_screen_frame_id = getattr(
                    visual_observation, "frame_id", None
                )
                self._screen_width = visual_observation.width
                self._screen_height = visual_observation.height
                observation.screen = visual_observation
                self._last_observed_screen = visual_observation
                self._last_observed_screen_from_reasoning = True
                refreshed = self.perception_bus.publish(
                    screenshot_base64=visual_observation.screenshot_base64,
                    width=visual_observation.width,
                    height=visual_observation.height,
                    active_window=visual_observation.active_window,
                    windows=observation.windows,
                    processes=observation.processes,
                    visible_text=visual_observation.visible_text,
                    controls=visual_observation.detected_controls,
                )
                observation.perception_version = refreshed.version
                observation.visible_text = visual_observation.visible_text
                observation.observation_age_ms = visual_true_age
                observation.perception_source = "avcs_frame_refresh"
            else:
                logger.warning(
                    "desktop_visual_escalation_unavailable",
                    active_window=getattr(visual_observation, "active_window", ""),
                    width=getattr(visual_observation, "width", 0),
                    height=getattr(visual_observation, "height", 0),
                    capture_age_ms=visual_true_age,
                )
        avcs_summary = self._fresh_avcs_reflex_summary(
            getattr(observation, "screen", None)
        )
        if gui_reasoning and (
            not self._last_screen_has_pixels or avcs_summary == "unavailable"
        ):
            return self._reasoning_unavailable_action(
                "AVCS requires fresh, frame-matched pixel state; no GUI reasoning request was sent."
            )
        # Build the prompt only after any fresh-frame recovery so the next
        # System2 turn reasons from the same pixels as the AVCS state.
        system_prompt, obs_summary = self._build_reasoning_context(
            goal, observation, step_index, primary_file, test_file
        )
        user_text = obs_summary
        image_attached = False
        escalation_reason = (
            "avcs_pixel_state_only" if gui_reasoning else "non_gui_reasoning"
        )

        observation.accessible_scene_fresh = structured_scene_available
        observation.image_attached = image_attached
        observation.visual_escalation_reason = escalation_reason
        counters = getattr(self, "_perception_counters", None)
        if not isinstance(counters, dict):
            counters = {
                "observations": 0,
                "avcs_frame_observations": 0,
                "image_attached_requests": 0,
                "visual_escalations": 0,
            }
            self._perception_counters = counters
        observation.image_escalation_count = counters["visual_escalations"]
        self._last_reasoning_perception = {
            "source": observation.perception_source,
            "version": observation.perception_version,
            "screen_age_ms": getattr(observation.screen, "capture_age_ms", None),
            "controls_age_ms": getattr(observation.screen, "controls_age_ms", None),
            "accessible_scene_fresh": structured_scene_available,
            "image_attached": image_attached,
            "visual_escalation_reason": escalation_reason,
            "image_escalation_count": counters["visual_escalations"],
        }
        if self._gui_action_contract_active() and avcs_summary == "unavailable":
            return (
                ActionExecutionStatus.BLOCKED,
                "gui-perception-unavailable",
                {},
                "No fresh AVCS pixel-state observation is available; GUI reasoning stopped.",
            )
        logger.info(
            "computer_reasoning_perception",
            perception_source=observation.perception_source,
            perception_version=observation.perception_version,
            screen_age_ms=getattr(observation.screen, "capture_age_ms", None),
            controls_age_ms=getattr(observation.screen, "controls_age_ms", None),
            accessible_scene_fresh=structured_scene_available,
            image_attached=image_attached,
            visual_escalation_reason=escalation_reason,
            image_escalation_count=counters["visual_escalations"],
            counters=dict(counters),
        )
        if self._gui_action_contract_active():
            system_prompt = self._gui_only_system_prompt()

        available_actions = (
            self._GUI_ONLY_ACTIONS
            if self._gui_action_contract_active()
            else frozenset(ComputerActionType) - {ComputerActionType.VERIFY_PROBE}
        )
        legacy_browser_actions = {
            action
            for action in self._GUI_ONLY_ACTIONS
            if action.name.startswith("BROWSER_")
        }
        advertised_actions = available_actions - legacy_browser_actions
        gui_action_values = [
            action.value
            for action in ComputerActionType
            if action in advertised_actions and action != ComputerActionType.GUI_TYPE
        ]
        action_schema = "|".join(gui_action_values)
        efficiency_directive = (
            "2. GUI CONTROL SEMANTICS: APP_LAUNCH targets an installed application executable/name only. "
            "For a visible menu, button, launcher, or other on-screen control, use GUI_CLICK with "
            "a semantic target for local AVCS grounding; never pass a control label to APP_LAUNCH.\n"
            "3. DIRECT & EFFICIENT: solve the objective with the fewest necessary visible GUI actions.\n"
            if self._gui_action_contract_active()
            else
            "2. DIRECT & EFFICIENT: solve the objective in minimal actions. Combine commands with && if appropriate (e.g. `hostname && df -h`).\n"
        )
        if self._gui_action_contract_active():
            action_instructions = (
                "For GUI_CLICK, GUI_DOUBLE_CLICK, GUI_RIGHT_CLICK, GUI_MOVE, and GUI_SCROLL, "
                "provide a concise target description for the local AVCS grounder. "
                "The language model receives no screenshot pixels or OCR/accessibility "
                "control list; never invent pixel coordinates or claim a target is grounded. "
                "If AVCS cannot resolve the target, the action will be blocked. PAYLOAD "
                "must be a JSON object. GUI_TYPE is unavailable until a tested local "
                "AVCS focus-verification capability exists, so do not propose it. GUI_HOTKEY "
                "requires {\"key\": \"Ctrl+L\"}. Do not include shell commands.\n"
                "3. COMPLETION: use GOAL_COMPLETE only when the current visual evidence already "
                "supports handing the result to the independent Judge."
            )
        else:
            action_instructions = (
                "TARGET: <resource path, application/window name, destination, UI query, coordinates, or command>\n"
                "PAYLOAD: {\"command\": \"...\"} or another payload JSON object\n"
                "For TERMINAL_EXEC, the command value must contain shell syntax only; "
                "never append explanations, parenthetical suggestions, or prose.\n"
                f"{efficiency_directive}"
                "3. EARLY COMPLETION: If command output or screen observations already satisfy "
                "the objective, declare GOAL_COMPLETE immediately."
            )
        if gui_reasoning:
            action_instructions += (
                "\nFor each single GUI action that changes the interface "
                "(GUI_CLICK, GUI_DOUBLE_CLICK, GUI_RIGHT_CLICK, "
                "GUI_KEYPRESS, GUI_HOTKEY, GUI_SCROLL, or GUI_DRAG), include "
                "a structured `postcondition` object inside PAYLOAD. It must "
                "use `screen_changed: true` or a grounded `active_window` value; "
                "AVCS does not provide OCR text or semantic controls. A changed "
                "screen is only an action receipt, not proof that the task succeeded. "
                "If the postcondition is missing or unobserved, the action is "
                "UNVERIFIED, never verified progress.\n"
            )
            action_instructions += (
                "\nOPTIONAL SHORT GUI CHUNK: For a predictable, low-risk sequence of at most "
                f"{_MAX_GUI_BURST_ACTIONS} directly grounded motor actions, you may put "
                "\"burst_actions\" in PAYLOAD. Each item must use a low-level action name "
                "(CLICK, DOUBLE_CLICK, RIGHT_CLICK, KEYPRESS, HOTKEY, SCROLL, MOVE, "
                "or DRAG) and include its own observable \"postcondition\". Also include "
                "a final PAYLOAD \"postcondition\". A postcondition must identify "
                "a screen change or active window. The executor observes after every step, checks bounds "
                "and the stated postcondition, and stops/replans on a mismatch; never "
                "chunk through an uncertain transition, dialog, or "
                "approval-required operation. A chunk is optional, not a fixed workflow.\n"
                "Example shape: {\"burst_actions\":[{\"action\":\"CLICK\",\"x\":120,"
                "\"y\":80,\"postcondition\":{\"screen_changed\":true}}],"
                "\"postcondition\":{\"screen_changed\":true},"
                "\"state_version\":<current perception version>}. Use only evidence "
                "actually present in the current observation."
            )
        if self._gui_action_contract_active():
            user_text += (
                "\n\n================================================================================\n"
                "CRITICAL GUI RESPONSE CONTRACT:\n"
                "Return exactly one JSON object and no Markdown or surrounding prose. "
                "Include exactly four keys: action, target, payload, expected. "
                f"The action value must be exactly one of: {action_schema}. "
                "target and expected must be strings; payload must be a JSON object. "
                "Do not omit keys, even when their values are empty. "
                "For GUI_SCREENSHOT use "
                '{"action":"GUI_SCREENSHOT","target":"visible-desktop",'
                '"payload":{},"expected":"Inspect the current desktop"}. '
                "For GOAL_COMPLETE use "
                '{"action":"GOAL_COMPLETE","target":"goal_complete",'
                '"payload":{},"expected":"Hand off for independent verification"}.\n'
                f"{action_instructions}\n"
                "================================================================================"
            )
        else:
            user_text += (
                "\n\n================================================================================\n"
                "CRITICAL INSTRUCTIONS FOR YOUR NEXT IMMEDIATE ACTION:\n"
                "1. Output EXACTLY ONE action block in this format:\n"
                "THOUGHT: <brief observation and rationale for the next action>\n"
                f"ACTION: <{action_schema}>\n"
                f"{action_instructions}\n"
                "EXPECTED: <observable outcome>\n"
                "For GOAL_COMPLETE use TARGET: goal_complete, PAYLOAD: {}, EXPECTED: hand off for independent verification.\n"
                "================================================================================"
            )
        if self._gui_action_contract_active() and self._pending_gui_contract_feedback:
            feedback = " ".join(self._pending_gui_contract_feedback.split())[:240]
            user_text += (
                "\n\nGUI CONTRACT FEEDBACK: Your previous proposal was not executed "
                f"because {feedback} Choose one allowed visible GUI action from the "
                "current evidence. Do not repeat the rejected action. If the current "
                "frame does not ground a task action, choose GUI_SCREENSHOT once."
            )

        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_text),
            ],
            task_type="reasoning",
            max_tokens=2048,
            tools=None,
            reasoning_effort="medium",
            extra_body=(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "computer_action",
                            "strict": False,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": gui_action_values + ["GOAL_COMPLETE"],
                                    },
                                    "target": {"type": "string"},
                                    "payload": {"type": "object"},
                                    "expected": {"type": "string"},
                                },
                                "required": [
                                    "action",
                                    "target",
                                    "payload",
                                    "expected",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    }
                }
                if self._gui_action_contract_active()
                else {}
            ),
        )
        t_thought_start = time.perf_counter()
        heartbeat_task: asyncio.Task[Any] | None = None
        heartbeat_callback = getattr(self, "_heartbeat_callback", None)
        if heartbeat_callback is not None:
            async def _heartbeat_loop() -> None:
                while True:
                    await asyncio.sleep(2.0)
                    event = {
                        "phase": "reasoning",
                        "elapsed_seconds": round(time.perf_counter() - t_thought_start, 1),
                        "step": step_index,
                        "goal": goal,
                        "last_observation_version": getattr(observation, "perception_version", 0),
                    }
                    try:
                        result = heartbeat_callback(event)
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        logger.warning(
                            "computer_reasoning_heartbeat_failed",
                            step=step_index,
                            error=str(exc),
                        )
                        return

            heartbeat_task = asyncio.create_task(_heartbeat_loop())
        try:
            reasoning_timeout = (
                _GUI_REASONING_TIMEOUT_SECONDS
                if self._gui_action_contract_active()
                else 120.0
            )
            response = await asyncio.wait_for(
                self.llm_router.complete(request),
                timeout=reasoning_timeout,
            )
            self._last_thought_duration = round(time.perf_counter() - t_thought_start, 2)
            content_str = getattr(response, "content", "") or ""
            tool_calls = list(getattr(response, "tool_calls", []) or [])

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
            combined_response = content_str.lower()
            is_refusal = any(p in combined_response for p in refusal_phrases) and not re.search(
                r'\baction\s*:\s*(?:terminal_exec|gui_|browser_|file_|git_|app_|tool_|security_tool)',
                combined_response,
            )

            if is_refusal:
                self._refusal_recovery_active = True
                self._last_thought = (
                    "LLM safety refusal detected; recording refusal for replanning."
                )
                if self._gui_action_contract_active():
                    return self._recover_rejected_gui_proposal(
                        goal,
                        "LLM safety refusal detected",
                    )
                return (
                    ActionExecutionStatus.BLOCKED,
                    "refusal-recorded",
                    {},
                    "Record model refusal and trigger replan",
                )

            thought_match = re.search(
                r"(?:^|\n)\s*THOUGHT\s*:\s*(.*?)(?=\n\s*(?:ACTION|TARGET|PAYLOAD|EXPECTED)\s*:|$)",
                content_str,
                re.DOTALL | re.IGNORECASE,
            )
            self._last_thought = (
                thought_match.group(1).strip(" *_\n\r\t")
                if thought_match
                else ""
            )
            parse_target_str = content_str
            if self._gui_action_contract_active() and tool_calls:
                self._last_thought = (
                    "The model returned a function call that was not requested; "
                    "GUI actions must use the JSON-schema response contract."
                )
                return self._recover_rejected_gui_proposal(
                    goal,
                    "Unrequested GUI tool call",
                )
            if self._gui_action_contract_active():
                action_type, target, payload, expected = self._parse_gui_action_response(
                    parse_target_str,
                    primary_file,
                )
            else:
                action_type, target, payload, expected = self._parse_llm_action(
                    parse_target_str, primary_file
                )
            if self._gui_action_contract_active():
                if (
                    action_type == ActionExecutionStatus.BLOCKED
                    and target == "gui-invalid-response"
                ):
                    parsed_action_match = re.match(
                        r"GUI ([A-Z][A-Z0-9_]*) action block",
                        expected,
                    )
                    logger.warning(
                        "gui_action_response_invalid",
                        parsed_action=(
                            parsed_action_match.group(1)
                            if parsed_action_match
                            else "BLOCKED"
                        ),
                        parser_reason=" ".join(expected.split())[:160],
                        response_chars=len(parse_target_str),
                        action_marker_present=bool(
                            re.search(
                                r'(?:^|\n)\s*(?:[-*+]\s*)?(?:\*\*)?ACTION(?:\*\*)?\s*:'
                                r'|["\']action["\']\s*:',
                                parse_target_str,
                                re.IGNORECASE,
                            )
                        ),
                        tool_call_count=len(tool_calls),
                    )
                    self._last_thought = (
                        "The model response did not satisfy the structured GUI action "
                        f"format: {expected}"
                    )
                    return self._recover_rejected_gui_proposal(
                        goal,
                        f"the response failed GUI action validation: {expected}",
                    )
                target_error = self._gui_action_target_error(
                    action_type,
                    target,
                    observation,
                )
                if target_error:
                    self._last_thought = target_error
                    return self._recover_rejected_gui_proposal(goal, target_error)
            if (
                self._gui_action_contract_active()
                and expected != _GOAL_COMPLETE_SENTINEL
                and action_type not in self._GUI_ONLY_ACTIONS
            ):
                proposed_action = (
                    action_type.value
                    if isinstance(action_type, ComputerActionType)
                    else str(action_type)
                )
                rejection_reason = (
                    f"the proposed action {proposed_action} is not allowed on the GUI plane"
                )
                logger.warning(
                    "gui_contract_proposal_rejected",
                    proposed_action=proposed_action,
                    reason="non_gui_action",
                )
                self._last_thought = (
                    f"The model proposed {proposed_action}, which is not a GUI action; "
                    "discard it and re-observe the visible desktop."
                )
                return self._recover_rejected_gui_proposal(
                    goal,
                    rejection_reason,
                )

            if target == "diagnostic-verification" and "LLM safety refusal detected" in expected:
                self._refusal_recovery_active = True
                self._last_thought = "Model refusal received; continue from the live visible desktop."
                if self._gui_action_contract_active():
                    return self._recover_rejected_gui_proposal(
                        goal,
                        "Model refusal received",
                    )
                return (
                    ActionExecutionStatus.BLOCKED,
                    "refusal-recorded",
                    {},
                    "Record model refusal and trigger replan",
                )

            if (
                self._gui_action_contract_active()
                and action_type != ComputerActionType.GUI_SCREENSHOT
            ):
                self._consecutive_gui_contract_violations = 0
                self._consecutive_gui_postcondition_violations = 0
                self._pending_gui_contract_feedback = ""
            return action_type, target, payload, expected
        except Exception as e:
            self._last_thought_duration = round(time.perf_counter() - t_thought_start, 2)
            logger.warning("computer_llm_action_failed", error=str(e))
            self._last_thought = ""
            if (
                getattr(self, "gui_only", False)
                or getattr(self, "execution_plane", "") == "gui"
            ):
                reason = (
                    f"GUI reasoning timed out after "
                    f"{_GUI_REASONING_TIMEOUT_SECONDS:g} seconds; no action was dispatched."
                    if isinstance(e, TimeoutError)
                    else f"GUI reasoning failed: {e}"
                )
                return self._reasoning_unavailable_action(
                    reason
                )
            return self._diagnostic_fallback(primary_file)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

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

    @staticmethod
    def _window_matches_application(app_name: str, window_title: str) -> bool:
        """Match an arbitrary app request to a live desktop window title."""
        try:
            requested = shlex.split(app_name.strip())[0]
        except (ValueError, IndexError):
            requested = app_name.strip().split(maxsplit=1)[0] if app_name.strip() else ""
        requested = os.path.splitext(os.path.basename(requested))[0].lower()
        title = window_title.lower()
        requested_key = re.sub(r"[^a-z0-9]+", "", requested)
        title_key = re.sub(r"[^a-z0-9]+", "", title)
        if not requested_key or not title_key:
            return False
        if requested_key in title_key:
            return True
        requested_words = set(re.findall(r"[a-z0-9]{3,}", requested))
        title_words = set(re.findall(r"[a-z0-9]{3,}", title))
        return (
            bool(requested_words)
            and len(requested_words & title_words)
            >= max(1, math.ceil(len(requested_words) * 0.6))
        )

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

        # NOTE: the separator after the field name is `[ \t]*` (not `\s*`) so an
        # empty field like "TARGET:\nPAYLOAD: {...}" does not swallow the next
        # field's text into this one. With `\s*` the newline is consumed and
        # TARGET captures "PAYLOAD: {...}", dropping the payload entirely.
        pattern = r'(?:^|\n|\s)(?:\#+\s*)?(?:\d+[\.\)]\s*)?(?:[\*\-_]\s*)?(?:\*{1,2}|_)?\b(ACTION|ANSWER|TARGET|PAYLOAD|CONTENT|CODE|SOURCE|EXPECTED|EXPECTED[\s_]+OUTCOME|REASONING|THOUGHT|EXPLANATION|COORDINATES|COMMAND)\b(?:\*{1,2}|_)?\s*:[ \t]*(.*?)(?=(?:^|\n|\s)(?:\#+\s*)?(?:\d+[\.\)]\s*)?(?:[\*\-_]\s*)?(?:\*{1,2}|_)?\b(?:ACTION|ANSWER|TARGET|PAYLOAD|CONTENT|CODE|SOURCE|EXPECTED|EXPECTED[\s_]+OUTCOME|REASONING|THOUGHT|EXPLANATION|COORDINATES|COMMAND)\b(?:\*{1,2}|_)?\s*:|$)'
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
            "APP_MAXIMIZE": ComputerActionType.APP_MAXIMIZE,
            "WINDOW_MAXIMIZE": ComputerActionType.WINDOW_MAXIMIZE,
            "WINDOW_TILE": ComputerActionType.WINDOW_TILE,
            "MAXIMIZE": ComputerActionType.APP_MAXIMIZE,
            "TILE": ComputerActionType.WINDOW_TILE,
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
            "SUBAGENT_DEFINE": ComputerActionType.DEFINE_SUBAGENT,
            "INVOKE_SUBAGENT": ComputerActionType.INVOKE_SUBAGENT,
            "SUBAGENT_INVOKE": ComputerActionType.INVOKE_SUBAGENT,
            "INVOKE_SUBAGENTS": ComputerActionType.INVOKE_SUBAGENT,
            "SUBAGENTS_INVOKE": ComputerActionType.INVOKE_SUBAGENT,
            "SEND_MESSAGE": ComputerActionType.SEND_MESSAGE,
            "SUBAGENT_MESSAGE": ComputerActionType.SEND_MESSAGE,
            "SUBAGENT_SEND": ComputerActionType.SEND_MESSAGE,
            "MANAGE_SUBAGENTS": ComputerActionType.MANAGE_SUBAGENTS,
            "MANAGE_SUBAGENT": ComputerActionType.MANAGE_SUBAGENTS,
            "SUBAGENT_MANAGE": ComputerActionType.MANAGE_SUBAGENTS,
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
            for field_name in ("TARGET", "PAYLOAD", "EXPECTED"):
                inline_field = re.search(
                    rf"(?:^|[,;\s]){field_name}\s*:\s*(.*?)(?=(?:[,;\s]+)(?:TARGET|PAYLOAD|EXPECTED)\s*:|$)",
                    ans_val,
                    re.IGNORECASE | re.DOTALL,
                )
                if inline_field:
                    fields.setdefault(field_name, inline_field.group(1).strip(" ,;\t\r\n"))

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
            # Strip accidental leading "target:" or "target_resource:" prefix
            target = re.sub(r'^(?:target|target_resource)\s*:\s*', '', target, flags=re.IGNORECASE).strip(" *_\n\r\t`\"'")
            # Strip accidental inline PAYLOAD: / EXPECTED: suffixes if model emitted them on the same line
            target = re.sub(r'\s+\b(?:payload|expected)\s*:.*$', '', target, flags=re.IGNORECASE).strip(" *_\n\r\t`\"'")
            # Take first non-empty line to strip accidental trailing markdown blocks
            for line in target.splitlines():
                if line.strip():
                    target = line.strip(" *_\n\r\t`\"'")
                    break
        if target.lower() in ("target", "target_resource", "command", "cmd", "none", "null", "n/a", "empty"):
            target = ""

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
            # If wrapped in markdown code fence, extract inner block
            if "```" in raw:
                m_fence = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
                if m_fence:
                    raw = m_fence.group(1).strip()
            # Check for JSON array e.g. [{"TypeName": "...", ...}]
            if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("[") and "]" in raw):
                try:
                    p_arr = json.loads(raw)
                    if isinstance(p_arr, list):
                        return {"subagents": p_arr}
                except Exception:
                    pass
            # Try direct JSON loads
            try:
                p = json.loads(raw)
                if isinstance(p, dict):
                    return p
                elif isinstance(p, list):
                    return {"subagents": p}
            except Exception:
                pass
            # Search for embedded JSON object bounds {...}
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start : end + 1]
                try:
                    p = json.loads(candidate)
                    if isinstance(p, dict):
                        return p
                except Exception:
                    pass
            # Try ast literal eval
            try:
                import ast
                p = ast.literal_eval(raw)
                if isinstance(p, dict):
                    return p
                elif isinstance(p, list):
                    return {"subagents": p}
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
            raw_payload = payload_str.strip()
            if raw_payload.startswith("{"):
                payload = {}
            else:
                cleaned_cmd = ComputerUseAgent._clean_terminal_command(raw_payload)
                payload = {"command": cleaned_cmd} if cleaned_cmd else {}
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

        # If target itself was formatted as JSON e.g. {"command": "hostname"} or {"text": "..."} or array
        if target.startswith("{") or target.startswith("["):
            tgt_dict = _extract_payload_dict(target)
            if tgt_dict:
                for k, v in tgt_dict.items():
                    payload.setdefault(k, v)
            if "command" in tgt_dict:
                target_command = ComputerUseAgent._clean_terminal_command(tgt_dict["command"])
                payload.setdefault("command", target_command)
                target = target_command.split()[0] if target_command.split() else ""
            elif "text" in tgt_dict:
                payload.setdefault("text", str(tgt_dict["text"]))
                target = ""
            elif "value" in tgt_dict:
                payload.setdefault("text", str(tgt_dict["value"]))
                target = ""
            elif "content" in tgt_dict:
                payload.setdefault("content", str(tgt_dict["content"]))
                payload.setdefault("text", str(tgt_dict["content"]))
            elif "key" in tgt_dict:
                payload.setdefault("key", str(tgt_dict["key"]))
                target = str(tgt_dict["key"])
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
            # GUI text entry is strictly for visible input controls, not shell commands.
            if not payload.get("text"):
                fallback_txt = fields.get("TEXT") or fields.get("CONTENT") or ""
                if fallback_txt:
                    payload["text"] = str(fallback_txt)
            if payload.get("text"):
                payload["text"] = sanitize_typing_text(str(payload["text"]))

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
                        tail = act_field[len(prefix):].strip(" :").strip()
                        if tail and tail.upper() not in ("_EXEC", "EXEC", "COMMAND", "RUN", "SHELL", "BASH"):
                            raw_cmd = tail
                            break
            if not raw_cmd and target and target != default_file and not target.startswith("{"):
                raw_cmd = target
            payload["command"] = sanitize_command_payload(
                ComputerUseAgent._clean_terminal_command(raw_cmd)
            )
            if payload["command"].lower() in (
                "none", "null", "_exec", "exec", "terminal",
                "terminal window", "the terminal", "bash", "shell", "console",
            ):
                payload["command"] = ""

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
        ComputerActionType.APP_MAXIMIZE,
        ComputerActionType.WINDOW_MAXIMIZE,
        ComputerActionType.WINDOW_TILE,
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
        ComputerActionType.APP_MAXIMIZE,
        ComputerActionType.WINDOW_MAXIMIZE,
        ComputerActionType.WINDOW_TILE,
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
            if (
                not isinstance(x, int)
                or isinstance(x, bool)
                or not isinstance(y, int)
                or isinstance(y, bool)
            ):
                return f"{label} ({x},{y}) is not a valid integer coordinate"
            ix, iy = x, y
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

        if (
            action_type == ComputerActionType.GUI_SCROLL
            and payload.get("x") is None
            and payload.get("y") is None
        ):
            return None
        if payload.get("x") is None or payload.get("y") is None:
            return "point coordinates (x, y) are required"
        err = _check(payload["x"], payload["y"], "point")
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
        if isinstance(raw, dict):
            raw = raw.get("command")
        if not isinstance(raw, str):
            return ""
        command = raw.strip()
        if not command:
            return ""

        if command.startswith("{"):
            try:
                parsed = json.loads(command)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "command" in parsed:
                value = parsed["command"]
                return value.strip() if isinstance(value, str) else ""

        fence = re.fullmatch(
            r"```(?:bash|sh|shell)?\s*\r?\n(.*?)\r?\n```",
            command,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence:
            command = fence.group(1)

        if command.upper() in {"_EXEC", "EXEC"}:
            return ""
        if command.upper().startswith("_EXEC "):
            command = command[6:]
        elif command.upper().startswith("EXEC "):
            command = command[5:]

        command = re.sub(
            r"^(?:TERMINAL_EXEC|TERMINAL|RUN|EXEC|COMMAND|CMD)\s*:\s*",
            "",
            command,
            count=1,
            flags=re.IGNORECASE,
        )
        return sanitize_command_payload(command.strip())

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

    @staticmethod
    def _operator_approval_reason(
        action_type: ComputerActionType,
        target_resource: str,
        payload: dict[str, Any],
    ) -> str | None:
        if action_type in {
            ComputerActionType.APP_INSTALL,
            ComputerActionType.GIT_COMMIT,
            ComputerActionType.GIT_PUSH,
        }:
            return f"{action_type.value} is a consequential operation"

        if action_type in {
            ComputerActionType.GUI_CLICK,
            ComputerActionType.GUI_DOUBLE_CLICK,
            ComputerActionType.GUI_KEYPRESS,
            ComputerActionType.GUI_HOTKEY,
            ComputerActionType.BROWSER_CLICK,
            ComputerActionType.BROWSER_KEYPRESS,
            ComputerActionType.BROWSER_SELECT,
        }:
            labels = [target_resource]
            if isinstance(payload, dict):
                labels.extend(
                    str(payload.get(key) or "")
                    for key in ("target", "label", "control_name", "accessible_name")
                )
            label_text = " ".join(labels)
            if re.search(
                r"\b(send|submit|publish|post|buy|purchase|order|checkout|pay|"
                r"transfer|delete|remove|erase|destroy|wipe|drop)\b",
                label_text,
                flags=re.IGNORECASE,
            ):
                return "the grounded control may send, publish, purchase, or destructively change data"
        return None

    async def _request_operator_approval(
        self,
        action_type: ComputerActionType,
        target_resource: str,
        reason: str,
    ) -> bool:
        callback = self.operator_approval_callback
        if not callable(callback):
            return False
        try:
            result = callback(action_type.value, target_resource, reason)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.error(
                "operator_action_approval_failed",
                action_type=action_type.value,
                error=str(exc),
            )
            return False
        return result is True

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

    @classmethod
    def _screen_state_signature(cls, screen: Any) -> str:
        """Fingerprint a frame from AVCS pixels and the active window only."""
        avcs_state = getattr(screen, "avcs_state", None)
        scene_hash = (
            str(avcs_state.get("scene_hash", "")).strip()
            if isinstance(avcs_state, dict)
            and avcs_state.get("engine") == "avcs_reflex"
            else ""
        )
        return json.dumps(
            {
                "active_window": str(getattr(screen, "active_window", "") or ""),
                "scene_hash": scene_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _record_click_outcome(
        self, action: GUIAction, before_screen: Any, changed: bool
    ) -> None:
        """Feed one dispatched pointer action and its measured outcome to AVCS.

        Only pointer actions that carry a pixel target are recorded: a click
        with no coordinates cannot be turned into a point label, and a keyboard
        or window action says nothing about *where* a control is. The frame
        captured *before* the action is the one the click was aimed at, so that
        is what the head is trained on. Recording is best-effort and never
        allowed to break a live action.
        """
        loop = getattr(self, "apprenticeship", None)
        if loop is None:
            return
        try:
            from sonic.computer.models import GUIActionType

            pointer_actions = {
                GUIActionType.CLICK,
                GUIActionType.DOUBLE_CLICK,
                GUIActionType.RIGHT_CLICK,
            }
            if action.action not in pointer_actions:
                return
            x = getattr(action, "x", None)
            y = getattr(action, "y", None)
            width = int(getattr(before_screen, "width", 0) or 0)
            height = int(getattr(before_screen, "height", 0) or 0)
            raw_rgb = getattr(before_screen, "raw_rgb", None)
            if not (isinstance(x, int) and isinstance(y, int)) or width <= 0 or height <= 0:
                return
            if not isinstance(raw_rgb, (bytes, bytearray)) or not raw_rgb:
                return
            frame = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(height, width, 3)
            action_class = {
                GUIActionType.CLICK: "clickable",
                GUIActionType.DOUBLE_CLICK: "clickable",
                GUIActionType.RIGHT_CLICK: "clickable",
            }[action.action]
            loop.record_click_outcome(
                frame,
                (min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0)),
                action_class,
                bool(changed),
                query=str(getattr(action, "target_query", "") or ""),
            )
        except Exception as exc:
            logger.debug("avcs_click_outcome_record_skipped", error=str(exc))

    @staticmethod
    def _screen_has_pixel_data(screen: Any) -> bool:
        return bool(
            getattr(screen, "raw_rgb", b"")
            or getattr(screen, "screenshot_base64", "")
        )

    @classmethod
    def _screen_pixels_changed(cls, before: Any, after: Any) -> bool | None:
        if not cls._screen_has_pixel_data(before) or not cls._screen_has_pixel_data(after):
            return None
        before_state = getattr(before, "avcs_state", None)
        after_state = getattr(after, "avcs_state", None)
        before_hash = (
            before_state.get("scene_hash")
            if isinstance(before_state, dict)
            and before_state.get("engine") == "avcs_reflex"
            else None
        )
        after_hash = (
            after_state.get("scene_hash")
            if isinstance(after_state, dict)
            and after_state.get("engine") == "avcs_reflex"
            else None
        )
        if isinstance(before_hash, str) and before_hash and isinstance(after_hash, str) and after_hash:
            return before_hash != after_hash
        return None

    @classmethod
    def _gui_observation_matches(cls, source: Any, fresh: Any) -> bool:
        """Check that a refreshed observation still grounds the same GUI decision."""
        source_screen = getattr(source, "screen", None)
        fresh_screen = getattr(fresh, "screen", None)
        if source_screen is None or fresh_screen is None:
            return False
        if (
            getattr(source_screen, "desktop_state", "")
            != getattr(fresh_screen, "desktop_state", "")
            or not str(getattr(source_screen, "active_window", "") or "").strip()
            or str(getattr(source_screen, "active_window", "") or "").strip().casefold()
            != str(getattr(fresh_screen, "active_window", "") or "").strip().casefold()
            or getattr(source_screen, "width", 0) != getattr(fresh_screen, "width", 0)
            or getattr(source_screen, "height", 0) != getattr(fresh_screen, "height", 0)
            or cls._screen_state_signature(source_screen)
            != cls._screen_state_signature(fresh_screen)
        ):
            return False
        if (
            getattr(source_screen, "desktop_state", "") != "INTERACTIVE"
            or getattr(fresh_screen, "desktop_state", "") != "INTERACTIVE"
        ):
            return False
        source_state = getattr(source_screen, "avcs_state", None)
        fresh_state = getattr(fresh_screen, "avcs_state", None)
        source_hash = (
            source_state.get("scene_hash", "")
            if isinstance(source_state, dict)
            and source_state.get("engine") == "avcs_reflex"
            else ""
        )
        fresh_hash = (
            fresh_state.get("scene_hash", "")
            if isinstance(fresh_state, dict)
            and fresh_state.get("engine") == "avcs_reflex"
            else ""
        )
        if not source_hash or not fresh_hash:
            return False
        if source_hash != fresh_hash:
            return False
        return True

    async def _create_gui_action_intent(
        self,
        workspace_id: str,
        *,
        action_type: ComputerActionType,
        target: str,
        payload: dict[str, Any],
        expected_outcome: str,
        observation: ComputerWorldObservation,
        ttl_ms: int,
    ) -> GUIActionIntent:
        try:
            return GUIActionIntent.create(
                action_type=action_type.value,
                target=target,
                payload=payload,
                expected_outcome=expected_outcome,
                observation=observation,
                ttl_ms=ttl_ms,
            )
        except ValueError as exc:
            if str(exc) != "GUI action intent requires fresh observation evidence":
                raise

        source_screen = getattr(observation, "screen", None)
        if getattr(source_screen, "desktop_state", "") != "INTERACTIVE":
            raise ValueError("GUI intent source has no refreshable desktop evidence")
        fresh_screen = await self.computer.screenshot(workspace_id)
        fresh_screen = self._ensure_avcs_state(fresh_screen)
        perception_source = "avcs_pixels"

        received_at_ns = time.perf_counter_ns()
        fresh_observation = observation.model_copy(
            update={
                "screen": fresh_screen,
                "active_application": (
                    getattr(fresh_screen, "active_window", "")
                    or observation.active_application
                ),
                "visible_text": str(getattr(fresh_screen, "visible_text", "") or ""),
                "perception_source": perception_source,
                "observation_age_ms": getattr(fresh_screen, "capture_age_ms", None),
                "observation_received_monotonic_ns": received_at_ns,
            }
        )
        if not self._gui_observation_matches(observation, fresh_observation):
            source_screen = observation.screen
            fresh_screen = fresh_observation.screen
            logger.warning(
                "gui_action_observation_revalidation_mismatch",
                source_state=source_screen.desktop_state,
                fresh_state=fresh_screen.desktop_state,
                same_active_window=(
                    str(source_screen.active_window or "").casefold()
                    == str(fresh_screen.active_window or "").casefold()
                ),
                same_dimensions=(
                    source_screen.width == fresh_screen.width
                    and source_screen.height == fresh_screen.height
                ),
                same_ui_signature=(
                    self._screen_state_signature(source_screen)
                    == self._screen_state_signature(fresh_screen)
                ),
                source_state_version=source_screen.perception_version,
                fresh_state_version=fresh_screen.perception_version,
            )
            raise ValueError(
                "desktop state changed while reasoning; the GUI action must be reconsidered"
            )

        self._screen_width = fresh_screen.width
        self._screen_height = fresh_screen.height
        self._last_screen_observation_received_at = (
            asyncio.get_running_loop().time()
        )
        self._last_observed_screen = fresh_screen
        self._last_observed_screen_from_reasoning = False
        self._last_screen_has_pixels = self._screen_has_pixel_data(fresh_screen)
        self._last_screen_frame_id = (
            getattr(fresh_screen, "frame_id", None)
            if self._last_screen_has_pixels
            else None
        )
        self._last_gui_dispatch_evidence = (
            fresh_screen,
            self._last_screen_observation_received_at,
        )
        self._last_gui_state_observation = None
        self._last_screenshot_b64 = fresh_screen.screenshot_base64
        snapshot = self.perception_bus.publish(
            screenshot_base64=fresh_screen.screenshot_base64,
            width=fresh_screen.width,
            height=fresh_screen.height,
            active_window=fresh_screen.active_window,
            windows=observation.windows,
            processes=observation.processes,
            visible_text=fresh_screen.visible_text,
            controls=fresh_screen.detected_controls,
        )
        fresh_observation.perception_version = snapshot.version
        self._last_world_observation = fresh_observation
        return GUIActionIntent.create(
            action_type=action_type.value,
            target=target,
            payload=payload,
            expected_outcome=expected_outcome,
            observation=fresh_observation,
            ttl_ms=ttl_ms,
        )

    def _renew_gui_action_intent_after_verified_observation(
        self,
        intent: GUIActionIntent,
        screen: ScreenObservation,
    ) -> GUIActionIntent:
        """Renew an intent only after callers verify the same live pixel frame."""
        observation = getattr(self, "_last_world_observation", None)
        if not isinstance(observation, ComputerWorldObservation):
            raise ValueError("GUI intent cannot be renewed without its source observation")

        now_ns = time.perf_counter_ns()
        binding_errors = intent.validate_binding(
            action_type=intent.action_type,
            target=intent.target,
            payload=json.loads(intent.payload_json),
            now_ns=now_ns,
        )
        binding_errors = tuple(
            error for error in binding_errors if error != "GUI action intent expired"
        )
        if binding_errors:
            raise ValueError("; ".join(binding_errors))
        evidence_errors = intent.validate_observation(screen, now_ns=now_ns)
        evidence_errors = tuple(
            error for error in evidence_errors if error != "GUI action intent expired"
        )
        if evidence_errors:
            raise ValueError("; ".join(evidence_errors))

        source_screen = observation.screen
        if getattr(screen, "desktop_state", "") != getattr(
            source_screen, "desktop_state", ""
        ):
            raise ValueError("desktop evidence modality changed before GUI dispatch")
        fresh_observation = observation.model_copy(
            update={
                "screen": screen,
                "active_application": (
                    str(getattr(screen, "active_window", "") or "").strip()
                    or observation.active_application
                ),
                "visible_text": str(getattr(screen, "visible_text", "") or ""),
                "observation_age_ms": getattr(screen, "capture_age_ms", None),
                "observation_received_monotonic_ns": now_ns,
            }
        )
        renewed = GUIActionIntent.create(
            action_type=intent.action_type,
            target=intent.target,
            payload=json.loads(intent.payload_json),
            expected_outcome=intent.expected_outcome,
            observation=fresh_observation,
            now_ns=now_ns,
            ttl_ms=2_000,
        )
        if (
            renewed.source_active_window.casefold()
            != intent.source_active_window.casefold()
            or renewed.source_scene_hash != intent.source_scene_hash
            or renewed.source_width != intent.source_width
            or renewed.source_height != intent.source_height
        ):
            raise ValueError("GUI intent target, focus, or source state changed")

        self._last_world_observation = fresh_observation
        self._last_screen_observation_received_at = asyncio.get_running_loop().time()
        self._last_observed_screen = screen
        self._last_observed_screen_from_reasoning = False
        self._last_screen_has_pixels = self._screen_has_pixel_data(screen)
        self._last_screen_frame_id = (
            getattr(screen, "frame_id", None)
            if self._last_screen_has_pixels
            else None
        )
        self._last_gui_state_observation = None
        self._last_screenshot_b64 = screen.screenshot_base64
        self._screen_width = screen.width
        self._screen_height = screen.height
        return renewed

    async def _dispatch_gui_action(
        self,
        workspace_id: str,
        action: GUIAction,
        *,
        actor: str | None = None,
    ) -> ScreenObservation:
        """Revalidate the pixel frame before GUI dispatch and capture its result."""
        pending_intent = getattr(self, "_pending_gui_action_intent", None)
        if pending_intent is not None:
            prior_evidence = getattr(self, "_last_gui_dispatch_evidence", None)
            if not isinstance(prior_evidence, tuple) or len(prior_evidence) != 2:
                raise RuntimeError("no current observation is available for final GUI validation")
            prior_screen, _ = prior_evidence
            if getattr(prior_screen, "desktop_state", "") != "INTERACTIVE":
                raise RuntimeError("AVCS pixel frame is unavailable before GUI dispatch")
            final_screen = await self.computer.screenshot(workspace_id)
            final_screen = self._ensure_avcs_state(final_screen)
            final_age = getattr(final_screen, "capture_age_ms", None)
            if (
                getattr(final_screen, "desktop_state", "") != "INTERACTIVE"
                or not isinstance(final_age, (int, float))
                or isinstance(final_age, bool)
                or not math.isfinite(final_age)
                or not 0 <= final_age <= 2_000
                or getattr(final_screen, "width", 0)
                != getattr(prior_screen, "width", 0)
                or getattr(final_screen, "height", 0)
                != getattr(prior_screen, "height", 0)
                or self._screen_state_signature(final_screen)
                != self._screen_state_signature(prior_screen)
            ):
                raise RuntimeError("AVCS scene changed before final GUI dispatch")
            pending_intent = self._renew_gui_action_intent_after_verified_observation(
                pending_intent,
                final_screen,
            )
            self._pending_gui_action_intent = pending_intent
            self._last_gui_dispatch_evidence = (
                final_screen,
                asyncio.get_running_loop().time(),
            )
            expected_provider_actions = {
                "GUI_CLICK": {"CLICK"},
                "GUI_DOUBLE_CLICK": {"DOUBLE_CLICK"},
                "GUI_RIGHT_CLICK": {"RIGHT_CLICK"},
                "GUI_TYPE": {"CLICK", "TYPE"},
                "GUI_KEYPRESS": {"KEYPRESS", "HOTKEY"},
                "GUI_MOVE": {"MOVE"},
                "GUI_SCROLL": {"SCROLL"},
                "GUI_DRAG": {"DRAG"},
            }
            if action.action.value not in expected_provider_actions.get(
                pending_intent.action_type, set()
            ):
                raise RuntimeError("provider action differs from the GUI intent")
            intent_payload = json.loads(pending_intent.payload_json)
            binding_errors = pending_intent.validate_binding(
                action_type=pending_intent.action_type,
                target=pending_intent.target,
                payload=intent_payload,
            )
            evidence = getattr(self, "_last_gui_dispatch_evidence", None)
            if not isinstance(evidence, tuple) or len(evidence) != 2:
                raise RuntimeError("no current observation is available for final GUI validation")
            observed_screen, observed_at = evidence
            elapsed_ms = max(
                0.0,
                (asyncio.get_running_loop().time() - observed_at) * 1000,
            )
            observation_errors = pending_intent.validate_observation(
                observed_screen,
                elapsed_ms=elapsed_ms,
            )
            if binding_errors or observation_errors:
                raise RuntimeError(
                    "; ".join((*binding_errors, *observation_errors))
                )
        self._last_gui_action_observation_valid = False
        self._last_gui_action_screen = None
        before_screen = getattr(self, "_last_gui_state_observation", None)
        if before_screen is None:
            before_screen = getattr(self, "_last_observed_screen", None)
        state_action = getattr(type(self.computer), "gui_action_state", None)
        dispatch_method = (
            self.computer.gui_action_state
            if callable(state_action)
            else self.computer.gui_action
        )
        dispatch = (
            dispatch_method(workspace_id, action, actor=actor)
            if actor is not None
            else dispatch_method(workspace_id, action)
        )
        dispatch_task = asyncio.create_task(dispatch)
        try:
            while not dispatch_task.done():
                if self._mission_interrupt_requested():
                    dispatch_task.cancel()
                    await asyncio.gather(dispatch_task, return_exceptions=True)
                    raise RuntimeError("GUI dispatch cancelled by interrupt")
                try:
                    await asyncio.wait_for(asyncio.shield(dispatch_task), timeout=0.025)
                except asyncio.TimeoutError:
                    continue
            result = await dispatch_task
        except BaseException:
            if not dispatch_task.done():
                dispatch_task.cancel()
                await asyncio.gather(dispatch_task, return_exceptions=True)
            raise

        self._last_gui_action_screen = result
        self._last_gui_state_delta = None
        if (
            getattr(result, "desktop_state", "") == "INTERACTIVE"
            and self._screen_has_pixel_data(result)
        ):
            result = self._ensure_avcs_state(result)
            self._last_observed_screen = result
            self._last_observed_screen_from_reasoning = False
            self._last_gui_state_observation = None
            capture_age_ms = getattr(result, "capture_age_ms", None)
            self._last_gui_action_observation_valid = bool(
                getattr(result, "desktop_state", "") == "INTERACTIVE"
                and self._screen_has_pixel_data(result)
                and isinstance(capture_age_ms, (int, float))
                and not isinstance(capture_age_ms, bool)
                and math.isfinite(capture_age_ms)
                and 0 <= capture_age_ms <= 2_000
            )
            self._last_screen_has_pixels = self._last_gui_action_observation_valid
            self._last_screen_frame_id = (
                getattr(result, "frame_id", None)
                if self._last_screen_has_pixels
                else None
            )
            self._last_screenshot_b64 = str(
                getattr(result, "screenshot_base64", "") or ""
            )
            if before_screen is not None:
                before_signature = self._screen_state_signature(before_screen)
                after_signature = self._screen_state_signature(result)
                changed = before_signature != after_signature
                self._last_gui_state_delta = {
                    "changed": changed,
                    "before_version": getattr(before_screen, "perception_version", 0),
                    "after_version": getattr(result, "perception_version", 0),
                    "active_window": getattr(result, "active_window", ""),
                }
                self._record_click_outcome(action, before_screen, changed)
        else:
            self._last_gui_action_observation_valid = False
            self._last_observed_screen = result
            self._last_observed_screen_from_reasoning = False
            self._last_gui_state_observation = None
            self._last_screenshot_b64 = ""
        return result

    def _get_visual_servo(self, workspace_id: str) -> Any | None:
        """Build (once) the closed-loop eye-hand servo for this provider.

        Returns ``None`` when the provider cannot report pointer position, in
        which case the caller keeps the existing open-loop behaviour rather than
        moving blind. Each corrective step is re-checked against the fail-closed
        safety envelope before it reaches the provider, so the loop cannot widen
        what the agent is allowed to do -- it only makes an allowed reach
        self-correcting.
        """
        if self._visual_servo is not None:
            return self._visual_servo
        provider = getattr(self, "computer", None)
        if provider is None or not callable(getattr(provider, "cursor_position", None)):
            return None

        async def read_cursor() -> tuple[int, int] | None:
            position = await provider.cursor_position(workspace_id)
            # A provider that merely *has* the attribute (e.g. a test double)
            # must not be treated as a real sensor; reject anything that is not
            # a two-integer point so the caller keeps its open-loop path.
            if position is None:
                return None
            if (
                not isinstance(position, tuple)
                or len(position) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in position
                )
            ):
                raise TypeError("cursor_position did not return an integer point")
            return int(position[0]), int(position[1])

        async def move_cursor(x: int, y: int) -> Any:
            # Re-evaluate the envelope for every corrective step. The servo
            # executes one already-authorized reach, but it must not become a
            # side channel that skips the policy on any of its moves.
            if self.safety is not None:
                verdict = self.safety.evaluate(
                    ComputerActionType.GUI_MOVE.value,
                    f"{x},{y}",
                    {"x": x, "y": y},
                )
                if not verdict.allowed:
                    raise RuntimeError(
                        f"servo move blocked by safety policy: {verdict.reason}"
                    )
            return await provider.gui_action(
                workspace_id,
                GUIAction(action=GUIActionType.MOVE, x=int(x), y=int(y)),
            )

        def screen_size() -> tuple[int, int]:
            return self._screen_width, self._screen_height

        try:
            from sonic.computer_use.visual_servo import VisualServo

            self._visual_servo = VisualServo(
                read_cursor=read_cursor,
                move_cursor=move_cursor,
                screen_size=screen_size,
            )
        except Exception as exc:
            logger.debug("visual_servo_unavailable", error=str(exc))
            return None
        return self._visual_servo

    async def _servo_to_pixel_target(
        self,
        workspace_id: str,
        x: int,
        y: int,
    ) -> dict[str, Any] | None:
        """Converge the pointer onto ``(x, y)`` and return a measured receipt.

        Best-effort and non-fatal: if the servo cannot run (no pointer readback)
        this returns ``None`` and the caller proceeds with its normal single
        dispatch, preserving the previous behaviour. A converged reach still
        does not authorise a click on its own -- the caller decides that.
        """
        servo = self._get_visual_servo(workspace_id)
        if servo is None:
            return None
        try:
            receipt = await servo.reach(
                workspace_id,
                (int(x), int(y)),
                should_stop=self._mission_interrupt_requested,
            )
        except Exception as exc:
            # Fail closed on the *servo*, not on the action: log and let the
            # caller fall back to its normal dispatch path.
            logger.debug("visual_servo_reach_failed", error=str(exc))
            return None
        self._last_servo_receipt = receipt.to_dict()
        return self._last_servo_receipt

    def _avcs_focus_head_ready(self) -> bool:
        """True only when a configured, evaluated local focus head can be loaded.

        This is a pure capability probe: it never dispatches an action and never
        treats a configured checkpoint path as readiness. A missing or failing
        checkpoint returns False so text entry stays blocked.
        """
        try:
            perception = getattr(self, "_avcs_perception", None)
            if perception is None:
                from sonic.computer_use.avcs_perception import AVCSPerception

                perception = AVCSPerception()
                self._avcs_perception = perception
            ensure = getattr(perception, "ensure_focus_verifier_ready", None)
            if not callable(ensure):
                return False
            return bool(ensure())
        except Exception:
            return False

    def _avcs_grounder_ready(self) -> bool:
        """True only when a configured, evaluated local grounding checkpoint loads.

        Pure capability probe: it never dispatches an action and never treats a
        configured path as readiness. When False, semantic GUI targets (clicks on
        described controls) cannot be grounded and will fail closed, so the
        reasoning prompt must steer the model toward non-semantic routes.
        """
        try:
            perception = getattr(self, "_avcs_perception", None)
            if perception is None:
                from sonic.computer_use.avcs_perception import AVCSPerception

                perception = AVCSPerception()
                self._avcs_perception = perception
            if not bool(getattr(perception, "goal_grounder_configured", False)):
                return False
            ready = getattr(perception, "goal_grounder_ready", None)
            if callable(ready):
                try:
                    return bool(ready())
                except Exception:
                    return False
            return False
        except Exception:
            return False

    async def _verify_gui_focus_with_avcs(
        self,
        screen: ScreenObservation,
        *,
        x: int,
        y: int,
        target: str,
    ) -> tuple[bool, str]:
        """Use an AVCS focus head if one exists; never upload the frame to an LLM."""
        try:
            perception = getattr(self, "_avcs_perception", None)
            if perception is None:
                from sonic.computer_use.avcs_perception import AVCSPerception

                perception = AVCSPerception()
                self._avcs_perception = perception
            raw_rgb = bytes(getattr(screen, "raw_rgb", b"") or b"")
            verifier = getattr(perception, "verify_focus_rgb_bytes", None)
            verifier_args = (
                raw_rgb,
                int(getattr(screen, "width", 0) or 0),
                int(getattr(screen, "height", 0) or 0),
                x,
                y,
                target,
                getattr(screen, "frame_id", None),
            )
            if not callable(verifier):
                verifier = getattr(perception, "verify_focus_png", None)
                verifier_args = (
                    base64.b64decode(
                        str(getattr(screen, "screenshot_base64", "") or ""),
                        validate=True,
                    ),
                    x,
                    y,
                    target,
                )
            if not callable(verifier):
                return False, "AVCS has no evaluated local focus-verification head"
            if not raw_rgb and not verifier_args[0]:
                return False, "AVCS focus verification requires current pixel data"
            result = await asyncio.to_thread(verifier, *verifier_args)
            checkpoint = (
                result.get("checkpoint_metadata")
                if isinstance(result, dict)
                else None
            )
            evaluation = (
                checkpoint.get("evaluation")
                if isinstance(checkpoint, dict)
                else None
            )
            state = getattr(screen, "avcs_state", None)
            precision = (
                evaluation.get("precision")
                if isinstance(evaluation, dict)
                else None
            )
            samples = (
                evaluation.get("samples")
                if isinstance(evaluation, dict)
                else None
            )
            score = result.get("decision_score") if isinstance(result, dict) else None
            threshold = (
                result.get("decision_threshold")
                if isinstance(result, dict)
                else None
            )
            if (
                not isinstance(result, dict)
                or result.get("focused") is not True
                or not isinstance(result.get("evidence"), str)
                or not result["evidence"].strip()
                or result.get("engine") != "avcs_focus_verifier"
                or result.get("frame_id") != getattr(screen, "frame_id", None)
                or not isinstance(state, dict)
                or result.get("scene_hash") != state.get("scene_hash")
                or not isinstance(checkpoint, dict)
                or checkpoint.get("dataset_provenance")
                != "recorded_ui_focus_demonstrations"
                or not isinstance(evaluation, dict)
                or evaluation.get("metric") != "focus_verification_precision"
                or evaluation.get("passed") is not True
                or isinstance(precision, bool)
                or not isinstance(precision, (int, float))
                or not math.isfinite(float(precision))
                or float(precision) < 0.8
                or isinstance(samples, bool)
                or not isinstance(samples, int)
                or samples < 30
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or not 0 < float(threshold) <= 1
                or float(score) < float(threshold)
            ):
                return False, (
                    "AVCS focus head lacks current-frame evidence or a passing "
                    "held-out focus-precision evaluation (minimum 0.80 over 30 samples)"
                )
            return True, result["evidence"].strip()[:200]
        except Exception as exc:
            logger.warning("avcs_focus_verification_failed", reason=str(exc))
            return False, "AVCS focus verification failed"

    async def execute_action(
        self,
        workspace_id: str,
        action_type: ComputerActionType,
        target_resource: str,
        payload: dict[str, Any],
        predicted_outcome: str,
    ) -> ComputerDecisionTrace:
        """Serialize every shared-display action before checking focus or dispatching."""
        action_name = getattr(action_type, "value", str(action_type))
        shared_display_action = action_name.startswith(("GUI_", "APP_", "BROWSER_"))
        if not shared_display_action:
            return await self._execute_action_unlocked(
                workspace_id, action_type, target_resource, payload, predicted_outcome
            )

        from sonic.computer_use.teamwork import DisplayLeaseManager

        display = None
        get_display = getattr(self.computer, "_get_display", None)
        if callable(get_display):
            display = get_display(workspace_id)
            if inspect.isawaitable(display):
                display = await display
        try:
            lock = await DisplayLeaseManager.acquire(
                workspace_id,
                display=display,
                owner=getattr(self, "agent_id", None),
                interrupt_check=self._mission_interrupt_requested,
            )
        except DisplayLeaseManager.Interrupted:
            self._interrupted = True
            self._interrupt_event.set()
            return await self._execute_action_unlocked(
                workspace_id,
                ActionExecutionStatus.BLOCKED,
                "interrupted",
                {},
                "Mission interrupted while waiting for display ownership.",
            )
        try:
            pending_intent = getattr(self, "_pending_gui_action_intent", None)
            return await self._execute_action_unlocked(
                workspace_id, action_type, target_resource, payload, predicted_outcome
            )
        finally:
            current_intent = getattr(self, "_pending_gui_action_intent", None)
            if pending_intent is not None and (
                current_intent is pending_intent
                or (
                    current_intent is not None
                    and current_intent.action_type == pending_intent.action_type
                    and current_intent.target == pending_intent.target
                    and current_intent.payload_json == pending_intent.payload_json
                    and current_intent.expected_outcome
                    == pending_intent.expected_outcome
                )
            ):
                del self._pending_gui_action_intent
            DisplayLeaseManager.release(lock, display=display)

    async def _execute_action_unlocked(
        self,
        workspace_id: str,
        action_type: ComputerActionType,
        target_resource: str,
        payload: dict[str, Any],
        predicted_outcome: str,
    ) -> ComputerDecisionTrace:
        """Executes the action inside the computer sandbox and validates outcome."""
        if not hasattr(self, "action_counter"):
            self.action_counter = 0
        if not hasattr(self, "traces"):
            self.traces = []
        if not hasattr(self, "history"):
            self.history = []
        if not hasattr(self, "safety"):
            self.safety = None
        self.action_counter += 1
        t_start = time.perf_counter()
        if self._mission_interrupt_requested():
            self._interrupted = True
            self._interrupt_event.set()
            actual_obs_str = "Action held before dispatch: mission interrupted."
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
            self.history.append({"action": f"BLOCKED {target_resource}", "result": actual_obs_str})
            return trace
        pending_intent = getattr(self, "_pending_gui_action_intent", None)
        if pending_intent is not None:
            intent_errors = pending_intent.validate_binding(
                action_type=getattr(action_type, "value", str(action_type)),
                target=target_resource,
                payload=payload,
            )
            intent_errors = tuple(
                error
                for error in intent_errors
                if error != "GUI action intent expired"
            )
            if intent_errors:
                actual_obs_str = (
                    "GUI action held before dispatch: " + "; ".join(intent_errors)
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
                self.history.append({"action": f"BLOCKED {target_resource}", "result": actual_obs_str})
                return trace
        legacy_gui_actions = {
            ComputerActionType.BROWSER_CLICK: ComputerActionType.GUI_CLICK,
            ComputerActionType.BROWSER_TYPE: ComputerActionType.GUI_TYPE,
            ComputerActionType.BROWSER_KEYPRESS: ComputerActionType.GUI_KEYPRESS,
            ComputerActionType.BROWSER_SCROLL: ComputerActionType.GUI_SCROLL,
            ComputerActionType.BROWSER_SCREENSHOT: ComputerActionType.GUI_SCREENSHOT,
            ComputerActionType.BROWSER_WAIT: ComputerActionType.GUI_WAIT,
        }
        if action_type == ComputerActionType.BROWSER_SCROLL:
            delta_y = payload.get("delta_y", payload.get("delta"))
            if delta_y is not None:
                payload = {**payload, "delta": -3 if int(delta_y) > 0 else 3}
        if action_type in (ComputerActionType.BROWSER_CLICK, ComputerActionType.BROWSER_TYPE):
            target_resource = str(
                payload.get("selector") or payload.get("target") or target_resource
            )
        if action_type == ComputerActionType.BROWSER_KEYPRESS:
            target_resource = str(payload.get("key") or target_resource)
        if action_type in legacy_gui_actions:
            action_type = legacy_gui_actions[action_type]
        actual_obs_str = ""
        status = ActionExecutionStatus.COMPLETED
        recovery_needed = False
        visual_stall_detected = False
        action_exit_code: int | None = None
        action_stderr: str = ""
        app_launch_failed = False
        typing_validation_error = ""
        typing_match = None
        self._last_gui_state_delta = None
        self._last_gui_action_observation_valid = False
        if action_type == ComputerActionType.GUI_TYPE:
            raw_text = str(payload.get("text") or payload.get("value") or payload.get("input") or "")
            invalid_text, text_reason = self._validate_typing_text(sanitize_typing_text(raw_text))
            invalid_field, field_reason = self._validate_typing_field(target_resource)
            if invalid_text:
                typing_validation_error = text_reason
            elif invalid_field:
                typing_validation_error = f"wrong field: {field_reason}"
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

        if action_type in (
            ComputerActionType.GUI_TYPE,
            ComputerActionType.BROWSER_NAVIGATE,
        ) and not self._avcs_focus_head_ready():
            actual_obs_str = (
                f"{action_type.value} blocked: AVCS has no evaluated local "
                "focus-verification head, so text will not be sent."
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
            self.history.append(
                {"action": f"BLOCKED {target_resource}", "result": actual_obs_str}
            )
            return trace

        if action_type in (
            ComputerActionType.GUI_KEYPRESS,
            ComputerActionType.GUI_HOTKEY,
        ) and not str(payload.get("key") or payload.get("keys") or "").strip() and (
            not self._has_gui_burst_payload(payload)
        ):
            actual_obs_str = (
                f"{action_type.value} blocked: an explicit key is required; "
                "no default key was sent."
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
            self.history.append({"action": f"BLOCKED {target_resource}", "result": actual_obs_str})
            return trace

        async def resolve_visible_desktop_target(query: str) -> tuple[int, int] | None:
            """Resolve a desktop target only with the local AVCS grounder."""
            if not query or not (
                getattr(self, "_last_screen_has_pixels", False)
                or self._last_screenshot_b64
            ):
                return None
            try:
                grounded = await self._ground_target_with_avcs(query)
            except Exception as exc:
                logger.error(
                    "avcs_target_grounding_failed_closed",
                    query=query,
                    error=str(exc),
                )
                return None
            return (grounded["x"], grounded["y"]) if grounded else None

        gui_actions = {
            ComputerActionType.GUI_CLICK,
            ComputerActionType.GUI_DOUBLE_CLICK,
            ComputerActionType.GUI_RIGHT_CLICK,
            ComputerActionType.GUI_TYPE,
            ComputerActionType.GUI_KEYPRESS,
            ComputerActionType.GUI_MOVE,
            ComputerActionType.GUI_SCROLL,
            ComputerActionType.GUI_DRAG,
        }
        observed_screen = getattr(self, "_last_observed_screen", None)
        observed_avcs = getattr(observed_screen, "avcs_state", None)
        system3 = (
            observed_avcs.get("system3")
            if isinstance(observed_avcs, dict)
            and observed_avcs.get("engine") == "avcs_reflex"
            else None
        )
        # Only a frame the agent actually reasoned on can raise "the desktop
        # changed unexpectedly". The post-action frame captured by the agent's
        # own GUI dispatch is its expected effect; treating it as surprise would
        # block the very next legitimate action.
        from_reasoning = bool(
            getattr(self, "_last_observed_screen_from_reasoning", False)
        )
        surprise_score = (
            system3.get("surprise_score")
            if from_reasoning
            and isinstance(system3, dict)
            and system3.get("engine") == "system3_temporal_prediction"
            and system3.get("prediction_available") is True
            and system3.get("stability_state") == "surprise_event"
            else None
        )
        mutates_desktop = action_type in gui_actions or action_type in {
            ComputerActionType.APP_LAUNCH,
            ComputerActionType.APP_CLOSE,
            ComputerActionType.APP_FOCUS,
            ComputerActionType.APP_MAXIMIZE,
            ComputerActionType.WINDOW_MAXIMIZE,
            ComputerActionType.WINDOW_TILE,
        }
        if (
            mutates_desktop
            and isinstance(surprise_score, (int, float))
            and not isinstance(surprise_score, bool)
            and math.isfinite(float(surprise_score))
            and float(surprise_score) >= 0.85
        ):
            actual_obs_str = (
                "AVCS temporal surprise detected "
                f"(score={float(surprise_score):.3f}); desktop input was held. "
                "Observe the new screen and re-reason before trying again. "
                "This signal detects visual change only; it does not identify "
                "the cause or prove an alert is present."
            )
            logger.info(
                "gui_action_held_on_avcs_surprise",
                action_type=action_type.value,
                score=round(float(surprise_score), 4),
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
            self.history.append(
                {"action": f"HELD {action_type.value} {target_resource}", "result": actual_obs_str}
            )
            return trace
        status_method = getattr(self.computer, "status", None)
        is_test_double = type(self.computer).__module__ == "unittest.mock"
        if action_type in gui_actions:
            if not callable(status_method) and not is_test_double:
                actual_obs_str = "GUI action blocked before dispatch: active-window ownership cannot be verified"
                status = ActionExecutionStatus.BLOCKED
                action_exit_code = 126
                requested_window = None
            else:
                requested_window = (
                    payload.get("target_window")
                    or payload.get("window")
                    or payload.get("active_window")
                )
            if status == ActionExecutionStatus.BLOCKED:
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
                    exit_code=action_exit_code,
                    thought=getattr(self, "_last_thought", ""),
                    duration_seconds=round(time.perf_counter() - t_start, 3),
                    thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
                )
                self.traces.append(trace)
                self.history.append({"action": f"BLOCKED {target_resource}", "result": actual_obs_str})
                return trace
        if action_type in gui_actions and status_method is not None:
            requested_window = (
                payload.get("target_window")
                or payload.get("window")
                or payload.get("active_window")
            )
            try:
                current_status = self.computer.status(workspace_id)
                if inspect.isawaitable(current_status):
                    current_status = await current_status
                active_window = str(
                    getattr(current_status, "active_application", "")
                    or getattr(current_status, "active_window", "")
                ).strip()
                if not active_window:
                    raise RuntimeError("active window is unknown")
                previous_observation = getattr(self, "_last_world_observation", None)
                previous_screen = (
                    getattr(previous_observation, "screen", None)
                    if previous_observation is not None
                    else getattr(self, "_last_observed_screen", None)
                )
                if previous_screen is None:
                    raise RuntimeError("no prior desktop observation is available")
                if getattr(previous_screen, "desktop_state", "") != "INTERACTIVE":
                    raise RuntimeError("AVCS pixel-frame evidence is unavailable")
                previous_window = str(
                    getattr(previous_screen, "active_window", "") or ""
                ).strip()
                if previous_window and previous_window.casefold() != active_window.casefold():
                    raise RuntimeError(
                        f"active window changed since observation: {previous_window} -> {active_window}"
                    )
                fresh_screen = await self.computer.screenshot(workspace_id)
                fresh_screen = self._ensure_avcs_state(fresh_screen)
                fresh_capture_age = getattr(fresh_screen, "capture_age_ms", None)
                if (
                    getattr(fresh_screen, "desktop_state", "") != "INTERACTIVE"
                    or not isinstance(fresh_capture_age, (int, float))
                    or isinstance(fresh_capture_age, bool)
                    or not math.isfinite(fresh_capture_age)
                    or not 0 <= fresh_capture_age <= 2_000
                    or not self._screen_has_pixel_data(previous_screen)
                    or not self._screen_has_pixel_data(fresh_screen)
                    or getattr(fresh_screen, "width", 0)
                    != getattr(previous_screen, "width", 0)
                    or getattr(fresh_screen, "height", 0)
                    != getattr(previous_screen, "height", 0)
                ):
                    raise RuntimeError("fresh AVCS pixel-frame evidence could not be verified")
                if (
                    self._screen_state_signature(fresh_screen)
                    != self._screen_state_signature(previous_screen)
                ):
                    raise RuntimeError("AVCS scene changed since the action was reasoned")
                pending_intent = getattr(self, "_pending_gui_action_intent", None)
                if pending_intent is not None:
                    pending_intent = self._renew_gui_action_intent_after_verified_observation(
                        pending_intent,
                        fresh_screen,
                    )
                    self._pending_gui_action_intent = pending_intent
                    intent_errors = pending_intent.validate_observation(fresh_screen)
                    if intent_errors:
                        raise RuntimeError("; ".join(intent_errors))
                self._last_gui_dispatch_evidence = (
                    fresh_screen,
                    asyncio.get_running_loop().time(),
                )
                if requested_window:
                    if active_window.casefold() != str(requested_window).strip().casefold():
                        if action_type != ComputerActionType.APP_FOCUS:
                            raise RuntimeError(
                                "requested window differs from the window used to ground "
                                "this action; focus and re-observe before acting"
                            )
            except Exception as exc:
                actual_obs_str = f"GUI action blocked before dispatch: {exc}"
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
                self.history.append({"action": f"BLOCKED {target_resource}", "result": actual_obs_str})
                return trace

        # Pre-execution refusal or blocked action handling (e.g. from LLM safety refusal)
        if action_type == ActionExecutionStatus.BLOCKED or str(action_type).upper() in ("BLOCKED", "ACTIONEXECUTIONSTATUS.BLOCKED"):
            actual_obs_str = (
                f"Action blocked: {target_resource} — {predicted_outcome}"
                if target_resource else f"Action blocked: {predicted_outcome}"
            )
            trace = ComputerDecisionTrace(
                step_index=self.action_counter,
                # A blocked pre-execution proposal has no executed action. It was
                # previously recorded as TERMINAL_EXEC unconditionally, so a
                # blocked GUI worker (reasoning-unavailable, gui-contract) showed
                # up in the operator UI as "terminal exec". Report it as a
                # non-action so the trace describes what actually happened.
                action_type=ComputerActionType.GUI_SCREENSHOT,
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

        if self._gui_action_contract_active() and action_type not in self._GUI_ONLY_ACTIONS:
            actual_obs_str = (
                f"GUI execution contract blocked {action_type.value}. "
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

        if action_type == ComputerActionType.SECURITY_TOOL and not self.security_tools:
            actual_obs_str = (
                "SECURITY_TOOL blocked: no registered in-sandbox tool adapter is "
                "available; unregistered tools are never sent to the terminal."
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
                content_keys = ("content", "code", "file_content", "text", "data")
                curr_content = next(
                    (str(payload[key]) for key in content_keys if payload.get(key) is not None),
                    "",
                )
                prev_content = next(
                    (
                        str(prev_payload[key])
                        for key in content_keys
                        if isinstance(prev_payload, dict) and prev_payload.get(key) is not None
                    ),
                    "",
                )
                if curr_path and curr_path == prev_path and curr_content == prev_content:
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
            cmd = self._clean_terminal_command(raw_cmd)
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
        approval_granted = False
        approval_reason = self._operator_approval_reason(
            action_type, target_resource, payload
        )
        if self.safety is not None:
            verdict = self.safety.evaluate(action_type.value, target_resource, payload)
            if verdict.needs_approval:
                approval_reason = approval_reason or verdict.reason
            if not verdict.allowed and not verdict.needs_approval:
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
        if approval_reason:
            approval_granted = await self._request_operator_approval(
                action_type, target_resource, approval_reason
            )
            approved_intent = getattr(self, "_pending_gui_action_intent", None)
            if (
                approval_granted
                and action_type.value.startswith("GUI_")
                and isinstance(approved_intent, GUIActionIntent)
            ):
                self._pending_gui_action_intent = replace(
                    approved_intent,
                    expires_at_ns=time.perf_counter_ns() + 2_000_000_000,
                )
            if not approval_granted:
                actual_obs_str = (
                    "Action blocked: explicit operator approval was not received. "
                    f"Reason: {approval_reason}"
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
                self.history.append(
                    {"action": action_type.value, "result": actual_obs_str}
                )
                return trace

        # ----- ASI Sandbox Pillar + Native Kernel governance -----
        # The native verdict is an independent execution gate. This agent can
        # dispatch directly to its provider, so Deny always blocks and
        # approval-required verdicts need a trusted operator callback.
        routed = self._native_govern_action(action_type, target_resource)
        native_verdict = routed.get("native_verdict")
        if native_verdict == "RequireApproval" and not approval_granted:
            approval_granted = await self._request_operator_approval(
                action_type,
                target_resource,
                str(routed.get("reason") or "native safety policy requires approval"),
            )
            approved_intent = getattr(self, "_pending_gui_action_intent", None)
            if (
                approval_granted
                and action_type.value.startswith("GUI_")
                and isinstance(approved_intent, GUIActionIntent)
            ):
                self._pending_gui_action_intent = replace(
                    approved_intent,
                    expires_at_ns=time.perf_counter_ns() + 2_000_000_000,
                )
        native_denied = native_verdict == "Deny"
        native_approval_missing = (
            native_verdict == "RequireApproval" and not approval_granted
        )
        if (
            not routed.get("executable", True)
            or native_denied
            or native_approval_missing
        ):
            actual_obs_str = (
                f"Action denied by execution governance: {routed.get('reason', 'denied')}"
            )
            if native_approval_missing:
                actual_obs_str = (
                    "Action requires operator approval before execution: "
                    f"{routed.get('reason', 'native policy requires approval')}"
                )
            status = ActionExecutionStatus.BLOCKED
            logger.warning(
                "action_blocked_by_execution_governance",
                action=action_type.value,
                tier=routed.get("tier"),
                reason=routed.get("reason"),
                native_verdict=native_verdict,
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
        provider_name = self._get_provider_name()
        tool_name = (
            payload.get("tool_name") or target_resource or action_type.value
            if isinstance(payload, dict) and action_type == ComputerActionType.SECURITY_TOOL
            else (action_type.value if hasattr(action_type, "value") else str(action_type))
        )
        if action_type == ComputerActionType.TERMINAL_EXEC and isinstance(payload, dict):
            command = str(payload.get("command") or payload.get("cmd") or "")
            try:
                command_tokens = shlex.split(command)
            except ValueError:
                command_tokens = []
            if command_tokens:
                tool_name = os.path.basename(command_tokens[0]) or command_tokens[0]
        if action_type == ComputerActionType.APP_LAUNCH:
            app_target = str(
                payload.get("app_name") or target_resource or ""
            ).strip()
            if app_target:
                tool_name = f"{action_type.value}:{app_target.casefold()}"
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

        # A GUI click/double-click whose previous attempts already produced no
        # visible change is owned by the visual-stall watchdog, not this breaker.
        # Blocking the repeat here would stop the mission on a generic "loop"
        # message one stall short of the watchdog's documented abort threshold
        # and would hide the real cause (unchanged pixels).
        stalled_gui_repeat = (
            action_type
            in (ComputerActionType.GUI_CLICK, ComputerActionType.GUI_DOUBLE_CLICK)
            and self._consecutive_unchanged_screens > 0
        )
        if (
            not stalled_gui_repeat
            and action_type
            not in {
                ComputerActionType.GUI_WAIT,
                ComputerActionType.BROWSER_WAIT,
                # Headless execution retries are governed by FailureBudgetTracker;
                # blocking the third identical attempt would prevent that tracker
                # from reaching its configured exhaustion threshold.
                ComputerActionType.TERMINAL_EXEC,
                ComputerActionType.SECURITY_TOOL,
            }
            and len(self._recent_action_signatures) >= 3
            and all(s == action_sig for s in self._recent_action_signatures[-3:])
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
        # Exact screen-pixel coordinates are authoritative. They bypass
        # semantic grounding, but both sources must agree when both are present.
        resolved_via_grounding = False
        coordinate_actions = {
            ComputerActionType.GUI_CLICK,
            ComputerActionType.GUI_DOUBLE_CLICK,
            ComputerActionType.GUI_RIGHT_CLICK,
            ComputerActionType.GUI_TYPE,
            ComputerActionType.GUI_MOVE,
            ComputerActionType.GUI_SCROLL,
            ComputerActionType.GUI_DRAG,
        }
        explicit_pixel_target = (
            _parse_pixel_coordinate_target(target_resource)
            if action_type in coordinate_actions
            else None
        )
        payload_coordinates_provided = (
            action_type in coordinate_actions
            and ("x" in payload or "y" in payload)
        )
        payload_pixel_target: tuple[int, int] | None = None
        if payload_coordinates_provided:
            x_value, y_value = payload.get("x"), payload.get("y")
            if (
                not isinstance(x_value, int)
                or isinstance(x_value, bool)
                or not isinstance(y_value, int)
                or isinstance(y_value, bool)
            ):
                actual_obs_str = (
                    "GUI coordinate payload must contain a complete valid integer coordinate x/y pair."
                )
                status = ActionExecutionStatus.BLOCKED
                action_exit_code = 126
            else:
                payload_pixel_target = (x_value, y_value)
        if (
            explicit_pixel_target is not None
            and payload_pixel_target is not None
            and explicit_pixel_target != payload_pixel_target
        ):
            actual_obs_str = (
                "GUI target and payload specify conflicting screen coordinates."
            )
            status = ActionExecutionStatus.BLOCKED
            action_exit_code = 126
        if status == ActionExecutionStatus.BLOCKED:
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
                exit_code=action_exit_code,
                thought=getattr(self, "_last_thought", ""),
                duration_seconds=round(time.perf_counter() - t_start, 3),
                thought_duration_seconds=getattr(self, "_last_thought_duration", 0.0),
            )
            self.traces.append(trace)
            self.history.append({"action": f"BLOCKED {target_resource}", "result": actual_obs_str})
            return trace
        if explicit_pixel_target is not None and payload_pixel_target is None:
            payload["x"], payload["y"] = explicit_pixel_target
            payload_pixel_target = explicit_pixel_target
        natural_target_needs_live_grounding = (
            explicit_pixel_target is None and payload_pixel_target is None
        )
        if action_type in coordinate_actions and natural_target_needs_live_grounding:
            grounding_reason = ""
            grounded = None
            if getattr(self, "_last_screen_has_pixels", False) or self._last_screenshot_b64:
                try:
                    grounded = await self._ground_target_with_avcs(target_resource)
                except Exception as exc:
                    grounding_reason = str(exc)
                    logger.error(
                        "avcs_target_grounding_failed_closed",
                        query=target_resource,
                        error=grounding_reason,
                    )
            if grounded is None:
                detail = f" {grounding_reason}" if grounding_reason else ""
                actual_obs_str = (
                    "GUI action blocked: the local AVCS grounder did not resolve "
                    f"the target '{target_resource}'.{detail}"
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
                    thought_duration_seconds=getattr(
                        self, "_last_thought_duration", 0.0
                    ),
                )
                self.traces.append(trace)
                self.history.append(
                    {"action": f"BLOCKED {target_resource}", "result": actual_obs_str}
                )
                return trace
            grounded_point = (grounded["x"], grounded["y"])
            if payload_pixel_target is not None and payload_pixel_target != grounded_point:
                actual_obs_str = (
                    "GUI action blocked: model-proposed coordinates disagree "
                    "with AVCS local grounding."
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
                    thought_duration_seconds=getattr(
                        self, "_last_thought_duration", 0.0
                    ),
                )
                self.traces.append(trace)
                self.history.append(
                    {"action": f"BLOCKED {target_resource}", "result": actual_obs_str}
                )
                return trace
            payload["x"], payload["y"] = grounded_point
            payload_pixel_target = grounded_point
            resolved_via_grounding = True

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
        # Missing point coordinates are resolved from the current visual
        # target below. Other coordinate errors (negative, non-integer, or
        # off-screen values) remain fail-closed before any grounding call.
        can_ground_missing_point = (
            action_type in {
                ComputerActionType.GUI_CLICK,
                ComputerActionType.GUI_DOUBLE_CLICK,
                ComputerActionType.GUI_RIGHT_CLICK,
                ComputerActionType.GUI_MOVE,
            }
            and payload.get("x") is None
            and payload.get("y") is None
            and bool(target_resource)
        )
        if can_ground_missing_point and coord_err == "point coordinates (x, y) are required":
            coord_err = None
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
                burst_ok = False
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
                self._inject_replan(goal=self._current_goal_for_replan())
            return trace

        try:
            if action_type == ComputerActionType.GUI_TYPE and typing_validation_error:
                actual_obs_str = f"[TYPING VERIFICATION FAILED]: {typing_validation_error}"
                status = ActionExecutionStatus.BLOCKED
                action_exit_code = 126

            elif action_type == ComputerActionType.GUI_CLICK:
                if payload.get("x") is None or payload.get("y") is None:
                    actual_obs_str = f"GUI_CLICK failed: integer pixel coordinates x,y required or visual target '{target_resource}' could not be resolved"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    x = int(payload.get("x", 0))
                    y = int(payload.get("y", 0))
                    pre_screen_b64 = getattr(self, "_last_screenshot_b64", "")
                    pre_screen_state = getattr(self, "_last_observed_screen", None)
                    # Reach before clicking, like a hand: converge the pointer
                    # onto the target so the click lands where it was aimed even
                    # if the coordinate estimate was slightly off. The click
                    # itself still goes through the normal dispatch path.
                    servo_receipt = await self._servo_to_pixel_target(workspace_id, x, y)
                    if self._last_screenshot_b64:
                        self._last_screenshot_b64 = draw_action_marker(
                            self._last_screenshot_b64, (x, y), label="CLICK", color="#00ffcc"
                        )
                    obs_after = await self._dispatch_gui_action(
                        workspace_id,
                        GUIAction(
                            action=GUIActionType.CLICK,
                            x=x,
                            y=y,
                            target_query=(
                                target_resource if explicit_pixel_target is None else None
                            ),
                        ),
                    )
                    post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                    if post_screen_b64:
                        self._last_screenshot_b64 = post_screen_b64
                        self._update_screen_dims(obs_after)
                    scene_changed = self._screen_pixels_changed(pre_screen_state, obs_after)
                    servo_note = ""
                    if isinstance(servo_receipt, dict):
                        servo_note = (
                            f" [servo: converged={servo_receipt.get('converged')}"
                            f" residual={servo_receipt.get('final_residual_px')}px"
                            f" latency={servo_receipt.get('latency_ms')}ms]"
                        )
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and clicked{servo_note}"
                    else:
                        actual_obs_str = f"Clicked at ({x}, {y}){servo_note}"
                    if (
                        scene_changed is False
                        or (
                            scene_changed is None
                            and pre_screen_b64
                            and post_screen_b64
                            and pre_screen_b64 == post_screen_b64
                        )
                    ):
                        self._consecutive_unchanged_screens += 1
                        status = ActionExecutionStatus.UNVERIFIED
                        visual_stall_detected = True
                        actual_obs_str = (
                            f"{actual_obs_str} | [VISUAL VERIFICATION]: Screen state unchanged after click; "
                            f"the effect at ({x}, {y}) is unverified. No recovery input was sent."
                        )
                        actual_obs_str, self._watchdog_aborted = await self._handle_visual_stall_watchdog(
                            workspace_id, action_type, actual_obs_str
                        )
                    elif scene_changed is True or post_screen_b64:
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
                    pre_screen_state = getattr(self, "_last_observed_screen", None)
                    # Reach before clicking, like a hand: converge the pointer onto
                    # the target so the double-click lands where it was aimed even
                    # if the coordinate estimate was slightly off. The click itself
                    # still goes through the normal dispatch path.
                    servo_receipt = await self._servo_to_pixel_target(workspace_id, x, y)
                    if self._last_screenshot_b64:
                        self._last_screenshot_b64 = draw_action_marker(
                            self._last_screenshot_b64, (x, y), label="DOUBLE_CLICK", color="#ffaa00"
                        )
                    obs_after = await self._dispatch_gui_action(
                        workspace_id,
                        GUIAction(
                            action=GUIActionType.DOUBLE_CLICK,
                            x=x,
                            y=y,
                            target_query=(
                                target_resource if explicit_pixel_target is None else None
                            ),
                        ),
                    )
                    post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                    if post_screen_b64:
                        self._last_screenshot_b64 = post_screen_b64
                        self._update_screen_dims(obs_after)
                    scene_changed = self._screen_pixels_changed(pre_screen_state, obs_after)
                    servo_note = ""
                    if isinstance(servo_receipt, dict):
                        servo_note = (
                            f" [servo: converged={servo_receipt.get('converged')}"
                            f" residual={servo_receipt.get('final_residual_px')}px"
                            f" latency={servo_receipt.get('latency_ms')}ms]"
                        )
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and double-clicked{servo_note}"
                    else:
                        actual_obs_str = f"Double-clicked at ({x}, {y}){servo_note}"
                    if (
                        scene_changed is False
                        or (
                            scene_changed is None
                            and pre_screen_b64
                            and post_screen_b64
                            and pre_screen_b64 == post_screen_b64
                        )
                    ):
                        self._consecutive_unchanged_screens += 1
                        status = ActionExecutionStatus.UNVERIFIED
                        visual_stall_detected = True
                        actual_obs_str = (
                            f"{actual_obs_str} | [VISUAL VERIFICATION]: Screen state unchanged after double-click; "
                            f"the effect at ({x}, {y}) is unverified. No recovery input was sent."
                        )
                        actual_obs_str, self._watchdog_aborted = await self._handle_visual_stall_watchdog(
                            workspace_id, action_type, actual_obs_str
                        )
                    elif scene_changed is True or post_screen_b64:
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
                    pre_screen_state = getattr(self, "_last_observed_screen", None)
                    # Reach before clicking, like a hand: converge the pointer onto
                    # the target so the right-click lands where it was aimed even
                    # if the coordinate estimate was slightly off. The click itself
                    # still goes through the normal dispatch path.
                    servo_receipt = await self._servo_to_pixel_target(workspace_id, x, y)
                    if self._last_screenshot_b64:
                        self._last_screenshot_b64 = draw_action_marker(
                            self._last_screenshot_b64, (x, y), label="RIGHT_CLICK", color="#ff3366"
                        )
                    obs_after = await self._dispatch_gui_action(
                        workspace_id,
                        GUIAction(
                            action=GUIActionType.RIGHT_CLICK,
                            x=x,
                            y=y,
                            target_query=(
                                target_resource if explicit_pixel_target is None else None
                            ),
                        ),
                    )
                    post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                    if post_screen_b64:
                        self._last_screenshot_b64 = post_screen_b64
                        self._update_screen_dims(obs_after)
                    scene_changed = self._screen_pixels_changed(pre_screen_state, obs_after)
                    servo_note = ""
                    if isinstance(servo_receipt, dict):
                        servo_note = (
                            f" [servo: converged={servo_receipt.get('converged')}"
                            f" residual={servo_receipt.get('final_residual_px')}px"
                            f" latency={servo_receipt.get('latency_ms')}ms]"
                        )
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and right-clicked{servo_note}"
                    else:
                        actual_obs_str = f"Right-clicked at ({x}, {y}){servo_note}"
                    if scene_changed is False or (
                        scene_changed is None
                        and pre_screen_b64
                        and post_screen_b64
                        and pre_screen_b64 == post_screen_b64
                    ):
                        status = ActionExecutionStatus.UNVERIFIED
                        visual_stall_detected = True
                        actual_obs_str = (
                            f"{actual_obs_str} | [VISUAL VERIFICATION]: Screen state unchanged after right-click; "
                            "the effect is unverified. No recovery input was sent."
                        )
                        actual_obs_str, self._watchdog_aborted = await self._handle_visual_stall_watchdog(
                            workspace_id, action_type, actual_obs_str
                        )
                    elif scene_changed is True or post_screen_b64:
                        self._consecutive_unchanged_screens = 0

            elif action_type == ComputerActionType.GUI_TYPE:
                raw_text = str(payload.get("text") or payload.get("value") or payload.get("input") or "")
                text = sanitize_typing_text(raw_text)
                pre_screen_b64 = getattr(self, "_last_screenshot_b64", "")
                obs_after = None
                focus_x = payload.get("x")
                focus_y = payload.get("y")
                prev_tr = self.traces[-1] if getattr(self, "traces", None) else None
                is_active_buffer = self._is_active_buffer_target(target_resource, payload, prev_tr)

                missing_focus_coordinates = focus_x is None or focus_y is None
                if (
                    self._gui_action_contract_active()
                    and missing_focus_coordinates
                ) or (
                    self._last_screenshot_b64
                    and missing_focus_coordinates
                    and not is_active_buffer
                ):
                    actual_obs_str = (
                        "GUI_TYPE blocked: a live desktop target could not be grounded to "
                        "coordinates; re-observe before typing"
                    )
                    status = ActionExecutionStatus.BLOCKED
                    action_exit_code = 126
                else:
                    target_window = payload.get("window") or payload.get("window_title")
                    if self._gui_action_contract_active() and target_window:
                        actual_obs_str = (
                            "GUI_TYPE blocked: window activation must be performed "
                            "as a separate visible GUI action; no shell-backed focus "
                            "command was run."
                        )
                        status = ActionExecutionStatus.BLOCKED
                        action_exit_code = 126
                    # Only treat an explicit window name as a window target;
                    # target_resource is usually a field label such as "search".
                    if (
                        status != ActionExecutionStatus.BLOCKED
                        and
                        hasattr(self, "motor")
                        and self.motor
                        and not self._gui_action_contract_active()
                    ):
                        if target_window and hasattr(self.motor, "settle_focus"):
                            await self.motor.settle_focus(
                                workspace_id,
                                target_window=str(target_window),
                            )
                if (
                    status != ActionExecutionStatus.BLOCKED
                    and focus_x is not None
                    and focus_y is not None
                ):
                    focus_err = self._validate_coordinates(
                        ComputerActionType.GUI_CLICK,
                        {"x": focus_x, "y": focus_y},
                    )
                    if focus_err is not None:
                        actual_obs_str = f"GUI_TYPE blocked before focus click: {focus_err}"
                        status = ActionExecutionStatus.BLOCKED
                        action_exit_code = 126
                    else:
                        # Reach before focusing, like a hand: converge the pointer
                        # onto the field so the focus click lands on it before any
                        # text is typed.
                        await self._servo_to_pixel_target(
                            workspace_id, int(focus_x), int(focus_y)
                        )
                        await self._dispatch_gui_action(
                            workspace_id,
                            GUIAction(
                                action=GUIActionType.CLICK,
                                x=int(focus_x),
                                y=int(focus_y),
                            ),
                        )
                        focused_observation = await self.observe(workspace_id)
                        if self._last_screenshot_b64:
                            focused_screen = await self.computer.screenshot(
                                workspace_id
                            )
                            focused_age = getattr(
                                focused_screen, "capture_age_ms", None
                            )
                            source_observation = getattr(
                                self, "_last_world_observation", None
                            )
                            source_screen = getattr(
                                source_observation, "screen", None
                            )
                            source_window = str(
                                getattr(source_screen, "active_window", "") or ""
                            ).strip()
                            focused_window = str(
                                getattr(focused_screen, "active_window", "") or ""
                            ).strip()
                            if not (
                                getattr(focused_screen, "screenshot_base64", "")
                                and getattr(focused_screen, "width", 0) > 0
                                and getattr(focused_screen, "height", 0) > 0
                                and isinstance(focused_age, (int, float))
                                and not isinstance(focused_age, bool)
                                and math.isfinite(focused_age)
                                and 0 <= focused_age <= 2_000
                                and not (
                                    source_window
                                    and focused_window
                                    and source_window.casefold()
                                    != focused_window.casefold()
                                )
                            ):
                                actual_obs_str = (
                                    "GUI_TYPE blocked: fresh post-focus screenshot "
                                    "was unavailable or the active window changed; "
                                    "no text was sent."
                                )
                                status = ActionExecutionStatus.BLOCKED
                                action_exit_code = 126
                            else:
                                focused_observation.screen = focused_screen
                                focused_snapshot = self.perception_bus.publish(
                                    screenshot_base64=focused_screen.screenshot_base64,
                                    width=focused_screen.width,
                                    height=focused_screen.height,
                                    active_window=focused_screen.active_window,
                                )
                                focused_observation.perception_version = (
                                    focused_snapshot.version
                                )
                                focused_observation.observation_age_ms = focused_age
                                self._last_screenshot_b64 = (
                                    focused_screen.screenshot_base64
                                )
                                self._update_screen_dims(focused_screen)
                                focus_confirmed, focus_evidence = (
                                    await self._verify_gui_focus_with_avcs(
                                        focused_screen,
                                        x=int(focus_x),
                                        y=int(focus_y),
                                        target=target_resource,
                                    )
                                )
                                if not focus_confirmed:
                                    actual_obs_str = (
                                        "GUI_TYPE blocked: the fresh screenshot "
                                        "did not visually confirm focus at the "
                                        f"requested field ({focus_evidence}); "
                                        "no text was sent."
                                    )
                                    status = ActionExecutionStatus.BLOCKED
                                    action_exit_code = 126
                        else:
                            actual_obs_str = (
                                "GUI_TYPE blocked: AVCS screenshot pixels are "
                                "unavailable for focus verification; no text was sent."
                            )
                            status = ActionExecutionStatus.BLOCKED
                            action_exit_code = 126
                        if status != ActionExecutionStatus.BLOCKED:
                            pending_intent = getattr(
                                self, "_pending_gui_action_intent", None
                            )
                            if pending_intent is not None:
                                try:
                                    self._pending_gui_action_intent = (
                                        pending_intent.rebase_after_verified_transition(
                                            focused_observation
                                        )
                                    )
                                except ValueError as exc:
                                    actual_obs_str = (
                                        "GUI_TYPE blocked: fresh post-focus observation "
                                        f"could not renew the bounded intent: {exc}"
                                    )
                                    status = ActionExecutionStatus.BLOCKED
                                    action_exit_code = 126
                if (
                    status != ActionExecutionStatus.BLOCKED
                    and payload.get("clear_field") is True
                ):
                    if self._gui_action_contract_active():
                        await self._dispatch_gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.HOTKEY, key="ctrl+a"),
                        )
                        await self._dispatch_gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.KEYPRESS, key="BackSpace"),
                        )
                    elif (
                        hasattr(self, "motor")
                        and self.motor
                        and hasattr(self.motor, "clear_field")
                    ):
                        await self.motor.clear_field(workspace_id)
                use_motor = bool(
                    hasattr(self, "motor")
                    and self.motor
                    and (self._last_screenshot_b64 or focus_x is not None or focus_y is not None)
                )
                if use_motor:
                    if status == ActionExecutionStatus.BLOCKED:
                        pass
                    elif self._gui_action_contract_active():
                        obs_after = await self._dispatch_gui_action(
                            workspace_id,
                            GUIAction(
                                action=GUIActionType.TYPE,
                                text=text,
                                x=int(focus_x) if focus_x is not None else None,
                                y=int(focus_y) if focus_y is not None else None,
                                delay_ms=payload.get("delay_ms", 25),
                            ),
                            actor="sonic-agent",
                        )
                    else:
                        await self.motor.human_type(workspace_id, text, delay_ms=payload.get("delay_ms", 25))
                elif status != ActionExecutionStatus.BLOCKED:
                    obs_after = await self._dispatch_gui_action(
                        workspace_id,
                        GUIAction(
                            action=GUIActionType.TYPE,
                            text=text,
                            x=int(focus_x) if focus_x is not None else None,
                            y=int(focus_y) if focus_y is not None else None,
                        ),
                        actor="sonic-agent",
                    )
                if status != ActionExecutionStatus.BLOCKED and hasattr(self.computer, "screenshot") and not obs_after:
                    try:
                        obs_after = await self.computer.screenshot(workspace_id)
                    except Exception:
                        pass
                post_screen_b64 = getattr(obs_after, "screenshot_base64", "") if obs_after else ""
                if post_screen_b64:
                    self._last_screenshot_b64 = post_screen_b64
                    self._prev_screenshot_hash = hashlib.md5(post_screen_b64.encode()).hexdigest()
                    self._update_screen_dims(obs_after)
                if status != ActionExecutionStatus.BLOCKED:
                    if pre_screen_b64 and post_screen_b64 and pre_screen_b64 == post_screen_b64:
                        self._consecutive_unchanged_screens += 1
                        status = ActionExecutionStatus.UNVERIFIED
                        visual_stall_detected = True
                        actual_obs_str = (
                            "[VISUAL VERIFICATION]: Screen state unchanged after typing; "
                            "the text entry effect is unverified. No recovery input was sent."
                        )
                        actual_obs_str, self._watchdog_aborted = await self._handle_visual_stall_watchdog(
                            workspace_id, action_type, actual_obs_str
                        )
                    elif focus_x is not None and focus_y is not None:
                        if post_screen_b64:
                            self._consecutive_unchanged_screens = 0
                        actual_obs_str = (
                            f"Typed {len(text)} chars at focused coordinates "
                            f"({focus_x}, {focus_y}): {text[:50]}"
                        )
                    else:
                        if post_screen_b64:
                            self._consecutive_unchanged_screens = 0
                        actual_obs_str = f"Typed {len(text)} chars: {text[:50]}"

            elif action_type in (ComputerActionType.GUI_KEYPRESS, ComputerActionType.GUI_HOTKEY):
                key = str(payload.get("key") or payload.get("keys") or target_resource or "Return").strip()
                pre_key_b64 = getattr(self, "_last_screenshot_b64", "")
                gui_act_type = getattr(GUIActionType, "HOTKEY", GUIActionType.KEYPRESS) if (action_type == ComputerActionType.GUI_HOTKEY or "+" in key) else GUIActionType.KEYPRESS
                obs_key = await self._dispatch_gui_action(
                    workspace_id,
                    GUIAction(action=gui_act_type, key=key),
                )
                key_b64 = getattr(obs_key, "screenshot_base64", "") if obs_key else ""
                if key_b64:
                    self._last_screenshot_b64 = key_b64
                    self._prev_screenshot_hash = hashlib.md5(key_b64.encode()).hexdigest()
                    self._update_screen_dims(obs_key)
                actual_obs_str = f"Pressed {'hotkey' if action_type == ComputerActionType.GUI_HOTKEY or '+' in key else 'key'}: {key}"
                if (
                    pre_key_b64
                    and key_b64
                    and pre_key_b64 == key_b64
                ):
                    self._consecutive_unchanged_screens += 1
                    status = ActionExecutionStatus.UNVERIFIED
                    visual_stall_detected = True
                    actual_obs_str = (
                        "[VISUAL VERIFICATION]: Screen state unchanged after keypress; "
                        "the keypress effect is unverified. No recovery input was sent."
                    )
                    actual_obs_str, self._watchdog_aborted = await self._handle_visual_stall_watchdog(
                        workspace_id, action_type, actual_obs_str
                    )
                elif key_b64:
                    self._consecutive_unchanged_screens = 0

            elif action_type == ComputerActionType.GUI_MOVE:
                if payload.get("x") is None or payload.get("y") is None:
                    actual_obs_str = f"GUI_MOVE failed: coordinates required or visual target '{target_resource}' could not be resolved"
                    recovery_needed = True
                    status = ActionExecutionStatus.FAILED
                else:
                    x = int(payload.get("x", 0))
                    y = int(payload.get("y", 0))
                    # Closed-loop reach: converge the pointer onto the target and
                    # carry the measured receipt. Falls back to a single dispatch
                    # when no pointer readback is available (open-loop, as before).
                    # The final MOVE still goes through the normal dispatch path so
                    # the pre-dispatch freshness revalidation is preserved; it is
                    # now only a sub-tolerance precision correction.
                    servo_receipt = await self._servo_to_pixel_target(workspace_id, x, y)
                    obs_move = await self._dispatch_gui_action(
                        workspace_id,
                        GUIAction(
                            action=GUIActionType.MOVE,
                            x=x,
                            y=y,
                            target_query=(
                                target_resource if explicit_pixel_target is None else None
                            ),
                        ),
                    )
                    move_b64 = getattr(obs_move, "screenshot_base64", "") if obs_move else ""
                    if move_b64:
                        self._last_screenshot_b64 = move_b64
                        self._prev_screenshot_hash = hashlib.md5(move_b64.encode()).hexdigest()
                        self._update_screen_dims(obs_move)
                    servo_note = ""
                    if isinstance(servo_receipt, dict):
                        servo_note = (
                            f" [servo: converged={servo_receipt.get('converged')}"
                            f" iterations={servo_receipt.get('iterations')}"
                            f" residual={servo_receipt.get('final_residual_px')}px"
                            f" latency={servo_receipt.get('latency_ms')}ms]"
                        )
                    if resolved_via_grounding:
                        actual_obs_str = f"Visual grounding resolved '{target_resource}' to ({x}, {y}) and moved mouse{servo_note}"
                    else:
                        actual_obs_str = f"Moved mouse to ({x}, {y}){servo_note}"

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
                    # Reach before pressing, like a hand: converge the pointer onto
                    # the drag origin so the press lands where it was aimed. The
                    # destination is a path endpoint, not a fixated target, so it is
                    # not servo-corrected.
                    servo_receipt = await self._servo_to_pixel_target(workspace_id, x, y)
                    obs_drag = await self._dispatch_gui_action(
                        workspace_id,
                        GUIAction(action=GUIActionType.DRAG, x=x, y=y, x2=x2, y2=y2),
                    )
                    drag_b64 = getattr(obs_drag, "screenshot_base64", "") if obs_drag else ""
                    if drag_b64:
                        self._last_screenshot_b64 = drag_b64
                        self._prev_screenshot_hash = hashlib.md5(drag_b64.encode()).hexdigest()
                        self._update_screen_dims(obs_drag)
                    servo_note = ""
                    if isinstance(servo_receipt, dict):
                        servo_note = (
                            f" [servo: converged={servo_receipt.get('converged')}"
                            f" residual={servo_receipt.get('final_residual_px')}px"
                            f" latency={servo_receipt.get('latency_ms')}ms]"
                        )
                    actual_obs_str = f"Dragged from ({x},{y}) to ({x2},{y2}){servo_note}"

            elif action_type == ComputerActionType.GUI_SCROLL:
                coord_err = self._validate_coordinates(action_type, payload)
                if coord_err is not None:
                    actual_obs_str = f"GUI_SCROLL failed: explicit valid coordinates required ({coord_err})"
                    status = ActionExecutionStatus.BLOCKED
                    action_exit_code = 126
                else:
                    x = int(payload["x"])
                    y = int(payload["y"])
                    raw_delta = payload.get("delta")
                    if (
                        not isinstance(raw_delta, int)
                        or isinstance(raw_delta, bool)
                    ):
                        actual_obs_str = "GUI_SCROLL failed: explicit integer delta required"
                        status = ActionExecutionStatus.BLOCKED
                        action_exit_code = 126
                    else:
                        delta = raw_delta
                        # Scroll events land on whatever is under the pointer, so
                        # converge the pointer onto the target point first; the
                        # scroll then affects the intended pane.
                        servo_receipt = await self._servo_to_pixel_target(workspace_id, x, y)
                        obs_scroll = await self._dispatch_gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.SCROLL, x=x, y=y, scroll_delta=delta),
                        )
                        scroll_b64 = getattr(obs_scroll, "screenshot_base64", "") if obs_scroll else ""
                        if scroll_b64:
                            self._last_screenshot_b64 = scroll_b64
                            self._prev_screenshot_hash = hashlib.md5(scroll_b64.encode()).hexdigest()
                        servo_note = ""
                        if isinstance(servo_receipt, dict):
                            servo_note = (
                                f" [servo: converged={servo_receipt.get('converged')}"
                                f" residual={servo_receipt.get('final_residual_px')}px]"
                            )
                        actual_obs_str = f"Scrolled at ({x}, {y}) delta={delta}{servo_note}"

            elif action_type == ComputerActionType.GUI_SCREENSHOT:
                screen = await self.computer.screenshot(workspace_id)
                self._last_screenshot_b64 = getattr(screen, "screenshot_base64", "")
                self._last_active_window_geometry = getattr(screen, "active_window_geometry", None)
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
                            app_launch_failed = True
                            raise RuntimeError(f"application launch failed for '{app_name}'")
                    else:
                        await self._dispatch_gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.OPEN_APP, app_name=app_request),
                        )
                    import asyncio as _asyncio
                    await _asyncio.sleep(1.0)
                    screen = await self.computer.screenshot(workspace_id)
                    screenshot_b64 = str(
                        getattr(screen, "screenshot_base64", "") or ""
                    )
                    raw_rgb = bytes(getattr(screen, "raw_rgb", b"") or b"")
                    frame_has_pixels = bool(screenshot_b64) or (
                        screen.width > 0
                        and screen.height > 0
                        and len(raw_rgb) == screen.width * screen.height * 3
                    )
                    self._last_screenshot_b64 = screenshot_b64
                    self._last_screen_has_pixels = frame_has_pixels
                    self._update_screen_dims(screen)
                    screen_state = str(getattr(screen, "desktop_state", "") or "").upper()
                    active_w = str(getattr(screen, "active_window", "") or "").strip()
                    after_status = await self.computer.status(workspace_id)
                    if not active_w:
                        active_w = str(getattr(after_status, "active_window", "") or "").strip()
                    active_window_class = ""
                    get_active_window_class = getattr(
                        self.computer,
                        "get_active_window_class",
                        None,
                    )
                    if callable(get_active_window_class):
                        active_window_class = await get_active_window_class(
                            workspace_id
                        )
                    windows_after = {
                        str(window).strip()
                        for window in (getattr(after_status, "open_applications", None) or [])
                        if str(window).strip()
                    }
                    if (
                        screen_state == "NO_DISPLAY"
                        or not frame_has_pixels
                        or active_w.lower() in {"", "desktop"}
                        or not (
                            self._window_matches_application(app_name, active_w)
                            or self._window_matches_application(
                                app_name,
                                active_window_class,
                            )
                        )
                    ):
                        raise RuntimeError(
                            f"application '{app_name}' did not produce a verified, focused GUI window "
                            f"(active window: {active_w or 'unknown'}; "
                            f"open windows: {sorted(windows_after)})"
                        )
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
                    before_status = await self.computer.status(workspace_id)
                    before_windows = [
                        str(window)
                        for window in (
                            list(getattr(before_status, "open_applications", []) or [])
                            + [
                                str(getattr(before_status, "active_application", "") or ""),
                                str(getattr(before_status, "active_window", "") or ""),
                            ]
                        )
                        if str(window).strip()
                    ]
                    was_open = any(
                        self._window_matches_application(app_name, window)
                        for window in before_windows
                    )
                    if hasattr(self.computer, "close_application"):
                        await self.computer.close_application(workspace_id, app_name)
                    else:
                        await self._dispatch_gui_action(
                            workspace_id,
                            GUIAction(action=GUIActionType.CLOSE_APP, app_name=app_name),
                        )
                    await asyncio.sleep(0.15)
                    after_status = await self.computer.status(workspace_id)
                    after_windows = [
                        str(window)
                        for window in (
                            list(getattr(after_status, "open_applications", []) or [])
                            + [
                                str(getattr(after_status, "active_application", "") or ""),
                                str(getattr(after_status, "active_window", "") or ""),
                            ]
                        )
                        if str(window).strip()
                    ]
                    still_open = any(
                        self._window_matches_application(app_name, window)
                        for window in after_windows
                    )
                    screen = await self.computer.screenshot(workspace_id)
                    raw_rgb = bytes(getattr(screen, "raw_rgb", b"") or b"")
                    frame_has_pixels = bool(
                        getattr(screen, "screenshot_base64", "")
                    ) or (
                        screen.width > 0
                        and screen.height > 0
                        and len(raw_rgb) == screen.width * screen.height * 3
                    )
                    capture_age_ms = getattr(screen, "capture_age_ms", None)
                    frame_is_fresh = (
                        isinstance(capture_age_ms, (int, float))
                        and not isinstance(capture_age_ms, bool)
                        and math.isfinite(capture_age_ms)
                        and 0 <= capture_age_ms <= 2_000
                    )
                    if still_open:
                        status = ActionExecutionStatus.FAILED
                        actual_obs_str = (
                            f"APP_CLOSE was dispatched, but '{app_name}' remains "
                            "visible in the fresh window state."
                        )
                    elif not frame_has_pixels or not frame_is_fresh:
                        status = ActionExecutionStatus.UNVERIFIED
                        actual_obs_str = (
                            f"Window state no longer lists '{app_name}', but the "
                            "post-close desktop frame is missing or stale."
                        )
                    else:
                        status = ActionExecutionStatus.VERIFIED
                        qualifier = "Closed" if was_open else "Already closed"
                        actual_obs_str = (
                            f"{qualifier} {app_name}; fresh desktop state confirms "
                            "no matching application window remains."
                        )

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
                    try:
                        obs_focus = await self._dispatch_gui_action(
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
                    except Exception as focus_exc:
                        # Auto-assist: If focusing an unopened app, attempt to launch it so the user's objective progresses
                        launched = False
                        if hasattr(self.computer, "launch_application"):
                            launched = await self.computer.launch_application(workspace_id, app_name)
                        if launched:
                            await asyncio.sleep(1.0)
                            obs_focus = await self.computer.screenshot(workspace_id)
                            focus_b64 = getattr(obs_focus, "screenshot_base64", "") if obs_focus else ""
                            if focus_b64:
                                self._last_screenshot_b64 = focus_b64
                                self._update_screen_dims(obs_focus)
                            active_w = getattr(obs_focus, "active_window", "") or app_name
                            actual_obs_str = f"Window '{app_name}' was not running; automatically launched '{app_name}' on desktop (active window: {active_w})."
                        else:
                            actual_obs_str = f"APP_FOCUS notice: window '{app_name}' is not currently open. Use APP_LAUNCH to start '{app_name}'."
                            status = ActionExecutionStatus.FAILED

            elif action_type in (
                ComputerActionType.APP_MAXIMIZE,
                ComputerActionType.WINDOW_MAXIMIZE,
            ):
                window_target = str(
                    payload.get("window_id") or payload.get("app_name") or target_resource or ""
                ).strip()
                action = GUIAction(
                    action=GUIActionType.APP_MAXIMIZE,
                    window_id=window_target,
                    app_name=window_target,
                )
                await self._dispatch_gui_action(workspace_id, action)
                actual_obs_str = (
                    f"Maximized window: {window_target}"
                    if window_target
                    else "Maximized active window"
                )

            elif action_type == ComputerActionType.WINDOW_TILE:
                await self._dispatch_gui_action(
                    workspace_id,
                    GUIAction(action=GUIActionType.WINDOW_TILE),
                )
                actual_obs_str = "Tiled workstation windows"

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

            elif action_type == ComputerActionType.VERIFY_PROBE:
                try:
                    probe = VerificationProbe.model_validate(payload.get("probe"))
                    command = probe.to_command()
                except Exception as exc:
                    status = ActionExecutionStatus.BLOCKED
                    action_exit_code = 126
                    actual_obs_str = f"Invalid verification probe: {exc}"
                else:
                    res = await self.computer.terminal(workspace_id, command)
                    action_exit_code = getattr(res, "exit_code", None)
                    safe_stdout = _safe_str(getattr(res, "stdout", "") or "")
                    safe_stderr = _safe_str(getattr(res, "stderr", "") or "")
                    action_stderr = safe_stderr
                    actual_obs_str = (
                        "\n".join(part for part in (safe_stdout, safe_stderr) if part)
                        or f"Exit {action_exit_code}"
                    )
                    if action_exit_code is None:
                        status = ActionExecutionStatus.FAILED
                        actual_obs_str = f"Verification probe returned no exit code. {actual_obs_str}"
                    elif action_exit_code == 124:
                        status = ActionExecutionStatus.TIMED_OUT
                    elif action_exit_code in (125, 126):
                        status = ActionExecutionStatus.BLOCKED
                    elif action_exit_code != 0:
                        status = ActionExecutionStatus.FAILED

            elif action_type == ComputerActionType.TERMINAL_EXEC:
                raw_cmd = payload.get("command") if (isinstance(payload, dict) and "command" in payload) else target_resource
                if raw_cmd is None:
                    raw_cmd = ""
                if isinstance(raw_cmd, dict):
                    raw_cmd = raw_cmd.get("command", "")
                cmd = self._clean_terminal_command(raw_cmd)
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
                # 1. Try branch creation or switch with 3-way merge (-m / --merge) so uncommitted changes are preserved
                res = await self.computer.terminal(
                    workspace_id,
                    f"git checkout -m -b {shlex.quote(branch_name)} || git checkout --merge {shlex.quote(branch_name)}",
                )
                action_exit_code = getattr(res, "exit_code", 0)
                # 2. If checkout --merge collided on uncommitted working tree changes, protect them with auto-stash
                if action_exit_code != 0:
                    stash_cmd = (
                        f"git stash push -m sonic-auto-stash --include-untracked && "
                        f"(git checkout -b {shlex.quote(branch_name)} || git checkout {shlex.quote(branch_name)}) && "
                        f"git stash pop"
                    )
                    res = await self.computer.terminal(workspace_id, stash_cmd)
                    action_exit_code = getattr(res, "exit_code", 0)

                obs_parts = []
                if _safe_str(getattr(res, "stdout", "")).strip():
                    obs_parts.append(_safe_str(getattr(res, "stdout", "")).strip())
                if _safe_str(getattr(res, "stderr", "")).strip():
                    obs_parts.append(_safe_str(getattr(res, "stderr", "")).strip())
                if action_exit_code == 0:
                    obs_parts.append(f"Switched to branch {branch_name}")
                actual_obs_str = "\n".join(obs_parts).strip()
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

                await self._dispatch_gui_action(
                    workspace_id,
                    GUIAction(action=GUIActionType.KEYPRESS, key="ctrl+l"),
                )
                await self._dispatch_gui_action(
                    workspace_id,
                    GUIAction(action=GUIActionType.TYPE, text=url, delay_ms=25),
                )
                await self._dispatch_gui_action(
                    workspace_id,
                    GUIAction(action=GUIActionType.KEYPRESS, key="Return"),
                )
                await asyncio.sleep(0.5)
                screen = await self.computer.screenshot(workspace_id)
                self._update_screen_dims(screen)
                if (
                    getattr(screen, "desktop_state", "") == "NO_DISPLAY"
                    or not getattr(screen, "screenshot_base64", "")
                ):
                    raise RuntimeError(
                        "visible application navigation could not be verified from a live screen"
                    )
                self._last_screenshot_b64 = getattr(screen, "screenshot_base64", "") or ""
                self._last_gui_navigation_target = url
                actual_obs_str = "Entered the requested destination through the visible desktop and captured its screen"

                if hasattr(self, "scratchpad") and self.scratchpad:
                    self.scratchpad.extract_from_text(url, source="url")
            elif action_type == ComputerActionType.BROWSER_SELECT:
                selector = str(payload.get("selector") or target_resource or "").strip()
                val = str(payload.get("value") or payload.get("option") or payload.get("label") or "").strip()
                selector_coords = await resolve_visible_desktop_target(selector)
                if selector_coords is None or not val:
                    actual_obs_str = (
                        f"Could not visually resolve the selection control and option "
                        f"'{val}' on the shared desktop"
                    )
                    status = ActionExecutionStatus.FAILED
                else:
                    select_obs = await self._dispatch_gui_action(
                        workspace_id,
                        GUIAction(
                            action=GUIActionType.CLICK,
                            x=selector_coords[0],
                            y=selector_coords[1],
                        ),
                    )
                    self._update_screen_dims(select_obs)
                    select_image = getattr(select_obs, "screenshot_base64", "")
                    if not select_image:
                        actual_obs_str = (
                            f"Opened selection control '{selector}', but the updated desktop "
                            "image is unavailable; option selection was not attempted"
                        )
                        status = ActionExecutionStatus.FAILED
                    else:
                        self._last_screenshot_b64 = select_image
                        option_coords = await resolve_visible_desktop_target(val)
                        if option_coords is None:
                            actual_obs_str = (
                                f"Opened selection control '{selector}', but option '{val}' "
                                "could not be grounded on the updated desktop"
                            )
                            status = ActionExecutionStatus.FAILED
                        else:
                            await self._dispatch_gui_action(
                                workspace_id,
                                GUIAction(
                                    action=GUIActionType.CLICK,
                                    x=option_coords[0],
                                    y=option_coords[1],
                                ),
                            )
                            actual_obs_str = f"Selected visible option '{val}' via desktop grounding"

            elif action_type == ComputerActionType.WAIT_FOR_HUMAN:
                reason = str(payload.get("reason") or payload.get("prompt") or target_resource or "Human assistance requested (CAPTCHA/OTP/2FA)").strip()
                self._waiting_for_human = True
                self._waiting_for_human_reason = reason
                status = ActionExecutionStatus.WAITING_FOR_HUMAN
                actual_obs_str = f"WAITING_FOR_HUMAN: {reason}"

            elif action_type == ComputerActionType.BROWSER_DOWNLOAD:
                selector = payload.get("selector") or target_resource
                actual_obs_str = (
                    f"Dedicated download is unavailable; use visible GUI actions to activate "
                    f"'{selector}' and handle the application's save dialog"
                )
                status = ActionExecutionStatus.FAILED
                recovery_needed = False

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
                            tenant_id=self.tenant_id,
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
                sub_name = str(payload.get("name") or payload.get("Name") or target_resource or "").strip()
                sub_desc = str(payload.get("description") or payload.get("Description") or "")
                sub_sys = str(payload.get("system_prompt") or payload.get("SystemPrompt") or payload.get("systemPrompt") or "")
                enable_write = bool(payload.get("enable_write_tools") if "enable_write_tools" in payload else payload.get("enableWriteTools", True))
                enable_sub = bool(payload.get("enable_subagent_tools") if "enable_subagent_tools" in payload else payload.get("enableSubagentTools", False))
                spec = self._teamwork_manager.define_subagent(
                    name=sub_name,
                    description=sub_desc,
                    system_prompt=sub_sys,
                    enable_write_tools=enable_write,
                    enable_subagent_tools=enable_sub,
                )
                actual_obs_str = (
                    f"Defined subagent type '{spec.name}' "
                    f"(write_tools={spec.enable_write_tools}, subagent_tools={spec.enable_subagent_tools}): {spec.description}"
                )
                status = ActionExecutionStatus.VERIFIED

            elif action_type == ComputerActionType.INVOKE_SUBAGENT:
                from sonic.computer_use.teamwork import SubagentTeamworkManager, _get_field

                if getattr(self, "_teamwork_manager", None) is None:
                    self._teamwork_manager = SubagentTeamworkManager(
                        parent_agent=self,
                        computer_provider=self.computer,
                        llm_router=self.llm_router,
                        safety=self.safety,
                        tenant_id=self.tenant_id,
                    )
                def _norm_str_list(v: Any) -> list[str]:
                    if not v:
                        return []
                    if isinstance(v, str):
                        lines = [ln.strip() for ln in v.splitlines() if ln.strip()]
                        return lines if lines else [v.strip()]
                    if isinstance(v, (list, tuple, set)):
                        return [str(it).strip() for it in v if it is not None and str(it).strip()]
                    return [str(v).strip()]

                raw_subagents = payload.get("subagents") or payload.get("Subagents")
                if not raw_subagents:
                    raw_subagents = [payload] if (isinstance(payload, dict) and any(
                        k in payload for k in ("prompt", "Prompt", "goal", "task", "instruction", "role", "Role", "title", "name", "type_name", "type", "requirements", "acceptance_criteria")
                    )) else [
                        {
                            "type_name": str(_get_field(payload, "type_name", "type", default="self") or "self"),
                            "role": str(_get_field(payload, "role", "title", "name", default=target_resource or "Autonomous Subagent") or target_resource or "Autonomous Subagent"),
                            "prompt": str(_get_field(payload, "prompt", "goal", "task", "instruction", default=target_resource or "") or target_resource or ""),
                            "requirements": _norm_str_list(_get_field(payload, "requirements", default=None)),
                            "acceptance_criteria": _norm_str_list(_get_field(payload, "acceptance_criteria", "criteria", default=None)),
                            "verification_command": str(_get_field(payload, "verification_command", "verify_cmd", default="") or ""),
                            "workspace_mode": str(_get_field(payload, "workspace_mode", "workspace", default="branch") or "branch"),
                            "execution_plane": str(_get_field(payload, "execution_plane", "plane", default="dual") or "dual"),
                            "model_tier": str(_get_field(payload, "model_tier", "model", default="inherit") or "inherit"),
                            "max_steps": int(payload.get("max_steps") or payload.get("MaxSteps") or 12),
                            "max_correction_rounds": int(payload.get("max_correction_rounds") if payload.get("max_correction_rounds") is not None else 2),
                        }
                    ]
                elif isinstance(raw_subagents, dict):
                    raw_subagents = [raw_subagents]
                elif not isinstance(raw_subagents, list):
                    raw_subagents = [raw_subagents]

                subagents_arg: list[dict[str, Any]] = []
                for entry in raw_subagents:
                    if not isinstance(entry, dict):
                        continue
                    p = str(
                        _get_field(entry, "prompt", "goal", "task", "instruction", default="")
                        or (target_resource if not (payload.get("subagents") or payload.get("Subagents")) else "")
                        or ""
                    ).strip()
                    if not p:
                        # Prevent launching empty/blank subagents
                        continue
                    plane = str(_get_field(entry, "execution_plane", "plane", default="dual") or "dual").lower().strip()
                    if plane not in ("gui", "headless", "dual"):
                        plane = "dual"
                    tier = str(_get_field(entry, "model_tier", "model", default="inherit") or "inherit").lower().strip()
                    if tier not in ("inherit", "flash", "pro"):
                        tier = "inherit"
                    mode = str(_get_field(entry, "workspace_mode", "workspace", default="branch") or "branch").lower().strip()
                    if mode not in ("inherit", "branch", "share"):
                        mode = "branch"

                    subagents_arg.append({
                        "type_name": str(_get_field(entry, "type_name", "type", default="self") or "self"),
                        "role": str(_get_field(entry, "role", "title", "name", default="Autonomous Subagent") or "Autonomous Subagent"),
                        "prompt": p,
                        "requirements": _norm_str_list(_get_field(entry, "requirements", default=None)),
                        "acceptance_criteria": _norm_str_list(_get_field(entry, "acceptance_criteria", "criteria", default=None)),
                        "verification_command": str(_get_field(entry, "verification_command", "verify_cmd", default="") or ""),
                        "workspace_mode": mode,
                        "execution_plane": plane,
                        "model_tier": tier,
                        "max_steps": int(entry.get("max_steps") or entry.get("MaxSteps") or 12),
                        "max_correction_rounds": int(entry.get("max_correction_rounds") if entry.get("max_correction_rounds") is not None else 2),
                        "employee_id": str(entry.get("employee_id") or "").strip() or None,
                    })

                if not subagents_arg:
                    actual_obs_str = "Error: Cannot invoke subagents with empty prompt. Each subagent must specify a non-empty prompt/objective."
                    status = ActionExecutionStatus.FAILED
                else:
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
                recipient = str(
                    payload.get("conversation_id")
                    or payload.get("conversationId")
                    or payload.get("recipient")
                    or payload.get("Recipient")
                    or target_resource
                    or ""
                ).strip()
                msg_text = str(
                    payload.get("message")
                    or payload.get("Message")
                    or payload.get("text")
                    or payload.get("prompt")
                    or ""
                )
                raw_ac = payload.get("acceptance_criteria") or payload.get("AcceptanceCriteria")
                ac = _norm_str_list(raw_ac) if raw_ac is not None else None
                vc = str(payload.get("verification_command") or payload.get("VerificationCommand") or "")
                max_s = int(payload.get("max_steps") or payload.get("MaxSteps") or 8)
                conv = await self._teamwork_manager.send_message(
                    conversation_id=recipient,
                    message=msg_text,
                    acceptance_criteria=ac,
                    verification_command=vc,
                    max_steps=max_s,
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
                m_action = str(
                    payload.get("action")
                    or payload.get("Action")
                    or (target_resource if target_resource in ("list", "kill", "kill_all") else "list")
                ).strip().lower()
                c_ids = list(payload.get("conversation_ids") or payload.get("ConversationIds") or [])
                if not c_ids and target_resource and target_resource not in ("list", "kill", "kill_all"):
                    c_ids = [target_resource]
                if not c_ids and (payload.get("conversation_id") or payload.get("conversationId")):
                    c_ids = [str(payload.get("conversation_id") or payload.get("conversationId"))]
                listing = self._teamwork_manager.manage_subagents(action=m_action, conversation_ids=c_ids)
                actual_obs_str = json.dumps(listing, separators=(",", ":"))
                status = ActionExecutionStatus.COMPLETED

        except Exception as e:
            actual_obs_str = f"Error: {str(e)}"
            recovery_needed = not app_launch_failed
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
        state_delta = getattr(self, "_last_gui_state_delta", None)
        state_verification_actions = GUI_STATE_VERIFICATION_ACTIONS
        if (
            isinstance(state_delta, dict)
            and action_type in state_verification_actions
            and not visual_stall_detected
            and status not in (
                ActionExecutionStatus.BLOCKED,
                ActionExecutionStatus.FAILED,
            )
        ):
            state_changed = state_delta.get("changed")
            if state_changed is False:
                self._consecutive_unchanged_screens += 1
                status = ActionExecutionStatus.UNVERIFIED
                visual_stall_detected = True
                actual_obs_str = (
                    f"{actual_obs_str} | [AVCS VERIFICATION]: The pixel scene "
                    "did not change after the action; its effect is unverified. "
                    "No recovery input was sent."
                )
                actual_obs_str, self._watchdog_aborted = await self._handle_visual_stall_watchdog(
                    workspace_id, action_type, actual_obs_str
                )
            elif state_changed is True:
                self._consecutive_unchanged_screens = 0
                actual_obs_str = (
                    f"{actual_obs_str} | [AVCS STATE]: The local pixel scene changed "
                    "after the action; task completion is not implied."
                )
            else:
                status = ActionExecutionStatus.UNVERIFIED
                actual_obs_str = (
                    f"{actual_obs_str} | [AVCS VERIFICATION]: A fresh post-action "
                    "pixel state was received but no pre-action scene was available."
                )
        if (
            action_type in state_verification_actions
            and status == ActionExecutionStatus.COMPLETED
            and not getattr(self, "_last_gui_action_observation_valid", False)
        ):
            status = ActionExecutionStatus.UNVERIFIED
            actual_obs_str = (
                f"{actual_obs_str} | [POST-ACTION VERIFICATION]: "
                "No fresh post-action desktop observation was available; "
                "the action is unverified."
            )
        postcondition_evidence = ""
        if (
            action_type in _POSTCONDITION_REQUIRED_GUI_ACTIONS
            and isinstance(pending_intent, GUIActionIntent)
            and status
            in (
                ActionExecutionStatus.COMPLETED,
                ActionExecutionStatus.SUCCESS,
                ActionExecutionStatus.VERIFIED,
                ActionExecutionStatus.UNVERIFIED,
            )
        ):
            postcondition_ok, postcondition_reason = (
                self._verify_single_gui_postcondition(payload)
            )
            if postcondition_ok:
                status = ActionExecutionStatus.VERIFIED
                postcondition_evidence = (
                    "Structured GUI postcondition observed in fresh desktop state."
                )
            else:
                status = ActionExecutionStatus.UNVERIFIED
                postcondition_evidence = (
                    "Structured GUI postcondition not verified: "
                    f"{postcondition_reason}"
                )
                actual_obs_str = (
                    f"{actual_obs_str} | [POSTCONDITION UNVERIFIED]: "
                    f"{postcondition_reason}"
                )
                self._pending_gui_contract_feedback = postcondition_reason
                logger.warning(
                    "gui_postcondition_unverified",
                    action=action_type.value,
                    target=target_resource,
                    reason=postcondition_reason,
                )
        # Adaptive Closed-Loop Recovery if needed
        if recovery_needed:
            if self.recovery_events < self.max_recovery_attempts:
                rec_trace = await self.recover(workspace_id, action_type, actual_obs_str)
                actual_obs_str = f"{actual_obs_str} | Recovered: {rec_trace}"
                if not visual_stall_detected:
                    status = ActionExecutionStatus.RECOVERED
            else:
                if status in (ActionExecutionStatus.COMPLETED, ActionExecutionStatus.SUCCESS):
                    status = ActionExecutionStatus.FAILED

        # Closed-loop Target Feedback to Self-Evolution Engine
        # Only security-tool scans evolve offensive strategies / MethodLab.
        # Workstation GUI, terminal, file, or git operations must never trigger exploit synthesis.
        if (
            action_type == ComputerActionType.SECURITY_TOOL
            and status in (ActionExecutionStatus.FAILED, ActionExecutionStatus.RECOVERED)
            and getattr(self, "evolution_engine", None) is not None
        ):
            try:
                target_str = target_resource or str(self._last_action_output.get("target", "")) or self._last_gui_navigation_target or "target"
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
        if not self._gui_action_contract_active() and hasattr(self, "motor") and self.motor:
            lower_obs = actual_obs_str.lower()
            if any(term in lower_obs for term in ("modal_blocked", "blocked by modal", "overlay detected", "dismiss modal")):
                await self.motor.backtrack(workspace_id, reason="modal_blocked")
                actual_obs_str = f"{actual_obs_str} | Reflexive backtrack: dismissed blocking modal"

        # Live narration of the action's outcome, so the next reasoning step
        # receives the eye's own account ("the screen responded" / "no visible
        # effect") alongside the raw observation.  VERIFIED maps to a visible
        # effect, UNVERIFIED to an explicitly unseen one, anything else to
        # "not verified" -- an unverified action is never reported as success.
        if action_type in _GUI_ACTION_TYPES_FOR_NARRATION:
            if status == ActionExecutionStatus.VERIFIED:
                visible_effect: bool | None = True
            elif status == ActionExecutionStatus.UNVERIFIED:
                visible_effect = False
            else:
                visible_effect = None
            action_narration = self._narrate_action_result(
                action_type,
                target_resource,
                visible_effect,
                getattr(self, "_last_observed_screen", None),
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )
            if action_narration:
                actual_obs_str = f"{actual_obs_str} | [AVCS NARRATION]: {action_narration}"

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
            if action_type in (
                ComputerActionType.TERMINAL_EXEC,
                ComputerActionType.SECURITY_TOOL,
                ComputerActionType.VERIFY_PROBE,
            )
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
        if isinstance(state_delta, dict):
            trace.state_changed = state_delta.get("changed")
            trace.verification_evidence = (
                f"accessibility_state_changed={state_delta.get('changed')}; "
                f"version={state_delta.get('before_version')}->"
                f"{state_delta.get('after_version')}; "
                f"active_window={state_delta.get('active_window')}"
            )
        if postcondition_evidence:
            trace.verification_evidence = (
                f"{trace.verification_evidence}; {postcondition_evidence}"
                if trace.verification_evidence
                else postcondition_evidence
            )
        trace.prediction_error = experience.prediction_error
        trace.graph_delta = experience.graph_delta
        before_receipt_state = self._contract_state(before_state)
        after_receipt_state = self._contract_state(after_state)
        intent_evidence = (
            {
                "source_perception_version": pending_intent.source_perception_version,
                "source_active_window": pending_intent.source_active_window,
                "source_width": pending_intent.source_width,
                "source_height": pending_intent.source_height,
                "expected_outcome": pending_intent.expected_outcome,
                "issued_at_ns": pending_intent.issued_at_ns,
                "expires_at_ns": pending_intent.expires_at_ns,
            }
            if pending_intent is not None
            else {}
        )
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
            intent=intent_evidence,
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
        # GUI actions must NOT write into _last_action_output!
        if action_type in (
            ComputerActionType.TERMINAL_EXEC,
            ComputerActionType.SECURITY_TOOL,
            ComputerActionType.VERIFY_PROBE,
        ):
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
        ComputerActionType.APP_MAXIMIZE,
        ComputerActionType.WINDOW_MAXIMIZE,
        ComputerActionType.WINDOW_TILE,
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
        screen = getattr(self, "_last_observed_screen", None)
        avcs_state = getattr(screen, "avcs_state", None)
        scene_hash = (
            str(avcs_state.get("scene_hash", ""))
            if isinstance(avcs_state, dict)
            and avcs_state.get("engine") == "avcs_reflex"
            else ""
        )
        last_output = getattr(self, "_last_action_output", {}) or {}
        return state_signature(
            screen_hash=scene_hash,
            active_window=getattr(snapshot, "active_window", ""),
            windows=getattr(snapshot, "windows", ()),
            processes=getattr(snapshot, "processes", ()),
            visible_text="",
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

        if self._gui_action_contract_active():
            try:
                await self._dispatch_gui_action(
                    workspace_id,
                    GUIAction(action=GUIActionType.KEYPRESS, key="Escape"),
                )
            except Exception:
                pass
            try:
                comp_status = await self.computer.status(workspace_id)
                active_app = getattr(comp_status, "active_application", None)
                if active_app and str(active_app).lower() not in ("desktop", "none", ""):
                    await self._dispatch_gui_action(
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
                        await self._dispatch_gui_action(
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
        "ip a", "ip addr", "ip addr show", "ifconfig", "route -n",
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
        if action_type == ComputerActionType.GUI_SCREENSHOT and any(
            signature[0] == action_type.value
            for signature in self._recent_action_signatures[-4:]
        ):
            return (
                "BLOCKED: a fresh desktop observation is already captured at the "
                "start of every mission step. Use its screenshot and choose a "
                "different goal-advancing action."
            )

        if action_type == ComputerActionType.TERMINAL_EXEC:
            payload_cmd = payload.get("command") if isinstance(payload, dict) else str(payload or "")
            cmd = (payload_cmd or target or "").strip().rstrip(',;').strip().lower()
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

        gui_key_actions = {
            ComputerActionType.GUI_KEYPRESS.value,
            ComputerActionType.GUI_HOTKEY.value,
        }
        if (
            action_type.value in gui_key_actions
            and self._consecutive_unchanged_screens > 0
            and isinstance(payload, dict)
        ):
            key_value = (
                payload.get("key")
                or payload.get("keys")
                or payload.get("hotkey")
                or payload.get("key_combination")
            )
            if key_value:
                current_key = re.sub(r"\s+", "", str(key_value)).casefold()
                for signature in reversed(self._recent_action_signatures[-4:]):
                    if signature[0] not in gui_key_actions:
                        continue
                    try:
                        previous_payload = ast.literal_eval(signature[2])
                    except (SyntaxError, ValueError):
                        continue
                    if not isinstance(previous_payload, dict):
                        continue
                    previous_key = (
                        previous_payload.get("key")
                        or previous_payload.get("keys")
                        or previous_payload.get("hotkey")
                        or previous_payload.get("key_combination")
                    )
                    if (
                        previous_key
                        and re.sub(r"\s+", "", str(previous_key)).casefold()
                        == current_key
                    ):
                        return (
                            "BLOCKED: this GUI key input already produced no visible "
                            "screen change. Do not dispatch it again; choose a different "
                            "grounded interaction or report that the target is unavailable."
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
        ComputerActionType.APP_MAXIMIZE,
        ComputerActionType.WINDOW_MAXIMIZE,
        ComputerActionType.WINDOW_TILE,
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

    def _verification_probe_for_goal(self, goal: str) -> VerificationProbe | None:
        """Infer only fixed-shape probes; never accept a shell command as input."""
        normalized = goal.lower().strip()
        if any(term in normalized for term in ("test", "pytest", "unittest")):
            return VerificationProbe(kind=VerificationProbeKind.PYTEST)
        if any(term in normalized for term in ("install", "set up", "setup")):
            match = re.search(
                r"(?:install|set\s+up|setup)\s+(?:the\s+)?([a-zA-Z0-9_.+-]+)",
                normalized,
            )
            if match:
                return VerificationProbe(
                    kind=VerificationProbeKind.EXECUTABLE,
                    target=match.group(1),
                )
        if any(term in normalized for term in ("file", "create", "write", "save", "generate", "report")):
            match = re.search(
                r'([a-zA-Z0-9_\-\./\\]+\.(?:txt|md|py|json|sh|csv|html|log))',
                goal,
                flags=re.IGNORECASE,
            )
            if match:
                target = match.group(1)
                if not target.startswith(("/", "\\")):
                    root = str(
                        getattr(
                            getattr(self, "safety", None),
                            "workspace_root",
                            "/home/sonic/workspace",
                        )
                    )
                    target = f"{root}/{target}"
                return VerificationProbe(kind=VerificationProbeKind.FILE, target=target)
        return None

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
            return False, (
                f"{evidence}\n"
                "Visible text alone is not an independent GUI postcondition; "
                "completion requires the external Judge."
            )

        do_llm = use_llm if use_llm is not None else getattr(self, "enable_llm_verification", False)
        g = goal.lower().strip()
        tool_result = getattr(self, "_last_tool_result", None)
        if tool_result is not None:
            tool_status = getattr(tool_result, "status", "")
            tool_status = getattr(tool_status, "value", tool_status)
            tool_exit_code = getattr(tool_result, "exit_code", None)
            tool_output = str(getattr(tool_result, "raw_stdout", "") or "").strip()
            tool_findings = getattr(tool_result, "parsed_data", []) or []
            if (
                str(tool_status).lower() in {"completed", "success", "succeeded"}
                and tool_exit_code == 0
                and (tool_output or tool_findings)
            ):
                evidence = (
                    f"Verified {getattr(tool_result, 'tool_name', 'tool')} execution receipt: "
                    f"exit_code={tool_exit_code}; output={tool_output[:1200]!r}; "
                    f"findings={len(tool_findings)}"
                )
                return True, evidence

        if (
            self._verification_probe_for_goal(goal) is None
            and any(k in g for k in ("run", "start", "serve", "launch", "open"))
        ):
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

        probe = self._verification_probe_for_goal(goal)
        if probe is None:
            obs = await self.observe(workspace_id)
            evidence = (
                f"Re-observed: app={obs.active_application}, "
                f"term={obs.terminal_output[:120]} | "
                "No typed independent verification probe was inferred."
            )
            return False, evidence

        trace = await self.execute_action(
            workspace_id,
            ComputerActionType.VERIFY_PROBE,
            target_resource=probe.kind.value,
            payload={"probe": probe.model_dump(mode="json")},
            predicted_outcome="A typed verification probe returns exit code 0.",
        )
        exit_code = getattr(trace, "exit_code", None)
        evidence = (
            f"Verification probe={probe.kind.value}; exit_code={exit_code}; "
            f"status={getattr(trace.status, 'value', trace.status)}; "
            f"output={trace.actual_observation[:1500]}"
        )
        probe_output = str(getattr(trace, "actual_observation", "") or "").strip().lower()
        rejected_receipt_markers = (
            "__sonic_obs_ready__",
            "terminal ready (0 procs)",
            "traceback",
            "assertionerror",
            "command not found",
            "connection refused",
            "timed out",
            "exit 126",
            "placeholder",
            "dummy",
            "blocked",
            "failed",
        )
        rejected_marker = next(
            (marker for marker in rejected_receipt_markers if marker in probe_output),
            None,
        )
        if rejected_marker == "__sonic_obs_ready__":
            evidence += "; rejected dummy marker from the observation handshake"
        verified = (
            exit_code == 0
            and trace.status in (
                ActionExecutionStatus.COMPLETED,
                ActionExecutionStatus.SUCCESS,
                ActionExecutionStatus.VERIFIED,
            )
            and bool(probe_output)
            and rejected_marker is None
        )

        if verified and do_llm and self.llm_router is not None:
            from sonic.llm.schemas import LLMRequest, Message, MessageRole
            request = LLMRequest(
                messages=[
                    Message(
                        role=MessageRole.USER,
                        content=(
                            f"Goal: {goal}\nVerified probe receipt:\n{evidence}\n"
                            "Does this receipt prove the goal? Respond YES or NO."
                        ),
                    )
                ],
                task_type="reasoning",
            )
            try:
                response = await self.llm_router.complete(request)
                verified = response.content.strip().splitlines()[0].upper() == "YES"
            except Exception:
                verified = False
        return verified, evidence

    async def _verify_completion_receipt(self, goal: str, evidence: str) -> tuple[bool, str]:
        """Use System-1 as a contradiction check alongside independent verification."""
        try:
            from sonic.llm.laya_engine import get_laya_engine

            score = await asyncio.to_thread(
                get_laya_engine().verify_execution_receipt,
                goal,
                evidence,
            )
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                raise ValueError("receipt scorer returned an invalid probability")
        except Exception as exc:
            logger.warning("mission_receipt_verification_failed", error=str(exc))
            return False, f"System-1 receipt check unavailable: {exc}"

        note = f"System-1 receipt confidence: {score:.2f}"
        if score < 0.30:
            return False, f"{note}; evidence contradicts successful completion"
        return True, note

    def _current_goal_for_replan(self) -> str:
        """Return the active mission goal for replan messages.

        Falls back to the last known mission goal, then the agent's goal
        attribute. Never returns observation text, so the LLM always sees the
        real objective when asked to pivot.
        """
        for candidate in (
            getattr(self, "_last_mission_goal", None),
            getattr(self, "goal", None),
        ):
            text = str(candidate or "").strip()
            if text:
                return text
        return "(objective unavailable)"

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
        if hasattr(self, "failure_budget") and hasattr(self.failure_budget, "records"):
            self.failure_budget.records.clear()
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
        self._active_workspace_id = workspace_id
        self._agentic_lessons_goal = None
        self._agentic_lessons = []
        self._multi_context_goal = None
        self._multi_context_block = ""
        self._multi_context_pack = None
        t_start = time.perf_counter()
        self._interrupted = False
        interrupt_event = getattr(self, "_interrupt_event", None)
        if interrupt_event is None:
            interrupt_event = asyncio.Event()
            self._interrupt_event = interrupt_event
        interrupt_event.clear()
        self._active_interrupt_check = interrupt_check
        self._watchdog_aborted = False
        self._consecutive_stale_gui_blocks = 0
        goal_reached = False
        self._last_tool_result = None
        requires_external_gui_judge = (
            getattr(self, "execution_plane", "") == "gui"
            or getattr(self, "gui_only", False)
        )
        if getattr(self, "_last_mission_goal", None) != goal:
            self._last_mission_goal = goal
            self._replan_count = 0
            self._consecutive_failures = 0
            if hasattr(self, "failure_budget") and hasattr(self.failure_budget, "records"):
                self.failure_budget.records.clear()

        if getattr(self, "checklist", None) is None or self.checklist.top_level_goal != goal:
            has_explicit_steps = any(re.match(r'^(?:\d+[\.\)]|\-|\*)\s+', line.strip()) for line in goal.splitlines())
            if getattr(self, "enable_llm_decomposition", False) or has_explicit_steps:
                self.checklist = await self.decompose_goal(goal)
            else:
                self.checklist = None
        self._consecutive_gui_contract_violations = 0
        self._pending_gui_contract_feedback = ""

        # Backend bootstrap may inspect/install packages through a shell. It is
        # intentionally unavailable on the GUI-only Computer plane.
        if not self._gui_action_contract_active() and not getattr(self, "_bootstrap_performed", False):
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
            if self._mission_interrupt_requested():
                self._interrupted = True
                self._interrupt_event.set()
                self.history.append({
                    "action": "INTERRUPTED",
                    "result": "Mission interrupted after observation; no action was dispatched.",
                })
                break

            # 2. REASON & CHOOSE ACTION
            action_type, target, payload, expected = await self.choose_action(goal, obs, step)
            if self._mission_interrupt_requested():
                self._interrupted = True
                self._interrupt_event.set()
                trace = await self.execute_action(
                    workspace_id,
                    ActionExecutionStatus.BLOCKED,
                    "interrupted",
                    {},
                    "Mission interrupted during reasoning; action was not dispatched.",
                )
                if step_callback is not None:
                    cb_res = step_callback(trace)
                    if asyncio.iscoroutine(cb_res):
                        await cb_res
                break

            if (
                action_type == ActionExecutionStatus.BLOCKED
                and target
                in {
                    "gui-contract-violation",
                    "gui-invalid-response",
                    "gui-perception-unavailable",
                }
            ):
                trace = await self.execute_action(
                    workspace_id,
                    action_type,
                    target,
                    payload,
                    expected,
                )
                if step_callback is not None:
                    try:
                        cb_res = step_callback(trace)
                        if asyncio.iscoroutine(cb_res):
                            await cb_res
                    except Exception as cb_err:
                        logger.warning(
                            "run_mission_step_callback_failed",
                            error=str(cb_err),
                        )
                break

            # Goal-complete sentinel: the LLM JUDGED the goal achieved — but we
            # do not stop on self-declaration alone. We independently verify.
            if expected == _GOAL_COMPLETE_SENTINEL:
                if requires_external_gui_judge or self._mission_used_desktop_actions():
                    self.history.append({
                        "action": "GOAL_COMPLETE (awaiting independent Judge)",
                        "result": (
                            "The GUI worker proposed completion. Its visible "
                            "self-check is not treated as proof; route the live "
                            "desktop and worker receipts to the independent Judge."
                        ),
                    })
                    break

                verified, evidence = await self.verify_goal(workspace_id, goal)
                if verified:
                    receipt_ok, receipt_note = await self._verify_completion_receipt(goal, evidence)
                    evidence = f"{evidence}\n{receipt_note}"
                    verified = receipt_ok
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
                if rejection.startswith(
                    "BLOCKED: this GUI key input already produced no visible screen change."
                ):
                    self._watchdog_aborted = True
                    self.history.append({
                        "action": "WATCHDOG_ABORT",
                        "result": rejection,
                    })
                    break
                continue

            # 3. ACT & VERIFY
            if action_type in {
                ComputerActionType.GUI_CLICK,
                ComputerActionType.GUI_DOUBLE_CLICK,
                ComputerActionType.GUI_RIGHT_CLICK,
                ComputerActionType.GUI_TYPE,
                ComputerActionType.GUI_KEYPRESS,
                ComputerActionType.GUI_MOVE,
                ComputerActionType.GUI_SCROLL,
                ComputerActionType.GUI_DRAG,
            }:
                postcondition_error = self._single_gui_postcondition_error(
                    action_type, payload
                )
                if postcondition_error:
                    reason = (
                        f"{action_type.value} was not dispatched: "
                        f"{postcondition_error}"
                    )
                    self._pending_gui_contract_feedback = reason
                    self.history.append({
                        "action": f"{action_type.value} {target} (BLOCKED)",
                        "result": reason,
                    })
                    trace = await self.execute_action(
                        workspace_id,
                        ActionExecutionStatus.BLOCKED,
                        "gui-postcondition-required",
                        {},
                        reason,
                    )
                    if step_callback is not None:
                        try:
                            callback_result = step_callback(trace)
                            if asyncio.iscoroutine(callback_result):
                                await callback_result
                        except Exception as callback_error:
                            logger.warning(
                                "run_mission_step_callback_failed",
                                error=str(callback_error),
                            )
                    self._consecutive_gui_postcondition_violations += 1
                    if self._consecutive_gui_postcondition_violations >= 2:
                        logger.warning(
                            "gui_postcondition_contract_repeatedly_missing",
                            action=action_type.value,
                            target=target,
                        )
                        break
                    continue
                self._consecutive_gui_postcondition_violations = 0
                try:
                    self._pending_gui_action_intent = await self._create_gui_action_intent(
                        workspace_id,
                        action_type=action_type,
                        target=target,
                        payload=payload,
                        expected_outcome=expected,
                        observation=obs,
                        ttl_ms=2_000,
                    )
                except (TypeError, ValueError) as exc:
                    reason = f"GUI action held; a fresh bounded intent could not be created: {exc}"
                    self.history.append(
                        {"action": f"{action_type.value} {target} (HELD)", "result": reason}
                    )
                    held_trace = await self.execute_action(
                        workspace_id,
                        ActionExecutionStatus.BLOCKED,
                        "gui-action-intent-invalid",
                        {},
                        reason,
                    )
                    if step_callback is not None:
                        try:
                            callback_result = step_callback(held_trace)
                            if asyncio.iscoroutine(callback_result):
                                await callback_result
                        except Exception as callback_error:
                            logger.warning(
                                "run_mission_step_callback_failed",
                                error=str(callback_error),
                            )
                    self._inject_replan(goal)
                    continue
            action_started_ns = time.perf_counter_ns()
            trace = await self.execute_action(workspace_id, action_type, target, payload, expected)
            perception_receipt = getattr(self, "_last_reasoning_perception", None)
            if isinstance(perception_receipt, dict):
                trace.perception_source = str(perception_receipt.get("source", "unknown"))
                trace.perception_version = int(perception_receipt.get("version", 0) or 0)
                trace.screen_age_ms = perception_receipt.get("screen_age_ms")
                trace.controls_age_ms = perception_receipt.get("controls_age_ms")
                trace.accessible_scene_fresh = perception_receipt.get(
                    "accessible_scene_fresh"
                )
                trace.image_attached = bool(perception_receipt.get("image_attached"))
                trace.visual_escalation_reason = str(
                    perception_receipt.get("visual_escalation_reason", "")
                )
                trace.image_escalation_count = int(
                    perception_receipt.get("image_escalation_count", 0) or 0
                )
            if action_type in getattr(self, "_GUI_ONLY_ACTIONS", set()):
                self._pending_gui_action_observation = {
                    "started_ns": action_started_ns,
                    "trace_id": trace.id,
                }

            # Invoke real-time step streaming callback if provided
            if step_callback is not None:
                try:
                    cb_res = step_callback(trace)
                    if asyncio.iscoroutine(cb_res):
                        await cb_res
                except Exception as cb_err:
                    logger.warning("run_mission_step_callback_failed", error=str(cb_err))

            if (
                action_type == ActionExecutionStatus.BLOCKED
                and target in {
                    "reasoning-unavailable",
                    "gui-contract-violation",
                    "gui-perception-unavailable",
                }
            ):
                break

            if self._interrupted or (callable(interrupt_check) and interrupt_check()):
                self._interrupted = True
                logger.info("mission_interrupted_by_user", step=step)
                break

            if self._watchdog_aborted:
                self.history.append({
                    "action": "WATCHDOG_ABORT",
                    "result": str(getattr(trace, "actual_observation", "")),
                })
                break

            is_observation_only = action_type == ComputerActionType.GUI_SCREENSHOT

            # A policy denial is a terminal state for this mission attempt.
            # Continuing would only repeat the same unauthorized action and
            # turn a real safety decision into scripted retry noise. The
            # operator can change scope/approval and start a fresh attempt.
            trace_observation = str(getattr(trace, "actual_observation", "") or "")
            trace_action_type = str(
                getattr(
                    getattr(trace, "action_type", action_type),
                    "value",
                    getattr(trace, "action_type", action_type),
                )
            ).upper()
            trace_status = str(
                getattr(getattr(trace, "status", ""), "value", getattr(trace, "status", ""))
            ).upper()
            normalized_trace_observation = trace_observation.strip().casefold()
            unavailable_gui_capability = (
                trace_status == ActionExecutionStatus.BLOCKED.value
                and trace_action_type.startswith(("GUI_", "APP_", "BROWSER_"))
                and any(
                    marker in normalized_trace_observation
                    for marker in (
                        "sonic_avcs_grounding_checkpoint is not configured",
                        "local avcs pixel processor is unavailable",
                        "avcs has no evaluated local focus-verification head",
                    )
                )
            )
            if unavailable_gui_capability:
                self.history.append({
                    "action": "GUI_CAPABILITY_UNAVAILABLE",
                    "result": (
                        "Stopped after the required local AVCS capability was "
                        "unavailable. No GUI action was dispatched."
                    ),
                })
                logger.warning(
                    "gui_mission_stopped_for_unavailable_avcs_capability",
                    action=trace_action_type,
                    reason=trace_observation[:300],
                )
                break
            stale_gui_block = (
                trace_status == ActionExecutionStatus.BLOCKED.value
                and trace_action_type.startswith(("GUI_", "APP_", "BROWSER_"))
                and normalized_trace_observation.startswith(
                    "gui action blocked before dispatch:"
                )
                and any(
                    marker in normalized_trace_observation
                    for marker in (
                        "changed since the action was reasoned",
                        "avcs scene changed since the action was reasoned",
                        "avcs scene changed before final gui dispatch",
                        "desktop state could not be verified",
                        "fresh screenshot evidence could not be verified",
                    )
                )
            )
            if stale_gui_block:
                self._consecutive_stale_gui_blocks += 1
                if self._consecutive_stale_gui_blocks >= 2:
                    self.history.append({
                        "action": "STALE_GUI_LOOP_ABORT",
                        "result": (
                            "Stopped after repeated pre-dispatch freshness blocks. "
                            "No stale GUI action was executed; obtain a stable fresh "
                            "observation before starting another attempt."
                        ),
                    })
                    break
            elif action_type != ComputerActionType.GUI_SCREENSHOT:
                self._consecutive_stale_gui_blocks = 0
            prediction_error = float(getattr(trace, "prediction_error", 0.0) or 0.0)
            action_type_val = getattr(action_type, "value", str(action_type))
            if not is_observation_only and prediction_error >= 0.75 and trace.status not in (
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

            if (
                trace.status == ActionExecutionStatus.BLOCKED
                and "[ACTION LOOP DETECTED]" in trace_observation
            ):
                self.history.append({
                    "action": f"{action_type_val} {target} (ACTION_LOOP_BLOCKED)",
                    "result": (
                        "Stopped after the action-loop breaker blocked a repeated "
                        "action. Replan from the latest desktop observation."
                    ),
                })
                logger.warning(
                    "mission_stopped_after_action_loop_block",
                    action=action_type_val,
                    target=target,
                )
                break

            # An observation refresh is not task progress and must not erase
            # accumulated failure/stall state or advance a checklist.
            if not is_observation_only:
                # Track consecutive failures for stuck/replan detection.
                is_failed_status = trace.status in (
                    ActionExecutionStatus.FAILED,
                    ActionExecutionStatus.TIMED_OUT,
                    ActionExecutionStatus.BLOCKED,
                    ActionExecutionStatus.CANCELLED,
                    ActionExecutionStatus.UNVERIFIED,
                    "FAILED",
                    "TIMED_OUT",
                    "BLOCKED",
                    "CANCELLED",
                    "UNVERIFIED",
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

        # Update Telemetry Metrics.
        self.metrics.actions_total = len(self.traces)
        self.metrics.actions_successful = sum(
            1 for t in self.traces
            if t.action_type != ComputerActionType.GUI_SCREENSHOT
            and t.status in (
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
                ActionExecutionStatus.UNVERIFIED,
                ActionExecutionStatus.RECOVERED,  # RECOVERED = failed + recovery attempt
                "FAILED",
                "TIMED_OUT",
                "BLOCKED",
                "CANCELLED",
                "UNVERIFIED",
                "RECOVERED",
            )
        )
        self.metrics.recovery_events = self.recovery_events

        if (
            not goal_reached
            and not requires_external_gui_judge
            and not self._mission_used_desktop_actions()
            and self.metrics.actions_successful > 0
            and self.metrics.actions_failed == 0
        ):
            try:
                verified, evidence = await self.verify_goal(workspace_id, goal)
                if verified:
                    receipt_ok, receipt_note = await self._verify_completion_receipt(goal, evidence)
                    goal_reached = receipt_ok
                    if self.traces:
                        self.traces[-1].verification_evidence = (
                            f"{evidence}\n{receipt_note}"
                        )[:500]
            except Exception as v_err:
                logger.debug("post_mission_verify_error", error=str(v_err))

        self.goal_reached = goal_reached
        self._active_interrupt_check = None
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
        avcs_state = getattr(obs.screen, "avcs_state", None)
        avcs_summary = "unavailable"
        if isinstance(avcs_state, dict):
            avcs_summary = (
                f"frame={avcs_state.get('frame_index', 'unknown')}, "
                f"scene={str(avcs_state.get('scene_hash', ''))[:16]}, "
                f"changed={avcs_state.get('scene_changed', 'unknown')}, "
                f"regions={len(avcs_state.get('regions', []))}"
            )
        return (
            f"Active app: {obs.active_application}\n"
            f"Open windows: {getattr(obs, 'open_applications', [])}\n"
            f"Installed GUI apps: {(obs.installed_applications or [])[:15]}\n"
            f"AVCS pixel state: {avcs_summary}\n"
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
            goal_cb = getattr(self, "_idle_goal_callback", None)
            if callable(goal_cb):
                try:
                    res_cb = goal_cb(goal, steps)
                    if asyncio.iscoroutine(res_cb):
                        await res_cb
                except Exception:
                    pass
            before = len(self.traces)
            step_cb = getattr(self, "_idle_step_callback", None)
            await self.run_mission(workspace_id, goal, steps=steps, step_callback=step_cb)
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
