"""
SONIC — Autonomous Computer-Using Engineer Models (Phase 14)
===================================================================
Data models for closed-loop computer world observations, action plans,
autonomy levels, engineering mission modes, decision traces, and metrics.
"""

from __future__ import annotations

import re
import shlex
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from sonic.computer.models import ScreenObservation


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str = "trace") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ============================================
# Enums
# ============================================

class ActionExecutionStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    COMPLETED = "COMPLETED"  # alias/eq with SUCCEEDED / SUCCESS
    SUCCESS = "SUCCESS"  # alias/eq with SUCCEEDED / COMPLETED
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    RECOVERED = "RECOVERED"
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"

    def __eq__(self, other: object) -> bool:
        if super().__eq__(other):
            return True
        aliases = {"COMPLETED", "SUCCESS", "SUCCEEDED"}
        self_val = getattr(self, "value", str(self))
        other_val = getattr(other, "value", str(other)) if (hasattr(other, "value") or isinstance(other, str)) else None
        if self_val in aliases and other_val in aliases:
            return True
        return False

    __hash__ = StrEnum.__hash__


class FailureClassification(StrEnum):
    TARGET_FAILURE = "TARGET_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    SANDBOX_FAILURE = "SANDBOX_FAILURE"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    PERMISSION_FAILURE = "PERMISSION_FAILURE"
    POLICY_BLOCK = "POLICY_BLOCK"
    TIMEOUT = "TIMEOUT"
    INVALID_COMMAND = "INVALID_COMMAND"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class StrategyState(StrEnum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    EXHAUSTED = "EXHAUSTED"
    ABANDONED = "ABANDONED"


class FailureRecord(BaseModel):
    """Record of an individual tool, provider, or environment failure."""
    tool: str
    provider: str
    error_class: FailureClassification
    environment: str = "sandbox"
    timestamp: str = Field(default_factory=_now)
    count: int = 1
    raw_error: str = ""


class SubGoalStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SubGoal(BaseModel):
    """Discrete, verifiable sequential milestone within a top-level mission."""
    id: str = Field(default_factory=lambda: f"sg-{uuid.uuid4().hex[:6]}")
    description: str
    verification_criteria: str = ""
    status: SubGoalStatus = SubGoalStatus.PENDING
    evidence: str = ""
    attempt_count: int = 0


class SubGoalChecklist(BaseModel):
    """Structured progression checklist tracking active vs completed mission sub-goals."""
    top_level_goal: str
    sub_goals: list[SubGoal] = Field(default_factory=list)
    active_index: int = 0

    def active_sub_goal(self) -> SubGoal | None:
        if 0 <= self.active_index < len(self.sub_goals):
            return self.sub_goals[self.active_index]
        return None

    def advance(self) -> bool:
        if self.active_index < len(self.sub_goals) - 1:
            self.active_index += 1
            if self.sub_goals[self.active_index].status == SubGoalStatus.PENDING:
                self.sub_goals[self.active_index].status = SubGoalStatus.IN_PROGRESS
            return True
        return False

    def mark_active_completed(self, evidence: str = "") -> bool:
        sg = self.active_sub_goal()
        if sg:
            sg.status = SubGoalStatus.COMPLETED
            if evidence:
                sg.evidence = evidence
            self.advance()
            return True
        return False

    def is_all_completed(self) -> bool:
        if not self.sub_goals:
            return False
        return all(sg.status in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED) for sg in self.sub_goals)

    def render_prompt_markdown(self) -> str:
        if not self.sub_goals:
            return "(no decomposed sub-goals)"
        lines = ["Execution Checklist:"]
        for i, sg in enumerate(self.sub_goals):
            if sg.status == SubGoalStatus.COMPLETED:
                mark = "[x]"
            elif i == self.active_index:
                mark = "[>]"
            elif sg.status == SubGoalStatus.FAILED:
                mark = "[!]"
            else:
                mark = "[ ]"
            status_str = f" ({sg.status.value})" if sg.status != SubGoalStatus.PENDING else ""
            lines.append(f"  {mark} Sub-Goal {i+1}: {sg.description}{status_str}")
            if sg.evidence:
                lines.append(f"      Evidence: {sg.evidence[:80]}")
        return "\n".join(lines)


class ComputerAutonomyLevel(StrEnum):
    L0_MANUAL = "L0_MANUAL"                          # Human executes
    L1_ASSISTED = "L1_ASSISTED"                      # SONIC recommends actions
    L2_SUPERVISED_AUTONOMOUS = "L2_SUPERVISED_AUTONOMOUS"  # SONIC executes with approval for high-risk
    L3_AUTONOMOUS = "L3_AUTONOMOUS"                  # SONIC executes complete mission


class EngineeringMissionMode(StrEnum):
    ENGINEERING_MODE = "ENGINEERING_MODE"            # Source inspection, debug, edit, test, commit
    SECURITY_RESEARCH_MODE = "SECURITY_RESEARCH_MODE"  # Recon, assessment, browser, evidence
    DEBUG_MODE = "DEBUG_MODE"                        # Focused bug reproduction, hypothesis testing
    GENERAL_ENGINEERING_MODE = "GENERAL_ENGINEERING_MODE"  # DevOps, QA, Cloud, Scripting


class ComputerActionType(StrEnum):
    GUI_CLICK = "GUI_CLICK"
    GUI_DOUBLE_CLICK = "GUI_DOUBLE_CLICK"
    GUI_RIGHT_CLICK = "GUI_RIGHT_CLICK"
    GUI_TYPE = "GUI_TYPE"
    GUI_KEYPRESS = "GUI_KEYPRESS"
    GUI_HOTKEY = "GUI_HOTKEY"
    GUI_MOVE = "GUI_MOVE"
    GUI_SCROLL = "GUI_SCROLL"
    GUI_SCREENSHOT = "GUI_SCREENSHOT"
    GUI_DRAG = "GUI_DRAG"
    GUI_WAIT = "GUI_WAIT"
    TERMINAL_EXEC = "TERMINAL_EXEC"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    APP_LAUNCH = "APP_LAUNCH"
    APP_CLOSE = "APP_CLOSE"
    APP_FOCUS = "APP_FOCUS"
    APP_MAXIMIZE = "APP_MAXIMIZE"
    WINDOW_MAXIMIZE = "WINDOW_MAXIMIZE"
    WINDOW_TILE = "WINDOW_TILE"
    APP_INSTALL = "APP_INSTALL"
    SERVICE_ACTION = "SERVICE_ACTION"
    GIT_STATUS = "GIT_STATUS"
    GIT_DIFF = "GIT_DIFF"
    GIT_BRANCH = "GIT_BRANCH"
    GIT_COMMIT = "GIT_COMMIT"
    GIT_PUSH = "GIT_PUSH"
    # Unified computer-use: browser actions share the same reasoning loop.
    BROWSER_NAVIGATE = "BROWSER_NAVIGATE"
    BROWSER_CLICK = "BROWSER_CLICK"
    BROWSER_TYPE = "BROWSER_TYPE"
    BROWSER_SELECT = "BROWSER_SELECT"
    BROWSER_KEYPRESS = "BROWSER_KEYPRESS"
    BROWSER_SCROLL = "BROWSER_SCROLL"
    BROWSER_SCREENSHOT = "BROWSER_SCREENSHOT"
    BROWSER_WAIT = "BROWSER_WAIT"
    BROWSER_DOWNLOAD = "BROWSER_DOWNLOAD"
    WAIT_FOR_HUMAN = "WAIT_FOR_HUMAN"
    SECURITY_TOOL = "SECURITY_TOOL"
    TOOL_AUTHOR = "TOOL_AUTHOR"
    TOOL_RUN = "TOOL_RUN"
    METHOD_INVENT = "METHOD_INVENT"
    DEFINE_SUBAGENT = "DEFINE_SUBAGENT"
    INVOKE_SUBAGENT = "INVOKE_SUBAGENT"
    SEND_MESSAGE = "SEND_MESSAGE"
    MANAGE_SUBAGENTS = "MANAGE_SUBAGENTS"
    VERIFY_PROBE = "VERIFY_PROBE"
    GOAL_COMPLETE = "GOAL_COMPLETE"


class VerificationProbeKind(StrEnum):
    PYTEST = "pytest"
    EXECUTABLE = "executable"
    FILE = "file"


class VerificationProbe(BaseModel):
    """A constrained verification request compiled to a fixed shell command."""

    kind: VerificationProbeKind
    target: str = ""

    @model_validator(mode="after")
    def validate_target(self) -> VerificationProbe:
        if self.kind == VerificationProbeKind.PYTEST:
            if self.target:
                raise ValueError("pytest probes do not accept a target")
        elif self.kind == VerificationProbeKind.EXECUTABLE:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", self.target):
                raise ValueError("executable probe target must be a simple binary name")
        elif self.kind == VerificationProbeKind.FILE:
            if not self.target or "\0" in self.target:
                raise ValueError("file probe requires a valid path")
        return self

    def to_command(self) -> str:
        if self.kind == VerificationProbeKind.PYTEST:
            return "python -m pytest -q --tb=short"
        if self.kind == VerificationProbeKind.EXECUTABLE:
            return f"command -v -- {shlex.quote(self.target)}"
        return (
            f"test -f {shlex.quote(self.target)} && "
            f"test -s {shlex.quote(self.target)} && "
            "printf '%s\\n' FILE_VERIFIED"
        )


# ============================================
# Models
# ============================================

class ComputerWorldObservation(BaseModel):
    """Unified multi-modal state observation of the computer environment."""
    screen: ScreenObservation = Field(default_factory=ScreenObservation)
    active_application: str = "Desktop"
    windows: list[str] = Field(default_factory=lambda: ["Desktop", "Terminal"])
    visible_text: str = ""
    filesystem_files: list[str] = Field(default_factory=list)
    installed_applications: list[str] = Field(default_factory=list)
    processes: list[str] = Field(default_factory=list)
    terminal_output: str = ""
    working_directory: str = "/home/sonic/workspace"
    browser_state: dict[str, Any] = Field(default_factory=lambda: {"url": "about:blank", "title": "New Tab"})
    ide_state: dict[str, Any] = Field(default_factory=lambda: {"active_file": "None", "cursor_line": 1})
    git_branch: str = "main"
    git_clean: bool = True
    perception_version: int = 0
    perception_latency_ns: int = 0
    observation_age_ms: float | None = Field(default=None, ge=0)
    observation_received_monotonic_ns: int | None = Field(default=None, exclude=True, ge=0)
    timestamp: str = Field(default_factory=_now)

    @property
    def desktop_state(self) -> str:
        return getattr(self.screen, "desktop_state", "")

    @property
    def screenshot_base64(self) -> str:
        return getattr(self.screen, "screenshot_base64", "")

    @property
    def active_window_geometry(self) -> tuple[int, int, int, int] | None:
        return getattr(self.screen, "active_window_geometry", None)

    @property
    def active_window_geom(self) -> tuple[int, int, int, int] | None:
        return getattr(self.screen, "active_window_geometry", None)



class ComputerActionPlan(BaseModel):
    """A closed-loop action plan translating mission goals into computer steps."""
    id: str = Field(default_factory=lambda: _new_id("plan"))
    mission_id: str
    objective: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    expected_outcomes: list[str] = Field(default_factory=list)
    fallback_actions: list[dict[str, Any]] = Field(default_factory=list)
    verification: str = "verify_tests_pass"
    created_at: str = Field(default_factory=_now)


class ComputerDecisionTrace(BaseModel):
    """Immutable record of an individual closed-loop action decision."""
    id: str = Field(default_factory=lambda: _new_id("trace"))
    action_id: str = Field(default_factory=lambda: _new_id("act"))
    step_index: int = 1
    action_type: ComputerActionType | ActionExecutionStatus | str = ComputerActionType.TERMINAL_EXEC
    target_resource: str
    payload: str | None = None
    predicted_outcome: str
    actual_observation: str
    expected_observation: str = ""
    info_gain: float = 1.0
    recovery_attempted: bool = False
    thought: str = ""
    status: ActionExecutionStatus | str = ActionExecutionStatus.COMPLETED

    duration_seconds: float = 0.0
    thought_duration_seconds: float = 0.0
    exit_code: int | None = None
    verification_evidence: str = ""
    state_changed: bool | None = None
    prediction_error: float | None = None
    graph_delta: str = ""
    # Executor-produced contract evidence.  This is populated by
    # ComputerUseAgent from live before/after observations; it is never used
    # to authorize or dispatch an action.
    action_receipt: Any | None = None
    timestamp: str = Field(default_factory=_now)

    @field_validator("action_type", mode="before")
    @classmethod
    def _coerce_action_type(cls, v: Any) -> Any:
        if isinstance(v, ActionExecutionStatus) or (
            isinstance(v, str) and v.strip().upper() in ("BLOCKED", "ACTIONEXECUTIONSTATUS.BLOCKED")
        ):
            return ComputerActionType.TERMINAL_EXEC
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in ComputerActionType._value2member_map_:
                return ComputerActionType(v_clean)
        return v

    @property
    def observation(self) -> str:
        return self.actual_observation

    @property
    def receipt(self) -> Any | None:
        """Compatibility accessor for executor-produced contract evidence."""
        return self.action_receipt

    @property
    def state_delta(self) -> Any | None:
        """Return the truthful before/after delta, when execution completed."""
        return getattr(self.action_receipt, "state_delta", None)


class AgentStepResult(ComputerDecisionTrace):
    """Result of an individual agent step execution.

    Can be unpacked as a 2-tuple (success, message), accessed via attributes
    (.success, .message), or used as a ComputerDecisionTrace.
    """
    success: bool = False
    message: str = ""

    def __init__(self, success: bool = False, message: str = "", **kwargs: Any):
        status = kwargs.pop("status", ActionExecutionStatus.COMPLETED if success else ActionExecutionStatus.FAILED)
        actual_obs = kwargs.pop("actual_observation", message)
        exit_code = kwargs.pop("exit_code", 0 if success else 1)
        target = kwargs.pop("target_resource", "")
        predicted = kwargs.pop("predicted_outcome", "")
        super().__init__(
            success=success,
            message=message,
            status=status,
            actual_observation=actual_obs,
            exit_code=exit_code,
            target_resource=target,
            predicted_outcome=predicted,
            action_type=ComputerActionType.TERMINAL_EXEC,
            prediction_error=0.0 if success else 1.0,
            info_gain=0.0,
            recovery_attempted=False,
            **kwargs,
        )

    def __iter__(self):
        yield self.success
        yield self.message

    def __getitem__(self, idx: int) -> Any:
        return (self.success, self.message)[idx]

    def __bool__(self) -> bool:
        return self.success

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return (self.success, self.message) == other
        if isinstance(other, AgentStepResult):
            return (self.success, self.message) == (other.success, other.message)
        return super().__eq__(other)


class ComputerUseMetrics(BaseModel):
    """Telemetry and performance metrics for a computer-use mission."""
    actions_total: int = 0
    actions_successful: int = 0
    actions_failed: int = 0
    recovery_events: int = 0
    unnecessary_actions: int = 0
    time_to_completion_seconds: float = 0.0
    verification_score: float = 0.0
    human_comparison_score: float = 0.0
