"""Data models for the Boss Agent + SubAgents orchestration system.

The Boss Agent is a strategic orchestrator that decomposes objectives into
phases, dispatches focused sub-missions to SubAgents (ComputerUseAgent workers),
collects their outputs, aggregates findings, and plans next phases until the
objective is complete.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str = "boss") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class SubMission(BaseModel):
    """A focused task assigned to a SubAgent by the Boss."""

    id: str = Field(default_factory=lambda: _new_id("sub"))
    goal: str
    max_steps: int = 5
    priority: int = 1  # Higher = execute first within a phase
    depends_on: list[str] = Field(default_factory=list)
    plane: str = "headless"  # "headless" (terminal/files/toolsmith) | "gui" (desktop/browser)
    process_mode: str = "deep"
    evidence_context: dict[str, Any] = Field(default_factory=dict)
    status: str = "PENDING"  # PENDING -> RUNNING -> COMPLETED -> FAILED

    @model_validator(mode="after")
    def _infer_plane_from_goal(self) -> SubMission:
        # If plane was explicitly specified (and not "auto"), preserve it unconditionally.
        if "plane" in self.model_fields_set and self.plane:
            plane_lower = str(self.plane).lower().strip()
            if plane_lower in ("gui", "headless"):
                self.plane = plane_lower
                return self

        goal_lower = self.goal.lower()

        # Explicit GUI indicators: tasks that genuinely require X11 display / browser interaction
        # Strictly target-agnostic and binary-free (complies with AGENTS.md Rule 1)
        gui_indicators = (
            "open browser", "launch browser", "browser navigation", "browser click",
            "browser window", "browser tab", "web browser", "browser",
            "visually verify", "visual verification", "visual inspection", "visually inspect",
            "visual completion", "gui_click", "gui_type", "gui_action", "gui_screenshot",
            "app_launch", "launch app", "click icon", "click button", "click on",
            "desktop application", "graphical window", "graphical interface", "graphical application",
            "desktop screen", "desktop view", "active window", "display :99", ":99",
        )

        # Headless indicators: tasks executed directly in the container sandbox without GUI
        headless_indicators = (
            "terminal", "bash", "shell", "mkdir", "script", "python", "probe",
            "file", "curl", "sha256", "grep", "ps ", "ss ", "netstat", "wmctrl",
            "directory", "tool_author", "toolsmith", "audit", "find ", "cat ",
            "server", "service", "author", "write", "endpoint", "api", "pip ",
            "background", "listen", "port", "compile", "clone", "git ",
        )

        # If goal clearly specifies a headless execution task and does not explicitly require browser/GUI interaction
        has_headless = any(h in goal_lower for h in headless_indicators)
        has_gui = any(g in goal_lower for g in gui_indicators)

        if has_gui and not has_headless:
            self.plane = "gui"
        elif has_headless and not has_gui:
            self.plane = "headless"
        elif has_headless and has_gui:
            # Ambiguous / compound: if it mentions browser or visual verification, check if primary intent is GUI
            if any(b in goal_lower for b in (
                "open browser", "launch browser", "visually verify",
                "visual verification", "visual inspection", "visual completion",
            )):
                self.plane = "gui"
            else:
                self.plane = "headless"
        elif any(
            w in goal_lower for w in ("browser", "gui_", "desktop application", "graphical window", "visual")
        ):
            self.plane = "gui"
        else:
            self.plane = "headless"

        return self


class SubMissionResult(BaseModel):
    """Output collected from a completed SubAgent."""

    sub_mission_id: str
    goal: str
    traces: list[Any] = Field(default_factory=list)
    findings_summary: str = ""
    success: bool = False
    key_discoveries: list[str] = Field(default_factory=list)
    actions_taken: int = 0
    duration_seconds: float = 0.0
    evidence_verified: bool = False
    judge_verdict: str = "UNVERIFIED"  # CONFIRMED | FALSIFIED | BENIGN | INCONCLUSIVE | UNVERIFIED
    judge_notes: str = ""
    status: str = "FAILED"  # SUCCESS | FAILED | TIMED_OUT | BLOCKED


class Phase(BaseModel):
    """A batch of SubMissions dispatched together by the Boss."""

    phase_number: int
    name: str = ""
    thinking: str = ""  # Boss's strategic reasoning for creating this phase
    sub_missions: list[SubMission] = Field(default_factory=list)
    results: list[SubMissionResult] = Field(default_factory=list)
    summary: str = ""  # Synthesized intelligence summary from this phase
    status: str = "PENDING"  # PENDING -> RUNNING -> COMPLETED
    started_at: str = ""
    completed_at: str = ""


class BossThinking(BaseModel):
    """A recorded thinking entry from the Boss Agent."""

    id: str = Field(default_factory=lambda: _new_id("think"))
    phase: int
    thinking_type: str  # strategic_decomposition | findings_analysis | reaction_planning | completion_check | final_synthesis
    content: str
    timestamp: str = Field(default_factory=_now)
    duration_seconds: float = 0.0


class BossReport(BaseModel):
    """Final structured report from the Boss Agent after orchestration completes."""

    objective: str
    status: str = "INCOMPLETE"  # COMPLETE | INCOMPLETE | PARTIAL
    phases: list[Phase] = Field(default_factory=list)
    thinking_log: list[BossThinking] = Field(default_factory=list)
    all_traces: list[Any] = Field(default_factory=list)
    total_sub_agents: int = 0
    total_actions: int = 0
    total_phases: int = 0
    findings_summary: str = ""
    duration_seconds: float = 0.0
    replay_metrics: dict[str, Any] = Field(default_factory=dict)
    replay_quality_gate_passed: bool = False
