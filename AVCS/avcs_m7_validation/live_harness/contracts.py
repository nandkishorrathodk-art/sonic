"""Contracts and data models for AVCS Milestone 8 Phase D Live Application Matrix.

Defines the typed interfaces, identity structures, and execution trace schemas
for real Windows process interaction across all 12 validation scenarios.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from avcs_m7_validation.contracts import (
    ActionStatus,
    AgentTask,
    EvidenceProvenance,
    Mode,
    OracleResult,
    OutcomeCategory,
)


@dataclass(frozen=True)
class LiveAppIdentity:
    """Authentic process and window identity of a running live Windows application."""
    pid: int
    hwnd: int
    title: str
    executable_path: str
    window_bounds: Tuple[int, int, int, int]  # x, y, width, height in physical screen pixels
    dpi_scale: float
    session_id: int
    desktop_name: str
    source_commit: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "hwnd": self.hwnd,
            "window_title": self.title,
            "executable_path": self.executable_path,
            "window_bounds": list(self.window_bounds),
            "dpi_scale": self.dpi_scale,
            "session_id": self.session_id,
            "desktop_name": self.desktop_name,
            "source_commit": self.source_commit,
        }


@dataclass(frozen=True)
class LiveExecutionTrace:
    """Empirical telemetry collected during a live Windows interaction trial."""
    run_id: str
    scenario_id: str
    mode: Mode
    seed: int
    app_identity: LiveAppIdentity
    pre_screenshot_path: str
    pre_screenshot_sha256: str
    post_screenshot_path: str
    post_screenshot_sha256: str
    visual_pixel_diff_ratio: float
    target_coord_screen: Tuple[int, int]
    dispatched_action: str
    capture_latency_ms: float
    action_latency_ms: float
    settle_latency_ms: float
    oracle_latency_ms: float
    oracle_result: OracleResult
    status: ActionStatus
    verified_success: bool
    false_confirmation: bool
    cleanup_successful: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["status"] = self.status.value
        d["app_identity"] = self.app_identity.to_dict()
        d["oracle_result"] = asdict(self.oracle_result)
        return d


class BaseLiveHarness(ABC):
    """Abstract base class for all 12 live Windows scenario harnesses."""

    @property
    @abstractmethod
    def scenario_id(self) -> str:
        """The canonical scenario identifier matching catalog.py."""

    @property
    @abstractmethod
    def target_property(self) -> str:
        """The target property evaluated by the independent oracle."""

    @abstractmethod
    def launch(self, workspace_dir: Path, seed: int) -> LiveAppIdentity:
        """Launches the controlled live application process and returns its verified identity."""

    @abstractmethod
    def get_agent_task(self, seed: int) -> AgentTask:
        """Returns the public AgentTask prompt visible to the agent."""

    @abstractmethod
    def get_target_screen_coords(self, seed: int) -> Tuple[int, int]:
        """Calculates expected target screen pixel coordinates (x, y) for physical dispatch."""

    @abstractmethod
    def dispatch_action(
        self,
        driver: Any,
        seed: int,
        target_coords: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Dispatches physical hardware inputs (clicks, drags, keyboard) to the live window."""

    @abstractmethod
    def evaluate_oracle(
        self,
        workspace_dir: Path,
        action_dispatched: bool,
        seed: int,
    ) -> OracleResult:
        """Independently verifies postcondition state from disk or process state files."""

    @abstractmethod
    def teardown(self) -> bool:
        """Cleanly terminates the application process and releases all desktop handles."""

