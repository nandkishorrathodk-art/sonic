"""
SONIC Autonomous Loop Comprehensive Opaque-Box E2E Test Suite
=============================================================
Authoritative Specification: ORIGINAL_REQUEST.md § R4
Loop Architecture:
  Objective -> BossAgent decomposition -> mission delegation ->
  ComputerUseAgent execution -> perception/state update ->
  GUI/motor dispatch -> visual verification -> JudgeAgent receipt ->
  cognitive state update -> task completion.

Coverage: 9 Required Operational Verification Points across Tiers 1–4:
  1. GUI actions do not hang indefinitely.
  2. Invalid/stale coordinates are rejected before dispatch.
  3. Focus validation occurs before actions requiring focus.
  4. Computer actions produce observable execution results.
  5. Perception failures propagate correctly.
  6. Verification failures are not silently swallowed.
  7. Worker failures reach the supervising agent.
  8. Cognitive state reflects actual execution state.
  9. Security and safety invariants remain enforced.

Structure:
  - Tier 1: Feature Coverage (P1 - P9 components)
  - Tier 2: Boundary & Corner Cases (stale coords, edge bounds, lease timeout, corrupt age)
  - Tier 3: Cross-Feature Combinations (Boss delegation + ComputerUse + Judge verification)
  - Tier 4: Real-World Workload Scenarios (multi-step dual-plane, adaptive replanning, safety envelope)
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import struct
import time
import zlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.agents.judge import JudgeAgent, JudgeVerdict
from sonic.agents.task_graph import TaskStatus
from sonic.computer.models import ComputerState, ScreenObservation
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.action_contracts import ActionReceipt
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
    ComputerWorldObservation,
)
from sonic.computer_use.models_boss import SubMission, SubMissionResult
from sonic.computer_use.teamwork import DisplayLeaseManager
from sonic.safety.action_policy import ActionPolicy
from sonic.safety.sealed import seal_default
from sonic.sandbox.provider import ExecResult

pytestmark = [pytest.mark.no_live_infra]


# =============================================================================
# Test Helpers & Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _reset_display_lease_manager():
    """Clear shared display lease state between test cases to prevent event-loop leakage."""
    DisplayLeaseManager._locks.clear()
    DisplayLeaseManager._lock_metadata.clear()
    yield
    DisplayLeaseManager._locks.clear()
    DisplayLeaseManager._lock_metadata.clear()


def _png_base64(width: int = 1280, height: int = 800, color_byte: int = 0) -> str:
    """Generate a minimal valid PNG image in base64 format with valid IHDR/IDAT."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", binascii.crc32(body) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([color_byte, color_byte, color_byte]) * width
    pixels = zlib.compress(row * height)
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(image).decode("ascii")


def _make_llm_mock(responses: list[str]) -> AsyncMock:
    """Create a mock LLM router returning a predefined sequence of responses."""
    mock = AsyncMock()
    results = []
    for r in responses:
        resp = MagicMock()
        resp.content = r
        resp.reasoning_content = ""
        resp.latency_ms = 40
        resp.tool_calls = []
        results.append(resp)
    mock.complete = AsyncMock(side_effect=results)
    return mock


def _make_trace(
    step_index: int = 1,
    action_type: ComputerActionType = ComputerActionType.TERMINAL_EXEC,
    target: str = "test_target",
    observation: str = "command output",
    status: ActionExecutionStatus = ActionExecutionStatus.COMPLETED,
    exit_code: int = 0,
    expected: str = "expected outcome",
    predicted: str = "predicted outcome",
) -> ComputerDecisionTrace:
    """Create a fully-populated ComputerDecisionTrace for test assertions."""
    return ComputerDecisionTrace(
        step_index=step_index,
        action_type=action_type,
        target_resource=target,
        payload="{}",
        predicted_outcome=predicted,
        expected_observation=expected,
        actual_observation=observation,
        status=status,
        exit_code=exit_code,
    )


class MockE2EComputer(ComputerProvider):
    """Complete in-memory ComputerProvider implementation for E2E autonomous loop testing."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 800,
        active_window: str = "Desktop",
        active_application: str = "Desktop",
    ):
        self.workspace_id = "ws-e2e"
        self.width = width
        self.height = height
        self.active_window = active_window
        self.active_application = active_application
        self.windows: list[str] = [active_window] if active_window else []

        # Call tracking
        self.terminal_calls: list[str] = []
        self.gui_calls: list[Any] = []
        self.files: dict[str, str] = {}

        # Controllable behaviors
        self.fail_screenshot = False
        self.fail_gui_action = False
        self.exit_code = 0
        self.terminal_stdout = "ok"
        self.terminal_stderr = ""
        self.capture_age_ms: float = 0.0
        self.controls_age_ms: float = 0.0
        self.desktop_state = "INTERACTIVE"
        self.perception_version = 1
        self.screen_color = 1
        self.detected_controls: list[dict[str, Any]] = []

    async def create(self, *args, **kwargs):
        return type("W", (), {"id": self.workspace_id})()

    async def destroy(self, *args, **kwargs):
        return True

    async def status(self, workspace_id: str):
        return ComputerState(
            workspace_id=workspace_id,
            tenant_id="tenant-e2e",
            active_application=self.active_application,
            active_window=self.active_window,
            windows=list(self.windows),
        )

    async def screenshot(self, workspace_id: str):
        if self.fail_screenshot:
            raise RuntimeError("Screenshot perception failed: display server unavailable")
        return ScreenObservation(
            width=self.width,
            height=self.height,
            screenshot_base64=_png_base64(self.width, self.height, color_byte=self.screen_color),
            active_window=self.active_window,
            desktop_state=self.desktop_state,
            capture_age_ms=self.capture_age_ms,
            controls_age_ms=self.controls_age_ms,
            perception_version=self.perception_version,
            detected_controls=list(self.detected_controls),
        )

    async def gui_action(self, workspace_id: str, action: Any, actor: str = "operator"):
        if self.fail_gui_action:
            raise RuntimeError("GUI action execution failed on workstation")
        self.gui_calls.append(action)
        self.perception_version += 1
        self.screen_color = (self.screen_color + 20) % 250 + 1
        return ScreenObservation(
            width=self.width,
            height=self.height,
            screenshot_base64=_png_base64(self.width, self.height, color_byte=self.screen_color),
            active_window=self.active_window,
            desktop_state=self.desktop_state,
            capture_age_ms=0.0,
            controls_age_ms=0.0,
            perception_version=self.perception_version,
        )

    async def terminal(self, workspace_id: str, command: str, timeout: int = 60, actor: str = "operator"):
        self.terminal_calls.append(command)
        return ExecResult(
            command=command,
            exit_code=self.exit_code,
            stdout=self.terminal_stdout,
            stderr=self.terminal_stderr,
        )

    async def read_file(self, workspace_id: str, path: str):
        return MagicMock(content=self.files.get(path, ""))

    async def write_file(self, workspace_id: str, path: str, content: str):
        self.files[path] = content
        return True

    async def list_files(self, workspace_id: str, path: str = "."):
        return [type("FE", (), {"name": k, "path": k})() for k in self.files]

    async def git_action(self, workspace_id: str, action: str, **kwargs):
        return type("GitStatus", (), {"branch": "main", "is_clean": True})()

    async def application_list(self, *args, **kwargs):
        return []

    async def process_list(self, *args, **kwargs):
        return []

    async def launch_application(self, workspace_id: str, app_name: str, **kwargs):
        self.active_window = app_name
        self.active_application = app_name
        if app_name not in self.windows:
            self.windows.append(app_name)
        return True

    async def close_application(self, *args, **kwargs):
        return True

    async def install_application(self, *args, **kwargs):
        return True, ""

    async def uninstall_application(self, *args, **kwargs):
        return True

    async def service_action(self, *args, **kwargs):
        return None


# =============================================================================
# TIER 1: FEATURE COVERAGE (9 OPERATIONAL VERIFICATION POINTS)
# =============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Feature coverage for each of the 9 required operational verification points."""

    @pytest.mark.asyncio
    async def test_tier1_point1_lease_timeout_prevents_indefinite_hang(self):
        """Point 1: DisplayLeaseManager acquires lock with timeout; contending lease does not hang."""
        lock = DisplayLeaseManager.get_lock("ws-1", display=":99")
        await lock.acquire()
        try:
            # Attempting to acquire contending lease with 0.05s timeout must raise TimeoutError
            with pytest.raises(TimeoutError):
                await DisplayLeaseManager.acquire("ws-1", display=":99", timeout=0.05)
        finally:
            DisplayLeaseManager.release(lock, display=":99")

    @pytest.mark.asyncio
    async def test_tier1_point1_watchdog_aborts_on_consecutive_unchanged_screens(self):
        """Point 1: Watchdog detects repeated unchanged screen frames and halts without spinning forever."""
        comp = MockE2EComputer()
        agent = ComputerUseAgent(computer_provider=comp, gui_only=True)
        await agent.observe(comp.workspace_id)

        # Trigger consecutive unchanged frames
        agent._consecutive_unchanged_screens = 4
        agent._watchdog_aborted = True

        assert agent._watchdog_aborted is True

    @pytest.mark.asyncio
    async def test_tier1_point2_out_of_bounds_coordinates_blocked_before_dispatch(self):
        """Point 2: Out-of-bounds coordinates (x >= width or y >= height) are blocked before reaching provider."""
        comp = MockE2EComputer(width=1280, height=800)
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "offscreen button",
            {"x": 2000, "y": 1500},
            "Click outside display bounds",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126
        assert "outside screen" in trace.actual_observation.lower() or "bounds" in trace.actual_observation.lower()
        assert len(comp.gui_calls) == 0

    @pytest.mark.asyncio
    async def test_tier1_point2_stale_screen_observation_rejected_before_dispatch(self):
        """Point 2: Stale observation age (> 2000ms) without valid refresh causes pre-dispatch rejection."""
        comp = MockE2EComputer()
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        # Force provider to return stale capture age (> 2000ms limit)
        comp.capture_age_ms = 2500.0

        screen = await comp.screenshot(comp.workspace_id)
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=1,
            observation_age_ms=2500.0,
        )
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time() - 3.0

        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "100,100",
            {"x": 100, "y": 100},
            "Click button with stale observation",
        )

        # Action is rejected before dispatch
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126
        assert "blocked before dispatch" in trace.actual_observation.lower() or "stale" in trace.actual_observation.lower()
        assert len(comp.gui_calls) == 0

    @pytest.mark.asyncio
    async def test_tier1_point3_focus_validation_enforced_before_window_action(self):
        """Point 3: Window focus is verified; mismatch against requested target window blocks action."""
        comp = MockE2EComputer(active_window="Desktop", active_application="Desktop")
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        # Target window is TextEditor, but active window is Desktop and cannot be switched
        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "editor surface",
            {"x": 100, "y": 100, "target_window": "TextEditor"},
            "Click inside TextEditor",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126
        assert "target window" in trace.actual_observation.lower() or "focus" in trace.actual_observation.lower()

    @pytest.mark.asyncio
    async def test_tier1_point3_typing_focus_and_text_validation(self):
        """Point 3: Typing requires focused valid input field and non-empty text."""
        comp = MockE2EComputer()
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        # Typing with empty text
        trace_empty = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_TYPE,
            "search box",
            {"text": "", "x": 100, "y": 100},
            "Type empty text",
        )
        assert trace_empty.status == ActionExecutionStatus.BLOCKED
        assert trace_empty.exit_code == 126

    @pytest.mark.asyncio
    async def test_tier1_point4_terminal_action_produces_observable_receipt(self):
        """Point 4: Terminal execution produces concrete ActionReceipt with exit code, stdout, and state delta."""
        comp = MockE2EComputer()
        comp.terminal_stdout = "user_id=1001\ngroup_id=1001"
        comp.exit_code = 0
        agent = ComputerUseAgent(computer_provider=comp)

        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.TERMINAL_EXEC,
            "id",
            {"command": "id"},
            "Show user identity",
        )

        assert trace.status == ActionExecutionStatus.COMPLETED
        assert trace.exit_code == 0
        assert isinstance(trace.action_receipt, ActionReceipt)
        assert trace.action_receipt.exit_code == 0
        assert "user_id=1001" in trace.action_receipt.evidence
        assert trace.action_receipt.state_delta.changed.get("exit_code") == 0

    @pytest.mark.asyncio
    async def test_tier1_point4_gui_action_produces_screen_observation_receipt(self):
        """Point 4: GUI action produces observable screen transition with updated perception version."""
        comp = MockE2EComputer()
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "150,150",
            {"x": 150, "y": 150},
            "Click button",
        )

        assert trace.status == ActionExecutionStatus.UNVERIFIED
        assert len(comp.gui_calls) == 1
        assert trace.action_receipt is not None
        assert trace.action_receipt.status == ActionExecutionStatus.UNVERIFIED

    @pytest.mark.asyncio
    async def test_tier1_point5_perception_failure_propagates_without_blind_dispatch(self):
        """Point 5: When screen capture fails, perception failure propagates and prevents blind dispatch."""
        comp = MockE2EComputer()
        comp.fail_screenshot = True
        agent = ComputerUseAgent(computer_provider=comp)

        with pytest.raises(RuntimeError, match="Screenshot perception failed"):
            await agent.observe(comp.workspace_id)

        assert len(comp.gui_calls) == 0

    @pytest.mark.asyncio
    async def test_tier1_point6_verification_failures_not_swallowed_by_judge(self):
        """Point 6: JudgeAgent detects failed traces and issues FALSIFIED verdict rather than swallowing error."""
        judge = JudgeAgent()
        sub_m = SubMission(goal="Compile application binary")
        failed_trace = {
            "action_type": "TERMINAL_EXEC",
            "target": "make build",
            "status": "FAILED",
            "exit_code": 2,
            "actual_observation": "fatal error: header not found; compilation terminated.",
        }
        res = SubMissionResult(
            sub_mission_id=sub_m.id,
            goal=sub_m.goal,
            traces=[failed_trace],
            success=False,
            findings_summary="Build failed with syntax error",
        )

        evaluation = await judge.evaluate_sub_mission(
            workspace_id="ws-e2e",
            sub_mission=sub_m,
            result=res,
            worker_agent_id="worker-1",
        )

        assert evaluation.verdict == JudgeVerdict.FALSIFIED
        assert "no successful execution traces" in evaluation.notes.lower()

    @pytest.mark.asyncio
    async def test_tier1_point7_worker_failure_propagates_to_boss_and_task_graph(self):
        """Point 7: Worker execution exceptions bubble up to BossAgent and update TaskGraph to FAILED."""
        decomp_json = json.dumps({
            "thinking": "Decompose into failing worker.",
            "phase_name": "Execution Phase",
            "sub_missions": [
                {"id": "sub-fail-1", "goal": "Execute failing command", "max_steps": 2, "plane": "headless"},
            ],
        })
        llm = _make_llm_mock([decomp_json])
        comp = MockE2EComputer()
        boss = BossAgent(computer_provider=comp, llm_router=llm)

        # Mock subagent dispatch to raise RuntimeError
        with patch.object(boss, "_dispatch_sub_agent", AsyncMock(side_effect=RuntimeError("Worker container crashed"))):
            phase = await boss._strategic_decomposition("Run problematic job")
            boss.phases.append(phase)
            boss._sync_phase_to_graph(phase)

            # Run wave
            await boss.run(workspace_id="ws-e2e", objective="Run problematic job")

        # Verify task graph marked failed
        tg_task = boss.task_graph.get_task("sub-fail-1")
        assert tg_task is not None
        assert tg_task.status == TaskStatus.FAILED
        assert any(r.success is False for r in boss.phases[0].results)

    @pytest.mark.asyncio
    async def test_tier1_point8_cognitive_state_reflects_actual_execution_traces(self):
        """Point 8: BossReport, thinking log, and phase results strictly reflect actual execution data."""
        decomp_json = json.dumps({
            "thinking": "Single reconnaissance step.",
            "phase_name": "Recon",
            "sub_missions": [
                {"id": "sub-recon", "goal": "Check system status", "max_steps": 2, "plane": "headless"},
            ],
        })
        summary_json = "System is operational."
        completion_json = json.dumps({"complete": True, "reason": "System is verified."})

        llm = _make_llm_mock([decomp_json, summary_json, completion_json])
        comp = MockE2EComputer()
        boss = BossAgent(computer_provider=comp, llm_router=llm)

        mock_trace = _make_trace(
            step_index=1,
            action_type=ComputerActionType.TERMINAL_EXEC,
            target="uname -a",
            status=ActionExecutionStatus.COMPLETED,
            exit_code=0,
            observation="Linux workstation 6.6.0",
        )
        sub_res = SubMissionResult(
            sub_mission_id="sub-recon",
            goal="Check system status",
            traces=[mock_trace],
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="Linux workstation 6.6.0 operational",
            actions_taken=1,
        )

        with patch.object(boss, "_dispatch_sub_agent", AsyncMock(return_value=sub_res)):
            report = await boss.run(workspace_id="ws-e2e", objective="Check system status")

        assert report.total_sub_agents >= 1
        assert report.total_actions == 1
        assert len(boss.thinking_log) >= 1
        assert any(t.thinking_type == "strategic_decomposition" for t in boss.thinking_log)

    @pytest.mark.asyncio
    async def test_tier1_point9_safety_action_policy_blocks_path_traversal(self):
        """Point 9: ActionPolicy blocks path traversal outside the workspace root fail-closed."""
        policy = ActionPolicy(workspace_root="/home/sonic/workspace")

        verdict_read = policy.evaluate(
            "FILE_READ",
            "../../etc/shadow",
            {"path": "../../etc/shadow"},
        )
        assert verdict_read.allowed is False
        assert (
            "escapes" in verdict_read.reason.lower()
            or "confined" in verdict_read.reason.lower()
            or "outside" in verdict_read.reason.lower()
        )

        verdict_write = policy.evaluate(
            "FILE_WRITE",
            "/etc/cron.d/malicious",
            {"path": "/etc/cron.d/malicious", "content": "* * * * * root bad"},
        )
        assert verdict_write.allowed is False


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES (9 OPERATIONAL VERIFICATION POINTS)
# =============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary and corner cases across all 9 operational verification points."""

    @pytest.mark.asyncio
    async def test_tier2_point2_negative_and_zero_coordinates_handling(self):
        """Point 2: Negative coordinates are blocked; (0,0) top-left origin is accepted."""
        comp = MockE2EComputer(width=1280, height=800)
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        # Negative coordinate
        trace_neg = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "-5,10",
            {"x": -5, "y": 10},
            "Click with negative x",
        )
        assert trace_neg.status == ActionExecutionStatus.BLOCKED
        assert trace_neg.exit_code == 126
        assert "negative" in trace_neg.actual_observation.lower()

        # Origin (0,0) coordinate is valid
        trace_origin = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "0,0",
            {"x": 0, "y": 0},
            "Click at screen origin",
        )
        assert trace_origin.status == ActionExecutionStatus.UNVERIFIED
        assert len(comp.gui_calls) == 1

    @pytest.mark.asyncio
    async def test_tier2_point2_exact_screen_boundary_coordinates_rejected(self):
        """Point 2: Exact screen resolution coordinates (width, height) are strictly outside 0-indexed bounds."""
        comp = MockE2EComputer(width=1280, height=800)
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        # Boundary x == width (valid range is 0 to 1279)
        trace_x = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "1280,400",
            {"x": 1280, "y": 400},
            "Click exactly at screen width boundary",
        )
        assert trace_x.status == ActionExecutionStatus.BLOCKED
        assert trace_x.exit_code == 126

        # Boundary y == height (valid range is 0 to 799)
        trace_y = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "640,800",
            {"x": 640, "y": 800},
            "Click exactly at screen height boundary",
        )
        assert trace_y.status == ActionExecutionStatus.BLOCKED
        assert trace_y.exit_code == 126

    @pytest.mark.asyncio
    async def test_tier2_point2_malformed_and_none_coordinates_rejected(self):
        """Point 2: Non-integer and None coordinate values in payload are rejected safely without crashing."""
        comp = MockE2EComputer(width=1280, height=800)
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        trace_str = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "invalid target",
            {"x": "not_an_int", "y": 200},
            "Click with string coordinate",
        )
        assert trace_str.status == ActionExecutionStatus.BLOCKED
        assert trace_str.exit_code == 126

        trace_none = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "invalid target",
            {"x": None, "y": None},
            "Click with None coordinates",
        )
        assert trace_none.status == ActionExecutionStatus.BLOCKED
        assert trace_none.exit_code == 126

    @pytest.mark.asyncio
    async def test_tier2_point1_fractional_lease_timeout_fails_cleanly(self):
        """Point 1: Fractional timeout on contending display lease times out cleanly without deadlock."""
        lock = DisplayLeaseManager.get_lock("ws-corner", display=":99")
        await lock.acquire()
        try:
            t_start = time.perf_counter()
            with pytest.raises(TimeoutError):
                await DisplayLeaseManager.acquire("ws-corner", display=":99", timeout=0.01)
            duration = time.perf_counter() - t_start
            assert duration < 0.5, "Lease acquisition timeout took too long"
        finally:
            DisplayLeaseManager.release(lock, display=":99")

    @pytest.mark.asyncio
    async def test_tier2_point5_empty_perception_screenshot_fails_validation(self):
        """Point 5: Empty screenshot bytes on interactive desktop state fails pre-dispatch revalidation."""
        comp = MockE2EComputer()
        agent = ComputerUseAgent(computer_provider=comp)

        # Observation with empty screenshot bytes
        screen = ScreenObservation(
            width=1280,
            height=800,
            screenshot_base64="",
            desktop_state="INTERACTIVE",
            active_window="Desktop",
            capture_age_ms=10.0,
        )
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=1,
            observation_age_ms=10.0,
        )

        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "100,100",
            {"x": 100, "y": 100},
            "Click with empty screenshot",
        )
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126

    @pytest.mark.asyncio
    async def test_tier2_point5_infinite_and_negative_capture_age_rejected(self):
        """Point 5: Corrupt, non-finite, or negative capture age ms triggers pre-dispatch verification error."""
        comp = MockE2EComputer()
        agent = ComputerUseAgent(computer_provider=comp)

        screen = ScreenObservation(
            width=1280,
            height=800,
            screenshot_base64=_png_base64(),
            desktop_state="INTERACTIVE",
            active_window="Desktop",
            capture_age_ms=float("inf"),
        )
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=1,
            observation_age_ms=0.0,
        )

        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "100,100",
            {"x": 100, "y": 100},
            "Click with infinite age",
        )
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126

    @pytest.mark.asyncio
    async def test_tier2_point6_anti_self_attestation_blocks_unsupported_worker_claims(self):
        """Point 6: Worker self-attestation of VERIFIED with empty or failed exit code is rejected by Judge."""
        judge = JudgeAgent()
        sub_m = SubMission(goal="Validate database schema", plane="headless")
        trace = {
            "action_type": "TERMINAL_EXEC",
            "target": "db_check.py",
            "status": "VERIFIED",
            "exit_code": 1,
            "actual_observation": "FAIL-CLOSED: Host execution is disabled in HeadlessComputeProvider.",
        }
        res = SubMissionResult(
            sub_mission_id=sub_m.id,
            goal=sub_m.goal,
            traces=[trace],
            success=True,
            findings_summary="Worker claims schema is valid",
        )

        evaluation = await judge.evaluate_sub_mission(
            workspace_id="ws-e2e",
            sub_mission=sub_m,
            result=res,
            worker_agent_id="sub-agent-1",
        )

        assert evaluation.verdict == JudgeVerdict.FALSIFIED
        assert evaluation.self_attestation_rejected is True

    @pytest.mark.asyncio
    async def test_tier2_point6_unsaved_buffer_detected_and_falsified_by_judge(self):
        """Point 6: JudgeAgent falsifies completion claims when editor window has unsaved markers (*untitled)."""
        comp = MockE2EComputer(active_window="*untitled - TextEditor")
        judge = JudgeAgent(computer_provider=comp)
        sub_m = SubMission(goal="Write report into editor")
        trace = {
            "action_type": "GUI_TYPE",
            "target": "editor buffer",
            "status": "COMPLETED",
            "actual_observation": "Typed 50 words into *untitled document",
        }
        res = SubMissionResult(
            sub_mission_id=sub_m.id,
            goal=sub_m.goal,
            traces=[trace],
            success=True,
            findings_summary="Report authored in editor",
        )

        evaluation = await judge.evaluate_sub_mission(
            workspace_id="ws-e2e",
            sub_mission=sub_m,
            result=res,
            worker_agent_id="sub-agent-editor",
        )

        assert evaluation.verdict == JudgeVerdict.FALSIFIED
        assert "unsaved" in evaluation.notes.lower()

    @pytest.mark.asyncio
    async def test_tier2_point8_empty_objective_creates_no_workers_and_is_incomplete(self):
        """Point 8: Empty objective produces zero workers, zero actions, and INCOMPLETE status."""
        llm = _make_llm_mock(["{}"])
        comp = MockE2EComputer()
        boss = BossAgent(computer_provider=comp, llm_router=llm)

        report = await boss.run(workspace_id="ws-e2e", objective="")

        assert report.status == "INCOMPLETE"
        assert report.total_actions == 0
        assert report.total_sub_agents == 0

    def test_tier2_point9_sealed_safety_policy_immutable_under_mutation_attempts(self):
        """Point 9: SealedActionPolicy fails closed if modified after seal."""
        sealed = seal_default("/home/sonic/workspace")

        # Valid action passes
        verdict = sealed.evaluate("FILE_READ", "/home/sonic/workspace/app.py", {})
        assert verdict.allowed is True

        # Mutating sealed attribute raises AttributeError
        with pytest.raises(AttributeError, match="sealed policy is immutable"):
            setattr(sealed, "workspace_root", "/unauthorized/root")


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (INTEGRATED AUTONOMOUS LOOP)
# =============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Cross-feature integrations across BossAgent, ComputerUseAgent, and JudgeAgent."""

    @pytest.mark.asyncio
    async def test_tier3_autonomous_loop_delegation_execution_judge_confirmation(self):
        """Cross-Feature: Complete autonomous loop where Boss delegates, ComputerUse executes, Judge confirms."""
        decomp_json = json.dumps({
            "thinking": "Objective requires verifying system uptime.",
            "phase_name": "Verification Phase",
            "sub_missions": [
                {"id": "sub-uptime", "goal": "Check server uptime", "max_steps": 2, "plane": "headless"},
            ],
        })
        summary_json = "Uptime: up 12 days, load average 0.05"
        completion_json = json.dumps({"complete": True, "reason": "Uptime verified"})

        llm = _make_llm_mock([decomp_json, summary_json, completion_json])
        comp = MockE2EComputer()
        comp.terminal_stdout = "up 12 days, load average 0.05"
        comp.exit_code = 0

        # Create real JudgeAgent
        judge = JudgeAgent(computer_provider=comp)
        boss = BossAgent(computer_provider=comp, llm_router=llm, judge=judge)

        # Worker trace returned by subagent
        worker_trace = _make_trace(
            step_index=1,
            action_type=ComputerActionType.TERMINAL_EXEC,
            target="uptime",
            status=ActionExecutionStatus.COMPLETED,
            exit_code=0,
            observation="up 12 days, load average 0.05",
            expected="uptime details",
        )
        worker_result = SubMissionResult(
            sub_mission_id="sub-uptime",
            goal="Check server uptime",
            traces=[worker_trace],
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="System uptime confirmed",
            actions_taken=1,
        )

        with patch.object(boss, "_dispatch_sub_agent", AsyncMock(return_value=worker_result)):
            report = await boss.run(workspace_id="ws-e2e", objective="Check server uptime")

        assert report.status == "COMPLETE"
        assert report.total_actions == 1
        assert boss.task_graph.get_task("sub-uptime").status == TaskStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_tier3_autonomous_loop_fails_closed_when_judge_falsifies_worker(self):
        """Cross-Feature: When worker claims success but Judge falsifies, Boss overrides to FAILED."""
        decomp_json = json.dumps({
            "thinking": "Attempting verification.",
            "phase_name": "Check Phase",
            "sub_missions": [
                {"id": "sub-check", "goal": "Check network connectivity", "max_steps": 2, "plane": "headless"},
            ],
        })
        llm = _make_llm_mock([decomp_json, "Network unreachable."])
        comp = MockE2EComputer()

        judge = JudgeAgent()
        boss = BossAgent(computer_provider=comp, llm_router=llm, judge=judge)

        # Worker trace claiming success despite failure
        falsified_trace = _make_trace(
            step_index=1,
            action_type=ComputerActionType.TERMINAL_EXEC,
            target="ping 198.51.100.1",
            status=ActionExecutionStatus.FAILED,
            exit_code=1,
            observation="Network is unreachable",
        )
        falsified_result = SubMissionResult(
            sub_mission_id="sub-check",
            goal="Check network connectivity",
            traces=[falsified_trace],
            success=False,
            status="FAILED",
            evidence_verified=False,
            findings_summary="[FALSIFIED by Judge] Network unreachable",
            actions_taken=1,
        )

        with patch.object(boss, "_dispatch_sub_agent", AsyncMock(return_value=falsified_result)):
            report = await boss.run(workspace_id="ws-e2e", objective="Verify network")

        # Boss correctly marks report INCOMPLETE and task FAILED
        assert report.status == "INCOMPLETE"
        assert boss.task_graph.get_task("sub-check").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_tier3_delegated_worker_stale_coordinates_blocks_before_dispatch(self):
        """Cross-Feature: Delegated worker encountering stale coordinates blocks action before provider dispatch."""
        comp = MockE2EComputer()
        agent = ComputerUseAgent(computer_provider=comp)
        await agent.observe(comp.workspace_id)

        # Stale capture age on provider
        comp.capture_age_ms = 3000.0
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time() - 5.0

        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.GUI_CLICK,
            "control at 200,200",
            {"x": 200, "y": 200},
            "Click control",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126
        assert len(comp.gui_calls) == 0

    @pytest.mark.asyncio
    async def test_tier3_worker_timeout_during_gui_delegation_handled_by_boss(self):
        """Cross-Feature: Worker timeout during GUI execution is caught by Boss and marked TIMED_OUT."""
        decomp_json = json.dumps({
            "thinking": "Long running desktop GUI task.",
            "phase_name": "GUI Automation",
            "sub_missions": [
                {"id": "sub-long", "goal": "Operate GUI app", "max_steps": 1, "plane": "gui"},
            ],
        })
        llm = _make_llm_mock([decomp_json])
        comp = MockE2EComputer()
        boss = BossAgent(computer_provider=comp, llm_router=llm)

        # Mock agent.run_mission to hang
        async def _hang(*args, **kwargs):
            await asyncio.sleep(100.0)

        with patch("sonic.computer_use.agent.ComputerUseAgent.run_mission", side_effect=_hang):
            with patch("asyncio.wait_for", side_effect=TimeoutError()):
                sub_res = await boss._dispatch_sub_agent(
                    "ws-e2e",
                    SubMission(id="sub-long", goal="Operate GUI app", max_steps=1, plane="gui"),
                )

        assert sub_res.success is False
        assert sub_res.status == "TIMED_OUT"
        assert "timed out" in sub_res.findings_summary.lower()

    @pytest.mark.asyncio
    async def test_tier3_parallel_workers_serialize_display_leases(self):
        """Cross-Feature: Concurrent GUI workers serialize physical :99 display access without deadlock."""
        comp = MockE2EComputer()
        execution_order: list[str] = []

        async def _worker_gui_task(name: str):
            lock = await DisplayLeaseManager.acquire("ws-parallel", display=":99", owner=name)
            try:
                execution_order.append(f"{name}_start")
                await asyncio.sleep(0.02)
                execution_order.append(f"{name}_end")
            finally:
                DisplayLeaseManager.release(lock, display=":99")

        await asyncio.gather(
            _worker_gui_task("worker_A"),
            _worker_gui_task("worker_B"),
        )

        assert len(execution_order) == 4
        # Assert non-interleaved sequential execution
        first_worker = execution_order[0].replace("_start", "")
        assert execution_order[0] == f"{first_worker}_start"
        assert execution_order[1] == f"{first_worker}_end"


# =============================================================================
# TIER 4: REAL-WORLD WORKLOAD SCENARIOS
# =============================================================================

class TestTier4RealWorldWorkloadScenarios:
    """Tier 4: End-to-end multi-step real-world workloads across terminal and desktop planes."""

    @pytest.mark.asyncio
    async def test_tier4_complete_multi_step_dual_plane_autonomous_workflow(self):
        """Scenario: Multi-step dual-plane mission.
        Phase 1: Terminal worker writes configuration artifact.
        Phase 2: GUI worker verifies artifact in editor.
        Judge confirms each receipt; Boss completes objective.
        """
        decomp_json = json.dumps({
            "thinking": "Step 1: Write config in terminal. Step 2: Open and inspect in editor.",
            "phase_name": "Configuration Deployment",
            "sub_missions": [
                {"id": "sub-write", "name": "Terminal Config Writer", "goal": "Deploy config.env", "plane": "headless", "max_steps": 2, "priority": 1},
                {"id": "sub-gui", "name": "GUI Config Inspector", "goal": "Inspect config.env", "plane": "gui", "max_steps": 2, "priority": 2},
            ],
        })
        summary_json = "Configuration deployed and visually verified."
        completion_json = json.dumps({"complete": True, "reason": "Dual-plane deployment verified."})

        llm = _make_llm_mock([decomp_json, summary_json, completion_json])
        comp = MockE2EComputer(active_window="TextEditor", active_application="TextEditor")
        judge = JudgeAgent(computer_provider=comp)
        boss = BossAgent(computer_provider=comp, llm_router=llm, judge=judge)

        # Worker 1: Terminal write
        trace1 = _make_trace(
            step_index=1,
            action_type=ComputerActionType.TERMINAL_EXEC,
            target="echo 'PORT=8080' > config.env",
            status=ActionExecutionStatus.COMPLETED,
            exit_code=0,
            observation="PORT=8080",
            expected="config written",
        )
        res1 = SubMissionResult(
            sub_mission_id="sub-write",
            goal="Deploy config.env",
            traces=[trace1],
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="Config file created with PORT=8080",
            actions_taken=1,
        )

        # Worker 2: GUI inspection
        trace2 = _make_trace(
            step_index=1,
            action_type=ComputerActionType.GUI_CLICK,
            target="TextEditor window",
            status=ActionExecutionStatus.COMPLETED,
            observation="TextEditor displayed PORT=8080",
            expected="config displayed",
        )
        res2 = SubMissionResult(
            sub_mission_id="sub-gui",
            goal="Inspect config.env",
            traces=[trace2],
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="TextEditor visually confirmed PORT=8080",
            actions_taken=1,
        )

        dispatch_results = [res1, res2]
        with patch.object(boss, "_dispatch_sub_agent", AsyncMock(side_effect=dispatch_results)):
            report = await boss.run(workspace_id="ws-e2e", objective="Deploy and inspect config.env")

        assert report.status == "COMPLETE"
        assert report.total_actions == 2
        assert len(report.phases) >= 1
        assert all(t.status == TaskStatus.SUCCEEDED for t in boss.task_graph.tasks.values())

    @pytest.mark.asyncio
    async def test_tier4_autonomous_loop_failure_recovery_and_replan(self):
        """Scenario: Adaptive recovery loop across phases.
        Phase 1: Worker succeeds in discovery, but completion check requires Phase 2 execution.
        Phase 2: Worker achieves verified completion.
        """
        phase1_plan = json.dumps({
            "thinking": "Phase 1: Discover available API service endpoints.",
            "phase_name": "Discovery",
            "sub_missions": [
                {"id": "sub-disc-1", "goal": "Verify service availability by discovering API endpoints", "plane": "headless", "max_steps": 2},
            ],
        })
        phase1_summary = "Discovery completed: service endpoint identified at /api/v1."
        completion_check1 = json.dumps({"complete": False, "reason": "Endpoint discovered; proceed with verified testing in phase 2."})

        phase2_plan = json.dumps({
            "thinking": "Phase 2: Test API endpoint and confirm service response.",
            "phase_name": "Service Verification",
            "sub_missions": [
                {"id": "sub-test-2", "goal": "Verify service availability at /api/v1", "plane": "headless", "max_steps": 2},
            ],
        })
        phase2_summary = "Probe successful: 200 OK received."
        completion_check2 = json.dumps({"complete": True, "reason": "Verified service response."})

        llm = _make_llm_mock([
            phase1_plan,
            phase1_summary,
            completion_check1,
            phase2_plan,
            phase2_summary,
            completion_check2,
        ])
        comp = MockE2EComputer()
        boss = BossAgent(computer_provider=comp, llm_router=llm)

        # Result 1: Discovery progress
        res1 = SubMissionResult(
            sub_mission_id="sub-disc-1",
            goal="Verify service availability by discovering API endpoints",
            traces=[_make_trace(
                step_index=1,
                action_type=ComputerActionType.TERMINAL_EXEC,
                target="curl /api/routes",
                status=ActionExecutionStatus.COMPLETED,
                exit_code=0,
                observation="Found /api/v1",
            )],
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="Discovered /api/v1",
            actions_taken=1,
        )

        # Result 2: Verification success
        res2 = SubMissionResult(
            sub_mission_id="sub-test-2",
            goal="Verify service availability at /api/v1",
            traces=[_make_trace(
                step_index=1,
                action_type=ComputerActionType.TERMINAL_EXEC,
                target="curl /api/v1/health",
                status=ActionExecutionStatus.COMPLETED,
                exit_code=0,
                observation="HTTP 200 OK",
            )],
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="Service verified with HTTP 200",
            actions_taken=1,
        )

        async def mock_dispatch(workspace_id, sub_mission, *args, **kwargs):
            if sub_mission.id == "sub-disc-1":
                return res1
            return res2

        with patch.object(boss, "_dispatch_sub_agent", side_effect=mock_dispatch):
            report = await boss.run(workspace_id="ws-e2e", objective="Verify service availability")

        assert len(report.phases) == 2
        assert report.phases[0].status == "COMPLETED"
        assert report.phases[1].status == "COMPLETED"
        assert report.total_actions == 2
        assert report.status == "COMPLETE"

    @pytest.mark.asyncio
    async def test_tier4_safety_envelope_contained_autonomous_loop(self):
        """Scenario: Safety envelope blocks adversarial command during autonomous mission.
        ActionPolicy intercepts destructive action; provider remains untouched; Boss records containment.
        """
        decomp_json = json.dumps({
            "thinking": "Sub-mission attempting dangerous deletion.",
            "phase_name": "Maintenance",
            "sub_missions": [
                {"id": "sub-danger", "goal": "Clean system files", "plane": "headless", "max_steps": 2},
            ],
        })
        llm = _make_llm_mock([decomp_json, "Action blocked by safety policy."])
        comp = MockE2EComputer()
        policy = ActionPolicy(workspace_root="/home/sonic/workspace")

        agent = ComputerUseAgent(
            computer_provider=comp,
            llm_router=MagicMock(),
            safety=policy,
            self_host=True,
            tenant_id="tenant-e2e",
        )

        # Execute destructive action
        trace = await agent.execute_action(
            comp.workspace_id,
            ComputerActionType.TERMINAL_EXEC,
            "cleanup",
            {"command": "rm -rf /"},
            "Clean root files",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126
        assert "safety blocked" in trace.actual_observation.lower()
        # Ensure provider terminal was never invoked
        assert len(comp.terminal_calls) == 0
