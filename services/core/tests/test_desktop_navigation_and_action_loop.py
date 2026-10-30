"""Tests for visible destination navigation and desktop action loop protection."""

from __future__ import annotations

import pytest

from sonic.computer.models import ComputerState, FileEntry, GitStatusInfo, ScreenObservation
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
from tests.avcs_test_helpers import png_base64


class _MockComputerProvider(ComputerProvider):
    def __init__(self):
        self.workspace_id = "ws-test"
        self.commands_executed: list[str] = []
        self.gui_actions = []
        self.display_available = True

    async def create(self, tenant_id, engagement_id, **kw):
        return type("W", (), {"id": self.workspace_id})()

    async def destroy(self, workspace_id):
        return True

    async def status(self, workspace_id):
        return ComputerState(
            workspace_id=workspace_id,
            tenant_id="test",
            active_application="Document Application",
            active_window="Document Application",
            open_applications=["Desktop", "Document Application"],
            running_processes=[],
        )

    async def screenshot(self, workspace_id):
        return ScreenObservation(
            screenshot_base64=png_base64() if self.display_available else "",
            width=1280,
            height=800,
            visible_text="Desktop application",
            active_window="Document Application",
            capture_age_ms=0 if self.display_available else None,
            controls_age_ms=0 if self.display_available else None,
        )

    async def gui_action(self, workspace_id, action, **kwargs):
        self.gui_actions.append(action)
        return await self.screenshot(workspace_id)

    async def terminal(self, workspace_id, command, timeout=60, actor="operator"):
        self.commands_executed.append(command)
        raise AssertionError("Visible desktop navigation must not use terminal application discovery")

    async def read_file(self, workspace_id, path):
        return ""

    async def write_file(self, workspace_id, path, content, actor="operator"):
        return True

    async def list_files(self, workspace_id, path="."):
        return [FileEntry(name="README.md", path="README.md")]

    async def git_action(self, workspace_id, action, **kw):
        return GitStatusInfo()

    async def process_list(self, *a, **k):
        return []

    async def application_list(self, *a, **k):
        return []

    async def launch_application(self, *a, **k):
        return True

    async def close_application(self, *a, **k):
        return True

    async def install_application(self, *a, **k):
        return True

    async def uninstall_application(self, *a, **k):
        return True

    async def service_action(self, *a, **k):
        return True

    async def snapshot(self, *a, **k):
        return {}

    async def restore_snapshot(self, *a, **k):
        return True


@pytest.mark.asyncio
async def test_destination_navigation_uses_visible_desktop_without_terminal_discovery():
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)
    target = "https://workspace.test"

    trace1 = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.BROWSER_NAVIGATE,
        target,
        {"url": target},
        "open target",
    )
    assert trace1.status == ActionExecutionStatus.BLOCKED
    assert "legacy browser_navigate is blocked" in trace1.actual_observation.lower()
    assert agent._last_gui_navigation_target == ""
    assert not comp.gui_actions
    assert not comp.commands_executed


@pytest.mark.asyncio
async def test_destination_navigation_fails_without_live_desktop_evidence():
    comp = _MockComputerProvider()
    comp.display_available = False
    agent = ComputerUseAgent(computer_provider=comp)
    previous_target = "https://previous.test"
    next_target = "https://next.test"
    agent._last_gui_navigation_target = previous_target

    trace = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.BROWSER_NAVIGATE,
        next_target,
        {"url": next_target},
        "open target",
    )
    assert trace.status == ActionExecutionStatus.BLOCKED
    assert "legacy browser_navigate is blocked" in trace.actual_observation.lower()
    assert agent._last_gui_navigation_target == previous_target
    assert not comp.commands_executed


@pytest.mark.asyncio
async def test_unresolved_semantic_click_stays_blocked_without_grounded_coordinates():
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)
    await agent.observe(comp.workspace_id)

    # Execute exact same action 3 times
    t1 = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.GUI_CLICK,
        "search bar",
        {"query": "search bar"},
        "click search",
    )
    assert t1.status == ActionExecutionStatus.BLOCKED
    assert "screen-pixel coordinates" in t1.actual_observation.lower()

    t2 = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.GUI_CLICK,
        "search bar",
        {"query": "search bar"},
        "click search",
    )
    assert t2.status == ActionExecutionStatus.BLOCKED

    t3 = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.GUI_CLICK,
        "search bar",
        {"query": "search bar"},
        "click search",
    )
    assert t3.status == ActionExecutionStatus.BLOCKED
    assert "[ACTION LOOP DETECTED]" in t3.actual_observation
    assert not comp.gui_actions
    assert comp.commands_executed == ["echo __sonic_obs_ready__"]


@pytest.mark.asyncio
async def test_alternating_action_loop_is_blocked():
    """An A-B-A-B oscillation must be blocked, not allowed to burn the budget.

    Reproduces a live failure: the worker alternated APP_LAUNCH thunar /
    APP_LAUNCH mousepad 18 times while the goal was to arrange the windows.
    The consecutive (3x identical) breaker never fires for an alternating
    pattern.
    """
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)
    await agent.observe(comp.workspace_id)

    # Two executed signatures that form a completed period-2 cycle.
    agent._recent_action_signatures = [
        (ComputerActionType.APP_LAUNCH.value, "thunar", "{}"),
        (ComputerActionType.APP_LAUNCH.value, "mousepad", "{}"),
        (ComputerActionType.APP_LAUNCH.value, "thunar", "{}"),
        (ComputerActionType.APP_LAUNCH.value, "mousepad", "{}"),
    ]

    rejection = agent._is_trivial_or_repeated_action(
        ComputerActionType.APP_LAUNCH, "thunar", {}
    )
    assert rejection is not None
    assert "alternating loop detected" in rejection.lower()


@pytest.mark.asyncio
async def test_alternating_loop_allows_a_different_action():
    """Breaking the cycle with a new action must not be blocked."""
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)
    await agent.observe(comp.workspace_id)

    agent._recent_action_signatures = [
        (ComputerActionType.APP_LAUNCH.value, "thunar", "{}"),
        (ComputerActionType.APP_LAUNCH.value, "mousepad", "{}"),
        (ComputerActionType.APP_LAUNCH.value, "thunar", "{}"),
        (ComputerActionType.APP_LAUNCH.value, "mousepad", "{}"),
    ]

    # WINDOW_TILE advances the goal and is exempt from repetition blocking.
    assert (
        agent._is_trivial_or_repeated_action(
            ComputerActionType.WINDOW_TILE, "", {}
        )
        is None
    )


@pytest.mark.asyncio
async def test_reasoning_context_uses_general_desktop_state():
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)
    obs = await agent.observe(comp.workspace_id)
    sys_prompt, user_prompt = agent._build_reasoning_context("explore target", obs, 2, "main.py", "")

    assert "Desktop Application State:" in user_prompt
    assert "Browser State:" not in user_prompt
    assert "BROWSER_" not in sys_prompt
