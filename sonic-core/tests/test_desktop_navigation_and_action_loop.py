"""Tests for visible destination navigation and desktop action loop protection."""

from __future__ import annotations

import pytest

from sonic.computer.models import ComputerState, FileEntry, GitStatusInfo, ScreenObservation
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType


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
            active_application="Desktop",
            open_applications=["Desktop", "Document Application"],
            running_processes=[],
        )

    async def screenshot(self, workspace_id):
        return ScreenObservation(
            screenshot_base64="c2NyZWVu" if self.display_available else "",
            visible_text="Desktop application",
            active_window="Document Application",
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
    assert trace1.status == ActionExecutionStatus.COMPLETED
    assert agent._last_gui_navigation_target == target
    assert [action.key for action in comp.gui_actions if action.key] == ["ctrl+l", "Return"]
    assert not comp.commands_executed


@pytest.mark.asyncio
async def test_app_focus_requires_matching_active_window_evidence():
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)

    trace = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.APP_FOCUS,
        "Another Application",
        {"app_name": "Another Application"},
        "Focus the requested application",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert "was not verified as the active window" in trace.actual_observation


@pytest.mark.asyncio
async def test_app_focus_succeeds_when_requested_window_is_visible():
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)

    trace = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.APP_FOCUS,
        "Document Application",
        {"app_name": "Document Application"},
        "Focus the document application",
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert "active window: Document Application" in trace.actual_observation


@pytest.mark.asyncio
async def test_app_focus_fails_without_live_display_evidence():
    comp = _MockComputerProvider()
    comp.display_available = False
    agent = ComputerUseAgent(computer_provider=comp)

    trace = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.APP_FOCUS,
        "Document Application",
        {"app_name": "Document Application"},
        "Focus the document application",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert "observed: unknown" in trace.actual_observation


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
    assert trace.status in {ActionExecutionStatus.FAILED, ActionExecutionStatus.RECOVERED}
    assert "could not be verified from a live screen" in trace.actual_observation
    assert agent._last_gui_navigation_target == previous_target
    assert not comp.commands_executed


@pytest.mark.asyncio
async def test_consecutive_action_loop_breaker():
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)

    # Execute exact same action 3 times
    t1 = await agent.execute_action(
        comp.workspace_id,
        ComputerActionType.GUI_CLICK,
        "search bar",
        {"query": "search bar"},
        "click search",
    )
    assert t1.status == ActionExecutionStatus.BLOCKED
    assert "visual grounding" in t1.actual_observation.lower()

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
    # The action remains blocked until a live frame grounds the target.
    assert t3.status == ActionExecutionStatus.BLOCKED


@pytest.mark.asyncio
async def test_reasoning_context_uses_general_desktop_state():
    comp = _MockComputerProvider()
    agent = ComputerUseAgent(computer_provider=comp)

    obs = await agent.observe(comp.workspace_id)
    sys_prompt, user_prompt = agent._build_reasoning_context("explore target", obs, 2, "main.py", "")

    assert "Desktop Application State:" in user_prompt
    assert "Browser State:" not in user_prompt
    assert "BROWSER_" not in sys_prompt
