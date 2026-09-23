from types import SimpleNamespace

import pytest

from sonic.computer.models import GUIActionType, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType


class _Provider:
    def __init__(self, *, change=True, fail_at=None):
        self.change = change
        self.fail_at = fail_at
        self.calls = 0
        self.frame = 0

    async def gui_action(self, workspace_id, action):
        index = self.calls
        self.calls += 1
        if index == self.fail_at:
            raise RuntimeError("dispatch failed")
        if self.change:
            self.frame += 1
        return ScreenObservation(
            screenshot_base64=f"frame-{self.frame}",
            width=800,
            height=600,
            active_window="Target",
            visible_text=f"step {self.frame}",
        )

    async def screenshot(self, workspace_id):
        return ScreenObservation(
            screenshot_base64=f"frame-{self.frame}",
            width=800,
            height=600,
            active_window="Target",
            visible_text=f"step {self.frame}",
        )

    async def status(self, workspace_id):
        return SimpleNamespace(
            open_applications=[],
            running_processes=[],
            active_application="Target",
        )

    async def list_files(self, workspace_id, path):
        return []

    async def git_action(self, workspace_id, action, **kwargs):
        return SimpleNamespace(stdout="", exit_code=0)

    async def terminal(self, workspace_id, command):
        return SimpleNamespace(stdout="", stderr="", exit_code=0)


def _payload(**extra):
    value = {
        "burst_actions": [
            {"action": GUIActionType.KEYPRESS.value, "key": "Tab"},
            {"action": GUIActionType.KEYPRESS.value, "key": "Enter"},
        ],
    }
    value.update(extra)
    return value


@pytest.mark.asyncio
async def test_agent_uses_motor_burst_and_verifies_live_postcondition():
    provider = _Provider()
    agent = ComputerUseAgent(provider, max_actions=5)
    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        _payload(postcondition={"visible_text": "step 2"}),
        "target advances",
    )
    assert trace.status == ActionExecutionStatus.COMPLETED
    assert provider.calls == 2
    assert trace.exit_code == 0


@pytest.mark.asyncio
async def test_burst_failure_stops_and_injects_replan():
    provider = _Provider(fail_at=1)
    agent = ComputerUseAgent(provider)
    trace = await agent.execute_action(
        "workspace", ComputerActionType.GUI_KEYPRESS, "target", _payload(), "advance"
    )
    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 2
    assert any("REPLAN" in item.get("result", "") for item in agent.history)


@pytest.mark.asyncio
async def test_unchanged_ui_and_failed_postcondition_fail_closed():
    provider = _Provider(change=False)
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")
    trace = await agent.execute_action(
        "workspace", ComputerActionType.GUI_KEYPRESS, "target", _payload(), "advance"
    )
    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 1
    assert trace.exit_code == 126


@pytest.mark.asyncio
async def test_stale_burst_version_never_dispatches():
    provider = _Provider()
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")
    current = agent.perception_bus.current().version
    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        _payload(state_version=current - 1),
        "advance",
    )
    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 0
    assert "stale" in trace.actual_observation.lower()
