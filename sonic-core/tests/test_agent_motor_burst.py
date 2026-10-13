import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from sonic.computer.models import GUIActionType, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType


def _frame(frame):
    image = Image.new("RGB", (800, 600), (frame * 40, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


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
            screenshot_base64=_frame(self.frame),
            width=800,
            height=600,
            active_window="Target",
            visible_text=f"step {self.frame}",
            capture_age_ms=0,
            controls_age_ms=0,
        )

    async def screenshot(self, workspace_id):
        return ScreenObservation(
            screenshot_base64=_frame(self.frame),
            width=800,
            height=600,
            active_window="Target",
            visible_text=f"step {self.frame}",
            capture_age_ms=0,
            controls_age_ms=0,
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
            {
                "action": GUIActionType.KEYPRESS.value,
                "key": "Tab",
                "postcondition": {"screen_changed": True},
            },
            {
                "action": GUIActionType.KEYPRESS.value,
                "key": "Enter",
                "postcondition": {"screen_changed": True},
            },
        ],
        "postcondition": {"screen_changed": True},
    }
    value.update(extra)
    return value


@pytest.mark.asyncio
async def test_agent_uses_motor_burst_and_verifies_live_postcondition():
    provider = _Provider()
    agent = ComputerUseAgent(provider, max_actions=5)
    await agent.observe("workspace")
    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        _payload(postcondition={"screen_changed": True}),
        "target advances",
    )
    assert trace.status == ActionExecutionStatus.COMPLETED
    assert provider.calls == 2
    assert trace.exit_code == 0


@pytest.mark.asyncio
async def test_burst_failure_stops_and_injects_replan():
    provider = _Provider(fail_at=1)
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")
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


@pytest.mark.asyncio
async def test_burst_requires_a_postcondition_for_each_action_and_the_final_state():
    provider = _Provider()
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")
    payload = _payload()
    payload["burst_actions"][0].pop("postcondition")

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        payload,
        "advance",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 0
    assert "every GUI burst action requires a postcondition" in trace.actual_observation


@pytest.mark.asyncio
async def test_burst_rejects_more_than_the_bounded_action_count():
    provider = _Provider()
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")
    payload = _payload()
    payload["burst_actions"] = payload["burst_actions"] * 3

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        payload,
        "advance",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 0
    assert "limited to 4 actions" in trace.actual_observation


@pytest.mark.asyncio
async def test_burst_checks_coordinates_before_any_action_is_dispatched():
    provider = _Provider()
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")
    payload = _payload()
    payload["burst_actions"][1] = {
        "action": GUIActionType.CLICK.value,
        "x": 900,
        "y": 20,
        "postcondition": {"screen_changed": True},
    }

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        payload,
        "advance",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 0
    assert "outside screen 800x600" in trace.actual_observation


@pytest.mark.asyncio
async def test_burst_holds_when_the_active_window_changes_unexpectedly():
    class _WindowChangingProvider(_Provider):
        def __init__(self):
            super().__init__()
            self.window_changed = False

        async def gui_action(self, workspace_id, action):
            result = await super().gui_action(workspace_id, action)
            self.window_changed = True
            return result

        async def screenshot(self, workspace_id):
            result = await super().screenshot(workspace_id)
            if self.window_changed:
                result.active_window = "Unexpected window"
            return result

    provider = _WindowChangingProvider()
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        _payload(),
        "advance",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 1
    assert "active window changed unexpectedly" in trace.actual_observation


@pytest.mark.asyncio
async def test_burst_rejects_postcondition_without_an_observable_predicate():
    provider = _Provider()
    agent = ComputerUseAgent(provider)
    await agent.observe("workspace")
    payload = _payload()
    payload["burst_actions"][0]["postcondition"] = {"require_state_change": True}

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_KEYPRESS,
        "target",
        payload,
        "advance",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert provider.calls == 0
    assert "must name an observable state" in trace.actual_observation
