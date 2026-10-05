"""Focused coverage for execution evidence exposed on computer-use traces."""

import asyncio

from sonic.computer.models import ComputerState, GitStatusInfo, ScreenObservation
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.action_contracts import ActionReceipt
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerWorldObservation,
)
from sonic.safety.action_policy import ActionPolicy
from sonic.sandbox.provider import ExecResult


class _ReceiptComputer(ComputerProvider):
    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        self.observe_state = None
        self.gui_calls = 0
        self.commands: list[str] = []

    async def create(self, *args, **kwargs): return type("Workspace", (), {"id": "workspace"})()
    async def destroy(self, *args, **kwargs): return True
    async def status(self, workspace_id):
        return ComputerState(
            workspace_id=workspace_id,
            tenant_id="test",
            active_application="Desktop",
        )

    async def screenshot(self, workspace_id):
        return ScreenObservation(
            screenshot_base64="before-screen",
            width=1280,
            height=800,
            active_window="Desktop",
            capture_age_ms=0,
            controls_age_ms=0,
        )
    async def list_files(self, *args, **kwargs): return []
    async def read_file(self, *args, **kwargs): return ""
    async def write_file(self, *args, **kwargs): return True
    async def git_action(self, *args, **kwargs): return GitStatusInfo(branch="main", is_clean=True)
    async def application_list(self, *args, **kwargs): return []
    async def process_list(self, *args, **kwargs): return []
    async def launch_application(self, *args, **kwargs): return True
    async def close_application(self, *args, **kwargs): return True
    async def install_application(self, *args, **kwargs): return True, ""
    async def uninstall_application(self, *args, **kwargs): return True
    async def service_action(self, *args, **kwargs): return None

    async def terminal(self, workspace_id, command, timeout=60, actor="operator"):
        self.commands.append(command)
        return ExecResult(
            command=command,
            exit_code=self.exit_code,
            stdout="real output" if self.exit_code == 0 else "",
            stderr="" if self.exit_code == 0 else "real failure",
        )

    async def gui_action(self, workspace_id, action, actor="operator"):
        self.gui_calls += 1
        return ScreenObservation(
            screenshot_base64="after-screen",
            capture_age_ms=0,
            controls_age_ms=0,
            active_window="observed window",
        )


class _StateRevalidatingComputer(_ReceiptComputer):
    async def observe_state(self, workspace_id):
        return ScreenObservation(
            screenshot_base64="",
            width=1280,
            height=800,
            active_window="Desktop",
            desktop_state="STATE_ONLY",
            capture_age_ms=0,
            controls_age_ms=0,
            visible_text="same desktop",
            perception_version=2,
        )


def _run(coro):
    return asyncio.run(coro)


def _execute_gui_move_from_stale_observation(computer, screen, perception_version=0):
    agent = ComputerUseAgent(computer_provider=computer)

    async def run():
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=perception_version,
            observation_age_ms=3_000,
        )
        agent._last_screen_observation_received_at = (
            asyncio.get_running_loop().time() - 3
        )
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_MOVE,
            "visible desktop",
            {"x": 640, "y": 400},
            "cursor reaches requested position",
        )

    return _run(run())


def test_terminal_trace_exposes_receipt_with_exit_code_and_delta():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.TERMINAL_EXEC,
            "inspect",
            {"command": "inspect"},
            "real output",
        )
    )

    assert isinstance(trace.action_receipt, ActionReceipt)
    assert trace.action_receipt.exit_code == 0
    assert trace.action_receipt.status == trace.status
    assert trace.action_receipt.evidence == "real output"
    assert trace.action_receipt.state_delta.changed["exit_code"] == 0


def test_failed_terminal_trace_receipt_is_not_successful():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer(exit_code=1))

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.TERMINAL_EXEC,
            "inspect",
            {"command": "inspect"},
            "real output",
        )
    )

    assert trace.status == ActionExecutionStatus.RECOVERED
    assert trace.action_receipt.status == ActionExecutionStatus.RECOVERED
    assert trace.action_receipt.exit_code == 1
    assert trace.action_receipt.after["exit_code"] == 1


def test_gui_trace_receipt_uses_observed_screen_transition():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)

    async def run_click():
        screen = await computer.screenshot("workspace")
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=0,
            observation_age_ms=0,
        )
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_CLICK,
            "observed control",
            {"x": 10, "y": 10},
            "screen changes",
        )

    trace = _run(run_click())

    assert trace.status != ActionExecutionStatus.BLOCKED, trace.actual_observation
    assert computer.gui_calls == 1
    assert trace.action_receipt.before["screen_hash"] == "before-screen"
    assert trace.action_receipt.after["screen_hash"] == "after-screen"
    assert "screen_hash" in trace.action_receipt.state_delta.changed


def test_slow_gui_reasoning_revalidates_unchanged_screenshot_before_dispatch():
    computer = _ReceiptComputer()
    screen = ScreenObservation(
        screenshot_base64="before-screen",
        width=1280,
        height=800,
        active_window="Desktop",
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
        controls_age_ms=0,
    )

    trace = _execute_gui_move_from_stale_observation(computer, screen)

    assert trace.status != ActionExecutionStatus.BLOCKED, trace.actual_observation
    assert computer.gui_calls == 1


def test_slow_gui_reasoning_fails_closed_when_screenshot_changed():
    class ChangedScreenComputer(_ReceiptComputer):
        async def screenshot(self, workspace_id):
            return ScreenObservation(
                screenshot_base64="changed-screen",
                width=1280,
                height=800,
                active_window="Desktop",
                desktop_state="INTERACTIVE",
                capture_age_ms=0,
                controls_age_ms=0,
            )

    computer = ChangedScreenComputer()
    screen = ScreenObservation(
        screenshot_base64="before-screen",
        width=1280,
        height=800,
        active_window="Desktop",
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
        controls_age_ms=0,
    )

    trace = _execute_gui_move_from_stale_observation(computer, screen)

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert "desktop screenshot changed" in trace.actual_observation
    assert computer.gui_calls == 0


def test_slow_gui_reasoning_revalidates_fresh_nsi_state_before_dispatch():
    computer = _StateRevalidatingComputer()
    del computer.observe_state
    screen = ScreenObservation(
        screenshot_base64="",
        width=1280,
        height=800,
        active_window="Desktop",
        desktop_state="STATE_ONLY",
        capture_age_ms=0,
        controls_age_ms=0,
        visible_text="same desktop",
        perception_version=1,
    )

    trace = _execute_gui_move_from_stale_observation(
        computer,
        screen,
        perception_version=1,
    )

    assert trace.status != ActionExecutionStatus.BLOCKED, trace.actual_observation
    assert computer.gui_calls == 1


def test_file_write_trace_exposes_receipt_without_fabricating_exit_code():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer())

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.FILE_WRITE,
            "notes.txt",
            {"path": "notes.txt", "content": "observed"},
            "file written",
        )
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert isinstance(trace.receipt, ActionReceipt)
    assert trace.receipt.exit_code is None
    assert trace.state_delta is trace.receipt.state_delta
    assert trace.receipt.evidence == "Wrote patch (8 bytes) to notes.txt"


def test_verify_goal_runs_typed_probe_through_action_receipt():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(
        computer_provider=computer,
        safety=ActionPolicy(workspace_root="/home/sonic/workspace"),
    )

    verified, evidence = _run(agent.verify_goal("workspace", "run pytest tests"))

    assert verified is True, evidence
    assert computer.commands == ["python -m pytest -q --tb=short"]
    trace = agent.traces[-1]
    assert trace.action_type == ComputerActionType.VERIFY_PROBE
    assert trace.exit_code == 0
    assert trace.action_receipt.exit_code == 0
    assert "exit_code=0" in evidence


def test_invalid_verification_probe_is_blocked_before_terminal():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(
        computer_provider=computer,
        safety=ActionPolicy(workspace_root="/home/sonic/workspace"),
    )

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.VERIFY_PROBE,
            "pytest",
            {"probe": {"kind": "pytest", "target": "; touch /tmp/pwned"}},
            "verification",
        )
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert trace.exit_code == 126
    assert computer.commands == []


def test_llm_verification_cannot_supply_a_shell_command():
    computer = _ReceiptComputer()
    router = type(
        "Judge",
        (),
        {"complete": staticmethod(
            lambda request: _async_response("YES")
        )},
    )()
    agent = ComputerUseAgent(
        computer_provider=computer,
        llm_router=router,
        safety=ActionPolicy(workspace_root="/home/sonic/workspace"),
    )

    verified, _ = _run(agent.verify_goal("workspace", "run pytest", use_llm=True))

    assert verified is True
    assert computer.commands == ["python -m pytest -q --tb=short"]


async def _async_response(content: str):
    return type("Response", (), {"content": content})()
