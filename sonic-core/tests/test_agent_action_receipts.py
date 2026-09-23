"""Focused coverage for execution evidence exposed on computer-use traces."""

import asyncio

from sonic.computer.models import ComputerState, GitStatusInfo, ScreenObservation
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.action_contracts import ActionReceipt
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
from sonic.sandbox.provider import ExecResult


class _ReceiptComputer(ComputerProvider):
    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        self.gui_calls = 0

    async def create(self, *args, **kwargs): return type("Workspace", (), {"id": "workspace"})()
    async def destroy(self, *args, **kwargs): return True
    async def status(self, workspace_id): return ComputerState(workspace_id=workspace_id, tenant_id="test")
    async def screenshot(self, workspace_id): return ScreenObservation(screenshot_base64="before-screen")
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
            active_window="observed window",
        )


def _run(coro):
    return asyncio.run(coro)


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
    agent._last_screenshot_b64 = "before-screen"

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.GUI_CLICK,
            "observed control",
            {"x": 10, "y": 10},
            "screen changes",
        )
    )

    assert trace.status != ActionExecutionStatus.BLOCKED
    assert computer.gui_calls == 1
    assert trace.action_receipt.before["screen_hash"] == "before-screen"
    assert trace.action_receipt.after["screen_hash"] == "after-screen"
    assert "screen_hash" in trace.action_receipt.state_delta.changed


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
