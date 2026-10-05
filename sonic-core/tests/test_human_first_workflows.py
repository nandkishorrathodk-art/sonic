"""Human-first workflow coverage using the existing computer-use test doubles.

These tests deliberately exercise ordinary desktop work rather than security
scenarios.  They keep the provider and agent in the loop while remaining
independent of Docker and a live display server.
"""

from __future__ import annotations

from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
from sonic.sandbox.provider import ExecResult
from tests.test_phase4_visible_desktop_actions import _VisibleDesktop


class _WorkflowComputer(_VisibleDesktop):
    """Small stateful extension of the shared provider helper."""

    def __init__(self, *, terminal_results=None):
        super().__init__()
        self.workspace_id = "ws-desktop"
        self.files: dict[str, str] = {}
        self.terminal_results = list(terminal_results or [])
        self.gui_states: list[str] = []
        self.commands: list[str] = []
        self.commands: list[str] = []

    async def write_file(self, workspace_id, path, content, actor="operator"):
        self.files[path] = content
        return True

    async def read_file(self, workspace_id, path):
        return self.files[path]

    async def terminal(self, workspace_id, command, timeout=60, actor="operator"):
        self.commands.append(command)
        if self.terminal_results:
            result = self.terminal_results.pop(0)
            return result
        return ExecResult(command=command, exit_code=0, stdout="ready", stderr="")

    async def screenshot(self, workspace_id):
        screen = await super().screenshot(workspace_id)
        screen.controls_age_ms = 0
        screen.detected_controls = [
            {
                "name": "title field",
                "role": "entry",
                "bbox": (90, 110, 200, 25),
                "source": "atspi",
                "confidence": 0.98,
                "freshness_ms": 0,
                "editable": True,
                "focused": self.focused,
            }
        ]
        return screen


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_document_workflow_writes_then_reads_real_provider_state():
    computer = _WorkflowComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    path = "/home/sonic/workspace/meeting-notes.md"

    written = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.FILE_WRITE,
        path,
        {"path": path, "content": "# Meeting notes\n\nNext steps."},
        "Save the document",
    ))
    read = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.FILE_READ,
        path,
        {"path": path},
        "Review the document",
    ))

    assert written.status == "SUCCESS"
    assert read.status == "SUCCESS"
    assert computer.files[path].startswith("# Meeting notes")
    assert "Read" in read.actual_observation


def test_gui_workflow_uses_grounded_controls_in_the_shared_desktop():
    computer = _WorkflowComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    _run(agent.observe(computer.workspace_id))

    typed = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.GUI_TYPE,
        "title field",
        {"x": 100, "y": 120, "text": "Quarterly notes"},
        "Enter the requested document title",
    ))

    assert typed.status == ActionExecutionStatus.COMPLETED, typed.actual_observation
    assert [action.action.value for action in computer.actions] == ["CLICK", "TYPE"]
    assert computer.actions[0].x == 100
    assert computer.actions[1].text == "Quarterly notes"


def test_gui_and_terminal_coordination_keeps_planes_distinct():
    computer = _WorkflowComputer(terminal_results=[
        ExecResult(command="printf 'draft-ready'", exit_code=0, stdout="draft-ready", stderr=""),
    ])
    agent = ComputerUseAgent(computer_provider=computer)

    focused = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.APP_FOCUS,
        "document window",
        {"window_id": "document window"},
        "Bring the document window forward",
    ))
    command = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.TERMINAL_EXEC,
        "printf 'draft-ready'",
        {"command": "printf 'draft-ready'"},
        "Check the saved draft from the operator terminal",
    ))

    assert focused.status == "SUCCESS"
    assert command.status == "SUCCESS"
    assert computer.commands == ["printf 'draft-ready'"]
    assert "draft-ready" in command.actual_observation


def test_unexpected_provider_failure_recovers_without_claiming_success():
    computer = _WorkflowComputer(terminal_results=[
        ExecResult(command="check", exit_code=1, stdout="", stderr="unexpected state"),
    ])
    agent = ComputerUseAgent(computer_provider=computer)

    trace = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.TERMINAL_EXEC,
        "check",
        {"command": "check"},
        "Inspect the current document state",
    ))

    assert trace.status in ("FAILED", "RECOVERED")
    assert trace.status != "SUCCESS"
    assert trace.recovery_attempted is True
    assert "unexpected state" in trace.actual_observation


def test_verification_requires_truthful_file_evidence():
    good = _WorkflowComputer(terminal_results=[
        ExecResult(
            command="verify",
            exit_code=0,
            stdout="FILE_VERIFIED: /home/sonic/workspace/notes.md\n# Notes",
            stderr="",
        )
    ])
    agent = ComputerUseAgent(computer_provider=good)
    verified, evidence = _run(agent.verify_goal(
        good.workspace_id, "create notes.md with the meeting summary"
    ))
    assert verified is True
    assert "FILE_VERIFIED" in evidence

    deceptive = _WorkflowComputer(terminal_results=[
        ExecResult(command="verify", exit_code=0, stdout="__sonic_obs_ready__", stderr="")
    ])
    agent = ComputerUseAgent(computer_provider=deceptive)
    verified, evidence = _run(agent.verify_goal(
        deceptive.workspace_id, "create notes.md with the meeting summary"
    ))
    assert verified is False
    assert "dummy marker" in evidence.lower()
