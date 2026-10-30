"""Human-first workflow coverage using the existing computer-use test doubles.

These tests deliberately exercise ordinary desktop work rather than security
scenarios.  They keep the provider and agent in the loop while remaining
independent of Docker, a display server, and Playwright.
"""

from __future__ import annotations

from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ComputerActionType
from sonic.sandbox.provider import ExecResult
from tests.test_phase4_unified_browser_in_loop import (
    _StubBrowser,
    _StubComputer,
    _StubPage,
)


class _WorkflowComputer(_StubComputer):
    """Small stateful extension of the shared provider helper."""

    def __init__(self, *, terminal_results=None):
        super().__init__()
        self.files: dict[str, str] = {}
        self.terminal_results = list(terminal_results or [])
        self.gui_states: list[str] = []

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


def test_browser_ui_workflow_uses_live_page_controls_without_named_apps():
    browser = _StubBrowser({
        "about:blank": _StubPage(
            url="about:blank",
            title="Document preview",
            elements=[{"tag": "input", "selector": "#title", "text": ""},
                      {"tag": "button", "selector": "#save", "text": "Save"}],
        )
    })
    browser.current = browser.pages["about:blank"]
    computer = _WorkflowComputer()
    agent = ComputerUseAgent(computer_provider=computer, browser=browser)

    typed = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.BROWSER_TYPE,
        "#title",
        {"selector": "#title", "text": "Quarterly notes"},
        "Fill the visible title field",
    ))
    clicked = _run(agent.execute_action(
        computer.workspace_id,
        ComputerActionType.BROWSER_CLICK,
        "#save",
        {"selector": "#save"},
        "Activate the visible save control",
    ))

    assert typed.status == "SUCCESS"
    assert clicked.status == "SUCCESS"
    assert ("type", {"selector": "#title", "text": "Quarterly notes"}) in browser.actions
    assert ("click", {"selector": "#save"}) in browser.actions


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
