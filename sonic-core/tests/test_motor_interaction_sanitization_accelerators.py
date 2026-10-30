"""
Tests for Motor Interaction, Input Sanitization & Keyboard Accelerators.
========================================================================
Verifies:
  1. Navigation instructions like 'File -> Open' in TARGET never fall back
     into GUI_TYPE typing payloads.
  2. Command and type payloads sanitize trailing thoughts, multiline commentary,
     and 'BeingCraft note:' blocks.
  3. GUI_TYPE focuses the grounded target before typing and only clears field
     contents when the action explicitly requests replacement.
  4. Standard keyboard accelerators (Alt+F, Ctrl+O, Ctrl+S, Ctrl+N) are present
     in prompts and action guidance.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.computer.models import GUIAction, GUIActionType, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ComputerActionType
from sonic.computer_use.motor import MotorReflexes
from sonic.llm import prompts


class _MockMotorComputer:
    """Mock computer for motor reflexes and computer-use agent tests."""

    def __init__(self):
        self.commands_executed: list[str] = []
        self.gui_actions: list[GUIAction] = []
        self._docker_exec = AsyncMock(side_effect=self._mock_docker_exec)
        self.gui_action = AsyncMock(side_effect=self._mock_gui_action)
        self.terminal = AsyncMock(side_effect=self._mock_terminal)
        self.screenshot = AsyncMock(return_value=ScreenObservation(width=1280, height=800))
    async def _mock_docker_exec(self, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
        self.commands_executed.append(cmd)
        return (0, "", "")

    async def _mock_gui_action(
        self, workspace_id: str, action: GUIAction, actor: str = "operator"
    ) -> ScreenObservation:
        self.gui_actions.append(action)
        return ScreenObservation(width=1280, height=800)

    async def _mock_terminal(self, workspace_id: str, cmd: str, timeout: int = 60):
        self.commands_executed.append(cmd)
        return MagicMock(exit_code=0, stdout="__sonic_obs_ready__", stderr="")

    async def status(self, workspace_id: str = ""):
        return MagicMock(
            active_application="Desktop",
            open_applications=["Desktop"],
            running_processes=["xvfb"],
        )


# ---------------------------------------------------------------------------
# 1. Target Fallback Prevention for GUI_TYPE
# ---------------------------------------------------------------------------

def test_gui_type_does_not_fallback_to_target_navigation_instruction():
    """Navigation instructions in TARGET must NEVER fall back into GUI_TYPE text."""
    text = (
        "THOUGHT: Open file menu\n"
        "ACTION: GUI_TYPE\n"
        "TARGET: File -> Open\n"
        "PAYLOAD: {}\n"
        "EXPECTED: Menu opens\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.GUI_TYPE
    assert "File -> Open" not in payload.get("text", "")
    assert payload.get("text") == "" or "text" not in payload


def test_gui_type_with_explicit_text_payload():
    """Explicit text payload is preserved, ignoring TARGET navigation text."""
    text = (
        "THOUGHT: Type filename into dialog\n"
        "ACTION: GUI_TYPE\n"
        "TARGET: File -> Open Dialog\n"
        "PAYLOAD: {\"text\": \"notes.txt\"}\n"
        "EXPECTED: notes.txt typed\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.GUI_TYPE
    assert payload.get("text") == "notes.txt"


# ---------------------------------------------------------------------------
# 2. Payload Sanitization: Thoughts, Commentary & BeingCraft Notes
# ---------------------------------------------------------------------------

def test_terminal_exec_sanitization_strips_beingcraft_note():
    """BeingCraft note blocks must be stripped from terminal commands."""
    text = (
        "THOUGHT: Check active network sockets\n"
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: terminal\n"
        "PAYLOAD: ss -tuln\n"
        "BeingCraft note: Verified listening ports on the workstation\n"
        "EXPECTED: Sockets listed\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") == "ss -tuln"
    assert "BeingCraft" not in payload.get("command", "")


def test_terminal_exec_sanitization_strips_json_embedded_beingcraft_note():
    """BeingCraft notes inside json dict payloads must be stripped."""
    text = (
        'ACTION: TERMINAL_EXEC\n'
        'TARGET: terminal\n'
        'PAYLOAD: {"command": "cat /etc/os-release\\nBeingCraft note: checked distribution"}\n'
        'EXPECTED: OS release info\n'
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") == "cat /etc/os-release"
    assert "BeingCraft" not in payload.get("command", "")


def test_terminal_exec_sanitization_strips_trailing_thoughts_and_commentary():
    """Trailing thoughts and natural language commentary must not leak into bash commands."""
    text = (
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: terminal\n"
        "PAYLOAD: git status --short\n"
        "Thought: We need to see modified files before creating a commit\n"
        "EXPECTED: Clean or dirty status\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") == "git status --short"
    assert "Thought" not in payload.get("command", "")


def test_gui_type_sanitization_strips_beingcraft_note_and_reasoning():
    """BeingCraft notes and trailing thoughts must not be typed into GUI fields."""
    text = (
        "ACTION: GUI_TYPE\n"
        "TARGET: search input\n"
        "PAYLOAD: python3\n"
        "BeingCraft note: Entered search term for python executable\n"
        "EXPECTED: Typed\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.GUI_TYPE
    assert payload.get("text") == "python3"
    assert "BeingCraft" not in payload.get("text", "")


# ---------------------------------------------------------------------------
# 3. Motor Reflexes: Field Clearing & Focus Settling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_motor_clear_field_dispatches_clearmodifiers_ctrl_a_backspace():
    """clear_field sends xdotool key --clearmodifiers ctrl+a BackSpace."""
    comp = _MockMotorComputer()
    motor = MotorReflexes(comp)

    await motor.clear_field("ws-1")
    executed = comp.commands_executed
    assert any("xdotool key --clearmodifiers ctrl+a BackSpace" in cmd for cmd in executed)


@pytest.mark.asyncio
async def test_motor_settle_focus_activates_window_and_settles():
    """settle_focus activates target window and pauses for wm transition."""
    comp = _MockMotorComputer()
    motor = MotorReflexes(comp)

    await motor.settle_focus("ws-1", target_window="gedit", settle_seconds=0.01)
    executed = comp.commands_executed
    assert any("wmctrl -a 'gedit'" in cmd or "windowactivate" in cmd for cmd in executed)


@pytest.mark.asyncio
async def test_motor_human_type_with_clear_field():
    """human_type with clear_field=True clears the field before typing."""
    comp = _MockMotorComputer()
    motor = MotorReflexes(comp)

    res = await motor.human_type("ws-1", "new_content", delay_ms=10, clear_field=True)
    assert "new_content" in res
    executed = comp.commands_executed
    assert len(executed) >= 2
    assert any("ctrl+a BackSpace" in cmd for cmd in executed)
    assert any("xdotool type --delay 10 --clearmodifiers" in cmd for cmd in executed)


@pytest.mark.asyncio
async def test_agent_gui_type_preserves_existing_field_contents_by_default():
    """Typing into a grounded control must not clear unrelated active content."""
    comp = _MockMotorComputer()
    agent = ComputerUseAgent(computer_provider=comp)

    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_TYPE,
        "input",
        {"text": "additional text", "x": 100, "y": 200},
        "Type additional text",
    )
    assert trace.status == "SUCCESS"
    assert not any("ctrl+a BackSpace" in cmd for cmd in comp.commands_executed)
    assert comp.gui_actions[0].action == GUIActionType.CLICK
    assert (comp.gui_actions[0].x, comp.gui_actions[0].y) == (100, 200)


@pytest.mark.asyncio
async def test_agent_gui_type_clears_only_after_grounded_target_is_focused():
    """Explicit replacement clears the selected field after the visible focus click."""
    comp = _MockMotorComputer()
    events: list[tuple[str, object]] = []
    comp._docker_exec.side_effect = lambda cmd, timeout=60: (
        events.append(("exec", cmd)) or (0, "", "")
    )
    comp.gui_action.side_effect = lambda workspace_id, action, actor="operator": (
        events.append(("gui", action))
        or ScreenObservation(width=1280, height=800)
    )
    agent = ComputerUseAgent(computer_provider=comp)

    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_TYPE,
        "input",
        {"text": "replacement text", "x": 100, "y": 200, "clear_field": True},
        "Replace the current input text",
    )
    assert trace.status == "SUCCESS"
    click_index = next(
        i for i, (kind, value) in enumerate(events)
        if kind == "gui" and value.action == GUIActionType.CLICK
    )
    clear_index = next(
        i for i, (kind, value) in enumerate(events)
        if kind == "exec" and "ctrl+a BackSpace" in value
    )
    type_index = next(
        i for i, (kind, value) in enumerate(events)
        if kind == "exec" and "xdotool type --delay" in value
    )
    assert click_index < clear_index < type_index


# ---------------------------------------------------------------------------
# 4. Keyboard Accelerators in Prompts
# ---------------------------------------------------------------------------

def test_prompts_include_standard_keyboard_accelerators():
    """Prompts and action guidance must provide standard accelerators Alt+F, Ctrl+O, Ctrl+S, Ctrl+N."""
    full_prompt = prompts.COMPUTER_USE_SYSTEM_PROMPT
    compact_prompt = prompts.COMPUTER_USE_SYSTEM_PROMPT_COMPACT

    # Check for standard accelerators in the system prompts
    for p in (full_prompt, compact_prompt):
        assert "alt+f" in p.lower()
        assert "ctrl+o" in p.lower()
        assert "ctrl+s" in p.lower()
        assert "ctrl+n" in p.lower()
