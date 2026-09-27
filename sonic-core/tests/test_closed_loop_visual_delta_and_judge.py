"""Tests for Closed-Loop Visual Delta, Recovery Watchdog & Judge Truthfulness.

Covers:
1. ComputerUseAgent _consecutive_unchanged_screens active watchdog
   (refocus at 2, Alt+F accelerator at 3, abort before burning quota at >= 4).
2. Typing verification rejecting garbage text and wrong fields instead of falsely marking SUCCESS.
3. JudgeAgent requiring verified application state or artifacts for GUI actions with exit_code=None,
   rejecting terminal syntax errors, and verifying file artifacts on disk.
4. BossAgent not advancing phases on actions_taken > 0 when all actions failed.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.agents.judge import JudgeAgent, JudgeVerdict
from sonic.computer.models import GUIAction, GUIActionType, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
    ComputerWorldObservation,
)
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult
from sonic.sandbox.provider import ExecResult


def _mock_static_gui_provider(screen_b64: str = "same_screen_bytes"):
    """Mock provider that always returns the exact same screenshot hash."""
    provider = AsyncMock()
    screen_obs = ScreenObservation(
        width=1280,
        height=800,
        screenshot_base64=screen_b64,
        active_window="Desktop",
    )
    provider.screenshot.return_value = screen_obs
    provider.gui_action.return_value = screen_obs
    provider.status.return_value = MagicMock(
        active_application="TextEditor",
        open_applications=["TextEditor"],
        running_processes=["gedit"],
    )
    provider.terminal.return_value = MagicMock(stdout="__sonic_obs_ready__", exit_code=0)
    return provider


# =========================================================================
# 1. ComputerUseAgent Watchdog Tests
# =========================================================================


@pytest.mark.asyncio
async def test_watchdog_refocuses_window_at_two_unchanged_screens():
    provider = _mock_static_gui_provider("fixed_screen_hash_123")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "fixed_screen_hash_123"

    # Action 1: First unchanged screen
    trace1 = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "100,100",
        {"x": 100, "y": 100},
        "Click button",
    )
    assert agent._consecutive_unchanged_screens == 1
    assert trace1.status == ActionExecutionStatus.FAILED

    # Action 2: Second unchanged screen -> watchdog level 1: refocus window
    trace2 = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "100,100",
        {"x": 100, "y": 100},
        "Click button again",
    )
    assert agent._consecutive_unchanged_screens == 2
    assert "WATCHDOG ADAPTIVE RECOVERY (count=2)" in trace2.actual_observation
    assert "Triggered window refocus" in trace2.actual_observation

    # Verify provider was called to refocus active window
    gui_calls = [
        call.args[1] for call in provider.gui_action.call_args_list
        if hasattr(call.args[1], "action")
    ]
    refocus_actions = [
        a for a in gui_calls
        if a.action in (GUIActionType.SELECT_WINDOW, GUIActionType.HOTKEY)
    ]
    assert len(refocus_actions) > 0


@pytest.mark.asyncio
async def test_watchdog_triggers_alt_f_accelerator_at_three_unchanged_screens():
    provider = _mock_static_gui_provider("fixed_screen_hash_456")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "fixed_screen_hash_456"
    agent._consecutive_unchanged_screens = 2  # Already at 2

    # Action 3: Third unchanged screen -> watchdog level 2: Alt+F accelerator
    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "200,200",
        {"x": 200, "y": 200},
        "Click stuck element",
    )
    assert agent._consecutive_unchanged_screens == 3
    assert "WATCHDOG ADAPTIVE RECOVERY (count=3)" in trace.actual_observation
    assert "Alt+F" in trace.actual_observation

    # Verify provider was dispatched Alt+F
    gui_calls = [call.args[1] for call in provider.gui_action.call_args_list]
    alt_f_calls = [
        a for a in gui_calls
        if getattr(a, "key", "") == "Alt+F" or getattr(a, "text", "") == "Alt+F"
    ]
    assert len(alt_f_calls) > 0


@pytest.mark.asyncio
async def test_watchdog_aborts_run_mission_before_burning_quota():
    provider = _mock_static_gui_provider("perpetually_stuck_pixels")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "perpetually_stuck_pixels"

    # Mock choose_action to repeatedly attempt GUI_CLICK
    async def _mock_choose(goal, obs, step):
        return ComputerActionType.GUI_CLICK, "stuck_btn", {"x": 50, "y": 50}, "Click"

    agent.choose_action = AsyncMock(side_effect=_mock_choose)

    # Request 15 steps; watchdog must abort well before burning quota
    traces = await agent.run_mission("ws-1", "Test mission", steps=15)
    assert len(traces) <= 5
    assert agent._consecutive_unchanged_screens >= 4
    assert any("WATCHDOG_ABORT" in h.get("action", "") for h in agent.history)


# =========================================================================
# 2. Typing Verification Tests
# =========================================================================


@pytest.mark.asyncio
async def test_typing_verification_rejects_garbage_and_sentinel_text():
    provider = _mock_static_gui_provider("diff_screen_1")
    agent = ComputerUseAgent(computer_provider=provider)

    garbage_payloads = [
        {"text": "undefined"},
        {"text": "null"},
        {"text": "[object Object]"},
        {"text": "{'text': 'value'}"},
        {"text": "   "},
        {"text": ""},
    ]

    for p in garbage_payloads:
        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            "input_box",
            p,
            "Type into input",
        )
        assert trace.status == ActionExecutionStatus.BLOCKED, f"Payload {p} was not fail-closed: {trace.status}"
        assert "[TYPING VERIFICATION FAILED]" in trace.actual_observation


@pytest.mark.asyncio
async def test_typing_verification_rejects_wrong_field_targets():
    provider = _mock_static_gui_provider("diff_screen_2")
    agent = ComputerUseAgent(computer_provider=provider)

    wrong_targets = [
        "submit button",
        "close btn",
        "checkbox agreement",
        "radio option",
        "icon minimize",
    ]

    for target in wrong_targets:
        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            target,
            {"text": "valid_user_text"},
            f"Type into {target}",
        )
        assert trace.status == ActionExecutionStatus.BLOCKED, f"Target {target} was not fail-closed: {trace.status}"
        assert "[TYPING VERIFICATION FAILED]" in trace.actual_observation
        assert "wrong field" in trace.actual_observation.lower()


@pytest.mark.asyncio
async def test_typing_verification_accepts_valid_field_and_text():
    provider = _mock_static_gui_provider("diff_screen_3")
    agent = ComputerUseAgent(computer_provider=provider)

    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_TYPE,
        "search input",
        {"text": "my search query"},
        "Type query",
    )
    assert trace.status == ActionExecutionStatus.COMPLETED
    assert "Typed 15 chars" in trace.actual_observation


# =========================================================================
# 3. JudgeAgent Execution Proof and Verification Tests
# =========================================================================


@pytest.mark.asyncio
async def test_judge_rejects_unverified_gui_actions_with_none_exit_code():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Click button on desktop application")
    # Trace has GUI action with exit_code=None and unverified observation
    trace = {
        "action_type": "GUI_CLICK",
        "target": "500,500",
        "status": "COMPLETED",
        "exit_code": None,
        "actual_observation": "Clicked at (500, 500)",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Worker claims button was clicked",
    )

    eval_result = await judge.evaluate_sub_mission("ws-1", sub_m, result)
    assert eval_result.verdict == JudgeVerdict.FALSIFIED
    assert eval_result.self_attestation_rejected is True


@pytest.mark.asyncio
async def test_judge_rejects_terminal_syntax_errors_without_silent_skipping():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Deploy service using script")
    # Worker had a syntax error in trace
    traces = [
        {
            "action_type": "TERMINAL_EXEC",
            "target": "python -c 'print(hello world)'",
            "status": "FAILED",
            "exit_code": 1,
            "actual_observation": "SyntaxError: invalid syntax",
        },
        {
            "action_type": "TERMINAL_EXEC",
            "target": "echo ok",
            "status": "COMPLETED",
            "exit_code": 0,
            "actual_observation": "ok",
        },
    ]
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=traces,
        success=True,
        findings_summary="Deployment finished",
    )

    eval_result = await judge.evaluate_sub_mission("ws-1", sub_m, result)
    assert eval_result.verdict == JudgeVerdict.FALSIFIED
    assert "syntax error" in eval_result.notes.lower()


@pytest.mark.asyncio
async def test_judge_verifies_file_artifact_exists_and_is_non_empty():
    mock_computer = MagicMock()
    # First: file does not exist / empty
    mock_computer.read_file = AsyncMock(return_value="")
    judge = JudgeAgent(computer_provider=mock_computer)

    sub_m = SubMission(goal="Write report into summary.md")
    trace = {
        "action_type": "FILE_WRITE",
        "target": "summary.md",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Wrote 100 bytes to summary.md",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Summary created",
    )

    eval_empty = await judge.evaluate_sub_mission("ws-1", sub_m, result)
    assert eval_empty.verdict == JudgeVerdict.FALSIFIED
    assert "Rule 15 violation" in eval_empty.notes

    # Second: file exists and has non-empty content
    mock_computer.read_file = AsyncMock(return_value="# Project Summary\nAll metrics nominal.")
    eval_ok = await judge.evaluate_sub_mission("ws-1", sub_m, result)
    assert eval_ok.verdict == JudgeVerdict.CONFIRMED


# =========================================================================
# 4. BossAgent Phase Advance Tests
# =========================================================================


@pytest.mark.asyncio
async def test_boss_does_not_advance_phase_if_all_actions_failed():
    llm = AsyncMock()
    decomp_json = json.dumps({
        "thinking": "Initial attack attempt",
        "phase_name": "Phase 1",
        "sub_missions": [{"goal": "Attempt exploit", "max_steps": 2, "priority": 1}],
    })
    llm.complete = AsyncMock(return_value=MagicMock(content=decomp_json, reasoning_content=""))
    computer = AsyncMock()

    boss = BossAgent(computer_provider=computer, llm_router=llm, max_phases=3)

    # Worker executes actions but ALL actions fail
    failing_trace = ComputerDecisionTrace(
        action_type=ComputerActionType.TERMINAL_EXEC,
        target_resource="curl http://target/exploit",
        predicted_outcome="200 OK",
        actual_observation="Connection refused (exit 1)",
        status=ActionExecutionStatus.FAILED,
        exit_code=1,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent") as MockAgentClass:
        mock_agent = MagicMock()
        mock_agent.run_mission = AsyncMock(return_value=[failing_trace])
        MockAgentClass.return_value = mock_agent

        report = await boss.run(workspace_id="ws-1", objective="Exploit target system")

        # Boss must stop at Phase 1 because all actions failed; must NOT advance to Phase 2
        assert len(report.phases) == 1
        assert report.phases[0].results[0].actions_taken > 0
        assert report.phases[0].results[0].success is False
