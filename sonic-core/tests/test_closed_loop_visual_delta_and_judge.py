"""Tests for Closed-Loop Visual Delta, Recovery Watchdog & Judge Truthfulness.

Covers:
1. ComputerUseAgent _consecutive_unchanged_screens active watchdog
   (no recovery input; abort before burning quota at >= 4).
2. Typing verification rejecting garbage text and wrong fields instead of falsely marking SUCCESS.
3. JudgeAgent requiring verified application state or artifacts for GUI actions with exit_code=None,
   rejecting terminal syntax errors, and verifying file artifacts on disk.
4. BossAgent not advancing phases on actions_taken > 0 when all actions failed.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.agents.judge import JudgeAgent, JudgeVerdict
from sonic.computer.models import GUIActionType, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
)
from sonic.computer_use.models_boss import SubMission, SubMissionResult


@pytest.fixture(autouse=True)
def _keep_visual_grounding_offline(monkeypatch):
    """Use deterministic mocked grounding; these unit tests never call a model."""
    async def _grounded_target(*args, **kwargs):
        return (240, 180)

    monkeypatch.setattr(
        "sonic.computer_use.agent.resolve_ui_target_async",
        _grounded_target,
    )


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
async def test_unchanged_click_is_unverified_without_refocusing_window():
    provider = _mock_static_gui_provider("fixed_screen_hash_123")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "fixed_screen_hash_123"

    # An unchanged frame cannot prove whether a click focused a control.
    trace1 = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "100,100",
        {"x": 100, "y": 100},
        "Click button",
    )
    assert agent._consecutive_unchanged_screens == 1
    assert trace1.status == ActionExecutionStatus.UNVERIFIED
    assert trace1.recovery_attempted is False

    # Re-observation is safe; the watchdog must not inject focus or key input.
    trace2 = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "100,100",
        {"x": 100, "y": 100},
        "Click button again",
    )
    assert agent._consecutive_unchanged_screens == 2
    assert trace2.status == ActionExecutionStatus.UNVERIFIED
    assert "No recovery input was sent" in trace2.actual_observation
    gui_calls = [call.args[1] for call in provider.gui_action.call_args_list]
    assert len(gui_calls) == 2
    assert all(action.action == GUIActionType.CLICK for action in gui_calls)


@pytest.mark.asyncio
async def test_watchdog_never_injects_application_specific_shortcuts():
    provider = _mock_static_gui_provider("fixed_screen_hash_456")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "fixed_screen_hash_456"
    agent._consecutive_unchanged_screens = 2  # Already at 2

    # An unchanged frame in an arbitrary application must not cause Alt+F.
    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "200,200",
        {"x": 200, "y": 200},
        "Click stuck element",
    )
    assert agent._consecutive_unchanged_screens == 3
    assert trace.status == ActionExecutionStatus.UNVERIFIED
    assert "No recovery input was sent" in trace.actual_observation
    gui_calls = [call.args[1] for call in provider.gui_action.call_args_list]
    assert len(gui_calls) == 1
    assert gui_calls[0].action == GUIActionType.CLICK


@pytest.mark.asyncio
async def test_watchdog_aborts_run_mission_before_burning_quota():
    provider = _mock_static_gui_provider("perpetually_stuck_pixels")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "perpetually_stuck_pixels"

    # Mock choose_action to repeatedly attempt GUI_CLICK
    async def _mock_choose(goal, obs, step):
        return ComputerActionType.GUI_CLICK, "50,50", {"x": 50, "y": 50}, "Click"

    agent.choose_action = AsyncMock(side_effect=_mock_choose)

    # Request 15 steps; watchdog must abort well before burning quota
    traces = await agent.run_mission("ws-1", "Test mission", steps=15)
    assert len(traces) <= 5
    assert agent._consecutive_unchanged_screens >= 4
    assert any("WATCHDOG_ABORT" in h.get("action", "") for h in agent.history)
    assert all(
        call.args[1].action == GUIActionType.CLICK
        for call in provider.gui_action.call_args_list
    )


@pytest.mark.asyncio
async def test_unchanged_focus_click_does_not_send_escape_before_typing():
    provider = _mock_static_gui_provider("unchanged_dialog_frame")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "unchanged_dialog_frame"

    click_trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "text field",
        {"x": 240, "y": 180},
        "Focus the visible text field",
    )

    assert click_trace.status == ActionExecutionStatus.UNVERIFIED
    assert click_trace.recovery_attempted is False
    assert provider.gui_action.await_count == 1
    assert provider.gui_action.await_args.args[1].action == GUIActionType.CLICK


@pytest.mark.asyncio
async def test_unchanged_typing_and_keypress_do_not_trigger_recovery():
    provider = _mock_static_gui_provider("unchanged_input_frame")
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_screenshot_b64 = "unchanged_input_frame"

    typing_trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_TYPE,
        "text field",
        {"x": 240, "y": 180, "text": "folder name"},
        "Enter text in the focused field",
    )
    keypress_trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_KEYPRESS,
        "text field",
        {"key": "Return"},
        "Submit the focused field",
    )

    assert typing_trace.status == ActionExecutionStatus.UNVERIFIED
    assert keypress_trace.status == ActionExecutionStatus.UNVERIFIED
    assert typing_trace.recovery_attempted is False
    assert keypress_trace.recovery_attempted is False
    actions = [call.args[1].action for call in provider.gui_action.call_args_list]
    assert actions == [
        GUIActionType.CLICK,
        GUIActionType.TYPE,
        GUIActionType.KEYPRESS,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("updated_text", "changed_region", "expected_status", "expected_changed"),
    [
        ("Document - Saved", None, ActionExecutionStatus.SUCCESS, True),
        ("Document - Untitled", None, ActionExecutionStatus.UNVERIFIED, False),
        (
            "Document - Untitled",
            (10, 20, 90, 70),
            ActionExecutionStatus.SUCCESS,
            True,
        ),
    ],
)
async def test_state_only_gui_receipt_verifies_delta_without_screenshot(
    updated_text, changed_region, expected_status, expected_changed
):
    class StateReceiptProvider:
        async def gui_action_state(self, workspace_id, action, actor="operator"):
            return ScreenObservation(
                screenshot_base64="",
                capture_age_ms=20,
                controls_age_ms=15,
                perception_version=2,
                width=1280,
                height=800,
                active_window="Document",
                visible_text=updated_text,
                changed_region=changed_region,
                detected_controls=[],
                desktop_state="STATE_ONLY",
            )

        async def gui_action(self, workspace_id, action, actor="operator"):
            raise AssertionError("state-only path should dispatch via gui_action_state")

        async def status(self, workspace_id):
            return MagicMock(active_application="Document", active_window="Document")

    provider = StateReceiptProvider()
    agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
    agent._last_observed_screen = ScreenObservation(
        screenshot_base64="",
        capture_age_ms=20,
        controls_age_ms=15,
        perception_version=1,
        width=1280,
        height=800,
        active_window="Document",
        visible_text="Document - Untitled",
        detected_controls=[],
        desktop_state="STATE_ONLY",
    )

    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "100,100",
        {"x": 100, "y": 100},
        "Save document",
    )

    assert trace.status == expected_status, trace.actual_observation
    assert trace.state_changed is expected_changed
    assert trace.verification_evidence.startswith("accessibility_state_changed=")
    assert "ACCESSIBILITY STATE" in trace.actual_observation or expected_changed is False
    assert agent._last_screenshot_b64 == ""
    assert agent._last_gui_state_observation.desktop_state == "STATE_ONLY"


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
    assert eval_result.verdict == JudgeVerdict.INCONCLUSIVE
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
