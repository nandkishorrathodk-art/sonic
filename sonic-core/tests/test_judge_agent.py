"""
Unit tests for JudgeAgent (Independent Verification & Anti-Self-Attestation).
Ensures Rule 15 (Empirical Falsification) and Rule 1 (No self-attestation).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from sonic.agents.judge import JudgeAgent, JudgeVerdict
from sonic.computer_use.models_boss import SubMission, SubMissionResult
from sonic.sandbox.provider import ExecResult


@pytest.mark.asyncio
async def test_judge_rejects_self_attestation_without_proof():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Audit authentication logic")
    # Worker claimed status="VERIFIED" on a trace that had no concrete output
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "cat auth.py",
        "status": "VERIFIED",
        "exit_code": 1,
        "actual_observation": "FAIL-CLOSED: Host execution is disabled in HeadlessComputeProvider.",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Worker claims auth is vulnerable",
    )

    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-1",
    )

    assert evaluation.verdict == JudgeVerdict.FALSIFIED
    assert evaluation.self_attestation_rejected is True
    assert "Self-attestation rejected" in evaluation.notes


@pytest.mark.asyncio
async def test_judge_flags_complete_failure():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Create task tracker folder")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "mkdir sonic_validation_task_tracker",
        "status": "FAILED",
        "exit_code": 126,
        "actual_observation": "FAIL-CLOSED: Host execution is disabled in HeadlessComputeProvider.",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=False,
        findings_summary="Execution failed",
    )

    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-2",
    )

    assert evaluation.verdict == JudgeVerdict.FALSIFIED
    assert "no successful execution traces" in evaluation.notes


@pytest.mark.asyncio
async def test_judge_confirms_valid_execution():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Run task tracker script")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "python task_tracker.py",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Tasks listed: 0 pending, 1 completed",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Task tracker executed successfully",
    )

    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-3",
    )

    assert evaluation.verdict == JudgeVerdict.CONFIRMED
    assert evaluation.confidence >= 0.8


@pytest.mark.asyncio
async def test_judge_does_not_execute_worker_supplied_security_poc():
    mock_computer = MagicMock()
    mock_computer.terminal = AsyncMock(return_value=ExecResult(
        command="curl -s http://127.0.0.1:8080/api/users",
        exit_code=0,
        stdout='[{"id": 1, "role": "admin"}]',
        stderr="",
    ))

    judge = JudgeAgent(computer_provider=mock_computer)
    sub_m = SubMission(goal="Test for IDOR vulnerability on api/users")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "curl -s http://127.0.0.1:8080/api/users",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": '[{"id": 1, "role": "admin"}]',
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        key_discoveries=["IDOR vulnerability exposed admin user"],
        findings_summary="Found IDOR on /api/users",
    )

    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-4",
    )

    assert evaluation.verdict == JudgeVerdict.INCONCLUSIVE
    assert evaluation.self_attestation_rejected is True
    mock_computer.terminal.assert_not_awaited()


@pytest.mark.asyncio
async def test_judge_falsifies_missing_or_empty_claimed_file():
    mock_computer = MagicMock()
    mock_computer.read_file = AsyncMock(return_value="")  # File missing/empty
    judge = JudgeAgent(computer_provider=mock_computer)
    sub_m = SubMission(goal="create file env_discovery.json with telemetry")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "touch env_discovery.json",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Exit 0",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Created env_discovery.json",
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-file",
    )
    assert evaluation.verdict == JudgeVerdict.FALSIFIED
    assert "Rule 15 violation" in evaluation.notes


@pytest.mark.asyncio
async def test_judge_falsifies_gui_mission_without_window_or_gui_action():
    mock_computer = MagicMock()
    # wmctrl returns only panels, no application window
    mock_computer.terminal = AsyncMock(return_value=ExecResult(
        command="DISPLAY=:99 wmctrl -l 2>/dev/null | grep -v ' -1 '",
        exit_code=0,
        stdout="",
        stderr="",
    ))
    judge = JudgeAgent(computer_provider=mock_computer, llm_router=MagicMock())
    sub_m = SubMission(goal="Launch browser on desktop and observe UI", plane="gui")
    # Worker bypassed GUI using terminal curl
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "curl -s http://127.0.0.1:12000/health",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": '{"status":"ok"}',
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Observed health endpoint via curl",
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-gui",
    )
    assert evaluation.verdict == JudgeVerdict.INCONCLUSIVE
    assert "no successful GUI interaction receipts" in evaluation.notes


@pytest.mark.asyncio
async def test_judge_confirms_gui_mission_with_real_window():
    mock_computer = MagicMock()
    mock_computer.screenshot = AsyncMock(return_value=type(
        "Screen", (), {
            "screenshot_base64": "c2NyZWVu",
            "capture_age_ms": 1,
            "desktop_state": "READY",
            "active_window": "Terminal",
        }
    )())
    mock_computer.terminal = AsyncMock(return_value=ExecResult(
        command="DISPLAY=:99 wmctrl -l 2>/dev/null | grep -v ' -1 '",
        exit_code=0,
        stdout="0x02600003  1 8e8d478afea9 Terminal - root@8e8d478afea9: /\n",
        stderr="",
    ))
    vision_judge = MagicMock()
    vision_judge.complete = AsyncMock(return_value=type(
        "Response", (), {
            "content": '{"verdict":"CONFIRMED","visible_evidence":"Terminal window is visible"}'
        }
    )())
    judge = JudgeAgent(computer_provider=mock_computer, llm_router=vision_judge)
    sub_m = SubMission(goal="Launch terminal window on desktop", plane="gui")
    trace = {
        "action_type": "APP_LAUNCH",
        "target": "xfce4-terminal",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Launched and focused xfce4-terminal",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Terminal window opened successfully",
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-gui",
    )
    assert evaluation.verdict == JudgeVerdict.CONFIRMED
