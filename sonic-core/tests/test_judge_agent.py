"""
Unit tests for JudgeAgent (Independent Verification & Anti-Self-Attestation).
Ensures Rule 15 (Empirical Falsification) and Rule 1 (No self-attestation).
"""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from sonic.agents.judge import JudgeAgent, JudgeVerdict
from sonic.computer_use.models_boss import SubMission, SubMissionResult
from sonic.sandbox.provider import ExecResult


def _png_data(width=1280, height=800):
    image = BytesIO()
    Image.new("RGB", (width, height), color=(32, 64, 96)).save(image, format="PNG")
    return image.getvalue()


def _base64_png(width=1280, height=800):
    return base64.b64encode(_png_data(width, height)).decode("ascii")


def _jpeg_data(width=1280, height=800):
    image = BytesIO()
    Image.new("RGB", (width, height), color=(96, 64, 32)).save(image, format="JPEG")
    return image.getvalue()


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
        "verification_evidence": (
            "Independent task tracker check confirms the task list contains one completed task."
        ),
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
async def test_judge_does_not_accept_generic_success_output_as_completion_proof():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Run task tracker script")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "echo completed successfully",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "completed successfully",
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
    )

    assert evaluation.verdict == JudgeVerdict.INCONCLUSIVE
    assert "lacks independent task evidence" in evaluation.notes


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
async def test_judge_does_not_treat_fresh_avcs_reflexes_as_gui_completion():
    mock_computer = MagicMock()
    mock_computer.screenshot = AsyncMock(return_value=type(
        "Screen", (), {
            "screenshot_base64": _base64_png(),
            "width": 1280,
            "height": 800,
            "avcs_state": {
                "engine": "avcs_reflex",
                "frame_index": 1,
                "scene_hash": "fresh-scene",
                "width": 1280,
                "height": 800,
                "scene_changed": True,
                "regions": [{"x": 1, "y": 2}],
                "events": [],
            },
            "desktop_state": "INTERACTIVE",
            "capture_age_ms": 0,
        }
    )())
    router = MagicMock()
    router.complete = AsyncMock(return_value=type(
        "Response", (), {
            "content": (
                '{"verdict":"INCONCLUSIVE","confidence":0.7,'
                '"visible_evidence":"No task-specific postcondition is visible."}'
            )
        }
    )())
    judge = JudgeAgent(computer_provider=mock_computer, llm_router=router)
    sub_mission = SubMission(goal="Save the completed report", plane="gui")
    result = SubMissionResult(
        sub_mission_id=sub_mission.id,
        goal=sub_mission.goal,
        traces=[{
            "action_type": "GUI_CLICK",
            "target": "save control",
            "status": "SUCCESS",
            "actual_observation": "Click dispatched",
        }],
        success=True,
    )

    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_mission,
        result=result,
    )

    assert evaluation.verdict == JudgeVerdict.INCONCLUSIVE
    assert any("AVCS frame 1" in item for item in evaluation.evidence_items)
    request = router.complete.await_args.args[0]
    assert request.task_type == "vision"
    assert request.messages[-1].images[0].base64 == _base64_png()
    assert "Vision-model judgment" in evaluation.evidence_items[-1]


@pytest.mark.asyncio
async def test_judge_confirms_gui_mission_with_real_window():
    mock_computer = MagicMock()
    mock_computer.screenshot = AsyncMock(return_value=type(
        "Screen", (), {
            "screenshot_base64": _base64_png(),
            "width": 1280,
            "height": 800,
            "avcs_state": {
                "engine": "avcs_reflex",
                "frame_index": 1,
                "scene_hash": "fresh-scene",
                "width": 1280,
                "height": 800,
            },
            "desktop_state": "INTERACTIVE",
            "active_window": "Terminal",
            "capture_age_ms": 0,
        }
    )())
    vision_judge = MagicMock()
    vision_judge.complete = AsyncMock(return_value=type(
        "Response", (), {
            "content": (
                '{"verdict":"CONFIRMED","confidence":0.97,'
                '"visible_evidence":"The requested terminal window is visible."}'
            )
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
    request = vision_judge.complete.await_args.args[0]
    assert request.task_type == "vision"
    assert request.messages[-1].images[0].base64 == _base64_png()
    assert "Vision-model judgment" in evaluation.evidence_items[-1]


@pytest.mark.asyncio
async def test_judge_accepts_browser_actions_in_gui_mission():
    mock_computer = MagicMock()
    mock_computer.screenshot = AsyncMock(return_value=type(
        "Screen", (), {
            "screenshot_base64": _base64_png(),
            "width": 1280,
            "height": 800,
            "avcs_state": {
                "engine": "avcs_reflex",
                "frame_index": 1,
                "scene_hash": "fresh-scene",
                "width": 1280,
                "height": 800,
            },
            "desktop_state": "INTERACTIVE",
            "active_window": "Browser",
            "capture_age_ms": 0,
        }
    )())
    vision_judge = MagicMock()
    vision_judge.complete = AsyncMock(return_value=type(
        "Response", (), {
            "content": (
                '{"verdict":"CONFIRMED","confidence":0.97,'
                '"visible_evidence":"The requested dashboard is visible in the page."}'
            )
        }
    )())
    judge = JudgeAgent(computer_provider=mock_computer, llm_router=vision_judge)
    sub_m = SubMission(goal="Navigate to local portal and verify dashboard", plane="gui")
    trace = {
        "action_type": "BROWSER_NAVIGATE",
        "target": "http://127.0.0.1:8080/dashboard",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Navigated to http://127.0.0.1:8080/dashboard",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Navigated to dashboard successfully",
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-browser",
    )
    assert evaluation.verdict == JudgeVerdict.CONFIRMED
    request = vision_judge.complete.await_args.args[0]
    assert request.task_type == "vision"
    assert request.messages[-1].images[0].base64 == _base64_png()
    assert "Vision-model judgment" in evaluation.evidence_items[-1]


@pytest.mark.asyncio
async def test_judge_does_not_falsify_on_git_branch_python_args_shell_globs():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Inspect git repository and python code")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "git branch; ls *.py",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "* main\n*args in func\n*.py files matched\n* bullet point",
        "verification_evidence": "Repository status and files inspected cleanly.",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Inspected repository on * main branch",
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-git",
    )
    # Must NOT be falsified due to * main or *args or *.py
    assert evaluation.verdict == JudgeVerdict.CONFIRMED


@pytest.mark.asyncio
async def test_judge_falsifies_actual_unsaved_buffer_indicators():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Edit document in editor")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "nano doc.txt",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Editor opened with *untitled buffer [modified]",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Edited file",
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-edit",
    )
    assert evaluation.verdict == JudgeVerdict.FALSIFIED
    assert "unsaved" in evaluation.notes.lower()


@pytest.mark.asyncio
async def test_judge_catches_unsaved_window_title_markers():
    mock_computer = MagicMock()
    mock_computer.status = AsyncMock(return_value=type(
        "Status", (), {
            "windows": ["* main.py - Editor", "Terminal"],
            "active_application": "Editor",
        }
    )())
    judge = JudgeAgent(computer_provider=mock_computer)
    sub_m = SubMission(goal="Update code file")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "code main.py",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Opened file",
        "verification_evidence": "Edited main.py",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Updated code",
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-editor",
    )
    assert evaluation.verdict == JudgeVerdict.FALSIFIED
    assert "unsaved" in evaluation.notes.lower()


@pytest.mark.asyncio
async def test_judge_allows_routine_open_port_and_bypass_markers():
    judge = JudgeAgent()
    sub_m = SubMission(goal="Check network service ports and caching")
    trace = {
        "action_type": "TERMINAL_EXEC",
        "target": "netstat -tuln",
        "status": "COMPLETED",
        "exit_code": 0,
        "actual_observation": "Active Internet connections: tcp 0.0.0.0:8080 LISTEN",
        "verification_evidence": "Service is actively listening on port 8080.",
    }
    result = SubMissionResult(
        sub_mission_id=sub_m.id,
        goal=sub_m.goal,
        traces=[trace],
        success=True,
        findings_summary="Verified open port 8080 and bypassed cache successfully",
        key_discoveries=["Service is on open port 8080"],
    )
    evaluation = await judge.evaluate_sub_mission(
        workspace_id="test-ws",
        sub_mission=sub_m,
        result=result,
        worker_agent_id="sub-agent-net",
    )
    # Should NOT be flagged as INCONCLUSIVE by vuln_markers check
    assert evaluation.verdict == JudgeVerdict.CONFIRMED
