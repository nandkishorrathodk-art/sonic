"""
Tests for BossAgent plane routing, FILE_WRITE JSON-unwrapping, and zero-hallucination failure summaries.
Validates Dual-Plane Concurrency (Mandates 1, 5) and Empirical Falsification (Rule 15).
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
)
from sonic.computer_use.models_boss import SubMission


def test_sub_mission_plane_inference_dual_plane():
    """Verify clean plane separation between headless coding and GUI browser verification."""
    # 1. Directory creation & python server authoring -> headless
    s1 = SubMission(
        goal="Create directory /home/sonic/workspace/task_manager_app and author server.py web service"
    )
    assert s1.plane == "headless"

    # 2. Launching background server and verifying with curl -> headless
    s2 = SubMission(
        goal="Launch server.py on port 5000 in background and test endpoints via curl"
    )
    assert s2.plane == "headless"

    # 3. Code inspection, file read, grep -> headless
    s3 = SubMission(
        goal="Inspect server.py code and check open ports with netstat or ss"
    )
    assert s3.plane == "headless"

    # 4. Opening browser and visually verifying UI -> gui
    s4 = SubMission(
        goal="Launch google-chrome to http://127.0.0.1:5000 and visually verify the task manager dark UI"
    )
    assert s4.plane == "gui"

    # 5. Clicking controls in graphical application -> gui
    s5 = SubMission(
        goal="Click button on screen to add a new task and verify visual completion"
    )
    assert s5.plane == "gui"


@pytest.mark.asyncio
async def test_requested_plane_overrides_decomposition_plane():
    router = MagicMock()
    router.complete = AsyncMock(return_value=type(
        "Response",
        (),
        {
            "content": json.dumps({
                "thinking": "One focused worker.",
                "phase_name": "Inspection",
                "sub_missions": [
                    {"goal": "Inspect the repository", "plane": "gui", "max_steps": 2}
                ],
            })
        },
    )())
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=router,
        requested_plane="headless",
    )

    phase = await boss._strategic_decomposition("inspect the repository")

    assert phase is not None
    assert phase.sub_missions[0].plane == "headless"


def test_shared_display_serializes_gui_workers_only():
    computer = type(
        "DisplayProvider",
        (),
        {"screenshot": lambda self: None, "gui_action": lambda self: None},
    )()
    boss = BossAgent(
        computer_provider=computer,
        llm_router=MagicMock(),
        max_parallel_workers=4,
    )

    assert boss._worker_limit_for_wave([
        SubMission(goal="inspect source", plane="headless"),
        SubMission(goal="run tests", plane="headless"),
    ]) == 4
    assert boss._worker_limit_for_wave([
        SubMission(goal="inspect source", plane="headless"),
        SubMission(goal="click the visible control", plane="gui"),
    ]) == 1


@pytest.mark.asyncio
async def test_file_write_unwraps_json_envelope():
    """FILE_WRITE must unwrap JSON string payloads into raw code, avoiding JSON-string pollution on disk."""
    mock_comp = MagicMock()
    mock_comp.write_file = AsyncMock(return_value=True)

    agent = ComputerUseAgent(
        computer_provider=mock_comp,
        llm_router=MagicMock(),
        tenant_id="test-tenant",
    )

    # Simulate LLM outputting a JSON string payload
    raw_json_payload = json.dumps({
        "path": "/home/sonic/workspace/app/server.py",
        "content": "from http.server import HTTPServer\nprint('Server started')\n"
    })

    result = await agent.execute_action(
        workspace_id="ws-test",
        action_type=ComputerActionType.FILE_WRITE,
        target_resource="/home/sonic/workspace/app/server.py",
        payload=raw_json_payload,
        predicted_outcome="File written successfully",
    )

    # Verify write_file was called with the unwrapped raw python code, NOT the JSON envelope!
    mock_comp.write_file.assert_awaited_once_with(
        "ws-test",
        "/home/sonic/workspace/app/server.py",
        "from http.server import HTTPServer\nprint('Server started')\n"
    )


@pytest.mark.asyncio
async def test_summarize_sub_agent_traces_deterministic_on_failure():
    """When all worker traces fail, _summarize_sub_agent_traces must return factual failure without LLM hallucination."""
    mock_comp = MagicMock()
    boss = BossAgent(
        computer_provider=mock_comp,
        llm_router=MagicMock(),
        tenant_id="test-tenant",
    )

    traces = [
        ComputerDecisionTrace(
            step_index=1,
            action_type=ComputerActionType.TERMINAL_EXEC,
            target_resource="mkdir -p /app",
            status=ActionExecutionStatus.FAILED,
            exit_code=1,
            actual_observation="Permission denied",
            predicted_outcome="Directory created",
        ),
        ComputerDecisionTrace(
            step_index=2,
            action_type=ComputerActionType.FILE_WRITE,
            target_resource="/app/server.py",
            status=ActionExecutionStatus.BLOCKED,
            exit_code=126,
            actual_observation="Policy block",
            predicted_outcome="File written",
        ),
    ]

    summary = await boss._summarize_sub_agent_traces("Setup app directory and server.py", traces)
    assert "Execution failed: no verified progress" in summary
    assert "TERMINAL_EXEC" in summary
    assert "Permission denied" in summary
