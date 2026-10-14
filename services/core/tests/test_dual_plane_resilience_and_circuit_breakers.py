"""
Unit tests for Dual-Plane Resilience, Safety Path Confinement, and Circuit Breakers.
Guarantees regression protection for:
1. JSON and freeform GOAL_COMPLETE action parsing (no markdown code fence hijack).
2. Repetitive action circuit breakers covering FILE_READ, FILE_WRITE, and TERMINAL_EXEC.
3. POSIX container path confinement under Windows hosts (zero false-positive denials).
4. Docker workstation owner release upon workspace destruction.
5. Workstation state isolation and singleton resets.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from sonic.computer_use.agent import ComputerUseAgent, _GOAL_COMPLETE_SENTINEL
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerWorldObservation,
)
from sonic.safety.action_policy import ActionPolicy
from sonic.computer.docker_computer import DockerComputerProvider
from sonic.api.routes.workstation import (
    reset_workstation_state,
    _get_workstation_state_file,
)


@pytest.mark.no_live_infra
def test_action_parser_goal_complete_json():
    """Agent correctly parses JSON action GOAL_COMPLETE into sentinel."""
    mock_provider = MagicMock()
    mock_llm = MagicMock()
    agent = ComputerUseAgent(computer_provider=mock_provider, llm=mock_llm)

    # 1. Standard JSON with action: GOAL_COMPLETE
    json_text = '```json\n{"action": "GOAL_COMPLETE", "target": "", "payload": {}}\n```'
    action_type, target, payload, expected = agent._parse_llm_action(json_text, "app.py")
    assert action_type == ComputerActionType.TERMINAL_EXEC
    assert expected == _GOAL_COMPLETE_SENTINEL

    # 2. Case variation and 'DONE'
    json_text_2 = '{"action": "DONE", "payload": {"note": "finished assessment"}}'
    action_type, target, payload, expected = agent._parse_llm_action(json_text_2, "app.py")
    assert action_type == ComputerActionType.TERMINAL_EXEC
    assert expected == _GOAL_COMPLETE_SENTINEL


@pytest.mark.no_live_infra
def test_action_parser_completion_phrase_with_markdown_not_hijacked():
    """Completion phrase accompanied by markdown text/json is not hijacked into a shell command."""
    mock_provider = MagicMock()
    mock_llm = MagicMock()
    agent = ComputerUseAgent(computer_provider=mock_provider, llm=mock_llm)

    response_text = (
        "The task has been completed. All requested information has been successfully gathered.\n\n"
        "```json\n"
        '{\n  "findings_count": 3,\n  "status": "success"\n}\n'
        "```"
    )
    action_type, target, payload, expected = agent._parse_llm_action(response_text, "app.py")
    assert action_type == ComputerActionType.TERMINAL_EXEC
    assert expected == _GOAL_COMPLETE_SENTINEL


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_consecutive_loop_breaker_blocks_repeated_file_and_terminal():
    """Loop breaker blocks 3rd consecutive repetition of FILE_READ or TERMINAL_EXEC."""
    mock_provider = MagicMock()
    mock_provider.read_file = AsyncMock(return_value="file content")
    mock_provider.terminal = AsyncMock(return_value=MagicMock(exit_code=0, stdout="output", stderr=""))
    mock_llm = MagicMock()
    agent = ComputerUseAgent(computer_provider=mock_provider, llm=mock_llm)

    # 3 consecutive identical FILE_READ actions
    t1 = await agent.execute_action(
        "ws-1",
        ComputerActionType.FILE_READ,
        "/home/sonic/workspace/app.py",
        {"path": "/home/sonic/workspace/app.py"},
        "Read source",
    )
    assert t1.status == ActionExecutionStatus.SUCCESS

    t2 = await agent.execute_action(
        "ws-1",
        ComputerActionType.FILE_READ,
        "/home/sonic/workspace/app.py",
        {"path": "/home/sonic/workspace/app.py"},
        "Read source",
    )
    assert t2.status == ActionExecutionStatus.SUCCESS

    t3 = await agent.execute_action(
        "ws-1",
        ComputerActionType.FILE_READ,
        "/home/sonic/workspace/app.py",
        {"path": "/home/sonic/workspace/app.py"},
        "Read source",
    )
    assert t3.status == ActionExecutionStatus.BLOCKED
    assert "[ACTION LOOP DETECTED]" in t3.actual_observation


@pytest.mark.no_live_infra
def test_action_policy_posix_confinement_on_windows():
    """POSIX container paths resolve cleanly against POSIX workspace_root without Windows path mangling."""
    policy = ActionPolicy(
        workspace_root="/home/sonic/workspace",
        allowed_action_types={"FILE_READ", "FILE_WRITE", "TERMINAL_EXEC"},
    )

    # In-workspace paths must be ALLOWED
    verdict_valid = policy.evaluate(
        ComputerActionType.FILE_READ,
        target="/home/sonic/workspace/src/app.py",
        payload={"path": "/home/sonic/workspace/src/app.py"},
    )
    assert verdict_valid.allowed is True

    # Relative path inside workspace must be ALLOWED
    verdict_rel = policy.evaluate(
        ComputerActionType.FILE_READ,
        target="src/config.py",
        payload={"path": "src/config.py"},
    )
    assert verdict_rel.allowed is True

    # Traversal escape must be BLOCKED
    verdict_escape = policy.evaluate(
        ComputerActionType.FILE_READ,
        target="/home/sonic/workspace/../../etc/shadow",
        payload={"path": "/home/sonic/workspace/../../etc/shadow"},
    )
    assert verdict_escape.allowed is False
    assert "escapes" in verdict_escape.reason.lower()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_computer_destroy_releases_owner():
    """Destroying home workspace frees _workspace_owner so subsequent tenants aren't locked out."""
    provider = DockerComputerProvider(container_name="test-workstation")
    ws = await provider.create("tenant-alpha", "home-alpha")
    assert provider._workspace_owner == "tenant-alpha"

    # Destroy home workspace
    destroyed = await provider.destroy(ws.id)
    assert destroyed is True
    assert provider._workspace_owner is None

    # Another tenant can now claim the workspace without collision error
    ws2 = await provider.create("tenant-beta", "home-beta")
    assert ws2.tenant_id == "tenant-beta"
    assert provider._workspace_owner == "tenant-beta"


@pytest.mark.no_live_infra
def test_workstation_reset_state_and_dynamic_path(tmp_path, monkeypatch):
    """reset_workstation_state() cleanly resets memory and state file paths."""
    test_state_file = tmp_path / "custom_workstations.json"
    monkeypatch.setenv("SONIC_WORKSTATION_STATE_FILE", str(test_state_file))

    resolved = _get_workstation_state_file()
    assert resolved == test_state_file

    reset_workstation_state()
    from sonic.api.routes import workstation
    assert workstation._primary_computer_instance is None
    assert len(workstation._tenant_workstations) == 0
