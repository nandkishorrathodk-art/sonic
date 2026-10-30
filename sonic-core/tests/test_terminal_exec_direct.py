"""
Tests for ComputerUseAgent TERMINAL_EXEC execution path.
Ensures that CLI commands (like python -c, bash -c) run directly via computer.terminal()
and are NEVER intercepted or hijacked into GUI launch_application (Rule 5 & Rule 6).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    AgentStepResult,
    ComputerActionType,
    ComputerAutonomyLevel,
    EngineeringMissionMode,
)
from sonic.sandbox.provider import ExecResult


@pytest.mark.asyncio
async def test_terminal_exec_runs_directly_via_terminal():
    mock_comp = MagicMock()
    mock_comp.terminal = AsyncMock(return_value=ExecResult(
        command="python -c 'print(1+1)'",
        exit_code=0,
        stdout="2\n",
        stderr="",
    ))
    mock_comp.launch_application = AsyncMock(return_value=True)
    mock_comp.application_list = AsyncMock(return_value=["python", "bash", "calc"])

    agent = ComputerUseAgent(
        computer_provider=mock_comp,
        autonomy_level=ComputerAutonomyLevel.L3_AUTONOMOUS,
        mode=EngineeringMissionMode.GENERAL_ENGINEERING_MODE,
        tenant_id="test-tenant",
    )

    result = await agent.execute_action(
        "test-ws",
        ComputerActionType.TERMINAL_EXEC,
        "python -c 'print(1+1)'",
        {"command": "python -c 'print(1+1)'"},
        "Evaluate python expression",
    )

    # Must invoke terminal, NEVER launch_application
    mock_comp.terminal.assert_awaited_once()
    mock_comp.launch_application.assert_not_awaited()
    assert "2" in result.actual_observation
    assert result.exit_code == 0
    assert result.status == ActionExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_terminal_exec_strips_trailing_commas_and_semicolons():
    mock_comp = MagicMock()
    mock_comp.terminal = AsyncMock(return_value=ExecResult(
        command="hostname",
        exit_code=0,
        stdout="sandbox-host\n",
        stderr="",
    ))

    agent = ComputerUseAgent(
        computer_provider=mock_comp,
        autonomy_level=ComputerAutonomyLevel.L3_AUTONOMOUS,
        mode=EngineeringMissionMode.GENERAL_ENGINEERING_MODE,
        tenant_id="test-tenant",
    )

    # Command with trailing comma and semicolon and whitespace
    result = await agent.execute_action(
        "test-ws",
        ComputerActionType.TERMINAL_EXEC,
        "hostname,;",
        {"command": "  hostname,;  "},
        "Get hostname",
    )

    mock_comp.terminal.assert_awaited_once_with("test-ws", "hostname")
    assert result.status == ActionExecutionStatus.COMPLETED
    assert "sandbox-host" in result.actual_observation


@pytest.mark.asyncio
async def test_terminal_exec_empty_command_returns_agent_step_result():
    mock_comp = MagicMock()
    mock_comp.terminal = AsyncMock()

    agent = ComputerUseAgent(
        computer_provider=mock_comp,
        autonomy_level=ComputerAutonomyLevel.L3_AUTONOMOUS,
        mode=EngineeringMissionMode.GENERAL_ENGINEERING_MODE,
        tenant_id="test-tenant",
    )

    # Empty command
    result = await agent.execute_action(
        "test-ws",
        ComputerActionType.TERMINAL_EXEC,
        "",
        {"command": ""},
        "Run empty command",
    )

    mock_comp.terminal.assert_not_awaited()
    assert isinstance(result, AgentStepResult)
    assert result == AgentStepResult(False, "Empty command provided")
    assert result.success is False
    assert result.message == "Empty command provided"
    assert result.status == ActionExecutionStatus.FAILED
    assert result.actual_observation == "Empty command provided"


@pytest.mark.asyncio
async def test_terminal_exec_punctuation_only_returns_agent_step_result():
    mock_comp = MagicMock()
    mock_comp.terminal = AsyncMock()

    agent = ComputerUseAgent(
        computer_provider=mock_comp,
        autonomy_level=ComputerAutonomyLevel.L3_AUTONOMOUS,
        mode=EngineeringMissionMode.GENERAL_ENGINEERING_MODE,
        tenant_id="test-tenant",
    )

    # Command consisting only of commas, semicolons, and whitespace
    result = await agent.execute_action(
        "test-ws",
        ComputerActionType.TERMINAL_EXEC,
        ",;, ",
        {"command": "  , ;, ;  "},
        "Run punctuation command",
    )

    mock_comp.terminal.assert_not_awaited()
    assert isinstance(result, AgentStepResult)
    assert result.success is False
    assert result.message == "Empty command provided"
    success, msg = result
    assert success is False
    assert msg == "Empty command provided"

