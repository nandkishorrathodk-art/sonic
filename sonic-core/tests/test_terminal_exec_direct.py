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
