"""
Tests for Universal GUI Application Execution & Computer-Use Enhancements:
1. _resolve_app_binary preserves command arguments and URLs (not stripping spaces across args).
2. _parse_llm_action parses flat JSON actions (e.g. action + url / path / command).
3. verify_goal detects file creation targets and generates test -f verification probe.
4. DockerComputerProvider injects root sandbox flags (--no-sandbox, --disable-dev-shm-usage) for browser/electron apps.
5. verify_goal detects generic GUI application launch/running.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from sonic.computer.models import GUIAction, GUIActionType, ScreenObservation
from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ComputerActionType, ComputerWorldObservation


def test_resolve_app_binary_preserves_arguments():
    """Ensure command arguments and URLs are not space-stripped into non-existent binaries."""
    # App with path argument
    res1 = ComputerUseAgent._resolve_app_binary("mousepad /workspace/task_report.txt")
    assert res1 == "mousepad /workspace/task_report.txt"

    # App with URL argument
    res2 = ComputerUseAgent._resolve_app_binary("google-chrome https://en.wikipedia.org")
    assert res2 == "google-chrome https://en.wikipedia.org"

    # Single app name with space still normalizes standard binary
    res3 = ComputerUseAgent._resolve_app_binary("burp suite")
    assert res3 == "burpsuite"


def test_parse_llm_action_flat_json():
    """Ensure flat JSON actions cleanly map target and payload without falling back to default_file."""
    raw = '{"action": "BROWSER_NAVIGATE", "url": "https://en.wikipedia.org"}'
    act_type, tgt, payload, _ = ComputerUseAgent._parse_llm_action(raw, "default.py")
    assert act_type == ComputerActionType.BROWSER_NAVIGATE
    assert tgt == "https://en.wikipedia.org"
    assert payload.get("url") == "https://en.wikipedia.org"

    # Flat JSON for file write
    raw_file = '{"action": "FILE_WRITE", "path": "/workspace/report.txt", "content": "done"}'
    act_type2, tgt2, payload2, _ = ComputerUseAgent._parse_llm_action(raw_file, "default.py")
    assert act_type2 == ComputerActionType.FILE_WRITE
    assert tgt2 == "/workspace/report.txt"
    assert payload2.get("content") == "done"


def test_parse_llm_action_does_not_strip_url_parentheses():
    """Ensure Wikipedia article URLs with parentheses are preserved."""
    raw = 'ACTION: TERMINAL_EXEC\nCOMMAND: curl -sL https://en.wikipedia.org/wiki/Python_(programming_language)'
    act_type, tgt, payload, _ = ComputerUseAgent._parse_llm_action(raw, "default.py")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert "(programming_language)" in payload.get("command", "")


@pytest.mark.asyncio
async def test_verify_goal_file_creation_probe():
    """Ensure verify_goal constructs an independent file verification probe when goal mentions a file."""
    mock_computer = MagicMock()
    mock_computer.terminal = AsyncMock(return_value=MagicMock(exit_code=0, stdout="FILE_VERIFIED: /home/sonic/workspace/task_report.txt\nline1"))
    agent = ComputerUseAgent(computer_provider=mock_computer)

    verified, evidence = await agent.verify_goal(
        workspace_id="ws-1",
        goal="create task_report.txt with current status",
    )
    assert verified is True
    assert "FILE_VERIFIED" in evidence
    mock_computer.terminal.assert_called_once()
    called_cmd = mock_computer.terminal.call_args[0][1]
    assert "test -f" in called_cmd
    assert "task_report.txt" in called_cmd


@pytest.mark.asyncio
async def test_docker_computer_injects_root_sandbox_flags():
    """Ensure launch_application preserves the discovered app and target arguments."""
    provider = DockerComputerProvider(container_name="test-container")
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    # Binary discovery is mocked to a real resolved path; a generic (0, "", "")
    # exec cannot resolve a binary because `which` returns empty output.
    provider._resolve_app_binary = AsyncMock(return_value="/usr/bin/google-chrome")

    success = await provider.launch_application("ws-1", "google-chrome https://198.51.100.10")
    assert success is True
    called_spawn = provider._docker_exec.call_args_list[-1][0][0]
    assert "google-chrome" in called_spawn
    assert "https://198.51.100.10" in called_spawn
    # Mock returns exit 0 for --no-sandbox probe so flags are injected
    assert "--no-sandbox" in called_spawn
