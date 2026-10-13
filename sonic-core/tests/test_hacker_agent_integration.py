"""
SONIC A-SEA — Hacker Cortex, Motor Reflexes, Scratchpad HUD & Wire Telemetry Integration Tests
=============================================================================================
Verifies end-to-end integration of:
  1. Hacker Scratchpad loot extraction and HUD prompt injection in ComputerUseAgent.
  2. Wire Telemetry HTTP stream recording and HUD prompt injection in ComputerUseAgent.
  3. System 1 Motor Reflexes (human typing cadence, hotkey navigation, tab budget).
  4. Reflexive modal backtracking on blocking UI overlays.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.computer.models import ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerWorldObservation,
)


def _mock_provider():
    """Build a mock ComputerProvider with async methods."""
    p = AsyncMock()
    p.screenshot.return_value = ScreenObservation(width=1280, height=800)
    p.status.return_value = MagicMock(
        active_application="Desktop",
        open_applications=["Desktop"],
        running_processes=["xvfb", "xfce4-panel"],
    )
    p.list_files.return_value = []
    p.git_action.return_value = MagicMock(branch="main", is_clean=True)
    p.terminal.return_value = MagicMock(
        stdout="OK", stderr="", exit_code=0
    )
    p.gui_action.return_value = ScreenObservation(width=1280, height=800)
    return p


class TestHackerAgentIntegration:
    """Test suite for integrated AI Human Hacker capabilities in ComputerUseAgent."""

    @pytest.mark.asyncio
    async def test_scratchpad_hud_injected_into_reasoning_context(self):
        """Discovered tokens must automatically populate the Scratchpad HUD in prompt context."""
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)

        # Pre-populate or extract a token
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        agent.scratchpad.extract_from_text(f"Token discovered: {jwt} in ?user_id=42", source="recon")

        obs = ComputerWorldObservation(
            visible_text="Dashboard login",
            terminal_output="",
            working_directory="/home/daytona/workspace",
        )

        _, obs_summary = agent._build_reasoning_context(
            goal="Penetrate target app",
            observation=obs,
            step_index=1,
            primary_file="",
            test_file="",
        )

        assert "WORKING MEMORY & ARTIFACT LEDGER:" in obs_summary
        assert "recon_jwt_1=eyJhbGciOi... (truncated)" in obs_summary
        assert "user_id" in obs_summary

    @pytest.mark.asyncio
    async def test_wire_telemetry_injected_into_reasoning_context(self):
        """Recorded network HTTP transactions must appear in prompt observation."""
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)

        agent.wire_telemetry.record_wire_event(
            method="POST",
            url="http://target.internal/api/v1/auth",
            status_code=401,
            response_body='{"error":"invalid token"}',
        )

        obs = ComputerWorldObservation(
            visible_text="Login screen",
            terminal_output="",
            working_directory="/home/daytona/workspace",
        )

        _, obs_summary = agent._build_reasoning_context(
            goal="Test auth flow",
            observation=obs,
            step_index=2,
            primary_file="",
            test_file="",
        )

        assert "LAST ACTION NETWORK WIRE (HTTP Stream):" in obs_summary
        assert "POST /api/v1/auth -> 401 Unauthorized" in obs_summary
        assert '{"error":"invalid token"}' in obs_summary

    @pytest.mark.asyncio
    async def test_terminal_curl_records_wire_and_loot(self):
        """Running a curl command via TERMINAL_EXEC extracts tokens and logs wire telemetry."""
        provider = _mock_provider()
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwiaWF0IjoxNTE2MjM5MDIyfQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        provider.terminal.return_value = MagicMock(
            exit_code=0,
            stdout=f"HTTP/1.1 200 OK\r\nAuthorization: Bearer {jwt}\r\n\r\nOK",
            stderr="",
        )

        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.TERMINAL_EXEC,
            "curl -s http://victim.internal/login",
            {"command": "curl -s http://victim.internal/login"},
            "Request auth token",
        )

        assert trace.status == ActionExecutionStatus.COMPLETED
        # Check scratchpad loot
        assert len(agent.scratchpad.tokens) >= 1
        # Check wire telemetry
        events = agent.wire_telemetry.fetch_latest_wire_events_sync(limit=1)
        assert len(events) == 1
        assert events[0]["url"] == "http://victim.internal/login"
        assert events[0]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_blocking_modal_triggers_reflexive_backtrack(self):
        """When an action observation encounters a blocking modal, motor backtrack (Escape) is triggered."""
        provider = _mock_provider()
        provider.terminal.return_value = MagicMock(
            stdout="Action blocked by modal overlay", stderr="", exit_code=0
        )
        agent = ComputerUseAgent(computer_provider=provider)
        agent.motor.backtrack = AsyncMock(return_value="backtracked: modal_blocked")

        # Simulate a command or action that results in a modal block message
        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.TERMINAL_EXEC,
            "echo 'Action blocked by modal overlay'",
            {"command": "echo 'Action blocked by modal overlay'"},
            "Execute test",
        )

        agent.motor.backtrack.assert_awaited_once_with("ws-1", reason="modal_blocked")
        assert "Reflexive backtrack: dismissed blocking modal" in trace.actual_observation

    @pytest.mark.asyncio
    async def test_legacy_browser_navigation_requires_live_desktop_evidence(self):
        """Legacy navigation aliases must use the visible desktop and fail closed without a frame."""
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        agent.motor.enforce_tab_budget = AsyncMock(return_value=1)

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.BROWSER_NAVIGATE,
            "https://" + "target",
            {"url": "https://" + "target"},
            "Navigate to target",
        )

        agent.motor.enforce_tab_budget.assert_not_awaited()
        provider.terminal.assert_not_awaited()
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "no evaluated local focus-verification head" in trace.actual_observation
