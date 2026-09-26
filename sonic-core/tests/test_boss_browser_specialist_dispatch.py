"""Tests verifying BossAgent's direct dispatch to BrowserAgent as a first-class specialist subagent,
propagation of Being, BeingMind, BeingCraft, BeingVault, execution planes, and error handling.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from sonic.agents.browser_agent import BrowserAgent, PageSnapshot
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult


class MockProvider:
    async def screenshot(self, *args, **kwargs):
        return b"mock_screenshot"

    async def terminal(self, *args, **kwargs):
        return "mock_terminal"

    async def status(self, *args, **kwargs):
        return {"status": "RUNNING"}


class MockLLM:
    async def complete(self, request, **kwargs):
        resp = MagicMock()
        resp.content = '{"action": "NAVIGATE", "target": "https://example.test", "thought": "Navigating to test site"}'
        return resp


@pytest.mark.asyncio
async def test_boss_dispatches_directly_to_browser_agent_as_specialist():
    """Verify that BossAgent routes browser tasks directly to BrowserAgent as a first-class specialist."""
    mock_browser = MagicMock(spec=BrowserAgent)
    mock_browser.run_mission = AsyncMock(return_value=[
        {
            "step_index": 1,
            "action_type": "BROWSER_NAVIGATE",
            "target_resource": "https://example.test",
            "thought": "Open website",
            "actual_observation": "Navigated to https://example.test (status: 200, title: Example)",
            "observation": "Navigated to https://example.test (status: 200, title: Example)",
            "status": "SUCCESS",
            "exit_code": 0,
            "duration_seconds": 0.5,
        },
        {
            "step_index": 2,
            "action_type": "GOAL_COMPLETE",
            "target_resource": "https://example.test",
            "thought": "Verified page content",
            "actual_observation": "GOAL_COMPLETE",
            "observation": "GOAL_COMPLETE",
            "status": "SUCCESS",
            "exit_code": 0,
            "duration_seconds": 0.01,
        },
    ])
    mock_browser.mission_summary = "Website loaded and verified successfully"
    mock_browser.vault = None

    mock_being = MagicMock()
    mock_being.being_id = "being-test-123"
    mock_mind = MagicMock()
    mock_craft = MagicMock()
    mock_vault = MagicMock()

    boss = BossAgent(
        computer_provider=MockProvider(),
        llm_router=MockLLM(),
        browser=mock_browser,
        being=mock_being,
        being_mind=mock_mind,
        craft=mock_craft,
        vault=mock_vault,
        tenant_id="tenant-browser-test",
    )

    sub_m = SubMission(
        id="sub-web-1",
        goal="Navigate to https://example.test and verify headline",
        role="browser",
        plane="gui",
        max_steps=3,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent") as mock_cua:
        res = await boss._dispatch_sub_agent("ws-1", sub_m, sub_agent_num=1)

        # ComputerUseAgent must NOT have been instantiated for browser specialist task
        mock_cua.assert_not_called()

    # BrowserAgent.run_mission must have been called directly
    mock_browser.run_mission.assert_called_once()
    call_kwargs = mock_browser.run_mission.call_args[1]
    assert call_kwargs["workspace_id"] == "ws-1"
    assert "https://example.test" in call_kwargs["goal"]
    assert call_kwargs["steps"] == 3

    # Being, BeingMind, BeingCraft, BeingVault must be propagated to BrowserAgent
    assert mock_browser.being == mock_being
    assert mock_browser.being_mind == mock_mind
    assert mock_browser.craft == mock_craft
    assert mock_browser.vault == mock_vault
    assert mock_browser.execution_plane == "gui"

    # Result verification
    assert isinstance(res, SubMissionResult)
    assert res.role == "browser"
    assert res.success is True
    assert res.status == "SUCCESS"
    assert res.actions_taken == 2
    assert "Website loaded and verified" in res.findings_summary


@pytest.mark.asyncio
async def test_boss_passes_being_mind_craft_vault_and_plane_to_computer_use_agent():
    """Verify ComputerUseAgent receives being, being_mind, craft, vault, and execution_plane."""
    mock_being = MagicMock()
    mock_being.being_id = "being-cua-test"
    mock_mind = MagicMock()
    mock_craft = MagicMock()
    mock_vault = MagicMock()

    boss = BossAgent(
        computer_provider=MockProvider(),
        llm_router=MockLLM(),
        being=mock_being,
        being_mind=mock_mind,
        craft=mock_craft,
        vault=mock_vault,
        tenant_id="tenant-cua-test",
    )

    sub_m = SubMission(
        id="sub-cli-1",
        goal="Run headless test command in workspace",
        role="subagent",
        plane="headless",
        max_steps=4,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent") as mock_cua_cls:
        mock_instance = MagicMock()
        mock_instance.run_mission = AsyncMock(return_value=[
            {
                "step_index": 1,
                "action_type": "TERMINAL_EXEC",
                "target_resource": "pytest",
                "actual_observation": "All tests passed",
                "status": "SUCCESS",
                "exit_code": 0,
            },
            {
                "step_index": 2,
                "action_type": "GOAL_COMPLETE",
                "target_resource": "pytest",
                "actual_observation": "GOAL_COMPLETE",
                "status": "SUCCESS",
                "exit_code": 0,
            },
        ])
        mock_instance.mission_summary = "Tests passed cleanly."
        mock_cua_cls.return_value = mock_instance

        res = await boss._dispatch_sub_agent("ws-1", sub_m, sub_agent_num=2)

        mock_cua_cls.assert_called_once()
        cua_kwargs = mock_cua_cls.call_args[1]
        assert cua_kwargs["being"] == mock_being
        assert cua_kwargs["being_mind"] == mock_mind
        assert cua_kwargs["craft"] == mock_craft
        assert cua_kwargs["vault"] == mock_vault
        assert cua_kwargs["execution_plane"] == "headless"
        assert cua_kwargs["observe_desktop"] is False  # Plane purity: headless skips desktop capture

        assert res.success is True
        assert res.actions_taken == 2
        assert "Tests passed cleanly" in res.findings_summary


@pytest.mark.asyncio
async def test_browser_agent_run_mission_standalone():
    """Verify BrowserAgent's run_mission executes actions, calls callbacks, and tracks traces."""
    agent = BrowserAgent(headless=True)

    # Mock low-level browser primitives
    agent._using_playwright = True
    agent._page = MagicMock()
    agent.navigate = AsyncMock(return_value=PageSnapshot(
        url="https://example.test/app",
        title="Example App",
        status_code=200,
        html_content="<html><body><h1>Welcome</h1></body></html>",
        screenshot_b64="",
    ))
    agent.current_page_state = AsyncMock(return_value=("https://example.test/app", "Example App"))
    agent.current_page_observation = AsyncMock(return_value={
        "url": "https://example.test/app",
        "title": "Example App",
        "text": "Welcome to the App",
        "interactive_elements": [],
    })
    agent.save_state_to_vault = AsyncMock()

    callback_traces = []
    async def step_cb(trace):
        callback_traces.append(trace)

    traces = await agent.run_mission(
        workspace_id="ws-browser",
        goal="Visit https://example.test/app and verify",
        steps=2,
        step_callback=step_cb,
    )

    assert len(traces) >= 1
    assert any("NAVIGATE" in t["action_type"] for t in traces)
    assert len(callback_traces) >= 1
    agent.save_state_to_vault.assert_called_once()


@pytest.mark.asyncio
async def test_boss_browser_dispatch_handles_exceptions_without_swallowing():
    """Verify that exceptions in BrowserAgent dispatch are reported accurately with status=FAILED."""
    mock_browser = MagicMock(spec=BrowserAgent)
    mock_browser.run_mission = AsyncMock(side_effect=RuntimeError("Browser process disconnected unexpectedly"))

    boss = BossAgent(
        computer_provider=MockProvider(),
        llm_router=MockLLM(),
        browser=mock_browser,
        tenant_id="err-test",
    )

    sub_m = SubMission(
        id="sub-err",
        goal="Navigate to https://crash.test",
        role="browser",
        plane="gui",
    )

    res = await boss._dispatch_sub_agent("ws-1", sub_m)

    assert res.success is False
    assert res.status == "FAILED"
    assert res.role == "browser"
    assert "Browser process disconnected unexpectedly" in res.findings_summary
