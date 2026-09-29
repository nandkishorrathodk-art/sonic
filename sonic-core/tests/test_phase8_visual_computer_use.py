"""
Phase 8 — Visual Computer Use done-gate tests
================================================
Verify that ComputerUseAgent can dispatch GUI actions (click, type, keypress,
move, scroll, screenshot) to the computer provider, that the LLM prompt
includes GUI actions in its schema, that the action parser handles coordinate
formats, and that the safety policy allows GUI actions.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from sonic.computer.models import GUIActionType, ScreenObservation
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerWorldObservation,
)
from sonic.computer_use.agent import ComputerUseAgent
from sonic.safety.action_policy import ActionPolicy


# ---- Helpers ----

def _mock_provider():
    """Build a mock ComputerProvider with async methods."""
    p = AsyncMock()
    p.screenshot.return_value = ScreenObservation(width=1280, height=800)
    p.status.return_value = MagicMock(
        active_application="Desktop", open_applications=["Desktop"],
        running_processes=["xvfb", "xfce4-panel"],
    )
    p.list_files.return_value = []
    p.git_action.return_value = MagicMock(branch="main", is_clean=True)
    p.terminal.return_value = MagicMock(stdout="__sonic_obs_ready__", stderr="", exit_code=0)
    p.gui_action.return_value = ScreenObservation(width=1280, height=800)
    return p


# ---- Tests ----

class TestGUIActionDispatch:
    """ComputerUseAgent dispatches GUI actions to the provider."""

    @pytest.mark.asyncio
    async def test_gui_click_dispatches_to_provider(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_CLICK, "640,400",
            {"x": 640, "y": 400}, "Click at center"
        )
        assert trace.status == "SUCCESS"
        assert "Clicked at (640, 400)" in trace.actual_observation
        provider.gui_action.assert_called_once()
        call_args = provider.gui_action.call_args
        gui_action = call_args[0][1]  # second positional arg
        assert gui_action.action == GUIActionType.CLICK

    @pytest.mark.asyncio
    async def test_gui_type_dispatches_to_provider(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_TYPE, "keyboard",
            {"text": "hello world"}, "Type text"
        )
        assert trace.status == "SUCCESS"
        assert "Typed 11 chars" in trace.actual_observation
        call_args = provider.gui_action.call_args
        gui_action = call_args[0][1]
        assert gui_action.action == GUIActionType.TYPE

    @pytest.mark.asyncio
    async def test_gui_type_focuses_grounded_input_before_typing(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
        agent._last_screenshot_b64 = "live-screen"
        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            "search field",
            {"x": 320, "y": 240, "text": "query"},
            "Focus and type in the grounded field",
        )
        assert trace.status == "SUCCESS"
        actions = [
            call.args[1].action for call in provider.gui_action.call_args_list
        ]
        assert actions[:2] == [GUIActionType.CLICK, GUIActionType.TYPE]
        click = provider.gui_action.call_args_list[0].args[1]
        assert (click.x, click.y) == (320, 240)

    @pytest.mark.asyncio
    async def test_gui_keypress_dispatches_to_provider(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_KEYPRESS, "keyboard",
            {"key": "Return"}, "Press Enter"
        )
        assert trace.status == "SUCCESS"
        assert "Pressed key: Return" in trace.actual_observation

    @pytest.mark.asyncio
    async def test_gui_only_blocks_terminal_before_provider_execution(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)

        trace = await agent.execute_action(
            "ws-1", ComputerActionType.TERMINAL_EXEC, "uname -a",
            {"command": "uname -a"}, "Inspect the desktop host"
        )

        assert trace.status == "BLOCKED"
        assert trace.exit_code == 126
        provider.terminal.assert_not_awaited()
        provider.gui_action.assert_not_called()

    @pytest.mark.asyncio
    async def test_gui_execution_plane_blocks_terminal_but_keeps_gui_actions_available(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(
            computer_provider=provider,
            execution_plane="gui",
        )

        terminal_trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.TERMINAL_EXEC,
            "gedit",
            {"command": "gedit"},
            "Launch a GUI application through the terminal",
        )

        assert agent.gui_only is False
        assert terminal_trace.status == ActionExecutionStatus.BLOCKED
        assert terminal_trace.exit_code == 126
        assert "GUI execution contract blocked TERMINAL_EXEC" in terminal_trace.actual_observation
        provider.terminal.assert_not_awaited()

        gui_trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_KEYPRESS,
            "Applications menu",
            {"key": "Super"},
            "Open the visible applications menu",
        )

        assert gui_trace.status == ActionExecutionStatus.SUCCESS
        provider.gui_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_gui_only_rejected_close_tabs_proposal_reobserves_visible_desktop(self):
        action_type, target, payload, expected = ComputerUseAgent._gui_only_recovery_action(
            "hi sonic, close browser and close all tabs",
            "Non-GUI proposal discarded",
        )

        assert action_type == ComputerActionType.GUI_SCREENSHOT
        assert target == "visible-desktop"
        assert payload == {}
        assert "visible desktop" in expected

    @pytest.mark.asyncio
    async def test_gui_only_close_tabs_intent_fails_closed_when_reasoning_fails(self):
        provider = _mock_provider()
        router = MagicMock()
        agent = ComputerUseAgent(
            computer_provider=provider,
            llm_router=router,
            gui_only=True,
        )

        action_type, target, payload, expected = await agent.choose_action(
            "hi sonic, close browser and close all tabs",
            ComputerWorldObservation(),
        )

        assert action_type == ActionExecutionStatus.BLOCKED
        assert target == "reasoning-unavailable"
        assert payload == {}
        assert "reasoning failed" in expected.lower()

    @pytest.mark.asyncio
    async def test_gui_plane_without_router_does_not_fall_back_to_file_probe(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(
            computer_provider=provider,
            execution_plane="gui",
        )
        agent.llm_router = None

        action_type, target, payload, expected = await agent.choose_action(
            "Open the document in the active application",
            ComputerWorldObservation(),
        )

        assert action_type == ActionExecutionStatus.BLOCKED
        assert target == "reasoning-unavailable"
        assert payload == {}
        assert "no reasoning model" in expected.lower()

    @pytest.mark.asyncio
    async def test_gui_reasoning_failure_records_one_block_and_stops_mission(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(
            computer_provider=provider,
            execution_plane="gui",
        )
        agent.llm_router = None
        agent._bootstrap_performed = True
        agent.observe = AsyncMock(return_value=ComputerWorldObservation())

        traces = await agent.run_mission("ws-1", "Open the requested application", steps=8)

        assert len(traces) == 1
        assert traces[0].status == ActionExecutionStatus.BLOCKED
        assert "No reasoning model" in traces[0].actual_observation
        provider.gui_action.assert_not_called()

    @pytest.mark.asyncio
    async def test_gui_verification_does_not_accept_keyword_overlap_as_proof(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(
            computer_provider=provider,
            execution_plane="gui",
        )
        agent.observe = AsyncMock(return_value=MagicMock(
            active_application="Calculator",
            visible_text="Calculator report saved",
            windows=["Calculator"],
            screen=MagicMock(visible_text="Calculator report saved"),
        ))

        verified, evidence = await agent.verify_goal(
            "ws-1",
            "Save Calculator report",
        )

        assert verified is False
        assert "No typed independent verification probe" in evidence

    @pytest.mark.asyncio
    async def test_gui_only_observation_does_not_probe_files_git_or_terminal(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)

        await agent.observe("ws-1")

        provider.terminal.assert_not_awaited()
        provider.list_files.assert_not_awaited()
        provider.git_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_only_browser_navigation_requires_live_screen_verification(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.BROWSER_NAVIGATE,
            "https://" + "target",
            {"url": "https://" + "target"},
            "Open the visible browser",
        )

        assert trace.status == "RECOVERED"
        assert "could not be verified from a live screen" in trace.actual_observation.lower()
        assert provider.terminal.await_count == 0
        assert provider.gui_action.await_count == 4
        actions = [call.args[1].action for call in provider.gui_action.call_args_list]
        assert [action.value for action in actions[:3]] == ["KEYPRESS", "TYPE", "KEYPRESS"]

    @pytest.mark.asyncio
    async def test_gui_only_modal_recovery_does_not_use_motor_backtrack(self):
        provider = _mock_provider()
        motor = MagicMock()
        motor.backtrack = AsyncMock()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True, motor=motor)

        async def blocked_gui_action(*args, **kwargs):
            raise RuntimeError("modal_blocked")

        provider.gui_action.side_effect = blocked_gui_action
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_CLICK, "100,100",
            {"x": 100, "y": 100}, "Click visible control",
        )

        assert trace.status in ("FAILED", "RECOVERED", "BLOCKED")
        motor.backtrack.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_double_click_dispatches(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_DOUBLE_CLICK, "100,200",
            {"x": 100, "y": 200}, "Double click"
        )
        assert trace.status == "SUCCESS"
        assert "Double-clicked" in trace.actual_observation
        gui_action = provider.gui_action.call_args[0][1]
        assert gui_action.action == GUIActionType.DOUBLE_CLICK

    @pytest.mark.asyncio
    async def test_gui_scroll_dispatches(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_SCROLL, "640,400",
            {"x": 640, "y": 400, "delta": -3}, "Scroll down"
        )
        assert trace.status == "SUCCESS"
        assert "Scrolled" in trace.actual_observation

    @pytest.mark.asyncio
    async def test_gui_screenshot_dispatches(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_SCREENSHOT, "screen",
            {}, "Take screenshot"
        )
        assert trace.status == "SUCCESS"
        assert "Screenshot captured" in trace.actual_observation


class TestActionParsing:
    """_parse_llm_action handles GUI coordinate format."""

    def test_parse_gui_click_with_coordinates(self):
        text = "ACTION: GUI_CLICK\nTARGET: 640,400\nPAYLOAD: {}\nEXPECTED: Click button"
        action_type, target, payload, expected = ComputerUseAgent._parse_llm_action(text, "test.py")
        assert action_type == ComputerActionType.GUI_CLICK
        assert payload.get("x") == 640
        assert payload.get("y") == 400

    def test_parse_gui_type(self):
        text = 'ACTION: GUI_TYPE\nTARGET: keyboard\nPAYLOAD: {"text": "hello"}\nEXPECTED: Type text'
        action_type, target, payload, expected = ComputerUseAgent._parse_llm_action(text, "test.py")
        assert action_type == ComputerActionType.GUI_TYPE
        assert payload.get("text") == "hello"

    def test_parse_gui_keypress(self):
        text = 'ACTION: GUI_KEYPRESS\nTARGET: keyboard\nPAYLOAD: {"key": "Escape"}\nEXPECTED: Press Escape'
        action_type, target, payload, expected = ComputerUseAgent._parse_llm_action(text, "test.py")
        assert action_type == ComputerActionType.GUI_KEYPRESS


class TestReasoningPrompt:
    """_build_reasoning_context includes GUI actions."""

    def test_system_prompt_includes_gui_actions(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        obs = MagicMock()
        obs.visible_text = "Desktop"
        obs.terminal_output = "ready"
        obs.active_application = "Desktop"
        obs.filesystem_files = []
        obs.git_branch = "main"
        obs.git_clean = True
        obs.browser_state = {}
        system_prompt, user_prompt = agent._build_reasoning_context(
            "test goal", obs, 1, "test.py", "test_test.py"
        )
        assert "GUI_CLICK" in system_prompt
        assert "GUI_TYPE" in system_prompt
        assert "GUI_KEYPRESS" in system_prompt
        assert "GUI_SCROLL" in system_prompt
        assert "GUI_SCREENSHOT" in system_prompt
        assert "EXECUTION GROUND-TRUTH" in system_prompt
        assert "visible cursor" in system_prompt
        assert "APP_FOCUS" in system_prompt
        assert "actual screen pixels" in user_prompt
        assert "Only the separate multimodal grounding response uses normalized 0-1000" in user_prompt


class TestSafetyPolicy:
    """Safety policy allows GUI actions (in-sandbox)."""

    def test_gui_click_allowed(self):
        policy = ActionPolicy()
        v = policy.evaluate("GUI_CLICK", "640,400", {"x": 640, "y": 400})
        assert v.allowed

    def test_gui_type_allowed(self):
        policy = ActionPolicy()
        v = policy.evaluate("GUI_TYPE", "keyboard", {"text": "hello"})
        assert v.allowed

    def test_gui_scroll_allowed(self):
        policy = ActionPolicy()
        v = policy.evaluate("GUI_SCROLL", "640,400", {"delta": -3})
        assert v.allowed

    def test_gui_screenshot_allowed(self):
        policy = ActionPolicy()
        v = policy.evaluate("GUI_SCREENSHOT", "screen", {})
        assert v.allowed

    def test_destructive_terminal_still_blocked(self):
        policy = ActionPolicy()
        v = policy.evaluate("TERMINAL_EXEC", "terminal", {"command": "rm -rf /"})
        assert not v.allowed

    @pytest.mark.asyncio
    async def test_gui_click_blocked_by_safety_policy_denied(self):
        """When safety policy denies GUI_CLICK, the action is BLOCKED."""
        provider = _mock_provider()
        # Create a policy that does NOT allow GUI_CLICK
        restrictive_policy = ActionPolicy(
            allowed_action_types={"TERMINAL_EXEC", "FILE_READ"},
        )
        agent = ComputerUseAgent(
            computer_provider=provider,
            safety=restrictive_policy,
        )
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_CLICK, "640,400",
            {"x": 640, "y": 400}, "Click at center"
        )
        assert trace.status == "BLOCKED"
        provider.gui_action.assert_not_called()
