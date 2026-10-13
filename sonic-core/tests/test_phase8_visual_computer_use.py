"""
Phase 8 — Visual Computer Use done-gate tests
================================================
Verify that ComputerUseAgent can dispatch GUI actions (click, type, keypress,
move, scroll, screenshot) to the computer provider, that the LLM prompt
includes GUI actions in its schema, that the action parser handles coordinate
formats, and that the safety policy allows GUI actions.
"""
import base64
import binascii
import json
import struct
import time
import pytest
import zlib
from unittest.mock import AsyncMock, MagicMock

from sonic.computer.models import GUIActionType, ScreenObservation
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerWorldObservation,
    SubGoal,
    SubGoalChecklist,
    SubGoalStatus,
)
from sonic.computer_use.agent import ComputerUseAgent
from sonic.safety.action_policy import ActionPolicy


# ---- Helpers ----

def _png_base64(color: int = 0) -> str:
    """Create a valid frame matching the mock display dimensions."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", binascii.crc32(body) & 0xFFFFFFFF
        )

    width, height = 1280, 800
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = (
        b"\x00" + bytes([color if y >= height // 2 else 0]) * (width * 3)
        for y in range(height)
    )
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(image).decode("ascii")


_FRESH_FRAME = _png_base64()
_AFTER_FRAME = _png_base64(255)
_FOCUSED_FRAME = _png_base64(128)


def _mock_provider():
    """Build a mock ComputerProvider with async methods."""
    p = AsyncMock()
    p.observe_state = None
    p.screenshot.return_value = ScreenObservation(
        width=1280,
        height=800,
        screenshot_base64=_FRESH_FRAME,
        capture_age_ms=0,
        controls_age_ms=0,
        active_window="Desktop",
    )
    p.status.return_value = MagicMock(
        active_application="Desktop", open_applications=["Desktop"],
        running_processes=["xvfb", "xfce4-panel"],
    )
    p.list_files.return_value = []
    p.git_action.return_value = MagicMock(branch="main", is_clean=True)
    p.terminal.return_value = MagicMock(stdout="__sonic_obs_ready__", stderr="", exit_code=0)
    p.gui_action.return_value = ScreenObservation(
        width=1280,
        height=800,
        screenshot_base64=_AFTER_FRAME,
        capture_age_ms=0,
        controls_age_ms=0,
        active_window="Desktop",
    )
    return p


def _seed_fresh_gui_observation(agent, *, active_window="Desktop", controls=None):
    screen = ScreenObservation(
        width=1280,
        height=800,
        screenshot_base64=_FRESH_FRAME,
        capture_age_ms=0,
        controls_age_ms=0,
        active_window=active_window,
        detected_controls=controls or [],
    )
    screen = agent._ensure_avcs_state(screen)
    observation = ComputerWorldObservation(
        screen=screen,
        active_application=active_window,
        perception_version=screen.perception_version,
        observation_age_ms=0,
    )
    agent._last_world_observation = observation
    agent._last_gui_state_observation = screen
    agent._last_screen_observation_received_at = time.monotonic()
    agent._last_screenshot_b64 = screen.screenshot_base64
    agent._screen_width = screen.width
    agent._screen_height = screen.height
    return observation


# ---- Tests ----

class TestGUIActionDispatch:
    """ComputerUseAgent dispatches GUI actions to the provider."""

    @pytest.mark.asyncio
    async def test_app_launch_verifies_active_window_by_wm_class(self):
        provider = _mock_provider()
        provider.status.return_value = MagicMock(
            active_application="workspace",
            active_window="workspace",
            open_applications=["Desktop", "workspace"],
            installed_applications=["thunar"],
            running_processes=["thunar"],
        )
        provider.screenshot.return_value = ScreenObservation(
            width=1280,
            height=800,
            active_window="workspace",
            screenshot_base64=_FRESH_FRAME,
            desktop_state="INTERACTIVE",
        )
        provider.launch_application.return_value = True
        provider.get_active_window_class = AsyncMock(return_value="thunar.Thunar")
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.APP_LAUNCH,
            "thunar",
            {"app_name": "thunar"},
            "Focus the installed file manager",
        )

        assert trace.status == ActionExecutionStatus.COMPLETED
        assert "active window: workspace" in trace.actual_observation
        provider.get_active_window_class.assert_awaited_once_with("ws-1")

    @pytest.mark.asyncio
    async def test_gui_click_dispatches_to_provider(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        await agent.observe("ws-1")
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

    @pytest.mark.parametrize(
        ("x", "y"),
        ((320.5, 240), (True, 240), (320, None), ("320", 240)),
    )
    @pytest.mark.asyncio
    async def test_gui_click_rejects_non_integer_payload_coordinates(self, x, y):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        await agent.observe("ws-1")

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_CLICK,
            "search field",
            {"x": x, "y": y},
            "Click the supplied pixel point",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "valid integer coordinate" in trace.actual_observation.lower() or (
            "coordinates (x, y) are required" in trace.actual_observation.lower()
        )
        provider.gui_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_scroll_without_grounded_point_is_blocked(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        await agent.observe("ws-1")

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_SCROLL,
            "current view",
            {"delta": -2},
            "Scroll the current view",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        provider.gui_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_type_without_pixel_target_is_blocked(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        await agent.observe("ws-1")
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_TYPE, "keyboard",
            {"text": "hello world"}, "Type text"
        )
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "focus-verification head" in trace.actual_observation
        provider.gui_action.assert_not_awaited()

    @pytest.mark.parametrize(
        "action_type",
        [ComputerActionType.GUI_TYPE, ComputerActionType.BROWSER_TYPE],
    )
    @pytest.mark.asyncio
    async def test_dual_plane_typing_without_avcs_focus_head_is_blocked(self, action_type):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)

        trace = await agent.execute_action(
            "ws-1",
            action_type,
            "search field",
            {"text": "must not be sent"},
            "Type into the search field",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "no evaluated local focus-verification head" in trace.actual_observation
        provider.gui_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_only_type_fails_closed_without_evaluated_focus_head(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock()
        await agent.observe("ws-1")
        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            "320,240",
            {
                "x": 320,
                "y": 240,
                "text": "query",
                "postcondition": {"screen_changed": True},
            },
            "The requested text appears",
        )
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "focus-verification head" in trace.actual_observation
        agent.llm_router.complete.assert_not_awaited()
        provider.gui_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_only_clear_field_fails_closed_without_using_shell_motor(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
        agent.motor.clear_field = AsyncMock()
        await agent.observe("ws-1")

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            "320,240",
            {
                "x": 320,
                "y": 240,
                "text": "replacement",
                "clear_field": True,
                "postcondition": {"screen_changed": True},
            },
            "Replace the visible field text",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "focus-verification head" in trace.actual_observation
        agent.motor.clear_field.assert_not_awaited()
        provider.gui_action.assert_not_awaited()
        provider.terminal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_only_type_rejects_shell_backed_window_focus_request(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
        agent.motor.settle_focus = AsyncMock()
        await agent.observe("ws-1")

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            "320,240",
            {
                "x": 320,
                "y": 240,
                "text": "must not be sent",
                "window_title": "Requested window",
                "postcondition": {"screen_changed": True},
            },
            "Type into the requested window",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        agent.motor.settle_focus.assert_not_awaited()
        assert all(
            call.args[1].action != GUIActionType.TYPE
            for call in provider.gui_action.await_args_list
        )

    @pytest.mark.asyncio
    async def test_gui_only_type_blocks_when_pixels_do_not_confirm_focus(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "focused": False,
                        "confidence": 0.98,
                        "evidence": "No caret or focus indicator is visible.",
                    }
                )
            )
        )
        await agent.observe("ws-1")

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            "320,240",
            {
                "x": 320,
                "y": 240,
                "text": "must not be sent",
                "postcondition": {"screen_changed": True},
            },
            "The requested text appears",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "no evaluated local focus-verification head" in trace.actual_observation
        assert all(
            call.args[1].action != GUIActionType.TYPE
            for call in provider.gui_action.await_args_list
        )

    @pytest.mark.asyncio
    async def test_gui_only_type_does_not_send_text_without_post_focus_pixels(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
        await agent.observe("ws-1")
        provider.screenshot.side_effect = [
            ScreenObservation(
                width=1280,
                height=800,
                screenshot_base64=_FRESH_FRAME,
                capture_age_ms=0,
                active_window="Desktop",
            ),
            ScreenObservation(
                width=1280,
                height=800,
                screenshot_base64=_FOCUSED_FRAME,
                capture_age_ms=0,
                active_window="Desktop",
            ),
            ScreenObservation(
                width=1280,
                height=800,
                capture_age_ms=0,
                active_window="Desktop",
            ),
        ]

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_TYPE,
            "320,240",
            {
                "x": 320,
                "y": 240,
                "text": "query",
                "postcondition": {"screen_changed": True},
            },
            "The requested text appears",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "no evaluated local focus-verification head" in trace.actual_observation
        assert all(
            call.args[1].action != GUIActionType.TYPE
            for call in provider.gui_action.await_args_list
        )

    @pytest.mark.asyncio
    async def test_gui_keypress_dispatches_to_provider(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        await agent.observe("ws-1")
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
    async def test_terminal_execution_preserves_quotes_and_trailing_semicolon(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        command = "printf '%s\\n' 'two words';"

        await agent.execute_action(
            "ws-1",
            ComputerActionType.TERMINAL_EXEC,
            command,
            {"command": command},
            "Print the argument",
        )

        assert provider.terminal.await_args.args[1] == command

    @pytest.mark.asyncio
    async def test_gui_execution_plane_blocks_terminal_but_keeps_gui_actions_available(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(
            computer_provider=provider,
            execution_plane="gui",
        )
        await agent.observe("ws-1")

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

    def test_repeated_gui_contract_violations_stop_instead_of_screenshot_loop(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider())

        first = agent._recover_rejected_gui_proposal("open app", "Non-GUI proposal")
        second = agent._recover_rejected_gui_proposal("open app", "Non-GUI proposal")

        assert first[0] == ComputerActionType.GUI_SCREENSHOT
        assert second[0] == ActionExecutionStatus.BLOCKED
        assert second[1] == "gui-contract-violation"

    def test_gui_type_requires_explicit_text(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")

        action, target, payload, reason = agent._parse_gui_action_response(
            "ACTION: GUI_TYPE\nTARGET: terminal_window\nPAYLOAD: {}\nEXPECTED: Type",
            "",
        )

        assert action == ActionExecutionStatus.BLOCKED
        assert target == "gui-invalid-response"
        assert payload == {}
        assert "requires explicit text" in reason

    def test_gui_action_parser_maps_pixel_coordinates_in_target_to_payload(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")

        action, target, payload, _ = agent._parse_gui_action_response(
            (
                "ACTION: GUI_TYPE\nTARGET: 115, 114\n"
                "PAYLOAD: {\"text\":\"hello\"}\nEXPECTED: Text appears"
            ),
            "",
        )

        assert action == ComputerActionType.GUI_TYPE
        assert target == "115, 114"
        assert payload == {"text": "hello", "x": 115, "y": 114}

    def test_gui_action_parser_maps_json_pixel_point_to_payload(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")

        action, target, payload, _ = agent._parse_gui_action_response(
            (
                'ACTION: GUI_TYPE\nTARGET: {"x": 330, "y": 300}\n'
                'PAYLOAD: {"text": "folder name"}\nEXPECTED: Text appears'
            ),
            "",
        )

        assert action == ComputerActionType.GUI_TYPE
        assert target == '{"x": 330, "y": 300}'
        assert payload == {"text": "folder name", "x": 330, "y": 300}

    def test_gui_action_parser_recovers_wrapped_single_quoted_payload_json(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")

        action, target, payload, _ = agent._parse_gui_action_response(
            (
                "ACTION: GUI_TYPE\nTARGET: 330, 300\n"
                "PAYLOAD:\n```json\n{'text': 'sonic-gui-acceptance-20261008',}\n```\n"
                "EXPECTED: The folder name appears in the dialog"
            ),
            "",
        )

        assert action == ComputerActionType.GUI_TYPE
        assert target == "330, 300"
        assert payload == {
            "text": "sonic-gui-acceptance-20261008",
            "x": 330,
            "y": 300,
        }

    def test_gui_action_parser_still_rejects_bare_payload_text(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")

        action, target, payload, reason = agent._parse_gui_action_response(
            (
                "ACTION: GUI_TYPE\nTARGET: folder name field\n"
                "PAYLOAD: sonic-gui-acceptance-20261008\nEXPECTED: Type folder name"
            ),
            "",
        )

        assert action == ActionExecutionStatus.BLOCKED
        assert target == "gui-invalid-response"
        assert payload == {}
        assert "PAYLOAD must be a JSON object" in reason

    @pytest.mark.asyncio
    async def test_gui_type_blocks_until_local_focus_verification_is_available(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        await agent.observe("ws-1")
        action, target, payload, expected = agent._parse_gui_action_response(
            (
                'ACTION: GUI_TYPE\nTARGET: {"x": 330, "y": 300}\n'
                'PAYLOAD: {"text": "folder name", '
                '"postcondition": {"screen_changed": true}}\n'
                'EXPECTED: Text appears'
            ),
            "",
        )

        trace = await agent.execute_action(
            "ws-1", action, target, payload, expected,
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "focus-verification head" in trace.actual_observation
        provider.gui_action.assert_not_awaited()

    @pytest.mark.parametrize(
        "target",
        (
            '{"x": 330.5, "y": 300}',
            '{"x": true, "y": 300}',
            '{"x": 330, "y": 300, "coordinate_space": "normalized"}',
        ),
    )
    def test_gui_action_parser_does_not_guess_non_pixel_coordinate_objects(self, target):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")

        action, _, payload, _ = agent._parse_gui_action_response(
            f'ACTION: GUI_CLICK\nTARGET: {target}\nPAYLOAD: {{}}\nEXPECTED: Click',
            "",
        )

        assert action == ActionExecutionStatus.BLOCKED
        assert "x" not in payload
        assert "y" not in payload

    def test_gui_action_parser_blocks_conflicting_target_and_payload_coordinates(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")

        action, target, payload, reason = agent._parse_gui_action_response(
            (
                'ACTION: GUI_CLICK\nTARGET: {"x": 330, "y": 300}\n'
                'PAYLOAD: {"x": 10, "y": 20}\nEXPECTED: Click'
            ),
            "",
        )

        assert action == ActionExecutionStatus.BLOCKED
        assert target == "gui-invalid-response"
        assert payload == {}
        assert "conflicting screen coordinates" in reason

    def test_gui_keypress_requires_explicit_key(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")
        action, target, payload, reason = agent._parse_gui_action_response(
            "ACTION: GUI_KEYPRESS\nTARGET: keyboard\nPAYLOAD: {}\nEXPECTED: Submit",
            "",
        )
        assert action == ActionExecutionStatus.BLOCKED
        assert target == "gui-invalid-response"
        assert payload == {}
        assert "explicit key" in reason

    @pytest.mark.asyncio
    async def test_malformed_keypress_never_dispatches_default_return(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_KEYPRESS,
            "keyboard",
            {},
            "Submit",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "no default key was sent" in trace.actual_observation
        provider.gui_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_pixel_payload_bypasses_grounding_and_uses_latest_bounds(self):
        provider = _mock_provider()
        provider.screenshot.return_value = ScreenObservation(
            width=1280,
            height=800,
            screenshot_base64=_FRESH_FRAME,
            capture_age_ms=0,
            controls_age_ms=0,
        )
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        await agent.observe("ws-1")
        accepted = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_CLICK,
            "visible control",
            {"x": 400, "y": 300},
            "Click the observed point",
        )
        assert accepted.status == ActionExecutionStatus.SUCCESS
        provider.gui_action.assert_awaited_once()

        provider.gui_action.reset_mock()
        rejected = await agent.execute_action(
            "ws-1",
            ComputerActionType.GUI_CLICK,
            "visible control",
            {"x": 1280, "y": 300},
            "Reject a point beyond the observed screen",
        )
        assert rejected.status == ActionExecutionStatus.BLOCKED
        assert "outside screen 1280x800" in rejected.actual_observation
        provider.gui_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gui_terminal_prefixed_goal_is_blocked_before_fast_path(self):
        llm = AsyncMock()
        agent = ComputerUseAgent(
            computer_provider=_mock_provider(),
            llm_router=llm,
            execution_plane="gui",
        )

        action, target, _, reason = await agent.choose_action(
            "terminal: echo must-not-run",
            ComputerWorldObservation(),
            1,
        )
        assert action == ActionExecutionStatus.BLOCKED
        assert target == "gui-contract-violation"
        assert "visible GUI plane" in reason
        llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_gui_agent_keeps_terminal_prefixed_fast_path(self):
        llm = AsyncMock()
        agent = ComputerUseAgent(computer_provider=_mock_provider(), llm_router=llm)

        action, target, payload, _ = await agent.choose_action(
            "terminal: echo allowed",
            ComputerWorldObservation(),
            1,
        )
        assert action == ComputerActionType.TERMINAL_EXEC
        assert target == "echo allowed"
        assert payload == {"command": "echo allowed"}
        llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_gui_proposal_gets_one_observation_then_stops_without_progress(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)
        goal = "1. capture screenshot"
        active = SubGoal(
            description="capture screenshot",
            status=SubGoalStatus.IN_PROGRESS,
        )
        agent.checklist = SubGoalChecklist(top_level_goal=goal, sub_goals=[active])
        agent._last_mission_goal = goal
        agent._consecutive_failures = 2
        agent._consecutive_unchanged_screens = 3
        agent.observe = AsyncMock(return_value=ComputerWorldObservation())
        agent.choose_action = AsyncMock(side_effect=[
            (
                ComputerActionType.GUI_SCREENSHOT,
                "visible-desktop",
                {},
                "Invalid proposal; inspect fresh screen",
            ),
            (
                ActionExecutionStatus.BLOCKED,
                "gui-contract-violation",
                {},
                "Stopped after repeated invalid model proposals outside the visible GUI action contract.",
            ),
        ])

        traces = await agent.run_mission("ws-1", goal, steps=5)

        assert agent.observe.await_count == 2
        assert agent.choose_action.await_count == 2
        assert len(traces) == 2
        assert traces[0].action_type == ComputerActionType.GUI_SCREENSHOT
        assert active.status == SubGoalStatus.IN_PROGRESS
        assert active.attempt_count == 0
        assert agent._consecutive_failures == 2
        assert agent._consecutive_unchanged_screens == 3
        assert agent.metrics.actions_successful == 0
        assert "outside the visible GUI action contract" in traces[1].actual_observation

    def test_mission_blocks_redundant_screenshot_action_after_fresh_observation(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider())
        agent._recent_action_signatures = [
            (ComputerActionType.GUI_SCREENSHOT.value, "screen", "{}")
        ]

        rejection = agent._is_trivial_or_repeated_action(
            ComputerActionType.GUI_SCREENSHOT,
            "desktop",
            {},
        )

        assert rejection is not None
        assert "fresh desktop observation" in rejection

    @pytest.mark.asyncio
    async def test_gui_only_close_tabs_intent_fails_closed_when_reasoning_fails(self):
        provider = _mock_provider()
        router = MagicMock()
        agent = ComputerUseAgent(
            computer_provider=provider,
            llm_router=router,
            gui_only=True,
        )
        router.complete = AsyncMock(side_effect=RuntimeError("reasoning failed"))
        observation = _seed_fresh_gui_observation(agent)

        action_type, target, payload, expected = await agent.choose_action(
            "hi sonic, close browser and close all tabs",
            observation,
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
        provider.screenshot.return_value = ScreenObservation(width=1280, height=800)
        agent = ComputerUseAgent(computer_provider=provider, gui_only=True)

        trace = await agent.execute_action(
            "ws-1",
            ComputerActionType.BROWSER_NAVIGATE,
            "https://" + "target",
            {"url": "https://" + "target"},
            "Open the visible browser",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "no evaluated local focus-verification head" in trace.actual_observation.lower()
        assert provider.terminal.await_count == 0
        provider.gui_action.assert_not_awaited()

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
        await agent.observe("ws-1")
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
        await agent.observe("ws-1")
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_SCROLL, "640,400",
            {"x": 640, "y": 400, "delta": -3}, "Scroll down"
        )
        assert trace.status == "SUCCESS"
        assert "Scrolled" in trace.actual_observation

    @pytest.mark.asyncio
    async def test_gui_scroll_requires_explicit_coordinates_and_delta(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider)
        trace = await agent.execute_action(
            "ws-1", ComputerActionType.GUI_SCROLL, "screen", {}, "Scroll down"
        )
        assert trace.status == "BLOCKED"
        provider.gui_action.assert_not_awaited()

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

    def test_gui_worker_rejects_prose_and_pseudo_action_output(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        action_type, target, payload, expected = agent._parse_gui_action_response(
            'I will use the tool now: {"name": "APP_FOCUS", "arguments": {"target": "app_focus"}}',
            "test.py",
        )

        assert action_type == ActionExecutionStatus.BLOCKED
        assert target == "gui-invalid-response"
        assert payload == {}
        assert "structured ACTION block" in expected

    def test_gui_worker_accepts_only_a_valid_structured_action(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        action_type, target, payload, expected = agent._parse_gui_action_response(
            'THOUGHT: The visible control is identified.\n'
            'ACTION: GUI_CLICK\n'
            'TARGET: visible button\n'
            'PAYLOAD: {"x": 400, "y": 300}\n'
            'EXPECTED: Button activates',
            "test.py",
        )

        assert action_type == ComputerActionType.GUI_CLICK
        assert target == "visible button"
        assert payload == {"x": 400, "y": 300}
        assert expected == "Button activates"

    def test_gui_worker_accepts_json_action_envelope(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        action_type, target, payload, expected = agent._parse_gui_action_response(
            '{"action":"GUI_HOTKEY","target":"keyboard",'
            '"payload":{"key":"Ctrl+L"},"expected":"Focus the location field"}',
            "",
        )

        assert action_type == ComputerActionType.GUI_HOTKEY
        assert target == "keyboard"
        assert payload == {"key": "Ctrl+L"}
        assert expected == "Focus the location field"

    def test_gui_worker_accepts_one_exact_json_envelope_inside_prose(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        response = (
            "The file manager is open. "
            '{"action":"GUI_HOTKEY","target":"keyboard",'
            '"payload":{"key":"Ctrl+L"},"expected":"Focus the location field"}'
            " I will inspect the resulting view next."
        )

        action_type, target, payload, expected = agent._parse_gui_action_response(
            response,
            "",
        )

        assert action_type == ComputerActionType.GUI_HOTKEY
        assert target == "keyboard"
        assert payload == {"key": "Ctrl+L"}
        assert expected == "Focus the location field"

    def test_gui_worker_rejects_ambiguous_or_extended_json_envelopes(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        envelope = (
            '{"action":"GUI_HOTKEY","target":"keyboard",'
            '"payload":{"key":"Ctrl+L"},"expected":"Focus the location field"}'
        )
        for response in (
            f"{envelope} {envelope}",
            envelope[:-1] + ',"commentary":"extra"}',
        ):
            action_type, target, _, _ = agent._parse_gui_action_response(
                response,
                "",
            )
            assert action_type == ActionExecutionStatus.BLOCKED
            assert target == "gui-invalid-response"

    def test_gui_worker_accepts_markdown_wrapped_action_fields(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        action_type, target, payload, expected = agent._parse_gui_action_response(
            "- **ACTION:** `GUI_HOTKEY` — focus the location field\n"
            "- **TARGET:** keyboard\n"
            '- **PAYLOAD:** {"key":"Ctrl+L"}\n'
            "- **EXPECTED:** Focus the location field",
            "",
        )

        assert action_type == ComputerActionType.GUI_HOTKEY
        assert target == "keyboard"
        assert payload == {"key": "Ctrl+L"}
        assert expected == "Focus the location field"

    def test_gui_worker_allows_visual_target_only_with_attached_grounding(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        response = (
            '{"action":"GUI_CLICK","target":"the unique file manager icon",'
            '"payload":{},"expected":"Open the file manager"}'
        )

        parsed_without_grounding, *_ = agent._parse_gui_action_response(response, "")
        assert parsed_without_grounding == ComputerActionType.GUI_CLICK

        agent._last_screenshot_b64 = _FRESH_FRAME
        agent._last_reasoning_perception = {"image_attached": True}
        agent.llm_router = MagicMock()
        action, target, *_ = agent._parse_gui_action_response(response, "")

        assert action == ComputerActionType.GUI_CLICK
        assert target == "the unique file manager icon"

    def test_gui_screenshot_can_omit_empty_target_and_payload(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)

        action_type, target, payload, _ = agent._parse_gui_action_response(
            "ACTION: GUI_SCREENSHOT",
            "",
        )

        assert action_type == ComputerActionType.GUI_SCREENSHOT
        assert target == "visible-desktop"
        assert payload == {}

    @pytest.mark.asyncio
    async def test_gui_reasoning_requests_json_action_output(self):
        agent = ComputerUseAgent(
            computer_provider=_mock_provider(),
            execution_plane="gui",
        )
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content=(
                '{"action":"GUI_SCREENSHOT","target":"visible-desktop",'
                '"payload":{},"expected":"Inspect the current desktop"}'
            ),
            reasoning_content="",
            tool_calls=[],
        ))
        observation = _seed_fresh_gui_observation(agent)

        action, *_ = await agent._llm_choose_action(
            "Inspect the visible desktop",
            observation,
            1,
            "",
            "",
        )
        request = agent.llm_router.complete.await_args.args[0]
        schema = request.extra_body["response_format"]["json_schema"]["schema"]

        assert action == ComputerActionType.GUI_SCREENSHOT
        assert request.extra_body["response_format"]["type"] == "json_schema"
        assert schema["properties"]["action"]["enum"] == [
            action.value
            for action in ComputerActionType
            if action in agent._GUI_ONLY_ACTIONS
            and action != ComputerActionType.GUI_TYPE
            and not action.name.startswith("BROWSER_")
        ] + ["GOAL_COMPLETE"]
        assert schema["required"] == ["action", "target", "payload", "expected"]
        assert "return exactly one JSON object" in request.messages[0].content
        assert "Return exactly one JSON object" in request.messages[1].content

    def test_gui_launch_does_not_treat_observed_controls_as_app_names(self, monkeypatch):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        monkeypatch.setattr(
            agent.perception_bus,
            "resolve",
            MagicMock(return_value=object()),
        )
        observation = MagicMock()

        error = agent._gui_action_target_error(
            ComputerActionType.APP_LAUNCH,
            "Applications menu",
            observation,
        )

        assert error is not None
        assert "use GUI_CLICK" in error

    def test_gui_focus_requires_an_observed_open_window(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), gui_only=True)
        observation = MagicMock()
        observation.active_application = "Desktop"
        observation.windows = ["Desktop", "workspace"]

        assert agent._gui_action_target_error(
            ComputerActionType.APP_FOCUS,
            "app_focus",
            observation,
        ) is not None
        assert agent._gui_action_target_error(
            ComputerActionType.APP_FOCUS,
            "workspace",
            observation,
        ) is None


class TestReasoningPrompt:
    """_build_reasoning_context includes GUI actions."""

    def test_system_prompt_includes_gui_actions(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        obs = _seed_fresh_gui_observation(agent)
        system_prompt, user_prompt = agent._build_reasoning_context(
            "test goal", obs, 1, "test.py", "test_test.py"
        )
        assert "GUI_CLICK" in system_prompt
        assert "GUI_KEYPRESS" in system_prompt
        assert "GUI_SCROLL" in system_prompt
        assert "GUI_SCREENSHOT" in system_prompt
        assert "EXECUTION GROUND-TRUTH" in system_prompt
        assert "visible cursor" in system_prompt
        assert "APP_FOCUS" in system_prompt
        assert "actual screen pixels" in user_prompt
        assert "Semantic GUI targets are resolved locally by AVCS" in user_prompt

    @pytest.mark.asyncio
    async def test_gui_action_selection_distinguishes_apps_from_controls(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content=(
                "ACTION: GUI_SCREENSHOT\nTARGET: visible desktop\n"
                "PAYLOAD: {}\nEXPECTED: Inspect desktop"
            ),
            reasoning_content="",
        ))
        obs = _seed_fresh_gui_observation(agent)
        obs.visible_text = "Desktop"
        obs.terminal_output = "ready"

        await agent._llm_choose_action("Inspect the desktop", obs, 1, "test.py", "test_test.py")

        request = agent.llm_router.complete.await_args.args[0]
        system_prompt = request.messages[0].content
        user_prompt = request.messages[1].content
        assert "Only these ACTION values are valid:" in system_prompt
        assert "GUI_CLICK" in system_prompt
        assert "TERMINAL_EXEC" not in system_prompt
        assert "Which GUI application window or terminal command" not in user_prompt
        assert "AUTONOMOUS TOOL AUTHORING" not in user_prompt
        assert "Never describe a desktop shortcut as an already-open window" in system_prompt
        assert "CURRENT AVCS PIXEL STATE" in user_prompt
        assert "no screenshot is sent" in user_prompt
        assert "Last Command Output" not in user_prompt
        assert '"command": "..."' not in user_prompt

    @pytest.mark.asyncio
    async def test_gui_action_selection_uses_json_schema_and_hides_private_reasoning(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content=(
                '{"action":"GUI_SCREENSHOT","target":"visible-desktop",'
                '"payload":{},"expected":"Refresh the visible desktop"}'
            ),
            reasoning_content="private reasoning that must not be surfaced",
            tool_calls=[],
        ))
        observation = _seed_fresh_gui_observation(agent)

        action, target, payload, expected = await agent._llm_choose_action(
            "Inspect the desktop",
            observation,
            1,
            "",
            "",
        )

        request = agent.llm_router.complete.await_args.args[0]
        assert action == ComputerActionType.GUI_SCREENSHOT
        assert target == "visible-desktop"
        assert payload == {}
        assert expected == "Refresh the visible desktop"
        assert request.tools is None
        assert request.extra_body["response_format"]["type"] == "json_schema"
        assert agent._last_thought == ""

    @pytest.mark.asyncio
    async def test_rejected_non_gui_proposal_is_explained_to_next_gui_decision(self):
        agent = ComputerUseAgent(
            computer_provider=_mock_provider(),
            execution_plane="gui",
        )
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(side_effect=[
            MagicMock(
                content=(
                    "ACTION: TERMINAL_EXEC\nTARGET: mkdir\n"
                    'PAYLOAD: {"command":"mkdir"}\nEXPECTED: Create folder'
                ),
                reasoning_content="",
                tool_calls=[],
            ),
            MagicMock(
                content=(
                    "ACTION: GUI_CLICK\nTARGET: 400,300\n"
                    "PAYLOAD: {}\nEXPECTED: Open the visible folder menu"
                ),
                reasoning_content="",
                tool_calls=[],
            ),
        ])
        observation = _seed_fresh_gui_observation(agent)

        first = await agent._llm_choose_action(
            "Create a folder using the visible desktop",
            observation,
            1,
            "",
            "",
        )
        assert first[0] == ComputerActionType.GUI_SCREENSHOT
        assert "TERMINAL_EXEC" in first[3]
        assert agent._pending_gui_contract_feedback

        second = await agent._llm_choose_action(
            "Create a folder using the visible desktop",
            observation,
            2,
            "",
            "",
        )
        second_request = agent.llm_router.complete.await_args_list[1].args[0]

        assert second[0] == ComputerActionType.GUI_CLICK
        assert "GUI CONTRACT FEEDBACK" in second_request.messages[1].content
        assert "TERMINAL_EXEC" in second_request.messages[1].content
        assert agent._pending_gui_contract_feedback == ""

    @pytest.mark.asyncio
    async def test_invalid_gui_response_gets_format_feedback_before_contract_stop(self):
        agent = ComputerUseAgent(
            computer_provider=_mock_provider(),
            execution_plane="gui",
        )
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(side_effect=[
            MagicMock(
                content="I will take a look at the screen now.",
                reasoning_content="",
                tool_calls=[],
            ),
            MagicMock(
                content=(
                    "ACTION: GUI_HOTKEY\nTARGET: keyboard\n"
                    'PAYLOAD: {"key":"Ctrl+L"}\nEXPECTED: Focus the location field'
                ),
                reasoning_content="",
                tool_calls=[],
            ),
        ])
        observation = _seed_fresh_gui_observation(agent)

        first = await agent._llm_choose_action(
            "Use the visible desktop",
            observation,
            1,
            "",
            "",
        )
        assert first[0] == ComputerActionType.GUI_SCREENSHOT
        assert "structured ACTION block" in first[3]
        assert agent._pending_gui_contract_feedback
        assert agent._consecutive_gui_contract_violations == 1

        second = await agent._llm_choose_action(
            "Use the visible desktop",
            observation,
            2,
            "",
            "",
        )
        second_request = agent.llm_router.complete.await_args_list[1].args[0]

        assert second[0] == ComputerActionType.GUI_HOTKEY
        assert "GUI CONTRACT FEEDBACK" in second_request.messages[1].content
        assert "structured ACTION block" in second_request.messages[1].content
        assert agent._consecutive_gui_contract_violations == 0
        assert agent._pending_gui_contract_feedback == ""

    @pytest.mark.asyncio
    async def test_repeated_invalid_gui_responses_stop_without_another_screenshot(self):
        agent = ComputerUseAgent(
            computer_provider=_mock_provider(),
            execution_plane="gui",
        )
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content="I will use the desktop.",
            reasoning_content="",
            tool_calls=[],
        ))
        observation = _seed_fresh_gui_observation(agent)

        first = await agent._llm_choose_action(
            "Use the visible desktop",
            observation,
            1,
            "",
            "",
        )
        second = await agent._llm_choose_action(
            "Use the visible desktop",
            observation,
            2,
            "",
            "",
        )

        assert first[0] == ComputerActionType.GUI_SCREENSHOT
        assert second[0] == ActionExecutionStatus.BLOCKED
        assert second[1] == "gui-contract-violation"
        assert agent.llm_router.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_valid_screenshot_does_not_reset_gui_contract_violation_streak(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")
        agent._consecutive_gui_contract_violations = 2
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content=(
                "ACTION: GUI_SCREENSHOT\nTARGET: visible desktop\n"
                "PAYLOAD: {}\nEXPECTED: Inspect desktop"
            ),
            reasoning_content="",
        ))
        observation = _seed_fresh_gui_observation(agent)

        action, *_ = await agent._llm_choose_action(
            "Inspect the desktop", observation, 1, "", ""
        )

        assert action == ComputerActionType.GUI_SCREENSHOT
        assert agent._consecutive_gui_contract_violations == 2

    @pytest.mark.asyncio
    async def test_repeated_non_gui_proposals_stop_gui_mission(self):
        provider = _mock_provider()
        agent = ComputerUseAgent(
            computer_provider=provider,
            llm_router=MagicMock(),
            execution_plane="gui",
        )
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content=(
                "ACTION: TERMINAL_EXEC\nTARGET: pwd\n"
                "PAYLOAD: {\"command\":\"pwd\"}\nEXPECTED: Inspect workspace"
            ),
            reasoning_content="",
        ))
        agent._bootstrap_performed = True
        agent.observe = AsyncMock(return_value=ComputerWorldObservation())
        step_callback = AsyncMock()

        traces = await agent.run_mission(
            "ws-1",
            "Use the visible desktop to inspect the workspace",
            steps=8,
            step_callback=step_callback,
        )

        assert len(traces) == 2
        assert traces[-1].status == ActionExecutionStatus.BLOCKED
        assert "gui-contract-violation" in traces[-1].actual_observation
        assert "outside the visible GUI action contract" in traces[-1].actual_observation
        assert step_callback.await_count == 2
        assert step_callback.await_args.args[0] is traces[-1]
        provider.gui_action.assert_not_awaited()

    def test_gui_reasoning_context_omits_structured_controls_from_visual_prompt(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")
        snapshot = agent.perception_bus.publish(
            screenshot_base64=_FRESH_FRAME,
            width=1280,
            height=800,
            controls=[
                {
                    "name": "File",
                    "role": "menuitem",
                    "bbox": (12, 20, 56, 42),
                    "bbox_format": "xyxy",
                    "source": "ocr",
                    "confidence": 0.98,
                }
            ],
        )
        observation = ComputerWorldObservation(
            screen=ScreenObservation(
                screenshot_base64=_FRESH_FRAME,
                width=1280,
                height=800,
                visible_text="File Edit View",
            ),
            visible_text="File Edit View",
            perception_version=snapshot.version,
        )

        _, user_prompt = agent._build_reasoning_context(
            "Create a folder", observation, 1, "", ""
        )

        assert "Grounded visible controls" not in user_prompt
        assert '"label":"File"' not in user_prompt
        assert "File Edit View" not in user_prompt
        assert "CURRENT AVCS PIXEL STATE" in user_prompt
        assert "no screenshot is sent" in user_prompt

    @pytest.mark.asyncio
    async def test_gui_reasoning_uses_local_avcs_state_without_exposing_controls(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")
        agent._last_screenshot_b64 = _FRESH_FRAME
        snapshot = agent.perception_bus.publish(
            screenshot_base64=_FRESH_FRAME,
            width=1280,
            height=800,
            controls=[
                {
                    "name": "File",
                    "role": "menuitem",
                    "bbox": (12, 20, 56, 42),
                    "bbox_format": "xyxy",
                    "source": "ocr",
                }
            ],
        )
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content=(
                "ACTION: GUI_CLICK\nTARGET: File\nPAYLOAD: {}\n"
                "EXPECTED: File menu opens"
            ),
            reasoning_content="Use the observed File menu.",
        ))
        screen = agent._ensure_avcs_state(
            ScreenObservation(
                screenshot_base64=_FRESH_FRAME,
                width=1280,
                height=800,
                capture_age_ms=0,
                controls_age_ms=0,
                visible_text="File Edit View",
            )
        )
        observation = ComputerWorldObservation(
            screen=screen,
            perception_version=screen.perception_version,
        )

        await agent._llm_choose_action("Open the File menu", observation, 1, "", "")

        request = agent.llm_router.complete.await_args.args[0]
        assert not getattr(request.messages[1], "images", [])
        assert request.task_type == "reasoning"
        assert "CURRENT AVCS PIXEL STATE" in request.messages[1].content
        assert "Grounded visible controls" not in request.messages[1].content
        assert "File Edit View" not in request.messages[1].content

    @pytest.mark.asyncio
    async def test_gui_reasoning_uses_avcs_state_without_controls(self):
        agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")
        agent._last_screenshot_b64 = _FRESH_FRAME
        snapshot = agent.perception_bus.publish(
            screenshot_base64=_FRESH_FRAME,
            width=1280,
            height=800,
        )
        agent.llm_router = MagicMock()
        agent.llm_router.complete = AsyncMock(return_value=MagicMock(
            content=(
                "ACTION: GUI_SCREENSHOT\nTARGET: desktop\nPAYLOAD: {}\n"
                "EXPECTED: Observe the desktop"
            ),
            reasoning_content="No structured controls were detected.",
        ))
        screen = agent._ensure_avcs_state(
            ScreenObservation(
                screenshot_base64=_FRESH_FRAME,
                width=1280,
                height=800,
                capture_age_ms=0,
                controls_age_ms=0,
            )
        )
        observation = ComputerWorldObservation(
            screen=screen,
            perception_version=screen.perception_version,
        )

        await agent._llm_choose_action("Inspect the desktop", observation, 1, "", "")

        request = agent.llm_router.complete.await_args.args[0]
        assert not getattr(request.messages[1], "images", [])
        assert "CURRENT AVCS PIXEL STATE" in request.messages[1].content
        assert request.task_type == "reasoning"


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
