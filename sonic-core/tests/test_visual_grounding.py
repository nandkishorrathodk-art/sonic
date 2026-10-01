"""
Tests for visual grounding engine (adapted from open-computer-use).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.computer.models import GUIActionType, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.grounding import draw_action_marker, extract_bbox_midpoint
from sonic.computer_use.models import ComputerActionType
from sonic.llm.prompts import grounding_user_prompt


def test_extract_bbox_midpoint_box_tags():
    # Tagged normalized 0-1000 format
    resp = "<|box_start|>(100, 200, 300, 400)<|box_end|>"
    pt = extract_bbox_midpoint(resp, width=1280, height=800)
    assert pt is not None
    # midpoint of 100 and 300 is 200 (200 / 1000 * 1280 = 256)
    # midpoint of 200 and 400 is 300 (300 / 1000 * 800 = 240)
    assert pt == (256, 240)


def test_extract_bbox_midpoint_float_range():
    # 0.0 - 1.0 range
    resp = "0.5, 0.5"
    pt = extract_bbox_midpoint(resp, width=1280, height=800)
    assert pt == (640, 400)


def test_grounding_prompt_declares_normalized_coordinate_space():
    prompt = grounding_user_prompt("visible button", 1280, 800)

    assert "normalized 0-1000 screen space" in prompt
    assert 'Return exactly one JSON object: {"x": <integer>, "y": <integer>}' in prompt
    assert '{"not_found": true}' in prompt


@pytest.mark.asyncio
async def test_multimodal_grounding_json_point_uses_normalized_screen_space():
    from sonic.computer_use.grounding import query_multimodal_grounding

    router = SimpleNamespace(
        complete=AsyncMock(return_value=SimpleNamespace(content='{"x": 500, "y": 500}'))
    )
    screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    coords = await query_multimodal_grounding(
        router,
        "visible button",
        screenshot,
        width=1280,
        height=800,
    )

    assert coords == (640, 400)
    request = router.complete.await_args.args[0]
    assert request.extra_body == {"response_format": {"type": "json_object"}}


@pytest.mark.asyncio
async def test_multimodal_grounding_rejects_prose_coordinates():
    from sonic.computer_use.grounding import query_multimodal_grounding

    router = SimpleNamespace(
        complete=AsyncMock(
            return_value=SimpleNamespace(
                content="The center of the menu is at (0.5, 0.5) normalized coordinates."
            )
        )
    )
    screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    coords = await query_multimodal_grounding(
        router,
        "Applications menu",
        screenshot,
        width=1280,
        height=800,
    )

    assert coords is None


@pytest.mark.asyncio
async def test_multimodal_grounding_accepts_strict_normalized_box():
    from sonic.computer_use.grounding import query_multimodal_grounding

    router = SimpleNamespace(
        complete=AsyncMock(
            return_value=SimpleNamespace(
                content="<|box_start|>(10, 20, 30, 40)<|box_end|>"
            )
        )
    )
    screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    coords = await query_multimodal_grounding(
        router,
        "Applications menu",
        screenshot,
        width=1280,
        height=800,
    )

    assert coords == (26, 24)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,expected",
    [
        ('{"x": 1000, "y": 500}', None),
        ('{"x": true, "y": 500}', None),
        ('{"not_found": true}', None),
        ('{"x": 0, "y": 0}', (0, 0)),
    ],
)
async def test_multimodal_grounding_rejects_invalid_or_offscreen_points(response, expected):
    from sonic.computer_use.grounding import query_multimodal_grounding

    router = SimpleNamespace(
        complete=AsyncMock(return_value=SimpleNamespace(content=response))
    )
    screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    coords = await query_multimodal_grounding(
        router,
        "visible control",
        screenshot,
        width=1280,
        height=800,
    )

    assert coords == expected


def test_draw_action_marker_fallback():
    dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    res = draw_action_marker(dummy_b64, (50, 50))
    assert res is not None
    assert len(res) > 0


def test_resolve_ui_target_direct_coords():
    from sonic.computer_use.grounding import resolve_ui_target
    pt = resolve_ui_target("800,600", width=1280, height=800)
    assert pt == (800, 600)


def test_resolve_ui_target_no_landmarks():
    from sonic.computer_use.grounding import resolve_ui_target
    # Verifies that arbitrary text queries without coordinates or vision return None (no landmark guessing)
    assert resolve_ui_target("Applications menu", width=1280, height=800) is None
    assert resolve_ui_target("Terminal icon", width=1280, height=800) is None
    assert resolve_ui_target("Google Chrome", width=1280, height=800) is None
    assert resolve_ui_target("Window close", width=1280, height=800) is None


def test_resolve_ui_target_grounding_fn():
    from sonic.computer_use.grounding import resolve_ui_target

    def mock_grounding(query: str, image_b64: str) -> str:
        return "<|box_start|>(500, 500, 500, 500)<|box_end|>"

    dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    pt = resolve_ui_target("Custom button", screenshot_b64=dummy_b64, width=1280, height=800, grounding_fn=mock_grounding)
    assert pt == (640, 400)


def test_resolve_ui_target_applies_coordinate_space_to_callback_tuple():
    from sonic.computer_use.grounding import resolve_ui_target

    dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    pt = resolve_ui_target(
        "visible application control",
        screenshot_b64=dummy_b64,
        width=1280,
        height=800,
        grounding_fn=lambda *_: (500, 400),
        coordinate_space="normalized_1000",
    )

    assert pt == (640, 320)


@pytest.mark.asyncio
async def test_resolve_ui_target_async_applies_coordinate_space_to_callback_tuple():
    from sonic.computer_use.grounding import resolve_ui_target_async

    dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    pt = await resolve_ui_target_async(
        "visible application control",
        screenshot_b64=dummy_b64,
        width=1280,
        height=800,
        grounding_fn=lambda *_: (500, 400),
        coordinate_space="normalized_1000",
    )

    assert pt == (640, 320)


def test_draw_action_marker_styles():
    dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    m_click = draw_action_marker(dummy_b64, (20, 20), label="CLICK")
    m_right = draw_action_marker(dummy_b64, (20, 20), label="RIGHT_CLICK")
    m_double = draw_action_marker(dummy_b64, (20, 20), label="DOUBLE_CLICK")
    assert m_click and m_right and m_double
    assert len(m_click) > 0


@pytest.mark.asyncio
async def test_agent_visual_grounding_click_dispatch():
    mock_computer = MagicMock()
    mock_computer.gui_action = AsyncMock(return_value=ScreenObservation(screenshot_base64="", width=1280, height=800))
    agent = ComputerUseAgent(computer_provider=mock_computer)
    agent._screen_width = 1280
    agent._screen_height = 800

    trace = await agent.execute_action(
        workspace_id="test-ws",
        action_type=ComputerActionType.GUI_CLICK,
        target_resource="Applications menu",
        payload={"x": 20, "y": 12},
        predicted_outcome="Open application menu",
    )

    assert trace.status == "SUCCESS"
    assert "Clicked at (20, 12)" in trace.actual_observation
    assert mock_computer.gui_action.called
    gui_call = mock_computer.gui_action.call_args[0][1]
    assert gui_call.action == GUIActionType.CLICK
    assert gui_call.x == 20
    assert gui_call.y == 12


@pytest.mark.asyncio
async def test_agent_click_blocked_when_unresolved():
    mock_computer = MagicMock()
    agent = ComputerUseAgent(computer_provider=mock_computer)
    trace = await agent.execute_action(
        workspace_id="test-ws",
        action_type=ComputerActionType.GUI_CLICK,
        target_resource="Applications menu",
        payload={},
        predicted_outcome="Open application menu",
    )
    assert trace.status == "BLOCKED"
    assert "could not be resolved from visual grounding" in trace.actual_observation
    assert not mock_computer.gui_action.called


@pytest.mark.asyncio
async def test_gui_worker_instructs_replanning_after_unresolved_target(monkeypatch):
    import sonic.computer_use.agent as agent_module

    computer = MagicMock()
    agent = ComputerUseAgent(computer_provider=computer, execution_plane="gui")
    agent.llm_router = MagicMock()
    agent._last_screenshot_b64 = "fresh-frame"
    agent._last_reasoning_perception = {"image_attached": True}
    monkeypatch.setattr(
        agent_module,
        "query_multimodal_grounding",
        AsyncMock(return_value=None),
    )

    trace = await agent.execute_action(
        "test-ws",
        ComputerActionType.GUI_CLICK,
        "File System",
        {},
        "Open the observed location",
    )

    assert trace.status == "BLOCKED"
    assert "Do not repeat that target" in agent._pending_gui_contract_feedback
    assert "keyboard navigation" in agent._pending_gui_contract_feedback
    assert not computer.gui_action.called


@pytest.mark.asyncio
async def test_agent_uses_explicit_payload_pixels_without_semantic_regrounding(monkeypatch):
    import sonic.computer_use.agent as agent_module

    computer = MagicMock()
    computer.status = AsyncMock(return_value=SimpleNamespace(
        active_application="Desktop",
        active_window="Desktop",
    ))
    computer.gui_action = AsyncMock(return_value=ScreenObservation(width=1280, height=800))
    agent = ComputerUseAgent(computer_provider=computer, llm_router=object())
    agent._last_screenshot_b64 = "visible-screen"
    grounding = AsyncMock(return_value=None)
    monkeypatch.setattr(
        agent_module,
        "query_multimodal_grounding",
        grounding,
    )

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_CLICK,
        "maximize button",
        {"x": 640, "y": 400},
        "Maximize the window",
    )

    assert trace.status == "COMPLETED"
    grounding.assert_not_awaited()
    computer.gui_action.assert_awaited_once()
    gui_action = computer.gui_action.await_args.args[1]
    assert (gui_action.x, gui_action.y) == (640, 400)


@pytest.mark.asyncio
async def test_agent_gui_right_click_dispatch():
    mock_computer = MagicMock()
    mock_computer.gui_action = AsyncMock(return_value=ScreenObservation(screenshot_base64="", width=1280, height=800))
    agent = ComputerUseAgent(computer_provider=mock_computer)
    agent._screen_width = 1280
    agent._screen_height = 800

    trace = await agent.execute_action(
        workspace_id="test-ws",
        action_type=ComputerActionType.GUI_RIGHT_CLICK,
        target_resource="Terminal",
        payload={"x": 48, "y": 12},
        predicted_outcome="Open terminal context menu",
    )

    assert trace.status == "SUCCESS"
    assert "Right-clicked at (48, 12)" in trace.actual_observation
    assert mock_computer.gui_action.called
    gui_call = mock_computer.gui_action.call_args[0][1]
    assert gui_call.action == GUIActionType.RIGHT_CLICK
    assert gui_call.x == 48
    assert gui_call.y == 12


@pytest.mark.asyncio
async def test_browser_fallback_does_not_type_when_desktop_target_is_unresolved(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    import sonic.computer_use.agent as agent_module

    computer = MagicMock()
    computer.gui_action = AsyncMock(return_value=ScreenObservation(width=1280, height=800))
    agent = ComputerUseAgent(computer_provider=computer, llm_router=object())
    agent._last_screenshot_b64 = "visible-screen"
    monkeypatch.setattr(agent_module, "query_multimodal_grounding", AsyncMock(return_value=None))

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.BROWSER_TYPE,
        "search field",
        {"selector": "search field", "text": "hello"},
        "type into the visible search field",
    )

    assert trace.status == "BLOCKED"
    computer.gui_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_browser_fallback_selects_grounded_visible_option(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    import sonic.computer_use.agent as agent_module

    computer = MagicMock()
    computer.gui_action = AsyncMock(
        side_effect=[
            ScreenObservation(screenshot_base64="updated-menu", width=1280, height=800),
            ScreenObservation(screenshot_base64="selected-menu", width=1280, height=800),
        ]
    )
    agent = ComputerUseAgent(computer_provider=computer, llm_router=object())
    agent._last_screenshot_b64 = "initial-screen"
    grounding = AsyncMock(side_effect=[(300, 200), (500, 400)])
    monkeypatch.setattr(agent_module, "query_multimodal_grounding", grounding)

    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.BROWSER_SELECT,
        "country selector",
        {"selector": "country selector", "value": "visible country"},
        "select the visible option",
    )

    assert trace.status == "SUCCESS"
    assert grounding.await_count == 2
    assert [call.args[1].action for call in computer.gui_action.await_args_list] == [
        GUIActionType.CLICK,
        GUIActionType.CLICK,
    ]
    assert (computer.gui_action.await_args_list[0].args[1].x,
            computer.gui_action.await_args_list[0].args[1].y) == (300, 200)
    assert (computer.gui_action.await_args_list[1].args[1].x,
            computer.gui_action.await_args_list[1].args[1].y) == (500, 400)
