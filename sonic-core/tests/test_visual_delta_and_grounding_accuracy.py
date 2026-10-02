"""
Unit tests for Visual Delta Verification, Grounding Accuracy, and Screen Scaling.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from sonic.computer_use.grounding import (
    extract_bbox_midpoint,
    resolve_ui_target,
)
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
from sonic.computer.models import ScreenObservation


def test_extract_bbox_midpoint_formats():
    # Direct tuple/list (x, y)
    assert extract_bbox_midpoint((100, 200), width=1280, height=800) == (100, 200)
    assert extract_bbox_midpoint([350, 450], width=1280, height=800) == (350, 450)

    # 4-element raw pixel box [x1, y1, x2, y2]
    mid_raw = extract_bbox_midpoint([100, 200, 300, 400], width=1280, height=800, is_normalized_1000=False)
    assert mid_raw == (200, 300)

    # 4-element model grounding tag <|box_start|>(200, 100, 400, 300)<|box_end|> normalized [0, 1000]
    mid_tag = extract_bbox_midpoint("<|box_start|>(200, 100, 400, 300)<|box_end|>", width=1280, height=800)
    # x = (200 + 400) / 2 = 300 -> 300/1000 * 1280 = 384
    # y = (100 + 300) / 2 = 200 -> 200/1000 * 800 = 160
    assert mid_tag == (384, 160)

    # Box coordinates exceeding screen bounds scale from [0, 1000]
    mid_scale = extract_bbox_midpoint([100, 900, 300, 950], width=1280, height=800)
    # y coordinates 900, 950 exceed height (800), triggering [0, 1000] normalization
    assert mid_scale[1] == int(((900 + 950) / 2.0 / 1000.0) * 800)


def test_grounding_rejects_offscreen_points_instead_of_clamping():
    assert extract_bbox_midpoint((1280, 400), width=1280, height=800) is None
    assert extract_bbox_midpoint((400, 800), width=1280, height=800) is None
    assert extract_bbox_midpoint((1300, 400), width=1280, height=800) is None
    assert extract_bbox_midpoint("-10, 400", width=1280, height=800) is None
    assert extract_bbox_midpoint("<|box_start|>(800, 500, 1000, 700)<|box_end|>", width=1280, height=800) == (1152, 480)


def test_landmark_substring_collision_prevention():
    # Without coordinates or vision grounding, text queries fail closed to None (no blind clicking)
    assert resolve_ui_target("token", width=1280, height=800) is None
    assert resolve_ui_target("ok", width=1280, height=800) is None
    assert resolve_ui_target("search bar", width=1280, height=800) is None

    # Direct coordinates resolve cleanly
    assert resolve_ui_target("704, 440", width=1280, height=800) == (704, 440)
    assert resolve_ui_target("640, 400", width=1280, height=800) == (640, 400)


@pytest.mark.asyncio
async def test_visual_verification_detects_unchanged_screen():
    mock_computer = MagicMock()
    # Mock gui_action returning the EXACT same screenshot as pre-action
    same_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    mock_obs = ScreenObservation(
        screenshot_base64=same_b64,
        width=1280,
        height=800,
        desktop_state="INTERACTIVE",
        active_window="Desktop",
        capture_age_ms=0,
        controls_age_ms=0,
    )
    mock_computer.gui_action = AsyncMock(return_value=mock_obs)
    mock_computer.execute = AsyncMock(return_value=MagicMock(exit_code=0, stdout="ok", stderr=""))
    mock_computer.screenshot = AsyncMock(return_value=mock_obs)
    mock_computer.status = AsyncMock(return_value=MagicMock(
        active_application="Desktop",
        active_window="Desktop",
        open_applications=["Desktop"],
        running_processes=[],
    ))

    agent = ComputerUseAgent(computer_provider=mock_computer)
    agent._last_screenshot_b64 = same_b64
    agent._last_observed_screen = mock_obs
    agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
    agent.perception_bus.publish(
        screenshot_base64=same_b64,
        width=1280,
        height=800,
        active_window="Desktop",
    )

    trace = await agent.execute_action(
        workspace_id="ws-test",
        action_type=ComputerActionType.GUI_CLICK,
        target_resource="500,300",
        payload={"x": 500, "y": 300},
        predicted_outcome="search bar clicked",
    )

    assert "[VISUAL VERIFICATION]: Screen state unchanged" in trace.actual_observation
    assert trace.status == ActionExecutionStatus.UNVERIFIED
    assert trace.recovery_attempted is False
