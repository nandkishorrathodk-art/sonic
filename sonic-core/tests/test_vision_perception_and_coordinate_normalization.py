"""
Tests for Vision Perception & Coordinate Normalization in sonic-core:
1. GUI/dual reasoning attaches desktop pixels when current structured state is
   unavailable, and avoids image transfer for a fresh, unambiguous accessibility
   snapshot.
2. extract_bbox_midpoint scales normalized 0-1000 coordinates to
   (raw_x / 1000.0) * width and (raw_y / 1000.0) * height in 1280x800 screens.
3. ScreenObservation supports active_window_geometry (x, y, w, h) in various formats.
4. crop_active_window_region crops high-res foveated window regions and passes them
   alongside the full desktop screenshot to the vision model.
"""

import base64
import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image

from sonic.computer.models import ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.grounding import (
    crop_active_window_region,
    extract_bbox_midpoint,
)
from sonic.computer_use.models import ComputerWorldObservation


def _make_test_image_b64(width: int = 1280, height: int = 800, color: str = "red") -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _mock_provider():
    p = AsyncMock()
    p.screenshot.return_value = ScreenObservation(width=1280, height=800)
    p.status.return_value = MagicMock(
        active_application="Desktop",
        active_window="Desktop",
        open_applications=["Desktop"],
        running_processes=["Xorg"],
    )
    p.list_files.return_value = []
    p.git_action.return_value = MagicMock(branch="main", is_clean=True)
    p.terminal.return_value = MagicMock(stdout="__sonic_obs_ready__", stderr="", exit_code=0)
    p.gui_action.return_value = ScreenObservation(width=1280, height=800)
    return p


# ==============================================================================
# 1. Image escalation vs. fresh structured GUI state
# ==============================================================================

@pytest.mark.asyncio
async def test_should_attach_image_without_fresh_structured_state_in_dual_plane():
    provider = _mock_provider()
    agent = ComputerUseAgent(
        computer_provider=provider,
        execution_plane="dual",
    )
    fake_b64 = _make_test_image_b64()
    agent._last_screenshot_b64 = fake_b64

    captured_requests = []
    async def mock_complete(req):
        captured_requests.append(req)
        mock_resp = MagicMock()
        mock_resp.content = 'THOUGHT: test\nACTION: GUI_CLICK\nTARGET: 640,400\nPAYLOAD: {"x": 640, "y": 400}\nEXPECTED: click'
        return mock_resp

    mock_router = MagicMock()
    mock_router.complete = AsyncMock(side_effect=mock_complete)
    agent.llm_router = mock_router

    obs = ComputerWorldObservation(
        screen=ScreenObservation(screenshot_base64=fake_b64, width=1280, height=800),
    )
    await agent._llm_choose_action(
        goal="Click button",
        observation=obs,
        step_index=0,
        primary_file="",
        test_file="",
    )

    assert len(captured_requests) == 1
    user_msg = captured_requests[0].messages[1]
    assert len(user_msg.images) >= 1
    assert user_msg.images[0].base64 == fake_b64


@pytest.mark.asyncio
async def test_should_attach_image_without_fresh_structured_state_in_gui_plane():
    provider = _mock_provider()
    agent = ComputerUseAgent(
        computer_provider=provider,
        execution_plane="gui",
    )
    fake_b64 = _make_test_image_b64()
    agent._last_screenshot_b64 = fake_b64

    captured_requests = []
    async def mock_complete(req):
        captured_requests.append(req)
        mock_resp = MagicMock()
        mock_resp.content = 'THOUGHT: test\nACTION: GUI_CLICK\nTARGET: 640,400\nPAYLOAD: {"x": 640, "y": 400}\nEXPECTED: click'
        return mock_resp

    mock_router = MagicMock()
    mock_router.complete = AsyncMock(side_effect=mock_complete)
    agent.llm_router = mock_router

    obs = ComputerWorldObservation(
        screen=ScreenObservation(screenshot_base64=fake_b64, width=1280, height=800),
    )
    await agent._llm_choose_action(
        goal="Click button",
        observation=obs,
        step_index=0,
        primary_file="",
        test_file="",
    )

    assert len(captured_requests) == 1
    user_msg = captured_requests[0].messages[1]
    assert len(user_msg.images) >= 1
    assert user_msg.images[0].base64 == fake_b64


@pytest.mark.asyncio
async def test_fresh_accessibility_state_avoids_image_transfer():
    provider = _mock_provider()
    agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
    fake_b64 = _make_test_image_b64()
    agent._last_screenshot_b64 = fake_b64
    snapshot = agent.perception_bus.publish(
        screenshot_base64=fake_b64,
        width=1280,
        height=800,
        active_window="Document",
        controls=[
            {
                "name": "Save",
                "role": "button",
                "bbox": (30, 40, 90, 70),
                "bbox_format": "xyxy",
                "source": "atspi",
                "confidence": 0.95,
                "freshness_ms": 10,
            }
        ],
    )

    captured_requests = []

    async def mock_complete(req):
        captured_requests.append(req)
        mock_resp = MagicMock()
        mock_resp.content = (
            'THOUGHT: grounded controls are clear\n'
            'ACTION: GUI_CLICK\nTARGET: Save\nPAYLOAD: {}\nEXPECTED: saved'
        )
        return mock_resp

    mock_router = MagicMock()
    mock_router.complete = AsyncMock(side_effect=mock_complete)
    agent.llm_router = mock_router

    obs = ComputerWorldObservation(
        screen=ScreenObservation(
            screenshot_base64=fake_b64,
            desktop_state="INTERACTIVE",
            capture_age_ms=0,
            controls_age_ms=10,
            width=1280,
            height=800,
            active_window="Document",
            detected_controls=[
                {
                    "name": "Save",
                    "role": "button",
                    "bbox": (30, 40, 90, 70),
                    "bbox_format": "xyxy",
                    "source": "atspi",
                    "confidence": 0.95,
                    "freshness_ms": 10,
                }
            ],
        ),
        active_application="Document",
        perception_version=snapshot.version,
        observation_age_ms=25,
    )
    await agent._llm_choose_action(
        goal="Save the current document",
        observation=obs,
        step_index=1,
        primary_file="",
        test_file="",
    )

    assert len(captured_requests) == 1
    user_msg = captured_requests[0].messages[1]
    assert user_msg.images == []
    assert "fresh, grounded accessibility controls" in user_msg.content
    assert "fresh accessibility/control inventory" in captured_requests[0].messages[0].content
    assert "attached current screenshot and live observation" not in captured_requests[0].messages[0].content
    assert obs.accessible_scene_fresh is True
    assert obs.image_attached is False
    assert obs.visual_escalation_reason == "fresh_accessibility_scene"


@pytest.mark.asyncio
async def test_stale_accessibility_state_escalates_to_image():
    provider = _mock_provider()
    agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
    fake_b64 = _make_test_image_b64()
    agent._last_screenshot_b64 = fake_b64
    snapshot = agent.perception_bus.publish(
        screenshot_base64=fake_b64,
        width=1280,
        height=800,
        active_window="Document",
        controls=[
            {
                "name": "Save",
                "role": "button",
                "bbox": (30, 40, 90, 70),
                "bbox_format": "xyxy",
                "source": "atspi",
            }
        ],
    )

    captured_requests = []

    async def mock_complete(req):
        captured_requests.append(req)
        mock_resp = MagicMock()
        mock_resp.content = (
            'THOUGHT: inspect the current visual state\n'
            'ACTION: GUI_CLICK\nTARGET: 640,400\n'
            'PAYLOAD: {"x":640,"y":400}\nEXPECTED: clicked'
        )
        return mock_resp

    mock_router = MagicMock()
    mock_router.complete = AsyncMock(side_effect=mock_complete)
    agent.llm_router = mock_router
    obs = ComputerWorldObservation(
        screen=ScreenObservation(screenshot_base64=fake_b64, width=1280, height=800),
        perception_version=snapshot.version,
        observation_age_ms=800,
    )

    await agent._llm_choose_action(
        goal="Inspect and continue",
        observation=obs,
        step_index=1,
        primary_file="",
        test_file="",
    )

    assert len(captured_requests) == 1
    assert captured_requests[0].messages[1].images
    assert obs.image_attached is True
    assert obs.visual_escalation_reason == "accessibility_scene_stale_ambiguous_or_missing"
    assert obs.image_escalation_count == 1
    assert agent._perception_counters["visual_escalations"] == 1


@pytest.mark.asyncio
async def test_ambiguous_state_fetches_visual_frame_only_on_demand():
    provider = _mock_provider()
    visual_frame = _make_test_image_b64(640, 480)
    provider.screenshot = AsyncMock(
        return_value=ScreenObservation(
            screenshot_base64=visual_frame,
            width=640,
            height=480,
            active_window="Document",
        )
    )
    agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
    agent._active_workspace_id = "ws-1"
    snapshot = agent.perception_bus.publish(
        width=1280,
        height=800,
        active_window="Document",
        controls=[
            {
                "name": "Save",
                "role": "text",
                "bbox": (30, 40, 90, 70),
                "bbox_format": "xyxy",
                "source": "ocr",
                "confidence": 96,
            }
        ],
    )

    captured_requests = []

    async def mock_complete(req):
        captured_requests.append(req)
        mock_resp = MagicMock()
        mock_resp.content = (
            'THOUGHT: the visual frame clarifies the control\n'
            'ACTION: GUI_CLICK\nTARGET: 300,200\n'
            'PAYLOAD: {"x":300,"y":200}\nEXPECTED: clicked'
        )
        return mock_resp

    mock_router = MagicMock()
    mock_router.complete = AsyncMock(side_effect=mock_complete)
    agent.llm_router = mock_router
    obs = ComputerWorldObservation(
        screen=ScreenObservation(
            width=1280,
            height=800,
            active_window="Document",
            capture_age_ms=25,
            controls_age_ms=25,
        ),
        active_application="Document",
        perception_version=snapshot.version,
        observation_age_ms=25,
    )

    await agent._llm_choose_action(
        goal="Save the document",
        observation=obs,
        step_index=1,
        primary_file="",
        test_file="",
    )

    assert provider.screenshot.await_count == 1
    assert len(captured_requests) == 1
    assert captured_requests[0].messages[1].images[0].base64 == visual_frame
    assert obs.screen.screenshot_base64 == visual_frame


def test_ambiguous_accessibility_controls_do_not_qualify_for_fast_path():
    agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")
    snapshot = agent.perception_bus.publish(
        screenshot_base64="frame",
        width=1280,
        height=800,
        active_window="Document",
        controls=[
            {
                "name": "Save",
                "role": "button",
                "bbox": (30, 40, 90, 70),
                "bbox_format": "xyxy",
                "source": "atspi",
            },
            {
                "name": "Save",
                "role": "button",
                "bbox": (930, 40, 990, 70),
                "bbox_format": "xyxy",
                "source": "atspi",
            },
        ],
    )
    observation = ComputerWorldObservation(
        screen=ScreenObservation(width=1280, height=800, active_window="Document"),
        perception_version=snapshot.version,
        observation_age_ms=25,
    )

    assert not agent._has_fresh_accessible_scene(observation)


@pytest.mark.asyncio
async def test_should_not_attach_image_in_headless_plane_without_screenshot_action():
    provider = _mock_provider()
    agent = ComputerUseAgent(
        computer_provider=provider,
        execution_plane="headless",
    )
    fake_b64 = _make_test_image_b64()
    agent._last_screenshot_b64 = fake_b64

    captured_requests = []
    async def mock_complete(req):
        captured_requests.append(req)
        mock_resp = MagicMock()
        mock_resp.content = 'THOUGHT: test\nACTION: TERMINAL_EXEC\nTARGET: ls\nPAYLOAD: {"command": "ls"}\nEXPECTED: list'
        return mock_resp

    mock_router = MagicMock()
    mock_router.complete = AsyncMock(side_effect=mock_complete)
    agent.llm_router = mock_router

    obs = ComputerWorldObservation(
        screen=ScreenObservation(screenshot_base64=fake_b64, width=1280, height=800),
        perception_version=1,
    )
    await agent._llm_choose_action(
        goal="List files",
        observation=obs,
        step_index=0,
        primary_file="",
        test_file="",
    )

    assert len(captured_requests) == 1
    user_msg = captured_requests[0].messages[1]
    assert len(user_msg.images) == 0


# ==============================================================================
# 2. Coordinate Scaling in 1280x800
# ==============================================================================

def test_extract_bbox_midpoint_normalized_scaling_1280x800():
    # Model returns normalized coordinates [500, 400] on a 1280x800 display
    # x = 500 / 1000 * 1280 = 640
    # y = 400 / 1000 * 800 = 320
    coords = extract_bbox_midpoint("[500, 400]", width=1280, height=800)
    assert coords == (640, 320)

    # Comma-separated normalized coordinates
    coords2 = extract_bbox_midpoint("500, 400", width=1280, height=800)
    assert coords2 == (640, 320)

    # Box tags normalized [0, 1000]
    coords_tag = extract_bbox_midpoint("<|box_start|>(200, 300, 400, 500)<|box_end|>", width=1280, height=800)
    # x = (200 + 400) / 2 = 300 -> 300/1000 * 1280 = 384
    # y = (300 + 500) / 2 = 400 -> 400/1000 * 800 = 320
    assert coords_tag == (384, 320)

    # Explicit pixel coordinates (is_normalized_1000=False) preserve raw values
    coords_px = extract_bbox_midpoint("[500, 400]", width=1280, height=800, is_normalized_1000=False)
    assert coords_px == (500, 400)


# ==============================================================================
# 3. ScreenObservation Active Window Geometry
# ==============================================================================

def test_screen_observation_active_window_geometry():
    # Direct tuple
    obs = ScreenObservation(
        active_window_geometry=(50, 100, 800, 600),
        active_window="Browser",
    )
    assert obs.active_window_geometry == (50, 100, 800, 600)
    assert obs.active_window_geom == (50, 100, 800, 600)

    # Dict format
    obs_dict = ScreenObservation(
        active_window_geom={"x": 30, "y": 40, "width": 640, "height": 480},
    )
    assert obs_dict.active_window_geometry == (30, 40, 640, 480)

    # Object with bbox
    class MockWindow:
        bbox = (10, 20, 500, 400)
    obs_obj = ScreenObservation(active_window_geometry=MockWindow())
    assert obs_obj.active_window_geometry == (10, 20, 500, 400)


# ==============================================================================
# 4. crop_active_window_region: foveated active window crop
# ==============================================================================

def test_crop_active_window_region_cropping():
    b64 = _make_test_image_b64(width=1280, height=800)
    geom = (100, 150, 500, 400)

    # Default returns (cropped_b64, (x, y))
    cropped_b64, offset = crop_active_window_region(b64, geometry=geom)
    assert offset == (100, 150)
    assert cropped_b64 != ""

    # Verify cropped image dimensions
    raw = base64.b64decode(cropped_b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (500, 400)

    # With padding=10
    cropped_pad, offset_pad = crop_active_window_region(b64, geometry=geom, padding=10)
    assert offset_pad == (90, 140)
    raw_pad = base64.b64decode(cropped_pad)
    img_pad = Image.open(io.BytesIO(raw_pad))
    assert img_pad.size == (520, 420)

    # With return_offset=False
    raw_str = crop_active_window_region(b64, geometry=geom, return_offset=False)
    assert isinstance(raw_str, str)
    assert len(raw_str) > 50


@pytest.mark.asyncio
async def test_llm_choose_action_attaches_foveated_active_window_crop():
    provider = _mock_provider()
    agent = ComputerUseAgent(
        computer_provider=provider,
        execution_plane="dual",
    )
    fake_b64 = _make_test_image_b64(width=1280, height=800)
    agent._last_screenshot_b64 = fake_b64

    captured_requests = []
    async def mock_complete(req):
        captured_requests.append(req)
        mock_resp = MagicMock()
        mock_resp.content = 'THOUGHT: test\nACTION: GUI_CLICK\nTARGET: 640,400\nPAYLOAD: {"x": 640, "y": 400}\nEXPECTED: click'
        return mock_resp

    mock_router = MagicMock()
    mock_router.complete = AsyncMock(side_effect=mock_complete)
    agent.llm_router = mock_router

    # Screen observation includes active window geometry
    obs = ComputerWorldObservation(
        screen=ScreenObservation(
            screenshot_base64=fake_b64,
            width=1280,
            height=800,
            active_window="CodeEditor",
            active_window_geometry=(100, 100, 600, 400),
        ),
    )
    await agent._llm_choose_action(
        goal="Edit file in editor",
        observation=obs,
        step_index=0,
        primary_file="",
        test_file="",
    )

    assert len(captured_requests) == 1
    user_msg = captured_requests[0].messages[1]
    # Expect 2 images: full desktop screenshot + foveated active window crop
    assert len(user_msg.images) == 2
    assert user_msg.images[0].base64 == fake_b64
    assert user_msg.images[1].base64 != fake_b64
