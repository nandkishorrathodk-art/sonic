"""
Tests for AVCS-local perception and coordinate normalization in sonic-core:
1. GUI reasoning requires a fresh AVCS pixel frame but does not upload it.
2. extract_bbox_midpoint scales normalized 0-1000 coordinates to
   (raw_x / 1000.0) * width and (raw_y / 1000.0) * height in 1280x800 screens.
3. ScreenObservation supports active_window_geometry (x, y, w, h) in various formats.
"""

import base64
import io
import sys
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from PIL import Image

from sonic.computer.models import ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use import grounding
from sonic.computer_use.grounding import extract_bbox_midpoint, resolve_ui_target
from sonic.computer_use.models import ActionExecutionStatus, ComputerWorldObservation
from tests.avcs_test_helpers import png_base64


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
# 1. AVCS frame requirements and local-only GUI perception
# ==============================================================================

@pytest.mark.asyncio
async def test_gui_only_worker_fails_closed_when_fresh_pixels_are_unavailable():
    provider = _mock_provider()
    agent = ComputerUseAgent(computer_provider=provider, execution_plane="gui")
    agent._active_workspace_id = "ws-1"
    provider.screenshot.return_value = ScreenObservation(
        width=1280,
        height=800,
        capture_age_ms=0,
    )
    router = MagicMock()
    router.complete = AsyncMock()
    agent.llm_router = router
    obs = ComputerWorldObservation(
        screen=ScreenObservation(
            width=1280,
            height=800,
            active_window="Document",
            capture_age_ms=0,
            detected_controls=[
                {
                    "name": "Save",
                    "role": "button",
                    "bbox": (30, 40, 90, 70),
                    "source": "atspi",
                    "confidence": 0.95,
                    "freshness_ms": 10,
                }
            ],
        ),
        active_application="Document",
        perception_version=1,
        observation_age_ms=25,
    )

    action, target, payload, thought = await agent._llm_choose_action(
        goal="Save the document",
        observation=obs,
        step_index=1,
        primary_file="",
        test_file="",
    )

    assert action == ActionExecutionStatus.BLOCKED
    assert target == "reasoning-unavailable"
    assert "no GUI reasoning request was sent" in thought
    router.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_reasoning_attaches_fresh_frame_to_computer_use_route():
    frame = _make_test_image_b64()
    agent = ComputerUseAgent(computer_provider=_mock_provider(), execution_plane="gui")
    agent.llm_router = MagicMock()
    agent.llm_router.complete = AsyncMock(
        return_value=SimpleNamespace(
            content=(
                '{"action":"GOAL_COMPLETE","target":"goal_complete",'
                '"payload":{},"expected":"Hand off for independent verification"}'
            ),
            tool_calls=[],
        )
    )
    screen = ScreenObservation(
        screenshot_base64=frame,
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 1,
            "scene_hash": "document-scene",
            "width": 1280,
            "height": 800,
        },
        width=1280,
        height=800,
        active_window="Document",
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
        perception_version=1,
    )
    observation = ComputerWorldObservation(
        screen=screen,
        active_application="Document",
        perception_version=1,
        observation_age_ms=0,
    )

    await agent._llm_choose_action(
        goal="Save the document",
        observation=observation,
        step_index=1,
        primary_file="",
        test_file="",
    )

    agent.llm_router.complete.assert_awaited_once()
    request = agent.llm_router.complete.await_args.args[0]
    user_content = request.messages[1].content
    assert isinstance(user_content, str)
    assert request.task_type == "computer_use"
    assert len(request.messages[1].images) == 1
    assert request.messages[1].images[0].base64 == frame
    assert request.messages[1].images[0].media_type == "image/png"
    assert observation.image_attached is True
    assert "ACCESSIBILITY CONTROLS" not in user_content


def test_gui_screenshot_content_encodes_raw_rgb_pixels_as_png():
    import base64
    from io import BytesIO

    from PIL import Image

    rgb = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255))
    screen = ScreenObservation(
        raw_rgb=rgb,
        screenshot_base64="",
        frame_id=1,
        width=2,
        height=2,
    )

    image = ComputerUseAgent._screen_image_content(screen)

    assert image is not None
    assert image.media_type == "image/png"
    decoded = base64.b64decode(image.base64, validate=True)
    with Image.open(BytesIO(decoded)) as png:
        assert png.mode == "RGB"
        assert png.size == (2, 2)
        assert png.tobytes() == rgb


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


def test_extract_bbox_midpoint_normalized_scaling_1920x1080():
    coords = extract_bbox_midpoint(
        '{"x": 500, "y": 500}',
        width=1920,
        height=1080,
        coordinate_space="normalized_1000",
    )
    assert coords == (960, 540)


def test_semantic_target_uses_only_ready_evaluated_local_avcs(monkeypatch):
    checkpoint = "evaluated-checkpoint.pt"
    monkeypatch.setenv("SONIC_AVCS_GROUNDING_CHECKPOINT", checkpoint)
    grounding._AVCS_PERCEPTION_CACHE.pop(checkpoint, None)

    class FakeAVCSPerception:
        goal_grounder_configured = True
        goal_grounder_ready = True

        def ground_target_png(
            self, image_bytes, query, width, height, *, window_geometry=None
        ):
            assert image_bytes == base64.b64decode(png_base64())
            assert query == "Save button"
            assert (width, height) == (1280, 800)
            return {
                "engine": "avcs_goal_grounder",
                "x": 400,
                "y": 300,
                "held_out_point_accuracy": 0.9,
                "query_score": 0.95,
                "query_score_threshold": 0.8,
                "checkpoint_metadata": {
                    "evaluation": {
                        "metric": "top1_point_accuracy_within_0.02_diagonal",
                        "passed": True,
                        "value": 0.9,
                        "samples": 20,
                    },
                    "query_confidence_calibration": {
                        "method": "held_out_empirical_precision",
                        "evaluation": {
                            "metric": "empirical_precision_at_threshold",
                            "samples": 20,
                            "threshold": 0.8,
                            "precision": 0.95,
                            "required_precision": 0.9,
                            "passed": True,
                        },
                    },
                },
            }

    monkeypatch.setitem(
        sys.modules,
        "sonic.computer_use.avcs_perception",
        SimpleNamespace(AVCSPerception=FakeAVCSPerception),
    )
    monkeypatch.setitem(
        sys.modules,
        "avcs.goal_grounder",
        SimpleNamespace(
            GROUNDING_ACCEPTANCE_METRIC="top1_point_accuracy_within_0.02_diagonal",
            GROUNDING_ACCEPTANCE_THRESHOLD=0.85,
        ),
    )

    try:
        assert resolve_ui_target(
            "Save button",
            screenshot_b64=png_base64(),
            width=1280,
            height=800,
        ) == (400, 300)
    finally:
        grounding._AVCS_PERCEPTION_CACHE.pop(checkpoint, None)


def test_semantic_target_fails_closed_without_evaluated_avcs(monkeypatch):
    monkeypatch.delenv("SONIC_AVCS_GROUNDING_CHECKPOINT", raising=False)
    callback = MagicMock(return_value=(400, 300))
    bus = MagicMock()
    bus.resolve_coordinates.return_value = (400, 300)

    assert resolve_ui_target(
        "Save button",
        screenshot_b64="aW1hZ2U=",
        grounding_fn=callback,
        perception_bus=bus,
    ) is None
    assert resolve_ui_target(
        "Save button",
        grounding_fn=callback,
        perception_bus=bus,
    ) is None
    callback.assert_not_called()
    bus.resolve_coordinates.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        {
            "engine": "avcs_goal_grounder",
            "x": 400,
            "y": 300,
            "held_out_point_accuracy": 0.9,
            "ambiguous": True,
            "checkpoint_metadata": {},
        },
        {
            "engine": "avcs_goal_grounder",
            "x": 1280,
            "y": 300,
            "held_out_point_accuracy": 0.9,
            "checkpoint_metadata": {
                "evaluation": {
                    "metric": "top1_point_accuracy_within_0.02_diagonal",
                    "passed": True,
                    "value": 0.9,
                    "samples": 20,
                }
            },
        },
        {
            "engine": "avcs_goal_grounder",
            "x": 400,
            "y": 300,
            "held_out_point_accuracy": 0.7,
            "checkpoint_metadata": {
                "evaluation": {
                    "metric": "top1_point_accuracy_within_0.02_diagonal",
                    "passed": False,
                    "value": 0.7,
                    "samples": 20,
                }
            },
        },
    ],
)
def test_semantic_target_rejects_ambiguous_offscreen_or_unevaluated_results(
    monkeypatch, result
):
    checkpoint = "invalid-checkpoint.pt"
    monkeypatch.setenv("SONIC_AVCS_GROUNDING_CHECKPOINT", checkpoint)
    grounding._AVCS_PERCEPTION_CACHE.pop(checkpoint, None)

    class FakeAVCSPerception:
        goal_grounder_configured = True
        goal_grounder_ready = True

        def ground_target_png(
            self, image_bytes, query, width, height, *, window_geometry=None
        ):
            return result

    monkeypatch.setitem(
        sys.modules,
        "sonic.computer_use.avcs_perception",
        SimpleNamespace(AVCSPerception=FakeAVCSPerception),
    )
    monkeypatch.setitem(
        sys.modules,
        "avcs.goal_grounder",
        SimpleNamespace(
            GROUNDING_ACCEPTANCE_METRIC="top1_point_accuracy_within_0.02_diagonal",
            GROUNDING_ACCEPTANCE_THRESHOLD=0.85,
        ),
    )

    try:
        assert resolve_ui_target(
            "Save button",
            screenshot_b64="aW1hZ2U=",
            width=1280,
            height=800,
        ) is None
    finally:
        grounding._AVCS_PERCEPTION_CACHE.pop(checkpoint, None)


def test_explicit_coordinates_keep_coordinate_space_and_screen_bounds():
    assert resolve_ui_target("500, 400", width=1280, height=800) == (500, 400)
    assert resolve_ui_target(
        "500, 400",
        width=1280,
        height=800,
        coordinate_space="normalized_1000",
    ) == (640, 320)
    assert resolve_ui_target("1280, 400", width=1280, height=800) is None
    assert resolve_ui_target("0, 0", width=1280, height=800) == (0, 0)


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
# ==============================================================================
