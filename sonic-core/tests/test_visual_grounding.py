"""Regression tests for strict, local-only workstation target grounding."""

import pytest

from sonic.computer_use.grounding import (
    draw_action_marker,
    extract_bbox_midpoint,
    resolve_ui_target,
    resolve_ui_target_async,
)


def test_extract_bbox_midpoint_uses_explicit_normalized_box():
    point = extract_bbox_midpoint(
        "<|box_start|>(100, 200, 300, 400)<|box_end|>",
        width=1280,
        height=800,
        coordinate_space="normalized_1000",
    )

    assert point == (256, 240)


def test_extract_bbox_midpoint_scales_normalized_point():
    assert extract_bbox_midpoint(
        (0.5, 0.5), width=1280, height=800, coordinate_space="normalized"
    ) == (640, 400)


def test_extract_bbox_midpoint_rejects_offscreen_point():
    assert extract_bbox_midpoint(
        (1280, 400), width=1280, height=800, coordinate_space="pixel"
    ) is None


def test_resolve_ui_target_accepts_explicit_screen_pixels():
    assert resolve_ui_target("800,600", width=1280, height=800) == (800, 600)


def test_resolve_ui_target_rejects_offscreen_pixels_without_clamping():
    assert resolve_ui_target("1280,600", width=1280, height=800) is None


def test_resolve_ui_target_does_not_guess_semantic_landmarks():
    assert resolve_ui_target("Applications menu", width=1280, height=800) is None
    assert resolve_ui_target("Terminal icon", width=1280, height=800) is None


def test_resolve_ui_target_does_not_trust_arbitrary_callback():
    point = resolve_ui_target(
        "custom control",
        screenshot_b64="frame",
        width=1280,
        height=800,
        grounding_fn=lambda *_: (640, 400),
    )

    assert point is None


@pytest.mark.asyncio
async def test_async_resolver_does_not_trust_arbitrary_callback():
    point = await resolve_ui_target_async(
        "custom control",
        screenshot_b64="frame",
        width=1280,
        height=800,
        grounding_fn=lambda *_: (640, 400),
    )

    assert point is None


def test_draw_action_marker_returns_annotated_image():
    image = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    marked = draw_action_marker(image, (0, 0), label="CLICK")

    assert marked
    assert marked != image
