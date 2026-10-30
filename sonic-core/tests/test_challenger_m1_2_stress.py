"""
Empirical Challenger M1-2 Stress Test Suite
===========================================
Stress-testing coordinate resolution, bounding box extraction, and grounding
under adversarial edge cases, boundary values, and regression conditions.
"""

import asyncio
import pytest
from sonic.computer_use.grounding import (
    extract_bbox_midpoint,
    resolve_ui_target,
    resolve_ui_target_async,
)

pytestmark = pytest.mark.no_live_infra

WIDTH = 1280
HEIGHT = 800


# ==============================================================================
# 1. Asymmetric Coordinate Normalization (DEFECT-CI-05): List vs String Discrepancy
# ==============================================================================

def test_stress_list_vs_string_asymmetry():
    """Verify discrepancy between list [x, y] and string 'x, y' when coordinate_space is None."""
    list_result = extract_bbox_midpoint([500, 400], width=WIDTH, height=HEIGHT)
    str_result = extract_bbox_midpoint("500, 400", width=WIDTH, height=HEIGHT)
    bracket_str_result = extract_bbox_midpoint("[500, 400]", width=WIDTH, height=HEIGHT)
    tuple_result = extract_bbox_midpoint((500, 400), width=WIDTH, height=HEIGHT)

    # list and tuple preserve raw screen coordinates (500, 400)
    assert list_result == (500, 400), f"Expected (500, 400), got {list_result}"
    assert tuple_result == (500, 400), f"Expected (500, 400), got {tuple_result}"

    # string formats are normalized by default (500/1000*1280=640, 400/1000*800=320)
    assert str_result == (640, 320), f"Expected (640, 320), got {str_result}"
    assert bracket_str_result == (640, 320), f"Expected (640, 320), got {bracket_str_result}"

    # EMPIRICAL PROOF OF ASYMMETRY:
    # Identical numerical values yield different coordinates depending on input type!
    assert list_result != str_result, "Discrepancy confirmed: list yields pixel coords, string yields normalized coords"


def test_stress_4_element_bbox_list_vs_box_tag_asymmetry():
    """Check 4-element bounding boxes [x1, y1, x2, y2] vs <|box_start|> tags."""
    # Midpoint of (200, 300, 400, 500) is (300, 400)
    # If treated as pixel coords: (300, 400)
    # If treated as normalized 1000: (300/1000*1280=384, 400/1000*800=320)
    list_bbox = extract_bbox_midpoint([200, 300, 400, 500], width=WIDTH, height=HEIGHT)
    tag_bbox = extract_bbox_midpoint("<|box_start|>(200, 300, 400, 500)<|box_end|>", width=WIDTH, height=HEIGHT)

    # In current code:
    # list_bbox has max(400, 500) <= 1000 and 400 <= 1280 and 500 <= 800 -> pixel coords (300, 400)
    assert list_bbox == (300, 400)
    # tag_bbox has tagged_box=True -> normalized 1000 coords (384, 320)
    assert tag_bbox == (384, 320)
    assert list_bbox != tag_bbox


# ==============================================================================
# 2. Boundary Values and Small Integer Pixel Edge Cases
# ==============================================================================

def test_stress_zero_zero_handling():
    """Verify zero coordinates behavior."""
    assert extract_bbox_midpoint([0.0, 0.0], width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint([0, 0], width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint((0, 0), width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint("0, 0", width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint("<|box_start|>(0, 0, 0, 0)<|box_end|>", width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint({"x": 0, "y": 0}, width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint({"point": [0, 0]}, width=WIDTH, height=HEIGHT) is None


def test_stress_small_pixel_integers_less_than_or_equal_to_one():
    """Small pixel coordinates (1, 1), (0, 1), (1, 0) get misclassified as normalized floats [0.0, 1.0]."""
    # Because max(x, y) <= 1.0, [1, 1] is treated as float (1.0, 1.0) -> scaled to (1280, 800) -> out of bounds -> None!
    res_1_1 = extract_bbox_midpoint([1, 1], width=WIDTH, height=HEIGHT)
    assert res_1_1 is None, "Bug verified: [1, 1] misclassified as float (1.0, 1.0) and rejected"

    res_0_1 = extract_bbox_midpoint([0, 1], width=WIDTH, height=HEIGHT)
    assert res_0_1 is None, "Bug verified: [0, 1] misclassified as float (0.0, 1.0) -> (0, 800) -> rejected"

    res_1_0 = extract_bbox_midpoint([1, 0], width=WIDTH, height=HEIGHT)
    assert res_1_0 is None, "Bug verified: [1, 0] misclassified as float (1.0, 0.0) -> (1280, 0) -> rejected"

    # For coordinates >= 2, max(x, y) > 1.0, so it correctly treats them as pixels:
    res_2_2 = extract_bbox_midpoint([2, 2], width=WIDTH, height=HEIGHT)
    assert res_2_2 == (2, 2)


def test_stress_screen_edges():
    """Verify boundary coordinates at screen borders."""
    # Top-right edge
    assert extract_bbox_midpoint([WIDTH - 1, 0], width=WIDTH, height=HEIGHT) == (WIDTH - 1, 0)
    # Bottom-left edge
    assert extract_bbox_midpoint([0, HEIGHT - 1], width=WIDTH, height=HEIGHT) == (0, HEIGHT - 1)
    # Bottom-right edge
    assert extract_bbox_midpoint([WIDTH - 1, HEIGHT - 1], width=WIDTH, height=HEIGHT) == (WIDTH - 1, HEIGHT - 1)

    # Exactly at screen width/height (off-screen by 1 pixel)
    assert extract_bbox_midpoint([WIDTH, HEIGHT], width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint([WIDTH, 100], width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint([100, HEIGHT], width=WIDTH, height=HEIGHT) is None


def test_stress_normalized_boundary_one_and_thousand():
    """Normalized float 1.0 and normalized int 1000 scale to exact width/height, which _screen_point rejects!"""
    # [1.0, 1.0] scales to int(1.0 * 1280) = 1280, int(1.0 * 800) = 800 -> rejected because x < width, y < height!
    res_float_1 = extract_bbox_midpoint([1.0, 1.0], width=WIDTH, height=HEIGHT)
    assert res_float_1 is None, "Normalized 1.0 scales to width/height and gets rejected"

    # [1000, 1000] with is_normalized_1000=True scales to int(1000/1000*1280) = 1280, 800 -> rejected!
    res_norm_1000 = extract_bbox_midpoint([1000, 1000], width=WIDTH, height=HEIGHT, is_normalized_1000=True)
    assert res_norm_1000 is None, "Normalized 1000 scales to width/height and gets rejected"

    # [999, 999] scales to 1278, 799 -> accepted
    res_norm_999 = extract_bbox_midpoint([999, 999], width=WIDTH, height=HEIGHT, is_normalized_1000=True)
    assert res_norm_999 == (1278, 799)


def test_stress_negative_coordinates():
    """Verify negative coordinates are safely rejected."""
    assert extract_bbox_midpoint([-10, 400], width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint([400, -10], width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint([-50, -50], width=WIDTH, height=HEIGHT) is None
    assert extract_bbox_midpoint("-10, 400", width=WIDTH, height=HEIGHT) is None


# ==============================================================================
# 3. resolve_ui_target Sync vs Async and Screenshot Attachment
# ==============================================================================

def test_resolve_ui_target_sync_with_and_without_screenshot():
    """Worker M1 fixed resolve_ui_target sync by removing the screenshot_b64 check."""
    fake_screenshot = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Without screenshot
    res_no_shot = resolve_ui_target("640, 400", screenshot_b64=None, width=WIDTH, height=HEIGHT)
    assert res_no_shot == (640, 400)

    # With screenshot attached (DEFECT-CI-10 fix check)
    res_with_shot = resolve_ui_target("640, 400", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT)
    assert res_with_shot == (640, 400), "Worker M1 fix in resolve_ui_target succeeds"

    # Edge screen target (0, 0)
    assert resolve_ui_target("0, 0", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT) == (0, 0)
    # Edge screen target (1279, 799)
    assert resolve_ui_target("1279, 799", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT) == (1279, 799)

    # With normalized_1000 space
    res_norm = resolve_ui_target("500, 500", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT, coordinate_space="normalized_1000")
    assert res_norm == (640, 400)


@pytest.mark.asyncio
async def test_resolve_ui_target_async_screenshot_bug():
    """DISCOVERY: Worker M1 fixed sync resolve_ui_target but FORGOT resolve_ui_target_async!"""
    fake_screenshot = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Without screenshot -> works
    res_no_shot = await resolve_ui_target_async("640, 400", screenshot_b64=None, width=WIDTH, height=HEIGHT)
    assert res_no_shot == (640, 400)

    # With screenshot -> now SUCCEEDS in resolve_ui_target_async because blocker is removed:
    res_with_shot = await resolve_ui_target_async("640, 400", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT)
    assert res_with_shot == (640, 400), "resolve_ui_target_async successfully resolves direct coords with screenshot!"


# ==============================================================================
# 4. Complex Bounding Boxes and Multimodal Grounding Callbacks
# ==============================================================================

def test_resolve_ui_target_with_grounding_callbacks():
    """Test resolution with various vision model output types via grounding_fn."""
    fake_screenshot = "dummy_base64"

    # Callback returns tagged box
    def mock_tag_fn(query, shot, w, h):
        return "<|box_start|>(100, 200, 300, 400)<|box_end|>"
    res_tag = resolve_ui_target("submit button", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT, grounding_fn=mock_tag_fn)
    # Midpoint of 100, 200, 300, 400 is 200, 300 -> 200/1000*1280=256, 300/1000*800=240
    assert res_tag == (256, 240)

    # Callback returns JSON string (valid JSON with double quotes)
    def mock_json_str_fn(query, shot, w, h):
        return '{"x": 500, "y": 400}'
    res_json = resolve_ui_target("search bar", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT, grounding_fn=mock_json_str_fn)
    assert res_json == (500, 400), f"Expected (500, 400) for JSON string, got {res_json}"

    # Callback returns Python dict object
    # BUG: extract_bbox_midpoint has no dict branch; it stringifies the dict to "{'x': 500, 'y': 400}",
    # regex extracts numbers, and string normalization scales 500, 400 to (640, 320)!
    def mock_dict_fn(query, shot, w, h):
        return {"x": 500, "y": 400}
    res_dict = resolve_ui_target("search bar", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT, grounding_fn=mock_dict_fn)
    assert res_dict == (640, 320), f"Dict object stringified and mis-scaled to (640, 320), got {res_dict}"

    # Callback returns 'NOT_FOUND'
    def mock_not_found(query, shot, w, h):
        return "NOT_FOUND"
    assert resolve_ui_target("missing icon", screenshot_b64=fake_screenshot, width=WIDTH, height=HEIGHT, grounding_fn=mock_not_found) is None
