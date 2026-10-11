import numpy as np
import pytest

from avcs.training.demo_recorder import (
    WindowInfo,
    build_click_record,
    crop_window_frame,
    find_exact_window_title,
    normalized_click,
)


def test_find_exact_window_title_rejects_partial_and_duplicate_matches():
    windows = [
        WindowInfo(1, "AVCS_DEMO_WINDOW", (10, 20, 110, 120)),
        WindowInfo(2, "AVCS_DEMO_WINDOW - Settings", (0, 0, 50, 50)),
    ]

    assert find_exact_window_title("AVCS_DEMO_WINDOW", windows).hwnd == 1
    with pytest.raises(RuntimeError, match="found 0"):
        find_exact_window_title("DEMO_WINDOW", windows)
    with pytest.raises(RuntimeError, match="found 2"):
        find_exact_window_title(
            "AVCS_DEMO_WINDOW",
            [windows[0], WindowInfo(3, "AVCS_DEMO_WINDOW", (0, 0, 20, 20))],
        )


def test_crop_window_frame_respects_virtual_screen_origin():
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    crop, screen_rect = crop_window_frame(frame, (-2, 11, 2, 14), (-3, 10))

    assert crop.shape == (3, 4, 3)
    np.testing.assert_array_equal(crop, frame[1:4, 1:5])
    assert screen_rect == (-2, 11, 2, 14)


def test_crop_window_frame_rejects_invalid_or_offscreen_frames():
    with pytest.raises(ValueError, match="RGB frame"):
        crop_window_frame(np.zeros((10, 10), dtype=np.uint8), (0, 0, 5, 5))
    with pytest.raises(ValueError, match="outside"):
        crop_window_frame(np.zeros((10, 10, 3), dtype=np.uint8), (20, 20, 30, 30))


def test_normalized_click_uses_exact_window_bounds_and_rejects_outside_points():
    rect = (100, 200, 300, 400)

    assert normalized_click((100, 200), rect) == (0.0, 0.0)
    assert normalized_click((200, 300), rect) == (0.5, 0.5)
    assert normalized_click((300, 300), rect) is None
    assert normalized_click((99, 300), rect) is None


def test_click_record_contains_no_captured_keystrokes_or_text():
    record = build_click_record(
        task="Open the test menu",
        target_label="File menu",
        frame_path="frames/sample.png",
        point_norm=(0.25, 0.75),
        recorded_at_utc="2026-10-11T00:00:00+00:00",
    )

    assert record["action"] == {
        "type": "mouse_click",
        "button": "left",
        "point_norm": [0.25, 0.75],
    }
    assert "typed_text" not in record
    assert "key" not in record["action"]
