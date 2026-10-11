"""Unit tests for AVCS Keyboard Motor Driver and ScreenEye typing methods."""
from __future__ import annotations

import sys
import pytest

from forge.vision.cortex.motor import (
    VirtualCursorDriver,
    CornerBreakoutFailSafe,
    SafetyBreakoutException,
    get_default_driver,
)
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


def test_virtual_keyboard_typing_and_history():
    """Verify virtual keyboard driver records typed text and special keys."""
    driver = VirtualCursorDriver(failsafe=False)

    driver.type_text("Hello AVCS!", clear_first=True, press_enter=True)
    assert driver.typed_history == ["Hello AVCS!"]
    assert "ctrl+a" in driver.key_history
    assert "backspace" in driver.key_history
    assert "enter" in driver.key_history

    driver.press_key("tab")
    assert "tab" in driver.key_history

    driver.press_hotkey("ctrl", "shift", "p")
    assert "ctrl+shift+p" in driver.key_history


def test_screeneye_keyboard_helpers_virtual():
    """Verify ScreenEye high-level typing, clicking, and keyboard shortcuts."""
    eye = ScreenEye(source=SyntheticScreen(width=320, height=240), device="cpu")

    eye.type_text("search query", clear_first=True, press_enter=True, force_virtual=True)
    eye.press_key("escape", force_virtual=True)
    eye.press_hotkey("ctrl", "c", force_virtual=True)

    # Test type_into with virtual driver
    ok = eye.type_into("search bar", text="test typing", force_virtual=True)
    assert ok is True

    eye.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 keyboard driver requires Windows OS")
def test_win32_keyboard_driver_structures_and_key_resolution():
    """Verify Win32CursorDriver key mapping resolution and internal structure setup."""
    from forge.vision.cortex.motor.win32 import (
        Win32CursorDriver,
        VK_RETURN,
        VK_TAB,
        VK_BACK,
        KEY_NAME_TO_VK,
    )

    driver = Win32CursorDriver(failsafe=False)
    assert driver._resolve_vk("enter") == VK_RETURN
    assert driver._resolve_vk("tab") == VK_TAB
    assert driver._resolve_vk("backspace") == VK_BACK
    assert driver._resolve_vk("a") == ord("A")
    assert driver._resolve_vk("f5") == 0x70 + 4  # VK_F5
    assert driver._resolve_vk(123) == 123

    # Test punctuation symbol fallback resolution (e.g. '-' or '.')
    dot_vk = driver._resolve_vk(".")
    assert dot_vk > 0

    with pytest.raises(ValueError):
        driver._resolve_vk("unknown_invalid_key_xyz")

    driver.close()


def test_failsafe_latching_and_reset():
    """Verify CornerBreakoutFailSafe latches on breakout and unlatches on reset()."""
    failsafe = CornerBreakoutFailSafe(corner_margin_px=15, enabled=True)
    assert failsafe.is_tripped is False

    # Breakout into top-left corner
    with pytest.raises(SafetyBreakoutException):
        failsafe.check(5, 5, 1920, 1080)

    assert failsafe.is_tripped is True

    # Subsequent check even in safe area must remain latched
    with pytest.raises(SafetyBreakoutException):
        failsafe.check(500, 500, 1920, 1080)

    # Reset allows safe operation again
    failsafe.reset()
    assert failsafe.is_tripped is False
    failsafe.check(500, 500, 1920, 1080)  # Should not raise

    # Test check_override
    with pytest.raises(SafetyBreakoutException):
        failsafe.check_override((0.5, 0.5), (0.7, 0.7), threshold_norm=0.08)
    assert failsafe.is_tripped is True

