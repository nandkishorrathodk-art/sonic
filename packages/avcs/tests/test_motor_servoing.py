"""Comprehensive Unit Tests for AVCS Physical OS Motor Driver & Hand-Eye Servoing."""
from __future__ import annotations

import math
import sys
import pytest

from forge.vision.cortex.motor import (
    VirtualCursorDriver,
    CornerBreakoutFailSafe,
    SafetyBreakoutException,
    HandEyeServoingController,
    get_default_driver,
)
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


def test_virtual_driver_lifecycle():
    """Verify virtual cursor driver mechanics, movement and click logging."""
    driver = VirtualCursorDriver(screen_size=(1920, 1080), initial_position=(0.5, 0.5), failsafe=False)

    assert driver.screen_size == (1920, 1080)
    assert driver.get_position_normalized() == (0.5, 0.5)

    px, py = driver.get_position_pixels()
    assert 950 <= px <= 970
    assert 530 <= py <= 550

    driver.move_to_normalized(0.2, 0.8)
    assert driver.get_position_normalized() == (0.2, 0.8)

    driver.click(button="left")
    assert len(driver.click_history) == 1
    assert driver.click_history[0] == (0.2, 0.8, "left")

    driver.double_click(button="left")
    assert len(driver.click_history) == 3


def test_corner_breakout_failsafe_all_corners():
    """Verify that moving into any of the 4 screen corners raises SafetyBreakoutException."""
    failsafe = CornerBreakoutFailSafe(corner_margin_px=15, enabled=True)
    w, h = 1920, 1080

    # Safe central coordinate
    failsafe.check(500, 500, w, h)

    # Top-Left corner (0, 0)
    with pytest.raises(SafetyBreakoutException) as exc1:
        failsafe.check(5, 5, w, h)
    assert "top-left" in str(exc1.value)

    # Top-Right corner (1915, 5)
    with pytest.raises(SafetyBreakoutException) as exc2:
        failsafe.check(1915, 5, w, h)
    assert "top-right" in str(exc2.value)

    # Bottom-Left corner (5, 1075)
    with pytest.raises(SafetyBreakoutException) as exc3:
        failsafe.check(5, 1075, w, h)
    assert "bottom-left" in str(exc3.value)

    # Bottom-Right corner (1915, 1075)
    with pytest.raises(SafetyBreakoutException) as exc4:
        failsafe.check(1915, 1075, w, h)
    assert "bottom-right" in str(exc4.value)


def test_hand_eye_servoing_convergence_and_teleport_prevention():
    """Verify closed-loop Fitts' Law convergence and zero coordinate jumping."""
    driver = VirtualCursorDriver(
        screen_size=(1920, 1080),
        initial_position=(0.1, 0.1),
        failsafe=False,
    )
    controller = HandEyeServoingController(
        driver=driver,
        tick_rate_hz=100,  # fast simulation rate
        foveal_contact_radius=0.025,
        add_human_jitter=True,
    )

    target = (0.75, 0.75)
    result = controller.servo_to(
        target_gaze=target,
        timeout_seconds=2.0,
        click_on_arrival=True,
        speed_factor=2.0,
        realtime=False,
    )

    assert result.contact_acquired is True
    fx, fy = result.final_position
    dist_error = math.hypot(target[0] - fx, target[1] - fy)
    assert dist_error <= 0.035, f"Final distance error {dist_error} exceeds basin tolerance"

    # Verify click on arrival was logged
    assert len(driver.click_history) == 1
    assert driver.click_history[0][2] == "left"

    # Verify zero coordinate teleportation across all steps
    assert len(result.trajectory) >= 5
    for i in range(1, len(result.trajectory)):
        p_prev = result.trajectory[i - 1]
        p_curr = result.trajectory[i]
        step_delta = math.hypot(p_curr.x - p_prev.x, p_curr.y - p_prev.y)
        # In a 10-16ms tick, cursor should never teleport more than 6% screen distance
        assert step_delta < 0.08, f"Detected coordinate jump artifact: delta={step_delta}"


def test_screen_eye_physical_servoing_integration():
    """Verify high-level ScreenEye.steer_hand_physical API."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    res = eye.steer_hand_physical(
        target_gaze=(0.4, 0.4),
        force_virtual=True,
        speed_factor=2.5,
        realtime=False,
    )
    assert res is not None
    assert res.contact_acquired is True
    assert len(res.trajectory) > 0


def test_win32_driver_read_only_on_windows():
    """Verify Win32CursorDriver reads valid physical metrics on Windows."""
    if sys.platform != "win32":
        pytest.skip("Test requires Microsoft Windows OS")

    from forge.vision.cortex.motor.win32 import Win32CursorDriver

    driver = Win32CursorDriver(failsafe=True)
    w, h = driver.screen_size
    assert w > 0
    assert h > 0

    px, py = driver.get_position_pixels()
    assert 0 <= px <= w
    assert 0 <= py <= h

    nx, ny = driver.get_position_normalized()
    assert 0.0 <= nx <= 1.0
    assert 0.0 <= ny <= 1.0
