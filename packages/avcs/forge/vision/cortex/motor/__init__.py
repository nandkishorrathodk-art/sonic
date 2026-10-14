"""AVCS Physical OS Motor Driver Subsystem.

Provides closed-loop Hand-Eye coordination, hardware cursor drivers,
and safety fail-safe mechanisms for desktop automation.
"""
from __future__ import annotations

from forge.vision.cortex.motor.driver import (
    OSCursorDriver,
    VirtualCursorDriver,
    get_default_driver,
)
from forge.vision.cortex.motor.failsafe import (
    CornerBreakoutFailSafe,
    SafetyBreakoutException,
)
from forge.vision.cortex.motor.servoing_motor import (
    HandEyeServoingController,
    ServoingResult,
    ServoingTrajectoryPoint,
)

try:
    from forge.vision.cortex.motor.win32 import Win32CursorDriver
except ImportError:
    Win32CursorDriver = None  # type: ignore

__all__ = [
    "OSCursorDriver",
    "VirtualCursorDriver",
    "Win32CursorDriver",
    "get_default_driver",
    "CornerBreakoutFailSafe",
    "SafetyBreakoutException",
    "HandEyeServoingController",
    "ServoingResult",
    "ServoingTrajectoryPoint",
]

