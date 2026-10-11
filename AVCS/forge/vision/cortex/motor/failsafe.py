"""Emergency Safety Fail-Safe for AVCS Motor Control.

Protects the user by instantly relinquishing cursor control whenever the
mouse enters a designated emergency breakout corner or whenever a manual
override is detected.
"""
from __future__ import annotations

from typing import Tuple


class SafetyBreakoutException(RuntimeError):
    """Raised immediately when emergency fail-safe is triggered by the user."""


class CornerBreakoutFailSafe:
    """Detects emergency cursor breakout in any of the 4 screen corners."""

    def __init__(self, corner_margin_px: int = 15, enabled: bool = True) -> None:
        self.corner_margin_px = corner_margin_px
        self.enabled = enabled
        self._tripped = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        """Resets the safety fail-safe latch, allowing motor control to resume."""
        self._tripped = False

    def check(self, pixel_x: int, pixel_y: int, screen_width: int, screen_height: int) -> None:
        """Checks if the cursor is in an emergency breakout corner.

        Raises SafetyBreakoutException if triggered or already latched.
        """
        if not self.enabled:
            return

        margin = self.corner_margin_px

        top_left = pixel_x <= margin and pixel_y <= margin
        top_right = pixel_x >= (screen_width - margin) and pixel_y <= margin
        bottom_left = pixel_x <= margin and pixel_y >= (screen_height - margin)
        bottom_right = pixel_x >= (screen_width - margin) and pixel_y >= (screen_height - margin)

        if top_left or top_right or bottom_left or bottom_right:
            self._tripped = True
            corner_name = (
                "top-left" if top_left else
                "top-right" if top_right else
                "bottom-left" if bottom_left else
                "bottom-right"
            )
            raise SafetyBreakoutException(
                f"[AVCS Safety Fail-Safe] Emergency abort triggered: cursor entered {corner_name} "
                f"corner ({pixel_x}, {pixel_y}). Motor control relinquished to user."
            )

        if self._tripped:
            raise SafetyBreakoutException(
                "[AVCS Safety Fail-Safe] Motor control is latched in emergency abort state. "
                "Call failsafe.reset() to re-enable motor operations."
            )

    def check_override(
        self,
        actual_pos: Tuple[float, float],
        expected_pos: Tuple[float, float],
        threshold_norm: float = 0.08,
    ) -> None:
        """Detects physical human resistance (user snatching mouse during servoing)."""
        if not self.enabled:
            return
        dx = actual_pos[0] - expected_pos[0]
        dy = actual_pos[1] - expected_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > threshold_norm:
            self._tripped = True
            raise SafetyBreakoutException(
                f"[AVCS Safety Fail-Safe] Manual override detected: cursor diverted by {dist:.3f} "
                f"from planned trajectory. Control relinquished to user."
            )
        if self._tripped:
            raise SafetyBreakoutException(
                "[AVCS Safety Fail-Safe] Motor control is latched in emergency abort state. "
                "Call failsafe.reset() to re-enable motor operations."
            )

