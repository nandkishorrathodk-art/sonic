"""Unified Cross-Platform OS Cursor Driver Interface.

Provides a polymorphic driver abstraction supporting native Windows Win32,
cross-platform fallbacks, and headless Virtual drivers.
"""
from __future__ import annotations

import sys
import time
from typing import List, Optional, Protocol, Set, Tuple

from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe, SafetyBreakoutException


class OSCursorDriver(Protocol):
    """Abstract protocol for OS cursor drivers."""

    @property
    def screen_size(self) -> Tuple[int, int]:
        ...

    def get_position_pixels(self) -> Tuple[int, int]:
        ...

    def get_position_normalized(self) -> Tuple[float, float]:
        ...

    def move_to_pixels(self, x: int, y: int) -> None:
        ...

    def move_to_normalized(self, nx: float, ny: float) -> None:
        ...

    def move_relative(self, dx: int, dy: int) -> None:
        ...

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        ...

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        ...

    def type_text(
        self,
        text: str,
        key_delay_ms: int = 10,
        clear_first: bool = False,
        press_enter: bool = False,
    ) -> None:
        ...

    def press_key(self, key: str, hold_duration: float = 0.02) -> None:
        ...

    def key_down(self, key: str) -> None:
        ...

    def key_up(self, key: str) -> None:
        ...

    def press_hotkey(self, *keys: str) -> None:
        ...


class VirtualCursorDriver:
    """In-memory simulated cursor driver for headless testing and CI."""

    def __init__(
        self,
        screen_size: Tuple[int, int] = (1920, 1080),
        initial_position: Tuple[float, float] = (0.5, 0.5),
        failsafe: bool = True,
    ) -> None:
        self._screen_size = screen_size
        self._norm_x, self._norm_y = initial_position
        self.failsafe = CornerBreakoutFailSafe(enabled=failsafe)
        self.click_history: List[Tuple[float, float, str]] = []
        self.trajectory_history: List[Tuple[float, float]] = [(self._norm_x, self._norm_y)]
        self.typed_history: List[str] = []
        self.key_history: List[str] = []
        self.held_keys: Set[str] = set()

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._screen_size

    def _check_failsafe(self) -> None:
        px, py = self.get_position_pixels()
        w, h = self._screen_size
        self.failsafe.check(px, py, w, h)

    def get_position_pixels(self) -> Tuple[int, int]:
        w, h = self._screen_size
        return (int(round(self._norm_x * (w - 1))), int(round(self._norm_y * (h - 1))))

    def get_position_normalized(self) -> Tuple[float, float]:
        return (self._norm_x, self._norm_y)

    def move_to_pixels(self, x: int, y: int) -> None:
        w, h = self._screen_size
        clamped_x = max(0, min(w - 1, x))
        clamped_y = max(0, min(h - 1, y))

        self.failsafe.check(clamped_x, clamped_y, w, h)
        self._norm_x = clamped_x / max(1, w - 1)
        self._norm_y = clamped_y / max(1, h - 1)
        self.trajectory_history.append((self._norm_x, self._norm_y))

    def move_to_normalized(self, nx: float, ny: float) -> None:
        w, h = self._screen_size
        clamped_nx = max(0.0, min(1.0, nx))
        clamped_ny = max(0.0, min(1.0, ny))
        px = int(round(clamped_nx * (w - 1)))
        py = int(round(clamped_ny * (h - 1)))
        self.failsafe.check(px, py, w, h)
        self._norm_x = clamped_nx
        self._norm_y = clamped_ny
        self.trajectory_history.append((self._norm_x, self._norm_y))

    def move_relative(self, dx: int, dy: int) -> None:
        px, py = self.get_position_pixels()
        self.move_to_pixels(px + dx, py + dy)

    def mouse_down(self, button: str = "left") -> None:
        self._check_failsafe()
        self.click_history.append((self._norm_x, self._norm_y, f"{button}_down"))

    def mouse_up(self, button: str = "left") -> None:
        self.click_history.append((self._norm_x, self._norm_y, f"{button}_up"))

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        self._check_failsafe()
        self.click_history.append((self._norm_x, self._norm_y, button))

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        self.click(button=button)
        self.click(button=button)

    def type_text(
        self,
        text: str,
        key_delay_ms: int = 10,
        clear_first: bool = False,
        press_enter: bool = False,
    ) -> None:
        self._check_failsafe()
        if clear_first:
            self.key_history.append("ctrl+a")
            self.key_history.append("backspace")
        self.typed_history.append(text)
        if press_enter:
            self.key_history.append("enter")

    def key_down(self, key: str) -> None:
        self._check_failsafe()
        k = str(key).lower().strip()
        self.held_keys.add(k)
        self.key_history.append(f"{k}_down")

    def key_up(self, key: str) -> None:
        k = str(key).lower().strip()
        self.held_keys.discard(k)
        self.key_history.append(f"{k}_up")

    def press_key(self, key: str, hold_duration: float = 0.02) -> None:
        self._check_failsafe()
        self.key_history.append(key)

    def press_hotkey(self, *keys: str) -> None:
        self._check_failsafe()
        self.key_history.append("+".join(str(k) for k in keys))

    def panic_release(self) -> None:
        self.held_keys.clear()

    def close(self) -> None:
        self.panic_release()



def get_default_driver(force_virtual: bool = False, failsafe: bool = True) -> OSCursorDriver:
    """Factory: selects optimal hardware driver for current environment."""
    if not force_virtual and sys.platform == "win32":
        try:
            from forge.vision.cortex.motor.win32 import Win32CursorDriver
            return Win32CursorDriver(failsafe=failsafe)
        except Exception:
            pass

    return VirtualCursorDriver(failsafe=failsafe)
