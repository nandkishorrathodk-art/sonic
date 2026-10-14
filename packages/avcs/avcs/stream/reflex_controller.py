"""Continuous Real-Time Game & Dynamic Reflex Controller.

Provides low-latency (sub-16ms) continuous motor actuation, analog steering,
persistent key holds, and biological surprise-triggered reflexes for dynamic video games.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from forge.vision.cortex.motor import OSCursorDriver, get_default_driver
from forge.vision.cortex.sdk import ScreenEye
from avcs.stream.neuro_stream import NeuroPacket, NeuroStreamBus


class GameReflexController:
    """High-frequency continuous reflex actuator for real-time video games and fluid UI."""

    def __init__(
        self,
        eye: ScreenEye,
        bus: Optional[NeuroStreamBus] = None,
        driver: Optional[OSCursorDriver] = None,
        force_virtual: bool = False,
    ) -> None:
        self.eye = eye
        self.bus = bus
        self.driver = driver or get_default_driver(force_virtual=force_virtual)
        self.force_virtual = force_virtual
        self._held_keys: Set[str] = set()
        self._lock = threading.Lock()
        self._surprise_hooks: List[Tuple[float, Callable[[NeuroPacket], None]]] = []
        self._unsubscribe_bus: Optional[Callable[[], None]] = None

        if self.bus is not None:
            self._unsubscribe_bus = self.bus.subscribe(self._on_neuro_packet)

    def _on_neuro_packet(self, packet: NeuroPacket) -> None:
        """Evaluates live visual surprise spikes to trigger sub-millisecond reflexes."""
        for threshold, callback in list(self._surprise_hooks):
            if packet.surprise_score >= threshold:
                try:
                    callback(packet)
                except Exception:
                    pass

    def on_surprise_reflex(
        self,
        threshold: float = 0.35,
    ) -> Callable[[Callable[[NeuroPacket], None]], Callable[[NeuroPacket], None]]:
        """Decorator to register a hardware reflex callback on sudden visual changes."""
        def decorator(cb: Callable[[NeuroPacket], None]) -> Callable[[NeuroPacket], None]:
            with self._lock:
                self._surprise_hooks.append((threshold, cb))
            return cb
        return decorator

    def remove_surprise_hook(self, cb: Callable[[NeuroPacket], None]) -> None:
        """Removes a registered surprise callback."""
        with self._lock:
            self._surprise_hooks = [h for h in self._surprise_hooks if h[1] != cb]

    # --- Continuous Keyboard Actuation ---

    def hold_key(self, key: str) -> None:
        """Holds a key down continuously (e.g., 'w' to move forward, 'shift' to sprint)."""
        with self._lock:
            key_clean = key.lower().strip()
            if key_clean not in self._held_keys:
                if hasattr(self.driver, "key_down"):
                    self.driver.key_down(key_clean)
                elif hasattr(self.driver, "press_key"):
                    self.driver.press_key(key_clean)
                self._held_keys.add(key_clean)

    def release_key(self, key: str) -> None:
        """Releases a currently held key."""
        with self._lock:
            key_clean = key.lower().strip()
            if key_clean in self._held_keys:
                self._held_keys.remove(key_clean)
                if hasattr(self.driver, "key_up"):
                    self.driver.key_up(key_clean)

    def release_all_keys(self) -> None:
        """Safety release for all active key holds."""
        with self._lock:
            for key in list(self._held_keys):
                if hasattr(self.driver, "key_up"):
                    try:
                        self.driver.key_up(key)
                    except Exception:
                        pass
            self._held_keys.clear()

    def tap_key(self, key: str, duration: float = 0.03) -> None:
        """Presses and releases a key with a calibrated physical hold duration."""
        self.driver.press_key(key, hold_duration=duration)

    def hotkey(self, *keys: str) -> None:
        """Executes a key combination (e.g., 'ctrl', 'space')."""
        self.driver.press_hotkey(*keys)

    # --- Continuous Mouse Steering & Shooting ---

    def aim_towards(
        self,
        target_norm: Tuple[float, float],
        gain: float = 0.4,
        max_step: float = 0.05,
    ) -> Tuple[float, float]:
        """Performs a single proportional visual servoing aim adjustment towards target."""
        curr_x, curr_y = self.driver.get_position_normalized()
        tx, ty = target_norm

        # IEEE 754 NaN / Inf safety guard
        if not (math.isfinite(tx) and math.isfinite(ty)):
            return (curr_x, curr_y)

        dx = (tx - curr_x) * gain
        dy = (ty - curr_y) * gain

        # Direction-preserving Euclidean vector clamping
        dist = math.hypot(dx, dy)
        if dist > max_step and dist > 1e-9:
            scale = max_step / dist
            dx *= scale
            dy *= scale

        new_x = max(0.0, min(1.0, curr_x + dx))
        new_y = max(0.0, min(1.0, curr_y + dy))
        self.driver.move_to_normalized(new_x, new_y)
        return (new_x, new_y)

    def steer_relative(self, dx_pixels: int, dy_pixels: int) -> None:
        """Moves cursor by relative pixel offsets (e.g. FPS game camera rotation)."""
        self.driver.move_relative(dx_pixels, dy_pixels)

    def shoot(self, button: str = "left", hold_duration: float = 0.04) -> None:
        """Triggers a fast weapon fire / primary action click."""
        self.driver.click(button=button, hold_duration=hold_duration)

    def close(self) -> None:
        self.release_all_keys()
        if self._unsubscribe_bus:
            self._unsubscribe_bus()

    def __enter__(self) -> GameReflexController:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

