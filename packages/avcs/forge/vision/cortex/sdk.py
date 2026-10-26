"""AVCS Developer SDK (ScreenEye).

Provides a clean, intuitive, 4-line developer experience for AI agent builders,
RPA engineers, and computer-use automation systems.
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from forge.vision.capture import open_capture
from forge.vision.cortex.inference.engine import AVCSInferenceEngine
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState


class ScreenEye:
    """The developer-facing Sensory Vision SDK for Computer-Use AI Agents.

    Usage:
    ------
    >>> from forge.vision.cortex import ScreenEye
    >>> eye = ScreenEye(fps=25)
    >>> eye.start()
    >>>
    >>> # 1. Check if screen is stable or still loading
    >>> if eye.wait_until_stable():
    ...     # 2. Get coordinates for clickable elements
    ...     buttons = eye.clickable_hotspots()
    ...     # 3. Locate target without sending screenshots to cloud LLM
    ...     x, y = eye.locate("Search bar")
    """

    def __init__(
        self,
        fps: int = 25,
        device: Optional[str] = None,
        source: Optional[object] = None,
        weights_path: Optional[str] = None,
        load_pretrained: bool = True,
    ) -> None:
        self.fps = fps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.source = source or open_capture()
        self.cortex = ArtificialVisualCortex().to(self.device).eval()

        self.weights_loaded: bool = False
        if load_pretrained:
            from avcs.weights import load_avcs_weights
            self.weights_loaded = load_avcs_weights(
                self.cortex,
                path=weights_path,
                device=self.device,
                strict=False,
            )

        self._change_callbacks: List[Callable[[CortexState], None]] = []
        self._surprise_callbacks: List[Tuple[float, Callable[[CortexState], None]]] = []
        self._affordance_callbacks: List[Tuple[str, float, Callable[[CortexState], None]]] = []
        self._drivers: dict[bool, Any] = {}

        self.engine = AVCSInferenceEngine(
            cortex=self.cortex,
            source=self.source,
            device=self.device,
            fps_target=fps,
            on_state=self._dispatch_state,
            on_surprise=self._dispatch_surprise,
        )

    def _get_driver(self, force_virtual: bool = False) -> Any:
        if force_virtual not in self._drivers:
            from forge.vision.cortex.motor import get_default_driver
            self._drivers[force_virtual] = get_default_driver(force_virtual=force_virtual)
        return self._drivers[force_virtual]

    def start(self) -> ScreenEye:
        """Starts real-time background perception stream."""
        self.engine.start()
        return self

    def stop(self) -> None:
        """Stops background perception stream."""
        self.engine.stop()

    def perceive_now(self, frame: Optional[np.ndarray] = None) -> CortexState:
        """Forces an instantaneous perception pass and returns current state."""
        return self.engine.step(frame=frame)

    @property
    def current_state(self) -> Optional[CortexState]:
        """Returns the most recent perceived screen state."""
        return self.engine.latest_state

    def is_stable(self) -> bool:
        """Returns True if the screen is settled and not animating or transitioning."""
        state = self.current_state or self.perceive_now()
        return state.stability_state == "stable"

    def wait_until_stable(self, timeout: float = 4.0, poll_interval: float = 0.05) -> bool:
        """Blocks until animations/page loads finish, or until timeout expires."""
        t_start = time.time()
        while time.time() - t_start < timeout:
            state = self.perceive_now()
            if state.stability_state == "stable" and state.surprise_score < 0.2:
                return True
            time.sleep(poll_interval)
        return False

    def look_at(self, query: str, frame: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Directs the foveal gaze to lock onto the target element.

        The query is encoded by the same real character n-gram encoder the
        training pipeline uses, so the gaze is steered by the actual words of the
        query. (Previously this seeded ``torch.randn`` from a SHA-256 hash of the
        query, producing a random vector unrelated to the text -- the grounding
        was driven by noise.)
        """
        from forge.vision.cortex.core.modulation import encode_text_query

        text_emb = encode_text_query(
            query, out_dim=self.cortex.cfg.latent_dim, device=self.device
        )

        with self.engine._lock:
            if frame is None:
                frame = self.source.grab()
            state = self.cortex.perceive(frame, text_embedding=text_emb, device=self.device)
            target = state.query_target or state.fixation
            # Update retinal gaze fixation
            self.cortex.retina.update_fixation(target_fixation=target)
            return target

    def locate(self, query: str, frame: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Alias for look_at: returns normalized (x, y) target coordinates."""
        return self.look_at(query, frame=frame)

    def steer_hand(
        self,
        current_cursor: Tuple[float, float],
        target_gaze: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[float, float], bool]:
        """Biological Hand-Eye coordination step.

        Returns:
            (motor_velocity: (vx, vy), contact_acquired: bool)
        Steers the hand toward the gaze point without calculating absolute pixel jumps.
        """
        gaze = target_gaze or self.cortex.retina.current_fixation
        motor_state = self.cortex.servoing.compute_motor_flow(
            gaze_target=gaze,
            current_cursor=current_cursor,
        )
        return motor_state.motor_velocity, motor_state.contact_acquired

    def is_aligned(self, current_cursor: Tuple[float, float], target_gaze: Optional[Tuple[float, float]] = None) -> bool:
        """Returns True if the cursor is locked onto the foveal target basin."""
        _, aligned = self.steer_hand(current_cursor, target_gaze=target_gaze)
        return aligned

    def steer_hand_physical(
        self,
        target_gaze: Optional[Tuple[float, float]] = None,
        click_on_arrival: bool = False,
        speed_factor: float = 1.0,
        timeout_seconds: float = 1.5,
        force_virtual: bool = False,
        realtime: bool = True,
    ) -> Any:
        """Physically steers the operating system mouse cursor to the gaze target.

        Uses closed-loop Fitts' Law velocity curves with zero coordinate jumping.
        Features emergency corner breakout safety abort.
        """
        from forge.vision.cortex.motor import HandEyeServoingController, get_default_driver
        driver = get_default_driver(force_virtual=force_virtual)
        controller = HandEyeServoingController(driver=driver)
        gaze = target_gaze or self.cortex.retina.current_fixation
        return controller.servo_to(
            target_gaze=gaze,
            timeout_seconds=timeout_seconds,
            click_on_arrival=click_on_arrival,
            speed_factor=speed_factor,
            realtime=realtime,
        )

    def click_target(
        self,
        query: str,
        button: str = "left",
        timeout_seconds: float = 1.5,
        force_virtual: bool = False,
        realtime: bool = False,
    ) -> bool:
        """Locates visual target by query and physically steers cursor to click it."""
        target = self.locate(query)
        res = self.steer_hand_physical(
            target_gaze=target,
            click_on_arrival=True,
            timeout_seconds=timeout_seconds,
            force_virtual=force_virtual,
            realtime=realtime,
        )
        return bool(res.contact_acquired)

    def type_text(
        self,
        text: str,
        key_delay_ms: int = 10,
        clear_first: bool = False,
        press_enter: bool = False,
        force_virtual: bool = False,
    ) -> None:
        """Types text via the OS hardware or virtual keyboard driver."""
        driver = self._get_driver(force_virtual=force_virtual)
        driver.type_text(
            text=text,
            key_delay_ms=key_delay_ms,
            clear_first=clear_first,
            press_enter=press_enter,
        )

    def press_key(self, key: str, force_virtual: bool = False) -> None:
        """Presses and releases a single key via keyboard driver."""
        driver = self._get_driver(force_virtual=force_virtual)
        driver.press_key(key)

    def press_hotkey(self, *keys: str, force_virtual: bool = False) -> None:
        """Presses and releases a hotkey combination via keyboard driver."""
        driver = self._get_driver(force_virtual=force_virtual)
        driver.press_hotkey(*keys)

    def type_into(
        self,
        query: str,
        text: str,
        clear_first: bool = True,
        press_enter: bool = True,
        force_virtual: bool = False,
    ) -> bool:
        """Locates an input field, clicks it to gain focus, and types text."""
        success = self.click_target(query=query, force_virtual=force_virtual)
        if success:
            self.type_text(
                text=text,
                clear_first=clear_first,
                press_enter=press_enter,
                force_virtual=force_virtual,
            )
        return success

    def get_affordances(self, action_type: str = "clickable") -> List[Tuple[float, float, float]]:
        """Returns list of normalized (x, y, confidence) points for interactive controls."""
        state = self.current_state or self.perceive_now()
        spots = []
        for x, y, act, conf in (state.action_hotspots or []):
            if act == action_type:
                spots.append((x, y, conf))
        return spots

    def clickable_hotspots(self) -> List[Tuple[float, float, float]]:
        """Convenience helper: returns top clickable focus points."""
        return self.get_affordances(action_type="clickable")

    def typeable_hotspots(self) -> List[Tuple[float, float, float]]:
        """Convenience helper: returns top input/typeable focus points."""
        return self.get_affordances(action_type="typeable")

    def on_change(self, callback: Callable[[CortexState], None]) -> Callable[[CortexState], None]:
        """Decorator or method to register a screen state change listener."""
        self._change_callbacks.append(callback)
        return callback

    def on_surprise(
        self,
        callback_or_threshold: Any = None,
        threshold: float = 0.3,
    ) -> Any:
        """Decorator or method to register a surprise/anomaly event listener.

        Supports both styles:
        >>> @eye.on_surprise
        ... def alert(state): ...
        >>> @eye.on_surprise(threshold=0.45)
        ... def alert(state): ...
        """
        if callable(callback_or_threshold):
            # Bare decorator: @eye.on_surprise
            self._surprise_callbacks.append((threshold, callback_or_threshold))
            return callback_or_threshold

        # Parameterized decorator: @eye.on_surprise(threshold=0.45)
        effective_thresh = (
            float(callback_or_threshold)
            if isinstance(callback_or_threshold, (int, float))
            else threshold
        )

        def decorator(cb: Callable[[CortexState], None]) -> Callable[[CortexState], None]:
            self._surprise_callbacks.append((effective_thresh, cb))
            return cb

        return decorator

    def on_affordance(
        self,
        action_type: Any = "clickable",
        min_confidence: float = 0.5,
    ) -> Any:
        """Decorator to trigger callbacks when specific interactive affordances appear.

        Supports both bare and parameterized styles:
        >>> @eye.on_affordance
        ... def on_click(state): ...
        >>> @eye.on_affordance(action_type="typeable", min_confidence=0.6)
        ... def on_input(state): ...
        """
        if callable(action_type):
            # Bare decorator: @eye.on_affordance
            self._affordance_callbacks.append(("clickable", min_confidence, action_type))
            return action_type

        act_type = str(action_type)

        def decorator(cb: Callable[[CortexState], None]) -> Callable[[CortexState], None]:
            self._affordance_callbacks.append((act_type, min_confidence, cb))
            return cb

        return decorator

    def _dispatch_state(self, state: CortexState) -> None:
        for cb in self._change_callbacks:
            try:
                cb(state)
            except Exception as e:
                print(f"[ScreenEye] Callback error: {e}")

        for act_type, min_conf, cb in self._affordance_callbacks:
            try:
                has_affordance = any(
                    act == act_type and conf >= min_conf
                    for _, _, act, conf in (state.action_hotspots or [])
                )
                if has_affordance:
                    cb(state)
            except Exception as e:
                print(f"[ScreenEye] Affordance callback error: {e}")

    def _dispatch_surprise(self, state: CortexState) -> None:
        for thresh, cb in self._surprise_callbacks:
            if state.surprise_score >= thresh or state.stability_state == "surprise_event":
                try:
                    cb(state)
                except Exception as e:
                    print(f"[ScreenEye] Surprise callback error: {e}")

    def close(self) -> None:
        """Closes ScreenEye and releases capture and motor resources."""
        self.stop()
        if hasattr(self.source, "close"):
            try:
                self.source.close()
            except Exception:
                pass
        for driver in list(self._drivers.values()):
            if hasattr(driver, "close"):
                try:
                    driver.close()
                except Exception:
                    pass
        self._drivers.clear()

    def __enter__(self) -> ScreenEye:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


