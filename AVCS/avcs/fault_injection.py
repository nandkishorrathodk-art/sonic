"""AVCS Deterministic Fault-Injection Layer.

Provides structured failure simulation across OS input dispatch, visual capture,
window focus, layout changes, overlay occlusions, and verifier state machines.
"""
from __future__ import annotations

import contextlib
import enum
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from avcs.action_broker import ActionClass, ActionRequest, ActionResult, ActionStatus
from avcs.capture_adapter import BaseCaptureAdapter, CaptureError, DisplayGeometry
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.verifier import VisualRippleVerifier
from forge.vision.cortex.motor.driver import OSCursorDriver


class FaultType(str, enum.Enum):
    """Categorization of simulated and environmental runtime failures."""
    NONE = "none"
    FAILED_DISPATCH = "failed_dispatch"                   # OS/driver rejects or drops input
    STALE_OBSERVATION = "stale_observation"               # Observation reflects outdated screen state
    TARGET_MOVEMENT = "target_movement"                   # Control moves or reflows away from target coordinate
    FOCUS_THEFT = "focus_theft"                           # Active window changes right before dispatch
    UNEXPECTED_OVERLAY = "unexpected_overlay"             # Modal dialog occludes target
    DELAYED_TRANSITION = "delayed_transition"             # UI state transition takes longer than settlement window
    CAPTURE_ERROR = "capture_error"                       # Frame grab fails (GDI error / buffer failure)
    VERIFIER_EXCEPTION = "verifier_exception"             # Postcondition evaluator throws unexpected error
    SILENT_INPUT_DROP = "silent_input_drop"               # OS reports success, but UI ignores event / state never updates


@dataclass(frozen=True)
class FaultConfig:
    """Configuration parameterizing a deterministic fault trial."""
    fault_type: FaultType = FaultType.NONE
    delay_s: float = 0.0
    active_overlay_bbox: Optional[Tuple[float, float, float, float]] = None  # Normalized (x1, y1, x2, y2)
    stolen_window_title: str = "External Unauthorized App"
    exception_message: str = "Injected verifier fault"
    dispatch_error_message: str = "Injected driver dispatch fault: ERROR_ACCESS_DENIED"
    capture_error_message: str = "Injected capture device failure: BitBlt failed"
    target_offset: Tuple[float, float] = (0.15, 0.15)


class FaultInjectedDriver(OSCursorDriver):
    """Driver wrapper that intercepts physical mouse/keyboard events for fault injection."""

    def __init__(self, base_driver: OSCursorDriver, config: FaultConfig) -> None:
        self.base_driver = base_driver
        self.config = config
        self.dispatch_count: int = 0
        self.last_action_coords: Optional[Tuple[int, int]] = None
        self.armed: bool = True

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    @property
    def screen_size(self) -> Tuple[int, int]:
        return getattr(self.base_driver, "screen_size", (1920, 1080))

    def move_to_pixels(self, x: int, y: int) -> None:
        if self.armed and self.config.fault_type == FaultType.FAILED_DISPATCH:
            raise RuntimeError(self.config.dispatch_error_message)
        self.last_action_coords = (int(x), int(y))
        if hasattr(self.base_driver, "move_to_pixels"):
            self.base_driver.move_to_pixels(x, y)

    def move_to_normalized(self, nx: float, ny: float) -> None:
        if self.armed and self.config.fault_type == FaultType.FAILED_DISPATCH:
            raise RuntimeError(self.config.dispatch_error_message)
        w, h = self.screen_size
        self.last_action_coords = (int(round(nx * (w - 1))), int(round(ny * (h - 1))))
        if hasattr(self.base_driver, "move_to_normalized"):
            self.base_driver.move_to_normalized(nx, ny)

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        self.dispatch_count += 1
        if self.armed and self.config.fault_type == FaultType.FAILED_DISPATCH:
            raise RuntimeError(self.config.dispatch_error_message)
        if self.armed and self.config.fault_type == FaultType.SILENT_INPUT_DROP:
            # OS claims click dispatched, but hardware/application event queue swallows it
            return
        if hasattr(self.base_driver, "click"):
            self.base_driver.click(button=button, hold_duration=hold_duration)

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        self.dispatch_count += 1
        if self.armed and self.config.fault_type == FaultType.FAILED_DISPATCH:
            raise RuntimeError(self.config.dispatch_error_message)
        if self.armed and self.config.fault_type == FaultType.SILENT_INPUT_DROP:
            return
        if hasattr(self.base_driver, "double_click"):
            self.base_driver.double_click(button=button, interval=interval)

    def type_text(
        self,
        text: str,
        key_delay_ms: int = 10,
        clear_first: bool = False,
        press_enter: bool = False,
    ) -> None:
        self.dispatch_count += 1
        if self.armed and self.config.fault_type == FaultType.FAILED_DISPATCH:
            raise RuntimeError(self.config.dispatch_error_message)
        if self.armed and self.config.fault_type == FaultType.SILENT_INPUT_DROP:
            return
        if hasattr(self.base_driver, "type_text"):
            self.base_driver.type_text(
                text, key_delay_ms=key_delay_ms, clear_first=clear_first, press_enter=press_enter
            )

    def press_key(self, key: str, hold_duration: float = 0.02) -> None:
        self.dispatch_count += 1
        if self.armed and self.config.fault_type == FaultType.FAILED_DISPATCH:
            raise RuntimeError(self.config.dispatch_error_message)
        if self.armed and self.config.fault_type == FaultType.SILENT_INPUT_DROP:
            return
        if hasattr(self.base_driver, "press_key"):
            self.base_driver.press_key(key, hold_duration=hold_duration)


class FaultInjectedCaptureAdapter(BaseCaptureAdapter):
    """Capture adapter that simulates hardware/OS capture failures and stale buffers."""

    def __init__(self, base_adapter: BaseCaptureAdapter, config: FaultConfig) -> None:
        self.base_adapter = base_adapter
        self.config = config
        self._stale_frame: Optional[np.ndarray] = None
        self.grab_count: int = 0
        self.armed: bool = False

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def open(self) -> FaultInjectedCaptureAdapter:
        self.base_adapter.open()
        return self

    def close(self) -> None:
        self.base_adapter.close()

    def get_geometry(self) -> DisplayGeometry:
        return self.base_adapter.get_geometry()

    def grab(self) -> np.ndarray:
        self.grab_count += 1
        if self.armed and self.config.fault_type == FaultType.CAPTURE_ERROR:
            raise CaptureError(self.config.capture_error_message)

        frame = self.base_adapter.grab()

        if self.armed and self.config.fault_type == FaultType.STALE_OBSERVATION:
            if self._stale_frame is None:
                self._stale_frame = frame.copy()
            # Deliberately return the initial frozen frame rather than the newly rendered screen
            return self._stale_frame.copy()

        return frame

    def supported_modes(self) -> List[str]:
        return self.base_adapter.supported_modes()


class FaultInjectionEnvironment:
    """Manages an active fault injection scenario around an ActionBroker."""

    def __init__(
        self,
        config: FaultConfig,
        driver: OSCursorDriver,
        capture: BaseCaptureAdapter,
        eye: HumanEye,
    ) -> None:
        self.config = config
        self.driver = driver if isinstance(driver, FaultInjectedDriver) else FaultInjectedDriver(driver, config)
        self.capture = capture if isinstance(capture, FaultInjectedCaptureAdapter) else FaultInjectedCaptureAdapter(capture, config)
        self.eye = eye

    def arm(self) -> None:
        self.driver.arm()
        self.capture.arm()

    def disarm(self) -> None:
        self.driver.disarm()
        self.capture.disarm()

    def __enter__(self) -> FaultInjectionEnvironment:
        self.arm()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disarm()

    def build_postcondition(
        self,
        true_predicate: Callable[[HumanEyeState, HumanEyeState], bool],
    ) -> Callable[[HumanEyeState, HumanEyeState], bool]:
        """Creates a postcondition function that reflects the injected fault."""
        fault = self.config.fault_type

        def _evaluated_postcondition(pre: HumanEyeState, post: HumanEyeState) -> bool:
            if fault == FaultType.VERIFIER_EXCEPTION:
                raise ValueError(self.config.exception_message)

            if fault == FaultType.DELAYED_TRANSITION:
                # The state change is delayed; postcondition is currently False
                return False

            if fault == FaultType.SILENT_INPUT_DROP:
                # Input was swallowed; target never reached intended state
                return False

            if fault == FaultType.UNEXPECTED_OVERLAY:
                # Click hit the overlay, so underlying control was not triggered
                return False

            if fault == FaultType.TARGET_MOVEMENT:
                # Click hit empty space where the control used to be
                return False

            if fault == FaultType.STALE_OBSERVATION:
                # Target wasn't really there; action did not change physical state
                return False

            if fault == FaultType.FOCUS_THEFT:
                return False

            if fault == FaultType.FAILED_DISPATCH:
                return False

            # Nominal case
            return bool(true_predicate(pre, post))

        return _evaluated_postcondition
