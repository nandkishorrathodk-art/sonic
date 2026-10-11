"""AVCS Milestone 2: Action Safety Controller & Bounded Windows Action Broker.

Replaces blind click-and-assume-success with a formal action lifecycle state machine:
States: PENDING -> DISPATCHED -> (CONFIRMED | FAILED | AMBIGUOUS | BLOCKED)
Features:
- Action risk classification (IDEMPOTENT, NON_IDEMPOTENT, HIGH_RISK)
- Focus and target freshness pre-validation before SendInput
- Blind retry prevention on non-idempotent actions
- Semantic postcondition evaluation with local photometric delta evidence
"""
from __future__ import annotations

import enum
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from avcs.human_eye import HumanEye, HumanEyeState
from avcs.verifier import VerificationResult, VerificationStatus, VisualRippleVerifier
from forge.vision.cortex.motor.driver import OSCursorDriver, get_default_driver


class ActionClass(str, enum.Enum):
    """Risk and replay categorization of agent operations."""
    IDEMPOTENT = "idempotent"                         # Safe to repeat (e.g. switch tab, open dropdown)
    CONDITIONALLY_REPEATABLE = "cond_repeatable"     # Repeatable only if precondition still holds
    NON_IDEMPOTENT = "non_idempotent"                 # Must never blind-retry (e.g. Submit, Delete, Pay)
    HIGH_RISK = "high_risk"                           # Requires explicit caller authorization


class ActionStatus(str, enum.Enum):
    """Discrete terminal states of an action execution lifecycle."""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ActionRequest:
    """Bounded, validated interaction request."""
    action_type: str                                       # "click", "double_click", "type", "press_key"
    target_coord: Tuple[float, float]                      # Normalized (x, y) coordinates
    action_class: ActionClass = ActionClass.IDEMPOTENT
    action_id: str = field(default_factory=lambda: f"act_{uuid.uuid4().hex[:10]}")
    text_content: Optional[str] = None
    expected_postcondition: Optional[Callable[[HumanEyeState, HumanEyeState], bool]] = None
    precondition_validator: Optional[Callable[[HumanEyeState], bool]] = None
    authorized: bool = False                               # Required true for HIGH_RISK actions
    target_window_title: Optional[str] = None
    timeout_ms: float = 1500.0
    button: str = "left"


@dataclass(frozen=True)
class ActionResult:
    """Auditable, immutable result of an action execution."""
    action_id: str
    status: ActionStatus
    action_type: str
    target_coord: Tuple[float, float]
    elapsed_ms: float
    evidence: str
    optical_ripple_energy: float
    photometric_delta: float
    recovery_attempts: int
    error_category: Optional[str] = None

    @property
    def is_confirmed(self) -> bool:
        return self.status == ActionStatus.CONFIRMED


class ActionSafetyController:
    """Validates target freshness, window focus, and authorization boundaries."""

    def __init__(self, allow_destructive: bool = False) -> None:
        self.allow_destructive = allow_destructive

    def validate_preconditions(
        self,
        request: ActionRequest,
        current_state: HumanEyeState,
    ) -> Tuple[bool, Optional[str]]:
        """Checks target bounds, focus, and security authorizations."""
        tx, ty = request.target_coord
        if request.action_type not in {"click", "double_click", "type", "press_key"}:
            return False, f"Unsupported action type: {request.action_type}"
        if request.button not in {"left", "right", "middle"}:
            return False, f"Unsupported mouse button: {request.button}"
        if (
            isinstance(request.timeout_ms, bool)
            or not isinstance(request.timeout_ms, (int, float))
            or not math.isfinite(request.timeout_ms)
            or request.timeout_ms <= 0
        ):
            return False, f"timeout_ms must be a positive finite number: {request.timeout_ms}"
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            return False, f"Target coordinate ({tx:.3f}, {ty:.3f}) out of bounds [0, 1]."

        # Validate High-Risk authorization
        if request.action_class == ActionClass.HIGH_RISK:
            if not request.authorized and not self.allow_destructive:
                return False, f"Action '{request.action_id}' is HIGH_RISK and lacks authorization grant."

        # Validate Window Focus if specified
        if request.target_window_title and current_state.scene_tree:
            current_win = current_state.scene_tree.window_title or ""
            if request.target_window_title.lower() not in current_win.lower():
                return False, f"Focus mismatch: expected window '{request.target_window_title}', but active is '{current_win}'."

        # Validate Precondition Validator if specified on request
        if request.precondition_validator is not None:
            try:
                pre_ok = request.precondition_validator(current_state)
                if not pre_ok:
                    return False, "Precondition validator evaluated to False."
            except Exception as exc:
                return False, f"Precondition validator raised exception: {exc}"

        return True, None


class ActionBroker:
    """Executes actions through the bounded lifecycle state machine with verified postconditions."""

    def __init__(
        self,
        eye: HumanEye,
        driver: Optional[OSCursorDriver] = None,
        safety_controller: Optional[ActionSafetyController] = None,
        settle_delay_s: float = 0.04,
    ) -> None:
        self.eye = eye
        self.driver = driver or get_default_driver()
        self.verifier = VisualRippleVerifier(eye=self.eye, driver=self.driver)
        self.safety = safety_controller or ActionSafetyController()
        self.settle_delay_s = max(0.0, float(settle_delay_s))

    def observe(self) -> HumanEyeState:
        """Captures and returns current visual state through the eye."""
        return self.eye.observe()

    def execute(self, request: ActionRequest) -> ActionResult:
        """Executes a bounded action request through the state machine."""
        t0 = time.perf_counter()
        if (
            isinstance(request.timeout_ms, bool)
            or not isinstance(request.timeout_ms, (int, float))
            or not math.isfinite(request.timeout_ms)
            or request.timeout_ms <= 0
        ):
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.BLOCKED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                evidence=(
                    "timeout_ms must be a positive finite number: "
                    f"{request.timeout_ms!r}"
                ),
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="INVALID_TIMEOUT",
            )
        deadline = t0 + request.timeout_ms / 1000.0

        # 1. Capture Pre-State
        try:
            pre_state = self.eye.observe()
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=f"Observation capture failed: {exc}",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="CAPTURE_ERROR",
            )

        # 2. Safety Controller Precondition Validation
        valid, reason = self.safety.validate_preconditions(request, pre_state)
        if not valid:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.BLOCKED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=reason or "Precondition validation failed.",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="PRECONDITION_BLOCKED",
            )

        if time.perf_counter() >= deadline:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence="Action deadline expired before input dispatch.",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="TIMEOUT_BEFORE_DISPATCH",
            )

        # 3. Dispatch Physical Input to OS
        tx, ty = request.target_coord
        dispatch_error: Optional[Exception] = None
        try:
            self._dispatch_input(request, tx, ty)
        except Exception as exc:
            dispatch_error = exc

        if dispatch_error is not None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=f"Input dispatch failed: {dispatch_error}",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="DISPATCH_FAILED",
            )

        if time.perf_counter() >= deadline:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.AMBIGUOUS,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=(
                    "Input dispatch exceeded timeout_ms; the physical action may "
                    "have occurred and was not replayed."
                ),
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="TIMEOUT_AFTER_DISPATCH",
            )

        # Allow settlement for UI animation
        if self.settle_delay_s > 0:
            time.sleep(min(self.settle_delay_s, max(0.0, deadline - time.perf_counter())))

        # 4. Capture Post-State & Verify Outcome
        try:
            post_state = self.eye.observe()
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=f"Post-action capture failed: {exc}",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="CAPTURE_ERROR",
            )

        if time.perf_counter() >= deadline:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.AMBIGUOUS,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=(
                    "Post-action observation completed after timeout_ms; outcome "
                    "is ambiguous and the action was not replayed."
                ),
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="TIMEOUT_AFTER_DISPATCH",
            )

        ripple_energy = self.verifier.calculate_local_ripple(request.target_coord, post_state.motion_field)

        # Evaluate Photometric Local Delta
        photo_delta = 0.0
        if pre_state.motion_field is not None and post_state.motion_field is not None:
            pre_local = self.verifier.calculate_local_ripple(request.target_coord, pre_state.motion_field)
            photo_delta = abs(ripple_energy - pre_local)

        # 5. Semantic Postcondition Evaluation
        postcondition_passed = False
        if request.expected_postcondition is not None:
            try:
                postcondition_passed = request.expected_postcondition(pre_state, post_state)
            except Exception as exc:
                print(f"[ActionBroker] Postcondition evaluation exception: {exc}")
                postcondition_passed = False

        # Structural UI transition detected
        diff = post_state.visual_diff
        structural_change = bool(diff and (diff.added_elements or diff.removed_ids or diff.modified_elements))

        # Decision Logic: Determine Terminal State
        status = ActionStatus.FAILED
        evidence_msg = "No reaction observed."

        if request.expected_postcondition is not None:
            if postcondition_passed:
                status = ActionStatus.CONFIRMED
                evidence_msg = "Semantic postcondition explicitly verified."
            elif ripple_energy > 0.015 or structural_change:
                status = ActionStatus.AMBIGUOUS
                evidence_msg = "Visual motion detected, but expected semantic postcondition was not satisfied."
            else:
                status = ActionStatus.FAILED
                evidence_msg = "Neither semantic postcondition nor local optical motion occurred."
        else:
            # General UI action verification (requires localized ripple or state diff)
            if (ripple_energy >= 0.015 or photo_delta >= 0.015) and post_state.surprise_score <= 0.4:
                status = ActionStatus.CONFIRMED
                evidence_msg = "Physical visual ripple confirmed at target coordinate."
            elif structural_change and post_state.surprise_score <= 0.4:
                status = ActionStatus.CONFIRMED
                evidence_msg = "Structural UI state change confirmed."
            elif post_state.surprise_score > 0.3:
                status = ActionStatus.AMBIGUOUS
                evidence_msg = "Screen underwent global shift; target state confirmation is ambiguous."
            else:
                status = ActionStatus.FAILED
                evidence_msg = "Target showed zero optical reaction."

        # 6. Recovery Policy for Actions
        recovery_attempts = 1
        can_retry = False
        if status in (ActionStatus.FAILED, ActionStatus.AMBIGUOUS):
            can_retry = request.action_type in {"click", "double_click"}
            if not can_retry:
                evidence_msg += " (Non-idempotent input is never replayed.)"
            elif request.action_class == ActionClass.IDEMPOTENT:
                can_retry = True
            elif request.action_class == ActionClass.CONDITIONALLY_REPEATABLE:
                can_retry = (
                    request.precondition_validator is not None
                    and request.precondition_validator(post_state)
                )
                if not can_retry:
                    evidence_msg += " (Conditionally repeatable action: precondition validator did not pass on post-state)."
            elif request.action_class in (ActionClass.NON_IDEMPOTENT, ActionClass.HIGH_RISK):
                can_retry = False
                evidence_msg += " (Non-idempotent action: blind replay blocked by safety controller)."

        if can_retry:
            # Interior Micro-Jitter Retry (Offset by 3px within interior hitbox)
            offsets = [(0.003, 0.003), (-0.003, -0.003)]
            for ox, oy in offsets:
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    status = ActionStatus.AMBIGUOUS
                    evidence_msg = "Action deadline expired; no further input was dispatched."
                    break
                recovery_attempts += 1
                rx = float(np.clip(tx + ox, 0.0, 1.0))
                ry = float(np.clip(ty + oy, 0.0, 1.0))
                try:
                    pre_retry = self.eye.observe()
                    self._dispatch_input(request, rx, ry)
                    if self.settle_delay_s > 0:
                        time.sleep(
                            min(self.settle_delay_s, max(0.0, deadline - time.perf_counter()))
                        )
                    post_retry = self.eye.observe()
                    if time.perf_counter() >= deadline:
                        status = ActionStatus.AMBIGUOUS
                        evidence_msg = (
                            "Action deadline expired during recovery; outcome is "
                            "ambiguous and no further inputs were dispatched."
                        )
                        break
                    retry_ripple = self.verifier.calculate_local_ripple((rx, ry), post_retry.motion_field)
                except Exception:
                    break

                if request.expected_postcondition is not None:
                    try:
                        passed = request.expected_postcondition(pre_retry, post_retry)
                    except Exception:
                        passed = False
                    if passed:
                        status = ActionStatus.CONFIRMED
                        evidence_msg = f"Recovered via interior micro-jitter ({ox:+.3f}, {oy:+.3f}) and verified semantic postcondition."
                        ripple_energy = retry_ripple
                        break
                else:
                    if retry_ripple >= 0.015 or bool(post_retry.visual_diff and (post_retry.visual_diff.added_elements or post_retry.visual_diff.removed_ids)):
                        status = ActionStatus.CONFIRMED
                        evidence_msg = f"Recovered via interior micro-jitter ({ox:+.3f}, {oy:+.3f})."
                        ripple_energy = retry_ripple
                        break

        elapsed = (time.perf_counter() - t0) * 1000.0
        return ActionResult(
            action_id=request.action_id,
            status=status,
            action_type=request.action_type,
            target_coord=request.target_coord,
            elapsed_ms=elapsed,
            evidence=evidence_msg,
            optical_ripple_energy=ripple_energy,
            photometric_delta=photo_delta,
            recovery_attempts=recovery_attempts,
            error_category=None if status == ActionStatus.CONFIRMED else status.value.upper(),
        )

    def _dispatch_input(self, request: ActionRequest, x: float, y: float) -> None:
        """Dispatches concrete mouse/keyboard events through cursor driver."""
        if hasattr(self.driver, "move_to_normalized"):
            self.driver.move_to_normalized(x, y)
        elif hasattr(self.driver, "move_to"):
            self.driver.move_to(x, y)
        else:
            raise RuntimeError("Cursor driver does not support pointer movement.")

        if request.action_type == "click":
            self.driver.click(button=request.button)
        elif request.action_type == "double_click":
            self.driver.double_click(button=request.button)
        elif request.action_type == "type":
            self.driver.click(button="left")
            time.sleep(0.02)
            if request.text_content and hasattr(self.driver, "type_text"):
                self.driver.type_text(request.text_content)
            elif request.text_content:
                raise RuntimeError("Cursor driver does not support text entry.")
        elif request.action_type == "press_key":
            if request.text_content and hasattr(self.driver, "press_key"):
                self.driver.press_key(request.text_content)
            else:
                raise RuntimeError("press_key action requires text_content and driver support.")
        else:
            raise RuntimeError(f"Unsupported action type: {request.action_type}")

        # Notify interactive capture source of simulated input event
        cap = getattr(self.eye, "source", None)
        while cap is not None:
            if hasattr(cap, "trigger_motion"):
                cap.trigger_motion(x, y)
                break
            cap = getattr(cap, "source", None)
