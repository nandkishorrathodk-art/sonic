"""Tests for AVCS Milestone 1 & 2: Capture Abstraction and Action Broker State Machine.

Validates:
1. DisplayGeometry coordinate transforms across multi-monitor frames.
2. Precondition validation: out-of-bounds rejection, focus checking, high-risk authorization.
3. Action lifecycle states: CONFIRMED, FAILED, AMBIGUOUS, BLOCKED.
4. Non-idempotent blind retry prevention.
5. Semantic postcondition evaluation.
"""
from __future__ import annotations

import time

import pytest

from avcs.action_broker import (
    ActionBroker,
    ActionClass,
    ActionRequest,
    ActionResult,
    ActionSafetyController,
    ActionStatus,
)
from avcs.capture_adapter import DisplayGeometry, SyntheticCaptureAdapter
from avcs.human_eye import HumanEye, HumanEyeState
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.motor import VirtualCursorDriver


def test_display_geometry_coordinate_transforms():
    """Verify DisplayGeometry normalizes multi-monitor and negative coordinates."""
    # Dual monitor setup: Primary is 1920x1080 at (0, 0), Secondary is 1920x1080 at (-1920, 0)
    # Virtual bounding rect: left = -1920, top = 0, width = 3840, height = 1080
    geo = DisplayGeometry(
        virtual_left=-1920,
        virtual_top=0,
        virtual_width=3840,
        virtual_height=1080,
        dpi_scale=1.25,
        is_primary=True,
    )

    # Origin of secondary monitor (-1920, 0) -> normalized (0.0, 0.0)
    nx, ny = geo.to_normalized(-1920, 0)
    assert pytest.approx(nx) == 0.0
    assert pytest.approx(ny) == 0.0

    # Center of dual desktop (0, 540) -> normalized (0.5, 0.5)
    nx, ny = geo.to_normalized(0, 540)
    assert pytest.approx(nx) == 0.5
    assert pytest.approx(ny) == 0.5

    # Reverse transformation
    sx, sy = geo.to_screen(0.5, 0.5)
    assert sx == 0
    assert sy == 540


def test_action_safety_controller_blocks_out_of_bounds():
    """Verify ActionSafetyController blocks out-of-bounds targets."""
    controller = ActionSafetyController()
    req = ActionRequest(
        action_type="click",
        target_coord=(1.2, 0.5),
        action_class=ActionClass.IDEMPOTENT,
    )
    eye = HumanEye(source=SyntheticScreen())
    state = eye.observe()
    valid, reason = controller.validate_preconditions(req, state)
    assert not valid
    assert "out of bounds" in str(reason)


def test_action_safety_controller_blocks_unauthorized_high_risk():
    """Verify ActionSafetyController strictly blocks unauthorized HIGH_RISK actions."""
    controller = ActionSafetyController(allow_destructive=False)
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.HIGH_RISK,
        authorized=False,
    )
    eye = HumanEye(source=SyntheticScreen())
    state = eye.observe()
    valid, reason = controller.validate_preconditions(req, state)
    assert not valid
    assert "HIGH_RISK" in str(reason)

    # When authorized, precondition passes
    auth_req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.HIGH_RISK,
        authorized=True,
    )
    valid_auth, reason_auth = controller.validate_preconditions(auth_req, state)
    assert valid_auth
    assert reason_auth is None


def test_action_broker_blocks_blocked_request():
    """Verify ActionBroker returns BLOCKED status without dispatching OS input."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    broker = ActionBroker(eye=eye, driver=driver)

    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.HIGH_RISK,
        authorized=False,
    )
    result = broker.execute(req)
    assert isinstance(result, ActionResult)
    assert result.status == ActionStatus.BLOCKED
    assert not result.is_confirmed
    assert result.error_category == "PRECONDITION_BLOCKED"


def test_action_broker_prevents_blind_replay_on_non_idempotent():
    """Verify ActionBroker prohibits blind micro-jitter replay on non-idempotent actions."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    broker = ActionBroker(eye=eye, driver=driver)

    # Non-idempotent action (e.g. Submit Order)
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.NON_IDEMPOTENT,
    )
    result = broker.execute(req)
    # Recovery attempts must remain 1 (no blind retries)
    assert result.recovery_attempts == 1


def test_action_broker_semantic_postcondition_evaluation():
    """Verify ActionBroker evaluates semantic postcondition predicate."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    broker = ActionBroker(eye=eye, driver=driver)

    # Custom postcondition that asserts true
    def always_success_postcondition(pre: HumanEyeState, post: HumanEyeState) -> bool:
        return True

    req_pass = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        expected_postcondition=always_success_postcondition,
    )
    res_pass = broker.execute(req_pass)
    assert res_pass.status == ActionStatus.CONFIRMED
    assert "Semantic postcondition explicitly verified" in res_pass.evidence

    # Custom postcondition that asserts false
    def failing_postcondition(pre: HumanEyeState, post: HumanEyeState) -> bool:
        return False

    req_fail = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        expected_postcondition=failing_postcondition,
    )
    res_fail = broker.execute(req_fail)
    assert res_fail.status in (ActionStatus.AMBIGUOUS, ActionStatus.FAILED)
    assert not res_fail.is_confirmed


def test_action_broker_preserves_mouse_button_and_never_retries_typing():
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    clicks = []
    original_click = driver.click

    def record_click(button="left"):
        clicks.append(button)
        return original_click(button=button)

    driver.click = record_click
    broker = ActionBroker(eye=eye, driver=driver, settle_delay_s=0)
    result = broker.execute(
        ActionRequest(
            action_type="click",
            target_coord=(0.5, 0.5),
            button="right",
            expected_postcondition=lambda before, after: True,
        )
    )
    assert result.status == ActionStatus.CONFIRMED
    assert clicks == ["right"]

    typed = []
    driver.type_text = typed.append
    broker.verifier.calculate_local_ripple = lambda *args: 0.0
    type_result = broker.execute(
        ActionRequest(
            action_type="type",
            target_coord=(0.5, 0.5),
            text_content="do not duplicate",
            action_class=ActionClass.IDEMPOTENT,
            timeout_ms=2000,
        )
    )
    assert type_result.recovery_attempts == 1
    assert typed == ["do not duplicate"]


def test_action_broker_timeout_before_dispatch_does_not_click():
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    clicks = []
    driver.click = lambda button="left": clicks.append(button)
    broker = ActionBroker(eye=eye, driver=driver, settle_delay_s=0)

    def slow_precondition(_state):
        time.sleep(0.02)
        return True

    result = broker.execute(
        ActionRequest(
            action_type="click",
            target_coord=(0.5, 0.5),
            precondition_validator=slow_precondition,
            timeout_ms=1,
        )
    )
    assert result.status == ActionStatus.FAILED
    assert result.error_category == "TIMEOUT_BEFORE_DISPATCH"
    assert clicks == []


def test_action_broker_reports_ambiguous_when_dispatch_exceeds_timeout(monkeypatch):
    eye = HumanEye(source=SyntheticScreen())
    state = eye.observe()
    monkeypatch.setattr(eye, "observe", lambda: state)
    driver = VirtualCursorDriver()
    clicks = []

    def slow_click(button="left"):
        clicks.append(button)
        time.sleep(0.02)

    driver.click = slow_click
    broker = ActionBroker(eye=eye, driver=driver, settle_delay_s=0)
    result = broker.execute(
        ActionRequest(
            action_type="click",
            target_coord=(0.5, 0.5),
            timeout_ms=1,
        )
    )
    assert result.status == ActionStatus.AMBIGUOUS
    assert result.error_category == "TIMEOUT_AFTER_DISPATCH"
    assert clicks == ["left"]
