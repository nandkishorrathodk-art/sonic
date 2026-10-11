"""Comprehensive tests for ActionBroker recovery, retry policies, and non-idempotent safety."""
from __future__ import annotations

import time
import pytest
import numpy as np

from avcs.action_broker import (
    ActionBroker,
    ActionClass,
    ActionRequest,
    ActionResult,
    ActionSafetyController,
    ActionStatus,
)
from avcs.capture_adapter import SyntheticCaptureAdapter
from avcs.human_eye import HumanEye, HumanEyeState
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def _make_broker():
    driver = VirtualCursorDriver()
    cap = SyntheticCaptureAdapter().open()
    eye = HumanEye(source=cap)
    safety = ActionSafetyController(allow_destructive=False)
    broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)
    return broker, eye, driver


def test_idempotent_action_micro_jitter_recovery():
    broker, eye, driver = _make_broker()

    call_count = 0

    def recovering_postcondition(pre: HumanEyeState, post: HumanEyeState) -> bool:
        nonlocal call_count
        call_count += 1
        # Fails on the first dispatch, succeeds on the second (micro-jitter)
        return call_count > 1

    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.IDEMPOTENT,
        expected_postcondition=recovering_postcondition,
    )
    res = broker.execute(req)

    assert res.status == ActionStatus.CONFIRMED
    assert res.recovery_attempts > 1
    assert "Recovered via interior micro-jitter" in res.evidence
    assert res.is_confirmed


def test_non_idempotent_action_no_blind_retry():
    broker, eye, driver = _make_broker()

    call_count = 0

    def never_passing_postcondition(pre: HumanEyeState, post: HumanEyeState) -> bool:
        nonlocal call_count
        call_count += 1
        return False

    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.NON_IDEMPOTENT,
        expected_postcondition=never_passing_postcondition,
    )
    res = broker.execute(req)

    # Must NOT report confirmed
    assert res.status != ActionStatus.CONFIRMED
    assert not res.is_confirmed
    # Recovery attempts must be strictly 1 (no blind replay)
    assert res.recovery_attempts == 1
    assert "blind replay blocked by safety controller" in res.evidence


def test_conditionally_repeatable_precondition_satisfied():
    broker, eye, driver = _make_broker()

    call_count = 0

    def recovering_postcondition(pre: HumanEyeState, post: HumanEyeState) -> bool:
        nonlocal call_count
        call_count += 1
        return call_count > 1

    # Precondition validator returns True, indicating safe to retry
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.CONDITIONALLY_REPEATABLE,
        precondition_validator=lambda state: True,
        expected_postcondition=recovering_postcondition,
    )
    res = broker.execute(req)

    assert res.status == ActionStatus.CONFIRMED
    assert res.recovery_attempts > 1
    assert res.is_confirmed


def test_conditionally_repeatable_precondition_violated():
    broker, eye, driver = _make_broker()

    call_count = 0

    def never_passing_postcondition(pre: HumanEyeState, post: HumanEyeState) -> bool:
        nonlocal call_count
        call_count += 1
        return False

    # Precondition validator returns False (precondition state was altered/invalidated)
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.CONDITIONALLY_REPEATABLE,
        precondition_validator=lambda state: False,
        expected_postcondition=never_passing_postcondition,
    )
    res = broker.execute(req)

    # Precondition check before execution blocks it immediately
    assert res.status == ActionStatus.BLOCKED
    assert res.error_category == "PRECONDITION_BLOCKED"
    assert not res.is_confirmed
    assert res.recovery_attempts == 0


def test_high_risk_action_unauthorized_blocked():
    broker, eye, driver = _make_broker()

    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.HIGH_RISK,
        authorized=False,
    )
    res = broker.execute(req)

    assert res.status == ActionStatus.BLOCKED
    assert res.error_category == "PRECONDITION_BLOCKED"
    assert "lacks authorization grant" in res.evidence
    assert not res.is_confirmed


def test_ambiguous_outcome_never_reported_confirmed():
    broker, eye, driver = _make_broker()

    # In a scenario where optical ripple occurs but expected postcondition is false:
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.IDEMPOTENT,
        expected_postcondition=lambda pre, post: False,
    )
    res = broker.execute(req)

    assert res.status != ActionStatus.CONFIRMED
    assert not res.is_confirmed
    assert res.status in (ActionStatus.AMBIGUOUS, ActionStatus.FAILED)

