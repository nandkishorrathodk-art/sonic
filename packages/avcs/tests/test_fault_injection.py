"""Unit tests for the AVCS deterministic fault-injection layer."""
from __future__ import annotations

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
from avcs.capture_adapter import DisplayGeometry, SyntheticCaptureAdapter
from avcs.fault_injection import (
    FaultConfig,
    FaultInjectedCaptureAdapter,
    FaultInjectedDriver,
    FaultInjectionEnvironment,
    FaultType,
)
from avcs.human_eye import HumanEye, HumanEyeState
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def _make_test_broker(fault_config: FaultConfig) -> Tuple[ActionBroker, FaultInjectionEnvironment]:
    base_driver = VirtualCursorDriver()
    base_cap = SyntheticCaptureAdapter().open()
    cap = FaultInjectedCaptureAdapter(base_cap, fault_config)
    eye = HumanEye(source=cap)
    driver = FaultInjectedDriver(base_driver, fault_config)
    safety = ActionSafetyController(allow_destructive=False)
    broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)
    env = FaultInjectionEnvironment(config=fault_config, driver=driver, capture=cap, eye=eye)
    return broker, env


def test_fault_injection_failed_dispatch():
    cfg = FaultConfig(fault_type=FaultType.FAILED_DISPATCH)
    broker, env = _make_test_broker(cfg)

    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.IDEMPOTENT,
    )
    res = broker.execute(req)

    assert res.status == ActionStatus.FAILED
    assert res.error_category == "DISPATCH_FAILED"
    assert "dispatch failed" in res.evidence.lower()
    assert not res.is_confirmed


def test_fault_injection_capture_error():
    cfg = FaultConfig(fault_type=FaultType.CAPTURE_ERROR)
    broker, env = _make_test_broker(cfg)
    env.arm()
    import time
    time.sleep(0.05)

    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        action_class=ActionClass.IDEMPOTENT,
    )
    res = broker.execute(req)

    assert res.status == ActionStatus.FAILED
    assert res.error_category == "CAPTURE_ERROR"
    assert "capture failed" in res.evidence.lower()
    assert not res.is_confirmed


def test_fault_injection_verifier_exception():
    cfg = FaultConfig(fault_type=FaultType.VERIFIER_EXCEPTION)
    broker, env = _make_test_broker(cfg)

    postcondition = env.build_postcondition(lambda pre, post: True)
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        expected_postcondition=postcondition,
    )
    res = broker.execute(req)

    # Verifier exception must NEVER report CONFIRMED
    assert res.status != ActionStatus.CONFIRMED
    assert not res.is_confirmed


def test_fault_injection_silent_input_drop():
    cfg = FaultConfig(fault_type=FaultType.SILENT_INPUT_DROP)
    broker, env = _make_test_broker(cfg)

    postcondition = env.build_postcondition(lambda pre, post: True)
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        expected_postcondition=postcondition,
    )
    res = broker.execute(req)

    assert res.status != ActionStatus.CONFIRMED
    assert not res.is_confirmed


def test_fault_injection_delayed_transition():
    cfg = FaultConfig(fault_type=FaultType.DELAYED_TRANSITION)
    broker, env = _make_test_broker(cfg)

    postcondition = env.build_postcondition(lambda pre, post: True)
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        expected_postcondition=postcondition,
    )
    res = broker.execute(req)

    assert res.status != ActionStatus.CONFIRMED
    assert not res.is_confirmed


def test_fault_injection_focus_theft():
    cfg = FaultConfig(fault_type=FaultType.FOCUS_THEFT)
    broker, env = _make_test_broker(cfg)

    # Target window is specifically required
    req = ActionRequest(
        action_type="click",
        target_coord=(0.5, 0.5),
        target_window_title="Expected Application",
    )
    res = broker.execute(req)

    # Because active window is None or does not match Expected Application, focus check blocks
    assert res.status == ActionStatus.BLOCKED
    assert res.error_category == "PRECONDITION_BLOCKED"
    assert not res.is_confirmed
