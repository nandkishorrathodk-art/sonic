import time

import pytest

from sonic.computer.models import ScreenObservation
from sonic.computer_use.action_contracts import (
    ActionContract,
    ActionContractValidator,
    ActionReceipt,
    GUIActionIntent,
    StateDelta,
)
from sonic.computer_use.models import ActionExecutionStatus, ComputerWorldObservation
from sonic.safety.kernel import KernelVerdict, SafetyAuthorization


class _AllowKernel:
    def __init__(self, verdict=KernelVerdict.ALLOW):
        self.verdict = verdict
        self.calls = 0

    def authorize(self, action_type, parameters, provider=None):
        self.calls += 1
        return SafetyAuthorization(
            verdict=self.verdict,
            reason="test",
            action_type=action_type,
        )


def test_contract_validation_reuses_safety_and_checks_preconditions():
    kernel = _AllowKernel()
    validator = ActionContractValidator(safety_kernel=kernel)
    contract = ActionContract("ACTION", preconditions={"mode": "ready"})

    result = validator.validate(contract, state={"mode": "ready"}, parameters={"x": 1})

    assert result.valid
    assert kernel.calls == 1


def test_validator_fails_closed_for_safety_and_precondition_failures():
    kernel = _AllowKernel(KernelVerdict.DENY)
    validator = ActionContractValidator(safety_kernel=kernel)
    contract = ActionContract("ACTION", preconditions={"mode": "ready"})

    result = validator.validate(contract, state={"mode": "blocked"})

    assert not result.valid
    assert any(reason.startswith("safety:") for reason in result.reasons)
    assert any(reason.startswith("precondition") for reason in result.reasons)


def test_receipt_requires_real_delta_preconditions_and_postconditions():
    contract = ActionContract(
        "ACTION",
        preconditions={"mode": "ready"},
        postconditions={"mode": "done"},
    )
    before = {"mode": "ready"}
    after = {"mode": "done", "count": 1}
    receipt = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.SUCCEEDED,
        before=before,
        after=after,
        state_delta=StateDelta.from_states(before, after),
        exit_code=0,
    )

    assert ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, receipt
    ).valid

    forged = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.SUCCEEDED,
        before=before,
        after=after,
        state_delta=StateDelta(),
        exit_code=0,
    )
    assert not ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, forged
    ).valid


def test_receipt_without_execution_evidence_or_matching_precondition_is_rejected():
    contract = ActionContract(
        "ACTION",
        preconditions={"mode": "ready"},
        postconditions={"mode": "done"},
    )
    before = {"mode": "blocked"}
    after = {"mode": "done"}
    receipt = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.SUCCEEDED,
        before=before,
        after=after,
        state_delta=StateDelta.from_states(before, after),
    )
    result = ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, receipt
    )
    assert not result.valid
    assert "neither a successful exit code nor evidence" in " ".join(result.reasons)
    assert "precondition failed in receipt: mode" in result.reasons


def test_failed_receipt_cannot_satisfy_postconditions():
    contract = ActionContract("ACTION", postconditions={"value": 1})
    receipt = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.FAILED,
        before={"value": 0},
        after={"value": 1},
        state_delta=StateDelta.from_states({"value": 0}, {"value": 1}),
    )

    result = ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, receipt
    )
    assert not result.valid


def _gui_observation(*, window="Editor", version=7, age_ms=10):
    received_at_ns = time.perf_counter_ns()
    screen = ScreenObservation(
        width=1280,
        height=800,
        active_window=window,
        desktop_state="STATE_ONLY",
        capture_age_ms=0,
        controls_age_ms=age_ms,
        perception_version=version,
    )
    return ComputerWorldObservation(
        screen=screen,
        active_application=window,
        perception_version=version,
        observation_age_ms=age_ms,
        observation_received_monotonic_ns=received_at_ns,
    )


def test_gui_intent_binds_action_target_payload_and_expected_outcome():
    observation = _gui_observation()
    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 50, "y": 60},
        expected_outcome="document saved",
        observation=observation,
    )

    assert intent.expected_outcome == "document saved"
    assert intent.source_perception_version == 7
    assert intent.validate_binding(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 50, "y": 60},
    ) == ()
    assert "GUI action target differs from intent" in intent.validate_binding(
        action_type="GUI_CLICK",
        target="Open",
        payload={"x": 50, "y": 60},
    )
    assert "GUI action payload differs from intent" in intent.validate_binding(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 50, "y": 61},
    )


def test_gui_intent_expires_from_observation_age_and_requires_matching_live_window():
    observation = _gui_observation(age_ms=240)
    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 50, "y": 60},
        expected_outcome="saved",
        observation=observation,
        ttl_ms=250,
    )

    assert intent.expires_at_ns - intent.issued_at_ns <= 10_000_000
    changed_window = ScreenObservation(
        width=1280,
        height=800,
        active_window="Different window",
        desktop_state="STATE_ONLY",
        capture_age_ms=0,
        controls_age_ms=0,
        perception_version=8,
    )
    assert "active window differs from GUI action intent" in intent.validate_observation(
        changed_window
    )
    assert "GUI action intent expired" in intent.validate_observation(
        changed_window,
        now_ns=intent.expires_at_ns + 1,
    )
    same_window_new_state = ScreenObservation(
        width=1280,
        height=800,
        active_window="Editor",
        desktop_state="STATE_ONLY",
        capture_age_ms=0,
        controls_age_ms=0,
        perception_version=8,
    )
    assert "desktop state version differs from GUI action intent" in (
        intent.validate_observation(same_window_new_state)
    )
    assert "fresh desktop evidence is unavailable" in intent.validate_observation(
        observation.screen,
        now_ns=intent.issued_at_ns + 1,
        elapsed_ms=251,
        max_age_ms=250,
    )


def test_gui_intent_binds_observed_focus_and_semantic_target():
    controls = [
        {
            "name": "Save",
            "role": "button",
            "bbox": [10, 20, 70, 40],
            "source": "atspi",
            "focused": False,
        },
        {
            "name": "Document title",
            "role": "entry",
            "bbox": [100, 20, 300, 45],
            "source": "atspi",
            "focused": True,
        },
    ]
    observation = _gui_observation()
    observation.screen.detected_controls = controls
    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 40, "y": 30},
        expected_outcome="Save the document",
        observation=observation,
    )

    assert intent.target_identity is not None
    assert intent.source_focus is not None
    changed = ScreenObservation(
        width=1280,
        height=800,
        active_window="Editor",
        desktop_state="STATE_ONLY",
        capture_age_ms=0,
        controls_age_ms=0,
        perception_version=7,
        detected_controls=[
            {**controls[0], "focused": True},
            {**controls[1], "focused": False},
        ],
    )
    errors = intent.validate_observation(changed)
    assert "focused control differs from GUI action intent" in errors
    missing_target = ScreenObservation(
        width=1280,
        height=800,
        active_window="Editor",
        desktop_state="STATE_ONLY",
        capture_age_ms=0,
        controls_age_ms=0,
        perception_version=7,
        detected_controls=[controls[1]],
    )
    assert "target control disappeared or changed since GUI action intent" in (
        intent.validate_observation(missing_target)
    )


def test_gui_intent_refuses_stale_or_unversioned_observations():
    stale = _gui_observation(age_ms=300)
    with pytest.raises(ValueError, match="fresh observation"):
        GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="Save",
            payload={"x": 50, "y": 60},
            expected_outcome="saved",
            observation=stale,
            ttl_ms=250,
        )

    unversioned = _gui_observation(version=0)
    with pytest.raises(ValueError, match="versioned active desktop"):
        GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="Save",
            payload={"x": 50, "y": 60},
            expected_outcome="saved",
            observation=unversioned,
        )
