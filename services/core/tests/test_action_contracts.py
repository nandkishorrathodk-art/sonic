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
        screenshot_base64="pixel-frame",
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": version,
            "scene_hash": "stable-scene",
            "width": 1280,
            "height": 800,
        },
        width=1280,
        height=800,
        active_window=window,
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
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
    assert intent.validate_binding(
        action_type=" GUI_CLICK ",
        target="Save",
        payload={"x": 50, "y": 60},
    )
    assert intent.validate_binding(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": float("nan")},
    )


def test_gui_intent_uses_pixels_and_avcs_without_accessibility_or_ocr_controls():
    observation = _gui_observation()
    observation.screen.visible_text = ""
    observation.screen.detected_controls = []
    observation.screen.controls_age_ms = None

    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="40,30",
        payload={"x": 40, "y": 30},
        expected_outcome="target changed",
        observation=observation,
    )

    assert intent.source_state_version == 7
    assert intent.source_scene_hash == "stable-scene"


def test_gui_intent_accepts_frame_bound_raw_rgb_without_base64_encoding():
    frame_id = 17
    received_at_ns = time.perf_counter_ns()
    screen = ScreenObservation(
        raw_rgb=bytes((24, 48, 72)) * (8 * 6),
        frame_id=frame_id,
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 5,
            "source_frame_id": frame_id,
            "scene_hash": "raw-scene",
            "width": 8,
            "height": 6,
        },
        width=8,
        height=6,
        active_window="Editor",
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
        perception_version=5,
    )
    observation = ComputerWorldObservation(
        screen=screen,
        active_application="Editor",
        perception_version=12,
        observation_age_ms=0,
        observation_received_monotonic_ns=received_at_ns,
    )

    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="40,30",
        payload={"x": 40, "y": 30},
        expected_outcome="target changed",
        observation=observation,
    )

    assert screen.screenshot_base64 == ""
    assert intent.source_state_version == 5
    assert intent.source_active_window == "Editor"
    assert intent.validate_observation(screen) == ()


def test_gui_intent_rejects_raw_rgb_with_mismatched_avcs_frame_id():
    observation = _gui_observation()
    screen = ScreenObservation(
        raw_rgb=bytes((24, 48, 72)) * (8 * 6),
        frame_id=18,
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 5,
            "source_frame_id": 17,
            "scene_hash": "raw-scene",
            "width": 8,
            "height": 6,
        },
        width=8,
        height=6,
        active_window="Editor",
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
        perception_version=5,
    )
    observation = observation.model_copy(update={"screen": screen})

    with pytest.raises(ValueError, match="trustworthy AVCS frame identity"):
        GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="40,30",
            payload={"x": 40, "y": 30},
            expected_outcome="target changed",
            observation=observation,
        )


def test_gui_intent_keeps_perception_bus_and_avcs_frame_versions_distinct():
    observation = _gui_observation(version=7)
    observation.perception_version = 42

    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="40,30",
        payload={"x": 40, "y": 30},
        expected_outcome="target changed",
        observation=observation,
    )

    assert intent.source_perception_version == 42
    assert intent.source_state_version == 7


@pytest.mark.parametrize(
    "screen_updates, observation_updates",
    [
        ({"avcs_state": {}}, {}),
        (
            {
                "avcs_state": {
                    "engine": "other",
                    "frame_index": 7,
                    "scene_hash": "stable-scene",
                    "width": 1280,
                    "height": 800,
                }
            },
            {},
        ),
        (
            {
                "avcs_state": {
                    "engine": "avcs_reflex",
                    "frame_index": 7,
                    "width": 1280,
                    "height": 800,
                }
            },
            {},
        ),
        (
            {
                "avcs_state": {
                    "engine": "avcs_reflex",
                    "frame_index": 7,
                    "scene_hash": "stable-scene",
                    "width": 1280,
                    "height": 800,
                },
                "perception_version": 6,
            },
            {},
        ),
        (
            {
                "avcs_state": {
                    "engine": "avcs_reflex",
                    "frame_index": 7,
                    "scene_hash": "stable-scene",
                    "width": 1024,
                    "height": 768,
                }
            },
            {},
        ),
    ],
)
def test_gui_intent_rejects_missing_or_untrustworthy_avcs_identity(
    screen_updates, observation_updates
):
    observation = _gui_observation()
    updated_screen = observation.screen.model_copy(update=screen_updates)
    observation = observation.model_copy(
        update={"screen": updated_screen, **observation_updates}
    )

    with pytest.raises(ValueError, match="trustworthy AVCS frame identity"):
        GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="Save",
            payload={"x": 50, "y": 60},
            expected_outcome="saved",
            observation=observation,
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
        screenshot_base64="pixel-frame",
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 8,
            "scene_hash": "stable-scene",
            "width": 1280,
            "height": 800,
        },
        width=1280,
        height=800,
        active_window="Different window",
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
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
        screenshot_base64="pixel-frame",
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 8,
            "scene_hash": "stable-scene",
            "width": 1280,
            "height": 800,
        },
        width=1280,
        height=800,
        active_window="Editor",
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
        perception_version=8,
    )
    assert intent.validate_observation(same_window_new_state) == ()
    changed_scene = same_window_new_state.model_copy(
        update={
            "avcs_state": {
                "engine": "avcs_reflex",
                "frame_index": 8,
                "scene_hash": "changed-scene",
                "width": 1280,
                "height": 800,
            }
        }
    )
    assert "AVCS scene differs from GUI action intent" in (
        intent.validate_observation(changed_scene)
    )
    assert "fresh desktop evidence is unavailable" in intent.validate_observation(
        observation.screen,
        now_ns=intent.issued_at_ns + 1,
        elapsed_ms=251,
        max_age_ms=250,
    )


def test_gui_intent_ttl_rounds_age_up_and_rejects_invalid_monotonic_arithmetic():
    observation = _gui_observation(age_ms=240.25)
    received_at_ns = observation.observation_received_monotonic_ns
    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 50, "y": 60},
        expected_outcome="saved",
        observation=observation,
        now_ns=received_at_ns,
        ttl_ms=250,
    )

    assert intent.expires_at_ns - intent.issued_at_ns == 9_750_000
    assert "GUI action intent expired" in intent.validate_binding(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 50, "y": 60},
        now_ns=intent.expires_at_ns,
    )

    future_receipt = observation.model_copy(
        update={"observation_received_monotonic_ns": received_at_ns + 1}
    )
    with pytest.raises(ValueError, match="valid observation timestamp"):
        GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="Save",
            payload={"x": 50, "y": 60},
            expected_outcome="saved",
            observation=future_receipt,
            now_ns=received_at_ns,
        )

    invalid_age = observation.model_copy(update={"observation_age_ms": float("nan")})
    with pytest.raises(ValueError, match="fresh observation"):
        GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="Save",
            payload={"x": 50, "y": 60},
            expected_outcome="saved",
            observation=invalid_age,
            now_ns=received_at_ns,
        )


def test_gui_observation_validation_rejects_stale_or_malformed_avcs_frame():
    observation = _gui_observation()
    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="Save",
        payload={"x": 50, "y": 60},
        expected_outcome="saved",
        observation=observation,
    )
    older_frame = observation.screen.model_copy(
        update={
            "perception_version": 6,
            "avcs_state": {
                "engine": "avcs_reflex",
                "frame_index": 6,
                "scene_hash": "stable-scene",
                "width": 1280,
                "height": 800,
            },
        }
    )
    assert "AVCS frame is older than the GUI action intent" in (
        intent.validate_observation(older_frame)
    )

    malformed_frame = observation.screen.model_copy(
        update={
            "avcs_state": {
                "engine": "avcs_reflex",
                "frame_index": True,
                "scene_hash": "stable-scene",
                "width": 1280,
                "height": 800,
            }
        }
    )
    assert "a trustworthy AVCS frame identity is unavailable" in (
        intent.validate_observation(malformed_frame)
    )


def test_gui_intent_uses_only_avcs_scene_and_active_window():
    observation = _gui_observation()
    intent = GUIActionIntent.create(
        action_type="GUI_CLICK",
        target="40,30",
        payload={"x": 40, "y": 30},
        expected_outcome="Save the document",
        observation=observation,
    )

    assert intent.source_scene_hash == "stable-scene"
    assert not hasattr(intent, "source_focus")
    assert not hasattr(intent, "target_identity")


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
    with pytest.raises(ValueError, match="versioned AVCS frame"):
        GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="Save",
            payload={"x": 50, "y": 60},
            expected_outcome="saved",
            observation=unversioned,
        )
