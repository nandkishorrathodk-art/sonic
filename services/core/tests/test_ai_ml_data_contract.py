"""Validation tests for the privacy-conscious offline-learning data contract."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from pydantic import ValidationError

from sonic.llm.trajectory import (
    ActionCandidate,
    ActionOutcome,
    ActionReceiptReference,
    DataArtifactReference,
    IndependentVerification,
    MissionOutcome,
    ModelProvenance,
    SafetyVerdict,
    TrainingExample,
    VerificationStatus,
)

_DIGEST = "a" * 64
_TENANT_REF = "hmac-sha256:" + "b" * 64
_PRIVACY_KEY = b"test-only-privacy-review-key-32-bytes"


def _ref(artifact_id: str) -> dict:
    return {
        "artifact_id": artifact_id,
        "sha256": _DIGEST,
        "redaction_status": "REDACTED",
    }


def _privacy_review(data: dict) -> dict:
    reviewed = {}

    def collect(value):
        if isinstance(value, dict):
            if "artifact_id" in value and "sha256" in value:
                reviewed[value["artifact_id"]] = value["sha256"]
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(data)
    receipt = {
        "review_id": "privacy-review-1",
        "policy_version": "redaction-policy-v1",
        "reviewer_id": "privacy-gate-1",
        "reviewed_artifacts": reviewed,
        "source_artifact_digests": reviewed.copy(),
        "passed": True,
        "reviewed_at": "2026-09-28T12:00:00+00:00",
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    receipt["signature_sha256"] = hmac.new(
        _PRIVACY_KEY, payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return receipt


def _example(**overrides) -> dict:
    data = {
        "episode_id": "episode-1",
        "decision_id": "decision-1",
        "tenant_ref": _TENANT_REF,
        "goal_ref": _ref("goal-redacted-1"),
        "observation_ref": _ref("observation-1"),
        "available_tools": ["GUI_CLICK", "GUI_SCREENSHOT"],
        "model": {
            "system": "llm_router",
            "provider": "nvidia",
            "model_id": "vision-model",
            "prompt_id": "computer-use",
            "prompt_version": "v3",
            "task_type": "vision",
            "latency_ms": 125.0,
            "cost_usd": 0.002,
        },
        "candidates": [
            {"candidate_id": "candidate-click", "action_type": "GUI_CLICK", "selected": True},
            {"candidate_id": "candidate-observe", "action_type": "GUI_SCREENSHOT"},
        ],
        "selected_action_id": "candidate-click",
        "receipt": {
            "receipt_id": "receipt-1",
            "candidate_id": "candidate-click",
            "outcome": ActionOutcome.FAILED,
            "safety_verdict": SafetyVerdict.ALLOW,
            "evidence_ref": _ref("receipt-evidence-1"),
            "state_changed": False,
        },
        "corrections": [
            {
                "prior_decision_id": "decision-1",
                "corrective_action_id": "decision-2",
                "reason_ref": _ref("correction-reason-1"),
            }
        ],
        "mission_outcome": MissionOutcome.SUCCESS,
        "verification": {
            "status": VerificationStatus.PASSED,
            "verifier_id": "independent-verifier-1",
            "evidence_ref": _ref("verification-1"),
            "independent": True,
        },
    }
    data.update(overrides)
    data.setdefault("privacy_review", _privacy_review(data))
    return data


def test_training_example_contract_links_decision_and_verified_outcome():
    example = TrainingExample.model_validate(_example())

    assert example.schema_version == "sonic.training-example/v1"
    assert example.candidates[0].candidate_id == example.selected_action_id
    assert example.receipt.outcome == ActionOutcome.FAILED
    assert len(example.corrections) == 1
    assert example.verification.independent

    serialized = example.model_dump(mode="json")
    assert "goal" not in serialized
    assert "observation" not in serialized
    assert "payload" not in serialized
    assert "reasoning" not in serialized
    assert serialized["goal_ref"]["sha256"] == _DIGEST


def test_contract_rejects_unlinked_selected_action_and_duplicate_candidates():
    no_selected = _example(
        candidates=[
            {"candidate_id": "candidate-click", "action_type": "GUI_CLICK"},
            {"candidate_id": "candidate-observe", "action_type": "GUI_SCREENSHOT"},
        ]
    )
    with pytest.raises(ValidationError, match="selected_action_id"):
        TrainingExample.model_validate(no_selected)

    duplicate_ids = _example(
        candidates=[
            {"candidate_id": "candidate-click", "action_type": "GUI_CLICK", "selected": True},
            {"candidate_id": "candidate-click", "action_type": "GUI_SCREENSHOT"},
        ]
    )
    with pytest.raises(ValidationError, match="candidate_id values must be unique"):
        TrainingExample.model_validate(duplicate_ids)


def test_contract_rejects_unverified_success_and_unsubstantiated_verification():
    with pytest.raises(ValidationError, match="successful mission outcome"):
        TrainingExample.model_validate(
            _example(
                verification={
                    "status": VerificationStatus.INCONCLUSIVE,
                    "independent": False,
                }
            )
        )

    with pytest.raises(ValidationError, match="independent verifier and evidence"):
        IndependentVerification.model_validate(
            {"status": VerificationStatus.PASSED, "independent": True}
        )


def test_contract_rejects_raw_content_fields_and_invalid_evidence_digests():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ActionCandidate.model_validate(
            {"candidate_id": "candidate-1", "action_type": "GUI_CLICK", "payload": "password=secret"}
        )

    with pytest.raises(ValidationError):
        DataArtifactReference.model_validate(
            {
                "artifact_id": "C:\\private\\screenshot.png",
                "sha256": "not-a-digest",
                "redaction_status": "REDACTED",
            }
        )

    with pytest.raises(ValidationError):
        ModelProvenance.model_validate({"system": "router", "latency_ms": -1})


def test_contract_requires_pseudonymous_tenant_and_explicit_redaction_attestation():
    with pytest.raises(ValidationError):
        TrainingExample.model_validate(_example(tenant_ref="tenant-123"))

    unreviewed_goal = _example(goal_ref={"artifact_id": "goal-1", "sha256": _DIGEST})
    with pytest.raises(ValidationError):
        TrainingExample.model_validate(unreviewed_goal)

    mismatched_receipt = _example(
        receipt={
            "receipt_id": "receipt-1",
            "candidate_id": "candidate-observe",
            "outcome": ActionOutcome.FAILED,
            "safety_verdict": SafetyVerdict.ALLOW,
        }
    )
    with pytest.raises(ValidationError, match="receipt candidate_id"):
        TrainingExample.model_validate(mismatched_receipt)


def test_contract_can_represent_policy_blocks_without_relabeling_them_as_success():
    receipt = ActionReceiptReference.model_validate(
        {
            "receipt_id": "receipt-blocked",
            "candidate_id": "candidate-blocked",
            "outcome": ActionOutcome.BLOCKED,
            "safety_verdict": SafetyVerdict.BLOCK,
            "exit_code": 126,
        }
    )

    assert receipt.outcome == ActionOutcome.BLOCKED
    assert receipt.safety_verdict == SafetyVerdict.BLOCK
    assert receipt.exit_code == 126
