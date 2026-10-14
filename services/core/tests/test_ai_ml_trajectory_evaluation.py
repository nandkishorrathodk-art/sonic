"""Offline descriptive metrics for validated AI/ML trajectory examples."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from pydantic import ValidationError

from sonic.llm.trajectory import (
    ActionOutcome,
    MissionOutcome,
    RedactionReviewReceipt,
    TrainingExample,
    VerificationStatus,
)
from sonic.mission_engine.trajectory_evaluation import evaluate_training_examples

_DIGEST = "c" * 64
_TENANT_REF = "hmac-sha256:" + "d" * 64
_PRIVACY_KEY = b"test-only-privacy-review-key-32-bytes"


def _artifact(name: str) -> dict:
    return {
        "artifact_id": name,
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
        "review_id": f"privacy-{data['decision_id']}",
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


def _example(
    *,
    decision_id: str,
    episode_id: str,
    outcome: ActionOutcome,
    mission_outcome: MissionOutcome,
    verification_status: VerificationStatus,
    tenant_ref: str = _TENANT_REF,
    latency_ms: float | None = 100.0,
    cost_usd: float | None = 0.01,
    correction: bool = False,
) -> TrainingExample:
    example = {
        "episode_id": episode_id,
        "decision_id": decision_id,
        "tenant_ref": tenant_ref,
        "goal_ref": _artifact(f"goal-{episode_id}"),
        "observation_ref": _artifact(f"observation-{decision_id}"),
        "available_tools": ["GUI_CLICK", "GUI_SCREENSHOT"],
        "model": {
            "system": "llm_router",
            "provider": "nvidia",
            "model_id": "vision-model",
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        },
        "candidates": [
            {"candidate_id": f"candidate-{decision_id}", "action_type": "GUI_CLICK", "selected": True},
        ],
        "selected_action_id": f"candidate-{decision_id}",
        "receipt": {
            "receipt_id": f"receipt-{decision_id}",
            "candidate_id": f"candidate-{decision_id}",
            "outcome": outcome,
            "safety_verdict": "ALLOW",
            "prediction_error": 0.25,
        },
        "mission_outcome": mission_outcome,
        "verification": (
            {
                "status": verification_status,
                "verifier_id": "verifier-1",
                "evidence_ref": _artifact(f"verification-{episode_id}"),
                "independent": True,
            }
            if verification_status != VerificationStatus.NOT_RUN
            else {
                "status": verification_status,
                "independent": False,
            }
        ),
    }
    if correction:
        example["corrections"] = [
            {
                "prior_decision_id": f"prior-{decision_id}",
                "corrective_action_id": f"corrective-{decision_id}",
            }
        ]
    example["privacy_review"] = _privacy_review(example)
    return TrainingExample.model_validate(example)


def test_empty_dataset_reports_unavailable_rates_not_a_fake_baseline():
    report = evaluate_training_examples([])

    assert report.example_count == 0
    assert report.episode_count == 0
    assert report.action_success_rate.value is None
    assert report.verified_mission_success_rate.value is None
    assert report.model_latency_ms.mean is None


def test_baseline_uses_unique_episodes_for_mission_metrics():
    examples = [
        _example(
            decision_id="decision-1",
            episode_id="episode-success",
            outcome=ActionOutcome.FAILED,
            mission_outcome=MissionOutcome.SUCCESS,
            verification_status=VerificationStatus.PASSED,
            latency_ms=100,
            cost_usd=0.02,
            correction=True,
        ),
        _example(
            decision_id="decision-2",
            episode_id="episode-success",
            outcome=ActionOutcome.SUCCEEDED,
            mission_outcome=MissionOutcome.SUCCESS,
            verification_status=VerificationStatus.PASSED,
            latency_ms=200,
            cost_usd=None,
        ),
        _example(
            decision_id="decision-3",
            episode_id="episode-failed",
            outcome=ActionOutcome.BLOCKED,
            mission_outcome=MissionOutcome.FAILED,
            verification_status=VerificationStatus.INCONCLUSIVE,
            latency_ms=None,
            cost_usd=0.04,
        ),
    ]

    report = evaluate_training_examples(examples)

    assert report.example_count == 3
    assert report.episode_count == 2
    assert report.action_outcomes == {"BLOCKED": 1, "FAILED": 1, "SUCCEEDED": 1}
    assert report.mission_outcomes == {"FAILED": 1, "SUCCESS": 1}
    assert report.action_success_rate.numerator == 1
    assert report.action_success_rate.denominator == 3
    assert report.verified_mission_success_rate.numerator == 1
    assert report.verified_mission_success_rate.denominator == 2
    assert report.independent_verification_coverage.value == 1.0
    assert report.correction_rate.value == pytest.approx(1 / 3, abs=1e-4)
    assert report.prediction_error.mean == 0.25
    assert report.model_latency_ms.count == 2
    assert report.model_latency_ms.mean == 150
    assert report.model_cost_usd.count == 2
    assert report.model_cost_usd.mean == 0.03
    assert report.verified_mission_success_rate.confidence_95 is not None
    assert report.verified_mission_success_rate.confidence_95[0] < 0.5
    assert report.verified_mission_success_rate.confidence_95[1] > 0.5


def test_failed_independent_verification_counts_as_coverage_not_success():
    example = _example(
        decision_id="decision-failed-verification",
        episode_id="episode-failed-verification",
        outcome=ActionOutcome.SUCCEEDED,
        mission_outcome=MissionOutcome.FAILED,
        verification_status=VerificationStatus.FAILED,
    )

    report = evaluate_training_examples([example])

    assert report.independent_verification_coverage.value == 1.0
    assert report.verified_mission_success_rate.value == 0.0


def test_episode_id_cannot_merge_outcomes_across_tenants():
    examples = [
        _example(
            decision_id=f"decision-tenant-{index}",
            episode_id="shared-episode-id",
            tenant_ref=f"hmac-sha256:{digit * 64}",
            outcome=ActionOutcome.SUCCEEDED,
            mission_outcome=MissionOutcome.SUCCESS,
            verification_status=VerificationStatus.PASSED,
        )
        for index, digit in enumerate(("e", "f"), start=1)
    ]

    with pytest.raises(ValueError, match="exactly one tenant"):
        evaluate_training_examples(examples)


def test_inconsistent_final_outcomes_for_one_episode_are_rejected():
    examples = [
        _example(
            decision_id="decision-failed",
            episode_id="episode-conflict",
            outcome=ActionOutcome.FAILED,
            mission_outcome=MissionOutcome.FAILED,
            verification_status=VerificationStatus.NOT_RUN,
        ),
        _example(
            decision_id="decision-success",
            episode_id="episode-conflict",
            outcome=ActionOutcome.SUCCEEDED,
            mission_outcome=MissionOutcome.SUCCESS,
            verification_status=VerificationStatus.PASSED,
        ),
    ]

    with pytest.raises(ValueError, match="inconsistent mission_outcome"):
        evaluate_training_examples(examples)


def test_unmeasured_latency_and_cost_are_not_coerced_to_zero():
    example = _example(
        decision_id="decision-no-telemetry",
        episode_id="episode-no-telemetry",
        outcome=ActionOutcome.UNKNOWN,
        mission_outcome=MissionOutcome.UNKNOWN,
        verification_status=VerificationStatus.NOT_RUN,
        latency_ms=None,
        cost_usd=None,
    )

    report = evaluate_training_examples([example])

    assert report.model_latency_ms.count == 0
    assert report.model_latency_ms.mean is None
    assert report.model_cost_usd.count == 0
    assert report.model_cost_usd.mean is None


def test_privacy_review_receipt_requires_a_digest_signature():
    with pytest.raises(ValidationError, match="signature_sha256"):
        RedactionReviewReceipt.model_validate(
            {
                "review_id": "review-1",
                "policy_version": "redaction-v1",
                "reviewer_id": "gate-1",
                "reviewed_artifacts": {"artifact-1": _DIGEST},
                "source_artifact_digests": {"artifact-1": _DIGEST},
                "passed": True,
            }
        )
