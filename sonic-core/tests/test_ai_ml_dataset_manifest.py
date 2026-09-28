"""Tests for the in-memory, privacy-attested dataset freeze gate."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from sonic.llm.trajectory import TrainingExample
from sonic.mission_engine.dataset_manifest import freeze_dataset
from sonic.mission_engine.privacy_gate import HmacPrivacyReviewAuthenticator

_PRIVACY_KEY = b"test-only-privacy-review-key-32-bytes"
_AUTHENTICATOR = HmacPrivacyReviewAuthenticator(lambda: _PRIVACY_KEY)
_DIGEST = "e" * 64


def _artifact(artifact_id: str) -> dict:
    return {
        "artifact_id": artifact_id,
        "sha256": _DIGEST,
        "redaction_status": "REDACTED",
    }


def _example(episode_number: int, tenant_number: int) -> TrainingExample:
    decision_id = f"decision-{episode_number:03}"
    candidate_id = f"candidate-{episode_number:03}"
    data = {
        "episode_id": f"episode-{episode_number:03}",
        "decision_id": decision_id,
        "tenant_ref": "hmac-sha256:" + f"{tenant_number:064x}",
        "goal_ref": _artifact(f"goal-{episode_number:03}"),
        "observation_ref": _artifact(f"observation-{episode_number:03}"),
        "available_tools": ["GUI_CLICK"],
        "model": {"system": "llm_router", "latency_ms": 10.0, "cost_usd": 0.001},
        "candidates": [
            {
                "candidate_id": candidate_id,
                "action_type": "GUI_CLICK",
                "selected": True,
            }
        ],
        "selected_action_id": candidate_id,
        "receipt": {
            "receipt_id": f"receipt-{episode_number:03}",
            "candidate_id": candidate_id,
            "outcome": "SUCCEEDED",
            "safety_verdict": "ALLOW",
        },
        "mission_outcome": "SUCCESS",
        "verification": {
            "status": "PASSED",
            "verifier_id": "independent-verifier",
            "evidence_ref": _artifact(f"verification-{episode_number:03}"),
            "independent": True,
        },
    }
    artifacts = {
        value["artifact_id"]: value["sha256"]
        for value in (
            data["goal_ref"],
            data["observation_ref"],
            data["verification"]["evidence_ref"],
        )
    }
    review = {
        "review_id": f"review-{episode_number:03}",
        "policy_version": "redaction-v1",
        "reviewer_id": "trusted-gate",
        "reviewed_artifacts": artifacts,
        "source_artifact_digests": artifacts.copy(),
        "passed": True,
        "reviewed_at": "2026-09-28T12:00:00Z",
    }
    payload = json.dumps(review, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    review["signature_sha256"] = hmac.new(
        _PRIVACY_KEY,
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    data["privacy_review"] = review
    return TrainingExample.model_validate(data)


def _dataset() -> list[TrainingExample]:
    # 60 tenants with five episodes each permits exact 150/75/75 tenant groups.
    return [
        _example(episode_number, tenant_number=(episode_number - 1) // 5)
        for episode_number in range(1, 301)
    ]


def test_freeze_is_deterministic_and_keeps_each_tenant_in_one_split():
    examples = _dataset()
    first = freeze_dataset(
        examples,
        seed="baseline-2026-09",
        privacy_review_verifier=_AUTHENTICATOR,
    )
    second = freeze_dataset(
        examples,
        seed="baseline-2026-09",
        privacy_review_verifier=_AUTHENTICATOR,
    )

    assert first.dataset_sha256 == second.dataset_sha256
    assert first.split_sha256 == second.split_sha256
    assert first.example_count == 300
    assert first.episode_count == 300
    assert first.tenant_count == 60
    assert first.episode_counts == {"train": 150, "validation": 75, "holdout": 75}
    assert first.tenant_counts == {"train": 30, "validation": 15, "holdout": 15}

    assignments = {entry.episode_id: entry.split for entry in first.assignments}
    tenants: dict[str, set] = {}
    for example in examples:
        tenants.setdefault(example.tenant_ref, set()).add(assignments[example.episode_id])
    assert all(len(splits) == 1 for splits in tenants.values())


def test_freeze_rejects_small_dataset_and_too_few_tenant_groups():
    with pytest.raises(ValueError, match="at least 300 unique episodes"):
        freeze_dataset(
            _dataset()[:299],
            seed="seed",
            privacy_review_verifier=_AUTHENTICATOR,
        )

    one_tenant = [_example(index, tenant_number=1) for index in range(1, 301)]
    with pytest.raises(ValueError, match="at least 3 distinct tenant groups"):
        freeze_dataset(
            one_tenant,
            seed="seed",
            privacy_review_verifier=_AUTHENTICATOR,
        )


def test_freeze_rejects_tampered_review_and_short_verification_key():
    examples = _dataset()
    tampered_review = examples[0].privacy_review.model_dump(mode="json")
    tampered_review["signature_sha256"] = "f" * 64
    tampered_data = examples[0].model_dump(mode="json")
    tampered_data["privacy_review"] = tampered_review
    tampered_example = TrainingExample.model_validate(tampered_data)
    with pytest.raises(ValueError, match="invalid privacy review signature"):
        freeze_dataset(
            [tampered_example, *examples[1:]],
            seed="seed",
            privacy_review_verifier=_AUTHENTICATOR,
        )

    weak_authenticator = HmacPrivacyReviewAuthenticator(lambda: b"short")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        freeze_dataset(
            examples,
            seed="seed",
            privacy_review_verifier=weak_authenticator,
        )


def test_freeze_rejects_duplicate_decisions():
    examples = _dataset()
    duplicate = examples[0].model_copy(update={"episode_id": "episode-duplicate"})
    with pytest.raises(ValueError, match="duplicate decision_id"):
        freeze_dataset(
            [*examples, duplicate],
            seed="seed",
            privacy_review_verifier=_AUTHENTICATOR,
        )
