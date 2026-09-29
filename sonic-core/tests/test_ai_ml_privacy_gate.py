"""Focused tests for trusted artifact inspection and redaction."""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from sonic.llm.trajectory import TrainingExample
from sonic.mission_engine.privacy_gate import (
    ArtifactContent,
    HmacPrivacyReviewAuthenticator,
    OCRDetection,
    OCRResult,
    PrivacyGateError,
    TrustedPrivacyRedactionGate,
    canonical_review_payload,
)

_KEY = b"external-test-key-for-redaction-gate-32-bytes"


class MemoryResolver:
    def __init__(self, artifacts: dict[str, ArtifactContent]) -> None:
        self.artifacts = artifacts

    def read(self, artifact_id: str) -> ArtifactContent:
        return self.artifacts[artifact_id]


class ScreenshotOCR:
    def inspect(self, image: object) -> OCRResult:
        assert isinstance(image, Image.Image)
        if image.getpixel((20, 20)) == (0, 0, 0):
            return OCRResult(detections=(), complete=True)
        return OCRResult(
            detections=(
                OCRDetection(
                    text="password: hunter2",
                    box=(5, 5, 45, 35),
                    confidence=0.99,
                ),
            ),
            complete=True,
        )


def _artifact_ref(artifact_id: str, content: bytes) -> dict:
    return {
        "artifact_id": artifact_id,
        "sha256": hashlib.sha256(content).hexdigest(),
        "redaction_status": "REDACTED",
    }


def _example(artifacts: dict[str, ArtifactContent]) -> TrainingExample:
    refs = {
        artifact_id: _artifact_ref(artifact_id, item.data)
        for artifact_id, item in artifacts.items()
    }
    data = {
        "episode_id": "episode-privacy-test",
        "decision_id": "decision-privacy-test",
        "tenant_ref": "hmac-sha256:" + "a" * 64,
        "goal_ref": refs["goal"],
        "observation_ref": refs["observation"],
        "available_tools": ["GUI_CLICK"],
        "model": {"system": "llm_router"},
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "action_type": "GUI_CLICK",
                "selected": True,
                **(
                    {"payload_ref": refs["candidate-payload"]}
                    if "candidate-payload" in refs
                    else {}
                ),
            }
        ],
        "selected_action_id": "candidate-1",
        "receipt": {
            "receipt_id": "receipt-1",
            "candidate_id": "candidate-1",
            "outcome": "FAILED",
            "safety_verdict": "ALLOW",
            **(
                {"predicted_outcome_ref": refs["predicted-outcome"]}
                if "predicted-outcome" in refs
                else {}
            ),
            **(
                {"evidence_ref": refs["receipt-evidence"]}
                if "receipt-evidence" in refs
                else {}
            ),
        },
        "corrections": (
            [
                {
                    "prior_decision_id": "decision-prior",
                    "corrective_action_id": "action-correction",
                    "reason_ref": refs["correction-reason"],
                }
            ]
            if "correction-reason" in refs
            else []
        ),
        "mission_outcome": "FAILED",
        "verification": {
            "status": "NOT_RUN",
            "independent": False,
            **(
                {"evidence_ref": refs["verification-evidence"]}
                if "verification-evidence" in refs
                else {}
            ),
        },
        "privacy_review": {
            "review_id": "untrusted-review",
            "policy_version": "untrusted",
            "reviewer_id": "untrusted",
            "reviewed_artifacts": {
                item["artifact_id"]: item["sha256"] for item in refs.values()
            },
            "source_artifact_digests": {
                item["artifact_id"]: item["sha256"] for item in refs.values()
            },
            "passed": True,
            "signature_sha256": "0" * 64,
        },
    }
    return TrainingExample.model_validate(data)


def _gate(
    artifacts: dict[str, ArtifactContent],
    *,
    ocr: ScreenshotOCR | None = None,
    key: bytes = _KEY,
) -> TrustedPrivacyRedactionGate:
    return TrustedPrivacyRedactionGate(
        resolver=MemoryResolver(artifacts),
        signer=HmacPrivacyReviewAuthenticator(lambda: key),
        reviewer_id="trusted-redaction-gate",
        policy_version="redaction-v1",
        ocr_inspector=ocr,
    )


@pytest.mark.parametrize(
    "payload,secret",
    [
        (b"password=correct-horse-123", b"correct-horse-123"),
        (b"Authorization: Bearer abcdefghijklmnopqrstuvwxyz", b"abcdefghijklmnopqrstuvwxyz"),
        (b"Cookie: sessionid=very-secret-cookie-value", b"very-secret-cookie-value"),
        (
            b'{"name":"sessionid","value":"json-cookie-secret"}',
            b"json-cookie-secret",
        ),
        (
            b'{"password":"a quoted credential with spaces"}',
            b"a quoted credential with spaces",
        ),
        (
            b"access_token=eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoxfQ.signature-value-123",
            b"eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoxfQ.signature-value-123",
        ),
    ],
)
def test_text_artifacts_redact_credentials_tokens_and_cookies(payload: bytes, secret: bytes):
    artifacts = {
        "goal": ArtifactContent(b"Inspect the supplied state.", "text/plain"),
        "observation": ArtifactContent(payload, "text/plain"),
    }
    result = _gate(artifacts).inspect_and_redact(_example(artifacts))

    redacted = {item.artifact_id: item for item in result.artifacts}
    observation_id = result.example.observation_ref.artifact_id
    assert secret not in redacted[observation_id].data
    assert b"[REDACTED]" in redacted[observation_id].data
    assert result.example.privacy_review.reviewed_artifacts == {
        item.artifact_id: item.sha256 for item in result.artifacts
    }
    review = result.example.privacy_review
    assert HmacPrivacyReviewAuthenticator(lambda: _KEY).verify(
        canonical_review_payload(review),
        review.signature_sha256,
    )
    assert set(review.source_artifact_digests.values()) == {
        hashlib.sha256(content.data).hexdigest() for content in artifacts.values()
    }
    tampered_review = review.model_copy(
        update={
            "source_artifact_digests": {
                artifact_id: "f" * 64
                for artifact_id in review.source_artifact_digests
            }
        }
    )
    assert not HmacPrivacyReviewAuthenticator(lambda: _KEY).verify(
        canonical_review_payload(tampered_review),
        review.signature_sha256,
    )


def test_terminal_output_secrets_are_redacted_before_attestation():
    terminal_output = b"service started\nSONIC_API_KEY=sk-1234567890abcdefghijklmnop\n"
    artifacts = {
        "goal": ArtifactContent(b"Read terminal result", "text/plain"),
        "observation": ArtifactContent(terminal_output, "text/plain"),
    }
    result = _gate(artifacts).inspect_and_redact(_example(artifacts))
    output = next(
        item.data
        for item in result.artifacts
        if item.artifact_id == result.example.observation_ref.artifact_id
    )

    assert b"sk-1234567890abcdefghijklmnop" not in output
    assert b"SONIC_API_KEY=[REDACTED]" in output


def test_every_training_example_artifact_reference_is_resolved_and_replaced():
    artifacts = {
        name: ArtifactContent(f"safe {name} content".encode(), "text/plain")
        for name in (
            "goal",
            "observation",
            "candidate-payload",
            "predicted-outcome",
            "receipt-evidence",
            "correction-reason",
            "verification-evidence",
        )
    }
    source_example = _example(artifacts)

    result = _gate(artifacts).inspect_and_redact(source_example)
    updated = result.example
    updated_refs = [
        updated.goal_ref,
        updated.observation_ref,
        updated.candidates[0].payload_ref,
        updated.receipt.predicted_outcome_ref,
        updated.receipt.evidence_ref,
        updated.corrections[0].reason_ref,
        updated.verification.evidence_ref,
    ]

    assert len(result.artifacts) == len(artifacts)
    assert all(ref is not None and ref.artifact_id.startswith("redacted-") for ref in updated_refs)
    redacted_by_id = {artifact.artifact_id: artifact for artifact in result.artifacts}
    assert all(
        reference.sha256 == redacted_by_id[reference.artifact_id].sha256
        for reference in updated_refs
        if reference is not None
    )
    assert set(updated.privacy_review.source_artifact_digests.values()) == {
        hashlib.sha256(artifact.data).hexdigest() for artifact in artifacts.values()
    }


def test_screenshot_secret_is_ocr_detected_redacted_and_rechecked():
    image_bytes = io.BytesIO()
    Image.new("RGB", (80, 50), "white").save(image_bytes, format="PNG")
    screenshot = image_bytes.getvalue()
    artifacts = {
        "goal": ArtifactContent(b"Inspect screenshot", "text/plain"),
        "observation": ArtifactContent(screenshot, "image/png"),
    }
    result = _gate(artifacts, ocr=ScreenshotOCR()).inspect_and_redact(_example(artifacts))
    output = next(
        item
        for item in result.artifacts
        if item.artifact_id == result.example.observation_ref.artifact_id
    )
    with Image.open(io.BytesIO(output.data)) as redacted:
        assert redacted.getpixel((20, 20)) == (0, 0, 0)
        assert redacted.info == {}


def test_screenshot_is_rejected_without_complete_ocr_or_inspector():
    image_bytes = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(image_bytes, format="PNG")
    screenshot = image_bytes.getvalue()
    artifacts = {
        "goal": ArtifactContent(b"Inspect screenshot", "text/plain"),
        "observation": ArtifactContent(screenshot, "image/png"),
    }

    with pytest.raises(PrivacyGateError, match="requires a trusted OCR"):
        _gate(artifacts).inspect_and_redact(_example(artifacts))

    class IncompleteOCR:
        def inspect(self, image: object) -> OCRResult:
            return OCRResult(detections=(), complete=False)

    with pytest.raises(PrivacyGateError, match="incomplete"):
        _gate(artifacts, ocr=IncompleteOCR()).inspect_and_redact(_example(artifacts))

    class LowConfidenceOCR:
        def inspect(self, image: object) -> OCRResult:
            return OCRResult(
                detections=(OCRDetection("password: secret", (1, 1, 10, 10), 0.2),),
                complete=True,
            )

    with pytest.raises(PrivacyGateError, match="confidence"):
        _gate(artifacts, ocr=LowConfidenceOCR()).inspect_and_redact(_example(artifacts))


def test_gate_rejects_digest_mismatch_unsupported_content_and_weak_keys():
    artifacts = {
        "goal": ArtifactContent(b"Inspect supplied state.", "text/plain"),
        "observation": ArtifactContent(b"safe content", "text/plain"),
    }
    example = _example(artifacts)
    changed_source = {
        **artifacts,
        "observation": ArtifactContent(b"changed source bytes", "text/plain"),
    }
    with pytest.raises(PrivacyGateError, match="digest mismatch"):
        _gate(changed_source).inspect_and_redact(example)

    unsupported = {
        "goal": ArtifactContent(b"Inspect supplied state.", "text/plain"),
        "observation": ArtifactContent(b"%PDF-1.7", "application/pdf"),
    }
    with pytest.raises(PrivacyGateError, match="unsupported artifact media type"):
        _gate(unsupported).inspect_and_redact(_example(unsupported))

    with pytest.raises(PrivacyGateError, match="at least 32 bytes"):
        _gate(artifacts, key=b"too-short").inspect_and_redact(_example(artifacts))

    with pytest.raises(PrivacyGateError, match="could not be resolved"):
        _gate({"goal": artifacts["goal"]}).inspect_and_redact(_example(artifacts))
