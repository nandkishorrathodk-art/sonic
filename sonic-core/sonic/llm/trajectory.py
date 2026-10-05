"""Versioned, privacy-conscious contract for offline agent-learning examples.

This module defines data shape only. It does not collect, persist, train on, or
promote examples. Content is referenced by opaque IDs and digests so raw prompts,
screenshots, tool payloads, and execution output are not stored in this contract.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
_TENANT_REF = re.compile(r"^hmac-sha256:[a-fA-F0-9]{64}$")
_ACTION_IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]{0,79}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,159}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    TIMED_OUT = "TIMED_OUT"
    RECOVERED = "RECOVERED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class VerificationStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_RUN = "NOT_RUN"


class MissionOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"
    UNKNOWN = "UNKNOWN"


class SafetyVerdict(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    NOT_EVALUATED = "NOT_EVALUATED"


class DataArtifactReference(ContractModel):
    """Reference to reviewed/redacted content stored outside the example."""

    artifact_id: str = Field(pattern=_OPAQUE_ID.pattern)
    sha256: str = Field(pattern=_SHA256.pattern)
    redaction_status: Literal["REDACTED"]


class ActionCandidate(ContractModel):
    candidate_id: str = Field(pattern=_OPAQUE_ID.pattern)
    action_type: str = Field(pattern=_ACTION_IDENTIFIER.pattern)
    payload_ref: DataArtifactReference | None = None
    selected: bool = False


class ModelProvenance(ContractModel):
    """Model/routing metadata; unknown values remain explicit rather than inferred."""

    system: str = Field(pattern=_SAFE_LABEL.pattern)
    provider: str | None = Field(default=None, pattern=_SAFE_LABEL.pattern)
    model_id: str | None = Field(default=None, pattern=_SAFE_LABEL.pattern)
    prompt_id: str | None = Field(default=None, pattern=_SAFE_LABEL.pattern)
    prompt_version: str | None = Field(default=None, pattern=_SAFE_LABEL.pattern)
    task_type: str | None = Field(default=None, pattern=_SAFE_LABEL.pattern)
    request_id: str | None = Field(default=None, pattern=_OPAQUE_ID.pattern)
    latency_ms: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class ActionReceiptReference(ContractModel):
    receipt_id: str = Field(pattern=_OPAQUE_ID.pattern)
    candidate_id: str = Field(pattern=_OPAQUE_ID.pattern)
    outcome: ActionOutcome
    exit_code: int | None = None
    safety_verdict: SafetyVerdict
    predicted_outcome_ref: DataArtifactReference | None = None
    evidence_ref: DataArtifactReference | None = None
    state_changed: bool | None = None
    prediction_error: float | None = Field(default=None, ge=0, le=1)


class CorrectionReference(ContractModel):
    """A correction links decisions without storing model reasoning or raw text."""

    prior_decision_id: str = Field(pattern=_OPAQUE_ID.pattern)
    corrective_action_id: str = Field(pattern=_OPAQUE_ID.pattern)
    reason_ref: DataArtifactReference | None = None


class IndependentVerification(ContractModel):
    status: VerificationStatus
    verifier_id: str | None = Field(default=None, max_length=120)
    evidence_ref: DataArtifactReference | None = None
    independent: bool = False
    verified_at: datetime | None = None

    @model_validator(mode="after")
    def require_evidence_for_pass(self) -> IndependentVerification:
        if self.status == VerificationStatus.PASSED:
            if not self.independent or not self.verifier_id or not self.evidence_ref:
                raise ValueError(
                    "passed verification requires an independent verifier and evidence reference"
                )
        return self


class RedactionReviewReceipt(ContractModel):
    """Attestation from a separate gate that inspected every referenced artifact."""

    review_id: str = Field(pattern=_OPAQUE_ID.pattern)
    policy_version: str = Field(pattern=_SAFE_LABEL.pattern)
    reviewer_id: str = Field(pattern=_OPAQUE_ID.pattern)
    reviewed_artifacts: dict[str, str] = Field(min_length=1)
    source_artifact_digests: dict[str, str] = Field(min_length=1)
    passed: Literal[True]
    signature_sha256: str = Field(pattern=_SHA256.pattern)
    reviewed_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def validate_reviewed_digests(self) -> RedactionReviewReceipt:
        if any(
            not _OPAQUE_ID.fullmatch(artifact_id)
            or not _SHA256.fullmatch(digest)
            for artifacts in (self.reviewed_artifacts, self.source_artifact_digests)
            for artifact_id, digest in artifacts.items()
        ):
            raise ValueError("artifact attestation maps must use opaque IDs and SHA-256 digests")
        if self.reviewed_artifacts.keys() != self.source_artifact_digests.keys():
            raise ValueError(
                "source_artifact_digests must map each redacted artifact to its source digest"
            )
        return self


class TrainingExample(ContractModel):
    """One action decision with linked execution and mission-level outcomes."""

    schema_version: Literal["sonic.training-example/v1"] = "sonic.training-example/v1"
    episode_id: str = Field(pattern=_OPAQUE_ID.pattern)
    decision_id: str = Field(pattern=_OPAQUE_ID.pattern)
    parent_episode_id: str | None = Field(default=None, pattern=_OPAQUE_ID.pattern)
    tenant_ref: str = Field(
        pattern=_TENANT_REF.pattern,
        description="HMAC-SHA256 tenant pseudonym; never store the raw tenant identifier.",
    )
    occurred_at: datetime = Field(default_factory=_utcnow)

    goal_ref: DataArtifactReference
    observation_ref: DataArtifactReference
    available_tools: list[str] = Field(min_length=1)
    model: ModelProvenance
    candidates: list[ActionCandidate] = Field(min_length=1)
    selected_action_id: str = Field(pattern=_OPAQUE_ID.pattern)
    receipt: ActionReceiptReference
    corrections: list[CorrectionReference] = Field(default_factory=list)
    mission_outcome: MissionOutcome
    verification: IndependentVerification
    privacy_review: RedactionReviewReceipt

    @model_validator(mode="after")
    def validate_decision_links(self) -> TrainingExample:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique within a decision")

        selected = [
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_action_id
        ]
        if len(selected) != 1 or not selected[0].selected:
            raise ValueError("selected_action_id must identify the selected candidate")
        if sum(candidate.selected for candidate in self.candidates) != 1:
            raise ValueError("exactly one candidate must be selected")
        if self.receipt.candidate_id != self.selected_action_id:
            raise ValueError("receipt candidate_id must match selected_action_id")

        artifact_refs = [
            self.goal_ref,
            self.observation_ref,
            *(candidate.payload_ref for candidate in self.candidates),
            self.receipt.predicted_outcome_ref,
            self.receipt.evidence_ref,
            *(correction.reason_ref for correction in self.corrections),
            self.verification.evidence_ref,
        ]
        required_digests = {
            ref.artifact_id: ref.sha256
            for ref in artifact_refs
            if ref is not None
        }
        if self.privacy_review.reviewed_artifacts != required_digests:
            raise ValueError(
                "privacy review must cover every referenced artifact with its matching digest"
            )

        tool_ids = [tool.strip() for tool in self.available_tools]
        if (
            any(not _ACTION_IDENTIFIER.fullmatch(tool) for tool in tool_ids)
            or len(tool_ids) != len(set(tool_ids))
        ):
            raise ValueError("available_tools must contain unique action/tool identifiers")
        correction_ids = [item.corrective_action_id for item in self.corrections]
        if len(correction_ids) != len(set(correction_ids)):
            raise ValueError("corrective_action_id values must be unique")

        if self.mission_outcome == MissionOutcome.SUCCESS:
            if self.verification.status != VerificationStatus.PASSED:
                raise ValueError("successful mission outcome requires passed verification")
        return self
