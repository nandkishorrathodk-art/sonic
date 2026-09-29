"""Trusted, in-memory inspection and redaction for trajectory artifacts."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Protocol
from uuid import uuid4

from sonic.llm.trajectory import (
    DataArtifactReference,
    RedactionReviewReceipt,
    TrainingExample,
)

MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MIN_OCR_CONFIDENCE = 0.8

_SECRET_PATTERNS = (
    re.compile(
        r"""(?ix)
        ["']?\b(?:password|passwd|pwd|secret|credential|credentials|token|
        client[_-]?secret|api[_-]?key|access[_-]?token|refresh[_-]?token|
        auth[_-]?token|session[_-]?token|
        private[_-]?key)\b["']?
        \s*[:=]\s*["'](?P<value>[^"'\r\n]+)["']
        """
    ),
    re.compile(
        r"""(?ix)
        ["']?\b(?:password|passwd|pwd|secret|credential|credentials|token|
        client[_-]?secret|api[_-]?key|access[_-]?token|refresh[_-]?token|
        auth[_-]?token|session[_-]?token|
        private[_-]?key)\b["']?
        \s*[:=]\s*(?P<value>(?!["'])[^\r\n]+)
        """
    ),
    re.compile(r"""(?ix)["']?(?:cookie|set-cookie|session[_-]?id)["']?\s*[:=]\s*["']?(?P<value>[^"'\r\n,;}]+)"""),
    re.compile(
        r"""(?isx)["']name["']\s*:\s*["'](?:sessionid|session[_-]?id|auth[_-]?token)["']\s*,.{0,160}?
        ["']value["']\s*:\s*["'](?P<value>[^"']+)["']"""
    ),
    re.compile(r"(?im)^\s*(?:authorization|proxy-authorization)\s*:\s*\S.*$"),
    re.compile(r"(?im)^\s*(?:cookie|set-cookie)\s*:\s*\S.*$"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b"),
    re.compile(
        r"(?is)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
)
_IMAGE_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
}
_TEXT_MEDIA_TYPES = {
    "application/json",
    "application/x-www-form-urlencoded",
    "application/xml",
    "application/javascript",
    "application/x-ndjson",
}


class PrivacyGateError(ValueError):
    """Artifact inspection or redaction could not be completed safely."""


@dataclass(frozen=True)
class ArtifactContent:
    data: bytes
    media_type: str


@dataclass(frozen=True)
class OCRDetection:
    text: str
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    detections: tuple[OCRDetection, ...]
    complete: bool


@dataclass(frozen=True)
class RedactedArtifact:
    artifact_id: str
    sha256: str
    media_type: str
    data: bytes


@dataclass(frozen=True)
class PrivacyGateResult:
    example: TrainingExample
    artifacts: tuple[RedactedArtifact, ...]


class ArtifactResolver(Protocol):
    def read(self, artifact_id: str) -> ArtifactContent: ...


class OCRInspector(Protocol):
    def inspect(self, image: object) -> OCRResult: ...


class PrivacyReviewSigner(Protocol):
    def sign(self, payload: bytes) -> str: ...


class PrivacyReviewVerifier(Protocol):
    def verify(self, payload: bytes, signature: str) -> bool: ...


class HmacPrivacyReviewAuthenticator:
    """HMAC signer/verifier backed by an external secret provider."""

    def __init__(self, key_provider: Callable[[], bytes]) -> None:
        self._key_provider = key_provider

    def sign(self, payload: bytes) -> str:
        key = self._get_key()
        return hmac.new(key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature)

    def _get_key(self) -> bytes:
        key = self._key_provider()
        if not isinstance(key, bytes) or len(key) < 32:
            raise PrivacyGateError("external privacy signing key must contain at least 32 bytes")
        return key


def canonical_review_payload(review: RedactionReviewReceipt | dict) -> bytes:
    """Serialize a review identically for the trusted signer and freeze verifier."""
    if isinstance(review, RedactionReviewReceipt):
        payload = review.model_dump(mode="json", exclude={"signature_sha256"})
    else:
        payload = {key: value for key, value in review.items() if key != "signature_sha256"}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _references(example: TrainingExample) -> list[DataArtifactReference]:
    return [
        reference
        for reference in (
            example.goal_ref,
            example.observation_ref,
            *(candidate.payload_ref for candidate in example.candidates),
            example.receipt.predicted_outcome_ref,
            example.receipt.evidence_ref,
            *(correction.reason_ref for correction in example.corrections),
            example.verification.evidence_ref,
        )
        if reference is not None
    ]


def _find_sensitive_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span("value") if "value" in match.groupdict() else match.span()
            if text[span[0] : span[1]].casefold() != "[redacted]":
                spans.append(span)
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _redact_text(data: bytes) -> bytes:
    if len(data) > MAX_TEXT_BYTES:
        raise PrivacyGateError("text artifact exceeds the inspection size limit")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PrivacyGateError("artifact is not verifiable UTF-8 text") from exc
    spans = _find_sensitive_spans(text)
    for start, end in reversed(spans):
        text = text[:start] + "[REDACTED]" + text[end:]
    if _find_sensitive_spans(text):
        raise PrivacyGateError("sensitive content remains after text redaction")
    return text.encode("utf-8")


def _validate_ocr_result(result: object, width: int, height: int) -> tuple[OCRDetection, ...]:
    if (
        not isinstance(result, OCRResult)
        or result.complete is not True
        or not isinstance(result.detections, tuple)
    ):
        raise PrivacyGateError("OCR inspection was incomplete")
    for detection in result.detections:
        if (
            not isinstance(detection, OCRDetection)
            or isinstance(detection.confidence, bool)
            or not isinstance(detection.confidence, (int, float))
            or not math.isfinite(detection.confidence)
            or not MIN_OCR_CONFIDENCE <= detection.confidence <= 1.0
            or not isinstance(detection.text, str)
        ):
            raise PrivacyGateError("OCR confidence is below the privacy threshold")
        if (
            not isinstance(detection.box, tuple)
            or len(detection.box) != 4
            or any(type(value) is not int for value in detection.box)
        ):
            raise PrivacyGateError("OCR returned an invalid screenshot region")
        x1, y1, x2, y2 = detection.box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise PrivacyGateError("OCR returned an invalid screenshot region")
    return result.detections


def _redact_image(
    data: bytes,
    media_type: str,
    inspector: OCRInspector | None,
) -> bytes:
    if inspector is None:
        raise PrivacyGateError("screenshot inspection requires a trusted OCR inspector")
    signature_types = [
        kind
        for kind, signatures in _IMAGE_SIGNATURES.items()
        if any(data.startswith(signature) for signature in signatures)
    ]
    if media_type not in _IMAGE_SIGNATURES or signature_types != [media_type]:
        raise PrivacyGateError("screenshot media type does not match its image signature")

    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise PrivacyGateError("Pillow is required to inspect screenshot artifacts") from exc

    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise PrivacyGateError("unsupported screenshot encoding")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise PrivacyGateError("screenshot exceeds the pixel inspection limit")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except PrivacyGateError:
        raise
    except Exception as exc:
        raise PrivacyGateError("screenshot could not be decoded") from exc

    initial = _validate_ocr_result(inspector.inspect(image.copy()), image.width, image.height)
    draw = ImageDraw.Draw(image)
    for detection in initial:
        x1, y1, x2, y2 = detection.box
        if _find_sensitive_spans(detection.text):
            draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))

    final_scan = _validate_ocr_result(
        inspector.inspect(image.copy()),
        image.width,
        image.height,
    )
    if any(
        _find_sensitive_spans(detection.text) for detection in final_scan
    ):
        raise PrivacyGateError("sensitive content remains visible in screenshot")

    output = io.BytesIO()
    output_format = "PNG" if media_type == "image/png" else "JPEG"
    image.save(output, format=output_format, optimize=True)
    return output.getvalue()


def _redact(content: ArtifactContent, inspector: OCRInspector | None) -> tuple[bytes, str]:
    data = content.data
    if not isinstance(data, bytes) or not data:
        raise PrivacyGateError("artifact resolver returned empty or invalid content")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise PrivacyGateError("artifact exceeds the inspection size limit")
    if not isinstance(content.media_type, str):
        raise PrivacyGateError("artifact resolver returned an invalid media type")
    media_type = content.media_type.split(";", 1)[0].strip().lower()
    if media_type in _IMAGE_SIGNATURES:
        return _redact_image(data, media_type, inspector), media_type
    if media_type.startswith("text/") or media_type in _TEXT_MEDIA_TYPES:
        return _redact_text(data), media_type
    raise PrivacyGateError(f"unsupported artifact media type: {media_type or 'missing'}")


def _replace_references(value: object, replacements: dict[tuple[str, str], dict]) -> object:
    if isinstance(value, dict):
        if {"artifact_id", "sha256", "redaction_status"} <= value.keys():
            key = (value["artifact_id"], value["sha256"])
            try:
                return replacements[key]
            except KeyError as exc:
                raise PrivacyGateError("artifact reference was not inspected") from exc
        return {
            key: _replace_references(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_references(child, replacements) for child in value]
    return value


class TrustedPrivacyRedactionGate:
    """Inspect and redact all referenced bytes before issuing an attestation.

    Artifact bytes and the signing key are handled in memory only. The caller
    remains responsible for any separately approved storage of returned bytes.
    """

    def __init__(
        self,
        *,
        resolver: ArtifactResolver,
        signer: PrivacyReviewSigner,
        reviewer_id: str,
        policy_version: str,
        ocr_inspector: OCRInspector | None = None,
    ) -> None:
        self._resolver = resolver
        self._signer = signer
        self._reviewer_id = reviewer_id
        self._policy_version = policy_version
        self._ocr_inspector = ocr_inspector

    def inspect_and_redact(self, example: TrainingExample) -> PrivacyGateResult:
        references = _references(example)
        source_digests: dict[str, str] = {}
        for reference in references:
            previous = source_digests.setdefault(reference.artifact_id, reference.sha256)
            if previous != reference.sha256:
                raise PrivacyGateError("an artifact ID is referenced with conflicting digests")

        replacements: dict[tuple[str, str], dict] = {}
        redacted_artifacts: list[RedactedArtifact] = []
        source_artifact_digests: dict[str, str] = {}
        for source_id, expected_digest in sorted(source_digests.items()):
            try:
                content = self._resolver.read(source_id)
            except (KeyError, FileNotFoundError) as exc:
                raise PrivacyGateError(f"artifact could not be resolved: {source_id}") from exc
            if not isinstance(content, ArtifactContent):
                raise PrivacyGateError("artifact resolver returned an invalid content record")
            if not isinstance(content.data, bytes):
                raise PrivacyGateError("artifact resolver returned non-byte content")
            if not content.data or len(content.data) > MAX_ARTIFACT_BYTES:
                raise PrivacyGateError("artifact is empty or exceeds the inspection size limit")
            actual_digest = hashlib.sha256(content.data).hexdigest()
            if not hmac.compare_digest(actual_digest, expected_digest):
                raise PrivacyGateError(f"source digest mismatch for artifact {source_id}")
            redacted_data, media_type = _redact(content, self._ocr_inspector)
            if len(redacted_data) > MAX_ARTIFACT_BYTES:
                raise PrivacyGateError("redacted artifact exceeds the output size limit")
            redacted_digest = hashlib.sha256(redacted_data).hexdigest()
            redacted_id = f"redacted-{uuid4().hex}"
            replacements[(source_id, expected_digest)] = {
                "artifact_id": redacted_id,
                "sha256": redacted_digest,
                "redaction_status": "REDACTED",
            }
            source_artifact_digests[redacted_id] = actual_digest
            redacted_artifacts.append(
                RedactedArtifact(
                    artifact_id=redacted_id,
                    sha256=redacted_digest,
                    media_type=media_type,
                    data=redacted_data,
                )
            )

        example_data = _replace_references(
            example.model_dump(mode="json", exclude={"privacy_review"}),
            replacements,
        )
        reviewed_artifacts = {
            item.artifact_id: item.sha256 for item in redacted_artifacts
        }
        review_data = {
            "review_id": f"review-{uuid4().hex}",
            "policy_version": self._policy_version,
            "reviewer_id": self._reviewer_id,
            "reviewed_artifacts": reviewed_artifacts,
            "source_artifact_digests": source_artifact_digests,
            "passed": True,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "signature_sha256": "0" * 64,
        }
        unsigned_review = RedactionReviewReceipt.model_validate(review_data)
        review_data["signature_sha256"] = self._signer.sign(
            canonical_review_payload(unsigned_review)
        )
        example_data["privacy_review"] = review_data
        reviewed_example = TrainingExample.model_validate(example_data)
        return PrivacyGateResult(
            example=reviewed_example,
            artifacts=tuple(redacted_artifacts),
        )
