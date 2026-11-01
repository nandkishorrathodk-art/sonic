"""Small, execution-free contracts for describing and checking actions.

Contracts deliberately sit *before* execution.  They add observability and
deterministic state checks, but do not replace the :class:`ActionBroker` or
the :class:`SafetyKernel`.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from sonic.computer_use.models import ActionExecutionStatus
from sonic.safety.kernel import SafetyAuthorization, SafetyKernel

if TYPE_CHECKING:
    from sonic.kernel.action_broker import ActionBroker


_MAX_NS = (1 << 63) - 1


def _finite_nonnegative_ms(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0 or numeric > _MAX_NS / 1_000_000:
        return None
    return numeric


def _valid_monotonic_ns(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_NS
    )


def _encode_payload(payload: Mapping[str, Any]) -> str:
    def validate_json_value(value: Any) -> None:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("payload object keys must be strings")
            for nested in value.values():
                validate_json_value(nested)
        elif isinstance(value, list):
            for nested in value:
                validate_json_value(nested)
        elif value is None or isinstance(value, (str, bool, int)):
            return
        elif isinstance(value, float) and math.isfinite(value):
            return
        else:
            raise ValueError("payload values must be strict JSON data")

    validate_json_value(payload)
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _frame_identity(screen: Any) -> tuple[int, str] | None:
    """Return AVCS identity or a versioned pixel hash for a fresh desktop frame."""
    state = getattr(screen, "avcs_state", None)
    screen_version = getattr(screen, "perception_version", None)
    if isinstance(state, Mapping) and state.get("engine") == "avcs_reflex":
        frame_index = state.get("frame_index")
        frame_id = getattr(screen, "frame_id", None)
        source_frame_id = state.get("source_frame_id")
        width = getattr(screen, "width", None)
        height = getattr(screen, "height", None)
        scene_hash = state.get("scene_hash")
        if (
            isinstance(frame_index, int)
            and not isinstance(frame_index, bool)
            and frame_index > 0
            and isinstance(screen_version, int)
            and not isinstance(screen_version, bool)
            and screen_version == frame_index
            and isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
            and state.get("width") == width
            and state.get("height") == height
            and isinstance(scene_hash, str)
            and scene_hash.strip()
            and (
                not getattr(screen, "raw_rgb", b"")
                or frame_id is None
                or (
                    isinstance(frame_id, int)
                    and not isinstance(frame_id, bool)
                    and frame_id > 0
                    and isinstance(source_frame_id, int)
                    and not isinstance(source_frame_id, bool)
                    and source_frame_id == frame_id
                )
            )
        ):
            return frame_index, scene_hash.strip()

    if (
        not isinstance(screen_version, int)
        or isinstance(screen_version, bool)
        or screen_version <= 0
        or not _has_pixel_frame(screen)
    ):
        return None

    raw_rgb = getattr(screen, "raw_rgb", None)
    if isinstance(raw_rgb, bytes) and raw_rgb:
        frame_bytes = raw_rgb
    else:
        encoded = str(getattr(screen, "screenshot_base64", "") or "").split(",", 1)[-1]
        try:
            frame_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None
    return screen_version, hashlib.sha256(frame_bytes).hexdigest()


def _has_pixel_frame(screen: Any) -> bool:
    """Accept encoded screenshots or a validated raw RGB frame from Docker."""
    screenshot = getattr(screen, "screenshot_base64", None)
    if isinstance(screenshot, str) and screenshot.strip():
        return True

    raw_rgb = getattr(screen, "raw_rgb", None)
    width = getattr(screen, "width", None)
    height = getattr(screen, "height", None)
    frame_id = getattr(screen, "frame_id", None)
    return bool(
        isinstance(raw_rgb, bytes)
        and raw_rgb
        and isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
        and len(raw_rgb) == width * height * 3
        and isinstance(frame_id, int)
        and not isinstance(frame_id, bool)
        and frame_id > 0
    )


def _frozen(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class GUIActionIntent:
    """One bounded GUI action tied to its source observation.

    Confidence and risk are optional annotations, never authorization inputs;
    the existing safety policy remains the only risk gate.
    """

    action_type: str
    target: str
    payload_json: str
    expected_outcome: str
    source_perception_version: int
    source_state_version: int
    source_scene_hash: str
    source_active_window: str
    source_width: int
    source_height: int
    issued_at_ns: int
    expires_at_ns: int
    confidence: float | None = None
    risk_level: str | None = None

    @classmethod
    def create(
        cls,
        *,
        action_type: str,
        target: str,
        payload: Mapping[str, Any],
        expected_outcome: str,
        observation: Any,
        now_ns: int | None = None,
        ttl_ms: int = 2_000,
    ) -> GUIActionIntent:
        """Create an intent only from a fresh, identified desktop observation."""
        if not isinstance(action_type, str):
            raise ValueError("GUI intents require a GUI action")
        action_name = action_type.strip()
        if not action_name.startswith("GUI_") or not action_name[4:]:
            raise ValueError("GUI intents require a GUI action")
        if not isinstance(payload, Mapping):
            raise ValueError("GUI action payload must be a mapping")
        if (
            not isinstance(ttl_ms, int)
            or isinstance(ttl_ms, bool)
            or ttl_ms <= 0
            or ttl_ms * 1_000_000 > _MAX_NS
        ):
            raise ValueError("GUI action intent TTL must be positive")

        screen = getattr(observation, "screen", None)
        if screen is None:
            raise ValueError("GUI action intent requires an observed screen")
        if (
            getattr(screen, "desktop_state", "") != "INTERACTIVE"
            or not _has_pixel_frame(screen)
        ):
            raise ValueError("GUI action intent requires a fresh pixel frame")
        screen_version = getattr(screen, "perception_version", None)
        if (
            not isinstance(screen_version, int)
            or isinstance(screen_version, bool)
            or screen_version <= 0
        ):
            raise ValueError("GUI action intent requires a versioned desktop frame")
        version = getattr(observation, "perception_version", None)
        identity = _frame_identity(screen)
        if identity is None:
            raise ValueError("GUI action intent requires trustworthy pixel frame identity")
        state_version, scene_hash = identity
        active_window = str(
            getattr(screen, "active_window", "")
            or getattr(observation, "active_application", "")
        ).strip()
        width = getattr(screen, "width", 0)
        height = getattr(screen, "height", 0)
        age_ms = getattr(observation, "observation_age_ms", None)
        capture_age_ms = getattr(screen, "capture_age_ms", None)
        capture_age = _finite_nonnegative_ms(capture_age_ms)
        observation_age = _finite_nonnegative_ms(age_ms)
        if capture_age is None:
            raise ValueError("GUI action intent requires known capture age")
        issued_at = now_ns if now_ns is not None else time.perf_counter_ns()
        if not _valid_monotonic_ns(issued_at):
            raise ValueError("GUI action intent requires a valid monotonic timestamp")
        received_at_ns = getattr(observation, "observation_received_monotonic_ns", None)
        if received_at_ns is not None and (
            not _valid_monotonic_ns(received_at_ns) or received_at_ns > issued_at
        ):
            raise ValueError("GUI action intent requires a valid observation timestamp")
        elapsed_ms = (
            (issued_at - received_at_ns) / 1_000_000
            if received_at_ns is not None
            else 0.0
        )
        effective_age_ms = max(
            capture_age + elapsed_ms,
            observation_age if observation_age is not None else 0.0,
        )
        if (
            not active_window
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version <= 0
        ):
            raise ValueError("GUI action intent requires a versioned active desktop")
        if observation_age is None or effective_age_ms >= ttl_ms:
            raise ValueError("GUI action intent requires fresh observation evidence")
        try:
            payload_json = _encode_payload(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("GUI action payload must be strict JSON data") from exc
        if not isinstance(target, str) or not isinstance(expected_outcome, str):
            raise ValueError("GUI action target and expected outcome must be strings")
        age_ns = math.ceil(effective_age_ms * 1_000_000)
        ttl_ns = ttl_ms * 1_000_000

        return cls(
            action_type=action_name,
            target=target,
            payload_json=payload_json,
            expected_outcome=expected_outcome,
            source_perception_version=version,
            source_state_version=state_version,
            source_scene_hash=scene_hash,
            source_active_window=active_window,
            source_width=width,
            source_height=height,
            issued_at_ns=issued_at,
            expires_at_ns=issued_at + ttl_ns - age_ns,
        )

    def validate_binding(
        self,
        *,
        action_type: str,
        target: str,
        payload: Mapping[str, Any],
        now_ns: int | None = None,
    ) -> tuple[str, ...]:
        """Reject action substitution or an intent that expired before dispatch."""
        reasons: list[str] = []
        current = now_ns if now_ns is not None else time.perf_counter_ns()
        if not _valid_monotonic_ns(current):
            reasons.append("GUI action validation timestamp is invalid")
        elif current >= self.expires_at_ns:
            reasons.append("GUI action intent expired")
        if not isinstance(action_type, str) or action_type != self.action_type:
            reasons.append("GUI action type differs from intent")
        if not isinstance(target, str) or target != self.target:
            reasons.append("GUI action target differs from intent")
        if not isinstance(payload, Mapping):
            reasons.append("GUI action payload is not serializable")
            return tuple(reasons)
        try:
            encoded_payload = _encode_payload(payload)
        except (TypeError, ValueError):
            reasons.append("GUI action payload is not serializable")
        else:
            if encoded_payload != self.payload_json:
                reasons.append("GUI action payload differs from intent")
        return tuple(reasons)

    def rebase_after_verified_transition(
        self,
        observation: Any,
        *,
        now_ns: int | None = None,
    ) -> GUIActionIntent:
        """Mint a continuation only after an observed, authorized UI transition."""
        current = now_ns if now_ns is not None else time.perf_counter_ns()
        if not _valid_monotonic_ns(current):
            raise ValueError("GUI action transition timestamp is invalid")
        remaining_ms = (self.expires_at_ns - current) // 1_000_000
        if remaining_ms <= 0:
            raise ValueError("GUI action intent expired before its verified transition")
        return type(self).create(
            action_type=self.action_type,
            target=self.target,
            payload=json.loads(self.payload_json),
            expected_outcome=self.expected_outcome,
            observation=observation,
            now_ns=current,
            ttl_ms=int(remaining_ms),
        )

    def validate_observation(
        self,
        screen: Any,
        *,
        now_ns: int | None = None,
        max_age_ms: int = 2_000,
        elapsed_ms: float = 0.0,
    ) -> tuple[str, ...]:
        """Require fresh evidence from the same window and display geometry."""
        current = now_ns if now_ns is not None else time.perf_counter_ns()
        reasons: list[str] = []
        if not _valid_monotonic_ns(current):
            reasons.append("GUI action validation timestamp is invalid")
        elif current >= self.expires_at_ns:
            reasons.append("GUI action intent expired")
        age_ms = _finite_nonnegative_ms(getattr(screen, "capture_age_ms", None))
        elapsed = _finite_nonnegative_ms(elapsed_ms)
        max_age = _finite_nonnegative_ms(max_age_ms)
        if age_ms is None or elapsed is None or max_age is None or age_ms + elapsed > max_age:
            reasons.append("fresh desktop evidence is unavailable")
        if (
            getattr(screen, "desktop_state", "") != "INTERACTIVE"
            or not _has_pixel_frame(screen)
        ):
            reasons.append("fresh pixel frame is unavailable")
        active_window = str(getattr(screen, "active_window", "") or "").strip()
        if active_window.casefold() != self.source_active_window.casefold():
            reasons.append("active window differs from GUI action intent")
        if (
            getattr(screen, "width", 0) != self.source_width
            or getattr(screen, "height", 0) != self.source_height
        ):
            reasons.append("display geometry differs from GUI action intent")
        identity = _frame_identity(screen)
        if identity is None:
            reasons.append("a trustworthy pixel frame identity is unavailable")
        elif identity[0] < self.source_state_version:
            reasons.append("pixel frame is older than the GUI action intent")
        if identity is not None and identity[1] != self.source_scene_hash:
            reasons.append("pixel frame differs from GUI action intent")
        return tuple(reasons)


@dataclass(frozen=True)
class ActionContract:
    """Typed metadata for one action's observable state transition."""

    action_type: str
    preconditions: Mapping[str, Any] = field(default_factory=dict)
    postconditions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, str) or not self.action_type.strip():
            raise ValueError("action_type must be a non-empty string")
        object.__setattr__(self, "action_type", self.action_type.strip())
        object.__setattr__(self, "preconditions", _frozen(self.preconditions))
        object.__setattr__(self, "postconditions", _frozen(self.postconditions))


@dataclass(frozen=True)
class StateDelta:
    """Deterministic shallow delta between two state snapshots."""

    changed: Mapping[str, Any] = field(default_factory=dict)
    removed: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed", _frozen(self.changed))
        object.__setattr__(self, "removed", tuple(sorted(set(self.removed))))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))

    @classmethod
    def from_states(
        cls, before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> StateDelta:
        changed = {
            key: after[key]
            for key in after
            if key not in before or before[key] != after[key]
        }
        removed = tuple(key for key in before if key not in after)
        return cls(changed=changed, removed=removed)


@dataclass(frozen=True)
class ActionReceipt:
    """Evidence produced by an executor; this module never creates execution."""

    action_type: str
    status: ActionExecutionStatus
    state_delta: StateDelta
    before: Mapping[str, Any] = field(default_factory=dict)
    after: Mapping[str, Any] = field(default_factory=dict)
    exit_code: int | None = None
    evidence: str = ""
    intent: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", _frozen(self.before))
        object.__setattr__(self, "after", _frozen(self.after))
        object.__setattr__(self, "intent", _frozen(self.intent))


@dataclass(frozen=True)
class ContractValidation:
    valid: bool
    reasons: tuple[str, ...] = ()
    authorization: SafetyAuthorization | None = None


class ActionContractValidator:
    """Validate contracts and receipts without dispatching an action."""

    def __init__(
        self,
        *,
        broker: ActionBroker | None = None,
        safety_kernel: SafetyKernel | None = None,
    ) -> None:
        if broker is not None and safety_kernel is not None:
            raise ValueError("provide broker or safety_kernel, not both")
        self._broker = broker
        self._safety_kernel = safety_kernel or (
            broker.safety_kernel if broker is not None else SafetyKernel()
        )

    def validate(
        self,
        contract: ActionContract,
        *,
        state: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
        provider: Any = None,
        approved: bool = False,
    ) -> ContractValidation:
        """Check safety and preconditions; never calls ``ActionBroker.execute``."""

        reasons: list[str] = []
        authorization: SafetyAuthorization | None = None
        try:
            authorization = self._safety_kernel.authorize(
                contract.action_type,
                dict(parameters or {}),
                provider=provider,
            )
            if not authorization.is_allowed:
                if authorization.verdict.value == "require_approval" and approved:
                    # Approval is explicit input, but the authorization remains
                    # owned by the existing broker/kernel execution path.
                    pass
                else:
                    reasons.append(f"safety: {authorization.reason}")
            for key, expected in contract.preconditions.items():
                if key not in state or state[key] != expected:
                    reasons.append(f"precondition failed: {key}")
        except Exception as exc:
            reasons.append(f"validation error: {type(exc).__name__}")
        return ContractValidation(not reasons, tuple(reasons), authorization)

    def validate_receipt(
        self,
        contract: ActionContract,
        receipt: ActionReceipt,
    ) -> ContractValidation:
        """Check an executor-produced receipt and its postconditions."""

        reasons: list[str] = []
        if receipt.action_type != contract.action_type:
            reasons.append("receipt action_type does not match contract")
        if receipt.status not in (
            ActionExecutionStatus.SUCCEEDED,
            ActionExecutionStatus.SUCCESS,
            ActionExecutionStatus.COMPLETED,
            ActionExecutionStatus.VERIFIED,
        ):
            reasons.append(f"receipt status is not successful: {receipt.status}")
        if receipt.exit_code not in (None, 0):
            reasons.append(f"receipt exit code is not successful: {receipt.exit_code}")
        evidence = receipt.evidence.strip() if isinstance(receipt.evidence, str) else ""
        if receipt.exit_code is None and not evidence:
            reasons.append("receipt has neither a successful exit code nor evidence")
        expected_delta = StateDelta.from_states(receipt.before, receipt.after)
        if expected_delta != receipt.state_delta:
            reasons.append("receipt state_delta does not match before/after state")
        for key, expected in contract.preconditions.items():
            if key not in receipt.before or receipt.before[key] != expected:
                reasons.append(f"precondition failed in receipt: {key}")
        for key, expected in contract.postconditions.items():
            if key not in receipt.after or receipt.after[key] != expected:
                reasons.append(f"postcondition failed: {key}")
        return ContractValidation(not reasons, tuple(reasons))
