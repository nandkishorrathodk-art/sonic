"""Small, execution-free contracts for describing and checking actions.

Contracts deliberately sit *before* execution.  They add observability and
deterministic state checks, but do not replace the :class:`ActionBroker` or
the :class:`SafetyKernel`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import math
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from sonic.computer_use.models import ActionExecutionStatus
from sonic.safety.kernel import SafetyAuthorization, SafetyKernel

if TYPE_CHECKING:
    from sonic.kernel.action_broker import ActionBroker


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
        action_name = str(action_type).strip()
        if not action_name.startswith("GUI_"):
            raise ValueError("GUI intents require a GUI action")
        if not isinstance(payload, Mapping):
            raise ValueError("GUI action payload must be a mapping")
        if ttl_ms <= 0:
            raise ValueError("GUI action intent TTL must be positive")

        screen = getattr(observation, "screen", None)
        if screen is None:
            raise ValueError("GUI action intent requires an observed screen")
        if (
            getattr(screen, "desktop_state", "") != "INTERACTIVE"
            or not getattr(screen, "screenshot_base64", "")
        ):
            raise ValueError("GUI action intent requires a fresh pixel frame")
        avcs_state = getattr(screen, "avcs_state", None)
        if not isinstance(avcs_state, Mapping) or avcs_state.get("engine") != "avcs_reflex":
            raise ValueError("GUI action intent requires current AVCS pixel state")
        state_version = avcs_state.get("frame_index")
        if (
            not isinstance(state_version, int)
            or isinstance(state_version, bool)
            or state_version <= 0
        ):
            raise ValueError("GUI action intent requires a versioned AVCS frame")
        scene_hash = avcs_state.get("scene_hash")
        if not isinstance(scene_hash, str) or not scene_hash.strip():
            raise ValueError("GUI action intent requires an AVCS scene fingerprint")
        active_window = str(
            getattr(screen, "active_window", "")
            or getattr(observation, "active_application", "")
        ).strip()
        width = getattr(screen, "width", 0)
        height = getattr(screen, "height", 0)
        version = getattr(observation, "perception_version", 0)
        age_ms = getattr(observation, "observation_age_ms", None)
        capture_age_ms = getattr(screen, "capture_age_ms", None)
        if (
            not isinstance(capture_age_ms, (int, float))
            or isinstance(capture_age_ms, bool)
            or not math.isfinite(capture_age_ms)
            or capture_age_ms < 0
        ):
            raise ValueError("GUI action intent requires known capture age")
        issued_at = now_ns if now_ns is not None else time.perf_counter_ns()
        received_at_ns = getattr(observation, "observation_received_monotonic_ns", None)
        elapsed_ms = (
            max(0, issued_at - received_at_ns) / 1_000_000
            if isinstance(received_at_ns, int) and not isinstance(received_at_ns, bool)
            else 0.0
        )
        effective_age_ms = max(
            capture_age_ms + elapsed_ms,
            float(age_ms)
            if isinstance(age_ms, (int, float)) and not isinstance(age_ms, bool)
            else 0.0,
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
        if (
            not isinstance(age_ms, (int, float))
            or isinstance(age_ms, bool)
            or not math.isfinite(age_ms)
            or age_ms < 0
            or effective_age_ms > ttl_ms
        ):
            raise ValueError("GUI action intent requires fresh observation evidence")

        return cls(
            action_type=action_name,
            target=str(target),
            payload_json=json.dumps(
                dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
            expected_outcome=str(expected_outcome),
            source_perception_version=version,
            source_state_version=state_version,
            source_scene_hash=scene_hash,
            source_active_window=active_window,
            source_width=width,
            source_height=height,
            issued_at_ns=issued_at,
            expires_at_ns=issued_at + max(0, ttl_ms - int(effective_age_ms)) * 1_000_000,
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
        if current > self.expires_at_ns:
            reasons.append("GUI action intent expired")
        if str(action_type).strip() != self.action_type:
            reasons.append("GUI action type differs from intent")
        if str(target) != self.target:
            reasons.append("GUI action target differs from intent")
        try:
            encoded_payload = json.dumps(
                dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
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
        if current > self.expires_at_ns:
            reasons.append("GUI action intent expired")
        age_ms = getattr(screen, "capture_age_ms", None)
        if (
            not isinstance(age_ms, (int, float))
            or isinstance(age_ms, bool)
            or not math.isfinite(age_ms)
            or age_ms < 0
            or age_ms + elapsed_ms > max_age_ms
        ):
            reasons.append("fresh desktop evidence is unavailable")
        if (
            getattr(screen, "desktop_state", "") != "INTERACTIVE"
            or not getattr(screen, "screenshot_base64", "")
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
        avcs_state = getattr(screen, "avcs_state", None)
        current_version = (
            avcs_state.get("frame_index")
            if isinstance(avcs_state, Mapping)
            and avcs_state.get("engine") == "avcs_reflex"
            else None
        )
        if (
            not isinstance(current_version, int)
            or isinstance(current_version, bool)
            or current_version <= 0
        ):
            reasons.append("a versioned AVCS frame is unavailable")
        current_scene_hash = (
            avcs_state.get("scene_hash")
            if isinstance(avcs_state, Mapping)
            and avcs_state.get("engine") == "avcs_reflex"
            else None
        )
        if current_scene_hash != self.source_scene_hash:
            reasons.append("AVCS scene differs from GUI action intent")
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
