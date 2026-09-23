"""Small, execution-free contracts for describing and checking actions.

Contracts deliberately sit *before* execution.  They add observability and
deterministic state checks, but do not replace the :class:`ActionBroker` or
the :class:`SafetyKernel`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from sonic.computer_use.models import ActionExecutionStatus
from sonic.kernel.action_broker import ActionBroker
from sonic.safety.kernel import SafetyAuthorization, SafetyKernel


def _frozen(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed", _frozen(self.changed))
        object.__setattr__(self, "removed", tuple(sorted(set(self.removed))))

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
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", _frozen(self.before))
        object.__setattr__(self, "after", _frozen(self.after))


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
        expected_delta = StateDelta.from_states(receipt.before, receipt.after)
        if expected_delta != receipt.state_delta:
            reasons.append("receipt state_delta does not match before/after state")
        for key, expected in contract.postconditions.items():
            if key not in receipt.after or receipt.after[key] != expected:
                reasons.append(f"postcondition failed: {key}")
        return ContractValidation(not reasons, tuple(reasons))
