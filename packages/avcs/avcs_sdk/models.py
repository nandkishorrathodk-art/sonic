"""Stable, dependency-free public data contracts for AVCS SDK v0.1."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping
from uuid import uuid4
import math
import time


class ActionKind(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"


class ActionClass(str, Enum):
    IDEMPOTENT = "idempotent"
    CONDITIONAL = "conditionally_repeatable"
    NON_IDEMPOTENT = "non_idempotent"


class ActionStatus(str, Enum):
    CONFIRMED = "confirmed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TargetProvenance(str, Enum):
    UI_AUTOMATION = "ui_automation"
    VISION = "vision"
    FUSED = "fused"
    MANUAL = "manual"


class RiskLevel(str, Enum):
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """Point in normalized virtual-desktop coordinates, both inclusive [0, 1]."""
    x: float
    y: float

    def __post_init__(self) -> None:
        _unit_interval(self.x, "x")
        _unit_interval(self.y, "y")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized rectangle using x/y origin plus width/height."""
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BoundingBox width and height must be positive")
        if self.x + self.width > 1.0 + 1e-9 or self.y + self.height > 1.0 + 1e-9:
            raise ValueError("BoundingBox must fit inside normalized desktop bounds")

    @property
    def center(self) -> NormalizedPoint:
        return NormalizedPoint(self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    captured_at: float
    width: int
    height: int
    active_window: str | None = None
    coordinate_space: str = "normalized_virtual_desktop"
    frame_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id cannot be empty")
        if not math.isfinite(self.captured_at) or self.captured_at <= 0:
            raise ValueError("captured_at must be a positive Unix timestamp")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.coordinate_space != "normalized_virtual_desktop":
            raise ValueError("v0.1 only supports normalized_virtual_desktop coordinates")


@dataclass(frozen=True, slots=True)
class Target:
    target_id: str
    observation_id: str
    label: str
    bounds: BoundingBox
    confidence: float
    provenance: TargetProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, TargetProvenance):
            raise TypeError("provenance must be a TargetProvenance value")
        if not self.target_id.strip() or not self.observation_id.strip():
            raise ValueError("target_id and observation_id cannot be empty")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not self.label.strip():
            raise ValueError("label cannot be empty")


@dataclass(frozen=True, slots=True)
class ActionRequest:
    kind: ActionKind
    target: Target | None = None
    point: NormalizedPoint | None = None
    observation_id: str | None = None
    text: str | None = None
    key: str | None = None
    scroll_delta: int | None = None
    action_class: ActionClass = ActionClass.NON_IDEMPOTENT
    risk: RiskLevel = RiskLevel.NORMAL
    authorization_ref: str | None = None
    expected_postcondition: str | None = None
    timeout_s: float = 5.0
    action_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            raise TypeError("kind must be an ActionKind value")
        if not isinstance(self.action_class, ActionClass):
            raise TypeError("action_class must be an ActionClass value")
        if not isinstance(self.risk, RiskLevel):
            raise TypeError("risk must be a RiskLevel value")
        if not self.action_id.strip():
            raise ValueError("action_id cannot be empty")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if self.kind in (ActionKind.CLICK, ActionKind.DOUBLE_CLICK) and self.target is None and self.point is None:
            raise ValueError(f"{self.kind.value} requires target or point")
        if self.target is not None and self.point is not None:
            raise ValueError("Provide target or point, not both")
        if self.point is not None and not (self.observation_id and self.observation_id.strip()):
            raise ValueError("coordinate-based actions require the observation_id used to derive the point")
        if self.target is not None and self.observation_id is not None and self.observation_id != self.target.observation_id:
            raise ValueError("observation_id must match the target's observation")
        if self.kind == ActionKind.TYPE_TEXT and self.text is None:
            raise ValueError("type_text requires text (empty text is allowed explicitly)")
        if self.kind == ActionKind.PRESS_KEY and not (self.key and self.key.strip()):
            raise ValueError("press_key requires a non-empty key")
        if self.kind == ActionKind.SCROLL and (self.scroll_delta is None or self.scroll_delta == 0):
            raise ValueError("scroll requires a non-zero scroll_delta")
        if self.risk == RiskLevel.HIGH and not (self.authorization_ref and self.authorization_ref.strip()):
            raise ValueError("high-risk actions require an authorization_ref")
        if self.expected_postcondition is not None and not self.expected_postcondition.strip():
            raise ValueError("expected_postcondition cannot be blank")


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    verified: bool
    method: str
    observed_at: float
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("verification method cannot be empty")
        if not math.isfinite(self.observed_at) or self.observed_at <= 0:
            raise ValueError("observed_at must be a positive Unix timestamp")


@dataclass(frozen=True, slots=True)
class ActionResult:
    action_id: str
    status: ActionStatus
    duration_ms: float
    evidence: VerificationEvidence | None = None
    error_code: str | None = None
    message: str | None = None
    attempt_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.status, ActionStatus):
            raise TypeError("status must be an ActionStatus value")
        if not self.action_id.strip():
            raise ValueError("action_id cannot be empty")
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative")
        if self.attempt_count < 0:
            raise ValueError("attempt_count cannot be negative")


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def now_timestamp() -> float:
    """Current Unix timestamp; exposed to make adapters/tests consistent."""
    return time.time()
