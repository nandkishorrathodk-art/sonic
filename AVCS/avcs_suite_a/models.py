"""Typed data contracts for the AVCS deterministic fixture benchmark.

This module has no Windows-only imports, so the scorer can be unit-tested on CI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ControlSpec:
    id: str
    kind: str
    label: str
    x: int
    y: int
    width: int
    height: int
    state_key: str | None = None
    initial: Any = None
    values: tuple[str, ...] = ()
    enabled: bool = True
    action: Mapping[str, Any] = field(default_factory=dict)
    group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["values"] = list(self.values)
        return item


@dataclass(frozen=True)
class FixtureCase:
    id: str
    category: str
    title: str
    task: str
    controls: tuple[ControlSpec, ...]
    initial_state: Mapping[str, Any]
    expected_state: Mapping[str, Any]
    target_id: str | None
    expected_status: str = "confirmed"
    timeout_s: float = 15.0
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "task": self.task,
            "controls": [c.to_dict() for c in self.controls],
            "initial_state": dict(self.initial_state),
            "expected_state": dict(self.expected_state),
            "target_id": self.target_id,
            "expected_status": self.expected_status,
            "timeout_s": self.timeout_s,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | None) -> "Rect | None":
        if not value:
            return None
        try:
            rect = cls(float(value["x"]), float(value["y"]),
                       float(value["width"]), float(value["height"]))
        except (KeyError, TypeError, ValueError):
            return None
        if rect.width <= 0 or rect.height <= 0:
            return None
        return rect

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class CaseScore:
    case_id: str
    category: str
    expected_status: str
    reported_status: str | None
    task_success: bool
    expected_state_match: bool
    target_scored: bool
    target_iou: float | None
    target_match: bool | None
    false_confirmation: bool
    elapsed_ms: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentAdapter(Protocol):
    """Implement this protocol in a project-specific adapter module.

    The adapter receives only the user-visible task and timeout. It must use the
    real AVCS SDK against the visible fixture window. It must NOT receive fixture
    IDs, expected state, target labels/bounds, or oracle credentials.

    Return a mapping with at least ``status`` (confirmed/failed/ambiguous/blocked).
    Optional ``duration_ms`` and ``actions`` are used for additional metrics.
    Each action may include ``status`` and ``target_bbox`` with screen-pixel
    x/y/width/height. Only report telemetry actually emitted by AVCS.
    """
    def execute(self, task: str, timeout_s: float) -> Mapping[str, Any]: ...
