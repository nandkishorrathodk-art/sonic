"""AVCS Multi-Turn Memory Manager Type Definitions and Schemas.

Defines the core data models for the Semantic DOM-less Scene Tree (SDST),
UI Markup Language (UIML), Visual Change Diffing, and Entity Lifecycle Tracking.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class AffordanceType(enum.Enum):
    """Categorization of interactive UI affordances."""
    STATIC = "static"
    CLICKABLE = "clickable"
    TYPEABLE = "typeable"
    SCROLLABLE = "scrollable"
    DRAGGABLE = "draggable"


class EntityLifecycle(enum.Enum):
    """State of an interactive entity across temporal turns."""
    VISIBLE = "visible"
    OFF_SCREEN_ABOVE = "off_screen_above"
    OFF_SCREEN_BELOW = "off_screen_below"
    OCCLUDED = "occluded"
    DESTROYED = "destroyed"


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates in [0.0, 1.0]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        self.xmin = max(0.0, min(1.0, float(self.xmin)))
        self.ymin = max(0.0, min(1.0, float(self.ymin)))
        self.xmax = max(self.xmin, min(1.0, float(self.xmax)))
        self.ymax = max(self.ymin, min(1.0, float(self.ymax)))

    @property
    def center(self) -> Tuple[float, float]:
        """Normalized centroid (cx, cy)."""
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, other: BoundingBox) -> bool:
        """Returns True if this bounding box fully encloses other."""
        return (
            self.xmin <= other.xmin
            and self.ymin <= other.ymin
            and self.xmax >= other.xmax
            and self.ymax >= other.ymax
        )

    def iou(self, other: BoundingBox) -> float:
        """Computes Intersection over Union (IoU) with another box."""
        ixmin = max(self.xmin, other.xmin)
        iymin = max(self.ymin, other.ymin)
        ixmax = min(self.xmax, other.xmax)
        iymax = min(self.ymax, other.ymax)

        iw = max(0.0, ixmax - ixmin)
        ih = max(0.0, iymax - iymin)
        intersection = iw * ih

        union = self.area + other.area - intersection
        if union <= 1e-8:
            return 0.0
        return intersection / union

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass
class UIMLElement:
    """A semantic affordance entity in the DOM-less scene tree."""
    element_id: str                      # Persistent ID (e.g. "#e1", "#btn_submit")
    role: AffordanceType                 # CLICKABLE, TYPEABLE, etc.
    label: str                           # Semantic text or visual label
    bbox: BoundingBox                    # Normalized screen coordinates
    parent_id: Optional[str] = None      # Container panel or window ID
    is_active: bool = False              # Focused, selected, or active state
    is_disabled: bool = False
    is_primary: bool = False
    current_value: Optional[str] = None  # Text inside typeable fields
    visual_embedding: Optional[np.ndarray] = None  # 64-d appearance latent

    def to_uiml(self, indent: int = 2) -> str:
        """Serializes to ultra-compact UIML format (<12 tokens per element)."""
        pad = " " * indent
        cx, cy = self.bbox.center
        flags = []
        if self.is_active:
            flags.append("active")
        if self.is_disabled:
            flags.append("disabled")
        if self.is_primary:
            flags.append("primary")
        if self.current_value:
            flags.append(f'val="{self.current_value}"')
        flag_str = (" " + " ".join(flags)) if flags else ""
        role_tag = self.role.value.upper()
        return f'{pad}[{role_tag}: "{self.label}" @({cx:.3f},{cy:.3f}) id={self.element_id}{flag_str}]'


@dataclass
class UIMLTree:
    """Hierarchical snapshot of the entire desktop UI in pure pixels."""
    timestamp: float
    window_title: str
    primary_category: str                # e.g. "editor_code", "browser_webpage"
    elements: Dict[str, UIMLElement] = field(default_factory=dict)
    containers: List[Dict[str, Any]] = field(default_factory=list)

    def to_string(self) -> str:
        """Emits indented UIML (<150 tokens)."""
        lines = [f'[WINDOW: "{self.window_title}" class={self.primary_category}]']
        for elem in self.elements.values():
            lines.append(elem.to_uiml(indent=2))
        return "\n".join(lines)

    def find_by_id(self, element_id: str) -> Optional[UIMLElement]:
        return self.elements.get(element_id)

    def find_by_label(self, query: str) -> List[UIMLElement]:
        q = query.strip().lower()
        return [e for e in self.elements.values() if q in e.label.lower()]

    def token_estimate(self) -> int:
        """Estimates total LLM input token count for this tree (approx 4 chars/token)."""
        text = self.to_string()
        return max(1, len(text) // 4)


@dataclass
class VisualDiff:
    """Delta transition between two consecutive turns."""
    turn_index: int
    prior_action: str                    # e.g. "click(#e1 - Submit)"
    surprise_score: float                # Predictive coding prediction error
    added_elements: List[UIMLElement] = field(default_factory=list)
    removed_ids: List[str] = field(default_factory=list)
    modified_elements: List[Tuple[UIMLElement, str]] = field(default_factory=list) # (elem, delta_desc)
    scroll_shift: Optional[Tuple[str, float]] = None # (container_id, dy)
    causal_summary: str = ""             # Compact natural language causal statement

    def to_prompt_delta(self) -> str:
        """Emits compact delta representation (30-50 tokens)."""
        lines = [f"Delta(turn={self.turn_index}, after='{self.prior_action}'):"]
        if self.causal_summary:
            lines.append(f"  Summary: {self.causal_summary}")
        for elem in self.added_elements:
            lines.append(f"  + {elem.to_uiml(indent=0)}")
        for rem_id in self.removed_ids:
            lines.append(f"  - id={rem_id}")
        for elem, desc in self.modified_elements:
            lines.append(f"  ~ id={elem.element_id} [{desc}]")
        if self.scroll_shift:
            lines.append(f"  ^ scroll({self.scroll_shift[0]}, dy={self.scroll_shift[1]:+.2f})")
        return "\n".join(lines)

    def token_estimate(self) -> int:
        """Estimates total LLM input token count for this delta."""
        return max(1, len(self.to_prompt_delta()) // 4)


@dataclass
class VisualMemorySnapshot:
    """Complete episodic memory history across multi-turn session."""
    total_turns: int
    initial_tree: UIMLTree
    deltas: List[VisualDiff] = field(default_factory=list)
    cumulative_tokens_saved: int = 0
    estimated_dollar_savings: float = 0.0
