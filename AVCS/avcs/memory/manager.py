"""Master Visual Memory Manager for Multi-Turn Computer Control.

Orchestrates the Semantic DOM-less Scene Tree (SDST), Persistent UI Registry,
and Visual Diff Engine (Delta_visual), preserving agent context across 50+ steps
while cutting vision token costs by >93%.
"""
from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional, Tuple, Deque

from avcs.memory.diff_engine import VisualDiffEngine
from avcs.memory.registry import PersistentUIRegistry
from avcs.memory.tree_extractor import SemanticSceneTreeExtractor
from avcs.memory.types import (
    UIMLElement,
    UIMLTree,
    VisualDiff,
    VisualMemorySnapshot,
)


class VisualMemoryManager:
    """Manages multi-turn visual episodic memory and context compression (<2ms per turn)."""

    def __init__(
        self,
        cost_threshold: float = 0.40,
        default_box_size: Tuple[float, float] = (0.08, 0.04),
    ) -> None:
        self.extractor = SemanticSceneTreeExtractor(default_box_size=default_box_size)
        self.registry = PersistentUIRegistry(cost_threshold=cost_threshold)
        self.diff_engine = VisualDiffEngine()

        self.turn_index: int = 0
        self.initial_tree: Optional[UIMLTree] = None
        self.latest_tree: Optional[UIMLTree] = None
        self.diff_history: Deque[VisualDiff] = collections.deque(maxlen=100)

        # Economics & Telemetry
        self.total_tokens_avoided: int = 0
        self.estimated_dollars_saved: float = 0.0

    def reset(self) -> None:
        """Clears memory history and resets registry for a new task session."""
        self.extractor = SemanticSceneTreeExtractor()
        self.registry = PersistentUIRegistry()
        self.diff_engine = VisualDiffEngine()
        self.turn_index = 0
        self.initial_tree = None
        self.latest_tree = None
        self.diff_history.clear()
        self.total_tokens_avoided = 0
        self.estimated_dollars_saved = 0.0

    def step(
        self,
        affordances: List[Tuple[float, float, str, float]],
        category: str = "general_ui",
        category_confidence: float = 1.0,
        foveal_inspection: Optional[Any] = None,
        moving_objects: Optional[List[Any]] = None,
        container_flow: Tuple[float, float] = (0.0, 0.0),
        prior_action: str = "observe",
        surprise_score: float = 0.0,
        window_title: Optional[str] = None,
    ) -> VisualDiff:
        """Processes a new visual observation turn and updates episodic memory (<2ms)."""
        self.turn_index += 1

        # 1. Extract raw scene tree from sensory outputs
        raw_tree = self.extractor.extract(
            affordances=affordances,
            category=category,
            category_confidence=category_confidence,
            foveal_inspection=foveal_inspection,
            moving_objects=moving_objects,
            window_title=window_title,
        )

        # 2. Align element identities via Persistent UI Registry (Hungarian Re-ID)
        observed_elements = list(raw_tree.elements.values())
        active_elements, disappeared_ids = self.registry.update(
            observed_elements=observed_elements,
            container_flow=container_flow,
        )

        # Re-pack tree with persistent element IDs
        current_tree = UIMLTree(
            timestamp=raw_tree.timestamp,
            window_title=raw_tree.window_title,
            primary_category=raw_tree.primary_category,
            elements={elem.element_id: elem for elem in active_elements},
        )

        # 3. Compute Delta Transition
        diff = self.diff_engine.compute_diff(
            current_tree=current_tree,
            prior_action=prior_action,
            surprise_score=surprise_score,
            turn_index=self.turn_index,
        )

        # Ensure disappeared IDs are flagged in the diff
        for did in disappeared_ids:
            if did not in diff.removed_ids:
                diff.removed_ids.append(did)

        # 4. Save trees
        if self.turn_index == 1 or self.initial_tree is None:
            self.initial_tree = current_tree
        self.latest_tree = current_tree
        self.diff_history.append(diff)

        # 5. Calculate Token & Dollar Savings
        # A raw 1080p screenshot sent to a VLM consumes ~1,600 vision tokens.
        # Our representation uses:
        # - Turn 1: tree token estimate (~110 tokens)
        # - Turn 2+: diff token estimate (~35 tokens)
        consumed_tokens = current_tree.token_estimate() if self.turn_index == 1 else diff.token_estimate()
        saved_tokens = max(0, 1600 - consumed_tokens)
        self.total_tokens_avoided += saved_tokens
        # Standard input token cost ($3.00 per 1,000,000 tokens)
        self.estimated_dollars_saved = (self.total_tokens_avoided / 1_000_000.0) * 3.00

        return diff

    def get_context_prompt(self, max_delta_turns: int = 10) -> str:
        """Emits an ultra-lean, high-signal multi-turn context prompt for LLM consumption (<300 tokens)."""
        if self.initial_tree is None and self.latest_tree is None:
            return "No visual context available."

        current_tree = self.latest_tree or self.initial_tree
        lines: List[str] = [
            "### Current Visual UI State (AVCS DOM-less UIML):",
            current_tree.to_string(),  # type: ignore
        ]

        # Recent causal delta history
        if len(self.diff_history) > 1:
            lines.append("\n### Recent Visual State Transitions:")
            recent_deltas = list(self.diff_history)[1:][-max_delta_turns:]
            for delta in recent_deltas:
                lines.append(delta.to_prompt_delta())

        return "\n".join(lines)

    def get_element(self, element_id: str) -> Optional[UIMLElement]:
        """Looks up a persistent element by ID."""
        return self.registry.get_element(element_id)

    def find_elements(self, query: str) -> List[UIMLElement]:
        """Finds elements matching a label query."""
        if not self.latest_tree:
            return []
        return self.latest_tree.find_by_label(query)

    def get_snapshot(self) -> VisualMemorySnapshot:
        """Returns complete session episodic memory snapshot."""
        return VisualMemorySnapshot(
            total_turns=self.turn_index,
            initial_tree=self.initial_tree or UIMLTree(0.0, "Empty", "unknown"),
            deltas=list(self.diff_history),
            cumulative_tokens_saved=self.total_tokens_avoided,
            estimated_dollar_savings=self.estimated_dollars_saved,
        )
