"""Visual Change Diff Engine (Delta_visual).

Computes minimal causal state deltas between consecutive turns (30-50 tokens)
instead of re-submitting expensive full screenshots (>1,600 tokens).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from avcs.memory.types import UIMLElement, UIMLTree, VisualDiff


class VisualDiffEngine:
    """Computes differential state transitions and causal summaries across turns (<1ms)."""

    def __init__(self) -> None:
        self.previous_tree: Optional[UIMLTree] = None

    def compute_diff(
        self,
        current_tree: UIMLTree,
        prior_action: str = "observe",
        surprise_score: float = 0.0,
        turn_index: int = 1,
    ) -> VisualDiff:
        """Diffs previous and current scene tree, producing a minimal causal VisualDiff."""
        if self.previous_tree is None:
            self.previous_tree = current_tree
            return VisualDiff(
                turn_index=turn_index,
                prior_action=prior_action,
                surprise_score=surprise_score,
                added_elements=list(current_tree.elements.values()),
                causal_summary=f"Initial screen loaded ({current_tree.primary_category})",
            )

        prev_elems = self.previous_tree.elements
        curr_elems = current_tree.elements

        added: List[UIMLElement] = []
        removed_ids: List[str] = []
        modified: List[Tuple[UIMLElement, str]] = []

        # 1. Identify Added & Mutated Elements
        for eid, elem in curr_elems.items():
            if eid not in prev_elems:
                added.append(elem)
            else:
                p_elem = prev_elems[eid]
                changes = []
                if p_elem.label != elem.label:
                    changes.append(f'label="{p_elem.label}" -> "{elem.label}"')
                if p_elem.current_value != elem.current_value:
                    changes.append(f'val="{p_elem.current_value}" -> "{elem.current_value}"')
                if p_elem.is_active != elem.is_active:
                    changes.append(f"active={elem.is_active}")
                if p_elem.is_disabled != elem.is_disabled:
                    changes.append(f"disabled={elem.is_disabled}")
                if p_elem.is_primary != elem.is_primary:
                    changes.append(f"primary={elem.is_primary}")
                if p_elem.role != elem.role:
                    changes.append(f"role={elem.role.value}")

                # Detect dimension resize (>2% width/height change)
                dw = abs(elem.bbox.width - p_elem.bbox.width)
                dh = abs(elem.bbox.height - p_elem.bbox.height)
                if dw > 0.02 or dh > 0.02:
                    changes.append(f"resized(w={elem.bbox.width:.2f},h={elem.bbox.height:.2f})")

                # Detect movement (>3% screen shift)
                dcx = elem.bbox.center[0] - p_elem.bbox.center[0]
                dcy = elem.bbox.center[1] - p_elem.bbox.center[1]
                if abs(dcx) > 0.03 or abs(dcy) > 0.03:
                    changes.append(f"moved({dcx:+.2f},{dcy:+.2f})")

                if changes:
                    modified.append((elem, ", ".join(changes)))

        # 2. Identify Removed Elements
        for eid in prev_elems:
            if eid not in curr_elems:
                removed_ids.append(eid)

        # 3. Detect Scroll Shift
        scroll_shift: Optional[Tuple[str, float]] = None
        if modified and len(modified) >= 2:
            # Check if multiple elements shifted uniformly in Y
            y_shifts = [
                elem.bbox.center[1] - prev_elems[elem.element_id].bbox.center[1]
                for elem, desc in modified
                if "moved" in desc and elem.element_id in prev_elems
            ]
            if y_shifts and abs(max(y_shifts) - min(y_shifts)) < 0.02:
                avg_dy = sum(y_shifts) / len(y_shifts)
                if abs(avg_dy) > 0.04:
                    scroll_shift = ("viewport", avg_dy)

        # 4. Synthesize High-Level Natural Language Causal Narrative (<25 words)
        causal_summary = self._synthesize_narrative(
            prior_action, added, removed_ids, modified, scroll_shift, surprise_score
        )

        diff = VisualDiff(
            turn_index=turn_index,
            prior_action=prior_action,
            surprise_score=surprise_score,
            added_elements=added,
            removed_ids=removed_ids,
            modified_elements=modified,
            scroll_shift=scroll_shift,
            causal_summary=causal_summary,
        )

        self.previous_tree = current_tree
        return diff

    def _synthesize_narrative(
        self,
        action: str,
        added: List[UIMLElement],
        removed: List[str],
        modified: List[Tuple[UIMLElement, str]],
        scroll: Optional[Tuple[str, float]],
        surprise: float,
    ) -> str:
        parts: List[str] = []
        clean_act = action.strip()
        if clean_act and clean_act != "observe":
            parts.append(f"Action '{clean_act}' executed.")

        if scroll:
            dy = scroll[1]
            direction = "down" if dy < 0 else "up"
            parts.append(f"Screen scrolled {direction} by {abs(dy):.2f}.")

        if removed:
            parts.append(f"{len(removed)} element(s) dismissed.")

        if added:
            labels = [f"'{e.label}'" for e in added[:2]]
            if len(added) > 2:
                labels.append(f"+{len(added) - 2} more")
            parts.append(f"Appeared: {', '.join(labels)}.")

        if modified and not scroll:
            parts.append(f"{len(modified)} element state(s) updated.")

        if not parts:
            if surprise > 0.2:
                return "Visual transition detected; scene elements stable."
            return "Screen static and ready for next action."

        return " ".join(parts)
