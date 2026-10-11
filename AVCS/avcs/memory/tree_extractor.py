"""Semantic DOM-less Scene Tree Extractor.

Extracts structured UIML trees directly from pure-pixel biological perception states,
completely independent of web DOM, browser APIs, or OS accessibility hooks.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from avcs.memory.types import AffordanceType, BoundingBox, UIMLElement, UIMLTree


class SemanticSceneTreeExtractor:
    """Extracts a structured hierarchical UIML scene tree from HumanEye perception states (<1ms)."""

    def __init__(self, default_box_size: Tuple[float, float] = (0.08, 0.04)) -> None:
        self.default_box_w, self.default_box_h = default_box_size
        self._counter = 0

    def extract(
        self,
        affordances: List[Tuple[float, float, str, float]],
        category: str = "general_ui",
        category_confidence: float = 1.0,
        foveal_inspection: Optional[Any] = None,
        moving_objects: Optional[List[Any]] = None,
        window_title: Optional[str] = None,
    ) -> UIMLTree:
        """Constructs a UIMLTree snapshot from biological perception outputs."""
        title = window_title or f"Active Desktop ({category})"
        tree = UIMLTree(
            timestamp=time.time(),
            window_title=title,
            primary_category=category,
        )

        elements_list: List[UIMLElement] = []

        # 1. Convert Affordance Hotspots into discrete UIMLElements
        for i, (hx, hy, action_type, conf) in enumerate(affordances):
            role = self._map_action_to_role(action_type)
            hw = self.default_box_w / 2.0
            hh = self.default_box_h / 2.0

            bbox = BoundingBox(
                xmin=max(0.0, hx - hw),
                ymin=max(0.0, hy - hh),
                xmax=min(1.0, hx + hw),
                ymax=min(1.0, hy + hh),
            )

            # Heuristic label based on screen region and category
            label = self._synthesize_label(hx, hy, action_type, category, i + 1)
            elem_id = f"#{action_type[:3]}_{i + 1}"

            elem = UIMLElement(
                element_id=elem_id,
                role=role,
                label=label,
                bbox=bbox,
                is_primary=(i == 0 and conf > 0.5),
            )
            elements_list.append(elem)

        # 2. Enrich with Foveal Inspection if active
        if foveal_inspection is not None:
            # If fovea inspected a target, merge or update with high acuity
            elem_kind = getattr(foveal_inspection, "element_kind", "element")
            f_bbox = getattr(foveal_inspection, "bbox", None)
            f_conf = getattr(foveal_inspection, "confidence", 1.0)

            if f_bbox is not None and len(f_bbox) == 4:
                x1, y1, x2, y2 = f_bbox
                target_box = BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)
                cx, cy = target_box.center

                # Check if it overlaps an existing affordance
                matched = False
                for existing in elements_list:
                    if existing.bbox.iou(target_box) > 0.2 or (
                        abs(existing.bbox.center[0] - cx) < 0.05
                        and abs(existing.bbox.center[1] - cy) < 0.05
                    ):
                        existing.role = self._map_foveal_kind_to_role(elem_kind)
                        existing.bbox = target_box
                        existing.is_active = True
                        matched = True
                        break

                if not matched:
                    # New fine-grained element discovered by high-acuity fovea
                    fovea_role = self._map_foveal_kind_to_role(elem_kind)
                    elem = UIMLElement(
                        element_id=f"#fov_{elem_kind[:3]}",
                        role=fovea_role,
                        label=f"{elem_kind.replace('_', ' ').title()}",
                        bbox=target_box,
                        is_active=True,
                        is_primary=True,
                    )
                    elements_list.append(elem)

        # 3. Add dynamic moving entities if detected
        if moving_objects:
            for idx, m in enumerate(moving_objects):
                mx = getattr(m, "x", 0.5)
                my = getattr(m, "y", 0.5)
                speed = getattr(m, "speed", 0.0)
                if speed > 0.05:
                    m_bbox = BoundingBox(
                        xmin=max(0.0, mx - 0.04),
                        ymin=max(0.0, my - 0.04),
                        xmax=min(1.0, mx + 0.04),
                        ymax=min(1.0, my + 0.04),
                    )
                    elem = UIMLElement(
                        element_id=f"#dyn_{idx + 1}",
                        role=AffordanceType.DRAGGABLE,
                        label="Dynamic Entity",
                        bbox=m_bbox,
                    )
                    elements_list.append(elem)

        # Store in tree indexed by ID
        for elem in elements_list:
            tree.elements[elem.element_id] = elem

        return tree

    def _map_action_to_role(self, action_name: str) -> AffordanceType:
        mapping = {
            "clickable": AffordanceType.CLICKABLE,
            "typeable": AffordanceType.TYPEABLE,
            "scrollable": AffordanceType.SCROLLABLE,
            "draggable": AffordanceType.DRAGGABLE,
        }
        return mapping.get(action_name.lower(), AffordanceType.CLICKABLE)

    def _map_foveal_kind_to_role(self, kind: str) -> AffordanceType:
        k = kind.lower()
        if k in ("button", "icon", "tab", "menu_item", "link"):
            return AffordanceType.CLICKABLE
        elif k in ("input_field", "search_bar", "text_area"):
            return AffordanceType.TYPEABLE
        elif k in ("scrollbar", "scroll_area"):
            return AffordanceType.SCROLLABLE
        elif k in ("slider", "splitter"):
            return AffordanceType.DRAGGABLE
        return AffordanceType.CLICKABLE

    def _synthesize_label(self, x: float, y: float, action: str, category: str, index: int) -> str:
        """Synthesizes human-readable label based on spatial position and category."""
        # Top bar / menu
        if y < 0.10:
            if x < 0.15:
                return "App Menu"
            elif x > 0.85:
                return "Close Window" if x > 0.93 else "Window Controls"
            return f"Header Nav {index}"

        # Sidebar
        if x < 0.20:
            return f"Sidebar Option {index}"

        # Footer / bottom actions
        if y > 0.88:
            if x > 0.70:
                return "Submit / Confirm" if action == "clickable" else "Bottom Action"
            elif x < 0.30:
                return "Status Item"

        # General content region
        if action == "typeable":
            return "Search or Input Field"
        elif action == "scrollable":
            return "Content Scroll View"
        return f"Action Button {index}"
