"""Unit Tests for Phase 3: Semantic DOM-less Scene Tree & Multi-Turn Episodic Memory."""
from __future__ import annotations

import time
import numpy as np
import pytest

from avcs.memory import (
    AffordanceType,
    BoundingBox,
    EntityLifecycle,
    PersistentUIRegistry,
    SemanticSceneTreeExtractor,
    UIMLElement,
    UIMLTree,
    VisualDiff,
    VisualDiffEngine,
    VisualMemoryManager,
    VisualMemorySnapshot,
)
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.grounding import FovealInspection
from avcs.sdk import HumanEyeClient
from forge.vision.capture import SyntheticScreen


def test_bounding_box_geometry():
    """Verify BoundingBox geometry, containment, and IoU computation."""
    b1 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)
    assert b1.center == (0.3, 0.3)
    assert pytest.approx(b1.width, abs=1e-5) == 0.4
    assert pytest.approx(b1.height, abs=1e-5) == 0.4
    assert pytest.approx(b1.area, abs=1e-5) == 0.16

    # Sub-box containment
    b2 = BoundingBox(xmin=0.2, ymin=0.2, xmax=0.4, ymax=0.4)
    assert b1.contains(b2) is True
    assert b2.contains(b1) is False

    # Disjoint box
    b3 = BoundingBox(xmin=0.6, ymin=0.6, xmax=0.9, ymax=0.9)
    assert b1.iou(b3) == 0.0

    # Overlapping box
    b4 = BoundingBox(xmin=0.3, ymin=0.3, xmax=0.7, ymax=0.7)
    iou = b1.iou(b4)
    assert 0.1 < iou < 0.5


def test_uiml_element_and_tree_serialization():
    """Verify UIMLElement formats to indented UIML and token estimation."""
    box = BoundingBox(xmin=0.8, ymin=0.88, xmax=0.95, ymax=0.95)
    elem = UIMLElement(
        element_id="#b1",
        role=AffordanceType.CLICKABLE,
        label="Submit Order",
        bbox=box,
        is_primary=True,
    )
    uiml_str = elem.to_uiml(indent=2)
    assert '[CLICKABLE: "Submit Order"' in uiml_str
    assert "id=#b1" in uiml_str
    assert "primary" in uiml_str

    tree = UIMLTree(
        timestamp=time.time(),
        window_title="Checkout Screen",
        primary_category="ecommerce",
        elements={"#b1": elem},
    )
    tree_text = tree.to_string()
    assert '[WINDOW: "Checkout Screen" class=ecommerce]' in tree_text
    assert tree.token_estimate() < 50
    assert tree.find_by_id("#b1") == elem
    assert len(tree.find_by_label("Submit")) == 1


def test_scene_tree_extractor():
    """Verify SemanticSceneTreeExtractor builds tree from affordance hotspots and fovea."""
    extractor = SemanticSceneTreeExtractor()
    affordances = [
        (0.85, 0.92, "clickable", 0.95),
        (0.50, 0.30, "typeable", 0.80),
        (0.10, 0.50, "scrollable", 0.60),
    ]

    fovea = FovealInspection(
        centroid=(0.85, 0.92),
        element_kind="button",
        confidence=0.98,
        bbox=(0.82, 0.89, 0.88, 0.95),
        contrast_score=0.75,
        foveal_crop_shape=(64, 64),
        latency_ms=1.2,
    )

    tree = extractor.extract(
        affordances=affordances,
        category="checkout_page",
        foveal_inspection=fovea,
        window_title="Store Checkout",
    )

    assert isinstance(tree, UIMLTree)
    assert len(tree.elements) >= 3
    # Check that high-acuity fovea merged with or enriched the submit button
    matched_buttons = [e for e in tree.elements.values() if e.role == AffordanceType.CLICKABLE]
    assert len(matched_buttons) >= 1
    assert tree.token_estimate() < 150


def test_persistent_ui_registry_reid():
    """Verify PersistentUIRegistry tracks persistent IDs and compensates for scroll flow."""
    registry = PersistentUIRegistry(cost_threshold=0.45)

    # Turn 1: Initial elements
    elem1 = UIMLElement(
        element_id="temp_1",
        role=AffordanceType.CLICKABLE,
        label="Search",
        bbox=BoundingBox(0.2, 0.2, 0.3, 0.25),
    )
    elem2 = UIMLElement(
        element_id="temp_2",
        role=AffordanceType.TYPEABLE,
        label="Query Input",
        bbox=BoundingBox(0.35, 0.2, 0.6, 0.25),
    )

    active1, disappeared1 = registry.update([elem1, elem2])
    assert len(active1) == 2
    id1, id2 = active1[0].element_id, active1[1].element_id
    assert id1 != id2
    assert disappeared1 == []

    # Turn 2: Elements shift slightly due to layout render (2% shift)
    elem1_shifted = UIMLElement(
        element_id="temp_3",
        role=AffordanceType.CLICKABLE,
        label="Search",
        bbox=BoundingBox(0.21, 0.21, 0.31, 0.26),
    )
    elem2_shifted = UIMLElement(
        element_id="temp_4",
        role=AffordanceType.TYPEABLE,
        label="Query Input",
        bbox=BoundingBox(0.36, 0.21, 0.61, 0.26),
    )

    active2, disappeared2 = registry.update([elem1_shifted, elem2_shifted])
    assert len(active2) == 2
    # Persistent IDs must match turn 1
    assert active2[0].element_id == id1
    assert active2[1].element_id == id2
    assert disappeared2 == []

    # Turn 3: Container optical scroll down (dy = +0.10)
    # Search scrolls up / disappears, new item appears at bottom
    elem_new = UIMLElement(
        element_id="temp_5",
        role=AffordanceType.CLICKABLE,
        label="Next Page",
        bbox=BoundingBox(0.5, 0.85, 0.7, 0.90),
    )
    active3, disappeared3 = registry.update([elem_new], container_flow=(0.0, 0.10))
    assert len(active3) == 1
    assert active3[0].label == "Next Page"
    # Old elements should transition to off-screen
    assert registry.lifecycles[id1] in (EntityLifecycle.OFF_SCREEN_BELOW, EntityLifecycle.OFF_SCREEN_ABOVE)


def test_visual_diff_engine():
    """Verify VisualDiffEngine computes minimal state deltas and causal narrative."""
    engine = VisualDiffEngine()

    # Turn 1: Initial tree
    e1 = UIMLElement("#e1", AffordanceType.CLICKABLE, "Save", BoundingBox(0.8, 0.8, 0.9, 0.9))
    e2 = UIMLElement("#e2", AffordanceType.TYPEABLE, "Input", BoundingBox(0.2, 0.2, 0.5, 0.3))
    tree1 = UIMLTree(timestamp=1.0, window_title="App", primary_category="editor", elements={"#e1": e1, "#e2": e2})

    diff1 = engine.compute_diff(tree1, prior_action="launch", turn_index=1)
    assert diff1.turn_index == 1
    assert len(diff1.added_elements) == 2
    assert "Initial screen" in diff1.causal_summary

    # Turn 2: User clicks Save, Modal dialog appears (+ #m1)
    e_modal = UIMLElement("#m1", AffordanceType.CLICKABLE, "Confirm Dialog", BoundingBox(0.4, 0.4, 0.6, 0.6))
    tree2 = UIMLTree(
        timestamp=2.0,
        window_title="App",
        primary_category="editor",
        elements={"#e1": e1, "#e2": e2, "#m1": e_modal},
    )

    diff2 = engine.compute_diff(tree2, prior_action="click(#e1 - Save)", turn_index=2)
    assert diff2.turn_index == 2
    assert len(diff2.added_elements) == 1
    assert diff2.added_elements[0].element_id == "#m1"
    assert "click(#e1 - Save)" in diff2.causal_summary
    assert diff2.token_estimate() <= 50


def test_visual_memory_manager_multi_turn():
    """Verify VisualMemoryManager orchestrates multi-turn tracking and context prompts."""
    mem = VisualMemoryManager()

    # Turn 1
    aff1 = [(0.1, 0.1, "clickable", 0.9), (0.5, 0.5, "typeable", 0.8)]
    diff1 = mem.step(aff1, category="editor", prior_action="open_file")
    assert mem.turn_index == 1
    assert mem.total_tokens_avoided > 1000

    # Turn 2
    aff2 = [(0.1, 0.1, "clickable", 0.9), (0.5, 0.5, "typeable", 0.8), (0.9, 0.9, "clickable", 0.95)]
    diff2 = mem.step(aff2, category="editor", prior_action="click(#e1)")
    assert mem.turn_index == 2
    assert mem.total_tokens_avoided > 2500

    prompt = mem.get_context_prompt(max_delta_turns=5)
    assert "AVCS DOM-less UIML" in prompt
    assert "Recent Visual State Transitions" in prompt
    assert len(prompt) < 1500  # <300 tokens

    snap = mem.get_snapshot()
    assert snap.total_turns == 2
    assert snap.cumulative_tokens_saved > 2500
    assert snap.estimated_dollar_savings > 0.0


def test_human_eye_and_sdk_integration():
    """Verify HumanEye and HumanEyeClient episodic memory integration."""
    client = HumanEyeClient(source="synthetic")
    with client:
        # 1. First perception tick
        state = client.perceive()
        assert state.scene_tree is not None
        assert isinstance(state.scene_tree, UIMLTree)
        assert state.visual_diff is not None
        assert isinstance(state.visual_diff, VisualDiff)

        # 2. Query scene tree and diff
        tree = client.get_scene_tree()
        assert isinstance(tree, UIMLTree)
        assert len(tree.elements) > 0

        diff = client.get_visual_diff(prior_action="look_at('Submit')")
        assert isinstance(diff, VisualDiff)

        # 3. Context prompt
        prompt = client.get_context_prompt()
        assert "AVCS DOM-less UIML" in prompt

        # 4. Snapshot
        snap = client.get_memory_snapshot()
        assert snap.total_turns >= 1
        assert snap.cumulative_tokens_saved > 0
