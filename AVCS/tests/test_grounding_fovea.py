"""Unit Tests for Phase 2: Open-Vocabulary Language Grounding and Dual-Fovea Acuity Engine."""
from __future__ import annotations

import math
import time
import numpy as np
import pytest
import torch

from avcs.grounding import (
    FOVEAL_ELEMENT_TYPES,
    FovealAcuityEngine,
    FovealAcuityHead,
    FovealInspection,
    GroundingResult,
    LanguageGazeGrounder,
)
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.sdk import HumanEyeClient
from forge.vision.capture import SyntheticScreen


def test_language_gaze_grounder_basic():
    """Verify LanguageGazeGrounder maps text queries to normalized coordinates."""
    grounder = LanguageGazeGrounder(visual_dim=64, text_dim=64)
    feat = torch.randn(1, 64, 16, 24)

    res = grounder.ground(feat, query="Submit Button")
    assert isinstance(res, GroundingResult)
    assert res.query == "Submit Button"
    assert 0.0 <= res.target_coord[0] <= 1.0
    assert 0.0 <= res.target_coord[1] <= 1.0
    assert 0.0 <= res.confidence <= 1.0
    assert res.relevance_map is not None
    assert res.relevance_map.shape == (16, 24)
    assert res.latency_ms > 0.0


def test_language_gaze_grounder_query_caching():
    """Verify query embedding caching speeds up repeated queries."""
    grounder = LanguageGazeGrounder(visual_dim=64, text_dim=64)
    feat = torch.randn(1, 64, 16, 24)

    # First lookup populates cache
    _ = grounder.ground(feat, query="Search Input")
    assert "search input" in grounder._query_cache

    # Second lookup retrieves from cache
    cached_emb = grounder.get_query_embedding("Search Input", device=feat.device)
    assert cached_emb is not None
    assert cached_emb.shape[-1] == 64


def test_language_gaze_grounder_affordance_snapping():
    """Verify grounder snaps coordinates to nearby conscious affordance hotspots."""
    grounder = LanguageGazeGrounder(visual_dim=64, text_dim=64)
    feat = torch.randn(1, 64, 16, 24)

    # First run without affordances to get raw coord
    raw_res = grounder.ground(feat, query="Close Window")
    rx, ry = raw_res.target_coord

    # Place an affordance hotspot right next to raw_res
    hotspot_x = rx + 0.02
    hotspot_y = ry + 0.02
    affordances = [(hotspot_x, hotspot_y, "clickable", 0.95)]

    snapped_res = grounder.ground(
        feat, query="Close Window", affordances=affordances, snap_distance_threshold=0.10
    )
    assert snapped_res.snapped_to_affordance is True
    assert pytest.approx(snapped_res.target_coord[0], abs=1e-3) == hotspot_x
    assert pytest.approx(snapped_res.target_coord[1], abs=1e-3) == hotspot_y


def test_foveal_acuity_engine_inspection():
    """Verify FovealAcuityEngine extracts crop and classifies element properties."""
    engine = FovealAcuityEngine(device="cpu", fovea_radius_frac=0.08)
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

    insp = engine.inspect(frame, centroid=(0.50, 0.50))
    assert isinstance(insp, FovealInspection)
    assert insp.element_kind in FOVEAL_ELEMENT_TYPES
    assert 0.0 <= insp.confidence <= 1.0
    assert len(insp.bbox) == 4
    x1, y1, x2, y2 = insp.bbox
    assert 0.0 <= x1 <= x2 <= 1.0
    assert 0.0 <= y1 <= y2 <= 1.0
    assert insp.contrast_score >= 0.0
    assert insp.foveal_crop_shape[0] > 0 and insp.foveal_crop_shape[1] > 0
    assert insp.latency_ms > 0.0


def test_foveal_acuity_engine_edge_boundaries():
    """Verify foveal crop handles extreme screen borders gracefully."""
    engine = FovealAcuityEngine(device="cpu")
    frame = np.ones((200, 300, 3), dtype=np.uint8) * 128

    for corner in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        insp = engine.inspect(frame, centroid=corner)
        assert insp.element_kind in FOVEAL_ELEMENT_TYPES
        assert len(insp.bbox) == 4


def test_human_eye_look_at_and_foveation():
    """Verify HumanEye.look_at sets gaze, grounds query, and updates fovea."""
    eye = HumanEye(source=SyntheticScreen())
    # Initial observation
    state0 = eye.observe()
    assert state0.active_query is None

    # Ground semantic query
    res = eye.look_at("Login Button")
    assert isinstance(res, GroundingResult)
    assert 0.0 <= res.target_coord[0] <= 1.0
    assert 0.0 <= res.target_coord[1] <= 1.0

    # Next observation should carry the active query and foveal inspection
    state1 = eye.observe()
    assert state1.active_query == "Login Button"
    assert state1.grounding_result is not None
    assert state1.foveal_inspection is not None
    assert state1.foveal_inspection.element_kind in FOVEAL_ELEMENT_TYPES

    # Smooth pursuit gaze should track towards target
    assert math.hypot(
        state1.gaze_fixation[0] - res.target_coord[0],
        state1.gaze_fixation[1] - res.target_coord[1],
    ) < 0.25


def test_look_at_with_explicit_frame_refreshes_visual_features():
    eye = HumanEye(source=SyntheticScreen())
    eye.observe()
    previous_features = eye._last_feat.clone()
    fresh_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    fresh_frame[40:280, 80:550] = 230

    eye.look_at("visible control", frame=fresh_frame)

    assert not torch.equal(eye._last_feat, previous_features)
    assert eye._latest_raw_frame is fresh_frame


def test_sdk_look_at_and_inspect():
    """Verify high-level HumanEyeClient look_at and inspect integration."""
    client = HumanEyeClient(source="synthetic")
    with client:
        # 1. Look at semantic query
        target = client.look_at("Confirm Payment")
        assert len(target) == 2
        assert 0.0 <= target[0] <= 1.0
        assert 0.0 <= target[1] <= 1.0

        # 2. Inspect fovea
        insp = client.inspect()
        assert isinstance(insp, FovealInspection)
        assert insp.element_kind in FOVEAL_ELEMENT_TYPES
        assert insp.confidence >= 0.0

        # 3. State dictionary verification
        state = client.perceive()
        d = state.to_dict()
        assert d["active_query"] == "Confirm Payment"
        assert d["foveal_element"] in FOVEAL_ELEMENT_TYPES
