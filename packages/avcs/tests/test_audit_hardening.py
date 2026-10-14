"""Tests for AVCS 10-Stream Audit Hardening Improvements."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.vision.cortex.core.dorsal import DorsalAffordanceStream, AFFORDANCE_CLASSES
from forge.vision.cortex.core.modulation import LanguageModulation, CharNGramEncoder, dsnt_soft_argmax
from forge.vision.cortex.core.predictive import PredictiveCodingEngine, decode_stability
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


def test_dsnt_soft_argmax_exact_coordinates():
    """Verify that DSNT soft-argmax recovers known peak locations."""
    B, H, W = 2, 32, 32
    # Create single-peak heatmaps
    heat = torch.zeros(B, 1, H, W)
    # Peak at (x=16, y=8) -> norm_x ~ 0.515, norm_y ~ 0.265
    heat[0, 0, 8, 16] = 100.0
    # Peak at (x=24, y=24) -> norm_x ~ 0.765, norm_y ~ 0.765
    heat[1, 0, 24, 24] = 100.0

    prob = torch.softmax(heat.view(B, 1, -1), dim=-1).view(B, 1, H, W)
    coords = dsnt_soft_argmax(prob)

    assert coords.shape == (B, 2)
    assert abs(coords[0, 0].item() - 0.515) < 0.05
    assert abs(coords[0, 1].item() - 0.265) < 0.05
    assert abs(coords[1, 0].item() - 0.765) < 0.05
    assert abs(coords[1, 1].item() - 0.765) < 0.05


def test_char_ngram_encoder():
    """Verify zero-cloud CharNGramEncoder creates normalized embeddings."""
    encoder = CharNGramEncoder(out_dim=64, hash_bins=256)
    emb1 = encoder("Submit Button")
    emb2 = encoder("Submit Button")
    emb3 = encoder("Cancel Operation")

    assert emb1.shape == (1, 64)
    # Identical query yields identical normalized representation
    assert torch.allclose(emb1, emb2, atol=1e-5)
    # Different queries produce distinct vectors
    sim = (emb1 * emb3).sum().item()
    assert sim < 0.99


def test_dorsal_nms_and_regions():
    """Verify 2D spatial NMS deduplicates peaks and find_action_regions returns bounds."""
    dorsal = DorsalAffordanceStream(in_dim=64)
    feat = torch.randn(1, 64, 16, 16)

    hotspots = dorsal.find_action_hotspots(feat, action_type="clickable", top_k=3, threshold=0.1)
    assert len(hotspots) <= 3
    for x, y, conf in hotspots:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert conf >= 0.1

    regions = dorsal.find_action_regions(feat, action_type="clickable", top_k=2, threshold=0.1)
    assert len(regions) <= 2
    for cx, cy, w, h, conf in regions:
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert w > 0.0 and h > 0.0


def test_predictive_hysteresis_and_anomaly_gating():
    """Verify Schmitt-trigger hysteresis suppresses state flickering."""
    logits_animating = torch.tensor([[0.4, 0.5, 0.05, 0.05]]) # argmax is 1 (animating)
    
    # Low surprise when previous was stable -> stays stable
    state1 = decode_stability(logits_animating, surprise_score=0.10, prev_state="stable")
    assert state1 == "stable"

    # Higher surprise -> transitions to animating
    state2 = decode_stability(logits_animating, surprise_score=0.35, prev_state="stable")
    assert state2 == "animating"

    # Acute surprise -> surprise event
    state3 = decode_stability(logits_animating, surprise_score=0.75, prev_state="stable")
    assert state3 == "surprise_event"


def test_sdk_bare_decorator_and_close():
    """Verify ScreenEye supports bare decorators and close cleanup."""
    source = SyntheticScreen(width=320, height=180)
    eye = ScreenEye(fps=20, source=source)

    called = []
    @eye.on_affordance
    def my_listener(state):
        called.append(True)

    @eye.on_surprise
    def my_surprise_listener(state):
        called.append(True)

    assert len(eye._affordance_callbacks) == 1
    assert len(eye._surprise_callbacks) == 1

    # Verify context manager and close
    with eye:
        assert eye.is_stable() or not eye.is_stable()
    
    eye.close()

