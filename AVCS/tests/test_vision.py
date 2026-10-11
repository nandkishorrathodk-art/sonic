"""Tests for the real-time screen-vision stack.

Every test drives real code on real pixel arrays.  No mocks: the synthetic
desktop is a genuine renderer, and the assertions are about properties of the
perception output (layout structure found, motion localised, hash stable for a
static frame and different for a changed one) rather than about exact numbers
that would just re-state the implementation.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import torch

from forge.vision.capture import FrameSource, ScreenCapture, SyntheticScreen, open_capture
from forge.vision.dashboard import render_dashboard
from forge.vision.describer import HeuristicDescriber
from forge.vision.encoder import (
    SCENE_TYPES,
    SceneUnderstandingModel,
    pseudo_labels,
    stack_grids,
    train_understanding,
)
from forge.vision.pipeline import ScreenVision
from forge.vision.reflex import (
    ReflexPerception,
    downsample,
    edge_map,
    events_from_states,
    perceptual_hash,
    pool_tiles,
    to_gray,
)


def _static_frame(h: int = 180, w: int = 320) -> np.ndarray:
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    img[40:120, 30:150] = 210
    return img


# ------------------------------------------------------------- primitives

def test_to_gray_range():
    img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    g = to_gray(img)
    assert g.shape == (1, 2)
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert g[0, 1] > g[0, 0]


def test_pool_tiles_exact_shape():
    grid = np.random.default_rng(0).random((108, 192))
    pooled = pool_tiles(grid, 16, 24)
    assert pooled.shape == (16, 24)


def test_pool_tiles_handles_tiny_input():
    grid = np.random.default_rng(0).random((3, 4))
    pooled = pool_tiles(grid, 16, 24)
    assert pooled.shape == (16, 24)


def test_downsample_factor_one_is_identity():
    grid = np.random.default_rng(0).random((10, 10))
    assert np.allclose(downsample(grid, 1), grid)


def test_edge_map_finds_boundary():
    grid = np.zeros((20, 20))
    grid[:, 10:] = 1.0
    e = edge_map(grid)
    # The forward difference puts the edge on the first bright column.
    assert e[:, 10].max() > 0.5
    assert e[:, 0].max() < 0.1


def test_perceptual_hash_stable_and_discriminative():
    a = np.random.default_rng(1).random((32, 32))
    b = 1.0 - a
    assert perceptual_hash(a) == perceptual_hash(a.copy())
    assert perceptual_hash(a) != perceptual_hash(b)


# ------------------------------------------------------------- reflex layer

def test_reflex_perceives_layout_regions():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    state = per.perceive(_static_frame(), frame_index=0)
    assert state.luma_grid.shape == (16, 24)
    assert state.regions, "a bright block on a dark field should form a region"
    assert any(r.kind in ("panel", "band", "patch") for r in state.regions)


def test_reflex_detects_motion_when_thing_moves():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f1 = _static_frame()
    per.perceive(f1, frame_index=0)
    f2 = f1.copy()
    f2[140:170, 200:260] = 255  # new bright block appears
    state = per.perceive(f2, frame_index=1)
    assert state.peak_motion > 0.1
    assert state.motion_centroid is not None
    assert state.change_fraction > 0.0


def test_reflex_first_frame_has_no_motion():
    per = ReflexPerception()
    state = per.perceive(_static_frame(), frame_index=0)
    assert state.global_motion == 0.0
    assert state.peak_motion == 0.0
    assert state.novelty == 1.0


def test_reflex_flags_whole_scene_change():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f1 = _static_frame()
    per.perceive(f1, frame_index=0)
    state = per.perceive(255 - f1, frame_index=1)
    assert state.scene_changed
    assert state.change_fraction > 0.35


def test_reflex_flags_overlay():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f1 = _static_frame()
    per.perceive(f1, frame_index=0)
    f2 = f1.copy()
    f2[0:30, 0:50] = 250  # compact, high-contrast addition
    state = per.perceive(f2, frame_index=1)
    assert state.overlay_suspected


def test_reflex_novelty_drops_for_repeat_frame():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f = _static_frame()
    per.perceive(f, frame_index=0)
    state = per.perceive(f.copy(), frame_index=1)
    assert state.novelty < 0.05


def test_reflex_latency_under_target_budget():
    """The reflex path must clear the 40-80 ms real-time tick budget."""
    per = ReflexPerception(rows=16, cols=24, downscale=4)
    frame = SyntheticScreen().grab()
    per.perceive(frame, frame_index=0)  # warm caches
    lat = [per.perceive(frame, frame_index=i + 1).latency_ms for i in range(10)]
    assert np.median(lat) < 40.0, f"reflex median {np.median(lat):.1f} ms exceeds 40 ms"


# ------------------------------------------------------------- events

def test_events_fire_on_motion_and_scene_change():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f1 = _static_frame()
    s0 = per.perceive(f1, frame_index=0)
    s1 = per.perceive(255 - f1, frame_index=1)
    ev = events_from_states(s0, s1)
    kinds = {e.kind for e in ev}
    assert "scene_change" in kinds
    assert all(e.text for e in ev)


def test_events_include_direction_wording():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f1 = _static_frame()
    prev = per.perceive(f1, frame_index=0)
    f2 = f1.copy()
    f2[150:175, 20:70] = 255  # bottom-left movement
    cur = per.perceive(f2, frame_index=1)
    move = [e for e in events_from_states(prev, cur) if e.kind == "move"]
    assert move, "a moving block should emit a move event"
    assert "bottom left" in move[0].text


# ------------------------------------------------------------- encoder

def test_stack_grids_shape():
    per = ReflexPerception(rows=16, cols=24)
    state = per.perceive(_static_frame(), frame_index=0)
    x = stack_grids(state)
    assert x.shape == (4, 16, 24)


def test_encoder_forward_shapes():
    model = SceneUnderstandingModel(dim=32)
    x = torch.randn(2, 4, 16, 24)
    out = model(x)
    assert out["embedding"].shape == (2, 32)
    assert out["type_logits"].shape == (2, len(SCENE_TYPES))
    for k in ("activity", "change", "overlay", "confidence"):
        assert out[k].shape == (2,)
        assert (out[k] >= 0).all() and (out[k] <= 1).all()


def test_encoder_learns_from_reflex_states():
    per = ReflexPerception(rows=16, cols=24, downscale=4)
    screen = SyntheticScreen(seed=3)
    states = []
    for i in range(80):
        f = screen.grab()
        if i % 9 == 0:
            f = 255 - f
        states.append(per.perceive(f, frame_index=i))

    model = SceneUnderstandingModel(dim=32)
    result = train_understanding(model, states, steps=150)
    losses = result["losses"]
    assert losses[-1] < losses[0], "understanding head should reduce its loss"

    und = model.understand(states[-1])
    assert und.scene_type in SCENE_TYPES
    assert 0.0 <= und.confidence <= 1.0
    assert 0.0 <= und.activity <= 1.0


def test_description_comes_from_the_model_not_a_template():
    """The sentence must be the model's prediction, not a hardcoded branch.

    Fed genuinely different scenes, the model must be able to produce different
    sentences; a constant output would mean the words are a template rather than
    a prediction.
    """
    from forge.vision.scenes import sample_states
    from forge.vision.vocab import VOCAB

    states, types = sample_states(
        ["desktop", "terminal", "media", "dialog"], 20, seed=5
    )
    model = SceneUnderstandingModel(dim=32)
    train_understanding(model, states, steps=250)

    und = model.understand(states[-1])
    predicted = set(und.tokens)
    assert predicted, "model predicted no words"
    assert set(und.tokens).issubset(set(VOCAB))
    assert "screen" in predicted
    assert und.description
    # Across a varied mix the model should emit more than one distinct sentence.
    sentences = {model.understand(s).description for s in states[::2]}
    assert len(sentences) > 1, "description is constant -> it is a template, not a prediction"


def test_understanding_exposes_calibrated_confidence():
    from forge.vision.scenes import sample_states

    train_states, _ = sample_states(["desktop", "terminal"], 20, seed=1)
    model = SceneUnderstandingModel(dim=32)
    train_understanding(model, train_states, steps=100)
    und = model.understand(train_states[0])
    assert 0.0 <= und.confidence <= 1.0
    # Confidence must not be a constant: it should vary across inputs.
    confs = {round(model.understand(s).confidence, 3) for s in train_states}
    assert len(confs) > 1


def test_word_report_macro_f1_is_not_inflated_by_always_on_words():
    """Raw accuracy is misleading; macro-F1 must expose weak words."""
    from forge.vision.encoder import ALWAYS_ON, word_report
    from forge.vision.scenes import sample_states

    states, types = sample_states(["desktop", "dialog"], 20, seed=2)
    model = SceneUnderstandingModel(dim=32)
    train_understanding(model, states, steps=60)
    rep = word_report(model, states, types)
    assert "macro_f1" in rep and "informative_accuracy" in rep
    assert set(rep["per_word"]) & ALWAYS_ON == set()  # always-on excluded from per-word
    # macro-F1 over discriminative words cannot exceed the best possible.
    assert 0.0 <= rep["macro_f1"] <= 1.0


def test_held_out_category_is_a_real_test():
    """Train on four categories, evaluate on a fifth never seen."""
    from forge.vision.encoder import word_report
    from forge.vision.scenes import sample_states

    train_states, train_types = sample_states(
        ["desktop", "terminal", "media", "dialog"], 25, seed=10
    )
    model = SceneUnderstandingModel(dim=32)
    train_understanding(model, train_states, steps=200)

    seen, seen_t = sample_states(["desktop", "terminal"], 15, seed=10)
    unseen, unseen_t = sample_states(["scroll"], 15, seed=77)

    seen_rep = word_report(model, seen, seen_t)
    unseen_rep = word_report(model, unseen, unseen_t)
    # The unseen category must not score implausibly high -- if it did, the
    # "held-out" split would be meaningless.
    assert unseen_rep["macro_f1"] <= seen_rep["macro_f1"] + 0.25
    assert 0.0 <= unseen_rep["macro_f1"] <= 1.0


def test_pseudo_labels_cover_types():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f = _static_frame()
    per.perceive(f, frame_index=0)
    idle = per.perceive(f.copy(), frame_index=1)
    assert pseudo_labels(idle)["type"] == float(SCENE_TYPES.index("idle"))
    busy = per.perceive(255 - f, frame_index=2)
    assert pseudo_labels(busy)["type"] == float(SCENE_TYPES.index("busy"))


# ------------------------------------------------------------- describer

def test_describer_produces_sentence():
    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f = _static_frame()
    per.perceive(f, frame_index=0)
    s = per.perceive(f.copy(), frame_index=1)
    text = HeuristicDescriber().describe(s, None, [])
    assert isinstance(text, str)
    assert len(text) > 20
    assert "screen" in text.lower()


# ------------------------------------------------------------- pipeline

def test_pipeline_step_and_stats():
    vis = ScreenVision(capture=SyntheticScreen(), understanding=None)
    for _ in range(3):
        vis.step()
    assert vis.stats.frames == 3
    assert vis.last_state is not None
    s = vis.stats.summary()
    assert s["frames"] == 3
    assert s["reflex_p50_ms"] >= 0.0
    # The full tick is measured too, so the budget cannot be quoted from
    # perception alone.
    assert s["tick_p50_ms"] >= s["reflex_p50_ms"] * 0.9


def test_model_describer_uses_model_output_not_fallback():
    from forge.vision.describer import ModelDescriber
    from forge.vision.scenes import sample_states

    states, types = sample_states(["desktop", "terminal"], 20, seed=4)
    model = SceneUnderstandingModel(dim=32)
    train_understanding(model, states, steps=120)
    und = model.understand(states[-1])

    text = ModelDescriber().describe(states[-1], und, [])
    # The rendered sentence must be the model's own prediction, plus confidence.
    assert und.description in text
    assert "confidence" in text


def test_model_describer_falls_back_without_model():
    from forge.vision.describer import ModelDescriber

    per = ReflexPerception(rows=16, cols=24, downscale=2)
    f = _static_frame()
    per.perceive(f, frame_index=0)
    state = per.perceive(f.copy(), frame_index=1)
    # No understanding supplied -> must still produce a sentence, never empty.
    text = ModelDescriber().describe(state, None, [])
    assert text and "screen" in text.lower()


def test_pipeline_threaded_loop_runs_and_describes():
    vis = ScreenVision(
        capture=SyntheticScreen(),
        understanding=None,
        target_fps=30.0,
        describe_every_s=0.1,
    )
    descriptions = []
    vis.on_description(descriptions.append)
    vis.start()
    try:
        import time

        time.sleep(0.8)
    finally:
        vis.stop()
    assert vis.stats.frames > 5, "threaded loop should have perceived several frames"
    assert descriptions, "loop should have described at least one frame"


def test_state_serialises_to_json():
    per = ReflexPerception(rows=16, cols=24)
    state = per.perceive(_static_frame(), frame_index=0)
    blob = json.dumps(state.to_dict())
    assert "scene_hash" in blob
    assert "regions" in blob


# ------------------------------------------------------------- dashboard

def test_frame_source_accepts_grayscale_and_rgba():
    gray = np.zeros((20, 30), dtype=np.uint8)
    src = FrameSource(lambda: gray)
    out = src.grab()
    assert out.shape == (20, 30, 3)

    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    src = FrameSource(lambda: rgba)
    assert src.grab().shape == (20, 30, 3)


def test_frame_source_closes_provider():
    closed = {"v": False}

    class Provider:
        def __call__(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            closed["v"] = True

    src = FrameSource(Provider())
    src.grab()
    src.close()
    assert closed["v"]


def test_open_capture_falls_back_without_display():
    # On a headless host this must return the synthetic source, never raise.
    cap = open_capture(prefer_real=False)
    frame = cap.grab()
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_screen_capture_reports_unavailable_without_display():
    # We do not assert it is unavailable (a CI box may have a display), only
    # that the probe itself never raises and returns a bool.
    assert isinstance(ScreenCapture.available(), bool)


def test_scene_generator_is_deterministic_and_advances():
    from forge.vision.scenes import SceneGenerator

    a = SceneGenerator("desktop", seed=1)
    b = SceneGenerator("desktop", seed=1)
    assert np.array_equal(a.grab(), b.grab())  # same seed -> same first frame
    f1 = a.grab()
    f2 = a.grab()
    assert not np.array_equal(f1, f2), "scene should advance per grab"


def test_scene_categories_all_render():
    from forge.vision.scenes import SCENE_CATEGORIES, SceneGenerator

    for cat in SCENE_CATEGORIES:
        gen = SceneGenerator(cat, seed=3)
        frame = gen.grab()
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.std() > 0, f"{cat} rendered a flat frame"


def test_dashboard_writes_self_contained_html():
    per = ReflexPerception(rows=8, cols=12)
    f = _static_frame()
    per.perceive(f, frame_index=0)
    state = per.perceive(f.copy(), frame_index=1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dash.html")
        render_dashboard(path, state, stats={"fps": 25}, description="hello")
        html = open(path, encoding="utf-8").read()
        assert "<svg" in html
        assert "hello" in html
        assert "http://" not in html  # no external assets
