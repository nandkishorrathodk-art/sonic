"""Unit tests for the official AVCS Developer SDK Package (avcs-sdk)."""
from __future__ import annotations

import os
from pathlib import Path

import avcs
import pytest
from avcs import (
    ArtificialVisualCortex,
    CortexState,
    ScreenEye,
    SyntheticScreen,
    get_weights_path,
)


def _has_weights_checkpoint() -> bool:
    filename = "avcs_cortex_weights.pt"
    custom_path = os.environ.get("AVCS_WEIGHTS_PATH")
    candidates = [
        Path(custom_path) if custom_path else None,
        Path(__file__).resolve().parents[1] / "runs" / "weights" / filename,
        Path(avcs.__file__).resolve().parent / filename,
    ]
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AVCS" / "weights"
    if os.name != "nt":
        cache_root = Path.home() / ".cache" / "avcs" / "weights"
    candidates.append(cache_root / filename)
    return any(path is not None and path.is_file() for path in candidates)


def test_avcs_import_and_metadata():
    """Verify top-level package metadata and exports."""
    assert avcs.__version__ == "0.2.0"
    assert "ScreenEye" in avcs.__all__
    assert "ArtificialVisualCortex" in avcs.__all__
    assert "VisualServoingController" in avcs.__all__
    assert "SyntheticScreen" in avcs.__all__
    assert "load_avcs_weights" in avcs.__all__


@pytest.mark.skipif(
    not _has_weights_checkpoint(),
    reason="The optional pretrained AVCS checkpoint is not available in this checkout.",
)
def test_screen_eye_initialization_with_pretrained_weights():
    """Verify ScreenEye initializes cleanly and automatically binds pre-trained weights."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source, load_pretrained=True)

    assert eye is not None
    assert eye.fps == 25
    assert eye.weights_loaded is True
    assert isinstance(eye.cortex, ArtificialVisualCortex)


def test_screen_eye_synchronous_perception():
    """Verify instantaneous perception tick returns structured CortexState."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    state = eye.perceive_now()
    assert isinstance(state, CortexState)
    assert 0.0 <= state.surprise_score <= 1.0
    assert state.stability_state in avcs.STABILITY_STATES
    assert state.fixation is not None
    assert 0.0 <= state.fixation[0] <= 1.0
    assert 0.0 <= state.fixation[1] <= 1.0


def test_screen_eye_affordance_hotspots():
    """Verify extraction of clickable and typeable affordances."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    clickables = eye.clickable_hotspots()
    typeables = eye.typeable_hotspots()

    assert isinstance(clickables, list)
    assert isinstance(typeables, list)
    for spot in clickables:
        x, y, conf = spot
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert 0.0 <= conf <= 1.0


def test_screen_eye_reactive_surprise_bare_decorator():
    """Verify @eye.on_surprise bare decorator syntax."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    events_received = []

    @eye.on_surprise
    def handle_surprise(state: CortexState):
        events_received.append(state.surprise_score)

    dummy_state = eye.perceive_now()
    # Force high surprise
    dummy_state.surprise_score = 0.85
    eye._dispatch_surprise(dummy_state)

    assert len(events_received) == 1
    assert events_received[0] == 0.85


def test_screen_eye_reactive_surprise_parameterized_decorator():
    """Verify @eye.on_surprise(threshold=...) parameterized syntax filters events."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    low_events = []
    high_events = []

    @eye.on_surprise(threshold=0.2)
    def on_minor(state: CortexState):
        low_events.append(state)

    @eye.on_surprise(threshold=0.7)
    def on_major(state: CortexState):
        high_events.append(state)

    test_state = eye.perceive_now()
    test_state.surprise_score = 0.5
    eye._dispatch_surprise(test_state)

    assert len(low_events) == 1
    assert len(high_events) == 0

    test_state.surprise_score = 0.9
    eye._dispatch_surprise(test_state)

    assert len(low_events) == 2
    assert len(high_events) == 1


def test_screen_eye_reactive_affordance_decorator():
    """Verify @eye.on_affordance decorator filters by interactive action type."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    affordance_events = []

    @eye.on_affordance(action_type="clickable", min_confidence=0.4)
    def on_button_visible(state: CortexState):
        affordance_events.append(state)

    state = eye.perceive_now()
    state.action_hotspots = [(0.5, 0.5, "clickable", 0.95)]
    eye._dispatch_state(state)

    assert len(affordance_events) == 1


def test_screen_eye_hand_eye_servoing():
    """Verify coordinate-free visual servoing velocity computation."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    # Hand at top-left, gaze at bottom-right
    current_cursor = (0.1, 0.1)
    target_gaze = (0.8, 0.8)

    (vx, vy), contact = eye.steer_hand(current_cursor=current_cursor, target_gaze=target_gaze)
    assert vx > 0.0  # Must steer right
    assert vy > 0.0  # Must steer down
    assert contact is False

    # When hand is on target
    aligned_cursor = (0.801, 0.802)
    _, aligned_contact = eye.steer_hand(current_cursor=aligned_cursor, target_gaze=target_gaze)
    assert aligned_contact is True
    assert eye.is_aligned(aligned_cursor, target_gaze=target_gaze) is True


def test_screen_eye_look_at_and_locate():
    """Verify language-directed attention and locate alias."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)

    gaze = eye.look_at("Submit Order Button")
    assert isinstance(gaze, tuple)
    assert len(gaze) == 2
    assert 0.0 <= gaze[0] <= 1.0
    assert 0.0 <= gaze[1] <= 1.0

    loc = eye.locate("Cancel Button")
    assert isinstance(loc, tuple)
    assert len(loc) == 2


@pytest.mark.skipif(
    not _has_weights_checkpoint(),
    reason="The optional pretrained AVCS checkpoint is not available in this checkout.",
)
def test_weights_auto_discovery():
    """Verify weights discovery locates runs/weights/avcs_cortex_weights.pt."""
    weights_path = get_weights_path()
    assert weights_path is not None
    assert weights_path.endswith("avcs_cortex_weights.pt")
