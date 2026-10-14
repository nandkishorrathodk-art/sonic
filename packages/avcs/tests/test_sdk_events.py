"""Tests for Universal HumanEyeClient SDK and Visual Event Streamer."""
from __future__ import annotations

import time
import numpy as np
import pytest

import avcs
from avcs.auth import LicenseManager, LicenseTier
from avcs.events import VisualEvent, VisualEventStreamer, VisualEventType
from avcs.human_eye import HumanEyeState, MovingObject
from forge.vision.capture import SyntheticScreen


def test_visual_event_streamer_detection():
    streamer = VisualEventStreamer()
    emitted_events: list[VisualEvent] = []

    streamer.subscribe(callback=lambda ev: emitted_events.append(ev))

    # Base initial state
    s0 = HumanEyeState(
        timestamp=time.time(),
        frame_index=1,
        latency_ms=10.0,
        fps=60.0,
        gaze_fixation=(0.5, 0.5),
        motion_magnitude=0.01,
        motion_field=np.zeros((16, 24, 2), dtype=np.float32),
        moving_objects=[],
        pursuit_target=None,
        pursuit_velocity=(0.0, 0.0),
        motor_flow=(0.0, 0.0),
        contact_acquired=False,
        stability="stable",
        surprise_score=0.1,
        affordance_hotspots=[],
        primary_category="editor_code",
        category_confidence=0.8,
    )
    streamer.process_state(s0)
    assert len(emitted_events) == 0

    # 1. Inject surprise alert (explosion or error modal)
    s1 = HumanEyeState(
        timestamp=time.time(),
        frame_index=2,
        latency_ms=10.0,
        fps=60.0,
        gaze_fixation=(0.5, 0.5),
        motion_magnitude=0.12,
        motion_field=np.zeros((16, 24, 2), dtype=np.float32),
        moving_objects=[],
        pursuit_target=None,
        pursuit_velocity=(0.0, 0.0),
        motor_flow=(0.0, 0.0),
        contact_acquired=False,
        stability="surprise",
        surprise_score=0.92,
        affordance_hotspots=[],
        primary_category="editor_code",
        category_confidence=0.8,
    )
    new_evs1 = streamer.process_state(s1)
    assert any(ev.event_type == VisualEventType.SURPRISE_ALERT for ev in new_evs1)

    # 2. Inject scene transition to browser
    s2 = HumanEyeState(
        timestamp=time.time(),
        frame_index=3,
        latency_ms=10.0,
        fps=60.0,
        gaze_fixation=(0.5, 0.5),
        motion_magnitude=0.03,
        motion_field=np.zeros((16, 24, 2), dtype=np.float32),
        moving_objects=[],
        pursuit_target=None,
        pursuit_velocity=(0.0, 0.0),
        motor_flow=(0.0, 0.0),
        contact_acquired=False,
        stability="stable",
        surprise_score=0.2,
        affordance_hotspots=[],
        primary_category="browser_webpage",
        category_confidence=0.75,
    )
    new_evs2 = streamer.process_state(s2)
    assert any(ev.event_type == VisualEventType.SCENE_TRANSITION for ev in new_evs2)


def test_human_eye_client_perceive_and_usage():
    screen = SyntheticScreen(width=640, height=480)
    client = avcs.eye(source=screen, force_virtual_motor=True, fps=60)

    # Perceive single frame
    state = client.perceive()
    assert isinstance(state, HumanEyeState)
    assert state.fps > 0

    # Look at target
    target = client.look_at((0.3, 0.7))
    assert target == (0.3, 0.7)

    # Affordances
    affs = client.get_affordances()
    assert isinstance(affs, list)

    # Events
    events = client.get_events()
    assert isinstance(events, list)

    # Usage metering snapshot
    usage = client.get_usage()
    assert usage["frames_processed"] >= 1
    assert usage["visual_queries"] >= 2
    assert "cost_usd" in usage

    # Signed receipt
    receipt = client.generate_receipt()
    assert receipt.receipt_id.startswith("RCPT_")
    assert receipt.signature != ""

    client.close()


def test_human_eye_client_auth_enforcement():
    screen = SyntheticScreen()

    # Enforce auth without keys -> raises PermissionError
    with pytest.raises(PermissionError):
        avcs.eye(source=screen, enforce_auth=True)

    # Generate valid API key
    mgr = LicenseManager(signing_key="master_key_secret")
    key = mgr.generate_api_key(org_id="org_test_runner", tier=LicenseTier.PRO)

    # Initialize with valid key
    client = avcs.eye(source=screen, api_key=key, signing_key="master_key_secret", enforce_auth=True)
    assert client.is_authenticated is True
    assert client.org_id == "org_test_runner"
    client.close()


def test_human_eye_client_background_stream():
    screen = SyntheticScreen()
    client = avcs.eye(source=screen, force_virtual_motor=True, fps=60)

    client.start()
    time.sleep(0.20)
    client.stop()

    usage = client.get_usage()
    # At 60 FPS for 200ms, should process multiple frames
    assert usage["frames_processed"] >= 2
    client.close()
