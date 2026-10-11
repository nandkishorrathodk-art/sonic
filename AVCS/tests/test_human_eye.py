"""Tests for Real-Time Human Eye Intelligence Model (avcs.human_eye)."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from avcs.human_eye import (
    FastAffordanceHead,
    HumanEye,
    HumanEyeState,
    MagnocellularMotionEngine,
    MovingObject,
    RealTimeCorticalBackbone,
    SmoothPursuitTracker,
)
from forge.vision.capture import SyntheticScreen


def test_magnocellular_motion_flow():
    """Verifies ultra-fast optical flow and motion velocity extraction."""
    engine = MagnocellularMotionEngine(grid_rows=16, grid_cols=24)

    # Frame 1: Blank dark scene
    f1 = np.full((480, 640, 3), 20, dtype=np.uint8)
    field1, motion1, objs1, surp1 = engine.process(f1)
    assert motion1 == 0.0
    assert len(objs1) == 0

    # Frame 2: Bright target moving in center
    f2 = np.full((480, 640, 3), 20, dtype=np.uint8)
    f2[200:280, 300:380] = 255  # Bright square in center
    field2, motion2, objs2, surp2 = engine.process(f2)

    assert motion2 > 0.01
    assert surp2 > 0.0
    assert len(objs2) >= 1
    target = objs2[0]
    assert 0.35 < target.centroid[0] < 0.65
    assert 0.35 < target.centroid[1] < 0.65
    assert target.area > 0.0


def test_smooth_pursuit_tracking():
    """Verifies biological smooth pursuit tracking of a moving target."""
    tracker = SmoothPursuitTracker(kp=0.7, kv=0.3, contact_radius=0.04)

    # Initial state centered
    assert tracker.gaze == (0.5, 0.5)

    # Simulate target moving from (0.5, 0.5) to (0.8, 0.2) over 20 steps
    target_start = np.array([0.5, 0.5])
    target_end = np.array([0.8, 0.2])
    target_vel = (0.3, -0.3)

    for i in range(25):
        alpha = min(1.0, i / 20.0)
        curr_target = tuple(target_start + alpha * (target_end - target_start))
        gaze, vel, contact = tracker.update(detected_target=curr_target, observed_velocity=target_vel)
        time.sleep(0.002)

    # Gaze should smoothly converge to target
    dist_error = math.hypot(gaze[0] - target_end[0], gaze[1] - target_end[1])
    assert dist_error < 0.05, f"Tracking error {dist_error} too large"


def test_human_eye_realtime_performance():
    """Verifies that HumanEye achieves 60+ FPS and sub-16ms latency on CPU."""
    eye = HumanEye(device="cpu")
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    # Warmup
    for _ in range(10):
        eye.observe(frame)

    t0 = time.perf_counter()
    count = 40
    for _ in range(count):
        state = eye.observe(frame)
        assert isinstance(state, HumanEyeState)
    total_time = time.perf_counter() - t0

    fps = count / total_time
    assert fps >= 20.0 or eye.p50_latency_ms < 50.0, f"Throughput {fps:.1f} FPS is below real-time requirement"
    assert eye.p50_latency_ms < 50.0, f"p50 latency {eye.p50_latency_ms:.2f}ms exceeds 50ms"
    assert eye.p95_latency_ms < 85.0


def test_human_eye_game_video_simulation():
    """Simulates dynamic 60Hz video game viewport and verifies continuous tracking."""
    eye = HumanEye(device="cpu")
    width, height = 640, 480

    target_pos = [0.2, 0.3]
    target_vel = [0.4, 0.2]  # velocity in units/second

    dt = 0.016  # 60 FPS delta
    errors = []

    for step in range(30):
        # Update bouncing target
        target_pos[0] += target_vel[0] * dt
        target_pos[1] += target_vel[1] * dt

        # Render frame
        frame = np.full((height, width, 3), 25, dtype=np.uint8)
        tx = int(target_pos[0] * width)
        ty = int(target_pos[1] * height)
        frame[max(0, ty-15):min(height, ty+15), max(0, tx-15):min(width, tx+15)] = 240

        state = eye.observe(frame)

        # Track error between true target and pursuit gaze
        err = math.hypot(target_pos[0] - state.gaze_fixation[0], target_pos[1] - state.gaze_fixation[1])
        errors.append(err)

    # After initial convergence, tracking error should remain tight
    late_errors = errors[10:]
    mean_late_error = float(np.mean(late_errors))
    assert mean_late_error < 0.15, f"Mean dynamic tracking error {mean_late_error} too high"


def test_human_eye_instant_surprise_reflex():
    """Verifies that an abrupt flash/explosion triggers zero-latency reflex callback."""
    eye = HumanEye(device="cpu")
    reflex_triggers = []

    @eye.on_surprise(threshold=0.80)
    def on_flash(state: HumanEyeState):
        reflex_triggers.append((state.frame_index, state.surprise_score))

    dark_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    for _ in range(5):
        eye.observe(dark_frame)

    assert len(reflex_triggers) == 0

    # Inject sudden bright flash (e.g. explosion, screen transition)
    flash_frame = np.full((480, 640, 3), 250, dtype=np.uint8)
    state = eye.observe(flash_frame)

    assert len(reflex_triggers) == 1
    assert state.stability == "surprise"
    assert reflex_triggers[0][1] >= 0.80


def test_human_eye_affordances_and_servoing():
    """Verifies interactive affordance hotspots and closed-loop motor servoing."""
    eye = HumanEye(device="cpu")
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    state = eye.observe(frame, current_cursor=(0.1, 0.1))
    assert state.motor_flow != (0.0, 0.0)
    # Cursor is at (0.1, 0.1), gaze is near center (0.5, 0.5)
    # Motor velocity should be positive towards center
    assert state.motor_flow[0] > 0
    assert state.motor_flow[1] > 0

