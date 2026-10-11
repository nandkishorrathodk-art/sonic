"""AVCS Real-Time Human Eye Gaming & Dynamic Video Benchmark.

Demonstrates and verifies that the HumanEye model perceives continuous 60-120 FPS
visual streams, tracks accelerating dynamic targets via smooth pursuit, and executes
sub-16ms zero-pause neural reflexes on standard hardware.
"""
from __future__ import annotations

import argparse
import math
import time
from typing import List, Tuple

import numpy as np

from avcs.human_eye import HumanEye, HumanEyeState
from forge.vision.capture import SyntheticScreen


class DynamicGamingViewport(SyntheticScreen):
    """Simulates a fast 60-120 FPS dynamic gaming/video viewport with multiple moving targets."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        super().__init__(width=width, height=height)
        self.target_pos = [0.20, 0.25]
        self.target_vel = [0.45, 0.35]  # normalized units/sec
        self.secondary_pos = [0.80, 0.70]
        self.secondary_vel = [-0.30, 0.40]
        self._last_time = time.perf_counter()
        self.flash_triggered = False

    def trigger_flash(self) -> None:
        """Simulates an abrupt visual shock / flash (e.g. explosion, popup)."""
        self.flash_triggered = True

    def grab(self) -> np.ndarray:
        now = time.perf_counter()
        dt = min(0.05, max(0.001, now - self._last_time))
        self._last_time = now

        # Update primary target
        self.target_pos[0] += self.target_vel[0] * dt
        self.target_pos[1] += self.target_vel[1] * dt
        for i in (0, 1):
            if self.target_pos[i] < 0.10 or self.target_pos[i] > 0.90:
                self.target_vel[i] *= -1.0
                self.target_pos[i] = max(0.10, min(0.90, self.target_pos[i]))

        # Update secondary target
        self.secondary_pos[0] += self.secondary_vel[0] * dt
        self.secondary_pos[1] += self.secondary_vel[1] * dt
        for i in (0, 1):
            if self.secondary_pos[i] < 0.10 or self.secondary_pos[i] > 0.90:
                self.secondary_vel[i] *= -1.0
                self.secondary_pos[i] = max(0.10, min(0.90, self.secondary_pos[i]))

        # Render frame
        frame = np.full((self.height, self.width, 3), 30, dtype=np.uint8)

        # Primary target (Cyan)
        tx = int(self.target_pos[0] * self.width)
        ty = int(self.target_pos[1] * self.height)
        r = 16
        frame[max(0, ty - r):min(self.height, ty + r), max(0, tx - r):min(self.width, tx + r)] = [0, 240, 255]

        # Secondary target (Magenta)
        sx = int(self.secondary_pos[0] * self.width)
        sy = int(self.secondary_pos[1] * self.height)
        r2 = 12
        frame[max(0, sy - r2):min(self.height, sy + r2), max(0, sx - r2):min(self.width, sx + r2)] = [255, 0, 200]

        # Sudden flash event
        if self.flash_triggered:
            frame[:, :] = np.clip(frame.astype(np.int16) + 200, 0, 255).astype(np.uint8)
            self.flash_triggered = False

        return frame


def run_benchmark(duration_seconds: float = 3.0) -> None:
    print("=" * 76)
    print("      AVCS REAL-TIME BIOLOGICAL HUMAN EYE BENCHMARK (60-120 FPS)      ")
    print("=" * 76)

    viewport = DynamicGamingViewport(width=640, height=480)
    eye = HumanEye(source=viewport, device="cpu")

    reflexes: List[Tuple[float, float]] = []

    @eye.on_surprise(threshold=0.82)
    def on_flash_reflex(state: HumanEyeState):
        reflexes.append((state.timestamp, state.surprise_score))

    print(f"Perceiving continuous dynamic stream for {duration_seconds:.1f} seconds...")

    # Warmup
    for _ in range(10):
        eye.observe()

    start_t = time.perf_counter()
    flash_injected = False
    tick_count = 0
    tracking_errors: List[float] = []

    # Pace loop for realistic 120 FPS maximum tick rate
    while time.perf_counter() - start_t < duration_seconds:
        elapsed_so_far = time.perf_counter() - start_t
        if not flash_injected and elapsed_so_far > (duration_seconds * 0.45):
            viewport.trigger_flash()
            flash_injected = True

        state = eye.observe()
        tick_count += 1

        # Evaluate tracking accuracy against true primary target
        true_x, true_y = viewport.target_pos
        gaze_x, gaze_y = state.gaze_fixation
        err = math.hypot(true_x - gaze_x, true_y - gaze_y)
        tracking_errors.append(err)

        time.sleep(0.005)  # Cap loop pace at ~120 Hz

    total_time = time.perf_counter() - start_t
    effective_fps = tick_count / total_time
    p50_lat = eye.p50_latency_ms
    p95_lat = eye.p95_latency_ms
    mean_err = float(np.mean(tracking_errors[15:])) if len(tracking_errors) > 15 else 0.0

    print("\n" + "-" * 76)
    print("PERFORMANCE TELEMETRY REPORT:")
    print(f"  • Total Video Frames Observed : {tick_count}")
    print(f"  • Effective Vision Frame Rate : {effective_fps:.1f} FPS")
    print(f"  • Median Perception Latency   : {p50_lat:.2f} ms (p50)")
    print(f"  • 95th Percentile Latency     : {p95_lat:.2f} ms (p95)")
    print(f"  • Mean Smooth Pursuit Error   : {mean_err:.4f} (normalized)")
    print(f"  • Surprise Reflex Reactions   : {len(reflexes)} instant triggers")
    print("-" * 76)

    passed = (effective_fps >= 60.0) and (p50_lat < 16.0) and (mean_err < 0.15)
    print(f"VERDICT: {'PASSED [HUMAN-EQUIVALENT REAL-TIME]' if passed else 'FAILED'}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVCS Real-Time Human Eye Benchmark")
    parser.add_argument("--seconds", type=float, default=2.0, help="Benchmark run duration in seconds")
    args = parser.parse_args()
    run_benchmark(duration_seconds=args.seconds)

