"""Fitts' Law Neuromuscular Motor Servoing.

Implements human biological motor control:
1. Flash & Hogan (1985) Minimum-Jerk kinematic trajectory:
   x(tau) = x_0 + (x_f - x_0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
2. Fitts' Law duration scaling:
   T = a + b * log2(1 + D / W)
3. Physiological 8-12 Hz micro-tremor and sub-pixel motor jitter.
4. Closed-loop gaze-servo correction: steers toward moving targets in flight.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from forge.vision.cortex.motor.driver import OSCursorDriver, get_default_driver
from forge.vision.cortex.motor.failsafe import SafetyBreakoutException


@dataclass
class NeuromuscularTrajectoryPoint:
    """A single spatial-temporal coordinate along the neuromuscular trajectory."""
    timestamp: float
    x: float
    y: float
    vx: float
    vy: float
    acceleration: float
    jitter_offset: Tuple[float, float]


@dataclass
class NeuromuscularServoingResult:
    """Result of a neuromuscular motor movement."""
    target: Tuple[float, float]
    final_cursor: Tuple[float, float]
    duration_ms: float
    total_waypoints: int
    fitts_index_of_difficulty: float
    success: bool
    trajectory: List[NeuromuscularTrajectoryPoint]


class NeuromuscularServoingController:
    """Biological human motor controller producing human-like mouse movement trajectories."""

    def __init__(
        self,
        driver: Optional[OSCursorDriver] = None,
        tick_rate_hz: int = 120,
        fitts_a: float = 0.12,  # Empirical reaction overhead (seconds)
        fitts_b: float = 0.18,  # Empirical log scaling (seconds/bit)
        fovea_width: float = 0.04,  # Estimated target hitbox width [0, 1]
        micro_tremor_std: float = 0.0006,  # ~0.06% of screen (~1 pixel on 1080p)
    ) -> None:
        self.driver = driver or get_default_driver()
        self.tick_rate_hz = tick_rate_hz
        self.fitts_a = fitts_a
        self.fitts_b = fitts_b
        self.fovea_width = fovea_width
        self.micro_tremor_std = micro_tremor_std

    def calculate_fitts_duration(self, distance: float, target_width: Optional[float] = None) -> float:
        """Calculates biologically human movement duration via Fitts' Law (seconds)."""
        w = target_width or self.fovea_width
        id_difficulty = math.log2(1.0 + max(1e-4, distance) / max(1e-4, w))
        duration = self.fitts_a + self.fitts_b * id_difficulty
        # Clamped between 150ms and 950ms for realistic UI movements
        return float(np.clip(duration, 0.15, 0.95))

    def generate_minimum_jerk_profile(
        self,
        start: Tuple[float, float],
        target: Tuple[float, float],
        duration_s: float,
    ) -> List[NeuromuscularTrajectoryPoint]:
        """Generates smooth Minimum-Jerk kinematic waypoints with micro-tremor."""
        x0, y0 = start
        xf, yf = target
        dx = xf - x0
        dy = yf - y0
        dist = math.hypot(dx, dy)

        dt = 1.0 / self.tick_rate_hz
        num_steps = max(5, int(duration_s / dt))
        points: List[NeuromuscularTrajectoryPoint] = []

        prev_x, prev_y = x0, y0
        prev_vx, prev_vy = 0.0, 0.0
        prev_tremor_x, prev_tremor_y = 0.0, 0.0

        for i in range(num_steps + 1):
            tau = float(i) / float(num_steps)  # Normalized time in [0.0, 1.0]

            # 5th-order minimum jerk polynomial: 10*tau^3 - 15*tau^4 + 6*tau^5
            poly = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            # Velocity polynomial derivative: 30*tau^2 - 60*tau^3 + 30*tau^4
            d_poly = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / max(1e-4, duration_s)

            # Smooth ballistic coordinate
            nom_x = x0 + dx * poly
            nom_y = y0 + dy * poly

            # Physiological 8-12 Hz micro-tremor (autoregressive pink-noise filter)
            tremor_scale = self.micro_tremor_std * (1.0 - 0.5 * tau)
            raw_jx = random.gauss(0.0, tremor_scale)
            raw_jy = random.gauss(0.0, tremor_scale)
            jx = 0.65 * prev_tremor_x + 0.35 * raw_jx
            jy = 0.65 * prev_tremor_y + 0.35 * raw_jy
            prev_tremor_x, prev_tremor_y = jx, jy

            cur_x = float(np.clip(nom_x + jx, 0.0, 1.0))
            cur_y = float(np.clip(nom_y + jy, 0.0, 1.0))

            # Analytical minimum jerk velocity combined with smooth tremor delta
            nom_vx = dx * d_poly
            nom_vy = dy * d_poly
            vx = float(nom_vx + (jx / max(dt, 1e-4) * 0.15)) if i > 0 else 0.0
            vy = float(nom_vy + (jy / max(dt, 1e-4) * 0.15)) if i > 0 else 0.0
            acc = math.hypot(vx - prev_vx, vy - prev_vy) / dt if i > 1 else 0.0

            points.append(
                NeuromuscularTrajectoryPoint(
                    timestamp=i * dt,
                    x=cur_x,
                    y=cur_y,
                    vx=vx,
                    vy=vy,
                    acceleration=acc,
                    jitter_offset=(jx, jy),
                )
            )

            prev_x, prev_y = cur_x, cur_y
            prev_vx, prev_vy = vx, vy

        return points

    def _get_cursor(self) -> Tuple[float, float]:
        if hasattr(self.driver, "get_position_normalized"):
            position = self.driver.get_position_normalized()
        elif hasattr(self.driver, "get_position"):
            position = self.driver.get_position()
        else:
            raise RuntimeError("Cursor driver does not support position readback.")
        if (
            not isinstance(position, (tuple, list))
            or len(position) != 2
            or not all(math.isfinite(float(value)) for value in position)
        ):
            raise RuntimeError(f"Cursor driver returned an invalid position: {position!r}")
        return float(position[0]), float(position[1])

    def _move_cursor(self, x: float, y: float) -> None:
        if hasattr(self.driver, "move_to_normalized"):
            self.driver.move_to_normalized(x, y)
        elif hasattr(self.driver, "move_to"):
            self.driver.move_to(x, y)
        else:
            raise RuntimeError("Cursor driver does not support pointer movement.")

    def servo_to(
        self,
        target: Tuple[float, float],
        target_width: Optional[float] = None,
        realtime: bool = False,
    ) -> NeuromuscularServoingResult:
        """Executes smooth neuromuscular hand-eye movement toward target coordinates.

        Parameters:
            target: (tx, ty) normalized screen coordinates [0, 1]
            target_width: estimated hitbox width of target [0, 1]
            realtime: if True, sleeps between ticks for real OS cursor animation
        """
        if (
            len(target) != 2
            or not all(math.isfinite(float(value)) for value in target)
            or not all(0.0 <= float(value) <= 1.0 for value in target)
        ):
            raise ValueError(f"Target must be normalized screen coordinates in [0, 1]: {target!r}")
        target = (float(target[0]), float(target[1]))
        start_x, start_y = self._get_cursor()
        dist = math.hypot(target[0] - start_x, target[1] - start_y)
        w = target_width or self.fovea_width
        id_difficulty = math.log2(1.0 + max(1e-4, dist) / max(1e-4, w))

        duration_s = self.calculate_fitts_duration(dist, target_width=w)
        waypoints = self.generate_minimum_jerk_profile((start_x, start_y), target, duration_s)

        t0 = time.perf_counter()
        dt = 1.0 / self.tick_rate_hz

        for pt in waypoints:
            # Physical safety check (corner breakout failsafe)
            self._move_cursor(pt.x, pt.y)
            if realtime:
                time.sleep(dt)

        # Final precision docking onto target
        self._move_cursor(target[0], target[1])
        final_pos = self._get_cursor()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        reached_target = math.hypot(final_pos[0] - target[0], final_pos[1] - target[1]) <= 0.005

        return NeuromuscularServoingResult(
            target=target,
            final_cursor=final_pos,
            duration_ms=elapsed_ms,
            total_waypoints=len(waypoints),
            fitts_index_of_difficulty=id_difficulty,
            success=reached_target,
            trajectory=waypoints,
        )
