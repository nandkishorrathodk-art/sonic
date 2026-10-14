"""Continuous Hand-Eye Servoing Controller.

Orchestrates biological visual motor flow vectors, Fitts' Law acceleration
profiles, physiological micro-jitter, and closed-loop contact acquisition.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from forge.vision.cortex.core.servoing import VisualServoingEngine, VisualMotorState
from forge.vision.cortex.motor.driver import OSCursorDriver, get_default_driver
from forge.vision.cortex.motor.failsafe import SafetyBreakoutException


@dataclass
class ServoingTrajectoryPoint:
    timestamp: float
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    distance_to_target: float
    contact: bool


@dataclass
class ServoingResult:
    target: Tuple[float, float]
    final_position: Tuple[float, float]
    contact_acquired: bool
    elapsed_ms: float
    total_steps: int
    trajectory: List[ServoingTrajectoryPoint]


class HandEyeServoingController:
    """Closed-loop motor controller connecting AVCS perception to OS cursor."""

    def __init__(
        self,
        driver: Optional[OSCursorDriver] = None,
        servoing_engine: Optional[VisualServoingEngine] = None,
        tick_rate_hz: int = 60,
        foveal_contact_radius: float = 0.025,
        add_human_jitter: bool = True,
        jitter_std: float = 0.0008,
    ) -> None:
        self.driver = driver or get_default_driver()
        self.servoing = servoing_engine or VisualServoingEngine(foveal_contact_radius=foveal_contact_radius)
        self.tick_rate_hz = tick_rate_hz
        self.add_human_jitter = add_human_jitter
        self.jitter_std = jitter_std

    def servo_to(
        self,
        target_gaze: Tuple[float, float],
        timeout_seconds: float = 1.5,
        click_on_arrival: bool = False,
        speed_factor: float = 1.0,
        realtime: bool = True,
    ) -> ServoingResult:
        """Executes smooth continuous hand-eye servoing toward target gaze.

        Parameters
        ----------
        target_gaze: (gx, gy) normalized target coordinates [0.0, 1.0]
        timeout_seconds: maximum time before timing out
        click_on_arrival: whether to execute click once contact is confirmed
        speed_factor: scaling factor for motor velocity
        realtime: whether to sleep between ticks to match real-time physics clock
        """
        gx, gy = target_gaze
        tick_dt = 1.0 / self.tick_rate_hz
        t_start = time.perf_counter()
        trajectory: List[ServoingTrajectoryPoint] = []

        consecutive_contacts = 0
        contact_confirmed = False
        prev_commanded_pos: Optional[Tuple[float, float]] = None

        # Autoregressive physiological tremor state (8-12 Hz biological tremor)
        tremor_x, tremor_y = 0.0, 0.0
        ar_rho = 0.85

        while True:
            now = time.perf_counter()
            elapsed = now - t_start
            if elapsed > timeout_seconds:
                break

            curr_pos = self.driver.get_position_normalized()

            # Physical user override detection: check if human snatched the mouse
            if prev_commanded_pos is not None and hasattr(self.driver, "failsafe"):
                if hasattr(self.driver.failsafe, "check_override"):
                    self.driver.failsafe.check_override(
                        actual_pos=curr_pos,
                        expected_pos=prev_commanded_pos,
                        threshold_norm=0.08,
                    )

            motor_state = self.servoing.compute_motor_flow(
                gaze_target=target_gaze,
                current_cursor=curr_pos,
            )

            vx, vy = motor_state.motor_velocity
            vx *= speed_factor
            vy *= speed_factor

            point = ServoingTrajectoryPoint(
                timestamp=elapsed,
                x=curr_pos[0],
                y=curr_pos[1],
                velocity_x=vx,
                velocity_y=vy,
                distance_to_target=motor_state.distance_to_target,
                contact=motor_state.contact_acquired,
            )
            trajectory.append(point)

            if motor_state.contact_acquired:
                consecutive_contacts += 1
                if consecutive_contacts >= 2:
                    contact_confirmed = True
                    break
            else:
                consecutive_contacts = 0

            # Compute next incremental position step
            step_x = vx * tick_dt
            step_y = vy * tick_dt

            # Autoregressive physiological tremor (faded smoothly near basin)
            if self.add_human_jitter and motor_state.distance_to_target > self.servoing.foveal_contact_radius:
                fade = min(1.0, (motor_state.distance_to_target - self.servoing.foveal_contact_radius) / 0.05)
                tremor_x = ar_rho * tremor_x + (1.0 - ar_rho) * random.gauss(0.0, self.jitter_std)
                tremor_y = ar_rho * tremor_y + (1.0 - ar_rho) * random.gauss(0.0, self.jitter_std)
                step_x += tremor_x * fade
                step_y += tremor_y * fade

            next_x = max(0.0, min(1.0, curr_pos[0] + step_x))
            next_y = max(0.0, min(1.0, curr_pos[1] + step_y))
            prev_commanded_pos = (next_x, next_y)

            # Apply incremental movement to physical driver
            self.driver.move_to_normalized(next_x, next_y)

            # Regulate tick rate if in real-time execution
            if realtime:
                loop_duration = time.perf_counter() - now
                sleep_time = tick_dt - loop_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)

        final_pos = self.driver.get_position_normalized()
        total_ms = (time.perf_counter() - t_start) * 1000.0

        if contact_confirmed and click_on_arrival:
            if hasattr(self.driver, "check_hardware_failsafe"):
                self.driver.check_hardware_failsafe()
            self.driver.click()

        return ServoingResult(
            target=target_gaze,
            final_position=final_pos,
            contact_acquired=contact_confirmed,
            elapsed_ms=total_ms,
            total_steps=len(trajectory),
            trajectory=trajectory,
        )
