"""Visual Servoing & Closed-Loop Hand-Eye Coordination.

In primate neurobiology, humans do not compute mathematical coordinate numbers
to click or grasp objects. Instead, the brain executes Visual Servoing:
1. Gaze Fixation: The eye's fovea locks onto the visual target (Point of Gaze).
2. Effector Tracking: The visual cortex continuously tracks the hand/cursor position.
3. Visual Motor Flow: The parietal-motor cortex computes the relative visual delta
   vector (Gaze - Cursor) to steer the hand smoothly toward the gaze target.
4. Contact Alignment: When the visual delta falls below the foveal threshold,
   the motor system triggers the action (click / grasp).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class VisualMotorState:
    """Continuous biological motor guidance state."""
    gaze_fixation: Tuple[float, float]   # Point of gaze (fx, fy) in [0, 1]
    cursor_position: Tuple[float, float] # Current hand/cursor location in [0, 1]
    motor_velocity: Tuple[float, float]  # Continuous directional flow (dx, dy)
    distance_to_target: float            # Normalized Euclidean visual error
    contact_acquired: bool               # True when cursor is locked on target
    confidence: float                    # Visual confidence of the target alignment


class VisualServoingEngine(nn.Module):
    """Closed-loop visual motor servoing with biological Fitts' Law dynamics."""

    def __init__(
        self,
        foveal_contact_radius: float = 0.025, # ~2.5% of screen width (foveal basin)
        max_speed: float = 1.0,               # Maximum normalized speed
        damping: float = 0.85,                # Deceleration damping as target nears
    ) -> None:
        super().__init__()
        self.foveal_contact_radius = foveal_contact_radius
        self.max_speed = max_speed
        self.damping = damping

    def compute_motor_flow(
        self,
        gaze_target: Tuple[float, float],
        current_cursor: Tuple[float, float],
        confidence: float = 1.0,
    ) -> VisualMotorState:
        """Computes continuous hand-eye steering velocity.

        Parameters
        ----------
        gaze_target: (gx, gy) where the eye is fixating
        current_cursor: (cx, cy) where the hand/cursor currently is
        """
        gx, gy = gaze_target
        cx, cy = current_cursor

        dx = gx - cx
        dy = gy - cy
        dist = math.hypot(dx, dy)

        # Contact check: has cursor entered the foveal target basin?
        contact = dist <= self.foveal_contact_radius

        if dist < 1e-6:
            vx, vy = 0.0, 0.0
        else:
            unit_x = dx / dist
            unit_y = dy / dist

            if dist > self.foveal_contact_radius:
                # Approach Zone: Fitts' law ballistic velocity tapering into foveal basin
                # Hermite smoothstep blend for minimum jerk deceleration
                t = min(1.0, max(0.0, (dist - self.foveal_contact_radius) / 0.15))
                blend = t * t * (3.0 - 2.0 * t)  # C1 smooth deceleration
                speed = 0.04 + (self.max_speed - 0.04) * blend
                speed = min(self.max_speed, speed * (dist * 3.5))
                vx = unit_x * speed
                vy = unit_y * speed
            else:
                # Foveal Basin Docking: Target contact acquired
                vx = 0.0
                vy = 0.0

        return VisualMotorState(
            gaze_fixation=(gx, gy),
            cursor_position=(cx, cy),
            motor_velocity=(float(vx), float(vy)),
            distance_to_target=float(dist),
            contact_acquired=contact,
            confidence=confidence,
        )

