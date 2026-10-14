"""OpenAI Operator & OSWorld Benchmark Environment Harness.

Integrates AVCS biological perception into standard Computer-Using Agent (CUA)
gym environments (OSWorld, DesktopEnv, WindowsArena). Transforms raw pixel frames
into structured semantic affordance tokens and provides visual servoing action execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from avcs import ScreenEye, CortexState


@dataclass
class OperatorVisualObservation:
    """Structured perception observation for Computer-Using Agents."""
    screen_category: str
    category_confidence: float
    is_animating: bool
    surprise_anomaly: float
    clickable_buttons: List[Dict[str, Any]]
    typeable_inputs: List[Dict[str, Any]]
    foveal_fixation: Tuple[float, float]
    inference_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen_category": self.screen_category,
            "category_confidence": self.category_confidence,
            "is_animating": self.is_animating,
            "surprise_anomaly": self.surprise_anomaly,
            "clickable_buttons": self.clickable_buttons,
            "typeable_inputs": self.typeable_inputs,
            "foveal_fixation": self.foveal_fixation,
            "inference_latency_ms": self.inference_latency_ms,
        }


class AVCSOperatorHarness:
    """Gym-compatible perception wrapper for OSWorld and Operator environments."""

    def __init__(self, eye: Optional[ScreenEye] = None) -> None:
        self.eye = eye or ScreenEye()

    def process_observation(self, raw_frame: Optional[np.ndarray] = None) -> OperatorVisualObservation:
        """Processes an environment frame into a structured high-level observation."""
        state = self.eye.perceive_now(frame=raw_frame)

        clickables = [
            {"x": round(x, 4), "y": round(y, 4), "confidence": round(conf, 3)}
            for x, y, conf in self.eye.clickable_hotspots()
        ]

        typeables = [
            {"x": round(x, 4), "y": round(y, 4), "confidence": round(conf, 3)}
            for x, y, conf in self.eye.typeable_hotspots()
        ]

        return OperatorVisualObservation(
            screen_category=state.primary_category,
            category_confidence=state.category_confidence,
            is_animating=state.stability_state in ["animating", "transitioning"],
            surprise_anomaly=state.surprise_score,
            clickable_buttons=clickables,
            typeable_inputs=typeables,
            foveal_fixation=state.fixation,
            inference_latency_ms=state.inference_ms,
        )

    def execute_servoing_step(
        self,
        current_cursor: Tuple[float, float],
        target_label_or_coords: Any,
    ) -> Tuple[Tuple[float, float], bool]:
        """Calculates next continuous visual servoing velocity vector.

        Returns:
            (velocity: (vx, vy), contact_acquired: bool)
        """
        if isinstance(target_label_or_coords, str):
            target = self.eye.look_at(target_label_or_coords)
        else:
            target = target_label_or_coords

        return self.eye.steer_hand(current_cursor=current_cursor, target_gaze=target)

