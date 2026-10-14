"""Advisory temporal prediction over AVCS's pixel-derived luminance grid."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class System3TemporalPredictor:
    """Use a one-frame persistence baseline to report scene surprise, never authorize actions."""

    STABILITY_STATES = ("initializing", "stable", "animating", "transitioning", "surprise_event")
    ANIMATING_THRESHOLD = 0.015
    TRANSITIONING_THRESHOLD = 0.08
    SURPRISE_THRESHOLD = 0.25
    SETTLED_THRESHOLD = 0.01

    def __init__(self) -> None:
        self._previous_latent: np.ndarray | None = None
        self._state = "initializing"
        self._frame_index: int | None = None

    def reset(self) -> None:
        self._previous_latent = None
        self._state = "initializing"
        self._frame_index = None

    def observe(self, latent: np.ndarray, frame_index: int) -> dict[str, Any]:
        """Compare a current AVCS luminance grid with the prior frame."""
        current = np.asarray(latent, dtype=np.float32)
        if current.ndim != 2 or current.size == 0:
            raise ValueError("System 3 requires a non-empty two-dimensional AVCS latent grid")
        if not np.isfinite(current).all() or np.any(current < 0) or np.any(current > 1):
            raise ValueError("System 3 AVCS latent values must be finite and within [0, 1]")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index <= 0:
            raise ValueError("System 3 requires a positive integer frame index")
        if self._frame_index is not None and frame_index <= self._frame_index:
            raise ValueError("System 3 frame indices must increase monotonically")

        has_prediction = self._previous_latent is not None
        if has_prediction and self._previous_latent.shape != current.shape:
            self.reset()
            has_prediction = False

        if has_prediction:
            surprise = float(np.mean(np.abs(current - self._previous_latent)))
            self._state = self._classify(surprise, self._state)
        else:
            surprise = 0.0
            self._state = "initializing"

        self._previous_latent = current.copy()
        self._frame_index = frame_index
        if not math.isfinite(surprise):
            raise ValueError("System 3 produced a non-finite prediction error")

        return {
            "engine": "system3_temporal_prediction",
            "model": "previous_frame_persistence_baseline",
            "frame_index": frame_index,
            "prediction_available": has_prediction,
            "surprise_score": round(surprise, 4),
            "stability_state": self._state,
            "trained": False,
            "action_authorization": False,
        }

    @classmethod
    def _classify(cls, surprise: float, previous_state: str) -> str:
        if surprise >= cls.SURPRISE_THRESHOLD:
            return "surprise_event"
        if previous_state == "surprise_event":
            return "transitioning" if surprise >= cls.SETTLED_THRESHOLD else "stable"
        if previous_state == "transitioning":
            return "animating" if surprise < cls.TRANSITIONING_THRESHOLD else "transitioning"
        if previous_state == "animating":
            return "stable" if surprise < cls.SETTLED_THRESHOLD else "animating"
        if surprise >= cls.TRANSITIONING_THRESHOLD:
            return "transitioning"
        if surprise >= cls.ANIMATING_THRESHOLD:
            return "animating"
        return "stable"
