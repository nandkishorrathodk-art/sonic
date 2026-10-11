"""AVCS pixel-only reflex perception for workstation screenshots."""

from __future__ import annotations

import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _avcs_source_root() -> Path:
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[3] / "AVCS",
        module_path.parents[2] / "AVCS",
    )
    for candidate in candidates:
        if (candidate / "forge" / "vision" / "reflex.py").is_file():
            return candidate
    raise RuntimeError("AVCS pixel-perception source is unavailable")


_AVCS_ROOT = str(_avcs_source_root())
if _AVCS_ROOT not in sys.path:
    sys.path.insert(0, _AVCS_ROOT)

from forge.vision.reflex import ReflexPerception, events_from_states  # noqa: E402
from forge.vision.scene import SceneState  # noqa: E402


class AVCSPerception:
    """Derive compact temporal visual state from real screenshot pixels."""

    def __init__(self) -> None:
        self._reflex = ReflexPerception()
        self._previous_state: SceneState | None = None
        self._frame_index = 0
        self._lock = threading.Lock()
        self._grounding_lock = threading.Lock()
        self._grounder = None
        self._grounding_checkpoint = os.environ.get("SONIC_AVCS_GROUNDING_CHECKPOINT", "").strip()

    @property
    def goal_grounder_configured(self) -> bool:
        return bool(self._grounding_checkpoint)

    def _load_goal_grounder(self) -> Any:
        from avcs.goal_grounder import load_grounding_checkpoint

        checkpoint = Path(self._grounding_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AVCS grounding checkpoint does not exist: {checkpoint}")
        return load_grounding_checkpoint(checkpoint)

    def ground_target_png(
        self,
        png_bytes: bytes,
        query: str,
        width: int,
        height: int,
    ) -> dict[str, Any] | None:
        """Resolve a target phrase from pixels using an evaluated AVCS checkpoint."""
        if not self.goal_grounder_configured:
            raise RuntimeError("SONIC_AVCS_GROUNDING_CHECKPOINT is not configured")
        if not png_bytes:
            raise ValueError("AVCS grounding requires non-empty screenshot pixels")
        if width <= 0 or height <= 0:
            raise ValueError("AVCS grounding requires positive screen dimensions")

        with Image.open(BytesIO(png_bytes)) as image:
            frame = np.array(image.convert("RGB"))
        if frame.shape[1] != width or frame.shape[0] != height:
            raise ValueError("AVCS grounding frame dimensions do not match the observation")

        with self._grounding_lock:
            if self._grounder is None:
                self._grounder = self._load_goal_grounder()
            grounder = self._grounder

        started = time.perf_counter()
        prediction = grounder.predict(frame, query)
        latency_ms = (time.perf_counter() - started) * 1_000
        point = prediction.point_norm
        if point is None:
            return None
        x_norm, y_norm = point
        x = int(x_norm * width)
        y = int(y_norm * height)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("AVCS grounding returned an off-screen point")
        return {
            "engine": "avcs_goal_grounder",
            "x": x,
            "y": y,
            "point_norm": [x_norm, y_norm],
            "held_out_point_accuracy": grounder.checkpoint_metadata["evaluation"]["value"],
            "latency_ms": round(latency_ms, 3),
        }

    def perceive_png(self, png_bytes: bytes) -> dict[str, Any]:
        if not png_bytes:
            raise ValueError("AVCS requires non-empty screenshot pixels")
        with Image.open(BytesIO(png_bytes)) as image:
            frame = np.array(image.convert("RGB"))

        with self._lock:
            self._frame_index += 1
            state = self._reflex.perceive(frame, frame_index=self._frame_index)
            events = events_from_states(self._previous_state, state)
            self._previous_state = state

        return {
            "engine": "avcs_reflex",
            "frame_index": state.frame_index,
            "width": state.width,
            "height": state.height,
            "scene_hash": state.scene_hash,
            "brightness": round(state.brightness, 3),
            "contrast": round(state.contrast, 3),
            "change_fraction": round(state.change_fraction, 4),
            "global_motion": round(state.global_motion, 4),
            "peak_motion": round(state.peak_motion, 4),
            "novelty": round(state.novelty, 3),
            "scene_changed": state.scene_changed,
            "overlay_suspected": state.overlay_suspected,
            "motion_centroid": (
                [round(value, 3) for value in state.motion_centroid]
                if state.motion_centroid is not None
                else None
            ),
            "regions": [region.to_dict() for region in state.regions],
            "events": [
                {
                    "kind": event.kind,
                    "text": event.text,
                    "salience": round(event.salience, 3),
                }
                for event in events
            ],
            "latency_ms": round(state.latency_ms, 3),
        }
