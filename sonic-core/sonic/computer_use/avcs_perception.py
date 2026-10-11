"""AVCS pixel-only reflex perception for workstation screenshots."""

from __future__ import annotations

import hashlib
import math
import os
import sys
import threading
import time
from collections.abc import Mapping
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
        self._last_frame_fingerprint: str | None = None
        self._last_scene_hash: str | None = None
        self._lock = threading.Lock()
        self._grounding_lock = threading.Lock()
        self._grounder = None
        self._grounding_checkpoint = os.environ.get("SONIC_AVCS_GROUNDING_CHECKPOINT", "").strip()

    @property
    def goal_grounder_configured(self) -> bool:
        return bool(self._grounding_checkpoint)

    @property
    def goal_grounder_ready(self) -> bool:
        """True only after a checkpoint with valid training/evaluation evidence loads."""
        if self._grounder is None:
            return False
        try:
            self._validated_checkpoint_metadata(self._grounder)
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

    def _load_goal_grounder(self) -> Any:
        from avcs.goal_grounder import (
            GROUNDING_CHECKPOINT_SCHEMA,
            GROUNDING_MODEL_TYPE,
            load_grounding_checkpoint,
        )

        checkpoint = Path(self._grounding_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AVCS grounding checkpoint does not exist: {checkpoint}")
        grounder = load_grounding_checkpoint(checkpoint)
        metadata = dict(getattr(grounder, "checkpoint_metadata", {}))
        metadata.setdefault("schema_version", GROUNDING_CHECKPOINT_SCHEMA)
        metadata.setdefault("model_type", GROUNDING_MODEL_TYPE)
        grounder.checkpoint_metadata = metadata
        self._validated_checkpoint_metadata(grounder)
        return grounder

    @staticmethod
    def _validated_checkpoint_metadata(grounder: Any) -> dict[str, Any]:
        from avcs.goal_grounder import (
            GROUNDING_ACCEPTANCE_METRIC,
            GROUNDING_ACCEPTANCE_THRESHOLD,
            GROUNDING_CHECKPOINT_SCHEMA,
            GROUNDING_MODEL_TYPE,
        )

        if getattr(grounder, "is_trained", None) is not True:
            raise ValueError("AVCS grounding checkpoint is not demonstration-trained")
        metadata = getattr(grounder, "checkpoint_metadata", None)
        if not isinstance(metadata, Mapping):
            raise ValueError("AVCS grounding checkpoint metadata is unavailable")
        if metadata.get("schema_version") != GROUNDING_CHECKPOINT_SCHEMA:
            raise ValueError("AVCS grounding checkpoint schema metadata is invalid")
        if metadata.get("model_type") != GROUNDING_MODEL_TYPE:
            raise ValueError("AVCS grounding checkpoint model metadata is invalid")
        if metadata.get("dataset_provenance") != "recorded_ui_click_demonstrations":
            raise ValueError("AVCS grounding checkpoint lacks UI demonstration provenance")
        for name in ("training_samples", "validation_samples", "held_out_test_samples"):
            count = metadata.get(name)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"AVCS grounding checkpoint has invalid {name}")

        evaluation = metadata.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("AVCS grounding checkpoint held-out evaluation is unavailable")
        samples = evaluation.get("samples")
        value = evaluation.get("value")
        if (
            evaluation.get("metric") != GROUNDING_ACCEPTANCE_METRIC
            or evaluation.get("passed") is not True
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples <= 0
            or samples != metadata["held_out_test_samples"]
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < GROUNDING_ACCEPTANCE_THRESHOLD
        ):
            raise ValueError("AVCS grounding checkpoint does not meet held-out evaluation gates")
        confidence_calibration = metadata.get("query_confidence_calibration")
        confidence_evaluation = (
            confidence_calibration.get("evaluation")
            if isinstance(confidence_calibration, Mapping)
            else None
        )
        if (
            not isinstance(confidence_calibration, Mapping)
            or not isinstance(confidence_calibration.get("method"), str)
            or not confidence_calibration["method"].strip()
            or not isinstance(confidence_evaluation, Mapping)
        ):
            raise ValueError("AVCS grounding checkpoint lacks per-query confidence calibration")
        confidence_samples = confidence_evaluation.get("samples")
        confidence_threshold = confidence_evaluation.get("threshold")
        measured_precision = confidence_evaluation.get("precision")
        required_precision = confidence_evaluation.get("required_precision")
        if (
            confidence_evaluation.get("metric") != "empirical_precision_at_threshold"
            or confidence_evaluation.get("passed") is not True
            or isinstance(confidence_samples, bool)
            or not isinstance(confidence_samples, int)
            or confidence_samples != metadata["held_out_test_samples"]
            or isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(float(confidence_threshold))
            or not 0 < float(confidence_threshold) <= 1
            or isinstance(measured_precision, bool)
            or not isinstance(measured_precision, (int, float))
            or not math.isfinite(float(measured_precision))
            or not 0 <= float(measured_precision) <= 1
            or isinstance(required_precision, bool)
            or not isinstance(required_precision, (int, float))
            or not math.isfinite(float(required_precision))
            or not 0 < float(required_precision) <= 1
            or float(measured_precision) < float(required_precision)
        ):
            raise ValueError("AVCS grounding checkpoint fails per-query confidence calibration gates")
        return dict(metadata)

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
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("AVCS grounding requires a positive integer screen width")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("AVCS grounding requires positive screen dimensions")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("AVCS grounding requires a non-empty target query")
        if len(query.strip().encode("utf-8")) > 96:
            raise ValueError("AVCS grounding queries must be at most 96 UTF-8 bytes")

        with Image.open(BytesIO(png_bytes)) as image:
            frame = np.array(image.convert("RGB"))
        if frame.shape[1] != width or frame.shape[0] != height:
            raise ValueError("AVCS grounding frame dimensions do not match the observation")
        fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
        with self._lock:
            if fingerprint != self._last_frame_fingerprint:
                raise RuntimeError("AVCS grounding pixels do not match the latest perceived frame")
            frame_index = self._frame_index
            scene_hash = self._last_scene_hash

        with self._grounding_lock:
            if self._grounder is None:
                self._grounder = self._load_goal_grounder()
            grounder = self._grounder
            checkpoint_metadata = self._validated_checkpoint_metadata(grounder)

        started = time.perf_counter()
        prediction = grounder.predict(frame, query)
        latency_ms = (time.perf_counter() - started) * 1_000
        with self._lock:
            if (
                self._frame_index != frame_index
                or self._last_frame_fingerprint != fingerprint
                or self._last_scene_hash != scene_hash
            ):
                raise RuntimeError("AVCS frame changed while target grounding was running")
        point = prediction.point_norm
        if point is None:
            return None
        calibration = checkpoint_metadata["query_confidence_calibration"]["evaluation"]
        query_score = getattr(prediction, "peak_probability", None)
        if (
            isinstance(query_score, bool)
            or not isinstance(query_score, (int, float))
            or not math.isfinite(float(query_score))
            or not 0 <= float(query_score) <= 1
            or float(query_score) < float(calibration["threshold"])
        ):
            return None
        if (
            not isinstance(point, (tuple, list))
            or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)
        ):
            raise ValueError("AVCS grounding returned an invalid normalized point")
        x_norm, y_norm = point
        if (
            not math.isfinite(float(x_norm))
            or not math.isfinite(float(y_norm))
            or not (0 <= x_norm < 1 and 0 <= y_norm < 1)
        ):
            raise ValueError("AVCS grounding returned an off-screen point")
        x = int(x_norm * width)
        y = int(y_norm * height)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("AVCS grounding returned an off-screen point")
        return {
            "engine": "avcs_goal_grounder",
            "x": x,
            "y": y,
            "point_norm": [x_norm, y_norm],
            "frame_index": frame_index,
            "scene_hash": scene_hash,
            "held_out_point_accuracy": checkpoint_metadata["evaluation"]["value"],
            "query_score": float(query_score),
            "query_score_threshold": float(calibration["threshold"]),
            "checkpoint_metadata": checkpoint_metadata,
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
            self._last_frame_fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
            self._last_scene_hash = state.scene_hash

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
