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

from sonic.computer_use.system3_predictor import System3TemporalPredictor


def _avcs_source_root() -> Path:
    module_path = Path(__file__).resolve()
    configured_root = os.environ.get("SONIC_AVCS_ROOT")
    candidates = ([Path(configured_root).resolve()] if configured_root else []) + [
        parent / "packages" / "avcs" for parent in module_path.parents
    ]
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
    """Derive compact temporal visual state directly from workstation pixels."""

    def __init__(self) -> None:
        self._reflex = ReflexPerception()
        self._system3_predictor = System3TemporalPredictor()
        self._previous_state: SceneState | None = None
        self._frame_index = 0
        self._last_source_frame_id: int | None = None
        self._source_frame_id_high_watermark: int | None = None
        self._last_frame_fingerprint: str | None = None
        self._last_scene_hash: str | None = None
        self._lock = threading.Lock()
        self._grounding_lock = threading.Lock()
        self._grounder = None
        self._grounding_checkpoint = os.environ.get("SONIC_AVCS_GROUNDING_CHECKPOINT", "").strip()
        self._focus_verifier = None
        self._focus_checkpoint = os.environ.get("SONIC_AVCS_FOCUS_CHECKPOINT", "").strip()
        # Semantic cortex head (trained UI-category + affordance model). Unlike
        # the reflex path this is a learned network, so it is only used when a
        # real trained checkpoint is configured and loads. Otherwise the agent
        # keeps the numpy reflex semantics and does not pretend to have them.
        self._cortex = None
        self._cortex_failed = False
        self._cortex_checkpoint = os.environ.get("SONIC_AVCS_CORTEX_CHECKPOINT", "").strip()
        self._cortex_device = os.environ.get("SONIC_AVCS_CORTEX_DEVICE", "cpu").strip() or "cpu"

    @property
    def cortex_configured(self) -> bool:
        return bool(self._cortex_checkpoint)

    @property
    def cortex_ready(self) -> bool:
        """True only when a trained cortex checkpoint actually loaded."""
        return self._cortex is not None

    def ensure_cortex_ready(self) -> bool:
        """Load the semantic cortex once, fail-closed on any error.

        A configured path is NOT readiness: a missing file, a shape mismatch or
        an unevaluated checkpoint leaves the cortex unloaded so callers cannot
        report learned semantics they do not have.
        """
        if self._cortex is not None:
            return True
        if self._cortex_failed or not self.cortex_configured:
            return False
        try:
            from forge.vision.cortex.training.evaluate import load_cortex_checkpoint

            self._cortex = load_cortex_checkpoint(
                self._cortex_checkpoint, device=self._cortex_device
            )
            return True
        except Exception:
            self._cortex_failed = True
            return False

    @property
    def goal_grounder_configured(self) -> bool:
        return bool(self._grounding_checkpoint)

    @property
    def focus_verifier_configured(self) -> bool:
        return bool(self._focus_checkpoint)

    @property
    def focus_verifier_ready(self) -> bool:
        """True only after a focus checkpoint with valid held-out evidence loads."""
        if self._focus_verifier is None:
            return False
        metadata = getattr(self._focus_verifier, "checkpoint_metadata", None)
        if not isinstance(metadata, Mapping):
            return False
        evaluation = metadata.get("evaluation")
        if not isinstance(evaluation, Mapping):
            return False
        return bool(evaluation.get("passed")) and evaluation.get("metric") == "focus_verification_precision"

    def _load_focus_verifier(self) -> Any:
        from avcs.focus_verifier import load_focus_checkpoint

        checkpoint = Path(self._focus_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AVCS focus checkpoint does not exist: {checkpoint}")
        verifier = load_focus_checkpoint(checkpoint)
        self._validated_focus_metadata(verifier)
        return verifier

    def ensure_focus_verifier_ready(self) -> bool:
        """Load the focus head once and report whether it is genuinely usable.

        Fail-closed: returns False when unconfigured or when the checkpoint fails
        its provenance/precision gates, so callers never treat a configured path
        as readiness.
        """
        if not self.focus_verifier_configured:
            return False
        if self._focus_verifier is None:
            try:
                with self._grounding_lock:
                    if self._focus_verifier is None:
                        self._focus_verifier = self._load_focus_verifier()
            except Exception:
                return False
        return self.focus_verifier_ready

    @staticmethod
    def _validated_focus_metadata(verifier: Any) -> dict[str, Any]:
        from avcs.focus_verifier import (
            FOCUS_ACCEPTANCE_METRIC,
            FOCUS_ACCEPTANCE_THRESHOLD,
            FOCUS_DATASET_PROVENANCE,
            FOCUS_MIN_EVALUATION_SAMPLES,
        )

        if getattr(verifier, "is_trained", None) is not True:
            raise ValueError("AVCS focus checkpoint is not demonstration-trained")
        metadata = getattr(verifier, "checkpoint_metadata", None)
        if not isinstance(metadata, Mapping):
            raise ValueError("AVCS focus checkpoint metadata is unavailable")
        if metadata.get("dataset_provenance") != FOCUS_DATASET_PROVENANCE:
            raise ValueError("AVCS focus checkpoint lacks focus demonstration provenance")
        evaluation = metadata.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("AVCS focus checkpoint held-out evaluation is unavailable")
        precision = evaluation.get("precision")
        samples = evaluation.get("samples")
        if (
            evaluation.get("metric") != FOCUS_ACCEPTANCE_METRIC
            or evaluation.get("passed") is not True
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples < FOCUS_MIN_EVALUATION_SAMPLES
            or isinstance(precision, bool)
            or not isinstance(precision, (int, float))
            or not math.isfinite(float(precision))
            or float(precision) < FOCUS_ACCEPTANCE_THRESHOLD
        ):
            raise ValueError("AVCS focus checkpoint fails the held-out precision gates")
        return dict(metadata)

    def verify_focus_rgb_bytes(
        self,
        frame_bytes: bytes,
        width: int,
        height: int,
        x: int,
        y: int,
        target: str,
        source_frame_id: int | None = None,
    ) -> dict[str, Any]:
        """Verify, from the latest local frame, whether a point is focused text entry.

        Fails closed: without a configured, evaluated focus checkpoint, or when
        the supplied pixels are not the latest perceived frame, this raises.
        """
        if not self.focus_verifier_configured:
            raise RuntimeError("SONIC_AVCS_FOCUS_CHECKPOINT is not configured")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
            or width * height > 64_000_000
        ):
            raise ValueError("AVCS focus verification requires valid bounded frame dimensions")
        if not isinstance(frame_bytes, bytes) or len(frame_bytes) != width * height * 3:
            raise ValueError("AVCS focus verification frame length does not match its dimensions")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("AVCS focus verification requires a non-empty target query")
        if source_frame_id is not None and (
            isinstance(source_frame_id, bool)
            or not isinstance(source_frame_id, int)
            or source_frame_id <= 0
        ):
            raise ValueError("AVCS source frame ID must be a positive integer")
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
        frame = self._validated_rgb_frame(frame)
        fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
        with self._lock:
            if fingerprint != self._last_frame_fingerprint:
                raise RuntimeError(
                    "AVCS focus verification pixels do not match the latest perceived frame"
                )
            if source_frame_id is not None and source_frame_id != self._last_source_frame_id:
                raise RuntimeError(
                    "AVCS focus verification frame ID does not match the latest perceived frame"
                )
            frame_id = self._last_source_frame_id
            scene_hash = self._last_scene_hash

        with self._grounding_lock:
            if self._focus_verifier is None:
                self._focus_verifier = self._load_focus_verifier()
            verifier = self._focus_verifier
            self._validated_focus_metadata(verifier)
            prediction = verifier.predict(frame, x, y, target)

        return {
            "engine": "avcs_focus_verifier",
            "focused": bool(prediction.focused),
            "decision_score": float(prediction.score),
            "decision_threshold": float(prediction.threshold),
            "frame_id": frame_id,
            "scene_hash": scene_hash,
            "patch_bounds": list(prediction.patch_bounds),
            "checkpoint_metadata": dict(verifier.checkpoint_metadata),
            "evidence": (
                f"local focus head score {prediction.score:.3f} "
                f"(threshold {prediction.threshold:.3f}) at ({x},{y}) for {target!r}"
            ),
        }

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
            GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES,
            GROUNDING_QUERY_REQUIRED_PRECISION,
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
        if metadata.get("coordinate_frame") not in (
            "cropped_target_window",
            "full_screen",
        ):
            raise ValueError("AVCS grounding checkpoint has an unsupported coordinate frame")
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
        accepted_samples = confidence_evaluation.get("accepted_samples")
        minimum_accepted_samples = confidence_evaluation.get("minimum_accepted_samples")
        confidence_threshold = confidence_evaluation.get("threshold")
        measured_precision = confidence_evaluation.get("precision")
        required_precision = confidence_evaluation.get("required_precision")
        if (
            confidence_evaluation.get("metric") != "empirical_precision_at_threshold"
            or confidence_evaluation.get("passed") is not True
            or isinstance(confidence_samples, bool)
            or not isinstance(confidence_samples, int)
            or confidence_samples != metadata["held_out_test_samples"]
            or isinstance(accepted_samples, bool)
            or not isinstance(accepted_samples, int)
            or accepted_samples < GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES
            or accepted_samples > confidence_samples
            or isinstance(minimum_accepted_samples, bool)
            or not isinstance(minimum_accepted_samples, int)
            or minimum_accepted_samples < GROUNDING_QUERY_MIN_ACCEPTED_SAMPLES
            or accepted_samples < minimum_accepted_samples
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
            or not GROUNDING_QUERY_REQUIRED_PRECISION <= float(required_precision) <= 1
            or float(measured_precision) < float(required_precision)
        ):
            raise ValueError("AVCS grounding checkpoint fails per-query confidence calibration gates")
        return dict(metadata)

    @staticmethod
    def _validated_rgb_frame(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError("AVCS raw frame must be a NumPy array")
        if frame.dtype != np.uint8:
            raise ValueError("AVCS raw frame must use uint8 pixels")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("AVCS raw frame must have RGB shape (height, width, 3)")
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("AVCS raw frame dimensions must be positive")
        if frame.shape[0] * frame.shape[1] > 64_000_000:
            raise ValueError("AVCS raw frame exceeds the pixel limit")
        return np.ascontiguousarray(frame)

    def ground_target_png(
        self,
        png_bytes: bytes,
        query: str,
        width: int,
        height: int,
        window_geometry: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any] | None:
        """Compatibility adapter for callers that still provide PNG frames."""
        if not png_bytes:
            raise ValueError("AVCS grounding requires non-empty screenshot pixels")
        with Image.open(BytesIO(png_bytes)) as image:
            frame = np.array(image.convert("RGB"))
        return self.ground_target_rgb(
            frame, query, width, height, window_geometry=window_geometry
        )

    def ground_target_rgb_bytes(
        self,
        frame_bytes: bytes,
        query: str,
        width: int,
        height: int,
        window_geometry: tuple[int, int, int, int] | None = None,
        source_frame_id: int | None = None,
    ) -> dict[str, Any] | None:
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
            or width * height > 64_000_000
        ):
            raise ValueError("AVCS raw grounding requires valid bounded frame dimensions")
        expected_size = width * height * 3
        if not isinstance(frame_bytes, bytes) or len(frame_bytes) != expected_size:
            raise ValueError("AVCS raw grounding frame length does not match its dimensions")
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
        return self.ground_target_rgb(
            frame,
            query,
            width,
            height,
            window_geometry=window_geometry,
            source_frame_id=source_frame_id,
        )

    def ground_target_rgb(
        self,
        frame: np.ndarray,
        query: str,
        width: int,
        height: int,
        window_geometry: tuple[int, int, int, int] | None = None,
        source_frame_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a target directly from the latest locally-held RGB frame."""
        if not self.goal_grounder_configured:
            raise RuntimeError("SONIC_AVCS_GROUNDING_CHECKPOINT is not configured")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("AVCS grounding requires a positive integer screen width")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("AVCS grounding requires positive screen dimensions")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("AVCS grounding requires a non-empty target query")
        if len(query.strip().encode("utf-8")) > 96:
            raise ValueError("AVCS grounding queries must be at most 96 UTF-8 bytes")
        if source_frame_id is not None and (
            isinstance(source_frame_id, bool)
            or not isinstance(source_frame_id, int)
            or source_frame_id <= 0
        ):
            raise ValueError("AVCS source frame ID must be a positive integer")
        frame = self._validated_rgb_frame(frame)
        if frame.shape[1] != width or frame.shape[0] != height:
            raise ValueError("AVCS grounding frame dimensions do not match the observation")
        fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
        with self._lock:
            if fingerprint != self._last_frame_fingerprint:
                raise RuntimeError("AVCS grounding pixels do not match the latest perceived frame")
            if source_frame_id is not None and source_frame_id != self._last_source_frame_id:
                raise RuntimeError("AVCS grounding frame ID does not match the latest perceived frame")
            frame_index = self._frame_index
            current_source_frame_id = self._last_source_frame_id
            scene_hash = self._last_scene_hash

        with self._grounding_lock:
            if self._grounder is None:
                self._grounder = self._load_goal_grounder()
            grounder = self._grounder
            checkpoint_metadata = self._validated_checkpoint_metadata(grounder)

        inference_frame = frame
        crop_origin = (0, 0)
        if checkpoint_metadata["coordinate_frame"] == "cropped_target_window":
            if (
                not isinstance(window_geometry, (tuple, list))
                or len(window_geometry) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in window_geometry
                )
            ):
                raise ValueError(
                    "AVCS grounding requires current active-window geometry "
                    "for a window-cropped checkpoint"
                )
            left, top, window_width, window_height = window_geometry
            right = min(width, left + window_width)
            bottom = min(height, top + window_height)
            left = max(0, left)
            top = max(0, top)
            if right <= left or bottom <= top:
                raise ValueError("AVCS active-window geometry does not intersect the screen")
            crop_origin = (left, top)
            inference_frame = frame[top:bottom, left:right]

        started = time.perf_counter()
        prediction = grounder.predict(inference_frame, query)
        latency_ms = (time.perf_counter() - started) * 1_000
        with self._lock:
            if (
                self._frame_index != frame_index
                or self._last_frame_fingerprint != fingerprint
                or self._last_source_frame_id != current_source_frame_id
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
        x = crop_origin[0] + int(x_norm * inference_frame.shape[1])
        y = crop_origin[1] + int(y_norm * inference_frame.shape[0])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("AVCS grounding returned an off-screen point")
        return {
            "engine": "avcs_goal_grounder",
            "x": x,
            "y": y,
            "point_norm": [x_norm, y_norm],
            "frame_index": frame_index,
            "source_frame_id": current_source_frame_id,
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
        return self.perceive_rgb(frame)

    def perceive_rgb(
        self,
        frame: np.ndarray,
        source_frame_id: int | None = None,
    ) -> dict[str, Any]:
        """Process one RGB uint8 frame without image-file or base64 encoding."""
        frame = self._validated_rgb_frame(frame)
        if source_frame_id is not None and (
            isinstance(source_frame_id, bool)
            or not isinstance(source_frame_id, int)
            or source_frame_id <= 0
        ):
            raise ValueError("AVCS source frame ID must be a positive integer")
        with self._lock:
            if (
                source_frame_id is not None
                and self._source_frame_id_high_watermark is not None
                and source_frame_id <= self._source_frame_id_high_watermark
            ):
                raise ValueError("AVCS source frame IDs must increase monotonically")
            self._frame_index += 1
            state = self._reflex.perceive(frame, frame_index=self._frame_index)
            events = events_from_states(self._previous_state, state)
            predictive_state = self._system3_predictor.observe(
                state.luma_grid,
                frame_index=state.frame_index,
            )
            self._previous_state = state
            self._last_source_frame_id = source_frame_id
            if source_frame_id is not None:
                self._source_frame_id_high_watermark = source_frame_id
            self._last_frame_fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
            self._last_scene_hash = state.scene_hash

        return {
            "engine": "avcs_reflex",
            "system3": predictive_state,
            "frame_index": state.frame_index,
            "source_frame_id": source_frame_id,
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

    def perceive_semantics_rgb(self, frame: np.ndarray) -> dict[str, Any]:
        """Learned UI semantics for one frame, or an honest empty result.

        Returns ``{}`` when no trained cortex checkpoint is loaded. The caller
        must treat an empty mapping as "semantics unavailable", never as a
        neutral/negative classification.
        """
        if not self.ensure_cortex_ready():
            return {}
        try:
            assert self._cortex is not None
            state = self._cortex.perceive(frame)
            return {
                "primary_category": state.primary_category,
                "category_confidence": float(state.category_confidence),
                "text_density": float(state.text_density),
                "stability_state": state.stability_state,
                "surprise_score": float(state.surprise_score),
                "action_hotspots": [
                    {"x": float(x), "y": float(y), "action": action, "confidence": float(conf)}
                    for x, y, action, conf in (state.action_hotspots or [])
                ],
                "latent_dim": int(getattr(state.visual_latent, "shape", (0,))[0]),
            }
        except Exception:
            self._cortex_failed = True
            return {}

    def perceive_rgb_bytes(
        self,
        frame_bytes: bytes,
        width: int,
        height: int,
        source_frame_id: int,
    ) -> dict[str, Any]:
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
            or width * height > 64_000_000
        ):
            raise ValueError("AVCS raw frame requires valid bounded dimensions")
        if not isinstance(frame_bytes, bytes) or len(frame_bytes) != width * height * 3:
            raise ValueError("AVCS raw frame byte length does not match its dimensions")
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
        return self.perceive_rgb(frame, source_frame_id=source_frame_id)
