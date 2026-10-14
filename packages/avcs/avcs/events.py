"""Universal Visual Event Streamer (Differential Vision).

Translates high-frequency 60-120 FPS continuous perception into discrete, typed,
high-level visual events (Scene Transitions, Surprise Alerts, Motion Pursuit,
and Affordance Shifts) so AI agents and applications can react instantly without polling.
"""
from __future__ import annotations

import collections
import dataclasses
import enum
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from avcs.human_eye import HumanEyeState, MovingObject


class VisualEventType(str, enum.Enum):
    """Categorized visual events emitted by the Human Eye."""
    SCENE_TRANSITION = "scene_transition"      # Application switch, webpage navigation
    SURPRISE_ALERT = "surprise_alert"          # Sudden modal dialog, popup, error, flash
    MOTION_DETECTED = "motion_detected"        # Dynamic motion or target movement
    STABILITY_REACHED = "stability_reached"    # Animations finished, page settled
    AFFORDANCE_SHIFT = "affordance_shift"      # Interactive hotspots appeared or disappeared


@dataclasses.dataclass(frozen=True)
class VisualEvent:
    """Discrete, typed visual perception event."""
    event_type: VisualEventType
    timestamp: float
    frame_index: int
    title: str
    description: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": round(self.timestamp, 4),
            "frame_index": self.frame_index,
            "title": self.title,
            "description": self.description,
            "data": dict(self.data),
        }


class VisualEventStreamer:
    """Processes continuous HumanEyeState sequence and extracts discrete visual events."""

    def __init__(self, history_size: int = 200) -> None:
        self.history_size = history_size
        self._lock = threading.Lock()
        self.event_history: collections.deque[VisualEvent] = collections.deque(maxlen=history_size)
        self._listeners: Dict[VisualEventType, List[Callable[[VisualEvent], None]]] = {
            t: [] for t in VisualEventType
        }
        self._all_listeners: List[Callable[[VisualEvent], None]] = []

        # Previous state tracking for differential analysis
        self._prev_category: Optional[str] = None
        self._prev_stability: Optional[str] = None
        self._prev_surprise: float = 0.0
        self._prev_moving_count: int = 0
        self._prev_hotspot_count: int = 0
        self._last_event_time: float = time.perf_counter()

    def subscribe(
        self,
        event_type: Optional[VisualEventType | str] = None,
        callback: Optional[Callable[[VisualEvent], None]] = None,
    ) -> Callable:
        """Subscribes a callback to specific or all visual events."""
        def decorator(fn: Callable[[VisualEvent], None]) -> Callable:
            with self._lock:
                if event_type is None:
                    self._all_listeners.append(fn)
                else:
                    t = VisualEventType(event_type) if isinstance(event_type, str) else event_type
                    self._listeners[t].append(fn)
            return fn

        if callback is not None:
            return decorator(callback)
        return decorator

    def process_state(self, state: HumanEyeState) -> List[VisualEvent]:
        """Analyzes a continuous perception tick and emits new visual events."""
        new_events: List[VisualEvent] = []
        now = state.timestamp

        # 1. Detect Surprise Alert (Explosions, Error popups, Sudden modal dialogs)
        if state.surprise_score >= 0.80 and self._prev_surprise < 0.75:
            ev = VisualEvent(
                event_type=VisualEventType.SURPRISE_ALERT,
                timestamp=now,
                frame_index=state.frame_index,
                title="Surprise Visual Event Detected",
                description=f"Sudden visual shift or alert detected (surprise score: {state.surprise_score:.2f})",
                data={
                    "surprise_score": round(state.surprise_score, 4),
                    "stability": state.stability,
                    "primary_category": state.primary_category,
                },
            )
            new_events.append(ev)

        # 2. Detect Scene Transitions (Application switch, navigation)
        if self._prev_category is not None and state.primary_category != self._prev_category:
            if state.category_confidence >= 0.40:
                ev = VisualEvent(
                    event_type=VisualEventType.SCENE_TRANSITION,
                    timestamp=now,
                    frame_index=state.frame_index,
                    title="Scene Category Transition",
                    description=f"Visual context transitioned from '{self._prev_category}' to '{state.primary_category}'",
                    data={
                        "from_category": self._prev_category,
                        "to_category": state.primary_category,
                        "confidence": round(state.category_confidence, 3),
                    },
                )
                new_events.append(ev)

        # 3. Detect Stability Settling
        if self._prev_stability in ("dynamic_video", "surprise", "tracking") and state.stability == "stable":
            ev = VisualEvent(
                event_type=VisualEventType.STABILITY_REACHED,
                timestamp=now,
                frame_index=state.frame_index,
                title="Visual Field Settled",
                description="Visual scene transitions completed; screen state is now stable and settled",
                data={
                    "motion_magnitude": round(state.motion_magnitude, 4),
                    "gaze_fixation": state.gaze_fixation,
                },
            )
            new_events.append(ev)

        # 4. Detect Motion / Dynamic Entities
        current_moving_count = len(state.moving_objects)
        if current_moving_count > 0 and self._prev_moving_count == 0:
            top_obj = state.moving_objects[0]
            ev = VisualEvent(
                event_type=VisualEventType.MOTION_DETECTED,
                timestamp=now,
                frame_index=state.frame_index,
                title="Dynamic Target Movement Detected",
                description=f"Detected {current_moving_count} moving visual entities (peak speed: {top_obj.speed:.2f})",
                data={
                    "moving_count": current_moving_count,
                    "target_centroid": top_obj.centroid,
                    "target_velocity": top_obj.velocity,
                    "speed": round(top_obj.speed, 3),
                },
            )
            new_events.append(ev)

        # 5. Detect Affordance Shifts (Buttons/Inputs appeared or disappeared)
        current_hotspot_count = len(state.affordance_hotspots)
        if abs(current_hotspot_count - self._prev_hotspot_count) >= 2 and current_hotspot_count > 0:
            ev = VisualEvent(
                event_type=VisualEventType.AFFORDANCE_SHIFT,
                timestamp=now,
                frame_index=state.frame_index,
                title="Interactive Affordances Updated",
                description=f"Screen interactive hotspots updated ({current_hotspot_count} available targets)",
                data={
                    "hotspot_count": current_hotspot_count,
                    "hotspots": list(state.affordance_hotspots[:5]),
                },
            )
            new_events.append(ev)

        # Update historical trackers
        self._prev_category = state.primary_category
        self._prev_stability = state.stability
        self._prev_surprise = state.surprise_score
        self._prev_moving_count = current_moving_count
        self._prev_hotspot_count = current_hotspot_count

        # Dispatch events to listeners
        for ev in new_events:
            with self._lock:
                self.event_history.append(ev)
                typed_cbs = list(self._listeners.get(ev.event_type, []))
                all_cbs = list(self._all_listeners)

            for fn in typed_cbs:
                try:
                    fn(ev)
                except Exception as exc:
                    print(f"[VisualEventStreamer] Error in event listener: {exc}")
            for fn in all_cbs:
                try:
                    fn(ev)
                except Exception as exc:
                    print(f"[VisualEventStreamer] Error in general listener: {exc}")

        return new_events

    def get_recent_events(self, limit: int = 20) -> List[VisualEvent]:
        """Returns the most recent visual events."""
        with self._lock:
            return list(self.event_history)[-limit:]

    def clear(self) -> None:
        """Clears event history."""
        with self._lock:
            self.event_history.clear()

