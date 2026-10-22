"""Continuous visual awareness: the EYES -> BRAIN link.

The workstation perception path can already turn a frame into a compact reflex
state, but nothing ever *asks* for the next frame unless an operator sends a
prompt. That is the difference between a camera you take a photo with and a pair
of eyes that stay open.

This module owns a single long-lived task that keeps sampling the desktop and
publishes perception events on the shared :class:`EventBus`. Consumers (the
being's cognition loop, the dashboard worklog, a Boss worker waiting for the
screen to settle) subscribe to those topics instead of polling a screenshot.

Design constraints, deliberately chosen:

* **It never acts.** Awareness only observes and publishes. No GUI action is
  dispatched from this task, so it cannot fight an agent that owns the desktop.
* **It is single-instance.** Two awareness loops on one display would double the
  capture cost and race on the same pixels; :func:`get_awareness_monitor`
  enforces one per workspace.
* **It is bounded.** Frame rate, event emission rate and history length are all
  capped, so a long-running desktop cannot grow memory without limit.
* **It fails quiet, not loud.** A capture error is logged and the loop keeps
  going; a dead monitor would otherwise silently end all awareness.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sonic.logger import get_logger

logger = get_logger(__name__)

# Topics published on the shared bus. Consumers subscribe to these rather than
# reaching into the monitor's internals.
TOPIC_FRAME = "perception.frame"
TOPIC_CHANGE = "perception.change"
TOPIC_OVERLAY = "perception.overlay"
TOPIC_STABLE = "perception.stable"

# Per-frame event kinds the reflex layer can raise, mapped to bus topics. Only
# the meaningful ones get their own topic; everything else is just a frame tick.
_EVENT_TOPICS = {
    "scene_change": TOPIC_CHANGE,
    "overlay": TOPIC_OVERLAY,
    "appear": TOPIC_CHANGE,
}


@dataclass
class AwarenessFrame:
    """One sampled frame's compact state, as published on the bus."""

    frame_index: int
    captured_at: float
    width: int
    height: int
    scene_hash: str
    changed: bool
    change_fraction: float
    global_motion: float
    events: list[str] = field(default_factory=list)
    semantics: dict[str, Any] = field(default_factory=dict)
    capture_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "captured_at": self.captured_at,
            "width": self.width,
            "height": self.height,
            "scene_hash": self.scene_hash,
            "changed": self.changed,
            "change_fraction": self.change_fraction,
            "global_motion": self.global_motion,
            "events": list(self.events),
            "semantics": dict(self.semantics),
            "capture_ms": self.capture_ms,
        }


class AwarenessMonitor:
    """Samples the desktop continuously and publishes perception events.

    Args:
        provider:  a ComputerProvider (anything exposing ``screenshot``).
        workspace_id: the desktop to watch.
        bus:       EventBus to publish on. When None, events are only kept in
                   the local ring buffer (useful for tests).
        fps:       target sampling rate. Capped to keep CPU sane; the real rate
                   is whatever the capture path can sustain.
        history:   number of recent frames kept for consumers to read back.
    """

    MAX_FPS = 10.0

    def __init__(
        self,
        provider: Any,
        workspace_id: str,
        bus: Any | None = None,
        fps: float = 2.0,
        history: int = 32,
    ) -> None:
        if fps <= 0:
            raise ValueError("Awareness fps must be positive")
        self._provider = provider
        self._workspace_id = workspace_id
        self._bus = bus
        self._fps = min(float(fps), self.MAX_FPS)
        self._interval = 1.0 / self._fps
        self._history: deque[AwarenessFrame] = deque(maxlen=max(1, history))
        self._task: asyncio.Task[None] | None = None
        self._stopping = threading.Event()
        self._frames_published = 0
        self._capture_errors = 0
        self._last_publish_at = 0.0
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> "AwarenessMonitor":
        """Start the sampling task (idempotent)."""
        if self.running:
            return self
        self._stopping.clear()
        self._task = asyncio.ensure_future(self._run())
        logger.info(
            "awareness_monitor_started",
            workspace_id=self._workspace_id,
            fps=self._fps,
        )
        return self

    async def stop(self) -> None:
        """Stop sampling and wait for the task to finish."""
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - cancel is expected
                pass
        logger.info(
            "awareness_monitor_stopped",
            workspace_id=self._workspace_id,
            frames=self._frames_published,
            capture_errors=self._capture_errors,
        )

    # -- sampling ----------------------------------------------------------
    async def _run(self) -> None:
        while not self._stopping.is_set():
            started = time.perf_counter()
            try:
                await self._sample_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._capture_errors += 1
                logger.warning(
                    "awareness_sample_failed",
                    workspace_id=self._workspace_id,
                    error=str(exc),
                )
            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.0, self._interval - elapsed))

    async def _sample_once(self) -> AwarenessFrame | None:
        """Capture one frame, derive state, publish. Returns the frame."""
        t0 = time.perf_counter()
        screen = await self._provider.screenshot(self._workspace_id)
        if screen is None or getattr(screen, "desktop_state", None) != "INTERACTIVE":
            return None

        perception = _shared_perception()
        state: dict[str, Any] = {}
        semantics: dict[str, Any] = {}
        if getattr(screen, "raw_rgb", b""):
            state = perception.perceive_rgb_bytes(
                screen.raw_rgb, screen.width, screen.height, screen.frame_id
            )
            try:
                import numpy as np

                frame = np.frombuffer(screen.raw_rgb, dtype=np.uint8).reshape(
                    screen.height, screen.width, 3
                )
                semantics = perception.perceive_semantics_rgb(frame)
            except Exception:
                semantics = {}
        elif getattr(screen, "screenshot_base64", ""):
            import base64

            state = perception.perceive_png(
                base64.b64decode(screen.screenshot_base64, validate=True)
            )

        if state.get("engine") != "avcs_reflex":
            return None

        capture_ms = (time.perf_counter() - t0) * 1000.0
        frame = AwarenessFrame(
            frame_index=int(state.get("frame_index", 0)),
            captured_at=time.time(),
            width=int(state.get("width", 0)),
            height=int(state.get("height", 0)),
            scene_hash=str(state.get("scene_hash", "")),
            changed=bool(state.get("scene_changed", False)),
            change_fraction=float(state.get("change_fraction", 0.0) or 0.0),
            global_motion=float(state.get("global_motion", 0.0) or 0.0),
            events=_event_kinds(state.get("events")),
            semantics=semantics,
            capture_ms=capture_ms,
        )

        with self._lock:
            previous = self._history[-1] if self._history else None
            self._history.append(frame)
        self._frames_published += 1

        self._publish(TOPIC_FRAME, frame, previous)
        return frame

    def _publish(
        self, topic: str, frame: AwarenessFrame, previous: AwarenessFrame | None
    ) -> None:
        if self._bus is None:
            return
        payload = frame.to_dict()
        if previous is not None:
            payload["previous_scene_hash"] = previous.scene_hash
            payload["since_last_ms"] = round(
                (frame.captured_at - previous.captured_at) * 1000.0, 2
            )
        try:
            self._bus.publish(topic, payload)
        except Exception as exc:
            logger.warning("awareness_publish_failed", topic=topic, error=str(exc))
            return

        for kind in frame.events:
            event_topic = _EVENT_TOPICS.get(kind)
            if event_topic:
                try:
                    self._bus.publish(event_topic, payload)
                except Exception as exc:
                    logger.warning(
                        "awareness_publish_failed", topic=event_topic, error=str(exc)
                    )

    # -- read-back ---------------------------------------------------------
    def latest(self) -> AwarenessFrame | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def recent(self, limit: int = 10) -> list[AwarenessFrame]:
        with self._lock:
            frames = list(self._history)
        return frames[-max(1, limit) :]

    def stats(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "running": self.running,
            "workspace_id": self._workspace_id,
            "target_fps": self._fps,
            "frames_published": self._frames_published,
            "capture_errors": self._capture_errors,
            "history_size": len(self._history),
            "last_scene_hash": latest.scene_hash if latest else "",
            "last_capture_ms": round(latest.capture_ms, 2) if latest else 0.0,
        }


def _event_kinds(events: Any) -> list[str]:
    kinds: list[str] = []
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                kind = event.get("kind")
                if isinstance(kind, str) and kind and kind not in kinds:
                    kinds.append(kind)
    return kinds


_perception_singleton: Any | None = None
_perception_lock = threading.Lock()


def _shared_perception() -> Any:
    """One AVCSPerception for the process.

    Constructing a fresh one per call would re-load the cortex checkpoint and
    discard the temporal memory that makes change detection meaningful.
    """
    global _perception_singleton
    if _perception_singleton is None:
        with _perception_lock:
            if _perception_singleton is None:
                from sonic.computer_use.avcs_perception import AVCSPerception

                _perception_singleton = AVCSPerception()
    return _perception_singleton


_monitors: dict[str, AwarenessMonitor] = {}
_monitors_lock = threading.Lock()


def get_awareness_monitor(
    provider: Any, workspace_id: str, bus: Any | None = None, **kwargs: Any
) -> AwarenessMonitor:
    """Return the single monitor for this workspace, creating it if needed."""
    with _monitors_lock:
        monitor = _monitors.get(workspace_id)
        if monitor is None:
            monitor = AwarenessMonitor(provider, workspace_id, bus=bus, **kwargs)
            _monitors[workspace_id] = monitor
        elif bus is not None:
            monitor._bus = bus
        return monitor


async def stop_awareness_monitor(workspace_id: str) -> bool:
    """Stop and forget the monitor for a workspace. Returns True if one ran."""
    with _monitors_lock:
        monitor = _monitors.pop(workspace_id, None)
    if monitor is None:
        return False
    await monitor.stop()
    return True


def reset_awareness_monitors() -> None:
    """Drop all monitor singletons (test helper)."""
    with _monitors_lock:
        _monitors.clear()
    global _perception_singleton
    with _perception_lock:
        _perception_singleton = None


def awareness_stats() -> dict[str, Any]:
    """Aggregate stats for every live monitor (observability surface)."""
    with _monitors_lock:
        monitors = list(_monitors.values())
    if not monitors:
        return {"running": False, "monitors": 0}
    per_monitor = {m._workspace_id: m.stats() for m in monitors}
    return {
        "running": any(m.running for m in monitors),
        "monitors": len(monitors),
        "per_workspace": per_monitor,
    }
