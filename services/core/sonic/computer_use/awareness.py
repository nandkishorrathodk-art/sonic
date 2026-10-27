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
import os
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
# The eye speaking: a human-readable line about what the screen is doing.  This
# is what a blind planner subscribes to.  Only published when the narrator has
# something worth saying, so a quiet screen stays quiet.
TOPIC_NARRATION = "perception.narration"

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
    # The eye's spoken line for this frame, or empty when it had nothing worth
    # saying.  A blind planner reads this instead of pixels.
    narration: str = ""
    narration_urgency: str = ""

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
            "narration": self.narration,
            "narration_urgency": self.narration_urgency,
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
        ambient_key: str = "",
    ) -> None:
        if fps <= 0:
            raise ValueError("Awareness fps must be positive")
        self._provider = provider
        self._workspace_id = workspace_id
        self._bus = bus
        # Shared identity used to hand the eye's lines to the brain.  The
        # workspace id alone is not enough: the awareness monitor and the agent
        # may name the same desktop differently, so both sides key the ambient
        # narration channel on a value they can each derive (the tenant id).
        self._ambient_key = ambient_key or workspace_id
        self._fps = min(float(fps), self.MAX_FPS)
        self._interval = 1.0 / self._fps
        self._history: deque[AwarenessFrame] = deque(maxlen=max(1, history))
        self._task: asyncio.Task[None] | None = None
        self._stopping = threading.Event()
        self._frames_published = 0
        self._capture_errors = 0
        self._last_publish_at = 0.0
        self._lock = threading.Lock()
        # The eye's voice.  Built lazily so a monitor can be constructed in an
        # environment without the AVCS narrator package; narration is then
        # simply absent and the monitor keeps publishing raw state.
        self._narrator: Any | None = None
        self._narrator_ready = False
        # One perception engine per monitor (i.e. per desktop).  Sharing a
        # single engine across desktops would mix their frames into one
        # temporal memory, producing phantom "changed/motion" lines and tripping
        # the monotonic frame-id guard between unrelated captures.
        self._perception: Any | None = None
        self._perception_ready = False
        # Apprenticeship: when enabled, frames the eye is already looking at
        # become honestly-labeled training data for its own cortex.  Off by
        # default; a normal boot pays nothing.  The window class is only probed
        # when the active window actually changes, so the cost is near zero.
        self._apprenticeship: Any | None = None
        self._apprenticeship_ready = False
        self._apprenticeship_key = ""
        self._apprenticeship_round = 0
        # Outcome count at the last affordance fine-tune, so a static desktop
        # does not re-run the trainer on every frame.
        self._last_affordance_trained_count = 0
        self._last_wm_class: str | None = None
        self._window_generation = 0

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

        perception = self._perception_engine()
        state: dict[str, Any] = {}
        semantics: dict[str, Any] = {}
        rgb_frame: Any = None
        if getattr(screen, "raw_rgb", b""):
            state = perception.perceive_rgb_bytes(
                screen.raw_rgb, screen.width, screen.height, screen.frame_id
            )
            try:
                import numpy as np

                rgb_frame = np.frombuffer(screen.raw_rgb, dtype=np.uint8).reshape(
                    screen.height, screen.width, 3
                )
                semantics = perception.perceive_semantics_rgb(rgb_frame)
            except Exception:
                semantics = {}
        elif getattr(screen, "screenshot_base64", ""):
            import base64

            png_bytes = base64.b64decode(screen.screenshot_base64, validate=True)
            state = perception.perceive_png(png_bytes)
            rgb_frame = _decode_png_to_rgb(png_bytes)

        if state.get("engine") != "avcs_reflex":
            return None

        capture_ms = (time.perf_counter() - t0) * 1000.0
        narration, narration_urgency = self._narrate(rgb_frame, state, screen)
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
            narration=narration,
            narration_urgency=narration_urgency,
        )

        with self._lock:
            previous = self._history[-1] if self._history else None
            self._history.append(frame)
        self._frames_published += 1

        self._publish(TOPIC_FRAME, frame, previous)
        await self._maybe_apprentice(rgb_frame, screen)
        return frame

    async def _maybe_apprentice(self, rgb_frame: Any, screen: Any) -> None:
        """Store this frame as a labeled sample, when apprenticeship is on.

        Best-effort and bounded: a capture/learn failure must never stop
        awareness.  The frame is only stored when its window resolves to an
        honest label; an unlabelable frame is skipped rather than guessed.
        """
        loop = self._apprenticeship_loop()
        if loop is None or rgb_frame is None:
            return
        try:
            wm_class, has_active_window = await self._window_identity(screen)
            label, source = await loop.labeler.label_for(wm_class, has_active_window)
            if label is None:
                return
            loop.store.add(
                rgb_frame, label, wm_class, source, group=self._apprenticeship_group()
            )
            self._apprenticeship_round += 1
            await self._maybe_learn(loop)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("awareness_apprenticeship_failed", error=str(exc))

    def _apprenticeship_group(self) -> str:
        """A group id that changes when the focused window changes.

        Frames of one continuous look at a window are near-duplicates and must
        stay on one side of the train/held-out split, so they share a group.
        """
        return f"{self._apprenticeship_key or 'desktop'}:{self._window_generation}"

    async def _window_identity(self, screen: Any) -> tuple[str, bool]:
        """The active window's WM_CLASS, probed only when the window changes."""
        active_window = str(getattr(screen, "active_window", "") or "")
        has_active_window = active_window not in ("", "Desktop", "None")
        get_class = getattr(self._provider, "get_active_window_class", None)
        if not callable(get_class):
            return "", has_active_window
        if self._last_wm_class is not None and self._last_wm_class == active_window:
            return self._apprenticeship_key, has_active_window
        try:
            wm_class = await get_class(self._workspace_id)
        except Exception:
            wm_class = ""
        self._last_wm_class = active_window
        self._apprenticeship_key = wm_class or ""
        self._window_generation += 1
        return self._apprenticeship_key, has_active_window

    async def _maybe_learn(self, loop: Any) -> None:
        """Trigger a learning round once enough new samples have accrued."""
        try:
            every = int(os.environ.get("SONIC_AVCS_APPRENTICE_LEARN_EVERY", "40") or 40)
        except ValueError:
            every = 40
        if every > 0 and self._apprenticeship_round >= every:
            self._apprenticeship_round = 0
            result = await loop.learn()
            logger.info(
                "awareness_apprenticeship_round",
                promoted=result.promoted,
                reason=result.reason,
                candidate_accuracy=result.candidate_accuracy,
                candidate_macro_f1=result.macro_f1,
            )
        await self._maybe_learn_affordance(loop)

    async def _maybe_learn_affordance(self, loop: Any) -> None:
        """Fine-tune the affordance head once enough new click outcomes exist.

        Clicks are far rarer than frames, so this is keyed on the *outcome*
        count, not the frame count. Training only runs when the count has grown
        by a real batch since the last round, so a static desktop does not spin
        the trainer. The head's readiness is still decided by the provenance
        gate; this only produces the candidate.
        """
        try:
            every = int(
                os.environ.get("SONIC_AVCS_AFFORDANCE_LEARN_EVERY", "40") or 40
            )
        except ValueError:
            every = 40
        if every <= 0:
            return
        try:
            total = int(loop.affordance_sample_counts().get("total", 0))
        except Exception:
            return
        if total < every:
            return
        if total - self._last_affordance_trained_count < every:
            return
        self._last_affordance_trained_count = total
        try:
            result = await loop.learn_affordance()
        except Exception as exc:
            logger.warning("awareness_affordance_learn_failed", error=str(exc))
            return
        logger.info(
            "awareness_affordance_round",
            trained=result.get("trained"),
            point_accuracy=result.get("held_out_point_accuracy"),
            samples=result.get("samples"),
        )

    def _apprenticeship_loop(self) -> Any | None:
        """Lazily build the apprenticeship loop when enabled (env-gated)."""
        if self._apprenticeship_ready:
            return self._apprenticeship
        self._apprenticeship_ready = True
        if os.environ.get("SONIC_AVCS_APPRENTICESHIP", "") != "1":
            return None
        try:
            from sonic.computer_use.apprenticeship import (
                get_shared_apprenticeship_loop,
            )

            # Share the process-wide loop so category frames and measured click
            # outcomes land in ONE store; separate loops split the eye's memory.
            self._apprenticeship = get_shared_apprenticeship_loop(self._provider)
            if self._apprenticeship is not None:
                logger.info(
                    "avcs_apprenticeship_enabled",
                    store=str(self._apprenticeship.store.root),
                )
        except Exception as exc:
            logger.warning("avcs_apprenticeship_init_failed", error=str(exc))
            self._apprenticeship = None
        return self._apprenticeship

    def _perception_engine(self) -> Any:
        """Lazily build this monitor's own AVCS perception engine.

        One engine per desktop, not one per process: the engine holds temporal
        state (previous frame, frame-id watermark, prediction memory) that is
        only meaningful within a single capture stream.  Sharing it would blend
        two desktops' frames together.
        """
        if not self._perception_ready:
            try:
                from sonic.computer_use.avcs_perception import AVCSPerception

                self._perception = AVCSPerception()
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("awareness_perception_unavailable", error=str(exc))
                self._perception = None
            self._perception_ready = True
        if self._perception is None:
            raise RuntimeError("AVCS perception engine is unavailable")
        return self._perception

    def _narrate(
        self, rgb_frame: Any, state: dict[str, Any], screen: Any
    ) -> tuple[str, str]:
        """Ask the eye to speak about this frame.

        Returns ``("", "")`` when there is nothing worth saying or the narrator
        is unavailable.  Never raises: narration is an enhancement and must not
        be able to stop the awareness loop.
        """
        if rgb_frame is None or not isinstance(state, dict):
            return "", ""
        if not self._narrator_ready:
            try:
                from sonic.computer_use.avcs_narrator import AVCSNarrator

                self._narrator = AVCSNarrator()
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("awareness_narrator_unavailable", error=str(exc))
                self._narrator = None
            self._narrator_ready = True
        if self._narrator is None:
            return "", ""
        active_window = str(getattr(screen, "active_window", "") or "")
        try:
            line = self._narrator.observe_frame(
                self._workspace_id, rgb_frame, state, active_window=active_window
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("awareness_narration_failed", error=str(exc))
            return "", ""
        if not isinstance(line, dict):
            return "", ""
        # The eye noticed something while the brain may have been busy.  Record
        # it on the ambient channel so the next reasoning turn hears about it
        # even though this observation was not requested by any agent.
        try:
            from sonic.computer_use.avcs_narrator import record_ambient_narration

            record_ambient_narration(self._ambient_key, line)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("awareness_ambient_record_failed", error=str(exc))
        return str(line.get("text") or ""), str(line.get("urgency") or "")

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

        # The eye speaking.  Published separately (and only when there is a
        # line) so a consumer can subscribe to *commentary* without also
        # receiving every raw frame tick.
        if frame.narration:
            try:
                self._bus.publish(TOPIC_NARRATION, payload)
            except Exception as exc:
                logger.warning(
                    "awareness_publish_failed", topic=TOPIC_NARRATION, error=str(exc)
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


def _decode_png_to_rgb(png_bytes: bytes) -> Any:
    """Decode PNG bytes to an ``(H, W, 3)`` array for narration, or ``None``.

    Best effort: the reflex state is already computed from the same bytes by
    ``perceive_png``, so a decode failure here only costs the narration line,
    not the perception.  No OCR and no resampling -- just pixels.
    """
    if not png_bytes:
        return None
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image

        with Image.open(BytesIO(png_bytes)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:  # pragma: no cover - environment dependent
        return None


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
