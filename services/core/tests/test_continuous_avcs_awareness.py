"""Done-gate for continuous AVCS awareness (the EYES -> BRAIN link).

These tests exercise the real :class:`AwarenessMonitor` against a stub provider
that returns genuine AVCS-processable frames, so the perception path runs for
real. They assert the properties that make awareness "living" rather than
on-demand:

* frames are sampled continuously without an operator prompt;
* real screen changes are published as change events;
* an unchanged screen does NOT publish a change event (no false alarms);
* the monitor never dispatches a GUI action (observation only);
* a capture error does not kill the loop;
* one monitor per workspace is enforced.
"""

from __future__ import annotations

import asyncio


import numpy as np
import pytest

from sonic.computer.models import ScreenObservation
from sonic.computer_use.awareness import (
    TOPIC_CHANGE,
    TOPIC_FRAME,
    AwarenessMonitor,
    get_awareness_monitor,
    reset_awareness_monitors,
    stop_awareness_monitor,
)


class _RecordingBus:
    """Minimal EventBus stand-in that records published topics."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, payload: dict | None = None) -> None:
        self.events.append((topic, payload or {}))

    def topics(self) -> list[str]:
        return [topic for topic, _ in self.events]

    def payloads(self, topic: str) -> list[dict]:
        return [payload for name, payload in self.events if name == topic]


class _StubProvider:
    """Returns a controllable RGB frame and records any gui_action calls."""

    def __init__(self, width: int = 320, height: int = 200) -> None:
        self.width = width
        self.height = height
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame[:, :] = 30
        self.frame_id = 0
        self.screenshot_calls = 0
        self.gui_action_calls = 0

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame

    async def screenshot(self, workspace_id: str) -> ScreenObservation:
        self.screenshot_calls += 1
        self.frame_id += 1
        rgb = np.ascontiguousarray(self._frame).tobytes()
        return ScreenObservation(
            workspace_id=workspace_id,
            screenshot_base64="",
            width=self.width,
            height=self.height,
            desktop_state="INTERACTIVE",
            raw_rgb=rgb,
            frame_id=self.frame_id,
        )

    async def gui_action(self, *args, **kwargs):  # pragma: no cover - must not run
        self.gui_action_calls += 1
        raise AssertionError("awareness must never dispatch GUI actions")


def _structured_frame(width: int = 320, height: int = 200, shift: int = 0) -> np.ndarray:
    """A frame with real structure so reflex produces a meaningful scene hash."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = 30
    # Window title bar
    frame[0:20, :] = (45, 45, 48)
    # A panel whose position shifts, so consecutive frames differ.
    x0 = 40 + shift
    frame[40:120, x0 : x0 + 120] = (200, 200, 210)
    return frame


@pytest.fixture(autouse=True)
def _clean_monitors():
    reset_awareness_monitors()
    yield
    reset_awareness_monitors()


def test_monitor_samples_continuously_without_prompt():
    """Frames must be captured repeatedly with no operator request."""

    async def scenario():
        provider = _StubProvider()
        provider.set_frame(_structured_frame())
        bus = _RecordingBus()
        monitor = AwarenessMonitor(provider, "ws-1", bus=bus, fps=20.0)
        monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()
        return provider, bus, monitor

    provider, bus, monitor = asyncio.run(scenario())
    assert provider.screenshot_calls >= 3, "expected continuous sampling"
    assert monitor.stats()["frames_published"] >= 3
    assert TOPIC_FRAME in bus.topics()


def test_real_screen_change_publishes_change_event():
    """A genuine visual change must surface as a perception.change event."""

    async def scenario():
        provider = _StubProvider()
        provider.set_frame(_structured_frame(shift=0))
        bus = _RecordingBus()
        monitor = AwarenessMonitor(provider, "ws-2", bus=bus, fps=20.0)
        monitor.start()
        await asyncio.sleep(0.15)
        # Move the panel a long way - an unmistakable change.
        provider.set_frame(_structured_frame(shift=120))
        await asyncio.sleep(0.35)
        await monitor.stop()
        return bus

    bus = asyncio.run(scenario())
    change_payloads = bus.payloads(TOPIC_CHANGE)
    assert change_payloads, "a real screen change must publish perception.change"
    assert any(p.get("changed") for p in change_payloads)


def test_unchanged_screen_publishes_no_change_event():
    """A static screen must not raise false change events."""

    async def scenario():
        provider = _StubProvider()
        provider.set_frame(_structured_frame())
        bus = _RecordingBus()
        monitor = AwarenessMonitor(provider, "ws-3", bus=bus, fps=20.0)
        monitor.start()
        # Poll instead of a fixed sleep: under a loaded suite the sampler can
        # publish fewer frames than the assertion needs even though it works.
        deadline = asyncio.get_running_loop().time() + 5.0
        while monitor.stats()["frames_published"] < 3:
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(0.02)
        await monitor.stop()
        return bus, monitor

    bus, monitor = asyncio.run(scenario())
    assert monitor.stats()["frames_published"] >= 3
    assert not bus.payloads(TOPIC_CHANGE), "static screen raised a false change"


def test_monitor_never_dispatches_gui_actions():
    """Awareness observes only - it must not act on the desktop."""

    async def scenario():
        provider = _StubProvider()
        provider.set_frame(_structured_frame())
        monitor = AwarenessMonitor(provider, "ws-4", bus=None, fps=20.0)
        monitor.start()
        await asyncio.sleep(0.25)
        await monitor.stop()
        return provider

    provider = asyncio.run(scenario())
    assert provider.gui_action_calls == 0


def test_capture_error_does_not_kill_the_loop():
    """A transient capture failure must be survived, not fatal."""

    class _FlakyProvider(_StubProvider):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = 1

        async def screenshot(self, workspace_id: str) -> ScreenObservation:
            if self.fail_next > 0:
                self.fail_next -= 1
                raise RuntimeError("transient capture failure")
            return await super().screenshot(workspace_id)

    async def scenario():
        provider = _FlakyProvider()
        provider.set_frame(_structured_frame())
        monitor = AwarenessMonitor(provider, "ws-5", bus=None, fps=20.0)
        monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()
        return monitor

    monitor = asyncio.run(scenario())
    stats = monitor.stats()
    assert stats["capture_errors"] >= 1, "the injected failure was not observed"
    assert stats["frames_published"] >= 2, "loop died after a transient error"


def test_one_monitor_per_workspace_is_enforced():
    provider = _StubProvider()
    first = get_awareness_monitor(provider, "ws-6", bus=None)
    second = get_awareness_monitor(provider, "ws-6", bus=None)
    assert first is second

    async def cleanup():
        assert await stop_awareness_monitor("ws-6") is True
        assert await stop_awareness_monitor("ws-6") is False

    asyncio.run(cleanup())


def test_fps_is_capped_to_protect_the_host():
    provider = _StubProvider()
    monitor = AwarenessMonitor(provider, "ws-7", bus=None, fps=1000.0)
    assert monitor.stats()["target_fps"] <= AwarenessMonitor.MAX_FPS


def test_invalid_fps_is_rejected():
    provider = _StubProvider()
    with pytest.raises(ValueError):
        AwarenessMonitor(provider, "ws-8", bus=None, fps=0)


def test_semantics_absent_without_trained_checkpoint(monkeypatch):
    """Without a trained cortex checkpoint, semantics must be honestly empty."""
    monkeypatch.delenv("SONIC_AVCS_CORTEX_CHECKPOINT", raising=False)
    from sonic.computer_use.avcs_perception import AVCSPerception

    perception = AVCSPerception()
    assert perception.cortex_configured is False
    assert perception.cortex_ready is False
    assert perception.ensure_cortex_ready() is False
    assert perception.perceive_semantics_rgb(_structured_frame()) == {}


def test_cortex_fails_closed_on_missing_checkpoint(tmp_path):
    """A configured-but-missing checkpoint must NOT report readiness."""
    from sonic.computer_use.avcs_perception import AVCSPerception

    perception = AVCSPerception()
    perception._cortex_checkpoint = str(tmp_path / "does_not_exist.pt")
    perception._cortex_failed = False
    assert perception.cortex_configured is True
    assert perception.ensure_cortex_ready() is False
    assert perception.cortex_ready is False
    assert perception.perceive_semantics_rgb(_structured_frame()) == {}


def test_cortex_loads_a_real_trained_checkpoint(tmp_path):
    """A genuine checkpoint must load and produce learned semantics."""
    import torch

    from forge.vision.cortex.config import ModelConfig
    from forge.vision.cortex.model import ArtificialVisualCortex
    from sonic.computer_use.avcs_perception import AVCSPerception

    checkpoint = tmp_path / "cortex.pt"
    torch.save(ArtificialVisualCortex(ModelConfig()).state_dict(), checkpoint)

    perception = AVCSPerception()
    perception._cortex_checkpoint = str(checkpoint)
    perception._cortex_failed = False
    assert perception.ensure_cortex_ready() is True
    assert perception.cortex_ready is True

    semantics = perception.perceive_semantics_rgb(_structured_frame())
    assert semantics, "a loaded cortex must emit semantics"
    assert "primary_category" in semantics
    assert 0.0 <= semantics["category_confidence"] <= 1.0


def test_awareness_stats_reflects_live_monitors():
    from sonic.computer_use.awareness import awareness_stats

    assert awareness_stats()["running"] is False

    async def scenario():
        provider = _StubProvider()
        provider.set_frame(_structured_frame())
        # Use the factory so the monitor is registered for observability.
        monitor = get_awareness_monitor(provider, "ws-9", bus=None, fps=20.0)
        monitor.start()
        await asyncio.sleep(0.15)
        stats = awareness_stats()
        await stop_awareness_monitor("ws-9")
        return stats

    stats = asyncio.run(scenario())
    assert stats["running"] is True
    assert stats["monitors"] == 1
    assert stats["per_workspace"]["ws-9"]["frames_published"] >= 1
    assert awareness_stats()["running"] is False


def test_capability_surface_reports_awareness_and_cortex_honestly(monkeypatch):
    """The capability report must not overstate AVCS readiness."""
    monkeypatch.delenv("SONIC_AVCS_CORTEX_CHECKPOINT", raising=False)
    monkeypatch.delenv("SONIC_AVCS_GROUNDING_CHECKPOINT", raising=False)
    monkeypatch.delenv("SONIC_AVCS_FOCUS_CHECKPOINT", raising=False)

    from sonic.api.routes.workstation_perception import avcs_capability_state

    capability = avcs_capability_state()
    assert capability["reflex_perception"] is True
    assert capability["learned_cortex_ready"] is False
    assert capability["gui_actions_ready"] is False
    assert "awareness_stats" in capability


def test_life_loop_consumes_perception_changes():
    """The being's cognition must actually receive AVCS change events."""
    from sonic.being.life_loop import BeingLifeLoop

    class _Agent:
        safety = object()

        async def idle_cycle(self, workspace_id, curiosity):
            class _R:
                learned_fact = None
                was_novel = False
                info_gain = 0.0
                proposed_goal = "noop"

            return _R()

    loop = BeingLifeLoop.__new__(BeingLifeLoop)
    loop.being = type("B", (), {"being_id": "b1", "tenant_id": "t1"})()
    loop.agent = _Agent()
    loop.curiosity = object()
    loop.workspace_id = "ws"
    loop.tick_interval = 0
    loop._task = None
    loop._stop = asyncio.Event()
    loop._operator_active = asyncio.Event()
    loop._tick_lock = asyncio.Lock()
    loop.cycles_completed = 0
    loop.before_tick_hooks = []
    loop.self_coder = None
    loop.self_coder_every = 20
    loop.self_coder_reports = []
    loop.recent_perception_changes = []
    from sonic.being.life_loop import EventBus

    loop.bus = EventBus()
    from sonic.computer_use.awareness import TOPIC_CHANGE, TOPIC_FRAME

    loop.bus.subscribe(TOPIC_CHANGE, loop._on_perception_change)
    loop.bus.subscribe(TOPIC_FRAME, loop._on_perception_frame)

    # Simulate the eyes: a real change event from the awareness monitor.
    loop.bus.publish(TOPIC_CHANGE, {"changed": True, "change_fraction": 0.4})
    loop.bus.publish(TOPIC_FRAME, {"frame_index": 7, "scene_hash": "abc"})

    context = loop.perception_context()
    assert context["change_count"] == 1
    assert context["latest_frame"]["frame_index"] == 7

    # Consuming the context clears the buffer (no double-counting next tick).
    assert loop.perception_context()["change_count"] == 0


class TestNarratorNeverBlocksPerception:
    """The VLM captioner must never stall the awareness loop.

    A synchronous VLM load/inference on the sampling loop blocked the event
    loop for ~19 s, so the loop produced 2 frames instead of >=3 and the
    monitor could not sample continuously.
    """

    def test_captioner_load_is_off_thread_and_caption_is_non_blocking(self):
        import threading
        import time

        from sonic.computer_use import avcs_narrator as module

        started = threading.Event()
        release = threading.Event()

        class _SlowDescriber:
            def caption(self, frame):  # noqa: ANN001
                started.set()
                release.wait(5)
                return "A window is open."

        class _SlowLoader:
            @staticmethod
            def try_load(model_name=""):  # noqa: ANN001
                started.set()
                release.wait(5)
                return _SlowDescriber()

        import sys
        import types

        fake_describer_mod = types.ModuleType("forge.vision.describer")
        fake_describer_mod.VLMDescriber = _SlowLoader  # type: ignore[attr-defined]
        fake_pkg = types.ModuleType("forge.vision")
        fake_pkg.describer = fake_describer_mod  # type: ignore[attr-defined]
        prev_forge = sys.modules.get("forge.vision")
        prev_desc = sys.modules.get("forge.vision.describer")
        sys.modules["forge.vision"] = fake_pkg
        sys.modules["forge.vision.describer"] = fake_describer_mod
        try:
            captioner = module._BackgroundVLMCaptioner("slow-model")
            # Construction must return immediately despite the 5 s load.
            t0 = time.perf_counter()
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            first = captioner.caption(frame)
            elapsed = time.perf_counter() - t0
            assert elapsed < 1.0, "caption() blocked the caller"
            assert first == "", "no caption can exist before the model loads"
        finally:
            release.set()
            if prev_forge is None:
                sys.modules.pop("forge.vision", None)
            else:
                sys.modules["forge.vision"] = prev_forge
            if prev_desc is None:
                sys.modules.pop("forge.vision.describer", None)
            else:
                sys.modules["forge.vision.describer"] = prev_desc

    def test_awareness_loop_keeps_sampling_while_vlm_loads(self, monkeypatch):
        """Perception must keep sampling while a VLM is loading.

        Keeps the real non-blocking captioner and only makes the underlying
        model load slow, so this measures the production wrapper rather than a
        hand-written stand-in.
        """
        import threading
        import sys
        import types

        gate = threading.Event()

        class _SlowDescriber:
            def caption(self, frame):  # noqa: ANN001
                gate.wait(10)
                return ""

        class _SlowLoader:
            @staticmethod
            def try_load(model_name=""):  # noqa: ANN001
                gate.wait(10)
                return _SlowDescriber()

        fake_mod = types.ModuleType("forge.vision.describer")
        fake_mod.VLMDescriber = _SlowLoader  # type: ignore[attr-defined]
        prev = sys.modules.get("forge.vision.describer")
        sys.modules["forge.vision.describer"] = fake_mod
        monkeypatch.setenv("SONIC_AVCS_VLM_DESCRIBER", "slow-fake")
        try:
            async def scenario():
                provider = _StubProvider()
                provider.set_frame(_structured_frame())
                monitor = AwarenessMonitor(provider, "ws-vlm", fps=20.0)
                monitor.start()
                await asyncio.sleep(0.4)
                await monitor.stop()
                return provider

            provider = asyncio.run(scenario())
        finally:
            gate.set()
            if prev is None:
                sys.modules.pop("forge.vision.describer", None)
            else:
                sys.modules["forge.vision.describer"] = prev
        assert provider.screenshot_calls >= 3, "a loading VLM stalled perception"
