"""Done-gate for the AVCS "talking eye" narration on the awareness path.

These tests exercise the real :class:`AwarenessMonitor` and the real
:class:`AVCSNarrator` wrapper against a stub provider that returns genuine
AVCS-processable RGB frames, so the narration path runs for real. They assert
the properties that make the eye "talk" rather than stay silent:

* a first frame produces a scene description (the eye introduces itself);
* an unchanged frame produces no line (silence is healthy, no flooding);
* a real visible change produces a line that mentions the change;
* narration carries the active window title when one is known;
* a narrator that raises never breaks the awareness loop;
* the learned cortex semantics ride alongside narration when configured.
"""

from __future__ import annotations

import asyncio

import numpy as np

from sonic.computer.models import ScreenObservation
from sonic.computer_use.avcs_narrator import AVCSNarrator
from sonic.computer_use.awareness import TOPIC_FRAME, AwarenessMonitor


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, payload: dict | None = None) -> None:
        self.events.append((topic, payload or {}))

    def payloads(self, topic: str) -> list[dict]:
        return [payload for name, payload in self.events if name == topic]


class _StubProvider:
    """Returns a controllable RGB frame; supports real AVCS perception."""

    def __init__(self, width: int = 320, height: int = 200) -> None:
        self.width = width
        self.height = height
        self._frame = np.full((height, width, 3), 30, dtype=np.uint8)
        self.frame_id = 0
        self.screenshot_calls = 0
        self.gui_actions: list[object] = []
        self.active_window = ""

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame

    async def screenshot(self, workspace_id: str) -> ScreenObservation:
        self.screenshot_calls += 1
        self.frame_id += 1
        return ScreenObservation(
            workspace_id=workspace_id,
            desktop_state="INTERACTIVE",
            width=self.width,
            height=self.height,
            raw_rgb=self._frame.tobytes(),
            frame_id=self.frame_id,
            active_window=self.active_window,
        )

    async def gui_action(self, workspace_id: str, action: object) -> dict:
        self.gui_actions.append(action)
        return {"status": "SUCCESS"}


def _run(coro):
    return asyncio.run(coro)


def _state(perception, frame: np.ndarray, frame_id: int = 1) -> dict:
    """Real AVCS reflex state dict, exactly what the awareness path produces."""
    height, width = frame.shape[:2]
    return perception.perceive_rgb_bytes(frame.tobytes(), width, height, frame_id)


class TestNarratorSpeaksAndStaysSilent:
    def test_first_frame_describes_the_scene(self) -> None:
        from sonic.computer_use.avcs_perception import AVCSPerception

        narrator = AVCSNarrator()
        perception = AVCSPerception()
        frame = np.full((200, 320, 3), 30, dtype=np.uint8)
        line = narrator.observe_frame("ws", frame, _state(perception, frame))
        assert isinstance(line, dict)
        assert line.get("text"), "the eye must introduce itself on the first frame"

    def test_unchanged_frame_is_silent(self) -> None:
        from sonic.computer_use.avcs_perception import AVCSPerception

        narrator = AVCSNarrator()
        perception = AVCSPerception()
        frame = np.full((200, 320, 3), 30, dtype=np.uint8)
        narrator.observe_frame("ws", frame, _state(perception, frame, 1))
        again = narrator.observe_frame("ws", frame, _state(perception, frame, 2))
        assert again is None, "an unchanged screen must not produce a line"

    def test_real_change_produces_a_line(self) -> None:
        from sonic.computer_use.avcs_perception import AVCSPerception

        narrator = AVCSNarrator()
        perception = AVCSPerception()
        base = np.full((200, 320, 3), 30, dtype=np.uint8)
        narrator.observe_frame("ws", base, _state(perception, base, 1))

        changed = base.copy()
        changed[20:120, 40:260] = [220, 40, 40]
        line = narrator.observe_frame(
            "ws", changed, _state(perception, changed, 2)
        )
        assert line is not None and line.get("text"), "a real change must be narrated"

    def test_narration_carries_active_window(self) -> None:
        from sonic.computer_use.avcs_perception import AVCSPerception

        narrator = AVCSNarrator()
        perception = AVCSPerception()
        frame = np.full((200, 320, 3), 30, dtype=np.uint8)
        line = narrator.observe_frame(
            "ws", frame, _state(perception, frame), active_window="Terminal - root"
        )
        assert line is not None
        assert "Terminal - root" in line.get("text", "")


class TestAwarenessPublishesNarration:
    def test_frames_carry_narration_and_stay_alive(self) -> None:
        async def scenario() -> tuple[int, list[dict], list[object]]:
            provider = _StubProvider()
            bus = _RecordingBus()
            monitor = AwarenessMonitor(provider, "ws", bus=bus, fps=20.0)
            monitor.start()
            await asyncio.sleep(0.25)
            frame = np.full((200, 320, 3), 30, dtype=np.uint8)
            frame[30:150, 50:280] = [40, 200, 90]
            provider.set_frame(frame)
            await asyncio.sleep(0.4)
            await monitor.stop()
            return (
                monitor.stats()["frames_published"],
                bus.payloads(TOPIC_FRAME),
                provider.gui_actions,
            )

        published, payloads, gui_actions = _run(scenario())
        assert published > 0, "the awareness loop must publish frames"
        spoken = [p for p in payloads if p.get("narration")]
        assert spoken, "the eye must speak about a real desktop change"
        assert not gui_actions, "awareness must never dispatch GUI actions"

    def test_narrator_failure_does_not_kill_the_loop(self) -> None:
        async def scenario() -> int:
            provider = _StubProvider()
            bus = _RecordingBus()
            monitor = AwarenessMonitor(provider, "ws", bus=bus, fps=20.0)

            class _Boom:
                @property
                def available(self) -> bool:
                    return True

                def observe_frame(self, *a, **k):
                    raise RuntimeError("narrator exploded")

            monitor._narrator = _Boom()  # type: ignore[assignment]
            monitor._narrator_ready = True
            monitor.start()
            await asyncio.sleep(0.4)
            await monitor.stop()
            return monitor.stats()["frames_published"]

        assert _run(scenario()) > 0, "a narrator failure must not stop awareness"


class TestCortexSemanticsRideAlong:
    def test_semantics_present_when_checkpoint_configured(self, tmp_path) -> None:
        import os

        import torch

        from forge.vision.cortex.config import CortexConfig
        from forge.vision.cortex.model import ArtificialVisualCortex

        model = ArtificialVisualCortex(CortexConfig())
        ckpt = tmp_path / "cortex.pt"
        torch.save(model.state_dict(), ckpt)
        os.environ["SONIC_AVCS_CORTEX_CHECKPOINT"] = str(ckpt)
        try:
            from sonic.computer_use.avcs_perception import AVCSPerception

            perception = AVCSPerception()
            assert perception.ensure_cortex_ready()
            frame = np.full((200, 320, 3), 30, dtype=np.uint8)
            frame[20:120, 40:260] = [230, 230, 230]
            semantics = perception.perceive_semantics_rgb(frame)
            assert "primary_category" in semantics
            assert "category_confidence" in semantics
        finally:
            os.environ.pop("SONIC_AVCS_CORTEX_CHECKPOINT", None)

    def test_semantics_empty_without_checkpoint(self) -> None:
        import os

        os.environ.pop("SONIC_AVCS_CORTEX_CHECKPOINT", None)
        from sonic.computer_use.avcs_perception import AVCSPerception

        perception = AVCSPerception()
        frame = np.full((200, 320, 3), 30, dtype=np.uint8)
        assert perception.perceive_semantics_rgb(frame) == {}
