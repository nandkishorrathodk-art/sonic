"""The eye's two-way loop: foveal focus, ambient commentary, and interruption.

These tests cover the three additions that turn AVCS from a sensor the agent
*polls* into a colleague that *speaks*:

* **foveal focus** -- a narration about a region carries a real pixel rectangle,
  and cropping to it lands on the thing that was described
* **ambient commentary** -- the continuous awareness loop narrates on its own,
  and the brain hears those lines on its next turn (once, not repeatedly)
* **cross-identity delivery** -- the eye and the brain name the same desktop
  differently, so the shared channel must be keyed on a value both can derive

Frames are real numpy arrays through the real reflex and real narrator; nothing
here mocks the narration logic.  What is deliberately *not* asserted: that AVCS
can read text.  It cannot -- there is no OCR in this path -- so no test claims
it does.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AVCS_SRC = REPO_ROOT / "packages" / "avcs"
CORE_SRC = REPO_ROOT / "services" / "core"
for candidate in (str(AVCS_SRC), str(CORE_SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from forge.vision.narrator import (  # noqa: E402
    FOCUS_MIN_PIXELS,
    LiveNarrator,
    crop_focus,
    focus_rect_pixels,
)
from forge.vision.reflex import ReflexPerception, events_from_states  # noqa: E402
from sonic.computer_use import avcs_narrator as narrator_module  # noqa: E402
from sonic.computer_use.awareness import (  # noqa: E402
    TOPIC_NARRATION,
    AwarenessMonitor,
)

W, H = 640, 480


def _grey_desktop() -> np.ndarray:
    return np.full((H, W, 3), 160, dtype=np.uint8)


def _with_red_dialog(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[180:300, 220:420] = (200, 40, 40)
    return out


def _narrate(frame: np.ndarray, previous: np.ndarray | None, window: str = "Desktop"):
    """Run one frame through the real reflex + narrator; return the Narration."""
    reflex = ReflexPerception()
    narrator = LiveNarrator(min_speak_interval_s=0.0)
    prev_state = reflex.perceive(previous) if previous is not None else None
    state = reflex.perceive(frame)
    events = events_from_states(prev_state, state) if prev_state is not None else []
    return narrator.observe(frame, state, events, active_window=window), state


class TestFovealFocus:
    """A line about a region must carry a real, croppable pixel rectangle."""

    def test_popup_line_carries_a_focus_rectangle(self):
        line, _ = _narrate(_with_red_dialog(_grey_desktop()), _grey_desktop())
        assert line is not None
        assert line.focus is not None
        for key in ("x", "y", "w", "h"):
            assert key in line.focus
        assert line.focus["w"] > 0 and line.focus["h"] > 0

    def test_focus_rectangle_covers_the_thing_that_was_described(self):
        """The crop must actually contain the dialog, not merely exist."""
        frame = _with_red_dialog(_grey_desktop())
        line, _ = _narrate(frame, _grey_desktop())
        crop = crop_focus(frame, line.focus)
        assert crop is not None
        # The dialog is a red block; a correct crop contains real red pixels.
        assert crop[:, :, 0].max() > 150 and crop[:, :, 1].min() < 100

    def test_focus_rectangle_is_inside_the_frame(self):
        frame = _with_red_dialog(_grey_desktop())
        line, state = _narrate(frame, _grey_desktop())
        x, y = line.focus["x"], line.focus["y"]
        assert 0 <= x < state.width
        assert 0 <= y < state.height
        assert x + line.focus["w"] <= state.width
        assert y + line.focus["h"] <= state.height

    def test_quiet_line_has_no_focus(self):
        """A still screen has nothing to point at."""
        line, _ = _narrate(_grey_desktop(), None)
        assert line is not None
        assert line.focus is None

    def test_focus_is_serialised_only_when_present(self):
        line, _ = _narrate(_with_red_dialog(_grey_desktop()), _grey_desktop())
        assert "focus" in line.to_dict()
        quiet, _ = _narrate(_grey_desktop(), None)
        assert "focus" not in quiet.to_dict()

    def test_tiny_block_is_grown_to_a_usable_crop(self):
        """A one-tile change must not produce a sliver too small to read."""
        reflex = ReflexPerception()
        narrator = LiveNarrator(min_speak_interval_s=0.0)
        base = _grey_desktop()
        tick = base.copy()
        tick[240, 320] = (255, 0, 0)
        st0 = reflex.perceive(base)
        st1 = reflex.perceive(tick)
        line = narrator.observe(tick, st1, events_from_states(st0, st1))
        if line is not None and line.focus is not None:
            assert line.focus["w"] >= FOCUS_MIN_PIXELS
            assert line.focus["h"] >= FOCUS_MIN_PIXELS

    def test_focus_rect_helpers_reject_bad_input(self):
        from forge.vision.scene import SceneState

        blank = SceneState(
            frame_index=0,
            timestamp=0.0,
            width=W,
            height=H,
            rows=24,
            cols=32,
            luma_grid=np.zeros((24, 32), dtype=np.float32),
            edge_grid=np.zeros((24, 32), dtype=np.float32),
            salience_grid=np.zeros((24, 32), dtype=np.float32),
            motion_grid=np.zeros((24, 32), dtype=np.float32),
            changed_grid=np.zeros((24, 32), dtype=np.float32),
            brightness=0.5,
            contrast=0.1,
            change_fraction=0.0,
            global_motion=0.0,
            peak_motion=0.0,
            novelty=0.0,
            scene_hash="x",
        )
        assert focus_rect_pixels(blank, None) is None
        assert crop_focus(_grey_desktop(), None) is None
        assert crop_focus(_grey_desktop(), {"x": 0, "y": 0, "w": 0, "h": 0}) is None
        assert crop_focus(None, {"x": 0, "y": 0, "w": 10, "h": 10}) is None


class _StubScreen:
    """A screenshot whose frame id increases, as the real provider's does."""

    _counter = 0

    def __init__(self, frame: np.ndarray) -> None:
        type(self)._counter += 1
        self.raw_rgb = frame.tobytes()
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.frame_id = type(self)._counter
        self.desktop_state = "INTERACTIVE"
        self.screenshot_base64 = ""
        self.active_window = "Desktop"


class _StubProvider:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0

    async def screenshot(self, workspace_id: str) -> _StubScreen:
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return _StubScreen(frame)


class _RecordingBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


@pytest.fixture(autouse=True)
def _clean_ambient():
    for key in ("default", "acme", "home-acme"):
        narrator_module.drain_ambient_narration(key)
    yield
    for key in ("default", "acme", "home-acme"):
        narrator_module.drain_ambient_narration(key)


class TestAmbientCommentary:
    """The eye speaks unprompted; the brain hears it on its next turn."""

    def _run(self, monitor: AwarenessMonitor, n: int) -> None:
        async def go() -> None:
            for _ in range(n):
                await monitor._sample_once()

        asyncio.run(go())

    def test_awareness_narrates_and_publishes_a_narration_topic(self):
        base = _grey_desktop()
        popup = _with_red_dialog(base)
        provider = _StubProvider([base, popup, popup])
        bus = _RecordingBus()
        monitor = AwarenessMonitor(provider, "home-acme", bus=bus, fps=10, ambient_key="acme")
        self._run(monitor, 3)
        topics = [t for t, _ in bus.published]
        assert TOPIC_NARRATION in topics
        lines = [p["narration"] for t, p in bus.published if t == TOPIC_NARRATION]
        assert any("popped up" in line for line in lines)

    def test_a_still_screen_does_not_repeat_itself(self):
        """Two identical frames must not produce two identical announcements."""
        base = _grey_desktop()
        provider = _StubProvider([base, base, base, base])
        bus = _RecordingBus()
        monitor = AwarenessMonitor(provider, "home-acme", bus=bus, fps=10, ambient_key="acme")
        self._run(monitor, 4)
        lines = [p["narration"] for t, p in bus.published if t == TOPIC_NARRATION]
        assert len(lines) == len(set(lines))

    def test_ambient_channel_delivers_lines_once(self):
        base = _grey_desktop()
        popup = _with_red_dialog(base)
        provider = _StubProvider([base, popup, popup])
        monitor = AwarenessMonitor(provider, "home-acme", fps=10, ambient_key="acme")
        self._run(monitor, 3)
        first = narrator_module.drain_ambient_narration("acme")
        assert first, "the eye should have recorded something"
        assert narrator_module.drain_ambient_narration("acme") == []

    def test_ambient_channel_is_cross_identity(self):
        """The eye names the workspace 'home-acme'; the brain only knows 'acme'."""
        base = _grey_desktop()
        popup = _with_red_dialog(base)
        provider = _StubProvider([base, popup, popup])
        monitor = AwarenessMonitor(provider, "home-acme", fps=10, ambient_key="acme")
        self._run(monitor, 3)
        assert narrator_module.drain_ambient_narration("acme")
        # The workspace id alone must NOT have been used as the key.
        assert narrator_module.drain_ambient_narration("home-acme") == []

    def test_narration_never_claims_to_read_text(self):
        """Honesty guard: no OCR exists, so no line may quote on-screen text."""
        base = _grey_desktop()
        popup = _with_red_dialog(base)
        provider = _StubProvider([base, popup, popup])
        bus = _RecordingBus()
        monitor = AwarenessMonitor(provider, "home-acme", bus=bus, fps=10, ambient_key="acme")
        self._run(monitor, 3)
        for _, payload in bus.published:
            line = str(payload.get("narration") or "")
            for forbidden in ("says", "reads", "text:", "reads:"):
                assert forbidden not in line.lower()

    def test_monitor_never_dispatches_an_action(self):
        """Awareness observes only: the provider exposes no action method."""
        base = _grey_desktop()
        provider = _StubProvider([base, base])
        assert not hasattr(provider, "gui_action")
        monitor = AwarenessMonitor(provider, "home-acme", fps=10, ambient_key="acme")
        self._run(monitor, 2)
        assert not hasattr(provider, "gui_action")

    def test_ambient_channel_is_bounded(self):
        for i in range(50):
            narrator_module.record_ambient_narration("acme", {"text": f"line {i}", "urgency": "info"})
        lines = narrator_module.peek_ambient_narration("acme")
        assert len(lines) <= narrator_module._MAX_AMBIENT_LINES

    def test_ambient_record_ignores_empty_text(self):
        narrator_module.record_ambient_narration("acme", {"text": ""})
        narrator_module.record_ambient_narration("acme", {})
        narrator_module.record_ambient_narration("acme", "not a dict")
        assert narrator_module.peek_ambient_narration("acme") == []


class TestAmbientInjectionIntoReasoning:
    """The brain must actually surface what the eye said."""

    def _agent(self, tenant: str = "acme"):
        from sonic.computer_use.agent import ComputerUseAgent

        agent = ComputerUseAgent.__new__(ComputerUseAgent)
        agent.tenant_id = tenant
        agent._active_workspace_id = "home-acme"
        return agent

    def test_agent_drains_the_eye_lines_into_a_prompt_block(self):
        narrator_module.record_ambient_narration(
            "acme", {"text": "A red box appeared.", "urgency": "alert"}
        )
        block = self._agent()._ambient_narration_block()
        assert "A red box appeared." in block
        # The alert marker is carried through so the model can weigh it.
        assert "!" in block

    def test_block_is_empty_when_the_eye_said_nothing(self):
        assert self._agent()._ambient_narration_block() == ""

    def test_block_is_drained_not_repeated(self):
        narrator_module.record_ambient_narration("acme", {"text": "One thing.", "urgency": "notice"})
        agent = self._agent()
        assert "One thing." in agent._ambient_narration_block()
        assert agent._ambient_narration_block() == ""

    def test_agent_falls_back_to_workspace_when_tenant_missing(self):
        from sonic.computer_use.agent import ComputerUseAgent

        agent = ComputerUseAgent.__new__(ComputerUseAgent)
        agent.tenant_id = ""
        agent._active_workspace_id = "home-acme"
        narrator_module.record_ambient_narration("home-acme", {"text": "Fallback line."})
        assert "Fallback line." in agent._ambient_narration_block()
