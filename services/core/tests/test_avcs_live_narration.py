"""Live AVCS narration: the eye describes what it sees, in words.

These tests exercise the real narrator against real pixel frames -- no mocks of
the narration logic itself.  What is asserted is deliberately narrow:

* the narrator speaks only when something a human would notice changed
* a coloured block that the luma reflex cannot see is still reported
* it never claims to read text, because there is no OCR in this path
* an unverified action is never narrated as success
"""
from __future__ import annotations

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
    LiveNarrator,
    chromatic_change_mask,
    dominant_color_name,
    rgb_tiles,
)
from forge.vision.reflex import ReflexPerception, events_from_states  # noqa: E402
from sonic.computer_use.avcs_narrator import AVCSNarrator  # noqa: E402
from sonic.computer_use.avcs_perception import AVCSPerception  # noqa: E402
from sonic.computer_use.models import ComputerActionType, ScreenObservation  # noqa: E402

W, H = 1920, 1080


def _base_frame() -> np.ndarray:
    frame = np.full((H, W, 3), 45, dtype=np.uint8)
    frame[100:1000, 100:1820] = 120
    return frame


def _screen(frame: np.ndarray, state: dict, active_window: str = "VS Code") -> ScreenObservation:
    screen = ScreenObservation()
    screen.raw_rgb = frame.tobytes()
    screen.width = frame.shape[1]
    screen.height = frame.shape[0]
    screen.avcs_state = state
    screen.desktop_state = "INTERACTIVE"
    screen.active_window = active_window
    screen.frame_id = state.get("source_frame_id")
    screen.capture_age_ms = 10.0
    return screen


class TestColorNaming:
    """Colour naming must be honest about achromatic content."""

    @pytest.mark.parametrize(
        ("rgb", "expected"),
        [
            ([200, 40, 40], "red"),
            ([40, 200, 60], "green"),
            ([50, 80, 220], "blue"),
            ([240, 240, 240], "white"),
            ([10, 10, 10], "black"),
            ([128, 128, 128], "grey"),
        ],
    )
    def test_primary_colours_are_named(self, rgb, expected):
        patch = np.array([[rgb]], dtype=np.uint8)
        assert dominant_color_name(patch) == expected

    def test_empty_patch_is_unknown_not_a_guess(self):
        assert dominant_color_name(np.zeros((0, 0, 3), dtype=np.uint8)) == "unknown"

    def test_grey_is_not_called_a_colour(self):
        patch = np.full((4, 4, 3), 130, dtype=np.uint8)
        assert dominant_color_name(patch) in {"grey", "white"}


class TestChromaticChange:
    """The luma reflex is blind to a saturated box of matching luminance."""

    def test_saturated_box_with_equal_luma_is_still_detected(self):
        prev = rgb_tiles(np.full((H, W, 3), 120, dtype=np.uint8), 16, 24)
        # [200,40,40] has nearly the same luma as mid-grey, so a luma-only
        # change test misses it entirely.
        frame = np.full((H, W, 3), 120, dtype=np.uint8)
        frame[300:520, 700:1250] = [200, 40, 40]
        cur = rgb_tiles(frame, 16, 24)
        mask = chromatic_change_mask(prev, cur)
        assert mask is not None
        assert mask.sum() >= 3, "a saturated box must register as a colour change"

    def test_missing_grid_returns_none_rather_than_guessing(self):
        tiles = rgb_tiles(np.zeros((H, W, 3), dtype=np.uint8), 16, 24)
        assert chromatic_change_mask(None, tiles) is None
        assert chromatic_change_mask(tiles, None) is None

    def test_identical_frames_produce_no_change(self):
        frame = _base_frame()
        tiles = rgb_tiles(frame, 16, 24)
        mask = chromatic_change_mask(tiles, tiles.copy())
        assert mask is not None and not mask.any()


class TestNarrationScenario:
    """The user-facing scenario: app open -> click -> coloured dialog."""

    def _run(self):
        reflex = ReflexPerception()
        narrator = LiveNarrator()
        first = _base_frame()
        state_a = reflex.perceive(first, frame_index=1)
        line_a = narrator.observe(first, state_a, [], active_window="VS Code")

        second = first.copy()
        second[500:560, 900:960] = 255  # a click ripple
        state_b = reflex.perceive(second, frame_index=2)
        line_b = narrator.observe(
            second, state_b, events_from_states(state_a, state_b), active_window="VS Code"
        )

        third = second.copy()
        third[300:520, 700:1250] = [200, 40, 40]  # a red dialog
        state_c = reflex.perceive(third, frame_index=3)
        line_c = narrator.observe(
            third, state_c, events_from_states(state_b, state_c), active_window="VS Code"
        )
        return line_a, line_b, line_c

    def test_first_frame_describes_the_scene(self):
        line_a, _, _ = self._run()
        assert line_a is not None
        assert "VS Code" in line_a.text

    def test_click_produces_a_change_line(self):
        _, line_b, _ = self._run()
        assert line_b is not None
        assert line_b.kind == "change"

    def test_red_dialog_is_reported_as_red_and_alerted(self):
        _, _, line_c = self._run()
        assert line_c is not None, "a coloured dialog must not be silent"
        assert line_c.color == "red"
        assert line_c.kind == "alert"
        assert line_c.urgency == "alert"
        assert "red" in line_c.text.lower()

    def test_narration_never_claims_to_read_text(self):
        line_a, line_b, line_c = self._run()
        for line in (line_a, line_b, line_c):
            if line is None:
                continue
            lowered = line.text.lower()
            # "structure reads as patch" means "looks like", which is honest
            # pixel wording.  What must never appear is a claim to have read
            # the *content* of on-screen text, which this path cannot do.
            for forbidden in (
                "it says",
                "the text says",
                "reading the text",
                "labelled",
                "labeled",
                "titled",
            ):
                assert forbidden not in lowered, f"narration claimed to read text: {line.text}"


class TestSilence:
    """A still screen must not spam the planner."""

    def test_unchanged_frames_are_silent(self):
        reflex = ReflexPerception()
        narrator = LiveNarrator()
        frame = _base_frame()
        narrator.observe(frame, reflex.perceive(frame, frame_index=1), [], active_window="App")
        spoken = 0
        for i in range(2, 8):
            state = reflex.perceive(frame, frame_index=i)
            if narrator.observe(frame, state, [], active_window="App") is not None:
                spoken += 1
        assert spoken == 0, "a static screen must produce no narration"

    def test_silence_ratio_is_high_for_a_static_scene(self):
        reflex = ReflexPerception()
        narrator = LiveNarrator()
        frame = _base_frame()
        for i in range(10):
            narrator.observe(frame, reflex.perceive(frame, frame_index=i + 1), [], active_window="App")
        assert narrator.silence_ratio >= 0.8

    def test_min_speak_interval_suppresses_rapid_repeats(self):
        reflex = ReflexPerception()
        narrator = LiveNarrator(min_speak_interval_s=60.0)
        frame = _base_frame()
        state = reflex.perceive(frame, frame_index=1, timestamp=1000.0)
        assert narrator.observe(frame, state, [], active_window="App") is not None
        changed = frame.copy()
        changed[300:520, 700:1250] = [200, 40, 40]
        state2 = reflex.perceive(changed, frame_index=2, timestamp=1000.5)
        assert narrator.observe(changed, state2, [], active_window="App") is None


class TestActionFeedback:
    """An action's outcome must never be upgraded to success."""

    def test_verified_action_reports_a_response(self):
        narrator = LiveNarrator()
        line = narrator.action_result("GUI_CLICK", "Run button", visible_effect=True)
        assert line.kind == "action"
        assert "responded" in line.text
        assert line.data["verified"] is True

    def test_unverified_action_reports_no_effect_and_alerts(self):
        narrator = LiveNarrator()
        line = narrator.action_result("GUI_CLICK", "Run button", visible_effect=False)
        assert line.kind == "alert"
        assert line.urgency == "alert"
        assert "no visible effect" in line.text
        assert line.data["verified"] is False

    def test_unknown_effect_is_not_reported_as_success(self):
        narrator = LiveNarrator()
        line = narrator.action_result("GUI_CLICK", "Run button", visible_effect=None)
        assert line.data["verified"] is None
        assert "responded" not in line.text
        assert "not verified" in line.text


class TestSonicBridge:
    """The SONIC-side bridge must work off a real AVCSPerception state."""

    def _states(self):
        perception = AVCSPerception()
        first = _base_frame()
        state_a = perception.perceive_rgb(first, source_frame_id=1)
        second = first.copy()
        second[300:520, 700:1250] = [200, 40, 40]
        state_b = perception.perceive_rgb(second, source_frame_id=2)
        return perception, first, state_a, second, state_b

    def test_avcs_state_carries_tile_grid_metadata(self):
        _, _, state_a, _, _ = self._states()
        assert state_a["rows"] > 0 and state_a["cols"] > 0
        assert len(state_a["changed_grid"]) == state_a["rows"]

    def test_bridge_narrates_a_coloured_popup(self):
        _, first, state_a, second, state_b = self._states()
        bridge = AVCSNarrator()
        assert bridge.available
        assert bridge.observe_screen("ws", _screen(first, state_a)) is not None
        line = bridge.observe_screen("ws", _screen(second, state_b))
        assert line is not None
        assert line["color"] == "red"
        assert line["urgency"] == "alert"

    def test_prompt_block_is_empty_without_narration(self):
        bridge = AVCSNarrator()
        assert bridge.format_for_prompt("never-seen") == ""

    def test_prompt_block_renders_alert_marker(self):
        _, first, state_a, second, state_b = self._states()
        bridge = AVCSNarrator()
        bridge.observe_screen("ws", _screen(first, state_a))
        bridge.observe_screen("ws", _screen(second, state_b))
        block = bridge.format_for_prompt("ws")
        assert "!" in block
        assert "red" in block.lower()

    def test_recent_lines_are_bounded(self):
        _, first, state_a, _, _ = self._states()
        bridge = AVCSNarrator()
        for i in range(30):
            bridge.observe_screen("ws", _screen(first, state_a))
        assert len(bridge.recent_lines("ws")) <= 12

    def test_malformed_frame_is_skipped_not_crashed(self):
        bridge = AVCSNarrator()
        screen = ScreenObservation()
        screen.raw_rgb = b"too-short"
        screen.width, screen.height = 1920, 1080
        screen.avcs_state = {"engine": "avcs_reflex", "rows": 16, "cols": 24}
        assert bridge.observe_screen("ws", screen) is None


class TestAgentIntegration:
    """The agent must expose narration and inject it into its prompt."""

    def _agent(self):
        from sonic.computer_use.agent import ComputerUseAgent

        agent = ComputerUseAgent.__new__(ComputerUseAgent)
        agent._active_workspace_id = "ws1"
        agent._avcs_narrator = None
        agent.observe_desktop = True
        return agent

    def test_agent_narrates_a_coloured_popup(self):
        agent = self._agent()
        perception = AVCSPerception()
        first = _base_frame()
        agent._narrate_screen(_screen(first, perception.perceive_rgb(first, source_frame_id=1)))
        second = first.copy()
        second[300:520, 700:1250] = [200, 40, 40]
        block = agent._narrate_screen(
            _screen(second, perception.perceive_rgb(second, source_frame_id=2))
        )
        assert "red" in block.lower()

    def test_agent_action_narration_reflects_verification(self):
        agent = self._agent()
        text = agent._narrate_action_result(
            ComputerActionType.GUI_CLICK, "Run button", False, None, latency_ms=12.0
        )
        assert "no visible effect" in text

    def test_agent_narration_is_empty_when_screen_is_missing(self):
        agent = self._agent()
        assert agent._narrate_screen(None) == ""

    def test_narration_never_breaks_the_agent_when_unavailable(self):
        from sonic.computer_use import avcs_narrator as module

        agent = self._agent()
        original = module.AVCSNarrator

        class Broken:
            @property
            def available(self):
                return True

            def observe_screen(self, *a, **k):
                raise RuntimeError("narrator exploded")

            def format_for_prompt(self, *a, **k):
                return "unreachable"

        module.AVCSNarrator = Broken
        try:
            assert agent._narrate_screen(_screen(_base_frame(), {"engine": "avcs_reflex"})) == ""
        finally:
            module.AVCSNarrator = original

    def test_narrator_is_built_once(self):
        agent = self._agent()
        first = agent._get_avcs_narrator()
        second = agent._get_avcs_narrator()
        assert first is not None and first is second
