"""End-to-end: what the eye notices reaches the prompt the brain actually sends.

The unit tests in ``test_avcs_foveal_and_ambient_loop.py`` prove the pieces
work in isolation.  These tests prove the *whole* chain is connected:

    real pixels -> awareness narration -> ambient channel
                -> ComputerUseAgent._build_reasoning_context -> prompt text

They drive the real ``_build_reasoning_context`` (not the block helper), so a
future refactor that drops the injection from the prompt would fail here even
though the helper still worked.  Both prompt paths are covered: the general
reasoning context and the GUI-only context, because the GUI worker is the one
most likely to be looking at a popup.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AVCS_SRC = REPO_ROOT / "packages" / "avcs"
CORE_SRC = REPO_ROOT / "services" / "core"
for candidate in (str(AVCS_SRC), str(CORE_SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from sonic.computer.models import ScreenObservation  # noqa: E402
from sonic.computer_use import avcs_narrator as narrator_module  # noqa: E402
from sonic.computer_use.agent import ComputerUseAgent  # noqa: E402
from sonic.computer_use.awareness import AwarenessMonitor  # noqa: E402
from sonic.computer_use.models import ComputerWorldObservation  # noqa: E402

W, H = 640, 480


def _grey() -> np.ndarray:
    return np.full((H, W, 3), 160, dtype=np.uint8)


def _with_red_dialog(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[180:300, 220:420] = (200, 40, 40)
    return out


class _StubScreen:
    _n = 0

    def __init__(self, frame: np.ndarray) -> None:
        type(self)._n += 1
        self.raw_rgb = frame.tobytes()
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.frame_id = type(self)._n
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


def _agent(tenant: str = "acme", gui_only: bool = False) -> ComputerUseAgent:
    provider = AsyncMock()
    provider.screenshot.return_value = ScreenObservation(width=W, height=H)
    provider.status.return_value = MagicMock(
        active_application="Desktop",
        open_applications=["Desktop"],
        running_processes=["xvfb"],
    )
    provider.list_files.return_value = []
    provider.git_action.return_value = MagicMock(branch="main", is_clean=True)
    agent = ComputerUseAgent(computer_provider=provider, tenant_id=tenant)
    agent._active_workspace_id = "home-acme"
    if gui_only:
        agent.gui_only = True
        agent.execution_plane = "gui"
    return agent


@pytest.fixture(autouse=True)
def _clean():
    for key in ("default", "acme", "home-acme"):
        narrator_module.drain_ambient_narration(key)
    yield
    for key in ("default", "acme", "home-acme"):
        narrator_module.drain_ambient_narration(key)


def _run_awareness(monitor: AwarenessMonitor, n: int) -> None:
    async def go() -> None:
        for _ in range(n):
            await monitor._sample_once()

    asyncio.run(go())


class TestEyeToBrainEndToEnd:
    """The eye's own commentary must land in the prompt the brain sends."""

    def test_eye_noticing_a_popup_reaches_the_general_prompt(self):
        base = _grey()
        popup = _with_red_dialog(base)
        monitor = AwarenessMonitor(
            _StubProvider([base, popup, popup]), "home-acme", fps=10, ambient_key="acme"
        )
        _run_awareness(monitor, 3)

        obs = ComputerWorldObservation(
            visible_text="Desktop",
            terminal_output="",
            working_directory="/home/sonic/workspace",
        )
        _, user_prompt = _agent()._build_reasoning_context(
            goal="Complete the operator task",
            observation=obs,
            step_index=1,
            primary_file="",
            test_file="",
        )
        assert "popped up" in user_prompt

    def test_gui_only_prompt_does_not_depend_on_ambient_avcs_narration(self):
        """GUI decisions use the fresh screenshot, not ambient narration."""
        base = _grey()
        popup = _with_red_dialog(base)
        monitor = AwarenessMonitor(
            _StubProvider([base, popup, popup]), "home-acme", fps=10, ambient_key="acme"
        )
        _run_awareness(monitor, 3)

        agent = _agent(gui_only=True)
        obs = ComputerWorldObservation(
            visible_text="Desktop",
            terminal_output="",
            working_directory="/home/sonic/workspace",
        )
        _, user_prompt = agent._build_reasoning_context(
            goal="Complete the operator task",
            observation=obs,
            step_index=1,
            primary_file="",
            test_file="",
        )
        assert "AVCS unsolicited observations" not in user_prompt

    def test_each_observation_is_delivered_once_across_steps(self):
        """A long-running alert must not be repeated in every later prompt."""
        base = _grey()
        popup = _with_red_dialog(base)
        monitor = AwarenessMonitor(
            _StubProvider([base, popup, popup]), "home-acme", fps=10, ambient_key="acme"
        )
        _run_awareness(monitor, 3)

        agent = _agent()
        obs = ComputerWorldObservation(
            visible_text="Desktop",
            terminal_output="",
            working_directory="/home/sonic/workspace",
        )
        _, first = agent._build_reasoning_context(
            goal="g", observation=obs, step_index=1, primary_file="", test_file=""
        )
        _, second = agent._build_reasoning_context(
            goal="g", observation=obs, step_index=2, primary_file="", test_file=""
        )
        assert "popped up" in first
        assert "popped up" not in second

    def test_still_screen_produces_no_alert_and_does_not_repeat(self):
        """A still screen may say one calm baseline line, never an alert.

        The eye narrating "the screen is essentially still" once is a useful
        baseline, not noise.  What must never happen is a false alert, or the
        same calm line repeated into the prompt on every step.
        """
        base = _grey()
        monitor = AwarenessMonitor(
            _StubProvider([base, base, base, base]), "home-acme", fps=10, ambient_key="acme"
        )
        _run_awareness(monitor, 4)

        lines = narrator_module.peek_ambient_narration("acme")
        urgencies = {str(line.get("urgency") or "") for line in lines}
        assert "alert" not in urgencies
        texts = [str(line.get("text") or "") for line in lines]
        assert len(texts) == len(set(texts)), "a still screen must not repeat itself"

        agent = _agent()
        obs = ComputerWorldObservation(
            visible_text="Desktop",
            terminal_output="",
            working_directory="/home/sonic/workspace",
        )
        _, first = agent._build_reasoning_context(
            goal="g", observation=obs, step_index=1, primary_file="", test_file=""
        )
        _, second = agent._build_reasoning_context(
            goal="g", observation=obs, step_index=2, primary_file="", test_file=""
        )
        # Whatever the eye said, it is delivered at most once: the ambient
        # section appears in the first prompt and is drained before the second.
        assert "unsolicited observations" in first.lower()
        assert "unsolicited observations" not in second.lower()


class TestCrossDesktopIsolation:
    """Two desktops must never share perception state.

    The engine holds temporal state (previous frame, frame-id watermark,
    prediction memory) that is only meaningful within one capture stream.  A
    shared engine blended two tenants' frames, which both invented phantom
    motion and tripped the monotonic frame-id guard, killing the second
    desktop's awareness loop outright.
    """

    def test_one_desktops_popup_does_not_leak_into_anothers_commentary(self):
        from sonic.computer_use.awareness import reset_awareness_monitors

        reset_awareness_monitors()
        for key in ("alpha", "beta"):
            narrator_module.drain_ambient_narration(key)
        base = _grey()
        popup = _with_red_dialog(base)
        monitor_a = AwarenessMonitor(
            _StubProvider([base, popup, popup]), "home-alpha", fps=10, ambient_key="alpha"
        )
        monitor_b = AwarenessMonitor(
            _StubProvider([base, base, base]), "home-beta", fps=10, ambient_key="beta"
        )

        async def go() -> None:
            for _ in range(3):
                await monitor_a._sample_once()
            for _ in range(3):
                await monitor_b._sample_once()

        asyncio.run(go())
        beta_lines = narrator_module.peek_ambient_narration("beta")
        assert beta_lines, "the still desktop should still report its own state"
        for line in beta_lines:
            assert line["urgency"] != "alert"
            assert "popped up" not in line["text"].lower()
        assert monitor_b._capture_errors == 0, (
            "the second desktop's loop must not be killed by the first's frames"
        )

    def test_each_monitor_owns_its_perception_engine(self):
        base = _grey()
        monitor_a = AwarenessMonitor(_StubProvider([base]), "ws-a", fps=10, ambient_key="a")
        monitor_b = AwarenessMonitor(_StubProvider([base]), "ws-b", fps=10, ambient_key="b")
        engine_a = monitor_a._perception_engine()
        engine_b = monitor_b._perception_engine()
        assert engine_a is not engine_b
        assert monitor_a._perception_engine() is engine_a
