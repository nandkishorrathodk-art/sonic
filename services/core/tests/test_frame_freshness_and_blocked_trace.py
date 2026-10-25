"""Regression tests for the Boss worker-visibility and frame-freshness fixes.

Three real defects are covered:

1. Frame freshness was measured from ``capture_age_ms`` alone, which is stamped
   at capture time and never grows. A reused/stale observation therefore looked
   fresh to the reasoning gate but stale to the AVCS gate, blocking a GUI worker
   whose pixels were actually fine. Freshness must use true wall-clock age.
2. A blocked pre-execution proposal was recorded as ``TERMINAL_EXEC``, so a
   blocked GUI worker (reasoning-unavailable / gui-contract) appeared in the
   operator UI as a terminal command.
3. The dashboard Agents registry hardcoded the ``default`` session and never
   polled, so Boss workers launched in the active session were invisible.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
)


class _StubComputer:
    workspace_id = "ws-test"

    async def screenshot(self, workspace_id: str):  # pragma: no cover - unused
        return SimpleNamespace(raw_rgb=b"", screenshot_base64="")

    async def status(self, workspace_id: str):  # pragma: no cover - unused
        return SimpleNamespace()


def _screen(*, capture_age_ms, capture_timestamp_ns, pixels=b"\x00\x01\x02"):
    return SimpleNamespace(
        raw_rgb=pixels,
        screenshot_base64="",
        capture_age_ms=capture_age_ms,
        capture_timestamp_ns=capture_timestamp_ns,
        frame_id=1,
    )


class TestFrameTrueAge:
    """``_frame_true_age_ms`` must not trust a frozen capture-time stamp."""

    def test_recent_frame_reports_capture_age(self):
        now_ns = time.time_ns()
        screen = _screen(capture_age_ms=40.0, capture_timestamp_ns=now_ns)
        age = ComputerUseAgent._frame_true_age_ms(screen)
        assert age is not None and age < 500

    def test_stale_frame_is_not_reported_fresh(self):
        """A frame captured a minute ago must not report ~40 ms of age."""
        stale_ns = time.time_ns() - 60_000 * 1_000_000
        screen = _screen(capture_age_ms=40.0, capture_timestamp_ns=stale_ns)
        age = ComputerUseAgent._frame_true_age_ms(screen)
        assert age is not None and age > 30_000

    def test_unmeasurable_age_returns_none(self):
        screen = _screen(capture_age_ms=None, capture_timestamp_ns=None)
        assert ComputerUseAgent._frame_true_age_ms(screen) is None

    def test_negative_or_bool_age_is_rejected(self):
        now_ns = time.time_ns()
        assert ComputerUseAgent._frame_true_age_ms(
            _screen(capture_age_ms=True, capture_timestamp_ns=None)
        ) is None
        # A negative capture-age stamp with no wall-clock reference is unusable.
        assert ComputerUseAgent._frame_true_age_ms(
            _screen(capture_age_ms=-5.0, capture_timestamp_ns=None)
        ) is None

    def test_clock_running_ahead_falls_back_to_capture_stamp(self):
        future_ns = time.time_ns() + 60_000 * 1_000_000
        age = ComputerUseAgent._frame_true_age_ms(
            _screen(capture_age_ms=25.0, capture_timestamp_ns=future_ns)
        )
        assert age == pytest.approx(25.0)


class TestStaleFrameBlocksGuiReasoning:
    """The reasoning gate must reject a stale reused frame consistently."""

    def test_stale_frame_is_not_adopted_as_reasoning_pixels(self):
        agent = ComputerUseAgent(
            computer_provider=_StubComputer(),
            execution_plane="gui",
        )
        stale_ns = time.time_ns() - 60_000 * 1_000_000
        screen = _screen(capture_age_ms=40.0, capture_timestamp_ns=stale_ns)
        fresh = agent._frame_is_fresh_for_reasoning(screen)
        assert fresh is False
        agent._adopt_reasoning_pixels(screen, fresh)
        assert agent._last_screen_has_pixels is False
        assert agent._last_screenshot_b64 == ""

    def test_fresh_frame_is_adopted_as_reasoning_pixels(self):
        agent = ComputerUseAgent(
            computer_provider=_StubComputer(),
            execution_plane="gui",
        )
        screen = _screen(
            capture_age_ms=40.0,
            capture_timestamp_ns=time.time_ns(),
        )
        fresh = agent._frame_is_fresh_for_reasoning(screen)
        assert fresh is True
        agent._adopt_reasoning_pixels(screen, fresh)
        assert agent._last_screen_has_pixels is True

    def test_pixel_less_frame_is_never_fresh(self):
        agent = ComputerUseAgent(
            computer_provider=_StubComputer(),
            execution_plane="gui",
        )
        screen = _screen(
            capture_age_ms=10.0,
            capture_timestamp_ns=time.time_ns(),
            pixels=b"",
        )
        assert agent._frame_is_fresh_for_reasoning(screen) is False


class TestBlockedTraceActionType:
    """A blocked proposal must not be reported as an executed terminal command."""

    @pytest.mark.asyncio
    async def test_blocked_proposal_is_not_labelled_terminal_exec(self):
        agent = ComputerUseAgent(
            computer_provider=_StubComputer(),
            execution_plane="gui",
        )
        trace = await agent.execute_action(
            workspace_id="ws-test",
            action_type=ActionExecutionStatus.BLOCKED,
            target_resource="reasoning-unavailable",
            payload={},
            predicted_outcome="GUI reasoning unavailable",
        )
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126
        assert trace.action_type != ComputerActionType.TERMINAL_EXEC

    def test_blocked_trace_model_still_coerces_status(self):
        trace = ComputerDecisionTrace(
            step_index=1,
            action_type=ActionExecutionStatus.BLOCKED,
            target_resource="blocked-action",
            payload="{}",
            predicted_outcome="blocked",
            actual_observation="Action blocked",
            expected_observation="blocked",
            info_gain=0.0,
            recovery_attempted=False,
            status=ActionExecutionStatus.BLOCKED,
        )
        assert trace.status == ActionExecutionStatus.BLOCKED
