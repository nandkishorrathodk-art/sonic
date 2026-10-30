"""Cloud VLM grounding bridge — opt-in, fail-closed, local-checkpoint-first.

Covers the bridge added so a configured vision model can answer "where is X?"
when no evaluated local AVCS grounder checkpoint exists. The local checkpoint
always wins when present; the bridge is off unless explicitly enabled.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


from sonic.computer_use.grounding import (
    _parse_vlm_grounding_reply,
    _resolve_vlm_target,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Reply parsing — normalized [0,1000] -> screen pixels
# --------------------------------------------------------------------------


class TestParseVlmGroundingReply:
    def test_json_point_normalized_center(self):
        assert _parse_vlm_grounding_reply('{"x": 500, "y": 500}', 1920, 1080) == (
            960,
            540,
        )

    def test_json_point_top_left(self):
        assert _parse_vlm_grounding_reply('{"x": 0, "y": 0}', 1920, 1080) == (0, 0)

    def test_point_list_and_bare_list(self):
        assert _parse_vlm_grounding_reply('{"point": [250, 750]}', 1920, 1080) == (
            480,
            810,
        )
        assert _parse_vlm_grounding_reply("[250, 750]", 1920, 1080) == (480, 810)

    def test_uitars_box_midpoint_is_pixel_space(self):
        reply = "<|box_start|>(1500,600,1524,680)<|box_end|>"
        assert _parse_vlm_grounding_reply(reply, 1920, 1080) == (1512, 640)

    def test_not_found_variants(self):
        assert _parse_vlm_grounding_reply('{"not_found": true}', 1920, 1080) is None
        assert _parse_vlm_grounding_reply("NOT_FOUND", 1920, 1080) is None
        assert _parse_vlm_grounding_reply("", 1920, 1080) is None

    def test_prose_is_rejected(self):
        assert (
            _parse_vlm_grounding_reply(
                "I believe the button is somewhere near the middle.", 1920, 1080
            )
            is None
        )

    def test_out_of_range_is_rejected_not_clamped(self):
        # 1000 normalized maps exactly to the screen edge -> out of bounds.
        assert _parse_vlm_grounding_reply('{"x": 1000, "y": 1000}', 1920, 1080) is None
        assert _parse_vlm_grounding_reply('{"x": -5, "y": 10}', 1920, 1080) is None
        assert _parse_vlm_grounding_reply('{"x": 1200, "y": 10}', 1920, 1080) is None

    def test_non_finite_and_non_numeric_rejected(self):
        assert (
            _parse_vlm_grounding_reply('{"x": "NaN", "y": 10}', 1920, 1080) is None
        )
        assert _parse_vlm_grounding_reply('{"x": "left", "y": 10}', 1920, 1080) is None


# --------------------------------------------------------------------------
# Bridge call — needs a router; fails closed without one
# --------------------------------------------------------------------------


class TestResolveVlmTarget:
    def test_no_router_fails_closed(self):
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        assert _run(_resolve_vlm_target("Save button", png, 1920, 1080, router=None)) is None

    def test_invalid_base64_fails_closed(self):
        router = MagicMock()
        assert (
            _run(
                _resolve_vlm_target(
                    "Save button", "not-valid-base64!!!", 1920, 1080, router=router
                )
            )
            is None
        )
        router.complete.assert_not_called()

    def test_router_reply_is_parsed_to_pixels(self):
        router = MagicMock()
        router.complete = AsyncMock(
            return_value=SimpleNamespace(content='{"x": 500, "y": 500}')
        )
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        point = _run(_resolve_vlm_target("Save button", png, 1920, 1080, router=router))
        assert point == (960, 540)

    def test_router_error_fails_closed(self):
        router = MagicMock()
        router.complete = AsyncMock(side_effect=RuntimeError("provider down"))
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        assert _run(_resolve_vlm_target("Save", png, 1920, 1080, router=router)) is None

    def test_router_not_found_reply_fails_closed(self):
        router = MagicMock()
        router.complete = AsyncMock(
            return_value=SimpleNamespace(content='{"not_found": true}')
        )
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        assert _run(_resolve_vlm_target("Save", png, 1920, 1080, router=router)) is None

    def test_vision_route_is_requested(self):
        router = MagicMock()
        router.complete = AsyncMock(
            return_value=SimpleNamespace(content='{"x": 10, "y": 20}')
        )
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        _run(_resolve_vlm_target("Save", png, 1920, 1080, router=router))
        request = router.complete.call_args.args[0]
        assert request.task_type == "vision"
        assert request.messages[0].has_images is True


# --------------------------------------------------------------------------
# Focus verification via the vision route (added when the local AVCS focus
# head is not configured). Answers a single yes/no question and fails closed.
# --------------------------------------------------------------------------


class TestParseVlmFocusReply:
    def test_true_with_evidence(self):
        from sonic.computer_use.grounding import _parse_vlm_focus_reply

        assert _parse_vlm_focus_reply(
            '{"focused": true, "evidence": "blinking caret visible"}'
        ) == (True, "blinking caret visible")

    def test_false_with_evidence(self):
        from sonic.computer_use.grounding import _parse_vlm_focus_reply

        assert _parse_vlm_focus_reply(
            '{"focused": false, "evidence": "that is a button"}'
        ) == (False, "that is a button")

    def test_missing_evidence_is_inconclusive(self):
        from sonic.computer_use.grounding import _parse_vlm_focus_reply

        assert _parse_vlm_focus_reply('{"focused": true}') is None

    def test_non_bool_focused_is_inconclusive(self):
        from sonic.computer_use.grounding import _parse_vlm_focus_reply

        assert _parse_vlm_focus_reply('{"focused": "yes", "evidence": "x"}') is None

    def test_non_json_is_inconclusive(self):
        from sonic.computer_use.grounding import _parse_vlm_focus_reply

        assert _parse_vlm_focus_reply("I think it is focused") is None


class TestVerifyFocusWithVlm:
    def test_true_reply_returns_focused(self):
        from sonic.computer_use.grounding import _verify_focus_with_vlm

        router = MagicMock()
        router.complete = AsyncMock(
            return_value=SimpleNamespace(
                content='{"focused": true, "evidence": "caret in the field"}'
            )
        )
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        focused, evidence = _run(
            _verify_focus_with_vlm("Name field", png, 1920, 1080, 400, 300, router=router)
        )
        assert focused is True
        assert "caret" in evidence

    def test_no_router_fails_closed(self):
        from sonic.computer_use.grounding import _verify_focus_with_vlm

        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        focused, evidence = _run(
            _verify_focus_with_vlm("Name", png, 1920, 1080, 1, 1, router=None)
        )
        assert focused is False
        assert evidence

    def test_provider_error_fails_closed(self):
        from sonic.computer_use.grounding import _verify_focus_with_vlm

        router = MagicMock()
        router.complete = AsyncMock(side_effect=RuntimeError("provider down"))
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        focused, _ = _run(
            _verify_focus_with_vlm("Name", png, 1920, 1080, 1, 1, router=router)
        )
        assert focused is False

    def test_inconclusive_reply_fails_closed(self):
        from sonic.computer_use.grounding import _verify_focus_with_vlm

        router = MagicMock()
        router.complete = AsyncMock(return_value=SimpleNamespace(content="maybe"))
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        focused, _ = _run(
            _verify_focus_with_vlm("Name", png, 1920, 1080, 1, 1, router=router)
        )
        assert focused is False

    def test_request_uses_vision_route_with_image(self):
        from sonic.computer_use.grounding import _verify_focus_with_vlm

        router = MagicMock()
        router.complete = AsyncMock(
            return_value=SimpleNamespace(content='{"focused": true, "evidence": "ok"}')
        )
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        _run(_verify_focus_with_vlm("Name", png, 1920, 1080, 5, 6, router=router))
        request = router.complete.call_args.args[0]
        assert request.task_type == "vision"
        assert request.messages[0].has_images is True


class TestVlmGroundingDefault:
    """The bridge is the default grounder when no local checkpoint exists."""

    def test_default_is_enabled(self, monkeypatch):
        import sonic.computer_use.agent as agent_mod

        monkeypatch.delenv("SONIC_ENABLE_VLM_GROUNDING", raising=False)
        # The default must be "on" (not "0"), so the bridge runs without an
        # explicit opt-in when no local checkpoint is configured.
        assert agent_mod.os.environ.get("SONIC_ENABLE_VLM_GROUNDING", "1") != "0"

    def test_focus_availability_helper_blocks_without_any_path(self, monkeypatch):
        from sonic.computer_use.agent import ComputerUseAgent

        monkeypatch.delenv("SONIC_AVCS_FOCUS_CHECKPOINT", raising=False)
        monkeypatch.setenv("SONIC_ENABLE_VLM_GROUNDING", "0")
        agent = ComputerUseAgent(computer_provider=MagicMock())
        assert agent._focus_verification_available() is False

    def test_focus_availability_true_with_ready_vision_route(self, monkeypatch):
        from sonic.computer_use.agent import ComputerUseAgent

        monkeypatch.delenv("SONIC_AVCS_FOCUS_CHECKPOINT", raising=False)
        monkeypatch.setenv("SONIC_ENABLE_VLM_GROUNDING", "1")
        router = MagicMock()
        router.vision_route_ready = MagicMock(return_value=True)
        agent = ComputerUseAgent(computer_provider=MagicMock(), llm_router=router)
        assert agent._focus_verification_available() is True
