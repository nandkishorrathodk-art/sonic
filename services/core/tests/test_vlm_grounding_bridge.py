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
