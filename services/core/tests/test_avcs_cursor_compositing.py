"""The AVCS frame stream must composite the X11 pointer into its frames.

``mss`` captures the root window, which excludes the hardware cursor: the X
server draws the pointer separately.  An agent that cannot see its own pointer
cannot verify where a click landed, and an operator watching the stream sees no
cursor at all.  These tests pin the compositor contract without needing a live
X server.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STREAM_PATH = _REPO_ROOT / "docker" / "avcs_frame_stream.py"


def _load_stream_module():
    """Import the stream script by path (it is not an installed package).

    ``mss`` is only installed inside the workstation container, so a lightweight
    stub is registered when it is missing; the compositor under test never uses
    it.
    """
    if "mss" not in sys.modules:
        try:
            import mss  # noqa: F401
        except ModuleNotFoundError:
            import types

            stub = types.ModuleType("mss")

            class _MSS:  # pragma: no cover - never instantiated here
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            stub.MSS = _MSS
            sys.modules["mss"] = stub
    spec = importlib.util.spec_from_file_location("avcs_frame_stream", _STREAM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stream_module():
    if not _STREAM_PATH.is_file():
        pytest.skip("avcs_frame_stream.py not present")
    return _load_stream_module()


class _FakeCompositor:
    """Minimal stand-in that reports a fixed cursor sprite."""

    def __init__(self, x, y, cw, ch, pixels):
        self._cursor = (x, y, cw, ch, np.asarray(pixels, dtype=np.uint32))

    def read(self):
        return self._cursor


class TestCursorCompositing:
    def test_cursor_is_blended_into_the_rgb_frame(self, stream_module):
        """A 2x2 fully opaque red cursor paints the expected pixels."""
        rgb = bytearray(8 * 8 * 3)  # black 8x8 frame
        pixels = np.array([0xFFFF0000] * 4, dtype=np.uint32)  # ARGB opaque red
        comp = _FakeCompositor(2, 3, 2, 2, pixels)
        drawn = stream_module.CursorCompositor.stamp(comp, rgb, 8, 8)
        assert drawn is True
        region = np.frombuffer(rgb, dtype=np.uint8).reshape(8, 8, 3)
        assert tuple(int(v) for v in region[3, 2]) == (255, 0, 0)
        assert tuple(int(v) for v in region[4, 3]) == (255, 0, 0)
        # A pixel outside the cursor keeps its original value.
        assert tuple(int(v) for v in region[0, 0]) == (0, 0, 0)

    def test_transparent_cursor_pixels_are_left_alone(self, stream_module):
        rgb = bytearray(8 * 8 * 3)
        pixels = np.array([0x00FFFFFF, 0xFFFFFFFF, 0x00FFFFFF, 0xFFFFFFFF], dtype=np.uint32)
        comp = _FakeCompositor(0, 0, 2, 2, pixels)
        assert stream_module.CursorCompositor.stamp(comp, rgb, 8, 8) is True
        region = np.frombuffer(rgb, dtype=np.uint8).reshape(8, 8, 3)
        assert tuple(int(v) for v in region[0, 0]) == (0, 0, 0)   # alpha 0 -> untouched
        assert tuple(int(v) for v in region[0, 1]) == (255, 255, 255)  # opaque white

    def test_partial_alpha_blends_toward_the_sprite(self, stream_module):
        rgb = bytearray(8 * 8 * 3)  # black
        # alpha 128 white over black -> about half brightness
        pixels = np.array([0x80FFFFFF], dtype=np.uint32)
        comp = _FakeCompositor(0, 0, 1, 1, pixels)
        stream_module.CursorCompositor.stamp(comp, rgb, 8, 8)
        region = np.frombuffer(rgb, dtype=np.uint8).reshape(8, 8, 3)
        assert 120 <= int(region[0, 0, 0]) <= 136

    def test_cursor_clipped_at_the_frame_edge_is_not_wrapped(self, stream_module):
        """A cursor partly off-screen must clip, never wrap to the other side."""
        rgb = bytearray(4 * 4 * 3)
        pixels = np.array([0xFFFFFFFF] * 4, dtype=np.uint32)
        comp = _FakeCompositor(-1, -1, 2, 2, pixels)
        assert stream_module.CursorCompositor.stamp(comp, rgb, 4, 4) is True
        region = np.frombuffer(rgb, dtype=np.uint8).reshape(4, 4, 3)
        # Only the bottom-right sprite pixel lands at (0,0); nothing wraps.
        assert tuple(int(v) for v in region[0, 0]) == (255, 255, 255)
        assert tuple(int(v) for v in region[3, 3]) == (0, 0, 0)

    def test_fully_offscreen_cursor_draws_nothing(self, stream_module):
        rgb = bytearray(4 * 4 * 3)
        pixels = np.array([0xFFFFFFFF] * 4, dtype=np.uint32)
        comp = _FakeCompositor(50, 50, 2, 2, pixels)
        assert stream_module.CursorCompositor.stamp(comp, rgb, 4, 4) is False
        assert bytes(rgb) == bytes(4 * 4 * 3)

    def test_all_transparent_cursor_reports_not_drawn(self, stream_module):
        rgb = bytearray(4 * 4 * 3)
        pixels = np.array([0x00FFFFFF] * 4, dtype=np.uint32)
        comp = _FakeCompositor(0, 0, 2, 2, pixels)
        assert stream_module.CursorCompositor.stamp(comp, rgb, 4, 4) is False

    def test_read_parses_32bit_argb_from_64bit_slots(self, stream_module, monkeypatch):
        """XFixes stores CARD32 ARGB values in C ``long`` (8-byte) slots."""
        # Two slots: 0x00000000_AARRGGBB little-endian each.
        raw = struct.pack("<Q", 0xFF112233) + struct.pack("<Q", 0x80445566)

        class _Cursor:
            x = 5
            y = 7
            width = 2
            height = 1
            pixels = 0xDEADBEEF  # dummy pointer; string_at is stubbed below

        class _Image:
            contents = _Cursor()

        class _Fixes:
            @staticmethod
            def XFixesGetCursorImage(_display):
                return _Image()

        class _Lib:
            @staticmethod
            def string_at(ptr, size):
                assert size == 2 * 8
                return raw

        comp = object.__new__(stream_module.CursorCompositor)
        comp._available = True
        comp._display = 1
        comp._fixes = _Fixes()
        monkeypatch.setattr(stream_module.ctypes, "string_at", _Lib.string_at)
        got = comp.read()
        assert got is not None
        x, y, w, h, values = got
        assert (x, y, w, h) == (5, 7, 2, 1)
        assert int(values[0]) == 0xFF112233
        assert int(values[1]) == 0x80445566

    def test_unavailable_xfixes_fails_soft(self, stream_module, monkeypatch):
        """Without XFixes the compositor must disable itself, not crash."""
        monkeypatch.setattr(stream_module.ctypes.util, "find_library", lambda name: None)
        comp = stream_module.CursorCompositor()
        assert comp.available is False
        assert comp.error
        assert comp.read() is None
        rgb = bytearray(4 * 4 * 3)
        assert comp.stamp(rgb, 4, 4) is False

    def test_stream_cli_exposes_a_no_cursor_escape_hatch(self):
        source = _STREAM_PATH.read_text(encoding="utf-8")
        assert "--no-cursor" in source
        # The compositor must actually be wired into the capture loop.
        assert "compositor.stamp(" in source
