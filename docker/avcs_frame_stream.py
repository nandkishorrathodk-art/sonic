from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import select
import struct
import sys
import time

import mss
import numpy as np

_HEADER = struct.Struct("!4sQQII")
_MAGIC = b"AVCS"
_MAX_RAW_FRAME_BYTES = 64 * 1024 * 1024
_MAX_FRAME_PIXELS = _MAX_RAW_FRAME_BYTES // 3


class _XFixesCursorImage(ctypes.Structure):
    """XFixesCursorImage layout (Xfixes.h)."""

    _fields_ = [
        ("x", ctypes.c_short),
        ("y", ctypes.c_short),
        ("width", ctypes.c_ushort),
        ("height", ctypes.c_ushort),
        ("xhot", ctypes.c_ushort),
        ("yhot", ctypes.c_ushort),
        ("cursor_serial", ctypes.c_ulong),
        ("pixels", ctypes.POINTER(ctypes.c_ulong)),
        ("atom", ctypes.c_ulong),
        ("name", ctypes.c_char_p),
    ]


class CursorCompositor:
    """Stamps the live X11 pointer onto captured frames.

    ``mss`` reads the root window, which does NOT include the hardware cursor
    (the X server composites it separately).  A computer-use agent that cannot
    see its own pointer is blind to the most important thing it controls: it
    cannot confirm where a click landed, and a human watching the stream sees
    no cursor at all.  This reads the cursor sprite through XFixes and blends it
    into the frame.

    Failure is soft: if XFixes is unavailable the frames are returned unchanged,
    because a missing cursor must never break perception entirely.
    """

    def __init__(self, display: bytes = b":99") -> None:
        self._display_name = display
        self._x11 = None
        self._fixes = None
        self._display = None
        self._available = False
        self._error = ""
        try:
            x11_name = ctypes.util.find_library("X11")
            fixes_name = ctypes.util.find_library("Xfixes")
            if not x11_name or not fixes_name:
                raise RuntimeError("libX11/libXfixes not found")
            self._x11 = ctypes.CDLL(x11_name)
            self._fixes = ctypes.CDLL(fixes_name)
            self._x11.XOpenDisplay.restype = ctypes.c_void_p
            self._fixes.XFixesGetCursorImage.restype = ctypes.POINTER(
                _XFixesCursorImage
            )
            self._display = self._x11.XOpenDisplay(display)
            if not self._display:
                raise RuntimeError(f"cannot open display {display!r}")
            self._available = True
        except Exception as exc:  # pragma: no cover - environment dependent
            self._error = str(exc)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def error(self) -> str:
        return self._error

    def read(self) -> tuple[int, int, int, int, object] | None:
        """Return ``(x, y, w, h, argb)`` for the cursor, or None."""
        if not self._available:
            return None
        try:
            image = self._fixes.XFixesGetCursorImage(ctypes.c_void_p(self._display))
            if not image:
                return None
            cursor = image.contents
            width = int(cursor.width)
            height = int(cursor.height)
            if width <= 0 or height <= 0:
                return None
            count = width * height
            raw = ctypes.string_at(cursor.pixels, count * ctypes.sizeof(ctypes.c_ulong))
            # Each sprite pixel is a CARD32 ARGB value held in a C ``long``, so
            # only the low 4 bytes of every 8-byte slot are meaningful.
            argb = np.frombuffer(raw, dtype=np.uint8).reshape(count, 8)[:, :4]
            values = argb.copy().view(np.uint32).reshape(count)
            return int(cursor.x), int(cursor.y), width, height, values
        except Exception:  # pragma: no cover - environment dependent
            return None

    def stamp(self, rgb: bytearray, width: int, height: int) -> bool:
        """Blend the cursor into an RGB bytearray in place.  True if drawn."""
        cursor = self.read()
        if cursor is None:
            return False
        x, y, cw, ch, pixels = cursor
        # ``x``/``y`` are the cursor's top-left in root coordinates.
        left = max(0, x)
        top = max(0, y)
        right = min(width, x + cw)
        bottom = min(height, y + ch)
        if left >= right or top >= bottom:
            return False
        sprite = pixels.reshape(ch, cw)
        crop = sprite[top - y : bottom - y, left - x : right - x]
        alpha = ((crop >> 24) & 0xFF).astype(np.uint16)
        opaque = alpha > 0
        if not opaque.any():
            return False
        region = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
        target = region[top:bottom, left:right]
        sprite_rgb = np.stack(
            [((crop >> 16) & 0xFF), ((crop >> 8) & 0xFF), (crop & 0xFF)], axis=-1
        ).astype(np.uint16)
        inverse = 255 - alpha
        blended = (sprite_rgb * alpha[..., None] + target.astype(np.uint16) * inverse[..., None]) // 255
        target[opaque] = blended[opaque].astype(np.uint8)
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--no-cursor",
        action="store_true",
        help="do not composite the X11 pointer into frames",
    )
    args = parser.parse_args()
    if not 1.0 <= args.fps <= 60.0:
        parser.error("--fps must be between 1 and 60")

    interval = 1.0 / args.fps
    frame_id = 0
    output = sys.stdout.buffer
    compositor = None
    if not args.no_cursor:
        compositor = CursorCompositor()
        if not compositor.available:
            print(
                f"[avcs_frame_stream] cursor compositing disabled: {compositor.error}",
                file=sys.stderr,
                flush=True,
            )
            compositor = None
    with mss.MSS() as capture:
        monitor = capture.monitors[0]
        width = int(monitor["width"])
        height = int(monitor["height"])
        if (
            width <= 0
            or height <= 0
            or width * height > _MAX_FRAME_PIXELS
        ):
            raise RuntimeError("X11 display dimensions are invalid or exceed the frame limit")

        while True:
            started = time.perf_counter()
            if select.select([sys.stdin.buffer], [], [], 0)[0]:
                if not sys.stdin.buffer.read(1):
                    return 0
            frame = capture.grab(monitor)
            if frame.width != width or frame.height != height:
                raise RuntimeError("X11 display dimensions changed during capture")
            rgb = frame.rgb
            if len(rgb) != width * height * 3:
                raise RuntimeError("X11 capture returned an invalid RGB frame")
            if compositor is not None:
                # ``frame.rgb`` is a fresh bytes object each grab; a bytearray is
                # needed for in-place blending.
                rgb = bytearray(rgb)
                if not compositor.stamp(rgb, width, height):
                    rgb = bytes(rgb)

            frame_id += 1
            capture_timestamp_ns = time.time_ns()
            output.write(
                _HEADER.pack(_MAGIC, frame_id, capture_timestamp_ns, width, height)
            )
            output.write(rgb)
            output.flush()

            remaining = interval - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
