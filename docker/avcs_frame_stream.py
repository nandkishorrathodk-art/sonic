from __future__ import annotations

import argparse
import select
import struct
import sys
import time

import mss

_HEADER = struct.Struct("!4sQQII")
_MAGIC = b"AVCS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if not 1.0 <= args.fps <= 60.0:
        parser.error("--fps must be between 1 and 60")

    interval = 1.0 / args.fps
    frame_id = 0
    output = sys.stdout.buffer
    with mss.MSS() as capture:
        monitor = capture.monitors[0]
        width = int(monitor["width"])
        height = int(monitor["height"])
        if width <= 0 or height <= 0 or width * height > 64_000_000:
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
