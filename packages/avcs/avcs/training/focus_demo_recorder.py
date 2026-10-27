"""Opt-in local recorder for labeled focus demonstrations on a Docker VNC desktop.

The AVCS focus-verification head answers "is this screen point a focused
text-entry control?" from the current frame alone.  It is trained from *recorded*
demonstrations, but no recorder produced the ``records.jsonl`` layout that
``avcs.training.focus_dataset`` consumes -- the trainer was reachable only from
hand-written test fixtures.  Without a recorder the focus checkpoint could never
be produced from real data, so text entry stayed permanently blocked.

This module closes that gap.  It reuses the VNC click recorder's isolation
guarantees (exact foreground window, cropped frames, no keystroke capture) and
adds the one thing a focus demonstration needs that a click demonstration does
not: a human ``focused`` ground-truth label for the clicked point.  The label is
never inferred from the OS or a model -- inferring it would defeat the purpose of
training a classifier on it.

Output layout (one session directory), matching ``discover_focus_demonstrations``:

* ``manifest.json`` -- ``status: "complete"``, ``task``, ``target_window_title``
* ``records.jsonl`` -- one JSON object per sample with ``schema_version: 1``,
  ``target_label``, ``frame`` (relative path), ``point_px`` (integer pixel in the
  *cropped window frame*), and the boolean ``focused`` label
* ``frames/<id>.png`` -- the cropped target-window frame at click time

Keyboard input is never captured and nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from avcs.training.demo_recorder import crop_window_frame
from avcs.training.vnc_demo_recorder import (
    _MAX_FRAME_AGE_SECONDS,
    _is_left_raw_button_press,
    _parse_raw_button_press,
    _parse_window_listing,
    _window_id,
)
from PIL import Image

FOCUS_RECORD_SCHEMA = 1
_CAPTURE_INTERVAL_SECONDS = 0.05
_WINDOW_LISTING_TIMEOUT = 2.0

# Explicit, non-inferred ground-truth tokens.  ``focused``/``not-focused`` is the
# single label the classifier is trained to predict; refusing anything else keeps
# a typo from silently becoming a mislabeled training sample.
_FOCUSED_TOKENS = frozenset({"focused", "focus", "yes", "y", "true", "1"})
_UNFOCUSED_TOKENS = frozenset(
    {"unfocused", "not-focused", "not_focused", "nofocus", "no", "n", "false", "0"}
)


def parse_focus_target(raw: str) -> tuple[str, bool]:
    """Parse one ``"<label>:<focused>"`` arming token into (label, focused).

    The focused flag is a human ground-truth label, so an unrecognised token is
    rejected rather than guessed.
    """
    if not isinstance(raw, str) or ":" not in raw:
        raise ValueError(
            "Focus targets must be '<label>:<focused>' (for example 'search field:focused')."
        )
    label, _, flag = raw.rpartition(":")
    label = label.strip()
    token = flag.strip().lower()
    if not label or len(label) > 160:
        raise ValueError("Focus target labels must contain 1 to 160 characters.")
    if token in _FOCUSED_TOKENS:
        return label, True
    if token in _UNFOCUSED_TOKENS:
        return label, False
    raise ValueError(
        f"Unrecognised focus flag {flag.strip()!r}; use focused or not-focused."
    )


def build_focus_record(
    *,
    target_label: str,
    frame_path: str,
    point_px: tuple[int, int],
    focused: bool,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Create a versioned, text-free focus label record for one clicked point."""
    if not isinstance(focused, bool):
        raise TypeError("focused must be a boolean ground-truth label")
    x, y = point_px
    return {
        "schema_version": FOCUS_RECORD_SCHEMA,
        "target_label": target_label,
        "frame": frame_path,
        "point_px": [int(x), int(y)],
        "focused": focused,
        "recorded_at_utc": recorded_at_utc,
    }


def cropped_point(
    pointer_position: tuple[int, int],
    crop_rect: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    """Map a screen-pixel point into the cropped window frame's pixel space."""
    left, top, right, bottom = crop_rect
    x, y = pointer_position
    if not (left <= x < right and top <= y < bottom):
        return None
    return x - left, y - top


class VNCFocusDemoRecorder:
    """Record human-labeled focus demonstrations in one exact, foreground VNC window."""

    def __init__(
        self,
        *,
        window_title: str,
        task: str,
        output_dir: Path,
        confirm_isolated_window: bool,
        targets: Sequence[tuple[str, bool]] = (),
        capture_factory: Any | None = None,
    ) -> None:
        if not window_title.strip() or len(window_title.strip()) > 160:
            raise ValueError("An exact window title of 1 to 160 characters is required.")
        if not task.strip() or len(task.strip()) > 500:
            raise ValueError("A task description of 1 to 500 characters is required.")
        if confirm_isolated_window is not True:
            raise ValueError("Recording requires --confirm-isolated-window.")
        if sys.platform != "linux":
            raise OSError("The Docker VNC focus recorder must run inside Linux.")
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("DISPLAY must point to the Docker VNC desktop.")
        self.window_title = window_title.strip()
        self.task = task.strip()
        self.output_dir = Path(output_dir).expanduser()
        self.capture_factory = capture_factory
        self._stop = threading.Event()
        self._targets: queue.Queue[tuple[str, bool]] = queue.Queue()
        for label, focused in targets:
            normalized = label.strip()
            if not normalized or len(normalized) > 160:
                raise ValueError("Focus target labels must contain 1 to 160 characters.")
            if not isinstance(focused, bool):
                raise TypeError("Focus target flags must be boolean ground truth.")
            self._targets.put((normalized, focused))
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot: (
            tuple[
                np.ndarray,
                tuple[int, int, int, int],
                tuple[int, int, int, int],
                float,
            ]
            | None
        ) = None
        self._record_lock = threading.Lock()
        self._records_file: Any = None
        self._sample_count = 0
        self._focused_count = 0
        self._started_at_utc = ""
        self._capture_error: list[BaseException] = []
        self._input_thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._event_thread: threading.Thread | None = None
        self._event_process: subprocess.Popen[str] | None = None

    def _run_text(self, args: Sequence[str]) -> str:
        result = subprocess.run(
            list(args),
            check=True,
            capture_output=True,
            text=True,
            timeout=_WINDOW_LISTING_TIMEOUT,
        )
        return result.stdout

    def _active_window(self) -> dict[str, Any] | None:
        active_id = _window_id(self._run_text(["xdotool", "getactivewindow"]))
        windows = _parse_window_listing(self._run_text(["wmctrl", "-lG"]))
        matches = [
            window
            for window in windows
            if window["window_id"] == active_id
            and window["title"] == self.window_title
        ]
        return matches[0] if len(matches) == 1 else None

    def _target_window_matches_snapshot(
        self, rect: tuple[int, int, int, int]
    ) -> bool:
        windows = _parse_window_listing(self._run_text(["wmctrl", "-lG"]))
        matches = [
            window
            for window in windows
            if window["title"] == self.window_title and window["rect"] == rect
        ]
        return len(matches) == 1

    def _capture_loop(self) -> None:
        capture_factory = self.capture_factory
        if capture_factory is None:
            import mss

            capture_factory = mss.MSS
        try:
            with capture_factory() as capture:
                monitor = capture.monitors[0]
                while not self._stop.is_set():
                    started = time.monotonic()
                    window = self._active_window()
                    if window is not None:
                        shot = capture.grab(monitor)
                        frame = (
                            np.frombuffer(shot.rgb, dtype=np.uint8)
                            .reshape(shot.height, shot.width, 3)
                            .copy()
                        )
                        window_rect = window["rect"]
                        crop, crop_rect = crop_window_frame(frame, window_rect)
                        with self._snapshot_lock:
                            self._latest_snapshot = (
                                crop,
                                crop_rect,
                                window_rect,
                                time.monotonic(),
                            )
                    else:
                        with self._snapshot_lock:
                            self._latest_snapshot = None
                    self._stop.wait(
                        max(
                            0.0,
                            _CAPTURE_INTERVAL_SECONDS - (time.monotonic() - started),
                        )
                    )
        except BaseException as exc:
            self._capture_error.append(exc)
            self._stop.set()

    def _read_targets(self) -> None:
        """Prompt for label + focused ground truth for each upcoming click."""
        while not self._stop.is_set():
            if not self._targets.empty():
                self._stop.wait(0.05)
                continue
            try:
                label = input(
                    "Target label for next click (:stop to finish): "
                ).strip()
                if label == ":stop":
                    self._stop.set()
                    return
                if not label or len(label) > 160:
                    print("Label must contain 1 to 160 characters.", file=sys.stderr)
                    continue
                flag = input("Is this point a focused text-entry control? [y/N]: ").strip()
                if flag.lower() in _FOCUSED_TOKENS:
                    focused = True
                elif flag.lower() in _UNFOCUSED_TOKENS or flag == "":
                    focused = False
                else:
                    print("Answer y or n.", file=sys.stderr)
                    continue
            except (EOFError, KeyboardInterrupt):
                self._stop.set()
                return
            self._targets.put((label, focused))
            print(f"Armed: {label!r} focused={focused}", flush=True)

    def _pointer_position(self) -> tuple[int, int]:
        output = self._run_text(["xdotool", "getmouselocation", "--shell"])
        values: dict[str, int] = {}
        for line in output.splitlines():
            key, separator, value = line.partition("=")
            if separator and key in {"X", "Y"}:
                values[key] = int(value)
        if set(values) != {"X", "Y"}:
            raise RuntimeError("Could not read the VNC pointer position.")
        return values["X"], values["Y"]

    def _record_current_pointer_click(self) -> None:
        self._record_click(self._pointer_position())

    def _record_click(self, pointer_position: tuple[int, int]) -> None:
        with self._snapshot_lock:
            snapshot = self._latest_snapshot
        if snapshot is None:
            print("Ignored click: the selected window is not foreground.", file=sys.stderr)
            return
        crop, crop_rect, window_rect, captured_at = snapshot
        if time.monotonic() - captured_at > _MAX_FRAME_AGE_SECONDS:
            print("Ignored click: the captured window frame is stale.", file=sys.stderr)
            return
        if not self._target_window_matches_snapshot(window_rect):
            print("Ignored click: the captured target window moved or closed.", file=sys.stderr)
            return
        point = cropped_point(pointer_position, crop_rect)
        if point is None:
            print("Ignored click: pointer is outside the target window.", file=sys.stderr)
            return
        try:
            label, focused = self._targets.get_nowait()
        except queue.Empty:
            print("Ignored click: enter a target label and focus flag first.", file=sys.stderr)
            return

        sample_id = uuid.uuid4().hex
        relative_frame = f"frames/{sample_id}.png"
        Image.fromarray(crop).save(self.output_dir / relative_frame)
        record = build_focus_record(
            target_label=label,
            frame_path=relative_frame,
            point_px=point,
            focused=focused,
            recorded_at_utc=datetime.now(UTC).isoformat(),
        )
        with self._record_lock:
            self._records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._records_file.flush()
            self._sample_count += 1
            if focused:
                self._focused_count += 1
            count = self._sample_count
        print(f"Recorded focus sample {count}: {label!r} focused={focused}")

    def _read_xinput_events(self) -> None:
        assert self._event_process is not None
        assert self._event_process.stdout is not None
        event_lines: list[str] = []
        click_handled = False
        for line in self._event_process.stdout:
            if self._stop.is_set():
                return
            if line.startswith("EVENT "):
                if event_lines and not click_handled:
                    pointer_position = _parse_raw_button_press(event_lines)
                    if pointer_position is not None:
                        self._record_click(pointer_position)
                event_lines = [line]
                click_handled = False
                continue
            if event_lines:
                event_lines.append(line)
                if (
                    not click_handled
                    and len(event_lines) > 1
                    and _is_left_raw_button_press(
                        event_lines[0].rstrip(),
                        line.strip() if line.strip().startswith("detail:") else "",
                    )
                ):
                    pointer_position = _parse_raw_button_press(event_lines)
                    if pointer_position is None:
                        self._record_current_pointer_click()
                    else:
                        self._record_click(pointer_position)
                    click_handled = True

    def _write_manifest(self, status: str) -> None:
        manifest = {
            "schema_version": FOCUS_RECORD_SCHEMA,
            "status": status,
            "task": self.task,
            "target_window_title": self.window_title,
            "started_at_utc": self._started_at_utc,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "sample_count": self._sample_count,
            "focused_sample_count": self._focused_count,
            "capture_scope": "cropped target window only",
            "keyboard_recording": False,
            "network_upload": False,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def run(self) -> int:
        if self.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {self.output_dir}")
        if self._active_window() is None:
            raise RuntimeError(
                f"Expected exactly one active window titled {self.window_title!r}."
            )
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "frames").mkdir()
        self._started_at_utc = datetime.now(UTC).isoformat()
        records_path = self.output_dir / "records.jsonl"
        status = "failed"
        self._records_file = records_path.open("x", encoding="utf-8", newline="\n")
        try:
            self._event_process = subprocess.Popen(
                ["xinput", "test-xi2", "--root"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="avcs-focus-capture", daemon=True
            )
            self._input_thread = threading.Thread(
                target=self._read_targets, name="avcs-focus-label-input", daemon=True
            )
            self._event_thread = threading.Thread(
                target=self._read_xinput_events,
                name="avcs-focus-input-events",
                daemon=True,
            )
            self._capture_thread.start()
            self._event_thread.start()
            print(
                "Focus recorder ready. Keep the exact isolated window in front. "
                "For each click: type a target label, answer the focused y/n "
                "prompt, then left-click that control in the VNC window. Enter "
                ":stop to finish. Keyboard input is not recorded.",
                flush=True,
            )
            if not self._targets.empty():
                print(
                    f"{self._targets.qsize()} focus target(s) armed from command line.",
                    flush=True,
                )
            self._input_thread.start()
            while not self._stop.is_set():
                if self._capture_error:
                    error = self._capture_error[0]
                    raise RuntimeError(
                        f"VNC frame capture failed: {type(error).__name__}: {error}"
                    ) from error
                if self._event_thread and not self._event_thread.is_alive():
                    raise RuntimeError("xinput click-event listener stopped unexpectedly")
                self._stop.wait(0.1)
            if self._capture_error:
                error = self._capture_error[0]
                raise RuntimeError(
                    f"VNC frame capture failed: {type(error).__name__}: {error}"
                ) from error
            status = "complete"
            return self._sample_count
        finally:
            self._stop.set()
            if self._event_process is not None and self._event_process.poll() is None:
                self._event_process.terminate()
                try:
                    self._event_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._event_process.kill()
                    self._event_process.wait()
            for thread in (self._capture_thread, self._event_thread):
                if thread is not None:
                    thread.join(timeout=2)
            self._records_file.close()
            self._write_manifest(status)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record human-labeled focus demonstrations in an isolated VNC window."
    )
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Pre-arm a click as '<label>:<focused>' where focused is "
            "focused/not-focused; may be repeated for a sequence of clicks."
        ),
    )
    parser.add_argument("--confirm-isolated-window", action="store_true")
    args = parser.parse_args()
    try:
        targets = [parse_focus_target(raw) for raw in args.target]
        recorder = VNCFocusDemoRecorder(
            window_title=args.window_title,
            task=args.task,
            output_dir=args.output_dir,
            confirm_isolated_window=args.confirm_isolated_window,
            targets=targets,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Focus demo recording failed: {exc}", file=sys.stderr)
        return 1
    try:
        count = recorder.run()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Focus demo recording failed: {exc}", file=sys.stderr)
        return 1
    print(f"Recording stopped with {count} labeled focus samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
