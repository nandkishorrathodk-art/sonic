"""Opt-in local recorder for labeled clicks made through a Docker VNC desktop."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
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
from avcs.training.demo_recorder import (
    build_click_record,
    crop_window_frame,
    normalized_click,
)
from PIL import Image

_WINDOW_LISTING_TIMEOUT = 2.0
_MAX_FRAME_AGE_SECONDS = 0.15
_CAPTURE_INTERVAL_SECONDS = 0.05


def _window_id(value: str) -> int:
    token = value.strip().lower()
    if token.startswith("0x"):
        return int(token, 16)
    return int(token, 10)


def _parse_window_listing(output: str) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=7)
        if len(fields) != 8:
            continue
        try:
            window_id = int(fields[0], 16)
            x, y, width, height = (int(value) for value in fields[2:6])
        except ValueError:
            continue
        title = fields[7].strip()
        if title and width > 0 and height > 0:
            windows.append(
                {
                    "window_id": window_id,
                    "title": title,
                    "rect": (x, y, x + width, y + height),
                }
            )
    return windows


def _is_left_raw_button_press(event_header: str, detail: str) -> bool:
    match = re.fullmatch(r"\s*EVENT type \d+ \(([^)]+)\)\s*", event_header)
    return bool(
        match
        and match.group(1) == "RawButtonPress"
        and detail.strip() == "detail: 1"
    )


def _parse_raw_button_press(event_lines: Sequence[str]) -> tuple[int, int] | None:
    if not event_lines or not event_lines[0].startswith("EVENT "):
        return None
    detail = next(
        (line.strip() for line in event_lines[1:] if line.strip().startswith("detail:")),
        "",
    )
    if not _is_left_raw_button_press(event_lines[0].rstrip(), detail):
        return None
    root = next(
        (line.strip()[len("root:") :].strip() for line in event_lines[1:]
         if line.strip().startswith("root:")),
        "",
    )
    match = re.fullmatch(
        r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)",
        root,
    )
    if match is None:
        return None
    return round(float(match.group(1))), round(float(match.group(2)))


class VNCClickDemoRecorder:
    """Record human-labeled left clicks in one exact, foreground VNC window."""

    def __init__(
        self,
        *,
        window_title: str,
        task: str,
        output_dir: Path,
        confirm_isolated_window: bool,
        target_labels: Sequence[str] = (),
        capture_factory: Any | None = None,
    ) -> None:
        if not window_title.strip() or len(window_title.strip()) > 160:
            raise ValueError("An exact window title of 1 to 160 characters is required.")
        if not task.strip() or len(task.strip()) > 500:
            raise ValueError("A task description of 1 to 500 characters is required.")
        if confirm_isolated_window is not True:
            raise ValueError("Recording requires --confirm-isolated-window.")
        if sys.platform != "linux":
            raise OSError("The Docker VNC demonstration recorder must run inside Linux.")
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("DISPLAY must point to the Docker VNC desktop.")
        self.window_title = window_title.strip()
        self.task = task.strip()
        self.output_dir = Path(output_dir).expanduser()
        self.capture_factory = capture_factory
        self._stop = threading.Event()
        self._labels: queue.Queue[str] = queue.Queue()
        for label in target_labels:
            normalized_label = label.strip()
            if not normalized_label or len(normalized_label) > 160:
                raise ValueError("Target labels must contain 1 to 160 characters.")
            self._labels.put(normalized_label)
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
        self._samples_file: Any = None
        self._sample_count = 0
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
                        frame = np.frombuffer(
                            shot.rgb, dtype=np.uint8
                        ).reshape(shot.height, shot.width, 3).copy()
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
                        max(0.0, _CAPTURE_INTERVAL_SECONDS - (time.monotonic() - started))
                    )
        except BaseException as exc:
            self._capture_error.append(exc)
            self._stop.set()

    def _read_labels(self) -> None:
        while not self._stop.is_set():
            if not self._labels.empty():
                self._stop.wait(0.05)
                continue
            try:
                label = input(
                    "Target label for next left-click (:stop to finish): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self._stop.set()
                return
            if label == ":stop":
                self._stop.set()
                return
            if not label or len(label) > 160:
                print("Label must contain 1 to 160 characters.", file=sys.stderr)
                continue
            self._labels.put(label)
            print(f"Label armed for next left-click: {label}", flush=True)

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
            print("Ignored VNC click: the selected window is not foreground.", file=sys.stderr)
            return
        crop, crop_rect, window_rect, captured_at = snapshot
        if time.monotonic() - captured_at > _MAX_FRAME_AGE_SECONDS:
            print("Ignored VNC click: the captured window frame is stale.", file=sys.stderr)
            return
        if not self._target_window_matches_snapshot(window_rect):
            print(
                "Ignored VNC click: the captured target window moved or closed.",
                file=sys.stderr,
            )
            return
        point_norm = normalized_click(pointer_position, crop_rect)
        if point_norm is None:
            print("Ignored VNC click: pointer is outside the target window.", file=sys.stderr)
            return
        try:
            label = self._labels.get_nowait()
        except queue.Empty:
            print("Ignored VNC click: enter a target label first.", file=sys.stderr)
            return

        sample_id = uuid.uuid4().hex
        relative_frame = f"frames/{sample_id}.png"
        Image.fromarray(crop).save(self.output_dir / relative_frame)
        record = build_click_record(
            task=self.task,
            target_label=label,
            frame_path=relative_frame,
            point_norm=point_norm,
            recorded_at_utc=datetime.now(UTC).isoformat(),
        )
        with self._record_lock:
            self._samples_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            self._samples_file.flush()
            self._sample_count += 1
            count = self._sample_count
        print(f"Recorded VNC click {count}: {label}")

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
                        line.strip()
                        if line.strip().startswith("detail:")
                        else "",
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
            "schema_version": 1,
            "status": status,
            "task": self.task,
            "target_window_title": self.window_title,
            "started_at_utc": self._started_at_utc,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "sample_count": self._sample_count,
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
        samples_path = self.output_dir / "samples.jsonl"
        status = "failed"
        self._samples_file = samples_path.open("x", encoding="utf-8", newline="\n")
        try:
            self._event_process = subprocess.Popen(
                ["xinput", "test-xi2", "--root"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="avcs-vnc-capture", daemon=True
            )
            self._input_thread = threading.Thread(
                target=self._read_labels, name="avcs-vnc-label-input", daemon=True
            )
            self._event_thread = threading.Thread(
                target=self._read_xinput_events,
                name="avcs-vnc-input-events",
                daemon=True,
            )
            self._capture_thread.start()
            self._event_thread.start()
            print(
                "VNC recorder ready. Keep the exact isolated window in front. "
                "Type a target label in this terminal, press Enter, and wait "
                "for 'Label armed' before each VNC left-click; enter :stop "
                "to finish. Keyboard input is not recorded.",
                flush=True,
            )
            if not self._labels.empty():
                print(
                    f"{self._labels.qsize()} target label(s) armed from command line.",
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
            status = "stopped"
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
            self._samples_file.close()
            self._write_manifest(status)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record human-labeled clicks in an isolated Docker VNC window."
    )
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--target-label",
        action="append",
        default=[],
        help="Pre-arm a click label; may be repeated for a sequence of clicks.",
    )
    parser.add_argument("--confirm-isolated-window", action="store_true")
    args = parser.parse_args()
    recorder = VNCClickDemoRecorder(
        window_title=args.window_title,
        task=args.task,
        output_dir=args.output_dir,
        confirm_isolated_window=args.confirm_isolated_window,
        target_labels=args.target_label,
    )
    try:
        count = recorder.run()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"VNC demo recording failed: {exc}", file=sys.stderr)
        return 1
    print(f"Recording stopped with {count} labeled click samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
