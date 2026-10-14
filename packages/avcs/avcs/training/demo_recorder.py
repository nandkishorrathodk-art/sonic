"""Opt-in local recorder for labeled clicks in one isolated Windows window."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Iterable, Optional, Sequence, Tuple
import uuid

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]


def find_exact_window_title(target: str, windows: Iterable[WindowInfo]) -> WindowInfo:
    """Require one visible window with an exact title match."""
    title = target.strip()
    if not title:
        raise ValueError("A non-empty exact window title is required.")
    matches = [window for window in windows if window.title == title]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one visible window titled {title!r}; found {len(matches)}."
        )
    return matches[0]


def crop_window_frame(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    virtual_origin: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a screen frame to a window; return crop pixels and screen-space bounds."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an RGB frame with shape (height, width, 3).")

    left, top, right, bottom = (int(value) for value in rect)
    origin_x, origin_y = virtual_origin
    x0 = max(0, left - origin_x)
    y0 = max(0, top - origin_y)
    x1 = min(frame.shape[1], right - origin_x)
    y1 = min(frame.shape[0], bottom - origin_y)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The target window is outside the captured screen frame.")

    screen_bounds = (x0 + origin_x, y0 + origin_y, x1 + origin_x, y1 + origin_y)
    return frame[y0:y1, x0:x1].copy(), screen_bounds


def normalized_click(
    point: Tuple[int, int],
    rect: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float]]:
    """Map an in-window screen-pixel point to the cropped window's [0, 1] space."""
    left, top, right, bottom = rect
    x, y = point
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0 or not (left <= x < right and top <= y < bottom):
        return None
    return (x - left) / width, (y - top) / height


def build_click_record(
    *,
    task: str,
    target_label: str,
    frame_path: str,
    point_norm: Tuple[float, float],
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Create a versioned, text-free action label record."""
    return {
        "schema_version": 1,
        "task": task,
        "target_label": target_label,
        "frame": frame_path,
        "action": {
            "type": "mouse_click",
            "button": "left",
            "point_norm": [round(point_norm[0], 6), round(point_norm[1], 6)],
        },
        "recorded_at_utc": recorded_at_utc,
    }


class WindowsDemoRecorder:
    """Capture pre-click window crops only while the selected app owns the foreground."""

    def __init__(
        self,
        *,
        window_title: str,
        task: str,
        output_dir: Path,
        target_label: Optional[str] = None,
        capture: Optional[Any] = None,
        user32: Optional[Any] = None,
        label_queue: Optional[queue.Queue[str]] = None,
        event_sink: Optional[Any] = None,
        poll_interval: float = 0.004,
        capture_interval: float = 0.05,
    ) -> None:
        if sys.platform != "win32" and user32 is None:
            raise OSError("The demonstration recorder requires Windows.")
        if not task.strip():
            raise ValueError("A task description is required.")
        if target_label is not None and (
            not target_label.strip() or len(target_label.strip()) > 160
        ):
            raise ValueError("A default target label must contain 1 to 160 characters.")
        self.window_title = window_title.strip()
        self.task = task.strip()
        self.target_label = target_label.strip() if target_label is not None else None
        self.output_dir = Path(output_dir)
        self.capture = capture
        self.user32 = user32
        self.label_queue = label_queue or queue.Queue()
        self.event_sink = event_sink
        self.poll_interval = poll_interval
        self.capture_interval = capture_interval
        self._stop = threading.Event()
        self._native_user32 = user32 is None
        self._user32 = user32 if user32 is not None else ctypes.windll.user32
        if poll_interval <= 0 or capture_interval <= 0:
            raise ValueError("Polling and capture intervals must be positive.")
        if self._native_user32:
            self._configure_user32()

    def _configure_user32(self) -> None:
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self._user32.GetWindowRect.restype = wintypes.BOOL
        self._user32.EnumWindows.argtypes = [
            ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
            wintypes.LPARAM,
        ]
        self._user32.EnumWindows.restype = wintypes.BOOL
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self._user32.GetAncestor.restype = wintypes.HWND
        self._user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self._user32.GetCursorPos.restype = wintypes.BOOL
        self._user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self._user32.GetAsyncKeyState.restype = ctypes.c_short

    def _window_text(self, hwnd: int) -> str:
        length = int(self._user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    def _visible_windows(self) -> list[WindowInfo]:
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        windows: list[WindowInfo] = []
        rect_type = wintypes.RECT()

        def visit(hwnd: int, _lparam: int) -> bool:
            if not self._user32.IsWindowVisible(hwnd):
                return True
            if not self._user32.GetWindowRect(hwnd, ctypes.byref(rect_type)):
                return True
            rect = (rect_type.left, rect_type.top, rect_type.right, rect_type.bottom)
            windows.append(WindowInfo(int(hwnd), self._window_text(hwnd), rect))
            return True

        callback = callback_type(visit)
        self._user32.EnumWindows(callback, 0)
        return windows

    def _selected_window(self) -> WindowInfo:
        return find_exact_window_title(self.window_title, self._visible_windows())

    def _owns_foreground(self, target_hwnd: int) -> bool:
        foreground = int(self._user32.GetForegroundWindow() or 0)
        if not foreground:
            return False
        root_owner = int(self._user32.GetAncestor(foreground, 3) or foreground)
        return root_owner == target_hwnd

    def _cursor_position(self) -> Tuple[int, int]:
        point = wintypes.POINT()
        if not self._user32.GetCursorPos(ctypes.byref(point)):
            raise OSError("GetCursorPos failed while recording a target-window click.")
        return int(point.x), int(point.y)

    def _capture_rgb(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        frame = self.capture.grab()
        geometry = self.capture.get_geometry()
        return frame, (geometry.virtual_left, geometry.virtual_top)

    def _write_manifest(self, *, status: str, sample_count: int) -> None:
        manifest = {
            "schema_version": 1,
            "status": status,
            "task": self.task,
            "target_window_title": self.window_title,
            "started_at_utc": self._started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": sample_count,
            "capture_scope": "cropped target window only",
            "keyboard_recording": False,
            "network_upload": False,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run(self) -> int:
        if self.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {self.output_dir}")

        window = self._selected_window()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir()
        self._started_at_utc = datetime.now(timezone.utc).isoformat()
        samples_path = self.output_dir / "samples.jsonl"
        sample_count = 0
        previous_left_down = False
        latest_crop: Optional[np.ndarray] = None
        latest_rect: Optional[Tuple[int, int, int, int]] = None
        latest_window_rect: Optional[Tuple[int, int, int, int]] = None
        latest_capture_time = 0.0
        last_capture_time = 0.0

        if self.capture is None:
            from avcs.capture_adapter import Win32GDICaptureAdapter

            self.capture = Win32GDICaptureAdapter()
        self._started_at_utc = datetime.now(timezone.utc).isoformat()
        status = "failed"
        capture_open = False
        samples_file = samples_path.open("x", encoding="utf-8", newline="\n")
        try:
            self.capture.open()
            capture_open = True
            if self.event_sink:
                label_hint = (
                    f" Default label is armed: {self.target_label!r}."
                    if self.target_label
                    else " Enter a target label in this terminal before each click."
                )
                self.event_sink(
                    "Recorder ready. Focus the selected window; only labeled left clicks "
                    f"are recorded.{label_hint} Enter :stop here to finish."
                )

            with samples_file:
                while not self._stop.is_set():
                    if not self._owns_foreground(window.hwnd):
                        previous_left_down = False
                        latest_crop = None
                        latest_rect = None
                        latest_window_rect = None
                        time.sleep(self.poll_interval)
                        continue

                    now = time.monotonic()
                    if now - last_capture_time >= self.capture_interval:
                        frame, origin = self._capture_rgb()
                        current_window = self._selected_window()
                        crop, crop_rect = crop_window_frame(frame, current_window.rect, origin)
                        latest_crop = crop
                        latest_rect = crop_rect
                        latest_capture_time = now
                        last_capture_time = now
                        latest_window_rect = current_window.rect

                    left_down = bool(int(self._user32.GetAsyncKeyState(0x01)) & 0x8000)
                    if left_down and not previous_left_down:
                        point = self._cursor_position()
                        point_norm = (
                            normalized_click(point, latest_rect) if latest_rect is not None else None
                        )
                        age = time.monotonic() - latest_capture_time
                        current_window = self._selected_window()
                        window_unchanged = current_window.rect == latest_window_rect
                        label = self._next_label() if window_unchanged else None
                        if (
                            label
                            and latest_crop is not None
                            and point_norm is not None
                            and age <= self.capture_interval * 3
                        ):
                            sample_id = uuid.uuid4().hex
                            relative_frame = f"frames/{sample_id}.png"
                            Image.fromarray(latest_crop).save(self.output_dir / relative_frame)
                            record = build_click_record(
                                task=self.task,
                                target_label=label,
                                frame_path=relative_frame,
                                point_norm=point_norm,
                                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                            )
                            samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            samples_file.flush()
                            sample_count += 1
                            if self.event_sink:
                                self.event_sink(f"Recorded labeled click {sample_count}: {label}")
                        elif self.event_sink:
                            if not window_unchanged:
                                reason = "the selected window moved after capture"
                            elif not label:
                                reason = (
                                    "no target label is armed; enter one in the recorder "
                                    "terminal before clicking"
                                )
                            elif point_norm is None:
                                reason = "the click is outside the selected window"
                            elif age > self.capture_interval * 3:
                                reason = "the captured frame is stale"
                            else:
                                reason = "the click could not be validated against the fresh frame"
                            self.event_sink(f"Click ignored: {reason}.")
                    previous_left_down = left_down
                    time.sleep(self.poll_interval)
            status = "stopped"
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            self._stop.set()
            try:
                samples_file.close()
            finally:
                try:
                    if capture_open:
                        self.capture.close()
                finally:
                    self._write_manifest(status=status, sample_count=sample_count)
        return sample_count

    def _next_label(self) -> Optional[str]:
        try:
            label = self.label_queue.get_nowait().strip()
        except queue.Empty:
            return self.target_label
        if not label or len(label) > 160:
            return None
        return label

    def stop(self) -> None:
        self._stop.set()


def _read_target_labels(recorder: WindowsDemoRecorder) -> None:
    while not recorder._stop.is_set():
        try:
            if recorder.target_label is None:
                prompt = "Next click target label (or :stop): "
            else:
                prompt = (
                    f"Next label [{recorder.target_label}] (Enter keeps it, :stop finishes): "
                )
            label = input(prompt).strip()
        except EOFError:
            recorder.stop()
            return
        if label.lower() == ":stop":
            recorder.stop()
            return
        if not label:
            if recorder.target_label is not None:
                continue
            print("Label cannot be empty; that click will not be recorded.", file=sys.stderr)
            continue
        if len(label) > 160:
            print("Label is longer than 160 characters; try a shorter description.", file=sys.stderr)
            continue
        recorder.label_queue.put(label)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record labeled left-click demonstrations from one isolated Windows window."
    )
    parser.add_argument("--window-title", required=True, help="Exact, unique visible window title.")
    parser.add_argument("--task", required=True, help="The operator's task for this demonstration.")
    parser.add_argument(
        "--target-label",
        help=(
            "Reuse this label for every click. Useful when recording repeated "
            "examples of one visible control."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New local directory for cropped frames, samples, and manifest.",
    )
    parser.add_argument(
        "--confirm-isolated-window",
        action="store_true",
        help="Confirm this is a disposable test window with no private content.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_isolated_window:
        parser.error("Pass --confirm-isolated-window only for a disposable, non-sensitive test window.")
    if args.target_label is not None and (
        not args.target_label.strip() or len(args.target_label.strip()) > 160
    ):
        parser.error("--target-label must contain 1 to 160 characters.")
    if not sys.stdin.isatty():
        parser.error("Interactive target labels require a terminal; no capture was started.")

    recorder = WindowsDemoRecorder(
        window_title=args.window_title,
        task=args.task,
        output_dir=args.output_dir,
        target_label=args.target_label,
        event_sink=print,
    )
    reader = threading.Thread(target=_read_target_labels, args=(recorder,), daemon=True)
    reader.start()
    try:
        count = recorder.run()
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f"Recorder stopped without claiming success: {exc}", file=sys.stderr)
        return 1
    print(f"Recorded {count} labeled click(s) locally in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
