"""Capture labeled real-desktop frames from the live workstation.

Opens a variety of real applications in different states and saves frames with
an honest scene label, so the AVCS cortex can be trained on genuine desktop
imagery instead of synthetic patterns. The label is the *scene class* the
cortex is asked to predict; nothing is invented about the pixels themselves.

Output layout (consumed by the cortex video/real-desktop trainer):

    <out>/frames/frame_XXXX.png
    <out>/manifest.json   # [{frame, label, app, note}, ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

CONTAINER = "sonic-desktop-workstation"
W, H = 1920, 1080


def _exec(cmd: str) -> None:
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", cmd], capture_output=True)


def _grab() -> np.ndarray | None:
    out = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-c",
         "DISPLAY=:99 import -window root -depth 8 rgb:- 2>/dev/null"],
        capture_output=True,
    )
    n = W * H * 3
    if len(out.stdout) < n:
        return None
    return np.frombuffer(out.stdout[:n], dtype=np.uint8).reshape(H, W, 3)


def _kill_all() -> None:
    _exec("pkill -f firefox-esr; pkill -f xfce4-terminal; pkill -f thunar; "
          "pkill -f mousepad; pkill -f xfce4-appfinder; sleep 1; true")


# Each scenario: (label, note, setup commands, settle seconds, frames to take)
SCENARIOS: list[tuple[str, str, list[str], float, int]] = [
    ("desktop_canvas", "bare XFCE desktop",
     ["pkill -f firefox-esr; pkill -f thunar; pkill -f mousepad; sleep 2; true"], 3, 8),
    ("desktop_canvas", "desktop with panel menus open",
     ["DISPLAY=:99 xdotool key super+escape; sleep 1; true"], 2, 4),
    ("browser_webpage", "firefox blank/new tab",
     ["DISPLAY=:99 nohup firefox-esr about:blank >/dev/null 2>&1 &"], 12, 8),
    ("browser_webpage", "firefox with a real page loaded",
     ["DISPLAY=:99 nohup firefox-esr https://example.com >/dev/null 2>&1 &"], 14, 10),
    ("browser_webpage", "firefox scrolled / different page",
     ["DISPLAY=:99 xdotool search --class firefox windowactivate key F5; sleep 6; true"], 4, 6),
    ("terminal_shell", "xfce terminal prompt",
     ["DISPLAY=:99 nohup xfce4-terminal >/dev/null 2>&1 &"], 6, 8),
    ("terminal_shell", "terminal running a command",
     ["DISPLAY=:99 xdotool search --class xfce4-terminal windowactivate "
      "type --delay 40 'ls -la /usr/share/doc | head -20'; sleep 1; "
      "DISPLAY=:99 xdotool key Return"], 4, 8),
    ("terminal_shell", "terminal with lots of output",
     ["DISPLAY=:99 xdotool search --class xfce4-terminal windowactivate "
      "type --delay 30 'ls -la /usr/bin | head -60'; sleep 1; "
      "DISPLAY=:99 xdotool key Return"], 5, 8),
    ("settings_panel", "xfce application finder",
     ["DISPLAY=:99 nohup xfce4-appfinder >/dev/null 2>&1 &"], 6, 8),
    ("settings_panel", "xfce settings dialog",
     ["DISPLAY=:99 nohup xfce4-settings-manager >/dev/null 2>&1 &"], 7, 8),
    ("settings_panel", "settings manager sub-panel",
     ["DISPLAY=:99 nohup xfce4-settings-manager >/dev/null 2>&1 & sleep 4; "
      "DISPLAY=:99 xdotool search --class xfce4-settings-manager windowactivate "
      "key Tab Tab Return; sleep 3; true"], 5, 8),
    ("settings_panel", "xfce display settings",
     ["DISPLAY=:99 nohup xfce4-display-settings >/dev/null 2>&1 &"], 6, 8),
    ("settings_panel", "xfce keyboard settings",
     ["DISPLAY=:99 nohup xfce4-keyboard-settings >/dev/null 2>&1 &"], 6, 8),
    ("settings_panel", "appearance settings",
     ["DISPLAY=:99 nohup xfce4-appearance-settings >/dev/null 2>&1 &"], 6, 8),
    ("editor_code", "mousepad text editor empty",
     ["DISPLAY=:99 nohup mousepad >/dev/null 2>&1 &"], 6, 8),
    ("editor_code", "mousepad with code text",
     ["DISPLAY=:99 xdotool search --class mousepad windowactivate "
      "type --delay 25 'def main():\\n    return 42\\n\\nif __name__ == \"__main__\":\\n"
      "    print(main())\\n'; sleep 1; true"], 4, 8),
    ("form_input", "file manager listing",
     ["DISPLAY=:99 nohup thunar >/dev/null 2>&1 &"], 7, 8),
    ("form_input", "file manager with a dialog",
     ["DISPLAY=:99 xdotool search --class Thunar windowactivate key ctrl+l; sleep 2; true"], 3, 6),
    ("media_playback", "image viewer",
     ["DISPLAY=:99 nohup ristretto /usr/share/backgrounds/*.jpg >/dev/null 2>&1 & "
      "|| DISPLAY=:99 nohup xfce4-terminal --command='echo image' >/dev/null 2>&1 &"], 8, 8),
    ("modal_dialog", "terminal plus overlapping windows",
     ["DISPLAY=:99 nohup thunar >/dev/null 2>&1 & sleep 2; "
      "DISPLAY=:99 xdotool search --class xfce4-terminal windowactivate; sleep 1; true"], 4, 6),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat the whole scenario list this many times")
    args = ap.parse_args()

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    idx = 0

    for rnd in range(args.rounds):
        for label, note, setup, settle, count in SCENARIOS:
            for cmd in setup:
                _exec(cmd)
            time.sleep(settle)
            for _ in range(count):
                frame = _grab()
                if frame is None:
                    continue
                name = f"frame_{idx:05d}.png"
                Image.fromarray(frame).resize((640, 360)).save(frames_dir / name)
                manifest.append({
                    "frame": f"frames/{name}",
                    "label": label,
                    "note": note,
                    "round": rnd,
                })
                idx += 1
                time.sleep(0.4)
            print(f"  captured {idx} frames (last: {label} / {note})", flush=True)

    _kill_all()
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"provenance": "real_desktop_capture",
                    "source": CONTAINER, "frames": manifest}, indent=2)
    )
    print(f"\nDONE: {idx} frames -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
