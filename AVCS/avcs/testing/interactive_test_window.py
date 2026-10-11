"""AVCS Milestone 8: Purpose-Built Interactive Test Window.

Isolated, non-destructive test application for Phase B live Windows certification.
Creates a controlled, single-purpose window with known geometry, HWND, and a stateful target button.
Writes state to a local JSON file to enable 100% independent oracle postcondition verification.
Fully Per-Monitor V2 DPI aware to align 1:1 with physical display coordinate space.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time

# Enable Per-Monitor V2 DPI awareness and attach to interactive desktop BEFORE initializing Tkinter
if sys.platform == "win32":
    try:
        h_desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
        if h_desk:
            ctypes.windll.user32.SetThreadDesktop(h_desk)
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

import tkinter as tk
from typing import Dict, Any


def get_hwnd_for_window(title: str) -> int:
    """Finds the HWND for a given window title using user32.FindWindowW."""
    if sys.platform != "win32":
        return 0
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    return int(hwnd)


class InteractiveTestApp:
    def __init__(self, state_file: Path, window_title: str = "AVCS_M8_LIVE_TEST_TARGET") -> None:
        self.state_file = state_file
        self.window_title = window_title
        self.click_count = 0
        self.status = "IDLE_STATE"
        self.created_at = time.time()
        self.last_clicked_at: float | None = None

        self.root = tk.Tk()
        self.root.title(self.window_title)
        # Position at (300, 300) with size 400x300 in physical pixels
        self.root.geometry("400x300+300+300")
        self.root.configure(bg="#202020")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # Header Label
        self.label = tk.Label(
            self.root,
            text="AVCS Certification Test Target",
            font=("Segoe UI", 12, "bold"),
            fg="#FFFFFF",
            bg="#202020",
        )
        self.label.pack(pady=20)

        # Stateful Target Button (prominent in center)
        self.button = tk.Button(
            self.root,
            text=f"TARGET: {self.status}",
            font=("Segoe UI", 14, "bold"),
            bg="#0078D7",
            fg="#FFFFFF",
            activebackground="#005A9E",
            activeforeground="#FFFFFF",
            padx=30,
            pady=15,
            command=self._on_button_click,
        )
        self.button.pack(expand=True)

        # Status text
        self.status_label = tk.Label(
            self.root,
            text="Awaiting Action",
            font=("Segoe UI", 10),
            fg="#AAAAAA",
            bg="#202020",
        )
        self.status_label.pack(pady=15)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Periodic command check
        self.root.after(50, self._check_command)

        # Save initial state
        self._write_state()

    def _on_button_click(self) -> None:
        self.click_count += 1
        self.last_clicked_at = time.time()
        self.status = "ACTIVATED_STATE"
        self.button.configure(text=f"ACTIVATED (Clicks: {self.click_count})", bg="#107C10")
        self.status_label.configure(text=f"Clicked at {time.strftime('%H:%M:%S')}", fg="#7FBA00")
        self._write_state()

    def _write_state(self) -> None:
        state_data: Dict[str, Any] = {
            "window_title": self.window_title,
            "pid": os.getpid(),
            "click_count": self.click_count,
            "status": self.status,
            "created_at": self.created_at,
            "last_clicked_at": self.last_clicked_at,
            "timestamp": time.time(),
        }
        tmp_file = self.state_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
        tmp_file.replace(self.state_file)

    def _check_command(self) -> None:
        cmd_file = self.state_file.with_name("command.json")
        if cmd_file.is_file():
            try:
                cmd = json.loads(cmd_file.read_text(encoding="utf-8"))
                if cmd.get("action") == "exit":
                    cmd_file.unlink(missing_ok=True)
                    self._on_close()
                    return
            except Exception:
                pass
        self.root.after(50, self._check_command)

    def _on_close(self) -> None:
        self._write_state()
        self.root.destroy()
        sys.exit(0)

    def run(self) -> None:
        self.root.update()
        hwnd = get_hwnd_for_window(self.window_title)
        if hwnd and sys.platform == "win32":
            ctypes.windll.user32.SetForegroundWindow(hwnd)

        # Accurate physical coordinates from Win32 ClientToScreen
        bx = self.button.winfo_rootx() + self.button.winfo_width() // 2
        by = self.button.winfo_rooty() + self.button.winfo_height() // 2
        
        info = {
            "status": "WINDOW_READY",
            "pid": os.getpid(),
            "hwnd": hwnd,
            "title": self.window_title,
            "window_x": self.root.winfo_rootx(),
            "window_y": self.root.winfo_rooty(),
            "window_w": self.root.winfo_width(),
            "window_h": self.root.winfo_height(),
            "target_center_x": bx,
            "target_center_y": by,
        }
        print(json.dumps(info), flush=True)
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Phase B Interactive Test Target Window")
    parser.add_argument("--state-file", required=True, help="Path to state JSON file")
    parser.add_argument("--title", default="AVCS_M8_LIVE_TEST_TARGET", help="Window title")
    args = parser.parse_args()

    app = InteractiveTestApp(state_file=Path(args.state_file), window_title=args.title)
    app.run()


if __name__ == "__main__":
    main()
