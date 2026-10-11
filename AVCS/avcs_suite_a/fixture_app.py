"""Windows GUI fixture app for AVCS Suite A.

Run on a Windows interactive desktop:
    python -m avcs_suite_a.fixture_app --port 8765
The admin/oracle endpoints require a bearer token supplied through the child
process environment as AVCS_FIXTURE_ORACLE_TOKEN. The visible UI contains no
expected labels, target IDs, or scoring metadata.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import threading
import time
import tkinter as tk
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tkinter import ttk
from typing import Any
from urllib.parse import urlparse

from .catalog import CASE_BY_ID


class FixtureWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.lock = threading.RLock()
        self.token = os.environ.get("AVCS_FIXTURE_ORACLE_TOKEN", "")
        self.current_case_id = "button_exact"
        self.state: dict[str, Any] = {}
        self.widgets: dict[str, tk.Widget] = {}
        self.target_boxes: dict[str, dict[str, float]] = {}
        self.variables: dict[str, Any] = {}
        self.dynamic_widgets: list[tk.Widget] = []
        self.case_title = tk.StringVar(value="AVCS Suite A")
        root.title("AVCS Deterministic Fixture Suite A")
        root.geometry("1000x680+80+60")
        root.minsize(800, 600)
        root.configure(bg="#f3f5f8")
        self.header = tk.Label(root, textvariable=self.case_title, font=("Segoe UI", 16, "bold"),
                               bg="#f3f5f8", fg="#172033", anchor="w")
        self.header.place(x=32, y=24, width=900, height=36)
        self.subtitle = tk.Label(root, text="Demo controls only • state is evaluated independently",
                                 font=("Segoe UI", 10), bg="#f3f5f8", fg="#5b6475", anchor="w")
        self.subtitle.place(x=34, y=63, width=900, height=24)
        self.surface = tk.Frame(root, bg="white", highlightbackground="#d6dbe5", highlightthickness=1)
        self.surface.place(x=24, y=105, width=950, height=535)
        self.root.after(150, self._refresh_boxes)

    def load_case(self, case_id: str, done: threading.Event | None = None) -> None:
        try:
            case = CASE_BY_ID[case_id]
            for widget in self.surface.winfo_children():
                widget.destroy()
            self.widgets.clear()
            self.variables.clear()
            self.dynamic_widgets.clear()
            self.current_case_id = case_id
            self.state = dict(case.initial_state)
            self.case_title.set(case.title)
            title = tk.Label(self.surface, text=case.title, font=("Segoe UI", 13, "bold"),
                             bg="white", fg="#172033", anchor="w")
            title.place(x=28, y=22, width=840, height=32)
            intro = tk.Label(self.surface, text="Use the visible controls to complete the requested demo task.",
                             font=("Segoe UI", 10), bg="white", fg="#5b6475", anchor="w")
            intro.place(x=30, y=60, width=840, height=25)
            self._control_widgets(case)
            self.root.update_idletasks()
            self._refresh_boxes()
        finally:
            if done:
                done.set()

    def _control_widgets(self, case) -> None:
        for spec in case.controls:
            opts = dict(x=spec.x - 24, y=spec.y - 120, width=spec.width, height=spec.height)
            # Catalog coordinates refer to the full root surface; translate to the fixture surface.
            opts["x"] = max(4, spec.x - 24)
            opts["y"] = max(4, spec.y - 120)
            kind = spec.kind
            widget: tk.Widget
            if kind == "label":
                widget = tk.Label(self.surface, text=spec.label, font=("Segoe UI", 10),
                                  bg="white", fg="#283247", anchor="w")
            elif kind == "button":
                widget = ttk.Button(self.surface, text=spec.label,
                                    command=lambda s=spec: self._activate(s))
                if not spec.enabled:
                    widget.configure(state="disabled")
            elif kind == "check":
                var = tk.BooleanVar(value=bool(spec.initial))
                self.variables[spec.id] = var
                widget = tk.Checkbutton(self.surface, text=spec.label, variable=var,
                                        command=lambda s=spec, v=var: self._set_state(s.state_key, bool(v.get())),
                                        font=("Segoe UI", 10), bg="white", activebackground="white",
                                        anchor="w")
                if spec.state_key:
                    self.state[spec.state_key] = bool(spec.initial)
            elif kind == "combo":
                var = tk.StringVar(value=str(spec.initial or ""))
                self.variables[spec.id] = var
                widget = ttk.Combobox(self.surface, textvariable=var, values=list(spec.values), state="readonly")
                if spec.state_key:
                    self.state[spec.state_key] = str(spec.initial or "")
                widget.bind("<<ComboboxSelected>>", lambda _e, s=spec, v=var: self._set_state(s.state_key, v.get()))
            elif kind == "entry":
                var = tk.StringVar(value=str(spec.initial or ""))
                self.variables[spec.id] = var
                widget = ttk.Entry(self.surface, textvariable=var)
                if spec.state_key:
                    self.state[spec.state_key] = str(spec.initial or "")
                var.trace_add("write", lambda *_a, s=spec, v=var: self._set_state(s.state_key, v.get()))
            else:
                continue
            widget.place(**opts)
            self.widgets[spec.id] = widget
            if case.id == "keyboard_equivalent_not_required" and spec.id == "apply":
                self.root.after_idle(widget.focus_set)

        # A visible overlay covers the underlying control in this adversarial fixture.
        if case.id == "overlay_close":
            overlay = tk.Frame(self.surface, bg="#fff4d6", highlightbackground="#c28b14", highlightthickness=2)
            overlay.place(x=265, y=64, width=420, height=235)
            message = tk.Label(overlay, text="Notice: action temporarily unavailable", bg="#fff4d6",
                               fg="#513b0d", font=("Segoe UI", 11, "bold"))
            message.place(x=15, y=40, width=385, height=32)
            close = ttk.Button(overlay, text="Close", command=lambda: self._set_state("overlay_closed", True))
            close.place(x=145, y=115, width=130, height=40)
            self.widgets["close_notice"] = close
            self.dynamic_widgets.extend([overlay, message, close])

        if case.id == "stale_layout_reflow":
            # The action causes a visible geometry change and publishes the new target bounds.
            button = self.widgets.get("expand")
            if button:
                # Button behavior is wired by _activate; this label makes the postcondition visible.
                note = tk.Label(self.surface, text="Details: collapsed", bg="white", fg="#5b6475",
                                font=("Segoe UI", 10), anchor="w")
                note.place(x=60, y=270, width=420, height=28)
                self.widgets["details_status"] = note

    def _set_state(self, key: str | None, value: Any) -> None:
        if key:
            with self.lock:
                self.state[key] = value
        self.root.after_idle(self._refresh_boxes)

    def _activate(self, spec) -> None:
        action = dict(spec.action)
        kind = action.get("type", "")
        if kind == "set":
            self._set_state(action.get("key", spec.state_key), action.get("value", True))
        elif kind == "submit_entry":
            entry_value = self.state.get(action.get("entry_key", ""), "")
            self._set_state(action.get("key", "submitted"), entry_value)
        elif kind == "delayed_set":
            label = self.widgets.get("status")
            if label and isinstance(label, tk.Label):
                label.configure(text="Status: running")
            self.root.after(int(action.get("delay_ms", 300)),
                            lambda: self._finish_delayed(action, label))
        elif kind == "expand_reflow":
            self._set_state(action.get("key", "expanded"), action.get("value", True))
            if spec.id in self.widgets:
                # Deliberately invalidate its old box after the action; a cached
                # target must not be reused without re-grounding.
                self.widgets[spec.id].place_configure(
                    x=spec.x - 24 + 38, y=spec.y - 120 + 12, width=spec.width + 18
                )
            note = self.widgets.get("details_status")
            if note:
                note.configure(text="Details: expanded — layout changed")
        self.root.after_idle(self._refresh_boxes)

    def _finish_delayed(self, action, label):
        self._set_state(action.get("key", "report_ready"), action.get("value", True))
        if label and isinstance(label, tk.Label):
            label.configure(text="Status: report ready")
        self.root.after_idle(self._refresh_boxes)

    def _refresh_boxes(self) -> None:
        boxes: dict[str, dict[str, float]] = {}
        try:
            self.root.update_idletasks()
            for control_id, widget in self.widgets.items():
                if not widget.winfo_exists() or not widget.winfo_viewable():
                    continue
                boxes[control_id] = {
                    "x": float(widget.winfo_rootx()), "y": float(widget.winfo_rooty()),
                    "width": float(widget.winfo_width()), "height": float(widget.winfo_height()),
                }
        except tk.TclError:
            return
        with self.lock:
            self.target_boxes = boxes
            self._published_state = dict(self.state)
            self._published_case_id = self.current_case_id

    def oracle_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "case_id": getattr(self, "_published_case_id", self.current_case_id),
                "state": dict(getattr(self, "_published_state", self.state)),
                "target_boxes": dict(self.target_boxes),
                "captured_at_monotonic": time.monotonic(),
            }


class Handler(BaseHTTPRequestHandler):
    server_version = "AVCSSuiteAFixture/1.0"

    def log_message(self, _format, *_args):
        return

    def _authorized(self) -> bool:
        expected = getattr(self.server, "oracle_token", "")
        supplied = self.headers.get("Authorization", "")
        return bool(expected) and supplied == f"Bearer {expected}"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        app: FixtureWindow = self.server.fixture_app  # type: ignore[attr-defined]
        if path == "/health":
            return self._json(200, {"ready": True, "service": "avcs-suite-a"})
        if path == "/oracle":
            if not self._authorized():
                return self._json(403, {"error": "forbidden"})
            return self._json(200, app.oracle_snapshot())
        return self._json(404, {"error": "not_found"})

    def do_POST(self):
        path = urlparse(self.path).path
        app: FixtureWindow = self.server.fixture_app  # type: ignore[attr-defined]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid_json"})

        if path == "/input/click":
            x = body.get("x")
            y = body.get("y")
            if x is None or y is None:
                return self._json(400, {"error": "missing_coords"})
            done = threading.Event()
            def _do_click():
                try:
                    w = app.root.winfo_containing(int(x), int(y))
                    if w:
                        if hasattr(w, "instate") and w.instate(["disabled"]):
                            pass
                        elif hasattr(w, "invoke"):
                            w.invoke()
                        elif isinstance(w, tk.Checkbutton):
                            w.invoke()
                        elif isinstance(w, ttk.Combobox):
                            vals = w["values"]
                            target_val = body.get("value")
                            if target_val and target_val in vals:
                                w.set(target_val)
                            elif vals:
                                cur = w.get()
                                idx = (list(vals).index(cur) + 1) % len(vals) if cur in vals else 0
                                w.set(vals[idx])
                            w.event_generate("<<ComboboxSelected>>")
                        elif isinstance(w, ttk.Entry):
                            w.focus_set()
                        else:
                            w.event_generate("<Button-1>")
                            w.event_generate("<ButtonRelease-1>")
                finally:
                    done.set()
            app.root.after(0, _do_click)
            done.wait(timeout=2.0)
            return self._json(200, {"clicked": True})

        if path == "/input/type":
            text = body.get("text", "")
            done = threading.Event()
            def _do_type():
                try:
                    focused = app.root.focus_get()
                    if isinstance(focused, ttk.Entry):
                        focused.delete(0, tk.END)
                        focused.insert(0, text)
                        var = app.variables.get("ticket_id")
                        if var:
                            var.set(text)
                finally:
                    done.set()
            app.root.after(0, _do_type)
            done.wait(timeout=2.0)
            return self._json(200, {"typed": True})

        if not self._authorized():
            return self._json(403, {"error": "forbidden"})
        if path == "/admin/reset":
            case_id = body.get("case_id")
            if case_id not in CASE_BY_ID:
                return self._json(400, {"error": "unknown_case"})
            done = threading.Event()
            app.root.after(0, app.load_case, case_id, done)
            if not done.wait(timeout=5.0):
                return self._json(504, {"error": "reset_timeout"})
            return self._json(200, {"reset": True, "case_id": case_id})
        if path == "/admin/shutdown":
            app.root.after(0, app.root.destroy)
            return self._json(200, {"shutdown": True})
        return self._json(404, {"error": "not_found"})


def _enable_per_monitor_v2_dpi_awareness() -> None:
    """Request physical-pixel-consistent fixture geometry before Tk creates HWNDs."""
    if os.name != "nt":
        return
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is the documented pseudo-handle -4.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        # Keep the fixture launchable on older Windows versions; the report must then
        # be interpreted as a non-mixed-DPI run unless the host is calibrated.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def main() -> int:
    _enable_per_monitor_v2_dpi_awareness()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    root = tk.Tk()
    app = FixtureWindow(root)
    app.load_case("button_exact")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.fixture_app = app  # type: ignore[attr-defined]
    server.oracle_token = app.token  # type: ignore[attr-defined]
    app.server = server
    thread = threading.Thread(target=server.serve_forever, name="avcs-fixture-http", daemon=True)
    thread.start()
    # Print one machine-readable readiness event; the runner captures stdout.
    print(json.dumps({"ready": True, "port": server.server_address[1]}), flush=True)
    try:
        root.mainloop()
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
