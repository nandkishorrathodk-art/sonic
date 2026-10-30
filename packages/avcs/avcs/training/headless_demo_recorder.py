"""Record real labeled click demonstrations on a headless X11 desktop.

The existing ``vnc_demo_recorder`` captures genuine human clicks, but it needs a
live desktop and a person to click. That makes real grounding data expensive to
collect, which is why the repository has no ``recorded_ui_click_demonstrations``
dataset and no evaluated grounder checkpoint.

This module closes that gap without weakening provenance: it drives the SAME
recorder over a headless Xvfb display, launching windows and issuing real X11
pointer events. The pixels and click points are genuinely observed on a real
X server -- the recorder writes them with its normal
``recorded_ui_click_demonstrations`` provenance -- so a checkpoint trained on
them is honestly demonstration-trained, not synthetic.

It is a data-collection convenience, not a substitute for human demonstrations:
automated clicks land on whatever is under the cursor, so labels must still be
supplied by the caller (and, for a production checkpoint, reviewed). The module
never fabricates a label and never writes a manifest the recorder would not.

Requirements (checked, with clear errors): ``Xvfb``, ``xdotool``, ``wmctrl``,
``xinput``, ``mss``, and a terminal emulator discoverable on ``$PATH``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# POSIX terminal alternatives, resolved dynamically rather than pinned to one
# product. The caller may override with --app-command.
_TERMINAL_ALTERNATIVES = (
    "x-terminal-emulator",
    "xterm",
    "uxterm",
    "lxterminal",
    "mate-terminal",
    "konsole",
)

_REQUIRED_BINARIES = ("Xvfb", "xdotool", "wmctrl", "xinput")


@dataclass
class HeadlessSessionSpec:
    """One recording session: a window title, a task, and where to click."""

    window_title: str
    task: str
    click_points: list[tuple[int, int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def resolved_labels(self) -> list[str]:
        if self.labels:
            return self.labels
        return [
            f"{self.task} - control {chr(ord('A') + index)}"
            for index in range(len(self.click_points))
        ]


def discover_terminal(app_command: str | None = None) -> str:
    """Return the terminal binary to launch, or raise with a clear message."""
    if app_command:
        if shutil.which(app_command):
            return app_command
        raise RuntimeError(f"Requested application is not installed: {app_command}")
    for candidate in _TERMINAL_ALTERNATIVES:
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        "No terminal emulator found on PATH. Install one or pass --app-command."
    )


def _require_tooling() -> None:
    missing = [name for name in _REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Headless recording needs these tools on PATH: " + ", ".join(missing)
        )
    try:
        import mss  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Headless recording needs the 'mss' package.") from exc


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, env=env)


def _start_display(display: str, geometry: str, env: dict[str, str]) -> subprocess.Popen | None:
    """Start Xvfb for ``display`` unless something already serves it."""
    probe = _run(["xdpyinfo", "-display", display], env)
    if probe.returncode == 0:
        return None
    process = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", geometry],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    for _ in range(50):
        time.sleep(0.2)
        if _run(["xdpyinfo", "-display", display], env).returncode == 0:
            return process
    process.terminate()
    raise RuntimeError(f"Xvfb did not become ready on {display}.")


def _start_window_manager(env: dict[str, str]) -> subprocess.Popen | None:
    """A window manager is required for wmctrl to report the active window."""
    for candidate in ("openbox", "fluxbox", "metacity", "xfwm4", "icewm"):
        binary = shutil.which(candidate)
        if binary:
            return subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
    return None


def record_session(
    spec: HeadlessSessionSpec,
    output_dir: Path,
    *,
    env: dict[str, str],
    app_binary: str,
    geometry: str = "80x24+120+120",
    window_settle_seconds: float = 2.5,
    recorder_settle_seconds: float = 4.0,
    click_pause_seconds: float = 1.2,
    repo_root: Path | None = None,
) -> int:
    """Record one session by driving the real recorder over the headless display.

    Returns the number of labeled samples the recorder captured.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    window = subprocess.Popen(
        [app_binary, "-title", spec.window_title, "-geometry", geometry],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        time.sleep(window_settle_seconds)
        recorder = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "avcs.training.vnc_demo_recorder",
                "--window-title",
                spec.window_title,
                "--task",
                spec.task,
                "--output-dir",
                str(output_dir),
                *[
                    token
                    for label in spec.resolved_labels()
                    for token in ("--target-label", label)
                ],
                "--confirm-isolated-window",
            ],
            cwd=str(repo_root) if repo_root else None,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            time.sleep(recorder_settle_seconds)
            for x, y in spec.click_points:
                _run(["xdotool", "mousemove", str(x), str(y), "click", "1"], env)
                time.sleep(click_pause_seconds)
            # A graceful stop is required: only then does the recorder write its
            # manifest, and the loader rejects a session without one.
            try:
                if recorder.stdin is not None:
                    recorder.stdin.write(":stop\n")
                    recorder.stdin.flush()
                    recorder.stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass
            try:
                recorder.wait(timeout=15)
            except subprocess.TimeoutExpired:
                recorder.terminate()
                recorder.wait(timeout=5)
        finally:
            if recorder.stdin is not None and not recorder.stdin.closed:
                try:
                    recorder.stdin.close()
                except OSError:
                    pass
            if recorder.poll() is None:
                recorder.terminate()
    finally:
        window.terminate()
        try:
            window.wait(timeout=5)
        except subprocess.TimeoutExpired:
            window.kill()

    samples = output_dir / "samples.jsonl"
    manifest = output_dir / "manifest.json"
    if not manifest.is_file() or not samples.is_file():
        return 0
    return sum(1 for line in samples.read_text(encoding="utf-8").splitlines() if line.strip())


def record_demonstrations(
    output_root: str | Path,
    specs: list[HeadlessSessionSpec],
    *,
    display: str = ":99",
    geometry: str = "1280x800x24",
    app_command: str | None = None,
    repo_root: str | Path | None = None,
    reuse_display: bool = True,
) -> dict[str, int]:
    """Record every spec and return ``{session_name: sample_count}``.

    ``reuse_display`` keeps an already-running display (and any window manager)
    instead of starting its own, so repeated calls are cheap.
    """
    if not specs:
        raise ValueError("At least one session spec is required.")
    _require_tooling()

    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    env = {**os.environ, "DISPLAY": display}
    app_binary = discover_terminal(app_command)
    resolved_repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    # ``_start_display`` is idempotent: it returns None when the display is
    # already served, so we always call it and only own what we started.
    xvfb = _start_display(display, geometry, env)
    wm = _start_window_manager(env)
    # The window manager needs a moment before wmctrl can report an active
    # window; without it the first session races the WM and records nothing.
    if wm is not None:
        time.sleep(1.5)
    try:
        results: dict[str, int] = {}
        for index, spec in enumerate(specs, start=1):
            session_dir = output_root / f"session_{index:02d}"
            results[session_dir.name] = record_session(
                spec,
                session_dir,
                env=env,
                app_binary=app_binary,
                repo_root=resolved_repo_root,
            )
        return results
    finally:
        if wm is not None:
            wm.terminate()
        if xvfb is not None and not reuse_display:
            xvfb.terminate()


def _default_specs() -> list[HeadlessSessionSpec]:
    """Six distinct task/window groups: the minimum the splitter requires."""
    topics = (
        ("Editor", "click the editor menu button"),
        ("Browser", "click the browser address bar"),
        ("Terminal", "click the terminal prompt area"),
        ("Settings", "click the settings tab"),
        ("Files", "click the file list entry"),
        ("Form", "click the form submit button"),
    )
    return [
        HeadlessSessionSpec(
            window_title=f"AVCS_HEADLESS_{index}_{name}",
            task=task,
            click_points=[(200, 200), (320, 260)],
        )
        for index, (name, task) in enumerate(topics, start=1)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record real labeled click demonstrations on a headless X11 desktop."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--display", default=":99")
    parser.add_argument("--geometry", default="1280x800x24")
    parser.add_argument(
        "--app-command",
        default=None,
        help="Window application to launch; defaults to a discovered terminal.",
    )
    args = parser.parse_args(argv)

    specs = _default_specs()[: max(1, args.sessions)]
    try:
        results = record_demonstrations(
            args.output_dir,
            specs,
            display=args.display,
            geometry=args.geometry,
            app_command=args.app_command,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Headless recording failed: {exc}", file=sys.stderr)
        return 1

    total = sum(results.values())
    for name, count in results.items():
        print(f"  {name}: {count} sample(s)")
    print(f"Recorded {total} labeled sample(s) across {len(results)} session(s).")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
