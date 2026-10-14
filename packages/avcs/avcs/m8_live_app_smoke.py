"""AVCS Milestone 8 Gate M8-B: Phase B Single Action Live Application Smoke Check.

Executes a controlled, non-destructive live interaction against an isolated,
purpose-built test window on the Windows desktop:
- Launches isolated interactive test window in a separate subprocess
- Acquires PID, HWND, and geometry with 1:1 physical Per-Monitor V2 DPI alignment
- Takes verified PRE screenshot and SHA-256 hash
- Confirms PRE state is IDLE via independent test oracle
- Executes one reversible click via Win32CursorDriver / ActionBroker
- Takes verified POST screenshot and SHA-256 hash
- Verifies postcondition independently: checks state.json (click_count == 1, ACTIVATED_STATE)
  and optical visual delta
- Shuts down test window cleanly
- Emits raw trial record conforming to M8B_LIVE_CERTIFICATION_RUNBOOK § 3
"""
from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

if sys.platform == "win32":
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

import numpy as np
from PIL import Image

from avcs.capture_adapter import Win32GDICaptureAdapter
from forge.vision.cortex.motor.win32 import Win32CursorDriver


@dataclass
class PhaseBLiveSmokeResult:
    run_id: str
    trial_id: str
    source_commit: str
    is_dirty: bool
    platform_info: Dict[str, Any]
    app_identity: Dict[str, Any]
    pre_state_verified: bool
    pre_screenshot_sha256: str
    pre_screenshot_path: str
    action_info: Dict[str, Any]
    post_state_verified: bool
    post_screenshot_sha256: str
    post_screenshot_path: str
    visual_delta_detected: bool
    visual_pixel_diff_ratio: float
    latencies_ms: Dict[str, float]
    passed: bool
    error: Optional[str] = None


def get_git_info(repo_root: Path) -> Tuple[str, bool]:
    """Retrieves current git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).strip()
        )
    except Exception:
        dirty = True
    return commit, dirty


def run_phase_b_live_smoke(
    output_dir: str = "m8-live-results", timeout_s: float = 15.0
) -> PhaseBLiveSmokeResult:
    """Executes Phase B live application interaction and independent verification."""
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    commit_sha, is_dirty = get_git_info(repo_root)
    run_id = f"m8-live-smoke-{int(time.time())}"
    trial_id = f"trial-phase-b-{commit_sha[:8]}"

    print("=" * 78)
    print("   AVCS // MILESTONE 8 GATE M8-B: PHASE B LIVE APPLICATION SMOKE CHECK     ")
    print("=" * 78)
    print(f"Commit SHA      : {commit_sha} {'(dirty)' if is_dirty else '(clean)'}")
    print(f"OS Platform     : {platform.system()} {platform.release()} (Build {platform.version()})")
    print(f"Architecture    : {platform.machine()}")
    print(f"Run ID          : {run_id}")

    state_file = evidence_dir / "target_state.json"
    command_file = evidence_dir / "command.json"
    if state_file.is_file():
        state_file.unlink()
    if command_file.is_file():
        command_file.unlink()

    # Step 1: Launch purpose-built test window in subprocess
    print("\n[1/6] Launching isolated interactive test window...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "avcs.testing.interactive_test_window",
            "--state-file",
            str(state_file),
            "--title",
            "AVCS_M8_LIVE_TEST_TARGET",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    app_info: Dict[str, Any] = {}
    latencies: Dict[str, float] = {}

    try:
        # Read the WINDOW_READY announcement line from stdout
        ready_line = proc.stdout.readline().strip()
        if not ready_line or "WINDOW_READY" not in ready_line:
            raise RuntimeError(f"Failed to receive WINDOW_READY from test app: {ready_line}")
        app_info = json.loads(ready_line)
        hwnd = int(app_info["hwnd"])
        print(f"      Target Window Attached: PID={app_info['pid']}, HWND={hwnd}")
        print(f"      Bounds (Physical)     : ({app_info['window_x']}, {app_info['window_y']}, {app_info['window_w']}x{app_info['window_h']})")
        print(f"      Target Button Screen  : ({app_info['target_center_x']}, {app_info['target_center_y']})")

        # Bring window to foreground and settle
        if hwnd and sys.platform == "win32":
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        time.sleep(0.35)

        # Step 2: Verify PRE state independently from state_file
        print("\n[2/6] Verifying independent PRE state...")
        if not state_file.is_file():
            raise RuntimeError("Initial state_file not written by test app")
        pre_state = json.loads(state_file.read_text(encoding="utf-8"))
        pre_ok = (pre_state.get("status") == "IDLE_STATE" and pre_state.get("click_count") == 0)
        print(f"      PRE State Verified     : {pre_ok} (Status: {pre_state.get('status')})")
        if not pre_ok:
            raise RuntimeError(f"Unexpected initial state: {pre_state}")

        # Step 3: Capture PRE screenshot and compute SHA-256 hash
        print("\n[3/6] Capturing PRE-action desktop state...")
        cap = Win32GDICaptureAdapter()
        cap.open()
        t0_cap = time.perf_counter()
        pre_frame = cap.grab()
        t1_cap = time.perf_counter()
        latencies["pre_capture_ms"] = round((t1_cap - t0_cap) * 1000.0, 3)

        pre_sha = hashlib.sha256(pre_frame.tobytes()).hexdigest()
        pre_img_path = evidence_dir / f"{trial_id}_pre.png"
        Image.fromarray(pre_frame).save(pre_img_path)
        print(f"      PRE Capture SHA-256   : {pre_sha}")
        print(f"      PRE Screenshot Saved  : {pre_img_path.name} (Latency: {latencies['pre_capture_ms']} ms)")

        # Step 4: Dispatch Physical Reversible Click to Target Button
        print("\n[4/6] Executing targeted reversible action...")
        driver = Win32CursorDriver(failsafe=False)
        target_x = int(app_info["target_center_x"])
        target_y = int(app_info["target_center_y"])

        t0_act = time.perf_counter()
        driver.move_to_pixels(target_x, target_y)
        time.sleep(0.05)
        driver.click(button="left", hold_duration=0.04)
        t1_act = time.perf_counter()
        latencies["action_dispatch_ms"] = round((t1_act - t0_act) * 1000.0, 3)
        print(f"      Action Dispatched     : Click at ({target_x}, {target_y}) in {latencies['action_dispatch_ms']} ms")

        # Settle 250 ms to allow OS message loop and tkinter repaint
        time.sleep(0.25)
        latencies["settling_ms"] = 250.0

        # Step 5: Capture POST screenshot and compute SHA-256 hash
        print("\n[5/6] Capturing POST-action desktop state...")
        t0_post_cap = time.perf_counter()
        post_frame = cap.grab()
        t1_post_cap = time.perf_counter()
        latencies["post_capture_ms"] = round((t1_post_cap - t0_post_cap) * 1000.0, 3)
        cap.close()

        post_sha = hashlib.sha256(post_frame.tobytes()).hexdigest()
        post_img_path = evidence_dir / f"{trial_id}_post.png"
        Image.fromarray(post_frame).save(post_img_path)
        print(f"      POST Capture SHA-256  : {post_sha}")
        print(f"      POST Screenshot Saved : {post_img_path.name} (Latency: {latencies['post_capture_ms']} ms)")

        # Calculate visual pixel diff over the window bounds
        wx, wy = max(0, app_info["window_x"]), max(0, app_info["window_y"])
        ww = min(app_info["window_w"], post_frame.shape[1] - wx)
        wh = min(app_info["window_h"], post_frame.shape[0] - wy)
        crop_pre = pre_frame[wy : wy + wh, wx : wx + ww]
        crop_post = post_frame[wy : wy + wh, wx : wx + ww]
        diff = np.abs(crop_post.astype(int) - crop_pre.astype(int))
        pixel_diff_ratio = float(np.count_nonzero(diff > 10) / max(1, diff.size))
        visual_delta = pixel_diff_ratio > 0.0005
        print(f"      Visual Delta Detected : {visual_delta} (Diff Ratio: {pixel_diff_ratio:.4%})")

        # Step 6: Verify POST state independently from state_file
        print("\n[6/6] Verifying independent POST state...")
        t0_ver = time.perf_counter()
        post_state = json.loads(state_file.read_text(encoding="utf-8"))
        post_ok = (post_state.get("status") == "ACTIVATED_STATE" and post_state.get("click_count") == 1)
        t1_ver = time.perf_counter()
        latencies["postcondition_check_ms"] = round((t1_ver - t0_ver) * 1000.0, 3)
        print(f"      POST State Verified    : {post_ok} (Status: {post_state.get('status')}, Clicks: {post_state.get('click_count')})")

        passed = bool(pre_ok and post_ok and visual_delta)

        result = PhaseBLiveSmokeResult(
            run_id=run_id,
            trial_id=trial_id,
            source_commit=commit_sha,
            is_dirty=is_dirty,
            platform_info={
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            app_identity=app_info,
            pre_state_verified=pre_ok,
            pre_screenshot_sha256=pre_sha,
            pre_screenshot_path=str(pre_img_path),
            action_info={
                "action_type": "click",
                "button": "left",
                "screen_target_px": [target_x, target_y],
                "dispatch_duration_ms": latencies["action_dispatch_ms"],
            },
            post_state_verified=post_ok,
            post_screenshot_sha256=post_sha,
            post_screenshot_path=str(post_img_path),
            visual_delta_detected=visual_delta,
            visual_pixel_diff_ratio=round(pixel_diff_ratio, 5),
            latencies_ms=latencies,
            passed=passed,
        )

    except Exception as exc:
        print(f"\n[ERROR] Phase B Live Smoke Check FAILED: {exc}")
        result = PhaseBLiveSmokeResult(
            run_id=run_id,
            trial_id=trial_id,
            source_commit=commit_sha,
            is_dirty=is_dirty,
            platform_info={"system": platform.system()},
            app_identity=app_info,
            pre_state_verified=False,
            pre_screenshot_sha256="",
            pre_screenshot_path="",
            action_info={},
            post_state_verified=False,
            post_screenshot_sha256="",
            post_screenshot_path="",
            visual_delta_detected=False,
            visual_pixel_diff_ratio=0.0,
            latencies_ms=latencies,
            passed=False,
            error=str(exc),
        )

    finally:
        # Gracefully signal test window to terminate
        print("\nCleaning up test window subprocess...")
        command_file.write_text(json.dumps({"action": "exit"}), encoding="utf-8")
        try:
            proc.wait(timeout=2.0)
            print("Test window exited cleanly.")
        except subprocess.TimeoutExpired:
            proc.kill()
            print("Test window killed forcefully after timeout.")

    # Save Phase B JSON Report
    report_path = out_dir / "phase_b_live_smoke_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    # Save to live_trials.jsonl formatted trial log conforming to contracts
    trial_record = {
        "run_id": result.run_id,
        "trial_id": result.trial_id,
        "scenario_id": "isolated_control_activation",
        "mode": "hybrid",
        "provenance": "live_application",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": result.source_commit,
        "is_dirty": result.is_dirty,
        "app_identity": {
            "process_name": "python.exe",
            "pid": result.app_identity.get("pid"),
            "hwnd": result.app_identity.get("hwnd"),
            "window_title": result.app_identity.get("title"),
            "bounds": [
                result.app_identity.get("window_x"),
                result.app_identity.get("window_y"),
                result.app_identity.get("window_w"),
                result.app_identity.get("window_h"),
            ],
        },
        "action": result.action_info,
        "postcondition": {
            "check_type": "independent_json_state_and_visual_delta",
            "expected_state": "ACTIVATED_STATE",
            "observed_state": "ACTIVATED_STATE" if result.post_state_verified else "UNKNOWN",
            "verifier_version": "VisualRippleVerifier-v0.2.0",
        },
        "evidence": {
            "pre_screenshot": result.pre_screenshot_path,
            "pre_sha256": result.pre_screenshot_sha256,
            "post_screenshot": result.post_screenshot_path,
            "post_sha256": result.post_screenshot_sha256,
            "state_file_ref": str(state_file),
        },
        "outcome": {
            "status": "confirmed" if result.passed else "failed",
            "verified_success": result.passed,
            "false_confirmation": False,
        },
        "latencies_ms": result.latencies_ms,
    }

    live_trials_file = out_dir / "live_trials.jsonl"
    with open(live_trials_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trial_record) + "\n")

    print("\n" + "=" * 78)
    print(f"Phase B Smoke Report Saved : {report_path}")
    print(f"Live Trial Record Appended : {live_trials_file}")
    print(f"Overall Phase B Status     : {'PASSED' if result.passed else 'FAILED'}")
    print("=" * 78)

    return result


if __name__ == "__main__":
    res = run_phase_b_live_smoke()
    sys.exit(0 if res.passed else 1)
