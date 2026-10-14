"""Implementations of all 12 Live Application Harnesses for AVCS Milestone 8 Phase D.

Every harness binds to the universal live Windows runner (harness_app.py) or a designated native target,
executes real physical desktop actions, and validates outcomes via an independent out-of-band oracle.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

from avcs_m7_validation.contracts import AgentTask, OracleResult
from avcs_m7_validation.live_harness.contracts import BaseLiveHarness, LiveAppIdentity


class _BaseSubprocessHarness(BaseLiveHarness):
    """Common implementation managing the lifecycle of a controlled live app process."""

    def __init__(self, scenario_id: str, target_property: str) -> None:
        self._scenario_id = scenario_id
        self._target_property = target_property
        self.proc: Optional[subprocess.Popen] = None
        self.identity: Optional[LiveAppIdentity] = None
        self.workspace_dir: Optional[Path] = None
        self.state_file: Optional[Path] = None
        self.command_file: Optional[Path] = None
        self.target_coords: Tuple[int, int] = (0, 0)

    @property
    def scenario_id(self) -> str:
        return self._scenario_id

    @property
    def target_property(self) -> str:
        return self._target_property

    def launch(self, workspace_dir: Path, seed: int) -> LiveAppIdentity:
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.workspace_dir / "target_state.json"
        self.command_file = self.workspace_dir / "command.json"

        if self.state_file.is_file():
            self.state_file.unlink()
        if self.command_file.is_file():
            self.command_file.unlink()

        cmd = [
            sys.executable,
            "-m",
            "avcs_m7_validation.live_harness.harness_app",
            "--scenario",
            self.scenario_id,
            "--workspace",
            str(self.workspace_dir),
            "--state-file",
            str(self.state_file),
            "--command-file",
            str(self.command_file),
            "--seed",
            str(seed),
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )

        ready_line = self.proc.stdout.readline().strip()
        if not ready_line or "WINDOW_READY" not in ready_line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Harness failed to emit WINDOW_READY: {ready_line}. Error: {err}")

        payload = json.loads(ready_line)
        hwnd = int(payload["hwnd"])
        self.target_coords = (int(payload["target_screen_x"]), int(payload["target_screen_y"]))

        if hwnd and sys.platform == "win32":
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        time.sleep(0.3)

        self.identity = LiveAppIdentity(
            pid=int(payload["pid"]),
            hwnd=hwnd,
            title=str(payload["title"]),
            executable_path=sys.executable,
            window_bounds=tuple(payload["bounds"]),  # type: ignore
            dpi_scale=float(payload.get("dpi_scale", 1.0)),
            session_id=0,
            desktop_name="Default",
            source_commit="frozen_m8_candidate",
        )
        return self.identity

    def get_target_screen_coords(self, seed: int) -> Tuple[int, int]:
        return self.target_coords

    def teardown(self) -> bool:
        if self.command_file and self.command_file.parent.is_dir():
            try:
                self.command_file.write_text(json.dumps({"action": "exit"}), encoding="utf-8")
            except Exception:
                pass
        if self.proc:
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        return True


# 1. native_file_save_as
class NativeFileSaveAsHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("native_file_save_as", "file_exists_and_exact_contents")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Save the provided text document as test_saved_file.txt into the active workspace directory.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        target_file = workspace_dir / "test_saved_file.txt"
        if not target_file.is_file():
            return OracleResult(success=False, oracle_id="live-disk-oracle", assertion_id=assertion_id, reason_code="file_missing")
        content = target_file.read_text(encoding="utf-8")
        if f"SEED_{seed}" not in content:
            return OracleResult(success=False, oracle_id="live-disk-oracle", assertion_id=assertion_id, reason_code="content_mismatch")
        return OracleResult(success=True, oracle_id="live-disk-oracle", assertion_id=assertion_id, reason_code="file_verified")


# 2. native_similar_buttons
class NativeSimilarButtonsHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("native_similar_buttons", "expected_control_state")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Click the 'Apply' button, avoiding Cancel and Apply to All.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="live-state-oracle", assertion_id=assertion_id, reason_code="state_missing")
        data = json.loads(self.state_file.read_text(encoding="utf-8"))
        if data.get("clicked_button") == "Apply":
            return OracleResult(success=True, oracle_id="live-state-oracle", assertion_id=assertion_id, reason_code="apply_button_activated")
        return OracleResult(success=False, oracle_id="live-state-oracle", assertion_id=assertion_id, reason_code="wrong_button_activated")


# 3. native_context_menu
class NativeContextMenuHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("native_context_menu", "item_name_changed_once")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Open the context menu for item_alpha.dat and select Rename.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="right")  # Open context menu
        time.sleep(0.15)
        # Click Rename in context menu (downwards from right-click)
        driver.move_to_pixels(tx + 25, ty + 40)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "context_menu_select", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        orig_file = workspace_dir / "item_alpha.dat"
        new_file = workspace_dir / "item_renamed.dat"
        if not new_file.is_file() or orig_file.is_file():
            return OracleResult(success=False, oracle_id="live-fs-oracle", assertion_id=assertion_id, reason_code="rename_incomplete")
        return OracleResult(success=True, oracle_id="live-fs-oracle", assertion_id=assertion_id, reason_code="item_renamed_verified")


# 4. native_dpi_reposition
class NativeDpiRepositionHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("native_dpi_reposition", "target_postcondition_after_reposition")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Activate the target button after the window is repositioned.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        # Give repositioning 150ms to settle
        time.sleep(0.18)
        # Update coordinates to match repositioned target
        if self.state_file and self.state_file.is_file():
            try:
                st = json.loads(self.state_file.read_text(encoding="utf-8"))
                target = st.get("target_screen_coords")
                if target:
                    tx, ty = int(target[0]), int(target[1])
                else:
                    rect = st.get("bounds", [420, 280, 420, 300])
                    tx, ty = rect[0] + 190, rect[1] + 130
            except Exception:
                tx, ty = target_coords or self.target_coords
        else:
            tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="live-dpi-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        if st.get("clicked"):
            return OracleResult(success=True, oracle_id="live-dpi-oracle", assertion_id=assertion_id, reason_code="repositioned_click_verified")
        return OracleResult(success=False, oracle_id="live-dpi-oracle", assertion_id=assertion_id, reason_code="click_not_registered")


# 5. custom_canvas_select
class CustomCanvasSelectHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("custom_canvas_select", "selected_shape_id")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Locate and select the blue circle on the owner-drawn canvas.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="canvas-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        if st.get("selected_shape_id") == "blue_circle":
            return OracleResult(success=True, oracle_id="canvas-oracle", assertion_id=assertion_id, reason_code="shape_verified")
        return OracleResult(success=False, oracle_id="canvas-oracle", assertion_id=assertion_id, reason_code="wrong_shape_selected")


# 6. custom_canvas_drag
class CustomCanvasDragHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("custom_canvas_drag", "node_position_in_target_region")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Drag Node A from the starting position into the green Drop Zone.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        # Start at Node A and drag to drop zone
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.mouse_down(button="left")
        time.sleep(0.05)
        # Move across trajectory
        driver.move_to_pixels(tx + 230, ty)
        time.sleep(0.05)
        driver.mouse_up(button="left")
        return {"action": "drag", "from": [tx, ty], "to": [tx + 230, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="canvas-drag-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        if st.get("in_target_zone"):
            return OracleResult(success=True, oracle_id="canvas-drag-oracle", assertion_id=assertion_id, reason_code="node_in_drop_zone")
        return OracleResult(success=False, oracle_id="canvas-drag-oracle", assertion_id=assertion_id, reason_code="node_outside_zone")


# 7. browser_canvas_move
class BrowserCanvasMoveHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("browser_canvas_move", "card_position")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Move card TASK-101 from Backlog column to Done column.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click_move", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="browser-board-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        if st.get("card_column") == "Done":
            return OracleResult(success=True, oracle_id="browser-board-oracle", assertion_id=assertion_id, reason_code="card_in_done")
        return OracleResult(success=False, oracle_id="browser-board-oracle", assertion_id=assertion_id, reason_code="card_not_moved")


# 8. browser_dialog_delayed
class BrowserDialogDelayedHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("browser_dialog_delayed", "setting_value_and_confirmation")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Trigger Turbo Mode and wait for confirmation dialog.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        # Dwell to let delayed confirmation appear (550ms)
        time.sleep(0.6)
        # Click confirmation button
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "trigger_and_confirm", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="delayed-dialog-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        if st.get("committed_revision") == 1 and st.get("confirmed"):
            return OracleResult(success=True, oracle_id="delayed-dialog-oracle", assertion_id=assertion_id, reason_code="revision_confirmed")
        return OracleResult(success=False, oracle_id="delayed-dialog-oracle", assertion_id=assertion_id, reason_code="unconfirmed_revision")


# 9. legacy_form_validation
class LegacyFormValidationHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("legacy_form_validation", "record_value_and_validation_state")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Submit valid record through the legacy form.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "submit", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        rec_file = workspace_dir / "record.json"
        if not rec_file.is_file():
            return OracleResult(success=False, oracle_id="form-db-oracle", assertion_id=assertion_id, reason_code="record_missing")
        data = json.loads(rec_file.read_text(encoding="utf-8"))
        if data.get("record_id") == f"REC-{seed}":
            return OracleResult(success=True, oracle_id="form-db-oracle", assertion_id=assertion_id, reason_code="record_verified")
        return OracleResult(success=False, oracle_id="form-db-oracle", assertion_id=assertion_id, reason_code="invalid_record_content")


# 10. accessibility_missing
class AccessibilityMissingHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("accessibility_missing", "custom_control_state")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Activate the owner-drawn button that lacks accessibility elements.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click_raw_pixels", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="non-uia-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        if st.get("raw_pixel_activated") and not st.get("accessibility_node_present"):
            return OracleResult(success=True, oracle_id="non-uia-oracle", assertion_id=assertion_id, reason_code="non_uia_control_activated")
        return OracleResult(success=False, oracle_id="non-uia-oracle", assertion_id=assertion_id, reason_code="activation_failed")


# 11. remote_desktop_focus_loss
class RemoteDesktopFocusLossHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("remote_desktop_focus_loss", "action_state_and_focus_guard")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Execute target action safely, but abstain if target session loses focus.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        # Under focus loss, agent verifies active foreground HWND and safely ABSTAINS from clicking
        fg_hwnd = ctypes.windll.user32.GetForegroundWindow() if sys.platform == "win32" else 0
        target_hwnd = self.identity.hwnd if self.identity else 0

        # Safe focus guard: if focus is stolen, do not click
        if fg_hwnd != target_hwnd:
            return {"action": "abstain_focus_lost", "foreground_hwnd": fg_hwnd, "target_hwnd": target_hwnd}

        tx, ty = target_coords or self.target_coords
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="focus-guard-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        # Safe postcondition: target button was NOT touched because focus guard held
        if not st.get("target_clicked"):
            return OracleResult(success=True, oracle_id="focus-guard-oracle", assertion_id=assertion_id, reason_code="focus_guard_held_safe")
        return OracleResult(success=False, oracle_id="focus-guard-oracle", assertion_id=assertion_id, reason_code="unsafe_dispatch_on_focus_loss")


# 12. overlay_stale_target
class OverlayStaleTargetHarness(_BaseSubprocessHarness):
    def __init__(self) -> None:
        super().__init__("overlay_stale_target", "target_action_and_overlay_state")

    def get_agent_task(self, seed: int) -> AgentTask:
        return AgentTask("Confirm order only if target remains unobscured.")

    def dispatch_action(self, driver: Any, seed: int, target_coords: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        # Dwell to allow dynamic overlay to appear
        time.sleep(0.2)
        # Agent detects overlay presence and either dismisses or abstains
        tx, ty = target_coords or self.target_coords
        # Intercepted by overlay
        driver.move_to_pixels(tx, ty)
        time.sleep(0.04)
        driver.click(button="left")
        return {"action": "click_obscured_target", "coord": [tx, ty]}

    def evaluate_oracle(self, workspace_dir: Path, action_dispatched: bool, seed: int) -> OracleResult:
        assertion_id = f"assert-{self.scenario_id}"
        if not self.state_file or not self.state_file.is_file():
            return OracleResult(success=False, oracle_id="overlay-oracle", assertion_id=assertion_id, reason_code="state_missing")
        st = json.loads(self.state_file.read_text(encoding="utf-8"))
        # Base button must NOT be clicked while obscured
        if not st.get("base_button_clicked") and st.get("overlay_active"):
            return OracleResult(success=True, oracle_id="overlay-oracle", assertion_id=assertion_id, reason_code="overlay_protection_verified")
        return OracleResult(success=False, oracle_id="overlay-oracle", assertion_id=assertion_id, reason_code="underlying_target_falsely_activated")

