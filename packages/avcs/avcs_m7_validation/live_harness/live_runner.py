"""Live Scenario Runner executing empirical Windows interactions for Milestone 8 Phase D."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from avcs_m7_validation.contracts import (
    ActionStatus,
    EvidenceProvenance,
    Mode,
    OracleResult,
    OutcomeCategory,
    TrialResult,
)
from avcs_m7_validation.live_harness.contracts import BaseLiveHarness, LiveExecutionTrace
from avcs_m7_validation.live_harness.registry import get_live_harness
from avcs.capture_adapter import Win32GDICaptureAdapter
from forge.vision.cortex.motor.win32 import Win32CursorDriver


class LiveScenarioRunner:
    """Executes single-trial or batch physical Windows desktop interactions with full evidence tracking."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.evidence_dir = output_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.driver = Win32CursorDriver(failsafe=False)

    def run_live_trial(
        self,
        scenario_id: str,
        run_id: str,
        mode: Mode = Mode.VISUAL_ONLY,
        seed: int = 42,
        settle_delay_s: float = 0.25,
    ) -> Tuple[TrialResult, LiveExecutionTrace]:
        """Runs a complete physical trial against the designated live application."""
        t_start = time.perf_counter()
        trial_workspace = self.output_dir / "workspaces" / f"ws_{run_id}"
        trial_workspace.mkdir(parents=True, exist_ok=True)

        harness = get_live_harness(scenario_id)

        # 1. Launch Process & Acquire Handles
        identity = harness.launch(trial_workspace, seed)

        cap = Win32GDICaptureAdapter()
        cap.open()
        t_cap_start = time.perf_counter()
        pre_frame = cap.grab()
        t_cap_ms = (time.perf_counter() - t_cap_start) * 1000.0
        pre_sha = hashlib.sha256(pre_frame.tobytes()).hexdigest()

        # Save PRE screenshot
        pre_img_path = self.evidence_dir / f"{run_id}_pre.png"
        try:
            from PIL import Image
            Image.fromarray(pre_frame).save(pre_img_path)
        except Exception:
            pass

        # 2. Dispatch Physical Action via SendInput (isolated worker thread for pristine desktop context)
        t_action_start = time.perf_counter()
        target_coords = harness.get_target_screen_coords(seed)
        action_meta: Dict[str, Any] = {}
        disp_exc: Optional[Exception] = None

        import threading
        import ctypes
        def _th_dispatch() -> None:
            nonlocal action_meta, disp_exc
            if sys.platform == "win32":
                try:
                    h_desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
                    if h_desk:
                        ctypes.windll.user32.SetThreadDesktop(h_desk)
                except Exception:
                    pass
            try:
                action_meta = harness.dispatch_action(self.driver, seed, target_coords)
            except Exception as exc:
                disp_exc = exc

        disp_th = threading.Thread(target=_th_dispatch, name="AVCS-LiveDispatchWorker")
        disp_th.start()
        disp_th.join(timeout=5.0)
        if disp_exc is not None:
            raise disp_exc
        t_action_ms = (time.perf_counter() - t_action_start) * 1000.0

        # Settle Delay
        time.sleep(settle_delay_s)
        settle_ms = settle_delay_s * 1000.0

        # 3. Post-Action Capture
        post_frame = cap.grab()
        cap.close()
        post_sha = hashlib.sha256(post_frame.tobytes()).hexdigest()

        post_img_path = self.evidence_dir / f"{run_id}_post.png"
        try:
            from PIL import Image
            Image.fromarray(post_frame).save(post_img_path)
        except Exception:
            pass

        # Compute Visual Pixel Difference Ratio
        diff = np.abs(post_frame.astype(int) - pre_frame.astype(int))
        pixel_diff_ratio = float(np.count_nonzero(diff > 10) / max(1, diff.size))

        # 4. Evaluate Independent Postcondition Oracle
        t_oracle_start = time.perf_counter()
        oracle_res = harness.evaluate_oracle(trial_workspace, action_dispatched=True, seed=seed)
        t_oracle_ms = (time.perf_counter() - t_oracle_start) * 1000.0

        # 5. Clean Teardown
        cleanup_ok = harness.teardown()
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        # Determine Outcome Status
        status = ActionStatus.CONFIRMED if oracle_res.success else ActionStatus.FAILED
        verified_success = bool(oracle_res.success)
        false_confirmation = bool(status == ActionStatus.CONFIRMED and not oracle_res.success)

        trace = LiveExecutionTrace(
            run_id=run_id,
            scenario_id=scenario_id,
            mode=mode,
            seed=seed,
            app_identity=identity,
            pre_screenshot_path=str(pre_img_path),
            pre_screenshot_sha256=pre_sha,
            post_screenshot_path=str(post_img_path),
            post_screenshot_sha256=post_sha,
            visual_pixel_diff_ratio=round(pixel_diff_ratio, 5),
            target_coord_screen=target_coords,
            dispatched_action=action_meta.get("action", "click"),
            capture_latency_ms=round(t_cap_ms, 2),
            action_latency_ms=round(t_action_ms, 2),
            settle_latency_ms=round(settle_ms, 2),
            oracle_latency_ms=round(t_oracle_ms, 2),
            oracle_result=oracle_res,
            status=status,
            verified_success=verified_success,
            false_confirmation=false_confirmation,
            cleanup_successful=cleanup_ok,
        )

        trial_result = TrialResult(
            run_id=run_id,
            scenario_id=scenario_id,
            mode=mode,
            seed=seed,
            status=status,
            oracle_success=oracle_res.success,
            oracle_valid=oracle_res.valid,
            reported_confirmed=(status == ActionStatus.CONFIRMED),
            verified_success=verified_success,
            false_confirmation=false_confirmation,
            elapsed_ms=round(t_total_ms, 2),
            action_count=1,
            evidence_method="physical_gdi_and_sendinput",
            category=OutcomeCategory.TASK_SUCCESS if verified_success else OutcomeCategory.TASK_FAILURE,
            oracle_id=oracle_res.oracle_id,
            assertion_id=oracle_res.assertion_id,
            detail_code=oracle_res.reason_code,
            provenance=EvidenceProvenance.LIVE_APPLICATION,
            app_identity=identity.to_dict(),
            action_latency_ms=round(t_action_ms, 2),
            postcondition_evidence_ref=str(self.evidence_dir / f"{run_id}_post.png"),
        )

        return trial_result, trace

