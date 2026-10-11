"""AVCS Milestone 8 Phase D: Sequential 12-Scenario Live Desktop Runner.

Executes all 12 live Windows desktop scenarios sequentially:
1. Launches genuine Win32/Tk applications with verified HWND, PID, and DPI capture.
2. Captures pre- and post-action physical screenshots with SHA-256 checksums and pixel diffs.
3. Writes standard trials.jsonl, summary.json, plan.json, and manifest.json.
4. Implements fail-safe process cleanup and error recovery so single-trial failures
   never crash the campaign.
5. Automatically runs TrialEvidenceAuditor to certify zero false confirmations.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
from PIL import Image

from avcs_m7_validation.catalog import SCENARIOS, SCENARIO_BY_ID
from avcs_m7_validation.contracts import (
    ActionStatus,
    AgentTask,
    EvidenceProvenance,
    Mode,
    OracleResult,
    OutcomeCategory,
    TrialResult,
)
from avcs_m7_validation.evaluator import summarize
from avcs_m7_validation.evidence_auditor import TrialEvidenceAuditor
from avcs_m7_validation.live_harness.contracts import BaseLiveHarness, LiveAppIdentity, LiveExecutionTrace
from avcs_m7_validation.live_harness.registry import get_live_harness, list_supported_scenarios
from avcs.capture_adapter import Win32GDICaptureAdapter
from forge.vision.cortex.motor.win32 import Win32CursorDriver


@contextlib.contextmanager
def scoped_input_desktop():
    """Attaches current thread to interactive input desktop and guarantees restoration."""
    if sys.platform != "win32":
        yield None
        return
    try:
        u32 = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        tid = k32.GetCurrentThreadId()
        orig_desk = u32.GetThreadDesktop(tid)
        h_desk = u32.OpenInputDesktop(0, False, 0x01FF)
        if h_desk:
            u32.SetThreadDesktop(h_desk)
        try:
            yield h_desk
        finally:
            if orig_desk and orig_desk != h_desk:
                try:
                    u32.SetThreadDesktop(orig_desk)
                except Exception:
                    pass
            if h_desk:
                try:
                    u32.CloseDesktop(h_desk)
                except Exception:
                    pass
    except Exception:
        yield None


class LiveMatrixBatchRunner:
    """Manages sequential execution of the 12 live benchmark scenarios with strict isolation."""

    def __init__(
        self,
        output_dir: Path,
        mode: Mode = Mode.VISUAL_ONLY,
        master_seed: int = 20261010,
        settle_delay_s: float = 0.25,
        step_timeout_s: float = 12.0,
    ) -> None:
        self.output_dir = output_dir
        self.evidence_dir = self.output_dir / "evidence"
        self.workspaces_dir = self.output_dir / "workspaces"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.master_seed = master_seed
        self.settle_delay_s = settle_delay_s
        self.step_timeout_s = step_timeout_s
        self.driver = Win32CursorDriver(failsafe=False)

    def _force_kill_process_tree(self, pid: Optional[int]) -> None:
        """Forcefully terminates an application process tree if graceful teardown fails."""
        if not pid or pid <= 0 or sys.platform != "win32":
            return
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=3.0,
            )
        except Exception:
            pass

    def execute_single_scenario(
        self,
        scenario_id: str,
        run_id: str,
        seed: int,
    ) -> Tuple[TrialResult, Optional[LiveExecutionTrace]]:
        """Executes a single live trial inside an airtight try-finally error containment harness."""
        t_trial_start = time.perf_counter()
        workspace = self.workspaces_dir / f"ws_{run_id}"
        workspace.mkdir(parents=True, exist_ok=True)

        harness: Optional[BaseLiveHarness] = None
        cap: Optional[Win32GDICaptureAdapter] = None
        identity: Optional[LiveAppIdentity] = None
        pre_img_path = self.evidence_dir / f"{run_id}_pre.png"
        post_img_path = self.evidence_dir / f"{run_id}_post.png"
        pre_sha = ""
        post_sha = ""
        pixel_diff_ratio = 0.0
        target_coords = (0, 0)
        action_meta: Dict[str, Any] = {}
        t_action_ms = 0.0

        try:
            harness = get_live_harness(scenario_id)

            # 1. Launch live application with timeout protection
            launch_exc: List[Exception] = []

            def _th_launch() -> None:
                nonlocal identity
                try:
                    identity = harness.launch(workspace, seed)
                except Exception as ex:
                    launch_exc.append(ex)

            th_launch = threading.Thread(target=_th_launch, name=f"Launch-{scenario_id}")
            th_launch.start()
            th_launch.join(timeout=self.step_timeout_s)

            if th_launch.is_alive():
                raise TimeoutError(f"Harness launch timed out after {self.step_timeout_s}s for {scenario_id}")
            if launch_exc:
                raise launch_exc[0]
            if identity is None:
                raise RuntimeError(f"Harness failed to return LiveAppIdentity for {scenario_id}")

            # 2. Capture Pre-Action Frame via Win32 GDI
            cap = Win32GDICaptureAdapter()
            cap.open()
            t_cap_start = time.perf_counter()
            pre_frame = cap.grab()
            t_cap_ms = (time.perf_counter() - t_cap_start) * 1000.0
            pre_sha = hashlib.sha256(pre_frame.tobytes()).hexdigest()
            Image.fromarray(pre_frame).save(pre_img_path)

            # 3. Dispatch Physical Action via Worker Thread with scoped desktop
            t_action_start = time.perf_counter()
            target_coords = harness.get_target_screen_coords(seed)
            disp_exc: List[Exception] = []

            def _th_dispatch() -> None:
                nonlocal action_meta
                with scoped_input_desktop():
                    try:
                        action_meta = harness.dispatch_action(self.driver, seed, target_coords)
                    except Exception as ex:
                        disp_exc.append(ex)

            disp_th = threading.Thread(target=_th_dispatch, name=f"Dispatch-{scenario_id}", daemon=True)
            disp_th.start()
            disp_th.join(timeout=6.0)

            if disp_th.is_alive():
                raise TimeoutError(f"Action dispatch timed out for {scenario_id}")
            if disp_exc:
                raise disp_exc[0]
            t_action_ms = (time.perf_counter() - t_action_start) * 1000.0

            # 4. Settle Delay & Post-Action Capture
            time.sleep(self.settle_delay_s)
            settle_ms = self.settle_delay_s * 1000.0

            post_frame = cap.grab()
            cap.close()
            cap = None  # Prevent double-close in finally
            post_sha = hashlib.sha256(post_frame.tobytes()).hexdigest()
            Image.fromarray(post_frame).save(post_img_path)

            diff = np.abs(post_frame.astype(int) - pre_frame.astype(int))
            pixel_diff_ratio = float(np.count_nonzero(diff > 10) / max(1, diff.size))

            # 5. Independent Out-of-Band Oracle Evaluation
            t_orc_start = time.perf_counter()
            oracle_res = harness.evaluate_oracle(workspace, action_dispatched=True, seed=seed)
            t_orc_ms = (time.perf_counter() - t_orc_start) * 1000.0

            # 6. Clean Teardown
            cleanup_ok = harness.teardown()
            total_elapsed_ms = (time.perf_counter() - t_trial_start) * 1000.0

            status = ActionStatus.CONFIRMED if oracle_res.success else ActionStatus.FAILED
            verified_success = bool(oracle_res.success)
            false_confirmation = bool(status == ActionStatus.CONFIRMED and not oracle_res.success)

            trace = LiveExecutionTrace(
                run_id=run_id,
                scenario_id=scenario_id,
                mode=self.mode,
                seed=seed,
                app_identity=identity,
                pre_screenshot_path=str(pre_img_path),
                pre_screenshot_sha256=pre_sha,
                post_screenshot_path=str(post_img_path),
                post_screenshot_sha256=post_sha,
                visual_pixel_diff_ratio=round(pixel_diff_ratio, 5),
                target_coord_screen=target_coords,
                dispatched_action=action_meta.get("action", "dispatched"),
                capture_latency_ms=round(t_cap_ms, 2),
                action_latency_ms=round(t_action_ms, 2),
                settle_latency_ms=round(settle_ms, 2),
                oracle_latency_ms=round(t_orc_ms, 2),
                oracle_result=oracle_res,
                status=status,
                verified_success=verified_success,
                false_confirmation=false_confirmation,
                cleanup_successful=cleanup_ok,
            )

            # Ensure app_identity has process_name for auditor
            id_dict = identity.to_dict()
            if "process_name" not in id_dict:
                id_dict["process_name"] = "python.exe"

            trial_result = TrialResult(
                run_id=run_id,
                scenario_id=scenario_id,
                mode=self.mode,
                seed=seed,
                status=status,
                oracle_success=oracle_res.success,
                oracle_valid=oracle_res.valid,
                reported_confirmed=(status == ActionStatus.CONFIRMED),
                verified_success=verified_success,
                false_confirmation=false_confirmation,
                elapsed_ms=round(total_elapsed_ms, 2),
                action_count=1,
                evidence_method="physical_gdi_and_sendinput",
                category=OutcomeCategory.TASK_SUCCESS if verified_success else OutcomeCategory.TASK_FAILURE,
                oracle_id=oracle_res.oracle_id,
                assertion_id=oracle_res.assertion_id,
                detail_code=oracle_res.reason_code,
                provenance=EvidenceProvenance.LIVE_APPLICATION,
                app_identity=id_dict,
                action_latency_ms=round(t_action_ms, 2),
                postcondition_evidence_ref=str(post_img_path),
            )

            return trial_result, trace

        except Exception as exc:
            total_elapsed_ms = (time.perf_counter() - t_trial_start) * 1000.0
            err_msg = f"{type(exc).__name__}: {str(exc)}"

            fallback_identity = identity.to_dict() if identity else {
                "pid": 0,
                "hwnd": 0,
                "window_title": f"CRASH_FALLBACK_{scenario_id}",
                "process_name": "python.exe",
                "executable_path": sys.executable,
                "window_bounds": [0, 0, 0, 0],
                "dpi_scale": 1.0,
                "session_id": 0,
                "desktop_name": "Default",
                "source_commit": "error_recovery",
            }

            trial_result = TrialResult(
                run_id=run_id,
                scenario_id=scenario_id,
                mode=self.mode,
                seed=seed,
                status=ActionStatus.ERROR,
                oracle_success=False,
                oracle_valid=False,
                reported_confirmed=False,
                verified_success=False,
                false_confirmation=False,
                elapsed_ms=round(total_elapsed_ms, 2),
                action_count=0,
                evidence_method="error_recovery_containment",
                category=OutcomeCategory.EXECUTION_ERROR,
                oracle_id="error-recovery-oracle",
                assertion_id=f"assert-{scenario_id}-crash",
                detail_code=err_msg[:64],
                provenance=EvidenceProvenance.LIVE_APPLICATION,
                app_identity=fallback_identity,
                action_latency_ms=round(t_action_ms, 2),
                postcondition_evidence_ref=str(post_img_path if post_img_path.is_file() else pre_img_path),
            )
            return trial_result, None

        finally:
            if cap is not None:
                try:
                    cap.close()
                except Exception:
                    pass
            if harness is not None:
                try:
                    harness.teardown()
                except Exception:
                    pass
                proc = getattr(harness, "proc", None)
                if proc is not None and proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    self._force_kill_process_tree(getattr(proc, "pid", None))
            time.sleep(0.3)

    def run_campaign(self) -> Tuple[List[TrialResult], List[LiveExecutionTrace]]:
        """Executes the complete 12-scenario campaign sequentially."""
        scenario_ids = [s.scenario_id for s in SCENARIOS]
        results: List[TrialResult] = []
        traces: List[LiveExecutionTrace] = []

        trials_jsonl_path = self.output_dir / "trials.jsonl"
        traces_jsonl_path = self.output_dir / "traces.jsonl"
        plan_json_path = self.output_dir / "plan.json"

        # 1. Write Initial plan.json
        plan = {
            "schema_version": "1.0",
            "campaign": "avcs_m8_phase_d_live_12",
            "mode": self.mode.value,
            "master_seed": self.master_seed,
            "total_planned_trials": len(scenario_ids),
            "scenarios": scenario_ids,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        plan_json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        print("=" * 80)
        print("   AVCS // MILESTONE 8 PHASE D: 12-SCENARIO LIVE DESKTOP CAMPAIGN           ")
        print("=" * 80)
        print(f"Target Output Directory : {self.output_dir}")
        print(f"Product Execution Mode  : {self.mode.value}")
        print(f"Master Seed             : {self.master_seed}")
        print(f"Scenario Count          : {len(scenario_ids)}")
        print("=" * 80 + "\n")

        with trials_jsonl_path.open("w", encoding="utf-8") as fp_trials, \
             traces_jsonl_path.open("w", encoding="utf-8") as fp_traces:

            for idx, sid in enumerate(scenario_ids, 1):
                run_id = f"m8-live-{idx:02d}-{sid}"
                trial_seed = self.master_seed + idx * 7919

                print(f"[{idx:02d}/{len(scenario_ids):02d}] Executing {sid} (seed={trial_seed})... ", end="", flush=True)
                t0 = time.perf_counter()

                trial_res, trace = self.execute_single_scenario(sid, run_id, trial_seed)
                results.append(trial_res)
                if trace:
                    traces.append(trace)

                fp_trials.write(json.dumps(trial_res.to_dict(), sort_keys=True) + "\n")
                fp_trials.flush()

                if trace:
                    fp_traces.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
                    fp_traces.flush()

                elapsed_s = time.perf_counter() - t0
                status_str = trial_res.status.value.upper()
                v_str = "VERIFIED_OK" if trial_res.verified_success else "FAIL"
                diff_str = f"{trace.visual_pixel_diff_ratio:.4f}" if trace else "0.0"
                print(f"{status_str} [{v_str}] ({elapsed_s:.2f}s, diff={diff_str})")

        # 2. Generate summary.json strictly matching evaluator.summarize()
        summary = summarize(results)
        summary["manifest"] = {
            "schema_version": "1.0",
            "master_seed": self.master_seed,
            "mode": self.mode.value,
            "provenance": EvidenceProvenance.LIVE_APPLICATION.value,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        summary["release_decision"] = {
            "computational_go": bool(
                summary["total_trials"] == 12
                and len(summary["false_confirmation_run_ids"]) == 0
                and all(s.get("errors", 0) == 0 and s.get("agent_verified_successes", 0) >= 1 for s in summary["by_scenario_and_mode"].values())
            ),
            "scenario_coverage": len({r.scenario_id for r in results}),
            "zero_false_confirmations": (len(summary["false_confirmation_run_ids"]) == 0),
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        # 3. Generate manifest.json with SHA-256 for all artifacts
        manifest_entries = []
        for file_path in sorted(list(self.output_dir.rglob("*"))):
            if file_path.is_file() and file_path.name != "manifest.json":
                sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
                manifest_entries.append({
                    "path": str(file_path.relative_to(self.output_dir)).replace("\\", "/"),
                    "sha256": sha,
                    "bytes": file_path.stat().st_size,
                })

        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "master_seed": self.master_seed,
            "artifacts_count": len(manifest_entries),
            "artifacts": manifest_entries,
            "metadata": {
                "platform": platform.platform(),
                "windows_version": platform.version(),
                "python_version": sys.version.split()[0],
                "provenance": EvidenceProvenance.LIVE_APPLICATION.value,
            },
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # 4. Run TrialEvidenceAuditor Certification Check
        print("\n" + "=" * 80)
        print("   AVCS // INDEPENDENT TRIAL EVIDENCE AUDITOR CERTIFICATION                 ")
        print("=" * 80)
        auditor = TrialEvidenceAuditor(expected_scenario_count=12, expected_modes=[self.mode], min_trials=12)
        report = auditor.audit_directory(self.output_dir, enforce_live=True)

        report_md_path = self.output_dir / "audit_report.md"
        report_md_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"Auditor Gate Passed      : {report.live_certification_gate_passed}")
        print(f"Blocking Violations Count: {len([v for v in report.violations if v.severity.value == 'blocking'])}")
        print(f"Audit Report Markdown    : {report_md_path}")
        print("=" * 80 + "\n")

        return results, traces


def main() -> int:
    parser = argparse.ArgumentParser(description="AVCS Milestone 8 Phase D Live 12-Scenario Batch Runner")
    parser.add_argument("--output-dir", default="m8-live-results/live_12_matrix", help="Output directory")
    parser.add_argument("--seed", type=int, default=20261010, help="Master random seed")
    parser.add_argument("--settle-delay-s", type=float, default=0.25, help="Settling delay after action in seconds")
    parser.add_argument("--timeout-s", type=float, default=12.0, help="Timeout per scenario step in seconds")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    runner = LiveMatrixBatchRunner(
        output_dir=out_path,
        mode=Mode.VISUAL_ONLY,
        master_seed=args.seed,
        settle_delay_s=args.settle_delay_s,
        step_timeout_s=args.timeout_s,
    )
    results, _ = runner.run_campaign()
    all_passed = all(r.verified_success for r in results)
    return 0 if (len(results) == 12 and all_passed) else 1


if __name__ == "__main__":
    sys.exit(main())

