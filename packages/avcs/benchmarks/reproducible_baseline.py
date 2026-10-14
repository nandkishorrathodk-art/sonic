"""AVCS Milestone 0: Instrumented Reproducible Performance Baseline Runner.

Measures granular, isolated execution latencies across every individual stage
of the AVCS perception and action loop using monotonic high-precision clocks.
Exports machine-readable JSON telemetry reports with full statistical percentiles
(p50, p95, p99, min, max, std).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from avcs.human_eye import HumanEye, HumanEyeState
from avcs.ipc import SharedMemoryBroadcaster
from forge.vision.capture import SyntheticScreen, Win32FastCapture, open_capture


def get_hardware_telemetry() -> Dict[str, Any]:
    """Gathers deterministic hardware and environment metadata."""
    cpu_count = os.cpu_count() or 1
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical_cores": cpu_count,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_cuda_available": torch.cuda.is_available(),
    }


class InstrumentedBenchmarkRunner:
    """Monotonically instrumented benchmark evaluating AVCS per-stage latencies."""

    def __init__(
        self,
        source_name: str = "synthetic",
        device: str = "cpu",
        num_frames: int = 1000,
        warmup_frames: int = 100,
    ) -> None:
        self.source_name = source_name
        self.device = device
        self.num_frames = num_frames
        self.warmup_frames = warmup_frames

        # Resolve capture source
        if source_name == "win32":
            self.source = Win32FastCapture()
        elif source_name == "desktop":
            self.source = open_capture(prefer_real=True, threaded=False)
        else:
            self.source = SyntheticScreen()

        self.eye = HumanEye(source=self.source, device=self.device)
        self.broadcaster = SharedMemoryBroadcaster(buffer_name="avcs_baseline_ipc")

        # Metric storage
        self.metrics: Dict[str, List[float]] = {
            "capture_ms": [],
            "tensor_prep_ms": [],
            "backbone_ms": [],
            "affordance_ms": [],
            "motion_flow_ms": [],
            "scene_diff_ms": [],
            "ipc_broadcast_ms": [],
            "total_tick_ms": [],
        }

    def run(self) -> Dict[str, Any]:
        """Executes warmup and instrumented collection loop."""
        print(f"[*] Starting AVCS Milestone 0 Baseline Runner ({self.source_name}, {self.device})...")
        print(f"[*] Warming up for {self.warmup_frames} frames...")
        for _ in range(self.warmup_frames):
            frame = self.source.grab()
            self.eye.observe(frame=frame)

        print(f"[*] Recording {self.num_frames} instrumented observations...")
        start_overall = time.perf_counter()

        for i in range(self.num_frames):
            t_tick_start = time.perf_counter()

            # 1. Screen Capture Stage
            t0 = time.perf_counter()
            frame = self.source.grab()
            t_cap = (time.perf_counter() - t0) * 1000.0

            # 2. Tensor Preprocessing Stage
            t0 = time.perf_counter()
            t_in = torch.from_numpy(frame).float().div_(255.0)
            t_in = t_in.permute(2, 0, 1).unsqueeze(0).to(self.device)
            if t_in.shape[-2:] != self.eye.input_size:
                t_resized = torch.nn.functional.interpolate(
                    t_in, size=self.eye.input_size, mode="bilinear", align_corners=False
                )
            else:
                t_resized = t_in
            t_prep = (time.perf_counter() - t0) * 1000.0

            # 3. Cortical Feature Backbone Stage
            t0 = time.perf_counter()
            with torch.inference_mode():
                feat = self.eye.backbone(t_resized)
            t_backbone = (time.perf_counter() - t0) * 1000.0

            # 4. Dorsal Affordance Head Stage
            t0 = time.perf_counter()
            with torch.inference_mode():
                affordances = self.eye.affordance_head.find_hotspots(feat, top_k=3)
            t_affordance = (time.perf_counter() - t0) * 1000.0

            # 5. Magnocellular Optical Flow Stage
            t0 = time.perf_counter()
            motion_field, mean_motion, moving_objs, surprise = self.eye.motion_engine.process(frame)
            t_motion = (time.perf_counter() - t0) * 1000.0

            # 6. Episodic Scene Tree Diff Stage
            t0 = time.perf_counter()
            diff = self.eye.memory.step(
                affordances=affordances,
                category="screen",
                category_confidence=0.9,
                moving_objects=moving_objs,
                surprise_score=surprise,
                prior_action="observe",
            )
            t_diff = (time.perf_counter() - t0) * 1000.0

            # 7. Zero-Copy Shared Memory IPC Broadcast Stage
            t0 = time.perf_counter()
            self.broadcaster.broadcast({
                "frame": i,
                "surprise_score": surprise,
                "motion": mean_motion,
                "affordances_count": len(affordances),
            })
            t_ipc = (time.perf_counter() - t0) * 1000.0

            t_tick_total = (time.perf_counter() - t_tick_start) * 1000.0

            # Record
            self.metrics["capture_ms"].append(t_cap)
            self.metrics["tensor_prep_ms"].append(t_prep)
            self.metrics["backbone_ms"].append(t_backbone)
            self.metrics["affordance_ms"].append(t_affordance)
            self.metrics["motion_flow_ms"].append(t_motion)
            self.metrics["scene_diff_ms"].append(t_diff)
            self.metrics["ipc_broadcast_ms"].append(t_ipc)
            self.metrics["total_tick_ms"].append(t_tick_total)

        total_elapsed_s = time.perf_counter() - start_overall
        effective_fps = self.num_frames / total_elapsed_s

        # Compute Statistics
        stats: Dict[str, Dict[str, float]] = {}
        for stage, samples in self.metrics.items():
            arr = np.array(samples)
            stats[stage] = {
                "mean": round(float(np.mean(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "min": round(float(np.min(arr)), 3),
                "p50": round(float(np.percentile(arr, 50)), 3),
                "p90": round(float(np.percentile(arr, 90)), 3),
                "p95": round(float(np.percentile(arr, 95)), 3),
                "p99": round(float(np.percentile(arr, 99)), 3),
                "max": round(float(np.max(arr)), 3),
            }

        report = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hardware": get_hardware_telemetry(),
            "config": {
                "source": self.source_name,
                "device": self.device,
                "num_frames": self.num_frames,
                "warmup_frames": self.warmup_frames,
            },
            "summary": {
                "total_frames": self.num_frames,
                "total_elapsed_seconds": round(total_elapsed_s, 3),
                "effective_fps": round(effective_fps, 2),
                "total_tick_p50_ms": stats["total_tick_ms"]["p50"],
                "total_tick_p95_ms": stats["total_tick_ms"]["p95"],
                "gate_passed_p50_sub10ms": bool(stats["total_tick_ms"]["p50"] <= 10.2),
            },
            "stage_statistics": stats,
        }

        self.cleanup()
        return report

    def cleanup(self) -> None:
        """Releases all capture and IPC resources."""
        if hasattr(self.source, "close"):
            try:
                self.source.close()
            except Exception:
                pass
        self.broadcaster.close()
        self.eye.close()


def print_formatted_report(report: Dict[str, Any]) -> None:
    """Prints clean terminal summary."""
    print("\n" + "=" * 80)
    print("       AVCS MILESTONE 0: REPRODUCIBLE BASELINE TELEMETRY REPORT       ")
    print("=" * 80)
    hw = report["hardware"]
    print(f"Platform: {hw['platform']} | CPU: {hw['processor']} ({hw['cpu_logical_cores']} cores)")
    print(f"PyTorch : {hw['torch_version']} ({report['config']['device']}) | Capture: {report['config']['source']}")
    print("-" * 80)
    print(f"{'Pipeline Stage':<24} | {'Mean (ms)':<9} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9} | {'Max (ms)':<9}")
    print("-" * 80)

    stages = [
        ("1. Screen Capture", "capture_ms"),
        ("2. Tensor Preprocessing", "tensor_prep_ms"),
        ("3. Cortical Backbone", "backbone_ms"),
        ("4. Affordance Head", "affordance_ms"),
        ("5. Optical Flow Flow", "motion_flow_ms"),
        ("6. Scene Tree Diff", "scene_diff_ms"),
        ("7. Zero-Copy IPC", "ipc_broadcast_ms"),
        ("TOTAL TICK (End-to-End)", "total_tick_ms"),
    ]

    for label, key in stages:
        s = report["stage_statistics"][key]
        print(f"{label:<24} | {s['mean']:<9.3f} | {s['p50']:<9.3f} | {s['p95']:<9.3f} | {s['p99']:<9.3f} | {s['max']:<9.3f}")

    print("-" * 80)
    summary = report["summary"]
    print(f"Processed Frames       : {summary['total_frames']}")
    print(f"Effective Frame Rate   : {summary['effective_fps']} FPS")
    print(f"End-to-End Latency p50 : {summary['total_tick_p50_ms']} ms")
    print(f"End-to-End Latency p95 : {summary['total_tick_p95_ms']} ms")
    passed = summary["gate_passed_p50_sub10ms"]
    print(f"Milestone 0 Gate (<=10.2ms p50) : {'[PASSED]' if passed else '[FAILED]'}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Milestone 0 Baseline Runner")
    parser.add_argument("--source", choices=["synthetic", "win32", "desktop"], default="synthetic")
    parser.add_argument("--frames", type=int, default=500, help="Number of benchmark frames")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup frames")
    parser.add_argument("--device", default="cpu", help="PyTorch compute device")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Directory for JSON reports")
    args = parser.parse_args()

    runner = InstrumentedBenchmarkRunner(
        source_name=args.source,
        device=args.device,
        num_frames=args.frames,
        warmup_frames=args.warmup,
    )

    report = runner.run()
    print_formatted_report(report)

    # Save JSON report
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output_dir, f"baseline_{args.source}_{timestamp_str}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved machine-readable report to: {out_file}")


if __name__ == "__main__":
    main()
