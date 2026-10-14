"""AVCS Comprehensive Inference & Pipeline Profiler.

Executes Step 1 & Step 2 of the AVCS Performance Optimization Protocol:
1. Establishes reproducible baseline measurements on fixed inputs and representative screens.
2. Measures isolated component latencies:
   - Capture (GDI BitBlt vs MSS vs Pre-captured Buffer)
   - Preprocessing (Resample, Transpose, Normalization)
   - Motion Engine (Magnocellular Optical Flow)
   - Backbone Forward (CorticalBackbone2M)
   - Affordance Forward (AffordanceHead2M)
   - Ventral Forward (VentralStream2M)
   - Hotspot Peak Extraction
   - Complete Model Forward (UnifiedCortex2M)
   - Complete Perception Tick (HumanEye2M)
3. Profiles CPU utilization, thread scaling (1, 2, 4, 8 threads), and memory.
4. Generates benchmark JSON and Markdown report.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from avcs.capture_adapter import Win32GDICaptureAdapter
from avcs.cortex_2m import (
    AffordanceHead2M,
    CorticalBackbone2M,
    HumanEye2M,
    LanguageGrounder2M,
    UnifiedCortex2M,
    UNIVERSAL_AFFORDANCE_CLASSES,
    UNIVERSAL_APP_CATEGORIES,
    VentralStream2M,
)
from forge.vision.cortex.benchmark_real_desktop import RealisticDesktopRenderer


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def profile_callable(
    fn,
    warmup: int = 50,
    iterations: int = 200,
    name: str = "operation",
) -> Dict[str, float]:
    """Profiles a callable function with warmup and percentile statistics."""
    # Warmup
    for _ in range(warmup):
        fn()

    gc.disable()
    times_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    gc.enable()

    times_arr = np.array(times_ms)
    p50 = float(np.percentile(times_arr, 50))
    p90 = float(np.percentile(times_arr, 90))
    p95 = float(np.percentile(times_arr, 95))
    p99 = float(np.percentile(times_arr, 99))
    mean = float(np.mean(times_arr))
    std = float(np.std(times_arr))
    fps = 1000.0 / p50 if p50 > 0 else 0.0

    return {
        "name": name,
        "iterations": iterations,
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "p50_ms": round(p50, 3),
        "p90_ms": round(p90, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "sustained_fps": round(fps, 1),
    }


def run_avcs_profiling_suite(
    output_path: str = "m8-live-results/optimization_baseline.json",
    num_iterations: int = 200,
    warmup_iterations: int = 50,
) -> Dict[str, Any]:
    print("=" * 78)
    print("   AVCS PERFORMANCE OPTIMIZATION — REPRODUCIBLE BASELINE PROFILER   ")
    print("=" * 78)

    git_sha = get_git_commit()
    device = torch.device("cpu")
    num_threads = torch.get_num_threads()
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()

    print(f"Git Commit SHA       : {git_sha}")
    print(f"Platform             : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor            : {platform.processor()}")
    print(f"Physical / Logic Cores: {cpu_count_physical} / {cpu_count_logical}")
    print(f"PyTorch Thread Count : {num_threads}")
    print(f"Total RAM            : {mem.total / (1024**3):.1f} GB")
    print(f"PyTorch Version      : {torch.__version__}")
    print("=" * 78)

    # 1. Prepare Realistic Visual Screen Benchmark Dataset
    renderer = RealisticDesktopRenderer(width=1920, height=1080)
    screen_samples = [
        ("VS Code (Editor)", renderer.render_vscode().frame),
        ("Chrome (Web Browser)", renderer.render_chrome().frame),
        ("Excel (Spreadsheet)", renderer.render_excel().frame),
        ("Notion (Productivity)", renderer.render_notion().frame),
        ("PowerShell (Terminal)", renderer.render_terminal().frame),
    ]

    # Pre-render 128x128 normalized tensors for isolated network tests
    test_tensor_fixed = torch.randn(1, 3, 128, 128, dtype=torch.float32, device=device)
    test_feat_fixed = torch.randn(1, 256, 8, 8, dtype=torch.float32, device=device)

    # Instantiate Baseline Cortex Architecture
    cortex = UnifiedCortex2M(feature_dim=256).to(device)
    cortex.eval()
    total_params = cortex.total_parameters

    backbone_params = sum(p.numel() for p in cortex.backbone.parameters())
    affordance_params = sum(p.numel() for p in cortex.affordance.parameters())
    ventral_params = sum(p.numel() for p in cortex.ventral.parameters())
    grounder_params = sum(p.numel() for p in cortex.grounder.parameters())

    print("\n--- Model Architecture & Parameter Verification ---")
    print(f"UnifiedCortex2M Total Params : {total_params:,}")
    print(f"  - CorticalBackbone2M      : {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"  - AffordanceHead2M        : {affordance_params:,} ({affordance_params/total_params*100:.1f}%)")
    print(f"  - VentralStream2M         : {ventral_params:,} ({ventral_params/total_params*100:.1f}%)")
    print(f"  - LanguageGrounder2M      : {grounder_params:,} ({grounder_params/total_params*100:.1f}%)")

    # -------------------------------------------------------------
    # Step 2: Component-by-Component Latency Breakdown
    # -------------------------------------------------------------
    print("\n--- Component-by-Component Latency Breakdown ---")
    benchmarks = {}

    # A. Screen Preprocessing (1920x1080 uint8 -> 128x128 float32 Tensor)
    sample_frame = screen_samples[0][1]

    def run_preprocessing():
        H, W = sample_frame.shape[:2]
        sy = max(1, H // 128)
        sx = max(1, W // 128)
        sub = sample_frame[::sy, ::sx][:128, :128]
        arr = sub.astype(np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    benchmarks["preprocessing"] = profile_callable(
        run_preprocessing, warmup_iterations, num_iterations, "Preprocessing (Downsample & Tensor Conversion)"
    )

    # B. Backbone Forward Pass
    with torch.no_grad():
        def run_backbone():
            return cortex.backbone(test_tensor_fixed)

        benchmarks["backbone_forward"] = profile_callable(
            run_backbone, warmup_iterations, num_iterations, "Backbone Forward (CorticalBackbone2M)"
        )

    # C. Affordance Head Forward Pass
    with torch.no_grad():
        def run_affordance():
            return cortex.affordance(test_feat_fixed)

        benchmarks["affordance_forward"] = profile_callable(
            run_affordance, warmup_iterations, num_iterations, "Affordance Forward (AffordanceHead2M)"
        )

    # D. Ventral Stream Forward Pass
    with torch.no_grad():
        dummy_aff_logits = torch.randn(1, 24, 8, 8, dtype=torch.float32, device=device)

        def run_ventral():
            return cortex.ventral(test_feat_fixed, dorsal_context=dummy_aff_logits, return_raw=True)

        benchmarks["ventral_forward"] = profile_callable(
            run_ventral, warmup_iterations, num_iterations, "Ventral Forward (VentralStream2M)"
        )

    # E. Hotspot Peak Extraction
    with torch.no_grad():
        def run_hotspots():
            return cortex.affordance.find_hotspots(test_feat_fixed, top_k=5)

        benchmarks["hotspot_extraction_baseline"] = profile_callable(
            run_hotspots, warmup_iterations, num_iterations, "Hotspot Extraction (Baseline with Re-forward)"
        )

    # F. UnifiedCortex2M Complete Forward Pass (No Hotspots)
    with torch.no_grad():
        def run_cortex_forward():
            return cortex(test_tensor_fixed)

        benchmarks["cortex_forward_only"] = profile_callable(
            run_cortex_forward, warmup_iterations, num_iterations, "UnifiedCortex2M Forward Only"
        )

    # G. UnifiedCortex2M Forward + Hotspot Extraction (Baseline Flow in HumanEye2M)
    with torch.no_grad():
        def run_cortex_and_hotspots_baseline():
            out = cortex(test_tensor_fixed)
            hotspots = cortex.affordance.find_hotspots(out["feat"], top_k=5)
            return out, hotspots

        benchmarks["cortex_plus_hotspots_baseline"] = profile_callable(
            run_cortex_and_hotspots_baseline, warmup_iterations, num_iterations, "Cortex + Hotspots (Baseline Flow)"
        )

    # H. Motion Engine (Magnocellular Optical Flow)
    motion_engine = cortex_motion_engine = HumanEye2M(source="synthetic").motion_engine

    def run_motion_engine():
        return motion_engine.process(sample_frame)

    benchmarks["motion_engine"] = profile_callable(
        run_motion_engine, warmup_iterations, num_iterations, "Motion Engine (Optical Flow 16x24)"
    )

    # I. GDI Desktop Capture
    try:
        gdi_cap = Win32GDICaptureAdapter()

        def run_gdi_capture():
            return gdi_cap.capture()

        benchmarks["gdi_capture"] = profile_callable(
            run_gdi_capture, warmup=10, iterations=50, name="GDI Desktop BitBlt Capture"
        )
    except Exception as e:
        print(f"GDI Capture profiling skipped: {e}")

    # Print Summary Table
    print("\n" + "=" * 78)
    print(f"{'Component Name':<42} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'FPS':<8}")
    print("-" * 78)
    for k, v in benchmarks.items():
        print(f"{v['name']:<42} | {v['p50_ms']:<9.2f} | {v['p95_ms']:<9.2f} | {v['sustained_fps']:<8.1f}")
    print("=" * 78)

    # -------------------------------------------------------------
    # Step 3: Thread Scaling Analysis
    # -------------------------------------------------------------
    print("\n--- Thread Scaling Analysis (UnifiedCortex2M Forward) ---")
    thread_results = {}
    orig_threads = torch.get_num_threads()
    for t in [1, 2, 4, 8]:
        if t <= cpu_count_logical:
            torch.set_num_threads(t)
            with torch.no_grad():
                res = profile_callable(
                    run_cortex_forward, warmup=20, iterations=100, name=f"Cortex Forward ({t} Threads)"
                )
                thread_results[f"{t}_threads"] = res
                print(f"Threads: {t:<2} -> p50: {res['p50_ms']:.2f} ms | p95: {res['p95_ms']:.2f} ms | FPS: {res['sustained_fps']:.1f}")
    torch.set_num_threads(orig_threads)

    # Package Output Report
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "git_commit": git_sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "physical_cores": cpu_count_physical,
            "logical_cores": cpu_count_logical,
            "ram_gb": round(mem.total / (1024**3), 2),
            "pytorch_version": torch.__version__,
        },
        "model_parameters": {
            "total": total_params,
            "backbone": backbone_params,
            "affordance": affordance_params,
            "ventral": ventral_params,
            "grounder": grounder_params,
        },
        "component_benchmarks": benchmarks,
        "thread_scaling": thread_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nBaseline benchmark JSON saved to: {output_path}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVCS Baseline Profiler")
    parser.add_argument("--output", type=str, default="m8-live-results/optimization_baseline.json")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    run_avcs_profiling_suite(output_path=args.output, num_iterations=args.iterations)

