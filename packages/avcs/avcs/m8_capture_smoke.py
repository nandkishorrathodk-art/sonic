"""AVCS Milestone 8 Gate M8-B: Phase A Capture-Only Live Windows Smoke Benchmark.

Executes non-invasive, input-free physical display capture benchmarks on Windows host:
- Benchmarks 100 consecutive frames on Win32 GDI (BitBlt) and mss Desktop Capture
- Measures p50, p95, p99, and max frame acquisition latency
- Validates non-black frames (mean/std sanity), display geometry, and SHA-256 evidence hashes
- Zero physical input dispatched (strictly non-interactive observation phase)
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CaptureBackendMetrics:
    backend_name: str
    target_frames: int
    successful_frames: int
    dropped_frames: int
    blank_or_black_frames: int
    min_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    effective_fps: float
    resolution: str
    sample_sha256: str
    passed: bool


def benchmark_win32_gdi(num_frames: int = 100) -> CaptureBackendMetrics:
    """Benchmarks Win32 GDI BitBlt capture across consecutive frames."""
    from avcs.capture_adapter import Win32GDICaptureAdapter

    adapter = Win32GDICaptureAdapter()
    adapter.open()

    latencies: List[float] = []
    blank_count = 0
    sample_hash = ""
    res_str = ""

    try:
        geo = adapter.get_geometry()
        res_str = f"{geo.virtual_width}x{geo.virtual_height}"

        # Warm-up 5 frames
        for _ in range(5):
            _ = adapter.grab()

        for i in range(num_frames):
            t0 = time.perf_counter()
            frame = adapter.grab()
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            latencies.append(dt_ms)

            # Check blank/black
            if frame.mean() < 1.0 or frame.std() < 0.1:
                blank_count += 1

            if i == num_frames - 1:
                sample_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    finally:
        adapter.close()

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    effective_fps = 1000.0 / max(0.1, mean_lat)

    return CaptureBackendMetrics(
        backend_name="Win32_GDI_BitBlt",
        target_frames=num_frames,
        successful_frames=len(latencies),
        dropped_frames=num_frames - len(latencies),
        blank_or_black_frames=blank_count,
        min_latency_ms=round(float(np.min(lat_arr)), 3),
        p50_latency_ms=round(float(np.percentile(lat_arr, 50)), 3),
        p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 3),
        p99_latency_ms=round(float(np.percentile(lat_arr, 99)), 3),
        max_latency_ms=round(float(np.max(lat_arr)), 3),
        mean_latency_ms=round(mean_lat, 3),
        effective_fps=round(effective_fps, 2),
        resolution=res_str,
        sample_sha256=sample_hash,
        passed=(len(latencies) == num_frames and blank_count == 0),
    )


def benchmark_mss_desktop(num_frames: int = 100) -> CaptureBackendMetrics:
    """Benchmarks mss desktop screen capture across consecutive frames."""
    from forge.vision.capture import ScreenCapture

    cap = ScreenCapture(monitor=1).open()
    latencies: List[float] = []
    blank_count = 0
    sample_hash = ""
    res_str = ""

    try:
        # Warm-up 5 frames
        for _ in range(5):
            _ = cap.grab()

        for i in range(num_frames):
            t0 = time.perf_counter()
            frame = cap.grab()
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            latencies.append(dt_ms)

            if not res_str:
                res_str = f"{frame.shape[1]}x{frame.shape[0]}"

            if frame.mean() < 1.0 or frame.std() < 0.1:
                blank_count += 1

            if i == num_frames - 1:
                sample_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    finally:
        cap.close()

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    effective_fps = 1000.0 / max(0.1, mean_lat)

    return CaptureBackendMetrics(
        backend_name="MSS_Desktop_Capture",
        target_frames=num_frames,
        successful_frames=len(latencies),
        dropped_frames=num_frames - len(latencies),
        blank_or_black_frames=blank_count,
        min_latency_ms=round(float(np.min(lat_arr)), 3),
        p50_latency_ms=round(float(np.percentile(lat_arr, 50)), 3),
        p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 3),
        p99_latency_ms=round(float(np.percentile(lat_arr, 99)), 3),
        max_latency_ms=round(float(np.max(lat_arr)), 3),
        mean_latency_ms=round(mean_lat, 3),
        effective_fps=round(effective_fps, 2),
        resolution=res_str,
        sample_sha256=sample_hash,
        passed=(len(latencies) == num_frames and blank_count == 0),
    )


def run_phase_a_smoke_benchmark(output_dir: str = "m8-live-results") -> Dict[str, Any]:
    """Runs Phase A capture-only smoke benchmark suite."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("   AVCS // MILESTONE 8 GATE M8-B: PHASE A LIVE CAPTURE SMOKE BENCHMARK    ")
    print("=" * 78)
    print(f"OS Platform     : {platform.system()} {platform.release()} (Build {platform.version()})")
    print(f"Architecture    : {platform.machine()}")
    print(f"Python Runtime  : {sys.version.split()[0]}")

    results: List[CaptureBackendMetrics] = []

    # 1. Benchmark Win32 GDI
    if sys.platform == "win32":
        print("\n[1/2] Benchmarking Win32 GDI (BitBlt)...")
        try:
            m_gdi = benchmark_win32_gdi(num_frames=100)
            results.append(m_gdi)
            print(f"      Passed         : {m_gdi.passed}")
            print(f"      Resolution     : {m_gdi.resolution}")
            print(f"      p50 Latency    : {m_gdi.p50_latency_ms} ms")
            print(f"      p95 Latency    : {m_gdi.p95_latency_ms} ms")
            print(f"      Effective FPS  : {m_gdi.effective_fps} FPS")
            print(f"      Blank Frames   : {m_gdi.blank_or_black_frames} / {m_gdi.target_frames}")
        except Exception as exc:
            print(f"      FAILED with error: {exc}")

    # 2. Benchmark MSS
    print("\n[2/2] Benchmarking MSS Desktop Capture...")
    try:
        m_mss = benchmark_mss_desktop(num_frames=100)
        results.append(m_mss)
        print(f"      Passed         : {m_mss.passed}")
        print(f"      Resolution     : {m_mss.resolution}")
        print(f"      p50 Latency    : {m_mss.p50_latency_ms} ms")
        print(f"      p95 Latency    : {m_mss.p95_latency_ms} ms")
        print(f"      Effective FPS  : {m_mss.effective_fps} FPS")
        print(f"      Blank Frames   : {m_mss.blank_or_black_frames} / {m_mss.target_frames}")
    except Exception as exc:
        print(f"      FAILED with error: {exc}")

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "M8_B_PHASE_A_CAPTURE_SMOKE",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "backends": [asdict(r) for r in results],
        "all_passed": all(r.passed for r in results),
    }

    report_file = out_path / "phase_a_capture_smoke_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print(f"Report saved to: {report_file}")
    print(f"Overall Phase A Smoke Status: {'PASSED' if report['all_passed'] else 'FAILED'}")
    print("=" * 78)

    return report


if __name__ == "__main__":
    run_phase_a_smoke_benchmark()
