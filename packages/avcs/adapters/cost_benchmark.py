"""Empirical Cost Benchmark: AVCS Token & Cost Reduction Evaluation.

Measures the concrete token, bandwidth, and cost savings achieved by using
AVCS's local System-1 reflex filter compared to standard naive cloud VLM polling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np

from adapters.claude_computer_use import AVCSClaudeStreamFilter
from avcs import SyntheticScreen, ScreenEye


@dataclass
class CostComparisonReport:
    workflow_duration_seconds: float
    total_screen_ticks_evaluated: int

    # Baseline (Naive 2-Second Polling)
    baseline_screenshots_sent: int
    baseline_tokens_consumed: int
    baseline_cost_usd: float

    # AVCS Reflex Filter
    avcs_screenshots_sent: int
    avcs_tokens_consumed: int
    avcs_cost_usd: float

    # Reductions
    tokens_saved: int
    token_reduction_percent: float
    cost_reduction_percent: float


def run_cost_reduction_benchmark(
    duration_seconds: float = 10.0,
    fps: int = 25,
) -> CostComparisonReport:
    """Executes a live comparative benchmark simulating real desktop interactions."""
    print("=" * 70)
    print("      AVCS vs NAIVE CLOUD VLM: EMPIRICAL COST BENCHMARK (M4.5)      ")
    print("=" * 70)

    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source, fps=fps)
    filter_layer = AVCSClaudeStreamFilter(eye=eye, surprise_threshold=0.3)

    print(f"Simulating continuous desktop stream for {duration_seconds:.1f} seconds at {fps} FPS...")
    total_ticks = int(duration_seconds * fps)
    dt = 1.0 / max(fps, 1)
    simulated_time = 1000.0

    for i in range(total_ticks):
        frame = source.grab()
        # Evaluate reflex filter with virtual clock progression
        filter_layer.should_emit_screenshot(frame=frame, timestamp=simulated_time)
        simulated_time += dt

    stats = filter_layer.stats

    # Baseline calculations: standard agent polls every 2 seconds
    baseline_count = max(1, int(duration_seconds / 2.0))
    baseline_tokens = baseline_count * AVCSClaudeStreamFilter.AVG_TOKENS_PER_SCREENSHOT
    baseline_cost = (baseline_tokens / 1_000_000.0) * AVCSClaudeStreamFilter.COST_PER_MILLION_INPUT_TOKENS

    # AVCS metrics
    avcs_count = stats.screenshots_sent_to_llm
    avcs_tokens = avcs_count * AVCSClaudeStreamFilter.AVG_TOKENS_PER_SCREENSHOT
    avcs_cost = (avcs_tokens / 1_000_000.0) * AVCSClaudeStreamFilter.COST_PER_MILLION_INPUT_TOKENS

    saved_tokens = max(0, baseline_tokens - avcs_tokens)
    reduction_pct = (saved_tokens / baseline_tokens) * 100.0 if baseline_tokens > 0 else 0.0
    cost_reduction_pct = ((baseline_cost - avcs_cost) / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0

    report = CostComparisonReport(
        workflow_duration_seconds=duration_seconds,
        total_screen_ticks_evaluated=total_ticks,
        baseline_screenshots_sent=baseline_count,
        baseline_tokens_consumed=baseline_tokens,
        baseline_cost_usd=round(baseline_cost, 4),
        avcs_screenshots_sent=avcs_count,
        avcs_tokens_consumed=avcs_tokens,
        avcs_cost_usd=round(avcs_cost, 4),
        tokens_saved=saved_tokens,
        token_reduction_percent=round(reduction_pct, 1),
        cost_reduction_percent=round(cost_reduction_pct, 1),
    )

    print("-" * 70)
    print(f"Total Frames Processed Locally : {report.total_screen_ticks_evaluated} frames (Zero Cloud Cost)")
    print(f"Baseline Cloud Screenshots     : {report.baseline_screenshots_sent} calls  -> {report.baseline_tokens_consumed:,} tokens (${report.baseline_cost_usd:.4f})")
    print(f"AVCS Filtered Screenshots      : {report.avcs_screenshots_sent} calls  -> {report.avcs_tokens_consumed:,} tokens (${report.avcs_cost_usd:.4f})")
    print(f"Tokens Saved                   : {report.tokens_saved:,} tokens")
    print(f"Empirical Token Reduction Rate : {report.token_reduction_percent:.1f}%")
    print(f"Cost Reduction Rate            : {report.cost_reduction_percent:.1f}%")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVCS Cost Reduction Benchmark")
    parser.add_argument("--seconds", type=float, default=6.0, help="Test duration")
    args = parser.parse_args()
    run_cost_reduction_benchmark(duration_seconds=args.seconds)
