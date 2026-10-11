"""AVCS Benchmark CLI Entry Point."""
from __future__ import annotations

import sys
from forge.vision.cortex.benchmark_real_desktop import run_desktop_diversity_benchmark, main as benchmark_main

def main() -> None:
    if hasattr(benchmark_main, "__call__"):
        benchmark_main()
    else:
        run_desktop_diversity_benchmark()

if __name__ == "__main__":
    main()

