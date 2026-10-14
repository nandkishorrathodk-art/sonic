"""Live Interactive Perception Demo for Artificial Visual Cortex (AVCS).

Runs the AVCS engine in real time on the desktop display (or synthetic screen),
visualizing gaze fixation, affordances, stability, and surprise metrics.
"""
from __future__ import annotations

import argparse
import sys
import time

from forge.vision.capture import open_capture
from forge.vision.cortex.inference.engine import AVCSInferenceEngine
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState


def run_live_demo(duration_seconds: float = 10.0, fps_target: int = 25):
    print("=" * 65)
    print("      LIVE ARTIFICIAL VISUAL CORTEX (AVCS) STREAM MONITOR        ")
    print("=" * 65)

    source = open_capture()
    print(f"Capture Source: {type(source).__name__} | Target FPS: {fps_target}")

    model = ArtificialVisualCortex()

    engine = AVCSInferenceEngine(
        cortex=model,
        source=source,
        fps_target=fps_target,
    )

    print("\nStarting live perception loop... (Press Ctrl+C to stop)\n")
    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < duration_seconds:
            t0 = time.perf_counter()
            state = engine.step()
            frame_count += 1
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Dynamic terminal status update
            status_line = (
                f"\r[Frame {frame_count:04d}] "
                f"Tick: {dt_ms:5.1f}ms | "
                f"State: {state.stability_state:<14} | "
                f"Surprise: {state.surprise_score:.3f} | "
                f"Gaze: ({state.fixation[0]:.2f}, {state.fixation[1]:.2f}) | "
                f"UI: {state.primary_category:<15}"
            )
            sys.stdout.write(status_line)
            sys.stdout.flush()

            # Throttle to target FPS
            sleep_needed = (1.0 / fps_target) - (dt_ms / 1000.0)
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

    elapsed = time.time() - start_time
    avg_fps = frame_count / max(elapsed, 1e-4)

    print("\n\n" + "-" * 65)
    print("LIVE RUN SUMMARY:")
    print(f"  • Total Frames Perceived: {frame_count}")
    print(f"  • Elapsed Time:           {elapsed:.2f} seconds")
    print(f"  • Average Perception FPS: {avg_fps:.1f} FPS")
    print("-" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    run_live_demo(duration_seconds=args.seconds, fps_target=args.fps)

