"""Live Demonstration and Telemetry Verification for AVCS Motor Servoing."""
from __future__ import annotations

import argparse
import sys
import time

from forge.vision.cortex.motor import (
    HandEyeServoingController,
    VirtualCursorDriver,
    get_default_driver,
    SafetyBreakoutException,
)


def run_motor_demo(live: bool = False, speed: float = 1.5) -> None:
    print("=" * 70)
    print("       AVCS PHYSICAL MOTOR DRIVER & HAND-EYE SERVOING DEMO       ")
    print("=" * 70)

    driver = get_default_driver(force_virtual=not live)
    driver_type = "Windows Native (Win32 user32.dll)" if live and sys.platform == "win32" else "Virtual In-Memory Driver"
    print(f"Active Driver     : {driver_type}")
    print(f"Screen Resolution : {driver.screen_size[0]}x{driver.screen_size[1]}")

    start_pos = driver.get_position_normalized()
    print(f"Initial Cursor    : ({start_pos[0]:.3f}, {start_pos[1]:.3f}) [Normalized]")

    # Choose a target across the screen
    target = (0.75, 0.65)
    print(f"Target Gaze Point : ({target[0]:.3f}, {target[1]:.3f})")
    print(f"Fitts' Law Dynamics: Smooth ballistic acceleration with foveal damping")
    print(f"Fail-Safe Guard   : Corner breakout ACTIVE (move mouse to corner to abort)")
    print("-" * 70)

    controller = HandEyeServoingController(driver=driver)

    try:
        t0 = time.perf_counter()
        result = controller.servo_to(
            target_gaze=target,
            timeout_seconds=2.0,
            click_on_arrival=False,
            speed_factor=speed,
            realtime=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"Servoing Result   : {'LOCKED (Contact Acquired)' if result.contact_acquired else 'TIMED OUT'}")
        print(f"Final Position    : ({result.final_position[0]:.3f}, {result.final_position[1]:.3f})")
        print(f"Convergence Time  : {result.elapsed_ms:.1f} ms (Wall clock: {elapsed_ms:.1f} ms)")
        print(f"Trajectory Steps  : {result.total_steps} ticks at 60 Hz")

        # Telemetry sample
        print("\nTrajectory Sample (First 5 ticks & Last 3 ticks):")
        for i, pt in enumerate(result.trajectory[:5]):
            print(f"  Tick {i:02d}: pos=({pt.x:.3f}, {pt.y:.3f}) vel=({pt.velocity_x:+.2f}, {pt.velocity_y:+.2f}) dist={pt.distance_to_target:.3f}")
        print("  ...")
        for i, pt in enumerate(result.trajectory[-3:], start=len(result.trajectory)-3):
            print(f"  Tick {i:02d}: pos=({pt.x:.3f}, {pt.y:.3f}) vel=({pt.velocity_x:+.2f}, {pt.velocity_y:+.2f}) dist={pt.distance_to_target:.3f} contact={pt.contact}")

    except SafetyBreakoutException as abort:
        print(f"\n[SAFETY ABORT TRIGGERED] {abort}")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVCS Motor Servoing Demo")
    parser.add_argument("--live", action="store_true", help="Steer the real OS mouse cursor")
    parser.add_argument("--speed", type=float, default=1.5, help="Speed scaling factor")
    args = parser.parse_args()
    run_motor_demo(live=args.live, speed=args.speed)

