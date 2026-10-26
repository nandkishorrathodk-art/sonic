"""Closed-loop eye-hand servo for the workstation: perceive, correct, converge.

The gap this closes
-------------------
AVCS could *see* fast (~16 ms/frame) and the provider could *move* the pointer
smoothly, but nothing tied the two together. A pointer action was open-loop: the
agent decided a coordinate, the provider moved the hand once, and whatever
happened happened. That is not how a person points at something. A person
reaches, keeps their eye on the target, and nudges until the hand is actually on
it.

This module is that loop. It reads the real pointer position, asks
``forge.vision.servo.HumanReachModel`` for the next corrective step, dispatches
it, re-reads, and repeats until the residual error is inside the foveal basin --
or until the bounded budget runs out, in which case it reports honestly that it
did not converge. No LLM is involved: this is the fast System-1 motor loop, so a
mis-estimated coordinate is corrected by the next observation instead of being
committed blindly.

Why the loop lives here and the model lives in ``forge``
--------------------------------------------------------
The reach model is pure, deterministic, torch-free math, so it belongs with the
rest of the pixel-only vision code and can be reused offline (video
apprenticeship shares its cursor detector). The *dispatch* half talks to the
compute provider and the safety envelope, so it belongs in ``services/core``
next to the agent.

Honesty contract
----------------
* Convergence is measured, never assumed: the receipt carries the real
  iteration count, the measured final residual, and the measured latency.
* The loop fails closed. If the pointer position cannot be read, or the target
  is outside the screen, it raises instead of clicking at a guessed point.
* It never fabricates a "reached" result. A budget-exhausted reach reports
  ``converged=False`` with the residual it actually ended on.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sonic.logger import get_logger

logger = get_logger(__name__)

__all__ = ["ServoReceipt", "VisualServo", "ServoUnavailable"]


class ServoUnavailable(RuntimeError):
    """Raised when the servo cannot run (no cursor readback, no provider)."""


@dataclass
class ServoReceipt:
    """What the loop actually did, in measured terms.

    ``converged`` is the only success signal, and it is derived from the final
    measured residual -- never set by decree. ``iterations`` and
    ``latency_ms`` are measured, so the loop's real-time behaviour is auditable.
    """

    converged: bool
    iterations: int
    final_residual_px: float
    tolerance_px: float
    latency_ms: float
    start_px: tuple[int, int]
    target_px: tuple[int, int]
    path_px: list[tuple[int, int]] = field(default_factory=list)
    cursor_source: str = "provider"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "final_residual_px": round(self.final_residual_px, 2),
            "tolerance_px": round(self.tolerance_px, 2),
            "latency_ms": round(self.latency_ms, 2),
            "start_px": list(self.start_px),
            "target_px": list(self.target_px),
            "path_px": [list(point) for point in self.path_px],
            "cursor_source": self.cursor_source,
            "reason": self.reason,
        }


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


class VisualServo:
    """Drive the pointer onto a pixel target using closed-loop corrections.

    The servo is intentionally provider-agnostic: it is handed two async
    callables -- read the pointer, move the pointer -- and a frame reader for the
    vision fallback. That keeps it testable without a live desktop and keeps the
    safety envelope in the agent, where every dispatched move still passes the
    normal GUI action path.
    """

    #: Pointer reads and moves are both bounded so a stuck display cannot hang
    #: a mission; the deadline is the real guard, the iteration cap is a
    #: backstop for a pointer that barely responds at all.
    DEFAULT_MAX_ITERATIONS = 12
    DEFAULT_TOLERANCE_PX = 6
    DEFAULT_DEADLINE_SECONDS = 6.0

    def __init__(
        self,
        *,
        read_cursor: Callable[[], Awaitable[tuple[int, int] | None]],
        move_cursor: Callable[[int, int], Awaitable[Any]],
        screen_size: Callable[[], tuple[int, int]] | None = None,
        read_frame: Callable[[], Awaitable[Any]] | None = None,
        max_iterations: int | None = None,
        tolerance_px: int | None = None,
        deadline_seconds: float | None = None,
        model: Any | None = None,
    ) -> None:
        self._read_cursor = read_cursor
        self._move_cursor = move_cursor
        self._screen_size = screen_size
        self._read_frame = read_frame
        self.max_iterations = max(
            1, max_iterations if max_iterations is not None
            else _env_int("SONIC_SERVO_MAX_ITERATIONS", self.DEFAULT_MAX_ITERATIONS)
        )
        self.tolerance_px = max(
            1, tolerance_px if tolerance_px is not None
            else _env_int("SONIC_SERVO_TOLERANCE_PX", self.DEFAULT_TOLERANCE_PX)
        )
        self.deadline_seconds = max(
            0.25, deadline_seconds if deadline_seconds is not None
            else _env_float("SONIC_SERVO_DEADLINE_SECONDS", self.DEFAULT_DEADLINE_SECONDS)
        )
        if model is None:
            from forge.vision.servo import HumanReachModel

            model = HumanReachModel()
        self._model = model

    # -- helpers ---------------------------------------------------------
    def _screen_dims(self) -> tuple[int, int]:
        if self._screen_size is not None:
            try:
                width, height = self._screen_size()
                if width > 0 and height > 0:
                    return int(width), int(height)
            except Exception:
                pass
        return 1920, 1080

    async def _cursor(
        self, workspace_id: str, frame: Any | None
    ) -> tuple[tuple[float, float], str] | None:
        """Return the cursor in normalized [0, 1] space and its source.

        The provider's own readback is preferred because it is exact and cheap.
        The pixel-blob vision fallback is used only when the provider cannot
        answer, and it is a weak measurement -- a low-confidence reading is
        discarded rather than trusted.
        """
        position = None
        try:
            position = await self._read_cursor()
        except Exception as exc:
            logger.debug("servo_cursor_readback_failed", error=str(exc))
            position = None

        width, height = self._screen_dims()
        if position is not None:
            x, y = position
            return (x / max(width, 1), y / max(height, 1)), "provider"

        if frame is None and self._read_frame is None:
            return None
        if frame is None:
            try:
                frame = await self._read_frame()
            except Exception as exc:
                logger.debug("servo_frame_read_failed", error=str(exc))
                return None
        rgb = _frame_rgb(frame)
        if rgb is None:
            return None
        try:
            from forge.vision.servo import locate_cursor

            located = locate_cursor(rgb)
        except Exception as exc:
            logger.debug("servo_visual_cursor_failed", error=str(exc))
            return None
        if located is None or located[2] < 0.25:
            return None
        return (located[0], located[1]), "vision"

    # -- public API ------------------------------------------------------
    async def reach(
        self,
        workspace_id: str,
        target_px: tuple[int, int],
        *,
        frame: Any | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> ServoReceipt:
        """Converge the pointer onto ``target_px`` and report what happened.

        Raises :class:`ServoUnavailable` if the pointer position cannot be
        measured at all -- the loop never guesses. A reach that runs out of
        budget or deadline returns ``converged=False`` with the real residual.
        """
        width, height = self._screen_dims()
        tx, ty = int(target_px[0]), int(target_px[1])
        if not (0 <= tx < width and 0 <= ty < height):
            raise ServoUnavailable(
                f"servo target ({tx}, {ty}) is outside the {width}x{height} screen"
            )

        started = time.perf_counter()
        deadline = started + self.deadline_seconds
        tolerance_px = float(self.tolerance_px)
        # Tolerance in normalized units, so the model's contact basin and the
        # pixel acceptance criterion describe the same region.
        span = max(width, height)
        norm_tolerance = tolerance_px / max(span, 1)

        reading = await self._cursor(workspace_id, frame)
        if reading is None:
            raise ServoUnavailable(
                "pointer position is unavailable; refusing to move blind"
            )
        (cx, cy), source = reading
        start_px = (int(round(cx * width)), int(round(cy * height)))
        target_norm = (tx / max(width, 1), ty / max(height, 1))
        path: list[tuple[int, int]] = [start_px]

        def residual_px(point: tuple[float, float]) -> float:
            return math.hypot(
                (point[0] - target_norm[0]) * width,
                (point[1] - target_norm[1]) * height,
            )

        iterations = 0
        converged = residual_px((cx, cy)) <= tolerance_px
        reason = "already within tolerance" if converged else ""

        # Align the model's foveal basin with the pixel acceptance criterion so
        # it keeps correcting until the pixel tolerance is genuinely met, rather
        # than stopping early at its own default radius.
        self._model.foveal_contact_radius = norm_tolerance

        while not converged and iterations < self.max_iterations:
            if should_stop is not None and should_stop():
                reason = "interrupted"
                break
            if time.perf_counter() > deadline:
                reason = "deadline exceeded"
                break

            step = self._model.next_submovement(
                target_norm, (cx, cy), ballistic=(iterations == 0)
            )
            next_x = cx + step.dx
            next_y = cy + step.dy
            move_px = (
                int(round(next_x * width)),
                int(round(next_y * height)),
            )
            if move_px == path[-1]:
                reason = "no further progress"
                break

            iterations += 1
            await self._move_cursor(move_px[0], move_px[1])
            path.append(move_px)

            # Re-observe: this is the whole point. The next correction is based
            # on where the hand really is, not where we assumed it went.
            after = await self._cursor(workspace_id, frame)
            if after is None:
                reason = "pointer readback lost mid-reach"
                break
            (cx, cy), source = after
            observed_px = (int(round(cx * width)), int(round(cy * height)))
            if observed_px != path[-1]:
                path.append(observed_px)
            converged = residual_px((cx, cy)) <= tolerance_px

        final_residual = residual_px((cx, cy))
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not converged and not reason:
            reason = "iteration budget exhausted"

        receipt = ServoReceipt(
            converged=converged,
            iterations=iterations,
            final_residual_px=final_residual,
            tolerance_px=tolerance_px,
            latency_ms=latency_ms,
            start_px=start_px,
            target_px=(tx, ty),
            path_px=path,
            cursor_source=source,
            reason=reason,
        )
        logger.info(
            "visual_servo_reach",
            converged=converged,
            iterations=iterations,
            residual_px=round(final_residual, 2),
            latency_ms=round(latency_ms, 1),
            cursor_source=source,
            reason=reason,
        )
        return receipt


def _frame_rgb(frame: Any) -> Any | None:
    """Extract an (H, W, 3) uint8 array from a ScreenObservation-like object."""
    raw = getattr(frame, "raw_rgb", b"") or b""
    width = int(getattr(frame, "width", 0) or 0)
    height = int(getattr(frame, "height", 0) or 0)
    if raw and width > 0 and height > 0 and len(raw) == width * height * 3:
        try:
            import numpy as np

            return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        except Exception:
            return None
    return None
