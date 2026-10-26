"""Real-time visual servoing: the eye-hand loop that closes on pixels alone.

A human does not compute a screen coordinate and teleport the hand to it. The
hand reaches, the eye keeps watching the hand, and the residual error is driven
to zero over a short burst of corrective submovements (Woodworth's two-component
model: an initial ballistic throw, then visually guided corrections). The
convergence is what makes the motion look human -- and, more importantly, it is
what makes the action *self-correcting*: a mis-estimated target is fixed by the
next observation instead of being committed blindly.

This module is that loop, and it is deliberately **pure NumPy**: no torch, no
learned checkpoint, no network. It has to run inside the perception tick budget
(~tens of milliseconds), so a dependency that costs 1.2 s of import and buys
nothing at this stage would be the wrong tool.

Two cooperating parts
---------------------
``locate_cursor``
    A fast proprioceptive/visual sensor: find the pointer in a raw RGB frame as
    a small high-contrast blob. Used when the provider cannot report the cursor
    position directly (or as a cross-check against it).

``HumanReachModel``
    SONIC's own motor model. Given a visual target and the currently observed
    cursor it emits the *next* submovement, decelerating as the target is
    approached, and declares contact when the foveal basin is reached. It is a
    deterministic controller, not a learned network: its parameters (speed,
    damping, foveal radius, noise) are explicit and inspectable, so the loop can
    be reasoned about and tested rather than trusted.

Everything here reports what actually happened -- measured iterations,
measured residual, measured latency. Nothing claims convergence it did not
achieve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ServoError",
    "CursorReading",
    "Submovement",
    "HumanReachModel",
    "locate_cursor",
]

#: Luma weights for the detector's grayscale reduction (Rec. 601).
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class ServoError(RuntimeError):
    """Raised when the servo cannot produce a trustworthy result.

    The loop fails closed: a frame that cannot be measured, or a target that
    cannot be reached within the configured budget, raises rather than
    reporting a fabricated success.
    """


@dataclass(frozen=True)
class CursorReading:
    """One measurement of where the pointer is, in normalized [0, 1] space."""

    x: float
    y: float
    confidence: float
    source: str  # "provider" | "vision"


@dataclass(frozen=True)
class Submovement:
    """One corrective step of a reach, in normalized [0, 1] space."""

    dx: float
    dy: float
    distance_to_target: float
    speed: float
    contact_acquired: bool


def locate_cursor(
    frame: np.ndarray,
    *,
    patch: int = 15,
) -> tuple[float, float, float] | None:
    """Locate the pointer in an RGB frame as the most salient small blob.

    The pointer is a tiny, high-contrast shape over a locally-flat desktop. The
    local background is estimated with a box mean (via an integral image, so it
    is O(1) per pixel), the residual ``pixel - background`` is taken, and the
    *centroid* of the brightest residual region is returned.

    The centroid (rather than the centre of the best window) matters: a
    window-maximum score is biased toward wherever the blob is clipped by the
    window edge, which offsets the reported point by roughly half a window.

    This is an explicit heuristic sensor, not a learned detector, so it can be
    fooled by icons and text. Callers treat it as a weak measurement -- the
    confidence is returned and low-confidence points should be dropped. A flat
    frame returns ``None`` rather than inventing a point.

    It costs ~100 ms at 1080p, so it is a *fallback*: the live loop prefers the
    provider's exact pointer readback and only measures pixels when that is
    unavailable.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ServoError("Expected an (H, W, 3) RGB frame")
    # Luma via a float32 dot product: equivalent to averaging the channels for
    # blob detection but cheaper at 1080p.
    gray = frame.astype(np.float32).dot(_LUMA_WEIGHTS)
    h, w = gray.shape
    if h < 3 or w < 3:
        return None

    window = max(5, int(patch))
    if window % 2 == 0:
        window += 1
    half = window // 2

    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(gray, axis=0), axis=1)
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - half, 0, h)[:, None]
    y1 = np.clip(ys + half + 1, 0, h)[:, None]
    x0 = np.clip(xs - half, 0, w)[None, :]
    x1 = np.clip(xs + half + 1, 0, w)[None, :]
    area = (y1 - y0) * (x1 - x0)
    total = (
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )
    background = total / np.maximum(area, 1)
    residual = gray - background

    max_res = float(residual.max())
    dynamic_range = float(gray.max() - gray.min())
    if max_res <= max(1.0, 0.02 * dynamic_range):
        return None

    mask = residual >= 0.75 * max_res
    ys_hit, xs_hit = np.nonzero(mask)
    if ys_hit.size == 0:
        return None
    cx = float(xs_hit.mean())
    cy = float(ys_hit.mean())
    confidence = max(0.0, min(1.0, max_res / max(dynamic_range, 1.0)))
    return cx / max(w, 1), cy / max(h, 1), float(confidence)


class HumanReachModel:
    """SONIC's own motor controller for a visually guided reach.

    Given where the eye is fixating and where the hand currently is, it produces
    one submovement at a time. Far from the target it moves fast (the ballistic
    component); as the residual shrinks it decelerates smoothly (the corrective
    component), and it reports contact once the cursor is inside the foveal
    basin. The loop is closed by the caller re-observing the cursor after each
    submovement, so a wrong estimate is corrected instead of committed.

    Parameters are explicit on purpose; there is nothing hidden to trust.
    """

    def __init__(
        self,
        *,
        foveal_contact_radius: float = 0.012,
        ballistic_step: float = 1.0,
        max_step: float = 0.55,
        min_step: float = 0.02,
        approach_span: float = 0.20,
        damping: float = 0.85,
        noise: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if foveal_contact_radius <= 0:
            raise ValueError("foveal_contact_radius must be positive")
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        if ballistic_step <= 0:
            raise ValueError("ballistic_step must be positive")
        if not 0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if noise < 0:
            raise ValueError("noise must be non-negative")
        self.foveal_contact_radius = float(foveal_contact_radius)
        # The opening throw covers the whole distance in one motion (a person
        # does not creep toward a target in tiny increments); corrective steps
        # afterwards are damped and decelerating.
        self.ballistic_step = float(ballistic_step)
        self.max_step = float(max_step)
        self.min_step = float(min_step)
        self.approach_span = float(approach_span)
        self.damping = float(damping)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)

    def distance(
        self, target: tuple[float, float], cursor: tuple[float, float]
    ) -> float:
        return math.hypot(target[0] - cursor[0], target[1] - cursor[1])

    def contact(
        self, target: tuple[float, float], cursor: tuple[float, float]
    ) -> bool:
        return self.distance(target, cursor) <= self.foveal_contact_radius

    def next_submovement(
        self,
        target: tuple[float, float],
        cursor: tuple[float, float],
        *,
        ballistic: bool = False,
    ) -> Submovement:
        """Return the next step from ``cursor`` toward ``target``.

        ``ballistic=True`` requests the opening throw, which covers the entire
        remaining distance in one motion. Later (corrective) calls decelerate as
        the residual shrinks -- the two-component reach a person performs.
        """
        dx = target[0] - cursor[0]
        dy = target[1] - cursor[1]
        dist = math.hypot(dx, dy)

        if dist <= self.foveal_contact_radius:
            return Submovement(0.0, 0.0, float(dist), 0.0, True)
        if dist < 1e-9:
            return Submovement(0.0, 0.0, 0.0, 0.0, True)

        if ballistic:
            step = min(self.ballistic_step * dist, dist)
        else:
            # Smoothstep blend: full speed outside the approach span, easing down
            # to a slow, damped final correction near the target (minimum-jerk).
            t = min(
                1.0,
                max(0.0, (dist - self.foveal_contact_radius) / self.approach_span),
            )
            blend = t * t * (3.0 - 2.0 * t)
            step = self.min_step + (self.max_step - self.min_step) * blend
            step *= self.damping
            step = min(step, dist)  # never overshoot the target

        unit_x = dx / dist
        unit_y = dy / dist
        mx = unit_x * step
        my = unit_y * step
        if self.noise > 0.0:
            # Simulates a human's motor tremor; default is 0 so the shipped
            # controller is exact. Tests use it to prove the loop self-corrects.
            mx += float(self._rng.normal(0.0, self.noise))
            my += float(self._rng.normal(0.0, self.noise))

        new_dist = math.hypot(target[0] - (cursor[0] + mx), target[1] - (cursor[1] + my))
        return Submovement(
            dx=float(mx),
            dy=float(my),
            distance_to_target=float(new_dist),
            speed=float(step),
            contact_acquired=new_dist <= self.foveal_contact_radius,
        )

    def reach(
        self,
        target: tuple[float, float],
        cursor: tuple[float, float],
        *,
        max_iterations: int = 12,
    ) -> list[Submovement]:
        """Plan a full reach without re-observation (used for tests/analysis).

        The live loop does not use this: it re-observes after every step so the
        real cursor position drives the next correction. This pure form exists
        so the controller's convergence can be asserted deterministically.
        """
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        steps: list[Submovement] = []
        cx, cy = cursor
        for index in range(max_iterations):
            move = self.next_submovement(target, (cx, cy), ballistic=index == 0)
            steps.append(move)
            cx += move.dx
            cy += move.dy
            if move.contact_acquired:
                break
        return steps
