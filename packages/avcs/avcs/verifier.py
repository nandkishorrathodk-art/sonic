"""Visual Ripple Assertion Verifier & Self-Healing Recovery Ladder.

Implements biological post-action visual verification:
1. Optical Flow Ripple: Measures local kinetic energy burst around action point.
2. Surprise Pulse: Evaluates predictive coding temporal prediction error (S_t).
3. State Transition: Verifies UI structural changes via DOM-less UIML diffs (Delta_visual).
4. 5-Tier Self-Healing Escalation Ladder: Automatically recovers from missed clicks,
   border deadzones, and delayed UI transitions.
"""
from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from avcs.human_eye import HumanEye, HumanEyeState
from avcs.memory.types import VisualDiff
from forge.vision.cortex.motor.driver import OSCursorDriver, get_default_driver


class VerificationStatus(enum.Enum):
    """Status code emitted by the Visual Ripple Verifier."""
    VERIFIED = "verified"                  # Tier 1: Action successfully verified on screen
    RETRY_MICRO_JITTER = "retry_jitter"    # Tier 2: Deadzone suspected; micro-offset retry
    KEYBOARD_FALLBACK = "key_fallback"     # Tier 3: Click unacknowledged; keyboard fallback
    SACCADIC_REGROUNDING = "regrounding"   # Tier 4: Target moved; gaze re-acquisition
    ESCALATE_BLOCKED = "escalate_blocked"  # Tier 5: Element unresponsive or modal blocked


@dataclass
class VerificationResult:
    """Comprehensive visual post-condition assertion result."""
    status: VerificationStatus
    action_name: str
    target_coord: Tuple[float, float]
    optical_ripple_energy: float
    surprise_pulse: float
    state_transition_detected: bool
    causal_summary: str
    recovery_attempt: int
    elapsed_ms: float
    diff: Optional[VisualDiff] = None

    @property
    def is_successful(self) -> bool:
        return self.status == VerificationStatus.VERIFIED


class VisualRippleVerifier:
    """Post-action visual assertion verifier and self-healing executor (<3ms evaluation)."""

    def __init__(
        self,
        eye: HumanEye,
        driver: Optional[OSCursorDriver] = None,
        ripple_radius: float = 0.08,  # ~8% of screen around target
        min_ripple_threshold: float = 0.015,
        min_surprise_threshold: float = 0.12,
    ) -> None:
        self.eye = eye
        self.driver = driver or get_default_driver()
        self.ripple_radius = ripple_radius
        self.min_ripple_threshold = min_ripple_threshold
        self.min_surprise_threshold = min_surprise_threshold

    def calculate_local_ripple(
        self,
        target: Tuple[float, float],
        motion_field: np.ndarray,
    ) -> float:
        """Computes kinetic optical flow energy within the target's visual neighborhood."""
        if motion_field is None or motion_field.ndim != 3:
            return 0.0

        rows, cols = motion_field.shape[:2]
        tx, ty = target
        center_r = int(ty * rows)
        center_c = int(tx * cols)

        radius_r = max(1, int(self.ripple_radius * rows))
        radius_c = max(1, int(self.ripple_radius * cols))

        r_min = max(0, center_r - radius_r)
        r_max = min(rows, center_r + radius_r + 1)
        c_min = max(0, center_c - radius_c)
        c_max = min(cols, center_c + radius_c + 1)

        patch = motion_field[r_min:r_max, c_min:c_max]
        if patch.size == 0:
            return 0.0

        # Sum of velocity magnitudes
        speeds = np.hypot(patch[..., 0], patch[..., 1])
        return float(np.mean(speeds))

    def verify_action(
        self,
        action_name: str,
        target: Tuple[float, float],
        pre_state: HumanEyeState,
        settle_ms: float = 60.0,
    ) -> VerificationResult:
        """Verifies if an action produced a physical optical ripple or UI state transition."""
        t0 = time.perf_counter()

        # Wait settlement period for UI animations to trigger
        if settle_ms > 0:
            time.sleep(settle_ms / 1000.0)

        # Perceive current screen state
        post_state = self.eye.observe()

        # 1. Local Optical Flow Ripple
        ripple_energy = self.calculate_local_ripple(target, post_state.motion_field)

        # 2. Photometric Local Delta against pre_state
        photometric_delta = 0.0
        if pre_state is not None and getattr(pre_state, "motion_field", None) is not None:
            pre_local = self.calculate_local_ripple(target, pre_state.motion_field)
            photometric_delta = abs(ripple_energy - pre_local)

        # 3. Predictive Coding Surprise Pulse
        surprise_pulse = post_state.surprise_score

        # 4. State Transition Detection (Delta_visual)
        diff = post_state.visual_diff
        structural_change = False
        if diff is not None:
            structural_change = bool(
                diff.added_elements or diff.removed_ids or diff.modified_elements
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Decision Logic: Did the screen react locally to the physical actuation?
        # Requires local ripple, structural change, local photometric delta, or localized surprise
        local_reaction = (
            ripple_energy >= self.min_ripple_threshold
            or structural_change
            or photometric_delta >= self.min_ripple_threshold
        )
        verified = (
            local_reaction
            or (surprise_pulse >= self.min_surprise_threshold and (ripple_energy > 0.005 or structural_change))
        )

        status = VerificationStatus.VERIFIED if verified else VerificationStatus.RETRY_MICRO_JITTER
        summary = (
            diff.causal_summary if diff and diff.causal_summary
            else ("Physical visual ripple verified." if verified else "No optical response detected.")
        )

        return VerificationResult(
            status=status,
            action_name=action_name,
            target_coord=target,
            optical_ripple_energy=ripple_energy,
            surprise_pulse=surprise_pulse,
            state_transition_detected=structural_change,
            causal_summary=summary,
            recovery_attempt=1,
            elapsed_ms=elapsed_ms,
            diff=diff,
        )

    def _move_cursor(self, x: float, y: float) -> None:
        if hasattr(self.driver, "move_to_normalized"):
            self.driver.move_to_normalized(x, y)
        elif hasattr(self.driver, "move_to"):
            self.driver.move_to(x, y)
        else:
            raise RuntimeError("Cursor driver does not support pointer movement.")

    def execute_and_verify(
        self,
        action_type: str,
        target: Tuple[float, float],
        text_content: Optional[str] = None,
        max_recovery_tier: int = 3,
        button: str = "left",
        keyboard_fallback_key: Optional[str] = None,
    ) -> VerificationResult:
        """Executes an action through the 5-Tier Self-Healing Escalation Ladder."""
        if action_type not in {"click", "double_click", "type"}:
            raise ValueError(f"Unsupported verified action type: {action_type}")
        if button not in {"left", "right", "middle"}:
            raise ValueError(f"Unsupported mouse button: {button}")
        t_global = time.perf_counter()
        tx, ty = target

        # Tier 1: Primary Action Execution
        pre_state = self.eye.observe()
        self._move_cursor(tx, ty)

        if action_type == "click":
            self.driver.click(button=button)
        elif action_type == "double_click":
            self.driver.double_click(button=button)
        elif action_type == "type":
            self.driver.click(button="left")
            time.sleep(0.04)
            if text_content and hasattr(self.driver, "type_text"):
                self.driver.type_text(text_content)
            elif text_content:
                raise RuntimeError("Cursor driver does not support text entry.")

        res = self.verify_action(f"{action_type}@{target}", target, pre_state)
        if res.is_successful or max_recovery_tier < 2:
            return res
        if action_type == "type":
            res.status = VerificationStatus.ESCALATE_BLOCKED
            res.causal_summary = (
                "Typing was dispatched but visual verification was inconclusive; "
                "it was not replayed to avoid duplicating text."
            )
            res.elapsed_ms = (time.perf_counter() - t_global) * 1000.0
            return res

        # Tier 2: Micro-Jitter Retry (Hitbox Border Offset)
        # Offset by +4 pixels (approx +0.003 normalized) to avoid dead border zones
        offsets = [(0.003, 0.003), (-0.003, -0.003), (0.003, -0.003)]
        for ox, oy in offsets:
            jitter_target = (float(np.clip(tx + ox, 0.0, 1.0)), float(np.clip(ty + oy, 0.0, 1.0)))
            pre_retry = self.eye.observe()
            self._move_cursor(jitter_target[0], jitter_target[1])

            if action_type == "double_click":
                self.driver.double_click(button=button)
            else:
                self.driver.click(button=button)

            res_jitter = self.verify_action(f"retry_{action_type}@{jitter_target}", jitter_target, pre_retry)
            if res_jitter.is_successful:
                res_jitter.status = VerificationStatus.VERIFIED
                res_jitter.recovery_attempt = 2
                res_jitter.causal_summary = f"Recovered via Tier 2 micro-jitter offset ({ox:+.3f},{oy:+.3f})."
                return res_jitter

        if max_recovery_tier < 3:
            res.status = VerificationStatus.RETRY_MICRO_JITTER
            return res

        # Keyboard activation is only safe when the caller names the intended key.
        if keyboard_fallback_key is not None and action_type == "click":
            pre_kb = self.eye.observe()
            if hasattr(self.driver, "press_key"):
                self.driver.press_key(keyboard_fallback_key)
            res_kb = self.verify_action(
                f"keyboard_{keyboard_fallback_key}", target, pre_kb
            )
            if res_kb.is_successful:
                res_kb.status = VerificationStatus.VERIFIED
                res_kb.recovery_attempt = 3
                res_kb.causal_summary = (
                    f"Recovered via explicit keyboard activation "
                    f"({keyboard_fallback_key})."
                )
                return res_kb

        if max_recovery_tier < 4:
            res.status = (
                VerificationStatus.KEYBOARD_FALLBACK
                if keyboard_fallback_key is not None
                else VerificationStatus.RETRY_MICRO_JITTER
            )
            return res

        # Tier 4: Saccadic Re-grounding (Re-acquire target)
        if action_type == "click" and hasattr(self.eye, "look_at") and self.eye._active_query:
            reground_res = self.eye.look_at(self.eye._active_query)
            if reground_res.target_coord != target:
                new_target = reground_res.target_coord
                pre_reground = self.eye.observe()
                self._move_cursor(new_target[0], new_target[1])
                self.driver.click(button=button)
                res_reground = self.verify_action(
                    f"reground_click@{new_target}", new_target, pre_reground
                )
                if res_reground.is_successful:
                    res_reground.status = VerificationStatus.VERIFIED
                    res_reground.recovery_attempt = 4
                    res_reground.causal_summary = f"Recovered via Tier 4 saccadic re-grounding to {new_target}."
                    return res_reground

        # Tier 5: Escalate Blocked
        res.status = VerificationStatus.ESCALATE_BLOCKED
        res.recovery_attempt = 5
        res.causal_summary = "All 5 recovery tiers exhausted; target unresponsive or blocked."
        res.elapsed_ms = (time.perf_counter() - t_global) * 1000.0
        return res
