"""Retired Milestone 7 simulation adapter.

Historical runs used synthetic capture and virtual input. The execution entry
point now fails closed; this class cannot produce live GUI certification.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from avcs_m7_validation.contracts import (
    ActionStatus,
    AgentTask,
    ExecutionResult,
    Mode,
)
from avcs.action_broker import (
    ActionBroker,
    ActionClass,
    ActionRequest,
    ActionSafetyController,
)
from forge.vision.cortex.motor.driver import VirtualCursorDriver
from avcs.capture_adapter import InteractiveSyntheticCaptureAdapter
from avcs.human_eye import HumanEye


class RepositoryValidationAdapter:
    """Retired fixture adapter; it is not an autonomous GUI task executor."""

    def __init__(
        self,
        broker: Optional[ActionBroker] = None,
        provider: str = "avcs-repository",
        model: str = "avcs-v0.2.0",
    ) -> None:
        self.provider = provider
        self.model = model
        if broker is not None:
            self.broker = broker
            self.capture = getattr(broker.eye, "_source", None)
        else:
            self.capture = InteractiveSyntheticCaptureAdapter(width=640, height=480)
            eye = HumanEye(source=self.capture, device="cpu", threaded_capture=False)
            driver = VirtualCursorDriver(failsafe=False)
            # Link simulated physical action dispatch directly to interactive visual motion
            orig_click = driver.click
            def _click_with_visual_pulse(*args: Any, **kwargs: Any) -> None:
                if self.capture is not None and hasattr(self.capture, "trigger_motion"):
                    self.capture.trigger_motion(0.5, 0.5)
                return orig_click(*args, **kwargs)
            driver.click = _click_with_visual_pulse  # type: ignore

            orig_press_key = driver.press_key
            def _press_key_with_visual_pulse(*args: Any, **kwargs: Any) -> None:
                if self.capture is not None and hasattr(self.capture, "trigger_motion"):
                    self.capture.trigger_motion(0.5, 0.5)
                return orig_press_key(*args, **kwargs)
            driver.press_key = _press_key_with_visual_pulse  # type: ignore

            safety = ActionSafetyController()
            self.broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

    def execute(
        self,
        *,
        run_id: str,
        task: AgentTask,
        mode: Mode,
        seed: int,
        timeout_s: float,
    ) -> ExecutionResult:
        """Refuse certification when no goal-conditioned visual worker is wired."""
        t0 = time.perf_counter()
        return ExecutionResult(
            status=ActionStatus.FAILED,
            elapsed_ms=max(0.0, (time.perf_counter() - t0) * 1000),
            action_count=0,
            evidence_method="none",
            detail_code="autonomous_gui_worker_unavailable",
            metadata={
                "provider": self.provider,
                "model": self.model,
                "mode": mode.value,
                "seed": seed,
                "provenance": "simulation_fixture",
                "certification_eligible": False,
            },
        )


def create_adapter() -> RepositoryValidationAdapter:
    """Factory producing the repository execution adapter."""
    return RepositoryValidationAdapter()
