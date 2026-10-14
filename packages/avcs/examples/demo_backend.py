"""In-memory demo backend. Does not control the operating system."""
from __future__ import annotations
import time
from typing import Sequence
from uuid import uuid4

from avcs_sdk import (
    ActionKind, ActionRequest, ActionResult, ActionStatus, BoundingBox,
    Observation, Target, TargetProvenance, VerificationEvidence,
)


class DemoBackend:
    def __init__(self):
        self._connected = False
        self._saved = False

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def observe(self) -> Observation:
        if not self._connected:
            raise RuntimeError("demo backend is not connected")
        return Observation(
            observation_id=str(uuid4()), captured_at=time.time(),
            width=1280, height=720, active_window="AVCS SDK Demo",
        )

    async def find_targets(self, description: str, observation: Observation) -> Sequence[Target]:
        if "save" not in description.casefold():
            return []
        return [Target(
            target_id="demo-save-button", observation_id=observation.observation_id,
            label="Save", bounds=BoundingBox(0.80, 0.08, 0.12, 0.06),
            confidence=0.99, provenance=TargetProvenance.VISION,
        )]

    async def execute(self, request: ActionRequest) -> ActionResult:
        if not self._connected:
            raise RuntimeError("demo backend is not connected")
        if request.kind is ActionKind.CLICK and request.target and request.target.target_id == "demo-save-button":
            self._saved = True
            return ActionResult(
                action_id=request.action_id, status=ActionStatus.CONFIRMED,
                duration_ms=2.0,
                evidence=VerificationEvidence(
                    verified=self._saved, method="demo_application_state",
                    observed_at=time.time(), details={"saved": str(self._saved).lower()},
                ),
            )
        return ActionResult(
            action_id=request.action_id, status=ActionStatus.BLOCKED,
            duration_ms=0.1, error_code="DEMO_UNSUPPORTED_ACTION",
            message="Demo backend supports only its synthetic Save button",
            attempt_count=0,
        )
