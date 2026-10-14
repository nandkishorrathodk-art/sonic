"""Retired Milestone 6 adapter.

The historical implementation bypassed the desktop and edited evaluated files
directly. Its public execution entry point now fails closed until a real visual
GUI task worker is available.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from avcs_pilot.contracts import ActionStatus, AgentExecution, PublicTask


class RepositoryPilotAdapter:
    """Retired adapter that refuses to claim autonomous GUI task completion."""

    def __init__(
        self,
        backend: Optional[Any] = None,
        provider: str = "avcs-repository",
        model: str = "avcs-v0.2.0",
    ) -> None:
        self.backend = backend
        self.provider = provider
        self.model = model

    async def execute(self, task: PublicTask) -> AgentExecution:
        """Fail closed because no goal-conditioned visual GUI worker is wired."""
        started = time.perf_counter()

        if task.risk_level.upper() != "LOW":
            return AgentExecution(
                status=ActionStatus.BLOCKED,
                action_count=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                provider=self.provider,
                model=self.model,
                message="Non-LOW risk task rejected by pilot safety policy",
                diagnostics={"error_code": "HIGH_RISK_BLOCKED", "dispatch_accepted": False},
            )

        return AgentExecution(
            status=ActionStatus.FAILED,
            action_count=0,
            duration_ms=(time.perf_counter() - started) * 1000,
            provider=self.provider,
            model=self.model,
            message=(
                "No goal-conditioned visual GUI worker is configured. "
                "No desktop action or file operation was performed."
            ),
            diagnostics={
                "error_code": "AUTONOMOUS_GUI_WORKER_UNAVAILABLE",
                "dispatch_accepted": False,
                "certification_eligible": False,
            },
        )


def create_adapter() -> RepositoryPilotAdapter:
    """Plugin factory for the retired fail-closed pilot adapter."""
    return RepositoryPilotAdapter()
