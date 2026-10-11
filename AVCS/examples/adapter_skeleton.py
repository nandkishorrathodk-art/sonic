"""Safe integration template connecting the pilot harness to the AVCS runtime.

Wraps avcs_pilot.repository_adapter.RepositoryPilotAdapter, enforcing zero-oracle
boundary, action budgets, and ActionBroker verification.
"""
from __future__ import annotations

from avcs_pilot.contracts import AgentExecution, PublicTask
from avcs_pilot.repository_adapter import (
    RepositoryPilotAdapter as CoreRepositoryPilotAdapter,
    create_adapter as core_create_adapter,
)


class RepositoryPilotAdapter:
    """Adapter wrapping the concrete AVCS repository runtime."""

    def __init__(self) -> None:
        self._core = core_create_adapter()

    async def execute(self, task: PublicTask) -> AgentExecution:
        """Execute the public task only; no scenario/evaluator object is provided."""
        return await self._core.execute(task)


def create_adapter() -> RepositoryPilotAdapter:
    """Factory producing the wired repository pilot adapter."""
    return RepositoryPilotAdapter()

