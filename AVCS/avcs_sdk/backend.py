"""Backend protocol boundary. Implement this against AVCS ActionBroker."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from .models import ActionRequest, ActionResult, Observation, Target


@runtime_checkable
class AVCSBackend(Protocol):
    """Adapter contract for a concrete local runtime.

    `execute` must route every operation through the bounded action broker.
    It must not dispatch an OS action directly from an LLM/provider callback.
    After possible dispatch, timeouts/errors must resolve as AMBIGUOUS unless
    the backend can prove the action was not dispatched.
    """
    async def connect(self) -> None: ...
    async def observe(self) -> Observation: ...
    async def find_targets(self, description: str, observation: Observation) -> Sequence[Target]: ...
    async def execute(self, request: ActionRequest) -> ActionResult: ...
    async def close(self) -> None: ...
