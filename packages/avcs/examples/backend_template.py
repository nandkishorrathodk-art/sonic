"""Template for adapting the real AVCS runtime to the public SDK.

Replace each NotImplementedError with calls to the existing capture, grounding,
ActionBroker and postcondition verifier. Never expose the fixture oracle to this
adapter and never dispatch raw OS input around ActionBroker.
"""
from __future__ import annotations
from typing import Sequence

from avcs_sdk import ActionRequest, ActionResult, Observation, Target


class RepositoryAVCSBackend:
    async def connect(self) -> None:
        raise NotImplementedError("Initialize the repository capture and broker here")

    async def observe(self) -> Observation:
        raise NotImplementedError("Map a real captured frame + runtime state to Observation")

    async def find_targets(self, description: str, observation: Observation) -> Sequence[Target]:
        raise NotImplementedError("Ground targets without access to benchmark oracle state")

    async def execute(self, request: ActionRequest) -> ActionResult:
        raise NotImplementedError("Translate to ActionBroker request and map independently verified result")

    async def close(self) -> None:
        raise NotImplementedError("Release capture/session resources")
