from __future__ import annotations
import time
from typing import Sequence

import pytest

from avcs_sdk import (
    ActionClass, ActionKind, ActionRequest, ActionResult, ActionStatus,
    AVCSClient, BackendActionError, BoundingBox, DispatchState,
    DuplicateActionIdError, NormalizedPoint, Observation, RiskLevel,
    StaleTargetError, Target, TargetProvenance, VerificationEvidence,
)


class FakeBackend:
    def __init__(self):
        self.connected = False
        self.execute_calls = 0
        self.targets: list[Target] = []
        self.next_result: ActionResult | None = None
        self.next_error: Exception | None = None
        self.observation_age_s = 0.0

    async def connect(self):
        self.connected = True

    async def close(self):
        self.connected = False

    async def observe(self):
        return Observation(
            observation_id=f"obs-{time.time_ns()}",
            captured_at=time.time() - self.observation_age_s,
            width=1920, height=1080, active_window="Fixture",
        )

    async def find_targets(self, description: str, observation: Observation) -> Sequence[Target]:
        return [Target(t.target_id, observation.observation_id, t.label, t.bounds, t.confidence, t.provenance) for t in self.targets]

    async def execute(self, request: ActionRequest):
        self.execute_calls += 1
        if self.next_error is not None:
            exc, self.next_error = self.next_error, None
            raise exc
        if self.next_result is not None:
            result, self.next_result = self.next_result, None
            return ActionResult(
                action_id=request.action_id, status=result.status,
                duration_ms=result.duration_ms, evidence=result.evidence,
                error_code=result.error_code, message=result.message,
                attempt_count=result.attempt_count,
            )
        return ActionResult(
            action_id=request.action_id, status=ActionStatus.CONFIRMED,
            duration_ms=1.0,
            evidence=VerificationEvidence(True, "fixture_postcondition", time.time(), {"state": "expected"}),
        )


def make_target(observation_id: str, confidence: float = 0.99) -> Target:
    return Target(
        target_id="save",
        observation_id=observation_id,
        label="Save",
        bounds=BoundingBox(0.1, 0.1, 0.1, 0.05),
        confidence=confidence,
        provenance=TargetProvenance.VISION,
    )


@pytest.mark.asyncio
async def test_requires_connection():
    client = AVCSClient(FakeBackend())
    with pytest.raises(Exception, match="connected"):
        await client.observe()


@pytest.mark.asyncio
async def test_context_lifecycle_and_observation():
    backend = FakeBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        assert obs.width == 1920
        assert backend.connected
    assert not backend.connected


@pytest.mark.asyncio
async def test_finds_targets_bound_to_same_observation():
    backend = FakeBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        backend.targets = [make_target(obs.observation_id)]
        results = await client.find_targets("Save", observation=obs)
        assert len(results) == 1
        assert results[0].observation_id == obs.observation_id


@pytest.mark.asyncio
async def test_target_from_prior_observation_is_rejected():
    backend = FakeBackend()
    async with AVCSClient(backend) as client:
        old = await client.observe()
        backend.targets = [make_target(old.observation_id)]
        target = (await client.find_targets("Save", observation=old))[0]
        await client.observe()
        with pytest.raises(StaleTargetError):
            await client.click(target=target)


@pytest.mark.asyncio
async def test_low_confidence_target_is_blocked_without_dispatch():
    backend = FakeBackend()
    async with AVCSClient(backend, minimum_target_confidence=0.95) as client:
        obs = await client.observe()
        backend.targets = [make_target(obs.observation_id, confidence=0.6)]
        target = (await client.find_targets("Save", observation=obs))[0]
        req = ActionRequest(kind=ActionKind.CLICK, target=target, action_id="low-conf")
        result = await client.act(req)
        assert result.status is ActionStatus.BLOCKED
        assert result.action_id == "low-conf"
        assert backend.execute_calls == 0


@pytest.mark.asyncio
async def test_confirmed_without_evidence_is_downgraded():
    backend = FakeBackend()
    backend.next_result = ActionResult("ignored", ActionStatus.CONFIRMED, 1.0, evidence=None)
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(point=NormalizedPoint(0.5, 0.5), observation=obs)
        assert result.status is ActionStatus.AMBIGUOUS
        assert result.error_code == "CONFIRMATION_WITHOUT_VERIFIED_EVIDENCE"


@pytest.mark.asyncio
async def test_unverified_evidence_does_not_confirm():
    backend = FakeBackend()
    backend.next_result = ActionResult(
        "ignored", ActionStatus.CONFIRMED, 1.0,
        evidence=VerificationEvidence(False, "pixel_delta", time.time()),
    )
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(point=NormalizedPoint(0.5, 0.5), observation=obs)
        assert result.status is ActionStatus.AMBIGUOUS


@pytest.mark.asyncio
async def test_definitely_not_dispatched_error_is_failed():
    backend = FakeBackend()
    backend.next_error = BackendActionError("not sent", dispatch_state=DispatchState.NOT_DISPATCHED)
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(point=NormalizedPoint(0.5, 0.5), observation=obs)
        assert result.status is ActionStatus.FAILED
        assert result.attempt_count == 0


@pytest.mark.asyncio
async def test_maybe_dispatched_error_is_ambiguous_and_not_retried():
    backend = FakeBackend()
    backend.next_error = BackendActionError("connection lost", dispatch_state=DispatchState.MAY_HAVE_DISPATCHED)
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        request = ActionRequest(kind=ActionKind.CLICK, point=NormalizedPoint(0.5, 0.5), observation_id=obs.observation_id, action_id="maybe")
        result1 = await client.act(request)
        result2 = await client.act(request)
        assert result1.status is ActionStatus.AMBIGUOUS
        assert result2 == result1
        assert backend.execute_calls == 1


@pytest.mark.asyncio
async def test_unclassified_backend_exception_is_ambiguous():
    backend = FakeBackend()
    backend.next_error = RuntimeError("unknown post-dispatch failure")
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(point=NormalizedPoint(0.5, 0.5), observation=obs)
        assert result.status is ActionStatus.AMBIGUOUS


@pytest.mark.asyncio
async def test_reusing_action_id_with_different_request_is_rejected():
    backend = FakeBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        await client.act(ActionRequest(kind=ActionKind.CLICK, point=NormalizedPoint(0.5, 0.5), observation_id=obs.observation_id, action_id="same"))
        with pytest.raises(DuplicateActionIdError):
            await client.act(ActionRequest(kind=ActionKind.CLICK, point=NormalizedPoint(0.2, 0.2), observation_id=obs.observation_id, action_id="same"))
        assert backend.execute_calls == 1


def test_high_risk_requires_authorization_reference():
    with pytest.raises(ValueError, match="authorization_ref"):
        ActionRequest(kind=ActionKind.CLICK, point=NormalizedPoint(0.5, 0.5), observation_id="obs", risk=RiskLevel.HIGH)


def test_coordinate_contract_rejects_invalid_values():
    with pytest.raises(ValueError):
        NormalizedPoint(1.01, 0.5)
    with pytest.raises(ValueError):
        BoundingBox(0.9, 0.9, 0.2, 0.2)


def test_action_request_validates_kind_payload():
    with pytest.raises(ValueError, match="press_key"):
        ActionRequest(kind=ActionKind.PRESS_KEY)
    with pytest.raises(ValueError, match="scroll"):
        ActionRequest(kind=ActionKind.SCROLL)

@pytest.mark.asyncio
async def test_caller_cancellation_does_not_duplicate_inflight_action():
    import asyncio

    class SlowBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def execute(self, request):
            self.execute_calls += 1
            self.started.set()
            await self.release.wait()
            return ActionResult(
                action_id=request.action_id, status=ActionStatus.CONFIRMED,
                duration_ms=1.0,
                evidence=VerificationEvidence(True, "postcondition", time.time()),
            )

    backend = SlowBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        request = ActionRequest(kind=ActionKind.CLICK, point=NormalizedPoint(0.5, 0.5), observation_id=obs.observation_id, action_id="cancel-safe")
        first = asyncio.create_task(client.act(request))
        await backend.started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        backend.release.set()
        result = await client.act(request)
        assert result.status is ActionStatus.CONFIRMED
        assert backend.execute_calls == 1

@pytest.mark.asyncio
async def test_malformed_backend_result_becomes_cached_ambiguous():
    class MalformedBackend(FakeBackend):
        async def execute(self, request):
            self.execute_calls += 1
            return {"status": "confirmed"}  # type: ignore[return-value]

    backend = MalformedBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        req = ActionRequest(kind=ActionKind.CLICK, point=NormalizedPoint(0.5, 0.5), observation_id=obs.observation_id, action_id="malformed")
        first = await client.act(req)
        second = await client.act(req)
        assert first.status is ActionStatus.AMBIGUOUS
        assert first.error_code == "BACKEND_CONTRACT_VIOLATION"
        assert second == first
        assert backend.execute_calls == 1


def test_public_models_reject_raw_enum_strings():
    with pytest.raises(TypeError, match="ActionKind"):
        ActionRequest(kind="click", point=NormalizedPoint(0.2, 0.3), observation_id="obs")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ActionStatus"):
        ActionResult(action_id="a", status="confirmed", duration_ms=1)  # type: ignore[arg-type]
