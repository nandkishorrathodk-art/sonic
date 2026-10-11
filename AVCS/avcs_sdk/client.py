"""High-level async AVCS client with strict freshness and confirmation rules."""
from __future__ import annotations

import asyncio
import time
from typing import Sequence

from .backend import AVCSBackend
from .errors import (
    BackendActionError, BackendContractError, ClientNotConnectedError,
    DispatchState, DuplicateActionIdError, ObservationNotKnownError,
    StaleTargetError,
)
from .models import (
    ActionClass, ActionKind, ActionRequest, ActionResult, ActionStatus,
    NormalizedPoint, Observation, RiskLevel, Target,
)


class AVCSClient:
    """Provider-neutral client façade around a broker-backed local runtime.

    The client never retries actions. Re-using the same action_id and identical
    request returns the previous result; re-using it with a different request
    raises DuplicateActionIdError.
    """

    def __init__(
        self,
        backend: AVCSBackend,
        *,
        max_observation_age_s: float = 2.0,
        minimum_target_confidence: float = 0.70,
        observation_history_size: int = 32,
    ) -> None:
        if max_observation_age_s <= 0:
            raise ValueError("max_observation_age_s must be positive")
        if not 0 <= minimum_target_confidence <= 1:
            raise ValueError("minimum_target_confidence must be in [0, 1]")
        if observation_history_size < 1:
            raise ValueError("observation_history_size must be >= 1")
        self._backend = backend
        self.max_observation_age_s = max_observation_age_s
        self.minimum_target_confidence = minimum_target_confidence
        self.observation_history_size = observation_history_size
        self._connected = False
        self._observations: dict[str, Observation] = {}
        self._latest_observation_id: str | None = None
        self._completed: dict[str, tuple[ActionRequest, ActionResult]] = {}
        self._inflight: dict[str, tuple[ActionRequest, asyncio.Task[ActionResult]]] = {}
        self._action_lock = asyncio.Lock()

    async def __aenter__(self) -> "AVCSClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._connected:
            return
        await self._backend.connect()
        self._connected = True

    async def close(self) -> None:
        if not self._connected:
            return
        # Finish known action operations before tearing down the backend. Each
        # backend must enforce request.timeout_s and surface uncertain dispatch
        # as AMBIGUOUS; closing must not interrupt an in-flight OS action.
        pending = [task for _, task in self._inflight.values()]
        if pending:
            await asyncio.gather(*(asyncio.shield(task) for task in pending), return_exceptions=True)
        try:
            await self._backend.close()
        finally:
            self._connected = False
            self._observations.clear()
            self._latest_observation_id = None

    async def observe(self) -> Observation:
        self._require_connected()
        observation = await self._backend.observe()
        if not isinstance(observation, Observation):
            raise BackendContractError("backend.observe() must return Observation")
        now = time.time()
        if observation.captured_at > now + 1.0:
            raise BackendContractError("backend returned an observation timestamp in the future")
        if now - observation.captured_at > self.max_observation_age_s:
            raise BackendContractError("backend returned an already-stale observation")
        self._observations[observation.observation_id] = observation
        self._latest_observation_id = observation.observation_id
        while len(self._observations) > self.observation_history_size:
            oldest_id = min(self._observations, key=lambda key: self._observations[key].captured_at)
            del self._observations[oldest_id]
        return observation

    async def find_targets(
        self,
        description: str,
        *,
        observation: Observation | None = None,
    ) -> list[Target]:
        self._require_connected()
        if not description.strip():
            raise ValueError("description cannot be empty")
        if observation is None:
            if self._latest_observation_id is None:
                observation = await self.observe()
            else:
                observation = self._observations[self._latest_observation_id]
        known = self._observations.get(observation.observation_id)
        if known != observation:
            raise ObservationNotKnownError("Observation was not created by this client or is no longer in history")
        self._validate_observation_freshness(observation)
        if observation.observation_id != self._latest_observation_id:
            raise StaleTargetError("Cannot ground against an observation older than the latest captured frame")
        targets = await self._backend.find_targets(description, observation)
        out: list[Target] = []
        for target in targets:
            if not isinstance(target, Target):
                raise BackendContractError("backend.find_targets() returned a non-Target value")
            if target.observation_id != observation.observation_id:
                raise BackendContractError("target refers to a different observation than the one queried")
            out.append(target)
        return out

    async def act(self, request: ActionRequest) -> ActionResult:
        self._require_connected()
        async with self._action_lock:
            # Resolve IDs before freshness checks: a retry of a completed action
            # must return its original result, never re-dispatch or fail merely
            # because the observation aged while the caller was waiting.
            cached = self._completed.get(request.action_id)
            if cached is not None:
                cached_request, cached_result = cached
                if cached_request != request:
                    raise DuplicateActionIdError("action_id already used for a different request")
                return cached_result
            inflight = self._inflight.get(request.action_id)
            if inflight is not None:
                inflight_request, task = inflight
                if inflight_request != request:
                    raise DuplicateActionIdError("action_id already in flight for a different request")
            else:
                if request.target is not None:
                    blocked = self._validate_target_for_action(request.target, request.action_id)
                    if blocked is not None:
                        # Reserve the action ID even for policy rejections.
                        self._completed[request.action_id] = (request, blocked)
                        return blocked
                elif request.point is not None:
                    self._validate_point_observation(request.observation_id)
                task = asyncio.create_task(self._execute_and_cache(request))
                self._inflight[request.action_id] = (request, task)
        # Shield the backend operation from cancellation of the caller. If the
        # caller cancels, the task still resolves/caches the terminal result; a
        # same-ID retry joins that task rather than dispatching a duplicate.
        return await asyncio.shield(task)

    async def _execute_and_cache(self, request: ActionRequest) -> ActionResult:
        started = time.perf_counter()
        try:
            result = await self._backend.execute(request)
        except BackendActionError as exc:
            status = ActionStatus.FAILED if exc.dispatch_state == DispatchState.NOT_DISPATCHED else ActionStatus.AMBIGUOUS
            result = ActionResult(
                action_id=request.action_id,
                status=status,
                duration_ms=(time.perf_counter() - started) * 1000,
                error_code=exc.error_code,
                message=str(exc),
                attempt_count=0 if status == ActionStatus.FAILED else 1,
            )
        except asyncio.CancelledError:
            # If the backend coroutine itself cancels after execute started, the
            # dispatch may have occurred. Preserve a terminal ambiguous result.
            result = ActionResult(
                action_id=request.action_id,
                status=ActionStatus.AMBIGUOUS,
                duration_ms=(time.perf_counter() - started) * 1000,
                error_code="BACKEND_EXECUTION_CANCELLED",
                message="Backend execution was cancelled; dispatch state is unknown",
                attempt_count=1,
            )
        except Exception as exc:  # noqa: BLE001 - uncertain dispatch must be conservative
            result = ActionResult(
                action_id=request.action_id,
                status=ActionStatus.AMBIGUOUS,
                duration_ms=(time.perf_counter() - started) * 1000,
                error_code="UNCLASSIFIED_BACKEND_EXCEPTION",
                message=f"Backend raised {type(exc).__name__}; dispatch state unknown",
                attempt_count=1,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            result = self._validate_result(request, result, elapsed_ms)
        except BackendContractError as exc:
            # A malformed result cannot prove whether dispatch took place.
            # Cache AMBIGUOUS so a caller cannot accidentally duplicate it.
            result = ActionResult(
                action_id=request.action_id,
                status=ActionStatus.AMBIGUOUS,
                duration_ms=elapsed_ms,
                error_code="BACKEND_CONTRACT_VIOLATION",
                message=str(exc),
                attempt_count=1,
            )
        self._completed[request.action_id] = (request, result)
        self._inflight.pop(request.action_id, None)
        return result

    async def click(
        self,
        *,
        target: Target | None = None,
        point: NormalizedPoint | None = None,
        observation: Observation | None = None,
        expected_postcondition: str | None = None,
        action_class: ActionClass = ActionClass.NON_IDEMPOTENT,
        risk: RiskLevel = RiskLevel.NORMAL,
        authorization_ref: str | None = None,
        timeout_s: float = 5.0,
        action_id: str | None = None,
    ) -> ActionResult:
        return await self.act(ActionRequest(
            kind=ActionKind.CLICK, target=target, point=point,
            observation_id=(observation.observation_id if observation is not None else None),
            expected_postcondition=expected_postcondition, action_class=action_class,
            risk=risk, authorization_ref=authorization_ref, timeout_s=timeout_s,
            **({"action_id": action_id} if action_id else {}),
        ))

    async def type_text(
        self,
        text: str,
        *,
        expected_postcondition: str | None = None,
        action_class: ActionClass = ActionClass.NON_IDEMPOTENT,
        timeout_s: float = 5.0,
        action_id: str | None = None,
    ) -> ActionResult:
        return await self.act(ActionRequest(
            kind=ActionKind.TYPE_TEXT, text=text,
            expected_postcondition=expected_postcondition, action_class=action_class,
            timeout_s=timeout_s, **({"action_id": action_id} if action_id else {}),
        ))

    async def press_key(
        self,
        key: str,
        *,
        expected_postcondition: str | None = None,
        action_class: ActionClass = ActionClass.NON_IDEMPOTENT,
        timeout_s: float = 5.0,
        action_id: str | None = None,
    ) -> ActionResult:
        return await self.act(ActionRequest(
            kind=ActionKind.PRESS_KEY, key=key,
            expected_postcondition=expected_postcondition, action_class=action_class,
            timeout_s=timeout_s, **({"action_id": action_id} if action_id else {}),
        ))

    def _validate_target_for_action(self, target: Target, action_id: str) -> ActionResult | None:
        known = self._observations.get(target.observation_id)
        if known is None:
            raise ObservationNotKnownError("Target observation is unknown or evicted")
        self._validate_observation_freshness(known)
        if target.observation_id != self._latest_observation_id:
            raise StaleTargetError("Target is not from the latest observation; observe and re-ground first")
        if target.confidence < self.minimum_target_confidence:
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.BLOCKED,
                duration_ms=0.0,
                error_code="TARGET_CONFIDENCE_BELOW_THRESHOLD",
                message=f"Target confidence {target.confidence:.3f} is below configured threshold {self.minimum_target_confidence:.3f}",
                attempt_count=0,
            )
        return None

    def _validate_observation_freshness(self, observation: Observation) -> None:
        age = time.time() - observation.captured_at
        if age < -1.0 or age > self.max_observation_age_s:
            raise StaleTargetError(f"Observation age {age:.3f}s exceeds freshness policy")

    def _validate_point_observation(self, observation_id: str | None) -> None:
        if not observation_id:
            raise ObservationNotKnownError("Coordinate actions require an observation_id")
        observation = self._observations.get(observation_id)
        if observation is None:
            raise ObservationNotKnownError("Coordinate action observation is unknown or evicted")
        self._validate_observation_freshness(observation)
        if observation_id != self._latest_observation_id:
            raise StaleTargetError("Coordinate action was derived from an old observation; observe and re-ground first")

    def _validate_result(self, request: ActionRequest, result: ActionResult, elapsed_ms: float) -> ActionResult:
        if not isinstance(result, ActionResult):
            raise BackendContractError("backend.execute() must return ActionResult")
        if result.action_id != request.action_id:
            raise BackendContractError("backend result action_id does not match request")
        if result.status == ActionStatus.CONFIRMED:
            if result.evidence is None or not result.evidence.verified:
                return ActionResult(
                    action_id=request.action_id,
                    status=ActionStatus.AMBIGUOUS,
                    duration_ms=max(result.duration_ms, elapsed_ms),
                    evidence=result.evidence,
                    error_code="CONFIRMATION_WITHOUT_VERIFIED_EVIDENCE",
                    message="Backend reported CONFIRMED without verified postcondition evidence; downgraded to AMBIGUOUS",
                    attempt_count=result.attempt_count,
                )
        return result

    def _require_connected(self) -> None:
        if not self._connected:
            raise ClientNotConnectedError("Client is not connected. Use `async with AVCSClient(...)` or await client.connect() first")
