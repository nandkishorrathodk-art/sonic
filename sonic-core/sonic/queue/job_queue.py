"""
SONIC-REDA — Redis-Backed Async Job Queue (Queue Plane)
=========================================================
Implements distributed job enqueuing, priority dispatching,
and event streaming with automatic in-memory fallback for offline dev/test.
"""

from __future__ import annotations

import asyncio
import os
import time

from sonic.logger import get_logger
from sonic.queue.models import Job, JobEvent, JobPriority, JobStatus

logger = get_logger(__name__)

_CLAIM_JOB_SCRIPT = """
for i = 1, #KEYS / 2 do
    local job_id = redis.call('RPOP', KEYS[i])
    if job_id then
        redis.call('ZADD', KEYS[i + (#KEYS / 2)], ARGV[1], job_id)
        return {job_id, i}
    end
end
return nil
"""

_REQUEUE_EXPIRED_SCRIPT = """
local requeued = 0
for i = 1, #KEYS / 2 do
    local expired = redis.call('ZRANGEBYSCORE', KEYS[i + (#KEYS / 2)], '-inf', ARGV[1])
    for _, job_id in ipairs(expired) do
        if redis.call('ZREM', KEYS[i + (#KEYS / 2)], job_id) == 1 then
            redis.call('LPUSH', KEYS[i], job_id)
            requeued = requeued + 1
        end
    end
end
return requeued
"""

class RedisJobQueue:
    """
    Asynchronous job queue supporting Redis backend and in-memory queue fallback.
    """

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self._redis_client = None
        self.visibility_timeout_seconds = 120
        # In-memory queues by priority
        self._memory_queues: dict[JobPriority, asyncio.Queue[Job]] = {
            JobPriority.CRITICAL: asyncio.Queue(),
            JobPriority.HIGH: asyncio.Queue(),
            JobPriority.MEDIUM: asyncio.Queue(),
            JobPriority.LOW: asyncio.Queue(),
        }
        self._jobs_store: dict[str, Job] = {}
        self._workspace_tenants: dict[str, str] = {}
        self._event_listeners: list[asyncio.Queue[JobEvent]] = []

    async def connect(self) -> bool:
        """Initialize Redis connection if available."""
        if self._redis_client is not None:
            try:
                await self._redis_client.ping()
                return True
            except Exception:
                self._redis_client = None
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
                await self._redis_client.ping()
                logger.info("redis_queue_connected", url=self.redis_url)
                return True
            except Exception as e:
                logger.warning("redis_connection_failed_fallback_memory", error=str(e))
                self._redis_client = None
        return False

    async def ensure_connected(self) -> bool:
        """Connect the queue and prohibit an in-memory production queue."""
        connected = await self.connect()
        if not connected and self._is_production():
            raise RuntimeError("Production Queue Failure: Redis is required but unavailable")
        return connected

    @staticmethod
    def _is_production() -> bool:
        from sonic.config import get_settings

        return (
            os.environ.get("APP_ENV", "").lower() == "production"
            or get_settings().is_production
        )

    async def close(self) -> None:
        """Close the Redis client cleanly during service shutdown."""
        if self._redis_client is None:
            return
        client, self._redis_client = self._redis_client, None
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    async def enqueue_job(self, job: Job) -> str:
        """Add job to priority queue."""
        is_production = self._is_production()
        job.status = JobStatus.QUEUED
        self._jobs_store[job.id] = job

        if self._redis_client:
            try:
                # Store job metadata in Redis Hash
                await self._redis_client.hset(f"sonic:job:{job.id}", mapping={
                    "data": job.model_dump_json(),
                    "tenant_id": job.tenant_id,
                    "status": job.status.value,
                })
                # Push ID to priority list
                await self._redis_client.lpush(f"sonic:queue:{job.priority.value}", job.id)
            except Exception as e:
                if is_production:
                    logger.error("redis_enqueue_failed_fail_closed_in_prod", error=str(e))
                    job.status = JobStatus.FAILED
                    raise RuntimeError(f"Production Queue Failure: Redis unavailable and fallback prohibited in production ({str(e)})")
                logger.error("redis_enqueue_failed_using_memory", error=str(e))
                await self._memory_queues[job.priority].put(job)
        else:
            if is_production:
                logger.error("redis_not_connected_fail_closed_in_prod")
                job.status = JobStatus.FAILED
                raise RuntimeError("Production Queue Failure: Redis client not connected and in-memory queue prohibited in production")
            await self._memory_queues[job.priority].put(job)

        await self.emit_event(JobEvent(
            tenant_id=job.tenant_id,
            engagement_id=job.engagement_id,
            job_id=job.id,
            event_type="JobQueued",
            details={"priority": job.priority.value, "job_type": job.job_type.value},
        ))

        logger.info("job_enqueued", job_id=job.id, tenant_id=job.tenant_id, priority=job.priority.value)
        return job.id

    def register_workspace(self, workspace_id: str, tenant_id: str) -> None:
        """Register an explicitly provisioned workspace for tenant-bound jobs."""
        if not workspace_id or not tenant_id:
            raise ValueError("workspace and tenant are required")
        existing = self._workspace_tenants.get(workspace_id)
        if existing and existing != tenant_id:
            raise PermissionError("workspace is already owned by another tenant")
        self._workspace_tenants[workspace_id] = tenant_id

    def workspace_owned_by(self, workspace_id: str, tenant_id: str) -> bool:
        return self._workspace_tenants.get(workspace_id) == tenant_id

    async def dequeue_job(self, timeout_seconds: float = 1.0) -> Job | None:
        """Fetch highest priority available job."""
        # 1. Try Redis queues in priority order (Critical -> High -> Medium -> Low)
        if self._redis_client:
            try:
                priorities = [
                    JobPriority.CRITICAL, JobPriority.HIGH,
                    JobPriority.MEDIUM, JobPriority.LOW,
                ]
                queue_keys = [f"sonic:queue:{priority.value}" for priority in priorities]
                processing_keys = [
                    f"sonic:queue:processing:{priority.value}" for priority in priorities
                ]
                keys = [*queue_keys, *processing_keys]
                now = time.time()
                await self._redis_client.eval(
                    _REQUEUE_EXPIRED_SCRIPT, len(keys), *keys, now
                )
                claimed = await self._redis_client.eval(
                    _CLAIM_JOB_SCRIPT,
                    len(keys),
                    *keys,
                    now + self.visibility_timeout_seconds,
                )
                if claimed:
                    job_id, priority_index = claimed
                    if isinstance(job_id, bytes):
                        job_id = job_id.decode()
                    raw_data = await self._redis_client.hget(f"sonic:job:{job_id}", "data")
                    if raw_data:
                        job = Job.model_validate_json(raw_data)
                        self._jobs_store[job.id] = job
                        return job
                    await self._redis_client.zrem(
                        processing_keys[int(priority_index) - 1], job_id
                    )
            except Exception as e:
                logger.error("redis_dequeue_error", error=str(e))
                if self._is_production():
                    raise RuntimeError(f"Production Queue Failure: Redis dequeue failed ({e})") from e

        # 2. In-memory queues in priority order
        for priority in [JobPriority.CRITICAL, JobPriority.HIGH, JobPriority.MEDIUM, JobPriority.LOW]:
            q = self._memory_queues[priority]
            if not q.empty():
                job = await q.get()
                return job

        return None

    async def renew_job_lease(self, job: Job) -> None:
        """Extend a claimed Redis job's visibility lease while a worker runs it."""
        if self._redis_client is None:
            return
        await self._redis_client.zadd(
            f"sonic:queue:processing:{job.priority.value}",
            {job.id: time.time() + self.visibility_timeout_seconds},
        )

    async def acknowledge_job(self, job: Job) -> None:
        """Remove a completed Redis job from its processing lease set."""
        if self._redis_client is None:
            return
        await self._redis_client.zrem(
            f"sonic:queue:processing:{job.priority.value}", job.id
        )

    async def get_job(self, job_id: str, tenant_id: str | None = None) -> Job | None:
        """Retrieve job by ID with optional tenant isolation."""
        job = self._jobs_store.get(job_id)
        if not job and self._redis_client:
            try:
                raw_data = await self._redis_client.hget(f"sonic:job:{job_id}", "data")
                if raw_data:
                    job = Job.model_validate_json(raw_data)
                    self._jobs_store[job.id] = job
            except Exception:
                pass

        if job and tenant_id and job.tenant_id != tenant_id:
            return None
        return job

    async def list_jobs(
        self,
        tenant_id: str,
        owner_id: str | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List persisted jobs visible to an owner within one tenant."""
        if self._redis_client is not None:
            try:
                async for key in self._redis_client.scan_iter(match="sonic:job:*"):
                    job_id = key.rsplit(":", 1)[-1]
                    if job_id not in self._jobs_store:
                        raw_data = await self._redis_client.hget(key, "data")
                        if raw_data:
                            job = Job.model_validate_json(raw_data)
                            self._jobs_store[job.id] = job
            except Exception as exc:
                if self._is_production():
                    raise RuntimeError("Production Queue Failure: Redis job listing failed") from exc
                logger.warning("redis_job_listing_failed", error=str(exc))

        jobs = [
            job
            for job in self._jobs_store.values()
            if job.tenant_id == tenant_id
            and (owner_id is None or job.owner_id == owner_id)
        ]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    async def update_job(self, job: Job) -> None:
        """Update job state in store and Redis."""
        self._jobs_store[job.id] = job
        if self._redis_client:
            try:
                await self._redis_client.hset(f"sonic:job:{job.id}", mapping={
                    "data": job.model_dump_json(),
                    "status": job.status.value,
                })
            except Exception as e:
                if self._is_production():
                    raise RuntimeError(f"Production Queue Failure: Redis job update failed ({e})") from e
                logger.warning("redis_job_update_failed", job_id=job.id, error=str(e))

    async def emit_event(self, event: JobEvent) -> None:
        """Broadcast job telemetry event to subscribers."""
        for listener in list(self._event_listeners):
            try:
                listener.put_nowait(event)
            except Exception:
                pass


# Global singleton queue
_global_queue: RedisJobQueue | None = None


def get_job_queue() -> RedisJobQueue:
    """Get or create singleton job queue."""
    global _global_queue
    if _global_queue is None:
        _global_queue = RedisJobQueue()
    return _global_queue
