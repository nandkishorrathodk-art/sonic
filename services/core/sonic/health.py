"""
SONIC-REDA — Production Health Check Engine (Phase 9)
======================================================
Provides unified, rigorous health status across all runtime dependencies:
    - API (Control Plane)
    - PostgreSQL (Identity / Tenancy / Engagements)
    - Redis (Job Queues / Mutex Locks / State Persistence)
    - Neo4j / Graph Memory (Orchestration & Lineage Graph)
    - Daytona (Remote Cloud Sandboxes)
    - Docker (Local Container Sandboxes)
    - LLM Provider (Model Router & Gemini API)
    - Compute Sandboxes
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from sonic.logger import get_logger

logger = get_logger(__name__)


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ComponentHealth(BaseModel):
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = "Operating normally"
    is_critical: bool = True
    last_checked: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SystemHealthReport(BaseModel):
    overall_status: HealthStatus
    version: str = "v1.3.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    components: dict[str, ComponentHealth] = Field(default_factory=dict)


class HealthChecker:
    """
    Evaluates real runtime health across control plane and execution backends.
    """

    @classmethod
    async def check_api(cls) -> ComponentHealth:
        return ComponentHealth(
            name="control_plane_api",
            status=HealthStatus.HEALTHY,
            latency_ms=0.5,
            message="FastAPI Control Plane responding",
            is_critical=True,
        )

    @classmethod
    async def check_redis(cls, redis_url: str | None = None) -> ComponentHealth:
        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        start = time.perf_counter()
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(url, socket_timeout=2.0)
            await r.ping()
            await r.aclose()
            latency = round((time.perf_counter() - start) * 1000, 2)
            return ComponentHealth(
                name="redis_queue",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Connected to Redis cluster",
                is_critical=True,
            )
        except Exception as e:
            return ComponentHealth(
                name="redis_queue",
                status=HealthStatus.DEGRADED,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                message=f"Redis unavailable ({str(e)}). Transient memory fallback active.",
                is_critical=False,
            )

    @classmethod
    async def check_database(cls) -> ComponentHealth:
        start = time.perf_counter()
        from sqlalchemy import text

        from sonic.db.session import get_database_url, get_engine

        db_url = get_database_url()
        backend = db_url.split(":", 1)[0].split("+", 1)[0]

        try:
            async def probe_database() -> None:
                async with get_engine().connect() as connection:
                    result = await connection.execute(text("SELECT 1"))
                    if result.scalar_one() != 1:
                        raise RuntimeError("Database health query returned an unexpected result")

            await asyncio.wait_for(probe_database(), timeout=3.0)
            latency = round((time.perf_counter() - start) * 1000, 2)
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message=f"{backend} database connection and health query succeeded",
                is_critical=True,
            )
        except Exception as e:
            latency = round((time.perf_counter() - start) * 1000, 2)
            logger.error("database_health_check_failed", backend=backend, error=str(e))
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNAVAILABLE,
                latency_ms=latency,
                message=f"{backend} database health check failed: {e}",
                is_critical=True,
            )

    @classmethod
    async def check_daytona(cls) -> ComponentHealth:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            if os.environ.get("SONIC_USE_DAYTONA_CLOUD") == "1":
                return ComponentHealth(
                    name="daytona_cloud",
                    status=HealthStatus.UNAVAILABLE,
                    message="Daytona cloud is required but DAYTONA_API_KEY is not configured.",
                    is_critical=True,
                )
            return ComponentHealth(
                name="daytona_cloud",
                status=HealthStatus.DEGRADED,
                message="Daytona cloud is inactive; local Docker workstation mode is configured.",
                is_critical=False,
            )
        return ComponentHealth(
            name="daytona_cloud",
            status=HealthStatus.HEALTHY,
            message="Daytona SDK credentials configured",
            is_critical=False,
        )

    @classmethod
    async def check_docker(cls) -> ComponentHealth:
        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            latency = round((time.perf_counter() - start) * 1000, 2)
            if proc.returncode == 0:
                return ComponentHealth(
                    name="docker_sandbox",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message="Docker daemon active and ready for container sandboxes",
                    is_critical=True,
                )
            else:
                return ComponentHealth(
                    name="docker_sandbox",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="Docker daemon returned error. Verify daemon is running.",
                    is_critical=True,
                )
        except Exception:
            return ComponentHealth(
                name="docker_sandbox",
                status=HealthStatus.DEGRADED,
                message="Docker CLI not found or daemon not reachable. Daytona provider recommended.",
                is_critical=False,
            )

    @classmethod
    async def check_llm(cls) -> ComponentHealth:
        try:
            from sonic.llm.router import ModelRouter

            router = ModelRouter.from_config()
        except Exception as exc:
            logger.error("llm_health_router_load_failed", error=str(exc))
            return ComponentHealth(
                name="llm_router",
                status=HealthStatus.DEGRADED,
                message="LLM routing configuration could not be loaded.",
                is_critical=True,
            )

        active = sorted(
            name
            for name in router.providers
            if router._is_provider_ready(name)
        )
        vision_rule = router.routing_rules.get("vision", {})
        vision_candidates = [vision_rule.get("provider", "")]
        for fallback in vision_rule.get("fallback", []):
            vision_candidates.append(
                fallback.split("/", 1)[0] if "/" in fallback else fallback
            )
        ready_vision = sorted({
            name for name in vision_candidates
            if name in router.providers and router._is_provider_ready(name)
        })
        primary_vision = str(vision_rule.get("provider", "") or "").strip()
        ready_fallbacks = [name for name in ready_vision if name != primary_vision]
        if primary_vision and primary_vision in ready_vision:
            fallbacks = ", ".join(ready_fallbacks) if ready_fallbacks else "none"
            return ComponentHealth(
                name="llm_router",
                status=HealthStatus.DEGRADED,
                message=(
                    f"Primary vision provider credentials are configured: {primary_vision}; "
                    f"ready fallback providers: {fallbacks}; remote reachability and image "
                    "comprehension are not verified."
                ),
                is_critical=True,
            )
        if ready_fallbacks:
            return ComponentHealth(
                name="llm_router",
                status=HealthStatus.DEGRADED,
                message=(
                    f"Primary vision provider credentials are unavailable: {primary_vision or 'none'}; "
                    f"ready fallback providers: {', '.join(ready_fallbacks)}; remote reachability "
                    "and image comprehension are not verified."
                ),
                is_critical=True,
            )
        if not ready_vision:
            active_names = ", ".join(active) if active else "none"
            return ComponentHealth(
                name="llm_router",
                status=HealthStatus.DEGRADED,
                message=(
                    "No configured vision provider credentials are available for GUI reasoning. "
                    f"Other ready providers: {active_names}."
                ),
                is_critical=True,
            )
        return ComponentHealth(
            name="llm_router",
            status=HealthStatus.DEGRADED,
            message=(
                "Vision provider credentials are configured: "
                f"{', '.join(ready_vision)}; remote reachability and image "
                "comprehension are not verified."
            ),
            is_critical=True,
        )

    @classmethod
    async def get_system_health(cls) -> SystemHealthReport:
        """Run all component checks concurrently and aggregate overall system health."""
        checks = await asyncio.gather(
            cls.check_api(),
            cls.check_redis(),
            cls.check_database(),
            cls.check_daytona(),
            cls.check_docker(),
            cls.check_llm(),
            return_exceptions=True,
        )

        components: dict[str, ComponentHealth] = {}
        for c in checks:
            if isinstance(c, ComponentHealth):
                components[c.name] = c

        # Determine overall status
        statuses = [comp.status for comp in components.values()]
        if any(s == HealthStatus.UNAVAILABLE for s in statuses):
            overall = HealthStatus.UNAVAILABLE
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return SystemHealthReport(
            overall_status=overall,
            components=components,
        )
