"""
Tests for Phase 9: Health Checker Engine and Runtime Dependency Audits.
"""

import asyncio
from sonic.health import HealthChecker, HealthStatus, SystemHealthReport


def test_health_checker_api_and_llm():
    async def _run():
        api_health = await HealthChecker.check_api()
        assert api_health.status == HealthStatus.HEALTHY
        assert api_health.latency_ms >= 0.0

        llm_health = await HealthChecker.check_llm()
        assert llm_health.name == "llm_router"
        assert llm_health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    asyncio.run(_run())


def test_system_health_aggregation():
    async def _run():
        report = await HealthChecker.get_system_health()
        assert isinstance(report, SystemHealthReport)
        assert report.version == "v1.3.0"
        assert report.overall_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNAVAILABLE)
        assert "control_plane_api" in report.components
        assert "redis_queue" in report.components

    asyncio.run(_run())


def test_database_health_check_executes_query_on_configured_database(monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr("sonic.db.session.get_engine", lambda: engine)
    monkeypatch.setattr(
        "sonic.db.session.get_database_url",
        lambda: "sqlite+aiosqlite:///:memory:",
    )

    async def _run():
        try:
            health = await HealthChecker.check_database()
            assert health.name == "database"
            assert health.status == HealthStatus.HEALTHY
            assert health.latency_ms >= 0
            assert "sqlite" in health.message
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_database_health_check_reports_unreachable_database(monkeypatch):
    class _FailedConnection:
        async def __aenter__(self):
            raise ConnectionError("database unavailable")

        async def __aexit__(self, *_args):
            return False

    class _FailedEngine:
        def connect(self):
            return _FailedConnection()

    monkeypatch.setattr("sonic.db.session.get_engine", lambda: _FailedEngine())
    monkeypatch.setattr(
        "sonic.db.session.get_database_url",
        lambda: "postgresql+asyncpg://db.invalid/sonic",
    )

    health = asyncio.run(HealthChecker.check_database())

    assert health.name == "database"
    assert health.status == HealthStatus.UNAVAILABLE
    assert health.is_critical is True
    assert "database unavailable" in health.message
