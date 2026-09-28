"""
Comprehensive Test Suite for Phase 4:
  1. ComputeProvider-bound Security Tool Execution (Nmap, Nuclei, FFUF, HTTP).
  2. Fail-Closed tool execution without host fallback.
  3. Containerized Browser Runtime (Playwright/Chromium script execution).
  4. Sandbox Network Egress Control (Metadata & Private IP filtering).
  5. Async Redis Job Queue & SonicWorker Execution Engine.
  6. Multi-Tenant Job API routes.
"""

import asyncio
import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

from sonic.api.main import app
from sonic.auth.google_auth import create_jwt_token
from sonic.auth.models import User, UserRole
from sonic.browser.container_runtime import BrowserAction, ContainerizedBrowser
from sonic.queue.job_queue import RedisJobQueue
from sonic.queue.models import Job, JobPriority, JobStatus, JobType
from sonic.queue.worker import SonicWorker
from sonic.sandbox.egress import is_target_allowed
from sonic.sandbox.providers.local_dev_provider import LocalDevProvider
from sonic.tools.adapters.ffuf_adapter import FFUFAdapter
from sonic.tools.adapters.nmap_adapter import NmapAdapter
from sonic.tools.adapters.nuclei_adapter import NucleiAdapter
from sonic.tools.base import ToolRequest, ToolStatus

client = TestClient(app)


def get_test_auth_headers(role: UserRole = UserRole.OPERATOR, tenant_id: str = "tenant-phase4"):
    user = User(email=f"worker@{tenant_id}.com", name="Worker Test", tenant_id=tenant_id, role=role)
    token = create_jwt_token(user)
    return {"Authorization": f"Bearer {token.access_token}"}


# ==========================================================
# 1. Tool Adapter & Fail-Closed Tests
# ==========================================================

def test_tool_adapter_fail_closed_when_safety_locked():
    """Verify tools fail closed and never execute on host when safety locked."""
    async def _run():
        locked_provider = LocalDevProvider(allow_host_execution=False)
        nmap = NmapAdapter(locked_provider)

        req = ToolRequest(
            tenant_id="tenant-alpha",
            engagement_id="eng-01",
            workspace_id="sandbox-01",
            agent_id="recon-agent",
            tool_name="nmap",
            target="example.com",
        )

        res = await nmap.execute(req)
        assert res.status == ToolStatus.BLOCKED
        assert res.exit_code == 126
        assert "FAIL-CLOSED" in (res.error_message or "")

    asyncio.run(_run())


def test_tool_parsers():
    """Verify Nmap, Nuclei, and FFUF structured output parsers."""
    provider = LocalDevProvider(allow_host_execution=False)

    # Nmap parsing
    nmap = NmapAdapter(provider)
    sample_nmap_stdout = """
    PORT     STATE SERVICE VERSION
    22/tcp   open  ssh     OpenSSH 8.9p1 Ubuntu
    80/tcp   open  http    nginx 1.18.0
    443/tcp  open  https   nginx 1.18.0
    """
    ports = nmap.parse_output(sample_nmap_stdout, "")
    assert len(ports) == 3
    assert ports[0]["port"] == 22
    assert ports[0]["service"] == "ssh"

    # Nuclei parsing
    nuclei = NucleiAdapter(provider)
    sample_nuclei = '{"template-id": "cve-2023-1234", "info": {"name": "Test Vuln", "severity": "high", "description": "PoC Vuln"}}'
    findings = nuclei.parse_output(sample_nuclei, "")
    assert len(findings) == 1
    assert findings[0]["template_id"] == "cve-2023-1234"
    assert findings[0]["severity"] == "HIGH"

    # FFUF parsing
    ffuf = FFUFAdapter(provider)
    sample_ffuf = '{"results": [{"url": "http://target.com/admin", "status": 200, "length": 512, "input": {"FUZZ": "admin"}}]}'
    endpoints = ffuf.parse_output(sample_ffuf, "")
    assert len(endpoints) == 1
    assert endpoints[0]["url"] == "http://target.com/admin"
    assert endpoints[0]["status_code"] == 200


# ==========================================================
# 2. Containerized Browser Tests
# ==========================================================

def test_containerized_browser_fail_closed():
    """Verify ContainerizedBrowser fails closed if sandbox is unavailable."""
    async def _run():
        locked_provider = LocalDevProvider(allow_host_execution=False)
        browser = ContainerizedBrowser(locked_provider)

        actions = [BrowserAction(action="navigate", url="https://example.com")]
        res = await browser.execute_browser_script("sandbox-ws-01", actions)
        assert res.success is False
        assert "FAIL-CLOSED" in (res.error_message or "").upper()

    asyncio.run(_run())



# ==========================================================
# 3. Egress Security Filtering Tests
# ==========================================================

def test_egress_security_filtering():
    """Verify that cloud metadata and private IP ranges are blocked by default."""
    # Metadata blocked
    allowed, msg = is_target_allowed("169.254.169.254")
    assert allowed is False
    assert "blocked" in msg

    # Loopback blocked
    allowed_lb, msg_lb = is_target_allowed("127.0.0.1")
    assert allowed_lb is False

    # Private RFC 1918 blocked
    allowed_priv, _ = is_target_allowed("10.0.0.5")
    assert allowed_priv is False

    allowed_priv_c, _ = is_target_allowed("192.168.1.100")
    assert allowed_priv_c is False

    # Public domain allowed
    allowed_pub, _ = is_target_allowed("example.com")
    assert allowed_pub is True


# ==========================================================
# 4. Async Queue & Worker Engine Tests
# ==========================================================

def test_job_queue_and_worker_execution():
    """Verify job enqueuing, priority ordering, and worker dispatching."""
    async def _run():
        queue = RedisJobQueue()  # Uses in-memory fallback
        provider = LocalDevProvider(allow_host_execution=False)
        worker = SonicWorker(queue=queue, provider=provider, worker_id="test-worker")

        # 1. Enqueue job
        job = Job(
            tenant_id="tenant-phase4",
            engagement_id="eng-01",
            job_type=JobType.TOOL_EXECUTION,
            priority=JobPriority.HIGH,
            payload={"tool_name": "http_client", "target": "https://example.com"},
        )
        job_id = await queue.enqueue_job(job)
        assert job_id == job.id

        # 2. Dequeue and execute via worker
        processed_job = await worker.run_once()
        assert processed_job is not None
        assert processed_job.id == job_id
        # Because provider is safety locked, worker records failure gracefully
        assert processed_job.status in (JobStatus.QUEUED, JobStatus.FAILED)

    asyncio.run(_run())


def test_failed_security_tool_does_not_succeed_job(monkeypatch):
    from sonic.tools.base import ToolStatus

    async def _run():
        queue = RedisJobQueue()
        worker = SonicWorker(queue=queue, provider=LocalDevProvider(allow_host_execution=False))
        worker._tools = {
            "failing": SimpleNamespace(
                execute=lambda request: _failed_tool_result()
            )
        }
        monkeypatch.setattr(
            "sonic.safety.scope.get_scope_checker",
            lambda: SimpleNamespace(kill_switch_enabled=True),
        )
        monkeypatch.setattr(
            "sonic.safety.runtime_stop.get_runtime_stop_state",
            lambda: SimpleNamespace(is_stopped=lambda tenant_id: False, reason=lambda tenant_id: ""),
        )
        job = Job(
            tenant_id="tenant-worker-failure",
            engagement_id="eng-failure",
            job_type=JobType.TOOL_EXECUTION,
            payload={"tool_name": "failing", "target": "target"},
            max_retries=0,
        )
        result = await worker.execute_job(job)
        assert result.status == JobStatus.FAILED
        assert "Tool execution failed" in result.error_message

    async def _failed_tool_result():
        return SimpleNamespace(
            status=ToolStatus.FAILED,
            error_message="exit code 1",
            raw_stderr="failed",
        )

    asyncio.run(_run())


class _FakeRedis:
    def __init__(self):
        self.lists = {}
        self.sorted_sets = {}
        self.hashes = {}

    async def eval(self, script, numkeys, *values):
        keys = values[:numkeys]
        now = float(values[-1])
        half = numkeys // 2
        if "local requeued = 0" in script:
            moved = 0
            for index in range(half):
                queue_key = keys[index]
                processing_key = keys[index + half]
                expired = [
                    job_id for job_id, lease in self.sorted_sets.get(processing_key, {}).items()
                    if lease <= now
                ]
                for job_id in expired:
                    self.sorted_sets[processing_key].pop(job_id)
                    self.lists.setdefault(queue_key, []).insert(0, job_id)
                    moved += 1
            return moved

        for index, queue_key in enumerate(keys[:half], start=1):
            queue = self.lists.setdefault(queue_key, [])
            if queue:
                job_id = queue.pop()
                self.sorted_sets.setdefault(keys[index - 1 + half], {})[job_id] = now
                return [job_id, index]
        return None

    async def hset(self, key, mapping):
        self.hashes[key] = dict(mapping)

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def zrem(self, key, job_id):
        return int(self.sorted_sets.setdefault(key, {}).pop(job_id, None) is not None)

    async def zadd(self, key, mapping):
        self.sorted_sets.setdefault(key, {}).update(mapping)


def test_production_queue_requires_redis(monkeypatch):
    from sonic.config import get_settings

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="Redis is required"):
            asyncio.run(RedisJobQueue().ensure_connected())
    finally:
        get_settings.cache_clear()


def test_redis_dequeue_recovers_expired_job_and_acknowledges(monkeypatch):
    from sonic.queue import job_queue as queue_module

    async def _run():
        queue = RedisJobQueue(redis_url="redis://unused")
        redis = _FakeRedis()
        queue._redis_client = redis
        job = Job(
            tenant_id="tenant-lease",
            engagement_id="eng-lease",
            priority=JobPriority.HIGH,
            payload={"tool_name": "noop"},
        )
        await queue.enqueue_job(job)

        clock = {"now": 1000.0}
        monkeypatch.setattr(queue_module.time, "time", lambda: clock["now"])
        claimed = await queue.dequeue_job()
        assert claimed is not None and claimed.id == job.id
        processing_key = f"sonic:queue:processing:{job.priority.value}"
        assert job.id in redis.sorted_sets[processing_key]

        clock["now"] += queue.visibility_timeout_seconds + 1
        reclaimed = await queue.dequeue_job()
        assert reclaimed is not None and reclaimed.id == job.id
        await queue.acknowledge_job(reclaimed)
        assert job.id not in redis.sorted_sets[processing_key]

    asyncio.run(_run())


# ==========================================================
# 5. Multi-Tenant Job API Endpoints
# ==========================================================

def test_job_api_multi_tenant_isolation():
    """Verify Job API endpoints enforce authentication and tenant isolation."""
    headers_alpha = get_test_auth_headers(tenant_id="tenant-alpha-jobs")
    headers_beta = get_test_auth_headers(tenant_id="tenant-beta-jobs")

    # 1. Tenant Alpha submits job
    res_submit = client.post(
        "/jobs/",
        json={
            "engagement_id": "eng-alpha",
            "job_type": "tool_execution",
            "priority": "high",
            "payload": {"tool_name": "nmap", "target": "alpha-target.com"},
        },
        headers=headers_alpha,
    )
    assert res_submit.status_code == 200
    job_id = res_submit.json()["job_id"]

    # 2. Tenant Alpha can fetch job
    res_get_alpha = client.get(f"/jobs/{job_id}", headers=headers_alpha)
    assert res_get_alpha.status_code == 200
    assert res_get_alpha.json()["job"]["id"] == job_id

    # 3. Tenant Beta direct query -> MUST return 404 Not Found
    res_get_beta = client.get(f"/jobs/{job_id}", headers=headers_beta)
    assert res_get_beta.status_code == 404
