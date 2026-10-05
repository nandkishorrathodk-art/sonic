"""
Tests for Phase 13: ComputerProvider Lifecycle, State & Audit.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from sonic.computer.models import ComputerProfile, ComputerWorkspaceStatus, ComputerWorkspaceType
from sonic.computer.provider import UnifiedComputerProvider
from sonic.sandbox.providers.docker_provider import DockerProvider


def test_computer_lifecycle_create_status_destroy():
    async def _run():
        docker_provider = DockerProvider()
        comp = UnifiedComputerProvider(compute_provider=docker_provider)

        # 1. Create Computer
        ws = await comp.create(
            tenant_id="tenant-alpha",
            engagement_id="eng-alpha",
            workspace_type=ComputerWorkspaceType.MISSION_COMPUTER,
            profile=ComputerProfile.KALI_SECURITY,
        )
        assert ws.status == ComputerWorkspaceStatus.READY
        assert ws.id in comp.workspaces

        # 2. Get Status
        st = await comp.status(ws.id)
        assert st.workspace_id == ws.id
        assert st.tenant_id == "tenant-alpha"
        assert "Desktop" in st.open_applications

        # 3. Verify Audit Log
        assert len(comp.audit_log) >= 1
        assert comp.audit_log[-1].action == "CREATE_COMPUTER"

        # 4. Destroy Computer
        ok = await comp.destroy(ws.id)
        assert ok is True
        assert ws.id not in comp.workspaces

    asyncio.run(_run())


def test_destroy_failure_preserves_workspace_and_local_state():
    async def _run():
        compute = SimpleNamespace(destroy_workspace=AsyncMock(return_value=False))
        comp = UnifiedComputerProvider(compute_provider=compute)
        ws = SimpleNamespace(
            id="ws-failed",
            tenant_id="tenant-alpha",
            status=ComputerWorkspaceStatus.READY,
        )
        comp.workspaces[ws.id] = ws
        comp._running_apps[ws.id] = {"chromium"}

        result = await comp.destroy(ws.id)

        assert result is False
        assert ws.status == ComputerWorkspaceStatus.DEGRADED
        assert comp.workspaces[ws.id] is ws
        assert comp._running_apps[ws.id] == {"chromium"}
        assert comp.audit_log[-1].result == "FAILED"

    asyncio.run(_run())


def test_application_lifecycle_requires_a_tracked_process():
    async def _run():
        compute = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(exit_code=0, stdout="", stderr="")
            )
        )
        comp = UnifiedComputerProvider(compute_provider=compute)
        ws = SimpleNamespace(
            id="ws-app",
            tenant_id="tenant-alpha",
            status=ComputerWorkspaceStatus.READY,
        )
        comp.workspaces[ws.id] = ws

        launched = await comp.launch_application(ws.id, "chromium https://example.test")
        closed = await comp.close_application(ws.id, "chromium")

        assert launched is False
        assert closed is False
        assert comp._running_apps.get(ws.id, set()) == set()
        assert comp._app_process_ids.get(ws.id, {}) == {}
        assert compute.execute.await_count == 1
        assert "printf '%s\\n' \"$app_pid\"" in compute.execute.await_args.args[1]
        assert "pkill" not in compute.execute.await_args.args[1]

    asyncio.run(_run())
