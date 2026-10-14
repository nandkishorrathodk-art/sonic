from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from sonic.api.routes import workstation
from sonic.auth.models import User, UserRole


@pytest.mark.asyncio
async def test_operator_approval_resolves_exact_pending_action(monkeypatch):
    monkeypatch.setattr(workstation, "_persist_workstation_state", lambda: None)
    workstation.reset_workstation_state()
    tenant_id = "approval-tenant"
    session_id = "approval-session"
    state = workstation._get_or_create_session(tenant_id, session_id)
    callback = workstation._request_operator_action_approval(
        tenant_id, session_id, state
    )
    waiting = asyncio.create_task(
        callback("GIT_PUSH", "origin main", "publish changes")
    )
    await asyncio.sleep(0)

    approval_item = state["worklog"][-1]
    approval_id = approval_item["approval_id"]
    assert approval_item["approval_status"] == "pending"
    assert state["status"] == "WAITING_FOR_APPROVAL"

    other_user = User(
        email="other@example.invalid",
        name="Other",
        role=UserRole.OPERATOR,
        tenant_id="other-tenant",
    )
    with pytest.raises(HTTPException) as exc_info:
        await workstation.approve_workstation_action(
            workstation.WorkstationActionApprovalRequest(
                approval_id=approval_id, approved=True
            ),
            session_id,
            other_user,
        )
    assert exc_info.value.status_code == 404
    assert not waiting.done()

    operator = User(
        email="operator@example.invalid",
        name="Operator",
        role=UserRole.OPERATOR,
        tenant_id=tenant_id,
    )
    result = await workstation.approve_workstation_action(
        workstation.WorkstationActionApprovalRequest(
            approval_id=approval_id, approved=True
        ),
        session_id,
        operator,
    )

    assert result["status"] == "approved"
    assert await waiting is True
    assert approval_item["approval_status"] == "approved"
    assert state["status"] != "WAITING_FOR_APPROVAL"
    assert approval_id not in workstation._pending_operator_approvals


@pytest.mark.asyncio
async def test_operator_rejection_resolves_as_denial(monkeypatch):
    monkeypatch.setattr(workstation, "_persist_workstation_state", lambda: None)
    workstation.reset_workstation_state()
    tenant_id = "approval-reject-tenant"
    session_id = "approval-reject-session"
    state = workstation._get_or_create_session(tenant_id, session_id)
    callback = workstation._request_operator_action_approval(
        tenant_id, session_id, state
    )
    waiting = asyncio.create_task(
        callback("APP_INSTALL", "package", "install changes system software")
    )
    await asyncio.sleep(0)
    approval_id = state["worklog"][-1]["approval_id"]
    operator = User(
        email="operator@example.invalid",
        name="Operator",
        role=UserRole.OPERATOR,
        tenant_id=tenant_id,
    )

    result = await workstation.approve_workstation_action(
        workstation.WorkstationActionApprovalRequest(
            approval_id=approval_id, approved=False
        ),
        session_id,
        operator,
    )

    assert result["status"] == "rejected"
    assert await waiting is False


@pytest.mark.asyncio
async def test_operator_approval_http_route_resolves_waiting_agent(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from sonic.auth.middleware import require_operator

    monkeypatch.setattr(workstation, "_persist_workstation_state", lambda: None)
    workstation.reset_workstation_state()
    tenant_id = "approval-http-tenant"
    session_id = "approval-http-session"
    state = workstation._get_or_create_session(tenant_id, session_id)
    callback = workstation._request_operator_action_approval(
        tenant_id, session_id, state
    )
    waiting = asyncio.create_task(
        callback("GIT_PUSH", "origin main", "publish changes")
    )
    await asyncio.sleep(0)
    approval_id = state["worklog"][-1]["approval_id"]
    operator = User(
        email="operator@example.invalid",
        name="Operator",
        role=UserRole.OPERATOR,
        tenant_id=tenant_id,
    )
    app = FastAPI()
    app.include_router(getattr(workstation.router, "original_router", workstation.router))
    app.dependency_overrides[require_operator] = lambda: operator

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/workstation/action/approve?session_id={session_id}",
            json={"approval_id": approval_id, "approved": True},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert await waiting is True
