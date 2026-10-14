"""Operator-visible, tenant-scoped workstation checkpoint receipts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from sonic.api.routes.workstation import (
    _get_or_create_session,
    _persist_workstation_state,
    _resolve_tenant_id,
    _session_workspace_id,
    get_daytona_computer,
)
from sonic.auth.middleware import require_operator
from sonic.auth.models import User
from sonic.computer_use.checkpoint import CheckpointManager

router = APIRouter()
_managers: dict[tuple[str, str, str], CheckpointManager] = {}


class CheckpointCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    artifacts: list[str] = Field(default_factory=list, max_length=100)
    state: dict[str, Any] = Field(default_factory=dict)
    session_id: str = "default"


class CheckpointRollbackRequest(BaseModel):
    session_id: str = "default"


class _ProviderFileStore:
    """Adapter restricted to one already-resolved tenant workspace."""

    def __init__(self, provider: Any, workspace_id: str, actor: str) -> None:
        self.provider = provider
        self.workspace_id = workspace_id
        self.actor = actor

    async def read_file(self, path: str) -> bytes:
        value = await self.provider.read_file(self.workspace_id, path)
        if hasattr(value, "content"):
            value = value.content
        return value.encode() if isinstance(value, str) else bytes(value)

    async def write_file(self, path: str, content: bytes | str) -> bool:
        value = content.decode(errors="strict") if isinstance(content, bytes) else content
        return bool(await self.provider.write_file(
            self.workspace_id, path, value, actor=self.actor
        ))

    async def delete_file(self, path: str) -> bool:
        # Checkpoint rollback never deletes artifacts, but the protocol requires
        # this method so the adapter cannot accidentally broaden its scope.
        return False


def _manager(user: User, session_id: str) -> tuple[CheckpointManager, str, dict[str, Any]]:
    tenant_id = _resolve_tenant_id(user)
    session = _get_or_create_session(tenant_id, session_id)
    workspace_id = _session_workspace_id(user, session_id)
    if not workspace_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active workstation workspace",
        )
    key = (tenant_id, session_id, workspace_id)
    manager = _managers.get(key)
    if manager is None:
        manager = CheckpointManager(
            _ProviderFileStore(get_daytona_computer(), workspace_id, user.email)
        )
        _managers[key] = manager
    return manager, workspace_id, session


@router.post("/workstation/checkpoints")
async def create_checkpoint(
    request: CheckpointCreateRequest,
    user: User = Depends(require_operator),
):
    manager, workspace_id, _ = _manager(user, request.session_id)
    try:
        checkpoint = await manager.create(
            request.name,
            state=request.state,
            artifacts=tuple(request.artifacts),
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "id": checkpoint.id,
        "name": checkpoint.name,
        "session_id": request.session_id,
        "workspace_id": workspace_id,
        "artifacts": [item.path for item in checkpoint.artifacts],
        "state": checkpoint.state,
    }


@router.get("/workstation/checkpoints")
async def list_checkpoints(
    session_id: str = Query("default"),
    user: User = Depends(require_operator),
):
    manager, workspace_id, _ = _manager(user, session_id)
    return {
        "session_id": session_id,
        "workspace_id": workspace_id,
        "checkpoints": [
            {
                "id": item.id,
                "name": item.name,
                "artifacts": [artifact.path for artifact in item.artifacts],
                "state": item.state,
            }
            for item in manager.list_checkpoints()
        ],
    }


@router.post("/workstation/checkpoints/{checkpoint_id}/rollback")
async def rollback_checkpoint(
    checkpoint_id: str,
    request: CheckpointRollbackRequest,
    user: User = Depends(require_operator),
):
    manager, workspace_id, session = _manager(user, request.session_id)
    current_state = dict(session.get("checkpoint_state", {}))

    def apply_checkpoint_state(state: dict[str, Any]) -> None:
        # Checkpoint state is deliberately kept in its own session section.
        # Never merge operator-provided state into tenant/session/workspace
        # identity fields.
        session["checkpoint_state"] = dict(state)

    def read_checkpoint_state() -> dict[str, Any]:
        return dict(session.get("checkpoint_state", {}))

    receipt = await manager.rollback(
        checkpoint_id,
        current_state=current_state,
        apply_state=apply_checkpoint_state,
        read_state=read_checkpoint_state,
    )
    if receipt.status.value not in {"SUCCEEDED", "SUCCESS", "COMPLETED", "VERIFIED"}:
        # Preserve the truthful receipt rather than turning a blocked rollback
        # into an HTTP success.
        code = 404 if receipt.evidence == "unknown checkpoint" else 409
        raise HTTPException(
            status_code=code,
            detail={
                "action_receipt": _receipt_json(receipt),
                "workspace_id": workspace_id,
            },
        )
    _persist_workstation_state()
    return {"workspace_id": workspace_id, "action_receipt": _receipt_json(receipt)}


def _receipt_json(receipt: Any) -> dict[str, Any]:
    return {
        "action_type": receipt.action_type,
        "status": receipt.status.value,
        "state_delta": {
            "changed": dict(receipt.state_delta.changed),
            "removed": list(receipt.state_delta.removed),
        },
        "before": dict(receipt.before),
        "after": dict(receipt.after),
        "exit_code": receipt.exit_code,
        "evidence": receipt.evidence,
        "timestamp": receipt.timestamp,
    }
