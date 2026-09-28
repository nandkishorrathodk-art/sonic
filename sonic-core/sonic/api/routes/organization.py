"""Tenant-scoped API for SONIC's AI employee organization."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from sonic.auth.middleware import require_auth, require_tenant_admin
from sonic.auth.models import User
from sonic.organization.store import get_organization_store

router = APIRouter()


class EmployeeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=160)
    department_id: str
    team_id: str
    subsystem_ids: list[str] = Field(default_factory=list, min_length=1)
    read_only: bool = False
    employee_type: str = "ai"
    prompt: str = ""


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = ""
    lead_employee_id: str | None = None


class WorkItemCreate(BaseModel):
    project_id: str
    title: str = Field(min_length=1, max_length=240)
    description: str = ""
    subsystem_id: str
    employee_id: str
    github_repo: str = ""
    github_issue_number: int | None = None
    github_pr_number: int | None = None
    development_branch: str = ""


class WorkItemStatus(BaseModel):
    status: str


class GitHubReceipt(BaseModel):
    event_type: str
    repository: str
    ref: str = ""
    external_id: str
    approved: bool = False
    ci_success: bool = False


def _http_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/")
async def organization_snapshot(user: User = Depends(require_auth)):
    return await get_organization_store().snapshot(user.tenant_id)


@router.post("/employees", status_code=status.HTTP_201_CREATED)
async def create_employee(payload: EmployeeCreate, user: User = Depends(require_tenant_admin)):
    try:
        return await get_organization_store().create_employee(
            tenant_id=user.tenant_id, name=payload.name, role=payload.role,
            department_id=payload.department_id, team_id=payload.team_id,
            subsystem_ids=payload.subsystem_ids, read_only=payload.read_only,
            employee_type=payload.employee_type, prompt=payload.prompt,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/projects", status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreate, user: User = Depends(require_tenant_admin)):
    try:
        return await get_organization_store().create_project(
            tenant_id=user.tenant_id, name=payload.name,
            description=payload.description, lead_employee_id=payload.lead_employee_id,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/work-items", status_code=status.HTTP_201_CREATED)
async def create_work_item(payload: WorkItemCreate, user: User = Depends(require_tenant_admin)):
    try:
        return await get_organization_store().create_work_item(
            tenant_id=user.tenant_id, project_id=payload.project_id, title=payload.title,
            description=payload.description, subsystem_id=payload.subsystem_id,
            employee_id=payload.employee_id, github_repo=payload.github_repo,
            github_issue_number=payload.github_issue_number,
            github_pr_number=payload.github_pr_number,
            development_branch=payload.development_branch,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.patch("/work-items/{work_item_id}/status")
async def update_work_item_status(
    work_item_id: str,
    payload: WorkItemStatus,
    user: User = Depends(require_tenant_admin),
):
    try:
        return await get_organization_store().update_work_item_status(
            user.tenant_id, work_item_id, payload.status
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/github/receipt")
async def github_receipt(
    request: Request,
    user: User = Depends(require_tenant_admin),
    x_sonic_signature: str | None = Header(default=None, alias="X-SONIC-Signature-256"),
):
    secret = os.environ.get("SONIC_GITHUB_RECEIPT_SECRET", "").encode()
    if not secret or not x_sonic_signature:
        raise HTTPException(status_code=503, detail="GitHub receipt signing is not configured")
    body = await request.body()
    expected = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, x_sonic_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid GitHub receipt signature")
    try:
        payload = GitHubReceipt.model_validate_json(body)
        return await get_organization_store().record_github_receipt(
            tenant_id=user.tenant_id, event_type=payload.event_type,
            repository=payload.repository, ref=payload.ref,
            external_id=payload.external_id, approved=payload.approved,
            ci_success=payload.ci_success,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/validate-path")
async def validate_path(path: str, user: User = Depends(require_auth)) -> dict[str, Any]:
    del user
    try:
        return get_organization_store().validate_repository_path(path)
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
