"""Tenant-scoped AI employee organization and work ownership API."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from sonic.auth.middleware import require_auth, require_operator
from sonic.auth.models import User
from sonic.organization.store import get_organization_store

router = APIRouter()


class EmployeeCreateRequest(BaseModel):
    team_id: str
    name: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1, max_length=128)
    career_track: Literal["individual_contributor", "management"] = "individual_contributor"
    level: str = Field(default="engineer", min_length=1, max_length=64)
    capability_mode: Literal["read_only", "engineering"] = "engineering"
    system_prompt: str = Field(default="", max_length=8000)
    skills: list[str] = Field(default_factory=list, max_length=50)
    primary_ownership: list[str] = Field(default_factory=list, max_length=50)
    secondary_ownership: list[str] = Field(default_factory=list, max_length=50)
    subsystem_ids: list[str] = Field(default_factory=list, max_length=20)
    status: Literal["active", "inactive"] = "active"


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=8000)
    status: Literal["planned", "active", "blocked", "completed", "archived"] = "planned"
    lead_employee_id: str | None = None
    github_repository: str = Field(default="", max_length=500)


class WorkItemCreateRequest(BaseModel):
    project_id: str
    parent_id: str | None = None
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=12000)
    item_type: Literal["epic", "issue", "task", "pull_request", "evaluation"] = "task"
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    subsystem_id: str = ""
    assignee_employee_id: str | None = None
    github_issue_url: str = Field(default="", max_length=1000)


class WorkItemUpdateRequest(BaseModel):
    status: Literal[
        "backlog", "ready", "in_progress", "blocked", "in_review"
    ] | None = None
    assignee_employee_id: str | None = None
    github_issue_url: str | None = Field(default=None, max_length=1000)
    branch_name: str | None = Field(default=None, max_length=255)
    pull_request_url: str | None = Field(default=None, max_length=1000)


def _tenant(user: User) -> str:
    return user.tenant_id or "default"


def _url_or_empty(value: str, field_name: str) -> None:
    if value and not value.startswith(("https://", "http://")):
        raise HTTPException(status_code=422, detail=f"{field_name} must be an http(s) URL")


@router.get("/")
def get_organization(user: User = Depends(require_auth)):
    """Return the caller's seeded organization and tenant-owned work."""
    return get_organization_store().snapshot(_tenant(user))


@router.post("/employees", status_code=201)
def create_employee(
    request: EmployeeCreateRequest,
    user: User = Depends(require_operator),
):
    try:
        employee = get_organization_store().create_employee(
            _tenant(user), request.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"employee": employee}


@router.post("/projects", status_code=201)
def create_project(
    request: ProjectCreateRequest,
    user: User = Depends(require_operator),
):
    _url_or_empty(request.github_repository, "github_repository")
    try:
        project = get_organization_store().create_project(
            _tenant(user), user.email, request.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"project": project}


@router.post("/work-items", status_code=201)
def create_work_item(
    request: WorkItemCreateRequest,
    user: User = Depends(require_operator),
):
    _url_or_empty(request.github_issue_url, "github_issue_url")
    try:
        work_item = get_organization_store().create_work_item(
            _tenant(user), user.email, request.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"work_item": work_item}


@router.patch("/work-items/{work_item_id}")
def update_work_item(
    work_item_id: str,
    request: WorkItemUpdateRequest,
    user: User = Depends(require_operator),
):
    values = {
        key: value
        for key, value in request.model_dump(exclude_unset=True).items()
        if value is not None or key == "assignee_employee_id"
    }
    if "pull_request_url" in values:
        _url_or_empty(values["pull_request_url"], "pull_request_url")
    if "github_issue_url" in values:
        _url_or_empty(values["github_issue_url"], "github_issue_url")
    try:
        work_item = get_organization_store().update_work_item(
            _tenant(user), work_item_id, values
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Work item not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"work_item": work_item}


@router.post("/github/webhook")
async def github_webhook(request: Request):
    """Accept only signed GitHub PR/review/check-suite delivery receipts."""
    secret = os.environ.get("SONIC_GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="GitHub webhook secret is not configured")
    body = await request.body()
    supplied_signature = request.headers.get("x-hub-signature-256", "")
    expected_signature = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid GitHub webhook signature")
    event_name = request.headers.get("x-github-event", "")
    if not event_name:
        raise HTTPException(status_code=400, detail="Missing GitHub event header")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid webhook JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Webhook payload must be an object")
    updated = get_organization_store().record_github_event(event_name, payload)
    return {"accepted": True, "event": event_name, "work_items_updated": updated}
