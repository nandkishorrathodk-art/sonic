"""
SONIC-REDA — Asynchronous Job Management API Routes
======================================================
Authenticated endpoints for job submission, tracking, and cancellation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from sonic.auth.middleware import require_auth, require_operator
from sonic.auth.models import User, UserRole
from sonic.queue.job_queue import get_job_queue
from sonic.queue.models import Job, JobPriority, JobType
from sonic.sandbox.egress import is_target_allowed
from sonic.safety.scope import get_scope_checker

router = APIRouter()


class SubmitJobRequest(BaseModel):
    engagement_id: str
    job_type: JobType = JobType.TOOL_EXECUTION
    priority: JobPriority = JobPriority.MEDIUM
    payload: dict[str, Any]
    workspace_id: str = ""
    timeout_seconds: int = 180


@router.post("/")
async def submit_job(
    request: SubmitJobRequest,
    user: User = Depends(require_operator),
):
    """Submit an asynchronous job to the Redis queue (Operator/Admin)."""
    queue = get_job_queue()
    workspace_id = request.workspace_id
    if not workspace_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An explicitly provisioned tenant-owned workspace is required",
        )
    if not queue.workspace_owned_by(workspace_id, user.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace is not owned by the submitting tenant",
        )
    payload = dict(request.payload)
    if request.job_type == JobType.TOOL_EXECUTION:
        target = str(payload.get("target") or "").strip()
        if not target:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tool jobs require a target",
            )
        from sonic.api.routes.engagements import get_engagement_manager

        engagement = await get_engagement_manager().get_engagement_status(
            request.engagement_id,
            tenant_id=user.tenant_id,
        )
        if (
            not engagement
            or "error" in engagement
            or engagement.get("tenant_id") != user.tenant_id
            or not isinstance(engagement.get("scope"), dict)
            or not engagement["scope"]
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tool job requires an active tenant-owned engagement with scope",
            )
        scope = engagement["scope"]
        if not get_scope_checker().is_target_in_scope(target, scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tool target is outside the engagement's registered scope",
            )
        allowed, reason = is_target_allowed(target)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tool target is blocked by egress policy: {reason}",
            )
        payload["scope"] = scope
    job = Job(
        tenant_id=user.tenant_id,
        owner_id=user.email,
        engagement_id=request.engagement_id,
        agent_id=user.email,
        job_type=request.job_type,
        priority=request.priority,
        payload=payload,
        workspace_id=workspace_id,
        timeout_seconds=request.timeout_seconds,
    )
    job_id = await queue.enqueue_job(job)
    return {"job_id": job_id, "status": "queued", "tenant_id": user.tenant_id}


@router.get("/")
async def list_jobs(
    limit: int = Query(50, ge=1, le=100),
    user: User = Depends(require_auth),
):
    """List jobs for the caller's tenant."""
    queue = get_job_queue()
    jobs = await queue.list_jobs(
        tenant_id=user.tenant_id,
        owner_id=None if user.role == UserRole.SUPER_ADMIN else user.email,
        limit=limit,
    )
    return {"jobs": [j.model_dump() for j in jobs], "count": len(jobs)}


@router.get("/{job_id}")
async def get_job_status(
    job_id: str,
    user: User = Depends(require_auth),
):
    """Get job status and results (Tenant-isolated)."""
    queue = get_job_queue()
    job = await queue.get_job(
        job_id,
        tenant_id=user.tenant_id if user.role != UserRole.SUPER_ADMIN else None,
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or unauthorized",
        )
    if user.role != UserRole.SUPER_ADMIN and job.owner_id and job.owner_id != user.email:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or unauthorized",
        )
    return {"job": job.model_dump()}
