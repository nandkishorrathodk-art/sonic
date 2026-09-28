"""Persistent AI-employee organization store backed by SONIC's relational data plane."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from sonic.db.models import (
    AIEmployeeModel,
    GitHubReceiptModel,
    OrganizationProjectModel,
    OrganizationWorkItemModel,
)
from sonic.db.session import get_session_factory, init_db
from sonic.organization.architecture import (
    DEPARTMENTS,
    SUBSYSTEM_OWNERSHIP,
    TEAMS,
    Department,
    Team,
    ownership_for_path,
)


class OrganizationStore:
    """Persistent tenant-scoped employee/project/work-item repository."""

    _schema_ready = False

    @classmethod
    async def _ensure_schema(cls) -> None:
        if not cls._schema_ready:
            await init_db()
            cls._schema_ready = True

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _team_department(team_id: str) -> Department:
        for item in TEAMS:
            if item.name.value == team_id:
                return item.department
        raise ValueError(f"Unknown team: {team_id}")

    @staticmethod
    def _subsystem(subsystem_id: str):
        for item in SUBSYSTEM_OWNERSHIP:
            if item.subsystem == subsystem_id:
                return item
        raise ValueError(f"Unknown subsystem: {subsystem_id}")

    def _validate_assignment(
        self, department_id: str, team_id: str, subsystem_ids: list[str]
    ) -> None:
        try:
            department = Department(department_id)
            Team(team_id)
        except ValueError as exc:
            raise ValueError("Unknown department or team") from exc
        if self._team_department(team_id) != department:
            raise ValueError("Team is not owned by the selected department")
        for subsystem_id in subsystem_ids:
            owner = self._subsystem(subsystem_id)
            if owner.department != department or owner.team.value != team_id:
                raise ValueError(
                    f"Subsystem '{subsystem_id}' is not owned by {department_id}/{team_id}"
                )

    async def create_employee(
        self,
        tenant_id: str,
        name: str,
        role: str,
        department_id: str,
        team_id: str,
        subsystem_ids: list[str],
        read_only: bool = False,
        employee_type: str = "ai",
        prompt: str = "",
    ) -> dict[str, Any]:
        await self._ensure_schema()
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Employee name is required")
        self._validate_assignment(department_id, team_id, subsystem_ids)
        employee = AIEmployeeModel(
            id=f"emp-{uuid.uuid4().hex[:12]}", tenant_id=tenant_id,
            name=clean_name, role=role.strip(), employee_type=employee_type,
            department_id=department_id, team_id=team_id,
            subsystem_ids=subsystem_ids, read_only=read_only, prompt=prompt.strip(),
            status="active", created_at=self._now(),
        )
        async with get_session_factory()() as session:
            session.add(employee)
            await session.commit()
        return self._employee_dict(employee)

    async def get_employee(self, tenant_id: str, employee_id: str) -> dict[str, Any] | None:
        await self._ensure_schema()
        async with get_session_factory()() as session:
            result = await session.execute(
                select(AIEmployeeModel).where(
                    AIEmployeeModel.id == employee_id,
                    AIEmployeeModel.tenant_id == tenant_id,
                )
            )
            employee = result.scalar_one_or_none()
        return self._employee_dict(employee) if employee else None

    async def list_employees(self, tenant_id: str) -> list[dict[str, Any]]:
        await self._ensure_schema()
        async with get_session_factory()() as session:
            result = await session.execute(
                select(AIEmployeeModel)
                .where(AIEmployeeModel.tenant_id == tenant_id)
                .order_by(AIEmployeeModel.created_at)
            )
            return [self._employee_dict(item) for item in result.scalars().all()]

    async def create_project(
        self,
        tenant_id: str,
        name: str,
        description: str = "",
        lead_employee_id: str | None = None,
    ) -> dict[str, Any]:
        await self._ensure_schema()
        if lead_employee_id:
            lead = await self.get_employee(tenant_id, lead_employee_id)
            if lead is None:
                raise ValueError("Project lead employee not found in tenant")
        project = OrganizationProjectModel(
            id=f"proj-{uuid.uuid4().hex[:12]}", tenant_id=tenant_id,
            name=name.strip(), description=description.strip(),
            lead_employee_id=lead_employee_id, status="planned", created_at=self._now(),
        )
        async with get_session_factory()() as session:
            session.add(project)
            await session.commit()
        return self._project_dict(project)

    async def list_projects(self, tenant_id: str) -> list[dict[str, Any]]:
        await self._ensure_schema()
        async with get_session_factory()() as session:
            result = await session.execute(
                select(OrganizationProjectModel)
                .where(OrganizationProjectModel.tenant_id == tenant_id)
                .order_by(OrganizationProjectModel.created_at)
            )
            return [self._project_dict(item) for item in result.scalars().all()]

    async def create_work_item(
        self,
        tenant_id: str,
        project_id: str,
        title: str,
        description: str,
        subsystem_id: str,
        employee_id: str,
        github_repo: str = "",
        github_issue_number: int | None = None,
        github_pr_number: int | None = None,
        development_branch: str = "",
    ) -> dict[str, Any]:
        await self._ensure_schema()
        employee = await self.get_employee(tenant_id, employee_id)
        if employee is None:
            raise ValueError("Employee not found in tenant")
        if subsystem_id not in employee["subsystem_ids"]:
            raise ValueError("Employee does not own the requested subsystem")
        owner = self._subsystem(subsystem_id)
        if not project_id:
            raise ValueError("Project is required")
        async with get_session_factory()() as session:
            project = await session.execute(
                select(OrganizationProjectModel).where(
                    OrganizationProjectModel.id == project_id,
                    OrganizationProjectModel.tenant_id == tenant_id,
                )
            )
            if project.scalar_one_or_none() is None:
                raise ValueError("Project not found in tenant")
            item = OrganizationWorkItemModel(
                id=f"work-{uuid.uuid4().hex[:12]}", tenant_id=tenant_id,
                project_id=project_id, title=title.strip(), description=description.strip(),
                subsystem_id=owner.subsystem, department_id=owner.department.value,
                team_id=owner.team.value, assignee_employee_id=employee_id,
                github_repo=github_repo.strip(), github_issue_number=github_issue_number,
                github_pr_number=github_pr_number, development_branch=development_branch.strip(),
                status="assigned", created_at=self._now(), updated_at=self._now(),
            )
            session.add(item)
            await session.commit()
        return self._work_item_dict(item)

    async def list_work_items(self, tenant_id: str) -> list[dict[str, Any]]:
        await self._ensure_schema()
        async with get_session_factory()() as session:
            result = await session.execute(
                select(OrganizationWorkItemModel)
                .where(OrganizationWorkItemModel.tenant_id == tenant_id)
                .order_by(OrganizationWorkItemModel.created_at)
            )
            return [self._work_item_dict(item) for item in result.scalars().all()]

    async def update_work_item_status(
        self, tenant_id: str, work_item_id: str, status: str
    ) -> dict[str, Any]:
        await self._ensure_schema()
        normalized = status.strip().lower()
        allowed = {"assigned", "in_progress", "review", "evaluating", "blocked", "released"}
        if normalized not in allowed:
            raise ValueError(f"Unsupported work-item status: {status}")
        async with get_session_factory()() as session:
            result = await session.execute(
                select(OrganizationWorkItemModel).where(
                    OrganizationWorkItemModel.id == work_item_id,
                    OrganizationWorkItemModel.tenant_id == tenant_id,
                )
            )
            item = result.scalar_one_or_none()
            if item is None:
                raise ValueError("Work item not found in tenant")
            if normalized == "released":
                await self._assert_release_receipts(session, item)
            item.status = normalized
            item.updated_at = self._now()
            await session.commit()
            return self._work_item_dict(item)

    async def record_github_receipt(
        self,
        tenant_id: str,
        event_type: str,
        repository: str,
        ref: str,
        external_id: str,
        approved: bool = False,
        ci_success: bool = False,
    ) -> dict[str, Any]:
        await self._ensure_schema()
        receipt = GitHubReceiptModel(
            id=f"ghr-{uuid.uuid4().hex[:12]}", tenant_id=tenant_id,
            event_type=event_type, repository=repository, ref=ref,
            external_id=external_id, approved=approved, ci_success=ci_success,
            received_at=self._now(),
        )
        async with get_session_factory()() as session:
            session.add(receipt)
            await session.commit()
        return {
            "id": receipt.id, "event_type": event_type, "repository": repository,
            "ref": ref, "external_id": external_id, "approved": approved,
            "ci_success": ci_success, "received_at": receipt.received_at.isoformat(),
        }

    async def _assert_release_receipts(self, session: Any, item: OrganizationWorkItemModel) -> None:
        if not item.github_repo or item.github_pr_number is None:
            raise ValueError("Release requires a GitHub repository and PR reference")
        result = await session.execute(
            select(GitHubReceiptModel).where(
                GitHubReceiptModel.tenant_id == item.tenant_id,
                GitHubReceiptModel.repository == item.github_repo,
                GitHubReceiptModel.external_id == str(item.github_pr_number),
            )
        )
        receipts = result.scalars().all()
        if not any(r.approved for r in receipts):
            raise ValueError("Release blocked: no approved GitHub review receipt")
        if not any(r.ci_success for r in receipts):
            raise ValueError("Release blocked: no successful GitHub CI receipt")

    async def snapshot(self, tenant_id: str) -> dict[str, Any]:
        return {
            "departments": [d.value for d in DEPARTMENTS],
            "teams": [
                {"id": t.name.value, "department_id": t.department.value, "mission": t.mission}
                for t in TEAMS
            ],
            "subsystems": [
                {
                    "id": s.subsystem, "department_id": s.department.value,
                    "team_id": s.team.value, "path_prefixes": list(s.path_prefixes),
                    "responsibility": s.responsibility,
                }
                for s in SUBSYSTEM_OWNERSHIP
            ],
            "employees": await self.list_employees(tenant_id),
            "projects": await self.list_projects(tenant_id),
            "work_items": await self.list_work_items(tenant_id),
        }

    def validate_repository_path(self, path: str) -> dict[str, str]:
        owner = ownership_for_path(path)
        return {
            "department_id": owner.department.value,
            "team_id": owner.team.value,
            "subsystem_id": owner.subsystem,
        }

    @staticmethod
    def _employee_dict(item: AIEmployeeModel) -> dict[str, Any]:
        return {
            "id": item.id, "tenant_id": item.tenant_id, "name": item.name,
            "role": item.role, "employee_type": item.employee_type,
            "department_id": item.department_id, "team_id": item.team_id,
            "subsystem_ids": list(item.subsystem_ids or []), "read_only": item.read_only,
            "prompt": item.prompt, "status": item.status,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    @staticmethod
    def _project_dict(item: OrganizationProjectModel) -> dict[str, Any]:
        return {
            "id": item.id, "tenant_id": item.tenant_id, "name": item.name,
            "description": item.description, "lead_employee_id": item.lead_employee_id,
            "status": item.status,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    @staticmethod
    def _work_item_dict(item: OrganizationWorkItemModel) -> dict[str, Any]:
        return {
            "id": item.id, "tenant_id": item.tenant_id, "project_id": item.project_id,
            "title": item.title, "description": item.description,
            "subsystem_id": item.subsystem_id, "department_id": item.department_id,
            "team_id": item.team_id, "assignee_employee_id": item.assignee_employee_id,
            "github_repo": item.github_repo, "github_issue_number": item.github_issue_number,
            "github_pr_number": item.github_pr_number, "development_branch": item.development_branch,
            "status": item.status,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        }


_store = OrganizationStore()


def get_organization_store() -> OrganizationStore:
    return _store
