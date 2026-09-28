import pytest

from sonic.organization.store import OrganizationStore


@pytest.mark.asyncio
async def test_employee_assignment_uses_canonical_ownership(tmp_path, monkeypatch):
    db_path = tmp_path / "organization.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    import sonic.db.session as db_session

    db_session._engine = None
    db_session._session_factory = None
    OrganizationStore._schema_ready = False

    store = OrganizationStore()
    employee = await store.create_employee(
        tenant_id="tenant-a",
        name="Cognitive Worker",
        role="AI Agent Engineer",
        department_id="cognitive_systems",
        team_id="reasoning_orchestration",
        subsystem_ids=["agents"],
    )
    assert employee["subsystem_ids"] == ["agents"]

    with pytest.raises(ValueError, match="not owned"):
        await store.create_employee(
            tenant_id="tenant-a",
            name="Invalid Worker",
            role="Engineer",
            department_id="cognitive_systems",
            team_id="reasoning_orchestration",
            subsystem_ids=["browser"],
        )


@pytest.mark.asyncio
async def test_release_requires_review_and_ci_receipts(tmp_path, monkeypatch):
    db_path = tmp_path / "organization-release.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    import sonic.db.session as db_session

    db_session._engine = None
    db_session._session_factory = None
    OrganizationStore._schema_ready = False

    store = OrganizationStore()
    employee = await store.create_employee(
        tenant_id="tenant-a", name="Platform Worker", role="Backend Engineer",
        department_id="platform_systems", team_id="api_services", subsystem_ids=["api"],
    )
    project = await store.create_project("tenant-a", "API Modernization", lead_employee_id=employee["id"])
    item = await store.create_work_item(
        tenant_id="tenant-a", project_id=project["id"], title="API work", description="",
        subsystem_id="api", employee_id=employee["id"], github_repo="acme/sonic",
        github_issue_number=10, github_pr_number=11, development_branch="feat/api",
    )

    with pytest.raises(ValueError, match="approved GitHub review"):
        await store.update_work_item_status("tenant-a", item["id"], "released")

    await store.record_github_receipt("tenant-a", "pull_request_review", "acme/sonic", "feat/api", "11", approved=True)
    with pytest.raises(ValueError, match="successful GitHub CI"):
        await store.update_work_item_status("tenant-a", item["id"], "released")

    await store.record_github_receipt("tenant-a", "check_suite", "acme/sonic", "feat/api", "11", ci_success=True)
    released = await store.update_work_item_status("tenant-a", item["id"], "released")
    assert released["status"] == "released"


@pytest.mark.asyncio
async def test_tenant_isolation(tmp_path, monkeypatch):
    db_path = tmp_path / "organization-tenant.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    import sonic.db.session as db_session

    db_session._engine = None
    db_session._session_factory = None
    OrganizationStore._schema_ready = False

    store = OrganizationStore()
    await store.create_employee(
        tenant_id="tenant-a", name="A", role="Engineer",
        department_id="platform_systems", team_id="api_services", subsystem_ids=["api"],
    )
    assert await store.list_employees("tenant-b") == []
