from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sonic.api.routes.organization import router
from sonic.auth.middleware import require_auth, require_operator
from sonic.auth.models import User, UserRole
from sonic.organization.store import OrganizationStore
from sonic.safety.sealed import SealedActionPolicy


def test_organization_store_seeds_structure_and_persists(tmp_path):
    db_path = tmp_path / "organization.sqlite"
    first = OrganizationStore(db_path)

    snapshot = first.snapshot("tenant-a")

    assert len(snapshot["departments"]) == 7
    assert len(snapshot["teams"]) >= 10
    assert len(snapshot["subsystems"]) >= 30
    assert snapshot["subsystems"][0]["department_id"]
    assert snapshot["subsystems"][0]["paths"]
    assert len(snapshot["employees"]) == 6
    assert snapshot["work_items"] == []
    assert OrganizationStore(db_path).snapshot("tenant-a") == snapshot


def test_subsystem_ownership_is_enforced_at_employee_and_work_item_boundaries(tmp_path):
    store = OrganizationStore(tmp_path / "organization.sqlite")
    snapshot = store.snapshot("tenant-a")
    browser_team = next(
        team for team in snapshot["teams"] if team["id"] == "browser_web"
    )
    try:
        store.create_employee(
            "tenant-a",
            {
                "team_id": browser_team["id"],
                "name": "Invalid Owner",
                "role": "Engineer",
                "career_track": "individual_contributor",
                "level": "engineer",
                "capability_mode": "engineering",
                "system_prompt": "",
                "skills": [],
                "primary_ownership": [],
                "secondary_ownership": [],
                "subsystem_ids": ["safety"],
                "status": "active",
            },
        )
    except ValueError as exc:
        assert "employee's team" in str(exc)
    else:
        raise AssertionError("Cross-team subsystem ownership must be rejected")

    lead = snapshot["employees"][0]
    project = store.create_project(
        "tenant-a",
        "operator@example.test",
        {
            "name": "Ownership Enforcement",
            "description": "",
            "status": "active",
            "lead_employee_id": lead["id"],
            "github_repository": "",
        },
    )
    try:
        store.create_work_item(
            "tenant-a",
            "operator@example.test",
            {
                "project_id": project["id"],
                "title": "Wrong subsystem",
                "description": "",
                "item_type": "task",
                "priority": "medium",
                "subsystem_id": "dashboard",
                "assignee_employee_id": lead["id"],
                "github_issue_url": "",
            },
        )
    except ValueError as exc:
        assert "does not own" in str(exc)
    else:
        raise AssertionError("Work must not be assigned outside employee ownership")


def test_project_work_item_assignment_and_release_gate_are_tenant_scoped(tmp_path):
    store = OrganizationStore(tmp_path / "organization.sqlite")
    snapshot = store.snapshot("tenant-a")
    lead = snapshot["employees"][0]
    project = store.create_project(
        "tenant-a",
        "operator@example.test",
        {
            "name": "Computer Use Reliability",
            "description": "Improve verified desktop interaction.",
            "status": "active",
            "lead_employee_id": lead["id"],
            "github_repository": "",
        },
    )
    work_item = store.create_work_item(
        "tenant-a",
        "operator@example.test",
        {
            "project_id": project["id"],
            "title": "Verify focus-aware typing",
            "description": "Add a reliable focus check.",
            "item_type": "task",
            "priority": "high",
            "assignee_employee_id": lead["id"],
            "github_issue_url": "",
        },
    )

    assert work_item["assignee_employee_id"] == lead["id"]
    try:
        store.update_work_item(
            "tenant-a", work_item["id"], {"status": "released"}
        )
    except ValueError as exc:
        assert "verified GitHub events" in str(exc)
    else:
        raise AssertionError("Release without review/evaluation must be rejected")

    released = store.update_work_item(
        "tenant-a",
        work_item["id"],
        {
            "status": "in_review",
        },
    )
    assert released["status"] == "in_review"

    isolated = store.snapshot("tenant-b")
    assert isolated["projects"] == []
    assert isolated["work_items"] == []
    try:
        store.create_work_item(
            "tenant-b",
            "other@example.test",
            {
                "project_id": project["id"],
                "title": "Cross-tenant attempt",
                "description": "",
                "item_type": "task",
                "priority": "medium",
                "assignee_employee_id": None,
                "github_issue_url": "",
            },
        )
    except ValueError as exc:
        assert "Project does not exist" in str(exc)
    else:
        raise AssertionError("Cross-tenant project reference must be rejected")


def test_signed_github_receipts_drive_review_evaluation_and_release(tmp_path):
    store = OrganizationStore(tmp_path / "organization.sqlite")
    lead = store.snapshot("tenant-a")["employees"][0]
    project = store.create_project(
        "tenant-a",
        "operator@example.test",
        {
            "name": "Agent Reliability",
            "description": "",
            "status": "active",
            "lead_employee_id": lead["id"],
            "github_repository": "https://github.example.test/team/sonic",
        },
    )
    item = store.create_work_item(
        "tenant-a",
        "operator@example.test",
        {
            "project_id": project["id"],
            "title": "Fix agent recovery",
            "description": "",
            "item_type": "task",
            "priority": "high",
            "assignee_employee_id": lead["id"],
            "github_issue_url": "",
        },
    )
    store.update_work_item(
        "tenant-a", item["id"], {"branch_name": "fix/agent-recovery"}
    )
    repo = {"html_url": "https://github.example.test/team/sonic"}
    pr_url = "https://github.example.test/team/sonic/pull/42"
    sha = "a" * 40
    pull_request = {
        "html_url": pr_url,
        "head": {"ref": "fix/agent-recovery", "sha": sha},
        "merged": False,
    }
    assert store.record_github_event(
        "pull_request",
        {"repository": repo, "action": "opened", "pull_request": pull_request},
    ) == 1
    assert store.record_github_event(
        "check_suite",
        {
            "repository": repo,
            "action": "completed",
            "check_suite": {
                "status": "completed",
                "head_branch": "fix/agent-recovery",
                "head_sha": sha,
                "conclusion": "success",
                "id": 101,
                "html_url": "https://github.example.test/team/sonic/actions/runs/101",
            },
        },
    ) == 1
    assert store.record_github_event(
        "pull_request_review",
        {
            "repository": repo,
            "action": "submitted",
            "pull_request": {"html_url": pr_url},
            "review": {
                "commit_id": sha,
                "state": "approved",
                "user": {"login": "reviewer"},
            },
        },
    ) == 1
    pull_request["merged"] = True
    assert store.record_github_event(
        "pull_request",
        {"repository": repo, "action": "closed", "pull_request": pull_request},
    ) == 1

    final_item = next(
        work
        for work in store.snapshot("tenant-a")["work_items"]
        if work["id"] == item["id"]
    )
    assert final_item["status"] == "released"
    assert final_item["review_status"] == "approved"
    assert final_item["github_reviewed_by"] == "reviewer"
    assert final_item["evaluation_status"] == "passed"
    assert final_item["verification"]["source"] == "github_check_suite"

    stale_review = store.record_github_event(
        "pull_request_review",
        {
            "repository": repo,
            "action": "submitted",
            "pull_request": {"html_url": pr_url},
            "review": {
                "commit_id": "b" * 40,
                "state": "approved",
                "user": {"login": "stale-reviewer"},
            },
        },
    )
    assert stale_review == 0


def test_organization_api_exposes_tenant_snapshot_and_creates_work(tmp_path, monkeypatch):
    monkeypatch.setenv("SONIC_ORGANIZATION_DB_PATH", str(tmp_path / "api.sqlite"))
    from sonic.organization.store import reset_organization_store

    reset_organization_store()
    user = User(
        email="operator@example.test",
        name="Operator",
        tenant_id="tenant-api",
        role=UserRole.OPERATOR,
    )
    app = FastAPI()
    app.include_router(router, prefix="/organization")
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[require_operator] = lambda: user
    client = TestClient(app)

    initial = client.get("/organization/")
    assert initial.status_code == 200
    assert len(initial.json()["departments"]) == 7
    team_id = initial.json()["teams"][0]["id"]
    employee = client.post(
        "/organization/employees",
        json={
            "team_id": team_id,
            "name": "Browser Automation Engineer",
            "role": "Browser Engineer",
            "skills": ["browser automation"],
            "primary_ownership": ["browser runtime"],
        },
    )
    assert employee.status_code == 201
    project = client.post(
        "/organization/projects",
        json={"name": "Browser Reliability", "github_repository": ""},
    )
    assert project.status_code == 201
    work = client.post(
        "/organization/work-items",
        json={
            "project_id": project.json()["project"]["id"],
            "title": "Add independent browser verification",
            "assignee_employee_id": employee.json()["employee"]["id"],
        },
    )
    assert work.status_code == 201
    assert work.json()["work_item"]["assignee_employee_id"] == employee.json()["employee"]["id"]
    invalid = client.post(
        "/organization/projects",
        json={"name": "Invalid URL", "github_repository": "not-a-url"},
    )
    assert invalid.status_code == 422


def test_github_webhook_requires_configured_secret_and_valid_signature(tmp_path, monkeypatch):
    monkeypatch.setenv("SONIC_ORGANIZATION_DB_PATH", str(tmp_path / "webhook.sqlite"))
    monkeypatch.setenv("SONIC_GITHUB_WEBHOOK_SECRET", "local-test-secret")
    from sonic.organization.store import reset_organization_store

    reset_organization_store()
    app = FastAPI()
    app.include_router(router, prefix="/organization")
    client = TestClient(app)
    body = json.dumps({"repository": {"html_url": "https://github.example.test/org/repo"}}).encode()
    signature = "sha256=" + hmac.new(
        b"local-test-secret", body, hashlib.sha256
    ).hexdigest()

    rejected = client.post(
        "/organization/github/webhook",
        content=body,
        headers={
            "x-github-event": "ping",
            "x-hub-signature-256": "sha256=invalid",
        },
    )
    assert rejected.status_code == 401
    accepted = client.post(
        "/organization/github/webhook",
        content=body,
        headers={
            "x-github-event": "ping",
            "x-hub-signature-256": signature,
        },
    )
    assert accepted.status_code == 200


def test_employee_profile_drives_subagent_role_scope_and_sealed_read_only_mode(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "employee-runtime.sqlite"
    monkeypatch.setenv("SONIC_ORGANIZATION_DB_PATH", str(db_path))
    from sonic.organization.store import reset_organization_store

    reset_organization_store()
    store = OrganizationStore(db_path)
    team_id = store.snapshot("tenant-runtime")["teams"][0]["id"]
    employee = store.create_employee(
        "tenant-runtime",
        {
            "team_id": team_id,
            "name": "Read-only evaluator",
            "role": "Evaluator",
            "career_track": "individual_contributor",
            "level": "senior",
            "capability_mode": "read_only",
            "system_prompt": "Inspect artifacts and report reproducible failures only.",
            "skills": ["test evaluation"],
            "primary_ownership": ["test outcomes"],
            "secondary_ownership": [],
            "status": "active",
        },
    )
    from sonic.computer_use.teamwork import SubagentTeamworkManager

    manager = SubagentTeamworkManager(
        parent_agent=SimpleNamespace(tenant_id="tenant-runtime", security_tools={}),
        computer_provider=object(),
        llm_router=None,
        safety=SealedActionPolicy(tenant_id="tenant-runtime"),
        tenant_id="tenant-runtime",
    )

    async def prepare_workspace(workspace_id, conversation_id, mode):
        return workspace_id, None

    async def skip_execution(conv, spec, formatted_goal):
        conv.verified = True
        conv.judge_verdict = "CONFIRMED"
        conv.state = "idle"
        conv.last_summary = "Tested fixture"

    manager._prepare_subagent_workspace = prepare_workspace
    manager._execute_and_verify_loop = skip_execution
    conversations = asyncio.run(
        manager.invoke_subagents(
            "workspace-runtime",
            [
                {
                    "employee_id": employee["id"],
                    "prompt": "Review the test result.",
                    "workspace_mode": "inherit",
                }
            ],
        )
    )

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation.employee_id == employee["id"]
    assert conversation.role == "Evaluator"
    assert "Owned subsystems: test outcomes" in conversation.messages[0]["content"]
    from sonic.computer_use.teamwork import _subagent_policy

    restricted = _subagent_policy(
        manager.safety,
        "read-only-check",
        None,
        read_only=True,
        disable_subagents=True,
    )
    assert restricted.sealed
    assert restricted.allowed_types == frozenset({
        "FILE_READ",
        "GIT_STATUS",
        "GIT_DIFF",
        "GUI_SCREENSHOT",
        "BROWSER_SCREENSHOT",
        "BROWSER_WAIT",
        "GUI_WAIT",
    })
    assert "FILE_WRITE" in manager.safety.allowed_types
