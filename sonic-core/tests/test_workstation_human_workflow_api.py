"""Focused API coverage for the planning-only human workflow endpoint."""

from fastapi.testclient import TestClient

from sonic.api.main import app
from sonic.api.routes import workstation
from sonic.auth.middleware import require_operator
from sonic.auth.models import User, UserRole


def test_operator_can_plan_without_execution(monkeypatch):
    user = User(
        email="operator@example.invalid",
        name="Operator",
        role=UserRole.OPERATOR,
        tenant_id="tenant-plan",
    )
    app.dependency_overrides[require_operator] = lambda: user
    monkeypatch.setattr(workstation, "_persist_workstation_state", lambda: None)
    monkeypatch.setattr(
        workstation,
        "get_daytona_computer",
        lambda: (_ for _ in ()).throw(AssertionError("planning must not access a computer")),
    )
    workstation.reset_workstation_state()

    try:
        response = TestClient(app).post(
            "/workstation/workflow/plan",
            json={"objective": "improve the service", "session_id": "planning"},
        )
        assert response.status_code == 200
        body = response.json()
        workflow = body["workflow"]
        assert body["status"] == "planned"
        assert body["session_id"] == "planning"
        assert [stage["name"] for stage in workflow["stages"]] == [
            "inspect", "reproduce", "patch", "test", "launch", "verify"
        ]
        assert workflow["ready_stages"] == ["inspect"]
        assert workflow["outcome"] == "incomplete"
        assert workflow["execution_started"] is False
        assert workflow["commands_executed"] is False
        assert workflow["commits_executed"] is False
    finally:
        app.dependency_overrides.pop(require_operator, None)


def test_workflow_plan_is_tenant_and_session_scoped(monkeypatch):
    monkeypatch.setattr(workstation, "_persist_workstation_state", lambda: None)
    workstation.reset_workstation_state()
    first = User(
        email="one@example.invalid", name="One",
        role=UserRole.OPERATOR, tenant_id="tenant-one",
    )
    second = User(
        email="two@example.invalid", name="Two",
        role=UserRole.OPERATOR, tenant_id="tenant-two",
    )

    app.dependency_overrides[require_operator] = lambda: first
    try:
        first_response = TestClient(app).post(
            "/workstation/workflow/plan",
            json={"objective": "first objective", "session_id": "same"},
        )
        app.dependency_overrides[require_operator] = lambda: second
        second_response = TestClient(app).post(
            "/workstation/workflow/plan",
            json={"objective": "second objective", "session_id": "same"},
        )
        assert first_response.status_code == second_response.status_code == 200
        assert first_response.json()["workflow"]["objective"] == "first objective"
        assert second_response.json()["workflow"]["objective"] == "second objective"
        assert workstation._tenant_workstations["tenant-one"]["same"]["human_workflow"]["objective"] == "first objective"
        assert workstation._tenant_workstations["tenant-two"]["same"]["human_workflow"]["objective"] == "second objective"
    finally:
        app.dependency_overrides.pop(require_operator, None)


def test_non_operator_is_rejected():
    response = TestClient(app).post(
        "/workstation/workflow/plan",
        json={"objective": "inspect the code"},
    )
    assert response.status_code == 401
