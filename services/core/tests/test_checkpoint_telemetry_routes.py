from __future__ import annotations

import pytest

from sonic.api.routes import checkpoints, workstation
from sonic.auth.models import User


class FakeProvider:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], str] = {}

    async def read_file(self, workspace_id: str, path: str) -> str:
        return self.files[(workspace_id, path)]

    async def write_file(
        self, workspace_id: str, path: str, content: str, *, actor: str
    ) -> bool:
        self.files[(workspace_id, path)] = content
        return True


@pytest.mark.asyncio
async def test_checkpoint_routes_are_operator_scoped_and_return_receipt(monkeypatch):
    checkpoints._managers.clear()
    workstation._tenant_workstations.clear()
    provider = FakeProvider()
    monkeypatch.setattr(checkpoints, "get_daytona_computer", lambda: provider)

    operator = User(email="a@example.invalid", name="A", tenant_id="tenant-a")
    session = workstation._get_or_create_session("tenant-a", "s1")
    session["desktop"]["workspace_id"] = "workspace-a"
    provider.files[("workspace-a", "notes.txt")] = "before"

    created = await checkpoints.create_checkpoint(
        checkpoints.CheckpointCreateRequest(
            name="before edit", artifacts=["notes.txt"], state={"phase": "inspect"}, session_id="s1"
        ),
        operator,
    )
    assert created["artifacts"] == ["notes.txt"]
    provider.files[("workspace-a", "notes.txt")] = "after"

    listed = await checkpoints.list_checkpoints(session_id="s1", user=operator)
    assert [item["id"] for item in listed["checkpoints"]] == [created["id"]]

    rolled = await checkpoints.rollback_checkpoint(
        created["id"],
        checkpoints.CheckpointRollbackRequest(session_id="s1"),
        operator,
    )
    assert rolled["action_receipt"]["status"] == "SUCCEEDED"
    assert "restored checkpoint" in rolled["action_receipt"]["evidence"]
    assert provider.files[("workspace-a", "notes.txt")] == "before"
    assert session["tenant_id"] == "tenant-a"
    assert session["desktop"]["workspace_id"] == "workspace-a"
    assert session["checkpoint_state"] == {"phase": "inspect"}

    other = User(email="b@example.invalid", name="B", tenant_id="tenant-b")
    with pytest.raises(Exception) as exc:
        await checkpoints.list_checkpoints(session_id="s1", user=other)
    assert getattr(exc.value, "status_code", None) == 503


@pytest.mark.asyncio
async def test_checkpoint_route_rejects_non_relative_artifact(monkeypatch):
    checkpoints._managers.clear()
    workstation._tenant_workstations.clear()
    monkeypatch.setattr(checkpoints, "get_daytona_computer", lambda: FakeProvider())
    user = User(email="a@example.invalid", name="A", tenant_id="tenant-a")
    session = workstation._get_or_create_session("tenant-a", "default")
    session["desktop"]["workspace_id"] = "workspace-a"

    with pytest.raises(Exception) as exc:
        await checkpoints.create_checkpoint(
            checkpoints.CheckpointCreateRequest(
                name="unsafe", artifacts=["../outside"], session_id="default"
            ),
            user,
        )
    assert getattr(exc.value, "status_code", None) == 400
