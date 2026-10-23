"""Boss worker identity, live activity projection, and agent drill-down API.

Covers the operator-facing fixes:

* Every launched worker gets a run-unique ``agent_uid`` (the positional
  ``agent_id`` is reused across runs, so it cannot address one worker).
* A worker that produced a terminal report is never shown as RUNNING.
* A worker whose owning background task is gone (finished mission, crash, or
  control-plane restart) is reported INTERRUPTED, not a stale RUNNING.
* A crashed worker (unhandled exception in the wave) still emits a terminal
  report so it cannot be stuck on RUNNING forever.
* ``GET /workstation/agents`` and ``/workstation/agents/{uid}`` expose this
  view, tenant-scoped and authenticated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sonic.computer_use.agent_activity import (
    collect_worker_activity,
    find_worker,
)
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult


class _MockProvider:
    async def screenshot(self, *args, **kwargs):
        return b"desktop"

    async def gui_action(self, *args, **kwargs):
        return True


class _MockLLM:
    async def complete(self, request, **kwargs):
        return MagicMock(content="{}")


def _dispatch_event(uid: str, agent_id: str = "sub-agent-1", **extra) -> dict:
    return {
        "type": "action",
        "title": f"Dispatching Worker ({agent_id})",
        "orchestration_event": "sub_dispatch",
        "agent_id": agent_id,
        "agent_uid": uid,
        "agent_name": "Endpoint Scout",
        "sub_mission_id": "sub-1",
        "sub_agent_number": 1,
        "goal": "Inspect the target",
        "worker_role": "subagent",
        "execution_plane": "gui",
        "status": "RUNNING",
        **extra,
    }


def _report_event(uid: str, status: str, success: bool, **extra) -> dict:
    return {
        "type": "info",
        "title": f"Endpoint Scout (sub-agent-1) Report — {status}",
        "orchestration_event": "sub_report",
        "agent_id": "sub-agent-1",
        "agent_uid": uid,
        "agent_name": "Endpoint Scout",
        "sub_mission_id": "sub-1",
        "sub_agent_number": 1,
        "status": status,
        "success": success,
        "findings_summary": "Verified the endpoint.",
        **extra,
    }


# ==========================================================
# Pure projection: collect_worker_activity
# ==========================================================

def test_running_worker_requires_live_task():
    worklog = [_dispatch_event("worker-1-abc")]
    # No live task -> the worker did not finish, so it must not read RUNNING.
    workers = collect_worker_activity(worklog, mission_task_alive=False)
    assert len(workers) == 1
    assert workers[0]["status"] == "INTERRUPTED"
    assert workers[0]["is_running"] is False

    # With a live task it is genuinely running.
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    assert workers[0]["status"] == "RUNNING"
    assert workers[0]["is_running"] is True


def test_terminal_report_wins_over_live_task():
    worklog = [
        _dispatch_event("worker-1-abc"),
        _report_event("worker-1-abc", "SUCCESS", True),
    ]
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    assert workers[0]["status"] == "SUCCESS"
    assert workers[0]["is_running"] is False


def test_failed_report_is_reported_failed():
    worklog = [
        _dispatch_event("worker-1-abc"),
        _report_event("worker-1-abc", "FAILED", False),
    ]
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    assert workers[0]["status"] == "FAILED"


def test_multiple_workers_are_tracked_independently():
    worklog = [
        _dispatch_event("worker-1-aaa", agent_id="sub-agent-1"),
        _dispatch_event(
            "worker-2-bbb",
            agent_id="sub-agent-2",
            sub_mission_id="sub-2",
            sub_agent_number=2,
        ),
        _report_event("worker-1-aaa", "SUCCESS", True),
    ]
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    by_uid = {w["agent_uid"]: w for w in workers}
    assert by_uid["worker-1-aaa"]["status"] == "SUCCESS"
    assert by_uid["worker-2-bbb"]["status"] == "RUNNING"


def test_step_items_attach_to_the_owning_worker_by_uid():
    worklog = [
        _dispatch_event("worker-1-abc"),
        {
            "type": "action",
            "sub_agent_id": "sub-agent-1",
            "sub_agent_uid": "worker-1-abc",
            "action_type": "GUI_CLICK",
            "target_resource": "Save button",
            "action_status": "VERIFIED",
            "step_index": 2,
        },
    ]
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    assert workers[0]["step_count"] == 1
    assert workers[0]["latest_action"] == "gui click · Save button"
    assert workers[0]["steps"][0]["action_type"] == "GUI_CLICK"


def test_two_workers_with_same_agent_id_are_not_merged_by_uid():
    """agent_id is reused across runs; the uid must keep them distinct."""
    worklog = [
        _dispatch_event("worker-1-run1", agent_id="sub-agent-1"),
        _dispatch_event("worker-1-run2", agent_id="sub-agent-1"),
    ]
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    assert len(workers) == 2
    assert {w["agent_uid"] for w in workers} == {"worker-1-run1", "worker-1-run2"}


def test_find_worker_resolves_by_uid_and_agent_id():
    worklog = [
        _dispatch_event("worker-1-abc"),
        _dispatch_event(
            "worker-2-def",
            agent_id="sub-agent-2",
            sub_mission_id="sub-2",
            sub_agent_number=2,
        ),
    ]
    assert find_worker(worklog, "worker-2-def")["sub_mission_id"] == "sub-2"
    assert find_worker(worklog, "sub-agent-1")["agent_uid"] == "worker-1-abc"
    assert find_worker(worklog, "2")["agent_uid"] == "worker-2-def"
    assert find_worker(worklog, "does-not-exist") is None


# ==========================================================
# Boss emits a run-unique uid and terminal reports for crashes
# ==========================================================

@pytest.mark.asyncio
async def test_boss_emits_run_unique_agent_uid_and_reports_it():
    """The wave loop must mint one uid and reuse it on dispatch and report."""
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    boss.max_phases = 1
    events: list[tuple[str, dict]] = []

    async def on_event(event_type, data):
        events.append((event_type, data))

    phase = Phase(
        phase_number=1,
        name="Single",
        sub_missions=[
            SubMission(id="sub-uid", goal="Inspect", plane="headless", max_steps=1)
        ],
    )

    async def fake_dispatch(workspace_id, sub_m, *args, **kwargs):
        return SubMissionResult(
            sub_mission_id=sub_m.id,
            goal=sub_m.goal,
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="ok",
            actions_taken=1,
        )

    with patch.object(boss, "_strategic_decomposition", AsyncMock(return_value=phase)), \
         patch.object(boss, "_dispatch_sub_agent", AsyncMock(side_effect=fake_dispatch)), \
         patch.object(boss, "_aggregate_findings", AsyncMock(return_value="")), \
         patch.object(boss, "_check_objective_complete", AsyncMock(return_value=(False, ""))), \
         patch.object(boss, "_plan_next_phase", AsyncMock(return_value=None)):
        await boss.run(
            workspace_id="ws-1",
            objective="Inspect the target",
            phase_callback=on_event,
        )

    dispatches = [d for t, d in events if t == "sub_dispatch"]
    reports = [d for t, d in events if t == "sub_report"]
    assert dispatches and reports
    uid = dispatches[0]["agent_uid"]
    assert uid.startswith("worker-1-")
    # Same identity flows through the terminal report.
    assert reports[0]["agent_uid"] == uid


@pytest.mark.asyncio
async def test_boss_mints_distinct_uids_for_each_worker():
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    boss.max_phases = 1
    events: list[tuple[str, dict]] = []

    async def on_event(event_type, data):
        events.append((event_type, data))

    phase = Phase(
        phase_number=1,
        name="Two",
        sub_missions=[
            SubMission(id="sub-a", goal="A", plane="headless", max_steps=1),
            SubMission(id="sub-b", goal="B", plane="headless", max_steps=1),
        ],
    )

    async def fake_dispatch(workspace_id, sub_m, *args, **kwargs):
        return SubMissionResult(
            sub_mission_id=sub_m.id,
            goal=sub_m.goal,
            success=True,
            status="SUCCESS",
            evidence_verified=True,
            findings_summary="ok",
            actions_taken=1,
        )

    with patch.object(boss, "_strategic_decomposition", AsyncMock(return_value=phase)), \
         patch.object(boss, "_dispatch_sub_agent", AsyncMock(side_effect=fake_dispatch)), \
         patch.object(boss, "_aggregate_findings", AsyncMock(return_value="")), \
         patch.object(boss, "_check_objective_complete", AsyncMock(return_value=(False, ""))), \
         patch.object(boss, "_plan_next_phase", AsyncMock(return_value=None)):
        await boss.run(
            workspace_id="ws-1",
            objective="Do A and B",
            phase_callback=on_event,
        )

    uids = [d["agent_uid"] for t, d in events if t == "sub_dispatch"]
    assert len(uids) == 2
    assert len(set(uids)) == 2, "each worker must get its own uid"


@pytest.mark.asyncio
async def test_crashed_worker_still_emits_terminal_report():
    """An unhandled exception must not leave the worker stuck on RUNNING."""
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    boss.max_phases = 1
    events: list[tuple[str, dict]] = []

    async def on_event(event_type, data):
        events.append((event_type, data))

    phase = Phase(
        phase_number=1,
        name="Crashy",
        sub_missions=[
            SubMission(id="sub-crash", goal="Explode", plane="headless", max_steps=1)
        ],
    )

    with patch.object(boss, "_strategic_decomposition", AsyncMock(return_value=phase)), \
         patch.object(
             boss,
             "_dispatch_sub_agent",
             AsyncMock(side_effect=RuntimeError("boom")),
         ), \
         patch.object(boss, "_aggregate_findings", AsyncMock(return_value="")), \
         patch.object(boss, "_check_objective_complete", AsyncMock(return_value=(False, ""))), \
         patch.object(boss, "_plan_next_phase", AsyncMock(return_value=None)):
        await boss.run(
            workspace_id="ws-1",
            objective="Explode safely",
            phase_callback=on_event,
        )

    dispatches = [d for t, d in events if t == "sub_dispatch"]
    reports = [d for t, d in events if t == "sub_report"]
    assert dispatches, "worker must have been dispatched"
    assert reports, "crashed worker must still emit a terminal report"
    assert reports[0]["status"] == "FAILED"
    assert reports[0]["agent_uid"] == dispatches[0]["agent_uid"]


# ==========================================================
# API surface: /workstation/agents
# ==========================================================

@pytest.fixture
def client():
    from sonic.api.main import app

    return TestClient(app)


def _login(client: TestClient, tenant_id: str = "default") -> str:
    res = client.post(
        "/auth/login",
        json={
            "email": f"u@{tenant_id}.com",
            "name": "U",
            "role": "operator",
            "tenant_id": tenant_id,
        },
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _seed_worklog(tenant_id: str, session_id: str, worklog: list[dict]) -> None:
    from sonic.api.routes import workstation

    state = workstation._get_or_create_session(tenant_id, session_id)
    state.setdefault("worklog", []).extend(worklog)


def test_agents_endpoint_requires_auth(client):
    assert client.get("/workstation/agents").status_code == 401


def test_agents_endpoint_lists_workers_and_status(client):
    token = _login(client)
    _seed_worklog(
        "default",
        "agent-list-session",
        [
            _dispatch_event("worker-1-abc"),
            _report_event("worker-1-abc", "SUCCESS", True),
        ],
    )
    res = client.get(
        "/workstation/agents?session_id=agent-list-session", headers=_auth(token)
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["count"] == 1
    assert body["workers"][0]["agent_uid"] == "worker-1-abc"
    assert body["workers"][0]["status"] == "SUCCESS"


def test_agent_detail_endpoint_and_404(client):
    token = _login(client)
    _seed_worklog("default", "agent-detail-session", [_dispatch_event("worker-9-xyz")])
    ok = client.get(
        "/workstation/agents/worker-9-xyz?session_id=agent-detail-session",
        headers=_auth(token),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["worker"]["agent_uid"] == "worker-9-xyz"

    missing = client.get(
        "/workstation/agents/nope?session_id=agent-detail-session",
        headers=_auth(token),
    )
    assert missing.status_code == 404


def test_agents_endpoint_is_tenant_scoped(client):
    token_a = _login(client, tenant_id="tenant-a")
    token_b = _login(client, tenant_id="tenant-b")
    _seed_worklog("tenant-a", "shared-session", [_dispatch_event("worker-1-abc")])

    res_a = client.get(
        "/workstation/agents?session_id=shared-session", headers=_auth(token_a)
    )
    res_b = client.get(
        "/workstation/agents?session_id=shared-session", headers=_auth(token_b)
    )
    assert res_a.json()["count"] == 1
    assert res_b.json()["count"] == 0


def test_state_endpoint_includes_workers_and_task_liveness(client):
    token = _login(client)
    _seed_worklog("default", "state-workers-session", [_dispatch_event("worker-1-abc")])
    res = client.get(
        "/workstation/state?session_id=state-workers-session", headers=_auth(token)
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert "workers" in body
    assert "mission_task_alive" in body
    assert body["workers"][0]["agent_uid"] == "worker-1-abc"
