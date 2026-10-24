"""Contract test: the worker drill-down API the operator dashboard consumes.

The dashboard's ``/agents`` page and the worklog ``WorkerDetail`` panel both
read the same two endpoints:

* ``GET /workstation/agents``            -> ``{count, running_count, workers[]}``
* ``GET /workstation/agents/{agent_uid}`` -> ``{worker: {...}}``

Those API functions (``api.getWorkstationAgents`` / ``api.getWorkstationAgent``
in ``apps/dashboard/lib/api.ts``) existed but had no caller, and the page
rendered static cards. This file pins the response *shape* the newly-wired
frontend depends on, so a backend rename cannot silently break the drill-down
while every existing test still passes.

Every field asserted here is read by a named line of the dashboard code; see
the comment on each assertion.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# ==========================================================
# Fixtures
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


def _dispatch(uid: str, name: str, goal: str, plane: str = "headless") -> dict:
    return {
        "type": "action",
        "title": f"Dispatching {name}",
        "orchestration_event": "sub_dispatch",
        "agent_id": "sub-agent-1",
        "agent_uid": uid,
        "agent_name": name,
        "sub_mission_id": "sub-1",
        "sub_agent_number": 1,
        "goal": goal,
        "worker_role": "subagent",
        "execution_plane": plane,
        "status": "RUNNING",
    }


def _step(uid: str, summary: str) -> dict:
    return {
        "type": "action",
        "title": "Worker step",
        "sub_agent_uid": uid,
        "sub_agent_id": "sub-agent-1",
        "sub_mission_id": "sub-1",
        "action_type": "TERMINAL_EXEC",
        "target_resource": summary,
        "action_status": "SUCCESS",
        "step_index": 1,
    }


def _thought(uid: str, content: str) -> dict:
    return {
        "type": "thought",
        "title": "Worker reasoning",
        "sub_agent_uid": uid,
        "sub_agent_id": "sub-agent-1",
        "sub_mission_id": "sub-1",
        "content": content,
    }


# ==========================================================
# List endpoint shape
# ==========================================================


def test_list_endpoint_exposes_the_envelope_the_page_reads(client):
    """`getWorkstationAgents` reads `data.workers`; the page reads `is_running`."""
    token = _login(client)
    _seed_worklog(
        "default",
        "contract-list",
        [_dispatch("worker-1-aaa", "Route Mapper", "map the routes")],
    )
    res = client.get("/workstation/agents?session_id=contract-list", headers=_auth(token))
    assert res.status_code == 200, res.text
    body = res.json()

    # `data.workers || []` in agents/page.tsx
    assert isinstance(body["workers"], list) and len(body["workers"]) == 1
    # header badge: `${runningWorkers}/${workers.length} workers`
    assert "count" in body and body["count"] == 1
    assert "running_count" in body

    worker = body["workers"][0]
    # card render: agent_name / agent_id / agent_uid / plane / status / goal
    assert worker["agent_name"] == "Route Mapper"
    assert worker["agent_id"] == "sub-agent-1"
    assert worker["agent_uid"] == "worker-1-aaa"
    assert worker["plane"] == "headless"
    assert worker["goal"] == "map the routes"
    assert isinstance(worker["status"], str)
    # `workers.filter((w) => w.is_running)`
    assert "is_running" in worker


def test_list_endpoint_carries_live_step_activity_for_the_card(client):
    """The collapsed card shows `latest_action` before any drill-down."""
    token = _login(client)
    _seed_worklog(
        "default",
        "contract-list-activity",
        [
            _dispatch("worker-1-bbb", "Param Fuzzer", "fuzz the params"),
            _step("worker-1-bbb", "curl /admin?id=1"),
            _thought("worker-1-bbb", "Trying an integer overflow next."),
        ],
    )
    res = client.get(
        "/workstation/agents?session_id=contract-list-activity", headers=_auth(token)
    )
    worker = res.json()["workers"][0]
    # `Latest: {worker.latest_action}` on the collapsed card
    assert "curl /admin?id=1" in worker["latest_action"]
    # `worker.latest_thought` in the expanded panel
    assert "integer overflow" in worker["latest_thought"]
    # `{detail.step_count} steps`
    assert worker["step_count"] == 1
    # `(detail.steps || []).slice(-8)` renders these
    assert isinstance(worker["steps"], list) and worker["steps"]


# ==========================================================
# Detail endpoint shape
# ==========================================================


def test_detail_endpoint_returns_worker_under_the_key_the_ui_reads(client):
    """`getWorkstationAgent` reads `response.worker` (WorklogFeed + agents page)."""
    token = _login(client)
    _seed_worklog(
        "default",
        "contract-detail",
        [
            _dispatch("worker-7-zzz", "Builder", "write the module", plane="gui"),
            _step("worker-7-zzz", "write src/a.py"),
        ],
    )
    res = client.get(
        "/workstation/agents/worker-7-zzz?session_id=contract-detail", headers=_auth(token)
    )
    assert res.status_code == 200, res.text
    body = res.json()

    # `const record = response?.worker;` -- a missing key would silently render
    # "showing last known events" forever.
    assert "worker" in body
    worker = body["worker"]
    assert worker["agent_uid"] == "worker-7-zzz"
    assert worker["agent_name"] == "Builder"
    assert worker["plane"] == "gui"
    assert "write src/a.py" in worker["latest_action"]
    assert worker["step_count"] == 1


def test_detail_endpoint_404s_an_unknown_uid(client):
    """The UI treats a 404 as a fetch error, not an empty worker."""
    token = _login(client)
    _seed_worklog("default", "contract-404", [_dispatch("worker-1-aaa", "W", "g")])
    res = client.get(
        "/workstation/agents/not-a-worker?session_id=contract-404", headers=_auth(token)
    )
    assert res.status_code == 404


def test_detail_endpoint_is_tenant_scoped(client):
    """A worker from another tenant must not be reachable."""
    token_a = _login(client, tenant_id="tenant-a")
    token_b = _login(client, tenant_id="tenant-b")
    _seed_worklog("tenant-a", "contract-tenant", [_dispatch("worker-1-aaa", "W", "g")])

    ok = client.get(
        "/workstation/agents/worker-1-aaa?session_id=contract-tenant", headers=_auth(token_a)
    )
    assert ok.status_code == 200

    leaked = client.get(
        "/workstation/agents/worker-1-aaa?session_id=contract-tenant", headers=_auth(token_b)
    )
    assert leaked.status_code == 404, "cross-tenant worker lookup must not resolve"
