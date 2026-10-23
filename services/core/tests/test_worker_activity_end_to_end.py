"""End-to-end test: real Boss phase events -> workstation worklog -> worker projection.

This closes a real gap. The pure projection
(``agent_activity.collect_worker_activity``) and the Boss dispatch path are each
tested in isolation, but the *seam* between them was not:

* the existing integration test emits ``sub_dispatch`` **without** ``agent_uid``,
  so the projection keyed off the positional ``agent_id``;
* production emits ``agent_uid`` (``worker-<n>-<hex>``), which is the only
  run-unique handle the operator UI can address a specific worker by.

A field-name mismatch across that seam would therefore pass every existing test
while the operator dashboard showed nothing. These tests drive the real
``_run_prompt_reasoning`` route with a fake Boss that emits the *production*
event shape, then assert the projection the UI actually reads.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.api.routes.workstation import (
    _get_or_create_session,
    _run_prompt_reasoning,
)
from sonic.computer_use.agent_activity import collect_worker_activity, find_worker
from sonic.computer_use.models import ComputerActionType, ComputerDecisionTrace
from sonic.computer_use.models_boss import BossReport, BossThinking, Phase, SubMission


def _make_mock_computer() -> MagicMock:
    mock_screen = MagicMock(
        desktop_state="LIVE", width=1280, height=800, visible_text="", detected_controls=[]
    )
    computer = MagicMock()
    computer.screenshot = AsyncMock(return_value=mock_screen)
    computer.terminal = AsyncMock(return_value=MagicMock(stdout="/root"))
    return computer


def _boss_report() -> BossReport:
    return BossReport(
        objective="enumerate the target's web surface",
        status="COMPLETE",
        phases=[Phase(phase_number=1, name="Recon", sub_missions=[SubMission(goal="map routes")])],
        thinking_log=[BossThinking(phase=1, thinking_type="strategic_decomposition", content="planning")],
        all_traces=[
            ComputerDecisionTrace(
                action_type=ComputerActionType.TERMINAL_EXEC,
                target_resource="curl target.test",
                predicted_outcome="response",
                actual_observation="HTTP 200",
                status="SUCCESS",
            )
        ],
        total_sub_agents=2,
        total_actions=2,
        total_phases=1,
        findings_summary="mapped the surface",
    )


async def _drive_route_with_production_events(monkeypatch, tenant_id: str, session_id: str):
    """Run the real route; a fake Boss emits the same event shape as production."""
    state = _get_or_create_session(tenant_id, session_id)
    state["desktop"] = {"workspace_id": "ws-activity-e2e"}
    state["worklog"] = []

    monkeypatch.setattr(
        "sonic.api.routes.workstation.get_daytona_computer", lambda: _make_mock_computer()
    )

    with patch("sonic.computer_use.boss.BossAgent") as mock_boss_cls:
        mock_boss = MagicMock()

        async def fake_run(workspace_id, objective, phase_callback=None, interrupt_check=None, requested_plane=None):
            if phase_callback:
                # Two workers, exactly the production payload shape: run-unique
                # agent_uid + positional agent_id + sub_agent_number.
                await phase_callback("sub_dispatch", {
                    "sub_agent_number": 1,
                    "sub_mission_id": "sub-map-routes",
                    "agent_id": "sub-agent-1",
                    "agent_uid": "worker-1-aaa111",
                    "agent_name": "Route Mapper",
                    "status": "RUNNING",
                    "goal": "map routes",
                    "role": "terminal_auditor",
                    "plane": "headless",
                    "max_steps": 5,
                })
                await phase_callback("sub_step", {
                    "trace": {
                        "step_index": 1,
                        "action_type": "TERMINAL_EXEC",
                        "target": "curl /admin",
                        "observation": "HTTP 403",
                        "status": "SUCCESS",
                        "exit_code": 0,
                    },
                    "agent_id": "sub-agent-1",
                    "agent_uid": "worker-1-aaa111",
                    "agent_name": "Route Mapper",
                    "sub_mission_id": "sub-map-routes",
                    "role": "terminal_auditor",
                    "plane": "headless",
                })
                await phase_callback("sub_report", {
                    "sub_agent_number": 1,
                    "sub_mission_id": "sub-map-routes",
                    "agent_id": "sub-agent-1",
                    "agent_uid": "worker-1-aaa111",
                    "agent_name": "Route Mapper",
                    "status": "SUCCESS",
                    "success": True,
                    "goal": "map routes",
                    "role": "terminal_auditor",
                    "plane": "headless",
                    "findings_summary": "12 routes",
                })
                # Second worker dispatches but never reports (interrupted).
                await phase_callback("sub_dispatch", {
                    "sub_agent_number": 2,
                    "sub_mission_id": "sub-fuzz-params",
                    "agent_id": "sub-agent-2",
                    "agent_uid": "worker-2-bbb222",
                    "agent_name": "Param Fuzzer",
                    "status": "RUNNING",
                    "goal": "fuzz params",
                    "role": "terminal_auditor",
                    "plane": "headless",
                    "max_steps": 5,
                })
            return _boss_report()

        mock_boss.run = AsyncMock(side_effect=fake_run)
        mock_boss_cls.return_value = mock_boss

        await _run_prompt_reasoning(tenant_id, session_id, "enumerate the target's web surface")

    return state


@pytest.mark.asyncio
async def test_projection_resolves_both_workers_by_run_unique_uid(monkeypatch):
    """The projection must key off agent_uid, so both workers are distinct."""
    state = await _drive_route_with_production_events(
        monkeypatch, "tenant-act-e2e", "sess-act-e2e-1"
    )
    workers = collect_worker_activity(state["worklog"], mission_task_alive=False)

    uids = {w["agent_uid"] for w in workers}
    assert uids == {"worker-1-aaa111", "worker-2-bbb222"}, (
        f"expected two run-unique workers, got {uids}"
    )
    # Distinct workers must not collapse onto the same key.
    assert len(workers) == 2


@pytest.mark.asyncio
async def test_finished_worker_is_completed_and_unreported_worker_is_interrupted(monkeypatch):
    """A worker with a terminal report is COMPLETED; one without is INTERRUPTED."""
    state = await _drive_route_with_production_events(
        monkeypatch, "tenant-act-e2e", "sess-act-e2e-2"
    )
    workers = {
        w["agent_uid"]: w
        for w in collect_worker_activity(state["worklog"], mission_task_alive=False)
    }
    assert workers["worker-1-aaa111"]["status"] in ("COMPLETED", "SUCCESS")
    assert workers["worker-1-aaa111"]["is_running"] is False
    # Never a stale RUNNING once the owning task is gone.
    assert workers["worker-2-bbb222"]["status"] == "INTERRUPTED"
    assert workers["worker-2-bbb222"]["is_running"] is False


@pytest.mark.asyncio
async def test_operator_can_drill_into_a_worker_by_uid(monkeypatch):
    """find_worker must resolve the uid the UI drills into, and show its activity."""
    state = await _drive_route_with_production_events(
        monkeypatch, "tenant-act-e2e", "sess-act-e2e-3"
    )
    worklog = state["worklog"]

    worker = find_worker(worklog, "worker-1-aaa111", mission_task_alive=False)
    assert worker is not None, "operator could not resolve a real worker uid"
    assert worker["agent_name"] == "Route Mapper"
    assert worker["step_count"] == 1
    assert "curl /admin" in worker["latest_action"]

    assert find_worker(worklog, "worker-2-bbb222", mission_task_alive=False) is not None
    assert find_worker(worklog, "no-such-worker", mission_task_alive=False) is None


@pytest.mark.asyncio
async def test_worker_running_only_while_mission_task_alive(monkeypatch):
    """The same worklog flips an unreported worker to RUNNING while the task lives."""
    state = await _drive_route_with_production_events(
        monkeypatch, "tenant-act-e2e", "sess-act-e2e-4"
    )
    alive = {
        w["agent_uid"]: w
        for w in collect_worker_activity(state["worklog"], mission_task_alive=True)
    }
    # Completed worker stays terminal even while the task is alive.
    assert alive["worker-1-aaa111"]["status"] in ("COMPLETED", "SUCCESS")
    assert alive["worker-1-aaa111"]["is_running"] is False
    # Unreported worker is RUNNING only because the task is still alive.
    assert alive["worker-2-bbb222"]["status"] == "RUNNING"
    assert alive["worker-2-bbb222"]["is_running"] is True
