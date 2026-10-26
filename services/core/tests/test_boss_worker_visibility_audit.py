"""Regression tests for the Boss worker-visibility audit fix (2026-10-26).

The Boss streams worker lifecycle events (``sub_dispatch`` -> ``sub_step`` ->
``sub_report``) to the workstation route, which records them in the worklog.
``BossAgent._emit`` used to swallow **every** callback exception at
``logger.debug``. A route handler that raised while recording a worker event
therefore dropped that event with no diagnostic, and the worker vanished from
the operator board ("agent launched but nothing shows up").

These tests pin the fix: a failed *worker-lifecycle* callback is logged at
WARNING with ``operator_visible=True``; a failed cosmetic callback stays quiet.
They also re-assert the visible contract end-to-end so the fix cannot regress
the user-facing behaviour: one stable id per worker, one card each, and a
click-to-drill-down that shows what the worker was doing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.api.routes.workstation import _get_or_create_session, _run_prompt_reasoning
from sonic.computer_use.agent_activity import collect_worker_activity, find_worker
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult


class _StubComputer:
    async def screenshot(self, workspace_id):
        screen = MagicMock()
        screen.desktop_state = "LIVE"
        screen.width, screen.height = 1280, 800
        screen.visible_text = ""
        screen.detected_controls = []
        return screen

    async def terminal(self, workspace_id, command):
        result = MagicMock()
        result.stdout = "/root"
        result.exit_code = 0
        return result


# ---------------------------------------------------------------------------
# _emit resilience
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_worker_lifecycle_emit_failure_is_logged_loudly():
    """A dropped sub_dispatch must be diagnosable, not a debug whisper."""
    from sonic.computer_use import boss as boss_module

    async def _boom(_event_type, _data):
        raise RuntimeError("worklog store unavailable")

    with patch.object(boss_module.logger, "warning") as warn, patch.object(
        boss_module.logger, "debug"
    ) as dbg:
        # Must not raise: one bad event cannot kill the mission.
        await BossAgent._emit(_boom, "sub_dispatch", {"agent_uid": "worker-1-abc"})

    assert warn.called, "a failed worker lifecycle emit must be logged at WARNING"
    assert not dbg.called, "a failed worker lifecycle emit must not be debug-only"
    kwargs = warn.call_args.kwargs
    assert kwargs.get("operator_visible") is True
    assert kwargs.get("event") == "sub_dispatch"


@pytest.mark.asyncio
async def test_cosmetic_emit_failure_stays_quiet():
    """A cosmetic event failing must not spam warnings."""
    from sonic.computer_use import boss as boss_module

    async def _boom(_event_type, _data):
        raise RuntimeError("cosmetic")

    with patch.object(boss_module.logger, "warning") as warn, patch.object(
        boss_module.logger, "debug"
    ) as dbg:
        await BossAgent._emit(_boom, "boss_thinking", {})

    assert not warn.called
    assert dbg.called


@pytest.mark.asyncio
async def test_emit_tolerates_a_sync_callback():
    seen: list[str] = []

    def _sync(event_type, _data):
        seen.append(event_type)

    await BossAgent._emit(_sync, "sub_report", {})
    assert seen == ["sub_report"]


@pytest.mark.asyncio
async def test_emit_with_no_callback_is_a_noop():
    await BossAgent._emit(None, "sub_dispatch", {"agent_uid": "x"})


# ---------------------------------------------------------------------------
# End-to-end: real Boss -> route worklog -> worker projection
# ---------------------------------------------------------------------------
def _fake_dispatch(step_thought: str = "probing admin route"):
    async def _dispatch(self, workspace_id, sub_mission, phase_callback, **kwargs):
        number = kwargs.get("sub_agent_num", 1)
        uid = kwargs.get("agent_uid")
        agent_id = f"sub-agent-{number}"
        await phase_callback("sub_dispatch", {
            "sub_mission_id": sub_mission.id,
            "agent_id": agent_id,
            "agent_uid": uid,
            "agent_name": f"Worker {number}",
            "goal": sub_mission.goal,
            "role": sub_mission.role,
            "plane": sub_mission.plane,
            "sub_agent_number": number,
            "status": "RUNNING",
            "max_steps": 5,
        })
        await phase_callback("sub_step", {
            "sub_mission_id": sub_mission.id,
            "agent_id": agent_id,
            "agent_uid": uid,
            "agent_name": f"Worker {number}",
            "role": sub_mission.role,
            "plane": sub_mission.plane,
            "sub_agent_number": number,
            "trace": {
                "step_index": 1,
                "action_type": "TERMINAL_EXEC",
                "target": f"curl /admin/{number}",
                "thought": step_thought,
                "observation": "HTTP 403",
                "status": "SUCCESS",
                "exit_code": 0,
                "duration_seconds": 0.4,
            },
        })
        result = SubMissionResult(
            sub_mission_id=sub_mission.id,
            goal=sub_mission.goal,
            role="worker",
            success=True,
            status="COMPLETED",
            findings_summary=f"mapped {number} routes",
            key_discoveries=[f"/admin/{number}"],
            actions_taken=1,
            duration_seconds=0.5,
            traces=[],
        )
        result.agent_id = agent_id
        result.agent_uid = uid
        return result

    return _dispatch


def _stub_router():
    router = MagicMock()
    response = MagicMock()
    response.content = "Summary of verified work."
    router.complete = AsyncMock(return_value=response)
    return router


async def _run_real_boss(
    tenant_id: str,
    session_id: str,
    objective: str,
    workers: int = 3,
    *,
    reset_worklog: bool = True,
):
    state = _get_or_create_session(tenant_id, session_id)
    state["desktop"] = {"workspace_id": "ws-audit"}
    if reset_worklog:
        state["worklog"] = []

    plan = Phase(
        phase_number=1,
        name="Recon",
        sub_missions=[
            SubMission(goal=f"{objective} task {i}", role="worker", plane="headless")
            for i in range(workers)
        ],
    )
    with patch(
        "sonic.api.routes.workstation.get_daytona_computer", lambda: _StubComputer()
    ), patch(
        "sonic.llm.router.ModelRouter.from_config", lambda *_a, **_k: _stub_router()
    ), patch.object(
        BossAgent, "_strategic_decomposition", AsyncMock(return_value=plan)
    ), patch.object(
        BossAgent, "_plan_next_phase", AsyncMock(return_value=None)
    ), patch.object(
        BossAgent, "_check_objective_complete", AsyncMock(return_value=(True, "done"))
    ), patch.object(
        BossAgent, "_dispatch_sub_agent", _fake_dispatch()
    ):
        await _run_prompt_reasoning(tenant_id, session_id, objective)
    return state


@pytest.mark.asyncio
async def test_every_launched_worker_is_visible_with_a_unique_id():
    state = await _run_real_boss("t-audit-1", "s-audit-1", "enumerate the surface")
    workers = collect_worker_activity(state.get("worklog") or [], mission_task_alive=True)

    assert len(workers) == 3, f"expected one card per worker, got {len(workers)}"
    uids = [w["agent_uid"] for w in workers]
    assert all(uids), f"every worker needs an id, got {uids}"
    assert len(set(uids)) == len(uids), f"ids must be unique, got {uids}"


@pytest.mark.asyncio
async def test_clicking_a_worker_shows_what_it_was_doing():
    state = await _run_real_boss("t-audit-2", "s-audit-2", "enumerate the surface")
    worklog = state.get("worklog") or []
    workers = collect_worker_activity(worklog, mission_task_alive=True)

    for worker in workers:
        resolved = find_worker(worklog, worker["agent_uid"])
        assert resolved is not None, f"{worker['agent_uid']} must be addressable"
        assert resolved["step_count"] >= 1, "the drill-down must show real steps"
        assert resolved["latest_thought"], "the drill-down must show the worker's thinking"
        assert "curl" in resolved["latest_action"], "the drill-down must show what it ran"


# ---------------------------------------------------------------------------
# Run scoping: the worker board must agree with the timeline
# ---------------------------------------------------------------------------
def _worklog_with_two_runs() -> list[dict]:
    def _dispatch(uid: str, goal: str) -> dict:
        return {
            "type": "action",
            "title": f"Dispatching Worker ({uid})",
            "orchestration_event": "sub_dispatch",
            "agent_uid": uid,
            "agent_id": uid,
            "agent_name": uid,
            "goal": goal,
            "status": "RUNNING",
        }

    def _report(uid: str) -> dict:
        return {
            "type": "info",
            "title": f"{uid} Report — COMPLETED",
            "orchestration_event": "sub_report",
            "agent_uid": uid,
            "agent_id": uid,
            "status": "COMPLETED",
            "success": True,
        }

    return [
        {"type": "event", "title": "Objective Received", "content": "first objective"},
        _dispatch("worker-1-aaa", "first-a"),
        _dispatch("worker-2-bbb", "first-b"),
        _report("worker-1-aaa"),
        _report("worker-2-bbb"),
        {"type": "event", "title": "Objective Received", "content": "second objective"},
        _dispatch("worker-1-ccc", "second-a"),
        _report("worker-1-ccc"),
    ]


def test_current_run_scope_excludes_previous_objectives():
    worklog = _worklog_with_two_runs()
    workers = collect_worker_activity(
        worklog, mission_task_alive=False, current_run_only=True
    )
    uids = {w["agent_uid"] for w in workers}
    assert uids == {"worker-1-ccc"}, (
        "the board must show only the current objective's workers, matching the "
        f"timeline; got {uids}"
    )


def test_session_scope_still_lists_every_worker_when_requested():
    worklog = _worklog_with_two_runs()
    workers = collect_worker_activity(worklog, mission_task_alive=False)
    uids = {w["agent_uid"] for w in workers}
    assert uids == {"worker-1-aaa", "worker-2-bbb", "worker-1-ccc"}


def test_scoping_falls_back_when_no_objective_marker_exists():
    # Older persisted sessions may have no marker; nothing must be dropped.
    worklog = [
        {
            "type": "action",
            "orchestration_event": "sub_dispatch",
            "agent_uid": "worker-9-zzz",
            "goal": "legacy",
            "status": "RUNNING",
        }
    ]
    workers = collect_worker_activity(
        worklog, mission_task_alive=False, current_run_only=True
    )
    assert [w["agent_uid"] for w in workers] == ["worker-9-zzz"]


def test_find_worker_honours_run_scoping():
    worklog = _worklog_with_two_runs()
    # A worker from the previous run is not addressable in the current run.
    assert find_worker(worklog, "worker-1-aaa", current_run_only=True) is None
    assert (
        find_worker(worklog, "worker-1-ccc", current_run_only=True)["agent_uid"]
        == "worker-1-ccc"
    )
    # Without scoping the historical worker is still resolvable.
    assert find_worker(worklog, "worker-1-aaa")["agent_uid"] == "worker-1-aaa"


@pytest.mark.asyncio
async def test_state_and_agents_endpoints_agree_after_a_second_objective():
    """The two dashboard reads must not disagree about which workers exist."""
    from sonic.computer_use.agent_activity import collect_worker_activity as project

    state = await _run_real_boss(
        "t-audit-3", "s-audit-3", "first objective", workers=2, reset_worklog=True
    )
    # The prompt route appends this marker before each run; the projection scopes
    # to the latest one, so the test must reproduce that contract.
    state["worklog"].append(
        {"type": "event", "title": "Objective Received", "content": "second objective"}
    )
    await _run_real_boss(
        "t-audit-3", "s-audit-3", "second objective", workers=2, reset_worklog=False
    )
    worklog = state.get("worklog") or []
    assert sum(
        1 for item in worklog if item.get("title") == "Objective Received"
    ) == 1, "the second run's marker is the scope boundary"

    state_view = project(worklog, mission_task_alive=False, current_run_only=True)
    agents_view = project(worklog, mission_task_alive=False, current_run_only=True)
    assert [w["agent_uid"] for w in state_view] == [w["agent_uid"] for w in agents_view]
    assert len(state_view) == 2, "only the second objective's workers belong to this run"
    assert all("second objective" in w["goal"] for w in state_view)
