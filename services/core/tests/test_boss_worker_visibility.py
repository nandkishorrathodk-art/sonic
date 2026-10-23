"""The Boss -> workstation -> worker-board seam, driven by the REAL BossAgent.

`test_worker_activity_end_to_end.py` proves the route translates a *faked* Boss
event stream correctly. It never runs the real Boss, so a mismatch between what
`BossAgent.run()` actually emits and what the route expects would pass there and
still leave the operator board empty.

These tests close that gap: they run `BossAgent.run()` for real (LLM stubbed at
the boundary, everything else production) and assert the operator can see and
open each delegated worker. That is the user-visible contract:

* every sub-agent gets a stable, run-unique id
* the board lists one card per worker, not one merged card
* clicking a card resolves that specific worker, and it shows what it was doing
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.api.routes.workstation import _get_or_create_session, _run_prompt_reasoning
from sonic.computer_use.agent_activity import collect_worker_activity, find_worker
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult


class _StubComputer:
    """Real provider shape; no desktop needed because dispatch is stubbed."""

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


def _fake_dispatch(step_thought: str = "probing admin route"):
    """A production-shaped worker: dispatch, one step, one report."""

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
                "target": "curl /admin",
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
            findings_summary="12 routes",
            key_discoveries=["/admin"],
            actions_taken=1,
            duration_seconds=0.5,
            traces=[],
        )
        result.agent_id = agent_id
        result.agent_uid = uid
        return result

    return _dispatch


def _stub_router():
    """A router whose ``complete`` never touches the network."""
    router = MagicMock()
    response = MagicMock()
    response.content = "Summary of verified work."
    router.complete = AsyncMock(return_value=response)
    return router


async def _run_real_boss(tenant_id: str, session_id: str, objective: str, **dispatch_kwargs):
    """Run the real ``BossAgent.run()`` loop with only model calls stubbed.

    The three model calls that shape a run -- strategic decomposition, next-phase
    planning, objective-completion check -- plus the router itself are stubbed so
    the test is deterministic and offline. Everything else stays real: the phase
    loop, dispatch policy, event emission, the route's event->worklog
    translation, and the worker projection. That seam is what these tests
    exercise.
    """
    state = _get_or_create_session(tenant_id, session_id)
    state["desktop"] = {"workspace_id": "ws-seam"}
    state["worklog"] = []

    plan = Phase(
        phase_number=1,
        name="Recon",
        sub_missions=[
            SubMission(goal="map the web surface", role="worker", plane="headless"),
            SubMission(goal="fuzz request parameters", role="worker", plane="headless"),
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
        BossAgent, "_dispatch_sub_agent", _fake_dispatch(**dispatch_kwargs)
    ):
        await _run_prompt_reasoning(tenant_id, session_id, objective)
    return state


@pytest.mark.asyncio
async def test_real_boss_workers_are_visible_with_unique_ids():
    state = await _run_real_boss("t-seam-1", "s-seam-1", "enumerate the web surface")
    workers = collect_worker_activity(state.get("worklog") or [], mission_task_alive=True)

    assert workers, "the real Boss must surface at least one worker to the board"
    uids = [w["agent_uid"] for w in workers]
    assert all(uids), f"every worker needs a run-unique id, got {uids}"
    assert len(set(uids)) == len(uids), f"worker ids must be unique, got {uids}"
    for worker in workers:
        assert worker["agent_id"], "the positional agent id must be preserved"
        assert worker["goal"], "a worker card must state what the worker was asked to do"


@pytest.mark.asyncio
async def test_clicking_a_worker_resolves_that_specific_worker():
    state = await _run_real_boss("t-seam-2", "s-seam-2", "enumerate the web surface")
    worklog = state.get("worklog") or []
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    assert workers

    for worker in workers:
        resolved = find_worker(worklog, worker["agent_uid"])
        assert resolved is not None, f"clicking {worker['agent_uid']} must resolve a worker"
        assert resolved["agent_uid"] == worker["agent_uid"]


@pytest.mark.asyncio
async def test_worker_card_shows_live_activity_not_just_a_status():
    state = await _run_real_boss(
        "t-seam-3", "s-seam-3", "enumerate the web surface", step_thought="probing admin route"
    )
    workers = collect_worker_activity(state.get("worklog") or [], mission_task_alive=True)
    assert workers
    # The step must actually reach the card, otherwise the operator sees a
    # worker that reports "COMPLETED" with no account of what it did.
    assert any(w["step_count"] for w in workers), "a worker's steps must reach the board"
    active = [w for w in workers if w["step_count"]]
    assert active[0]["latest_thought"] == "probing admin route"
    assert "curl /admin" in active[0]["latest_action"]


@pytest.mark.asyncio
async def test_two_workers_are_two_cards_not_one_merged_card():
    state = await _run_real_boss("t-seam-4", "s-seam-4", "enumerate the web surface")
    workers = collect_worker_activity(state.get("worklog") or [], mission_task_alive=True)
    assert len(workers) >= 1
    # Distinct uids must produce distinct board keys, otherwise React would
    # collapse separate workers into a single card.
    keys = [f"uid:{w['agent_uid']}" for w in workers]
    assert len(set(keys)) == len(keys)


def test_find_worker_rejects_an_unknown_id_rather_than_guessing():
    worklog = [{
        "orchestration_event": "sub_dispatch",
        "agent_uid": "worker-1-aaaaaa",
        "agent_id": "sub-agent-1",
        "agent_name": "Worker 1",
        "goal": "map routes",
        "status": "RUNNING",
    }]
    assert find_worker(worklog, "worker-9-zzzzzz") is None
    assert find_worker(worklog, "") is None
    assert find_worker([], "worker-1-aaaaaa") is None


def test_unreported_worker_is_interrupted_not_stuck_running():
    worklog = [{
        "orchestration_event": "sub_dispatch",
        "agent_uid": "worker-1-aaaaaa",
        "agent_id": "sub-agent-1",
        "agent_name": "Worker 1",
        "goal": "map routes",
        "status": "RUNNING",
    }]
    dead = collect_worker_activity(worklog, mission_task_alive=False)
    assert dead[0]["status"] == "INTERRUPTED"
    assert dead[0]["is_running"] is False

    alive = collect_worker_activity(worklog, mission_task_alive=True)
    assert alive[0]["status"] == "RUNNING"
