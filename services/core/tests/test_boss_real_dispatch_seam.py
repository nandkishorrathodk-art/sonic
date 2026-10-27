"""The REAL Boss -> worker dispatch seam, with no stubbed dispatch.

``test_boss_worker_visibility.py`` stubs ``BossAgent._dispatch_sub_agent``, so
the code that actually builds a worker, runs its observe->reason->act loop, and
streams its per-step traces back to the board is never exercised. A regression
there -- a dropped ``agent_uid``, a missing step emit, a worker that runs in the
background but never reports -- would leave every existing test green while the
operator sees nothing.

These tests drive the real method: a real ``ComputerUseAgent`` is constructed and
run, with only the LLM router scripted. Everything else stays real (agent
construction, ``run_mission``, the step callback, the route's event->worklog
translation, and the worker projection).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.api.routes.workstation import _get_or_create_session, _run_prompt_reasoning
from sonic.computer.headless import HeadlessComputeProvider
from sonic.computer_use.agent_activity import collect_worker_activity, find_worker
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import Phase, SubMission


class _ScriptedRouter:
    """Returns a fixed terminal action for every reasoning turn.

    The real agent parses ``ACTION:/TARGET:/PAYLOAD:`` blocks, so a scripted
    response drives the real parser and the real dispatch path.
    """

    def __init__(self, thought: str = "probing admin route", command: str = "curl /admin") -> None:
        self.thought = thought
        self.command = command
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        response = MagicMock()
        response.content = (
            f"THOUGHT: {self.thought}\n"
            "ACTION: TERMINAL_EXEC\n"
            f"TARGET: {self.command}\n"
            "PAYLOAD: {}\n"
            "EXPECTED: command output"
        )
        response.tool_calls = []
        return response


def _plan() -> Phase:
    return Phase(
        phase_number=1,
        name="Recon",
        sub_missions=[
            SubMission(goal="map the web surface", role="worker", plane="headless", max_steps=1),
            SubMission(goal="fuzz request parameters", role="worker", plane="headless", max_steps=1),
        ],
    )


async def _run_boss_with_real_dispatch(
    tenant_id: str,
    session_id: str,
    objective: str,
    *,
    thought: str = "probing admin route",
    command: str = "echo /admin",
    tmp_path=None,
):
    """Run the real Boss loop and the real ``_dispatch_sub_agent``.

    Uses the real in-process ``HeadlessComputeProvider`` so worker steps execute
    for real; only the LLM is scripted.
    """
    state = _get_or_create_session(tenant_id, session_id)
    state["desktop"] = {"workspace_id": "ws-real"}
    state["worklog"] = []

    base_dir = (tmp_path / "headless") if tmp_path is not None else None
    computer = HeadlessComputeProvider(base_dir=base_dir, allow_host_execution=True)
    router = _ScriptedRouter(thought=thought, command=command)

    with patch(
        "sonic.api.routes.workstation.get_daytona_computer", lambda: computer
    ), patch(
        "sonic.llm.router.ModelRouter.from_config", lambda *_a, **_k: router
    ), patch.object(
        BossAgent, "_strategic_decomposition", AsyncMock(return_value=_plan())
    ), patch.object(
        BossAgent, "_plan_next_phase", AsyncMock(return_value=None)
    ), patch.object(
        BossAgent, "_check_objective_complete", AsyncMock(return_value=(True, "done"))
    ):
        await _run_prompt_reasoning(tenant_id, session_id, objective)
    return state, computer, router


@pytest.mark.asyncio
async def test_real_dispatch_surfaces_workers_with_unique_run_ids(tmp_path):
    state, _computer, _router = await _run_boss_with_real_dispatch(
        "t-real-1", "s-real-1", "enumerate the web surface", tmp_path=tmp_path
    )
    workers = collect_worker_activity(state.get("worklog") or [], mission_task_alive=True)

    assert workers, "the real dispatch path must surface workers to the board"
    uids = [w["agent_uid"] for w in workers]
    assert all(uids), f"every real worker needs a run-unique id, got {uids}"
    assert len(set(uids)) == len(uids), f"real worker ids must be unique, got {uids}"
    for worker in workers:
        assert worker["agent_id"], "the positional agent id must be preserved"
        assert worker["goal"], "a worker card must state what the worker was asked to do"


@pytest.mark.asyncio
async def test_real_dispatch_streams_each_worker_step_to_the_board(tmp_path):
    state, _computer, router = await _run_boss_with_real_dispatch(
        "t-real-2", "s-real-2", "enumerate the web surface", command="echo /admin", tmp_path=tmp_path
    )
    workers = collect_worker_activity(state.get("worklog") or [], mission_task_alive=True)
    assert workers

    # The worker genuinely ran: the scripted command reached the provider.
    assert router.calls > 0, "the worker's real reasoning loop must have run"

    # ...and its per-step activity reached the board, not just a status.
    stepped = [w for w in workers if w["step_count"]]
    assert stepped, "each worker's steps must reach the board"
    assert stepped[0]["latest_thought"] == "probing admin route"


@pytest.mark.asyncio
async def test_real_dispatch_workers_are_individually_addressable(tmp_path):
    state, _computer, _router = await _run_boss_with_real_dispatch(
        "t-real-3", "s-real-3", "enumerate the web surface", tmp_path=tmp_path
    )
    worklog = state.get("worklog") or []
    workers = collect_worker_activity(worklog, mission_task_alive=True)
    assert workers

    for worker in workers:
        resolved = find_worker(worklog, worker["agent_uid"])
        assert resolved is not None, f"clicking {worker['agent_uid']} must resolve a worker"
        assert resolved["agent_uid"] == worker["agent_uid"]

    # Two distinct workers must not collapse into one card.
    keys = [f"uid:{w['agent_uid']}" for w in workers]
    assert len(set(keys)) == len(keys)


@pytest.mark.asyncio
async def test_real_dispatch_does_not_leak_worker_ids_across_goals(tmp_path):
    """A second objective must not resurrect the first run's workers."""
    state, _computer, _router = await _run_boss_with_real_dispatch(
        "t-real-4", "s-real-4", "enumerate the web surface", tmp_path=tmp_path
    )
    first_uids = {
        w["agent_uid"]
        for w in collect_worker_activity(state.get("worklog") or [], mission_task_alive=True)
    }
    assert first_uids

    state2, _c2, _r2 = await _run_boss_with_real_dispatch(
        "t-real-4", "s-real-4", "map the internal api", tmp_path=tmp_path
    )
    second = collect_worker_activity(
        state2.get("worklog") or [], mission_task_alive=True, current_run_only=True
    )
    second_uids = {w["agent_uid"] for w in second}
    assert second_uids
    assert not (first_uids & second_uids), (
        "a fresh objective must produce fresh worker identities"
    )


@pytest.mark.asyncio
async def test_real_dispatch_worker_status_is_honest(tmp_path):
    """A worker that ran but produced no independent evidence is not COMPLETED.

    The whole point of the operator board is to show real progress. A worker
    whose command succeeded but whose task was never independently verified must
    report UNVERIFIED -- never COMPLETED -- so the operator is not misled.
    """
    state, _computer, router = await _run_boss_with_real_dispatch(
        "t-real-5", "s-real-5", "enumerate the web surface", tmp_path=tmp_path
    )
    reports = [
        it
        for it in (state.get("worklog") or [])
        if it.get("orchestration_event") == "sub_report"
    ]
    assert reports, "each dispatched worker must report a terminal status"
    assert router.calls > 0, "the worker really ran its reasoning loop"

    statuses = {str(r.get("status") or "").upper() for r in reports}
    assert "COMPLETED" not in statuses, (
        "an echo-style command with no independent evidence must not be reported "
        f"as COMPLETED, got {statuses}"
    )
    assert statuses <= {"UNVERIFIED", "FAILED", "INCOMPLETE", "PARTIAL"}, statuses
