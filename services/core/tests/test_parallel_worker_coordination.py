"""Parallel-worker coordination gaps: scope binding, steering, structured evidence.

Covers three real gaps in the "parallel agents" workflow:

1. Scope binding — a worker was never told which paths it owns, so the wave
   path-lock in the Boss had nothing to protect.
2. Mid-run steering — the operator could not message a specific running worker
   (the ``conversation_id`` teammate inbox is agent-authored, not operator).
3. Structured evidence — a worker's result carried only free text, so a
   confident but unbacked summary looked identical to real file/test evidence.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.computer_use.boss import BossAgent
from sonic.computer_use.boss import _available_worker_skill_names
from sonic.computer_use.coordinator_inbox import (
    CoordinatorInbox,
    get_coordinator_inbox,
    reset_coordinator_inbox,
)
from sonic.agents.judge import JudgeEvaluation, JudgeVerdict
from sonic.computer_use.models import ComputerActionType, ComputerDecisionTrace
from sonic.computer_use.models_boss import SubMission


# ---------------------------------------------------------------------------
# GAP 1 — owned_paths scope
# ---------------------------------------------------------------------------


def test_submission_owned_paths_default_empty_and_roundtrip():
    sub = SubMission(goal="edit the parser")
    assert sub.owned_paths == []
    scoped = SubMission(goal="edit the parser", owned_paths=["src/parser.py"])
    assert scoped.owned_paths == ["src/parser.py"]


@pytest.mark.asyncio
async def test_decomposition_parses_owned_paths(monkeypatch):
    boss = BossAgent(computer_provider=MagicMock(), llm_router=MagicMock())
    payload = (
        '{"domain":"d","thinking":"t","phase_name":"p","sub_missions":['
        '{"id":"sub-1","name":"A","role":"builder","goal":"edit a",'
        '"owned_paths":["src/a.py"," src/b.py "],"plane":"headless"},'
        '{"id":"sub-2","name":"B","role":"builder","goal":"edit b",'
        '"owned_paths":["src/c.py"],"plane":"headless"}]}'
    )

    async def fake_complete(_req):
        return MagicMock(content=payload)

    boss.llm_router.complete = fake_complete
    phase = await boss._strategic_decomposition("refactor the modules")
    assert phase is not None
    by_id = {s.id: s for s in phase.sub_missions}
    # Whitespace trimmed, no duplicates, order preserved.
    assert by_id["sub-1"].owned_paths == ["src/a.py", "src/b.py"]
    assert by_id["sub-2"].owned_paths == ["src/c.py"]


@pytest.mark.asyncio
async def test_decomposition_keeps_only_available_catalog_skills():
    available = _available_worker_skill_names()
    assert available
    boss = BossAgent(computer_provider=MagicMock(), llm_router=MagicMock())
    payload = json.dumps({
        "thinking": "Assign work from the current capability catalog.",
        "phase_name": "Profiled worker",
        "sub_missions": [{
            "id": "sub-profile",
            "name": "Profiled worker",
            "role": "Document operator",
            "skills": [available[0], "not-a-catalog-skill", available[0]],
            "goal": "Complete the assigned outcome",
            "plane": "headless",
        }],
    })
    boss.llm_router.complete = AsyncMock(return_value=MagicMock(content=payload))

    phase = await boss._strategic_decomposition("Complete the assigned outcome")

    assert phase is not None
    assert phase.sub_missions[0].role == "Document operator"
    assert phase.sub_missions[0].skills == [available[0]]


@pytest.mark.asyncio
async def test_replanned_workers_keep_role_skills_and_owned_paths():
    available = _available_worker_skill_names()
    assert available
    boss = BossAgent(computer_provider=MagicMock(), llm_router=MagicMock())
    payload = json.dumps({
        "thinking": "Continue with the assigned engineering role.",
        "phase_name": "Implementation",
        "sub_missions": [{
            "id": "sub-next",
            "name": "Scoped implementer",
            "role": "Implementation specialist",
            "skills": [available[0]],
            "goal": "Implement the remaining task",
            "owned_paths": ["src/component.py"],
            "plane": "headless",
            "depends_on": [],
            "max_steps": 8,
        }],
    })
    boss.llm_router.complete = AsyncMock(return_value=MagicMock(content=payload))

    phase = await boss._plan_next_phase(
        "Complete the implementation",
        "The design is ready.",
        [],
        requested_plane="dual",
    )

    assert phase is not None
    worker = phase.sub_missions[0]
    assert worker.role == "Implementation specialist"
    assert worker.skills == [available[0]]
    assert worker.owned_paths == ["src/component.py"]


@pytest.mark.asyncio
async def test_boss_dispatch_passes_role_and_skills_to_agent_context():
    available = _available_worker_skill_names()
    assert available
    boss = BossAgent(computer_provider=MagicMock(), llm_router=MagicMock())
    boss.judge.evaluate_sub_mission = AsyncMock(return_value=JudgeEvaluation(
        verdict=JudgeVerdict.CONFIRMED,
        confidence=0.99,
        notes="Test fixture confirms the worker receipt.",
    ))
    worker_trace = SimpleNamespace(
        action_type=ComputerActionType.TERMINAL_EXEC,
        target_resource="inspect workspace",
        actual_observation="Workspace inspected.",
        status="COMPLETED",
        exit_code=0,
        verification_evidence="Workspace state checked.",
    )
    agent = MagicMock()
    agent.run_mission = AsyncMock(return_value=[worker_trace])
    agent.mission_summary = "Workspace inspection completed."

    from unittest.mock import patch

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=agent) as agent_class:
        await boss._dispatch_sub_agent(
            "workspace",
            SubMission(
                id="profiled-worker",
                goal="Inspect the workspace",
                role="Workspace analyst",
                skills=[available[0]],
                plane="headless",
            ),
            sub_agent_num=1,
        )

    context = agent_class.call_args.kwargs["initial_context"]
    assert context["subagent_role"] == "Workspace analyst"
    assert context["delegated_skills"] == [available[0]]


def test_worker_prompt_renders_owned_scope():
    from sonic.computer_use.agent import ComputerUseAgent
    from sonic.computer_use.skill_catalog import load_skill_catalog

    skill = load_skill_catalog().by_status("available")[0].name

    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=None,
        initial_context={
            "subagent_role": "builder",
            "delegated_skills": [skill],
            "owned_paths": ["src/a.py", "tests/test_a.py"],
        },
    )
    agent._last_screenshot_b64 = ""
    ctx = agent._build_reasoning_context("edit a", MagicMock(), 1, "", "")[1]
    assert "ASSIGNED WRITE SCOPE" in ctx
    assert "src/a.py" in ctx
    assert "tests/test_a.py" in ctx
    assert skill in ctx
    assert "they grant no permissions" in ctx


def test_worker_prompt_omits_scope_block_without_owned_paths():
    from sonic.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=None,
        initial_context={"subagent_role": "builder"},
    )
    agent._last_screenshot_b64 = ""
    ctx = agent._build_reasoning_context("edit a", MagicMock(), 1, "", "")[1]
    assert "ASSIGNED WRITE SCOPE" not in ctx


# ---------------------------------------------------------------------------
# GAP 2 — coordinator inbox / mid-run steering
# ---------------------------------------------------------------------------


def test_inbox_post_and_drain_roundtrip():
    inbox = CoordinatorInbox()
    inbox.post("worker-1-aaa", "stop touching file X")
    assert inbox.pending_count("worker-1-aaa") == 1
    drained = inbox.drain("worker-1-aaa")
    assert drained[0]["message"] == "stop touching file X"
    # Drain is destructive: a message is delivered once, not replayed forever.
    assert inbox.drain("worker-1-aaa") == []
    assert inbox.pending_count("worker-1-aaa") == 0


def test_inbox_rejects_empty_message_and_missing_uid():
    inbox = CoordinatorInbox()
    with pytest.raises(ValueError):
        inbox.post("", "hello")
    with pytest.raises(ValueError):
        inbox.post("worker-1-aaa", "   ")


def test_inbox_is_bounded_per_worker():
    inbox = CoordinatorInbox()
    for i in range(50):
        inbox.post("worker-1-aaa", f"msg {i}")
    drained = inbox.drain("worker-1-aaa")
    assert len(drained) == 20  # _MAX_PENDING
    assert drained[-1]["message"] == "msg 49"


def test_agent_renders_and_drains_operator_messages():
    from sonic.computer_use.agent import ComputerUseAgent

    reset_coordinator_inbox()
    get_coordinator_inbox().post("worker-9-zzz", "also check the edge case in test Y")
    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=None,
        initial_context={},
        agent_uid="worker-9-zzz",
    )
    agent._last_screenshot_b64 = ""
    # Assert the WIRING: the directive must appear in the real reasoning
    # prompt the model receives, not merely in a helper we could call directly.
    ctx = agent._build_reasoning_context("do the task", MagicMock(), 1, "", "")[1]
    assert "OPERATOR INSTRUCTIONS" in ctx
    assert "edge case in test Y" in ctx
    # Second render is empty: the directive is delivered exactly once.
    assert agent._render_operator_messages() == ""
    reset_coordinator_inbox()


def test_agent_without_uid_never_renders_messages():
    from sonic.computer_use.agent import ComputerUseAgent

    reset_coordinator_inbox()
    get_coordinator_inbox().post("worker-9-zzz", "hello")
    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=None,
        initial_context={},
    )
    assert agent._render_operator_messages() == ""
    reset_coordinator_inbox()


# ---------------------------------------------------------------------------
# GAP 3 — structured evidence extraction
# ---------------------------------------------------------------------------


def _trace(action, target, obs, status="COMPLETED", exit_code=0):
    return ComputerDecisionTrace(
        action_type=action,
        target_resource=target,
        predicted_outcome="x",
        actual_observation=obs,
        status=status,
        exit_code=exit_code,
    )


def test_file_references_derived_from_real_traces_only():
    traces = [
        _trace(ComputerActionType.FILE_WRITE, "src/a.py", "wrote 12 lines"),
        _trace(ComputerActionType.FILE_READ, "src/a.py", "read"),
        _trace(ComputerActionType.TERMINAL_EXEC, "pytest", "1 passed"),
        # A failed write must NOT be reported as a touched file.
        _trace(ComputerActionType.FILE_WRITE, "src/b.py", "permission denied", status="FAILED"),
    ]
    refs = {r["path"]: r for r in BossAgent._extract_file_references(traces)}
    assert set(refs) == {"src/a.py"}
    assert set(refs["src/a.py"]["operations"]) == {"write", "read"}


def test_test_results_parsed_from_real_runner_output():
    traces = [
        _trace(
            ComputerActionType.TERMINAL_EXEC,
            "python -m pytest tests/",
            "collected 5 items\n5 passed in 0.42s",
            exit_code=0,
        ),
        _trace(ComputerActionType.TERMINAL_EXEC, "ls -la", "src tests", exit_code=0),
    ]
    results = BossAgent._extract_test_results(traces)
    assert len(results) == 1
    assert results[0]["passed"] == 5
    assert results[0]["failed"] == 0
    assert results[0]["exit_code"] == 0


def test_test_results_records_failures():
    traces = [
        _trace(
            ComputerActionType.TERMINAL_EXEC,
            "pytest",
            "3 passed, 2 failed in 1.10s",
            exit_code=1,
        )
    ]
    results = BossAgent._extract_test_results(traces)
    assert results[0]["passed"] == 3
    assert results[0]["failed"] == 2
    assert results[0]["exit_code"] == 1


def test_no_evidence_extracted_from_prose_only_traces():
    # A trace whose observation merely *mentions* a file must not create a
    # reference; only real file actions count.
    traces = [_trace(ComputerActionType.TERMINAL_EXEC, "echo", "I edited src/a.py successfully")]
    assert BossAgent._extract_file_references(traces) == []
    assert BossAgent._extract_test_results(traces) == []


def test_evidence_incomplete_flag_shape():
    """The flag is a plain bool so downstream consumers can gate on it."""
    from sonic.computer_use.models_boss import SubMissionResult

    result = SubMissionResult(sub_mission_id="sub-1", goal="g")
    assert result.evidence_incomplete is False
    assert result.file_references == []
    assert result.test_results == []
    result.evidence_incomplete = True
    assert result.evidence_incomplete is True
