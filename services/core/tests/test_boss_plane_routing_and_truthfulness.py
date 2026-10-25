"""
Tests for BossAgent plane routing, FILE_WRITE JSON-unwrapping, and zero-hallucination failure summaries.
Validates Dual-Plane Concurrency (Mandates 1, 5) and Empirical Falsification (Rule 15).
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
)
from sonic.computer_use.models_boss import SubMission, SubMissionResult


def test_sub_mission_plane_inference_dual_plane():
    """Verify clean plane separation between headless coding and GUI browser verification."""
    # 1. Directory creation & python server authoring -> headless
    s1 = SubMission(
        goal="Create directory /home/sonic/workspace/task_manager_app and author server.py web service"
    )
    assert s1.plane == "headless"

    # 2. Launching background server and verifying with curl -> headless
    s2 = SubMission(
        goal="Launch server.py on port 5000 in background and test endpoints via curl"
    )
    assert s2.plane == "headless"

    # 3. Code inspection, file read, grep -> headless
    s3 = SubMission(
        goal="Inspect server.py code and check open ports with netstat or ss"
    )
    assert s3.plane == "headless"

    # 4. Opening browser and visually verifying UI -> gui
    s4 = SubMission(
        goal="Launch google-chrome to http://127.0.0.1:5000 and visually verify the task manager dark UI"
    )
    assert s4.plane == "gui"

    # 5. Clicking controls in graphical application -> gui
    s5 = SubMission(
        goal="Click button on screen to add a new task and verify visual completion"
    )
    assert s5.plane == "gui"


@pytest.mark.asyncio
async def test_requested_plane_overrides_decomposition_plane():
    router = MagicMock()
    router.complete = AsyncMock(return_value=type(
        "Response",
        (),
        {
            "content": json.dumps({
                "thinking": "One focused worker.",
                "phase_name": "Inspection",
                "sub_missions": [
                    {"goal": "Inspect the repository", "plane": "gui", "max_steps": 2}
                ],
            })
        },
    )())
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=router,
        requested_plane="headless",
    )

    phase = await boss._strategic_decomposition("inspect the repository")

    assert phase is not None
    assert phase.sub_missions[0].plane == "headless"


def test_fallback_worker_uses_laya_plane_for_original_objective(monkeypatch):
    plane_router = SimpleNamespace(
        decide_execution_plane=lambda goal: (
            "headless" if "test" in goal else "gui",
            0.95,
        )
    )
    monkeypatch.setattr(
        "sonic.llm.laya_engine.get_laya_engine",
        lambda: plane_router,
    )
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=MagicMock(),
    )

    phase = boss._fallback_decomposition("Run the test suite")

    assert phase.sub_missions[0].goal == "Run the test suite"
    assert phase.sub_missions[0].plane == "headless"


def test_explicit_requested_plane_overrides_fallback_laya_plane(monkeypatch):
    monkeypatch.setattr(
        "sonic.llm.laya_engine.get_laya_engine",
        lambda: SimpleNamespace(
            decide_execution_plane=lambda _goal: ("headless", 0.95)
        ),
    )
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=MagicMock(),
        requested_plane="gui",
    )

    phase = boss._fallback_decomposition("Run the test suite")

    assert phase.sub_missions[0].plane == "gui"


@pytest.mark.asyncio
async def test_boss_interrupt_before_planning_prevents_worker_dispatch():
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=MagicMock(),
    )
    boss._strategic_decomposition = AsyncMock(
        side_effect=AssertionError("planning must not run after interrupt")
    )

    report = await boss.run(
        workspace_id="ws-1",
        objective="Use the desktop to complete a task",
        interrupt_check=lambda: True,
    )

    assert report.status == "INCOMPLETE"
    assert report.total_sub_agents == 0
    assert "before planning or worker dispatch" in report.findings_summary
    boss._strategic_decomposition.assert_not_awaited()


def test_shared_display_serializes_gui_workers_only():
    computer = type(
        "DisplayProvider",
        (),
        {"screenshot": lambda self: None, "gui_action": lambda self: None},
    )()
    boss = BossAgent(
        computer_provider=computer,
        llm_router=MagicMock(),
        max_parallel_workers=4,
    )

    assert boss._worker_limit_for_wave([
        SubMission(goal="inspect source", plane="headless"),
        SubMission(goal="run tests", plane="headless"),
    ]) == 4
    assert boss._worker_limit_for_wave([
        SubMission(goal="inspect source", plane="headless"),
        SubMission(goal="click the visible control", plane="gui"),
    ]) == 1


@pytest.mark.asyncio
async def test_file_write_unwraps_json_envelope():
    """FILE_WRITE must unwrap JSON string payloads into raw code, avoiding JSON-string pollution on disk."""
    mock_comp = MagicMock()
    mock_comp.write_file = AsyncMock(return_value=True)

    agent = ComputerUseAgent(
        computer_provider=mock_comp,
        llm_router=MagicMock(),
        tenant_id="test-tenant",
    )

    # Simulate LLM outputting a JSON string payload
    raw_json_payload = json.dumps({
        "path": "/home/sonic/workspace/app/server.py",
        "content": "from http.server import HTTPServer\nprint('Server started')\n"
    })

    result = await agent.execute_action(
        workspace_id="ws-test",
        action_type=ComputerActionType.FILE_WRITE,
        target_resource="/home/sonic/workspace/app/server.py",
        payload=raw_json_payload,
        predicted_outcome="File written successfully",
    )

    # Verify write_file was called with the unwrapped raw python code, NOT the JSON envelope!
    mock_comp.write_file.assert_awaited_once_with(
        "ws-test",
        "/home/sonic/workspace/app/server.py",
        "from http.server import HTTPServer\nprint('Server started')\n"
    )


@pytest.mark.asyncio
async def test_summarize_sub_agent_traces_deterministic_on_failure():
    """When all worker traces fail, _summarize_sub_agent_traces must return factual failure without LLM hallucination."""
    mock_comp = MagicMock()
    boss = BossAgent(
        computer_provider=mock_comp,
        llm_router=MagicMock(),
        tenant_id="test-tenant",
    )

    traces = [
        ComputerDecisionTrace(
            step_index=1,
            action_type=ComputerActionType.TERMINAL_EXEC,
            target_resource="mkdir -p /app",
            status=ActionExecutionStatus.FAILED,
            exit_code=1,
            actual_observation="Permission denied",
            predicted_outcome="Directory created",
        ),
        ComputerDecisionTrace(
            step_index=2,
            action_type=ComputerActionType.FILE_WRITE,
            target_resource="/app/server.py",
            status=ActionExecutionStatus.BLOCKED,
            exit_code=126,
            actual_observation="Policy block",
            predicted_outcome="File written",
        ),
    ]

    summary = await boss._summarize_sub_agent_traces("Setup app directory and server.py", traces)
    assert "Execution failed: no verified progress" in summary
    assert "TERMINAL_EXEC" in summary
    assert "Permission denied" in summary


@pytest.mark.asyncio
async def test_plan_next_phase_respects_headless_plane_without_explicit_constraint():
    """Verify that _plan_next_phase respects LLM's requested headless plane and avoids sticky Phase 1 poisoning."""
    from sonic.computer_use.models_boss import Phase

    router = MagicMock()
    router.complete = AsyncMock(return_value=type(
        "Response",
        (),
        {
            "content": json.dumps({
                "thinking": "Deploy headless test runner.",
                "phase_name": "Testing",
                "sub_missions": [
                    {"id": "sub-test", "goal": "Run pytest suite", "plane": "headless", "max_steps": 2}
                ],
            })
        },
    )())
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=router,
    )
    # Simulate Phase 1 having run and set self.requested_plane = "gui"
    boss.requested_plane = "gui"
    assert boss._explicit_requested_plane is None

    past_phase = Phase(
        phase_number=1,
        name="UI Setup",
        sub_missions=[SubMission(id="sub-gui", goal="Open app in browser", plane="gui")],
    )

    next_phase = await boss._plan_next_phase(
        objective="Test web application",
        all_findings="UI setup completed",
        phases=[past_phase],
    )

    assert next_phase is not None
    assert next_phase.sub_missions[0].plane == "headless"


@pytest.mark.asyncio
async def test_plan_next_phase_enforces_explicit_requested_plane():
    """When caller explicitly specifies requested_plane, _plan_next_phase forces all workers to that plane."""
    from sonic.computer_use.models_boss import Phase

    router = MagicMock()
    router.complete = AsyncMock(return_value=type(
        "Response",
        (),
        {
            "content": json.dumps({
                "thinking": "Deploy headless test runner.",
                "phase_name": "Testing",
                "sub_missions": [
                    {"id": "sub-test", "goal": "Run pytest suite", "plane": "headless", "max_steps": 2}
                ],
            })
        },
    )())
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=router,
        requested_plane="gui",
    )
    assert boss._explicit_requested_plane == "gui"

    past_phase = Phase(
        phase_number=1,
        name="UI Setup",
        sub_missions=[SubMission(id="sub-gui", goal="Open app in browser", plane="gui")],
    )

    next_phase = await boss._plan_next_phase(
        objective="Test web application",
        all_findings="UI setup completed",
        phases=[past_phase],
    )

    assert next_phase is not None
    assert next_phase.sub_missions[0].plane == "gui"


@pytest.mark.asyncio
async def test_shared_desktop_wave_admits_one_gui_worker_and_concurrent_headless_workers():
    """Wave scheduling limits shared desktop to 1 GUI worker, but allows headless workers to run concurrently."""
    from sonic.computer_use.models_boss import Phase

    computer = type(
        "DisplayProvider",
        (),
        {"screenshot": lambda self: None, "gui_action": lambda self: None},
    )()
    boss = BossAgent(
        computer_provider=computer,
        llm_router=MagicMock(),
        max_parallel_workers=4,
    )
    assert boss._shared_desktop is True

    # 4 unblocked ready candidates: GUI, headless, headless, GUI
    cands = [
        SubMission(id="g1", goal="GUI task 1", plane="gui"),
        SubMission(id="h1", goal="Headless task 1", plane="headless"),
        SubMission(id="h2", goal="Headless task 2", plane="headless"),
        SubMission(id="g2", goal="GUI task 2", plane="gui"),
    ]

    phase = Phase(phase_number=1, name="Mixed Phase", sub_missions=cands)
    boss._strategic_decomposition = AsyncMock(return_value=phase)
    boss._check_objective_complete = AsyncMock(return_value=(True, "All complete"))

    wave_dispatches = []
    current_wave_agents = []

    async def mock_dispatch(workspace_id, sub_m, *args, **kwargs):
        current_wave_agents.append(sub_m.id)
        return SubMissionResult(
            sub_mission_id=sub_m.id,
            goal=sub_m.goal,
            role=sub_m.role,
            success=True,
            status="SUCCESS",
            findings_summary="done",
            key_discoveries=[],
            actions_taken=1,
            duration_seconds=0.1,
            traces=[],
            agent_id="",
            agent_name="",
            judge_verdict="CONFIRMED",
            judge_notes="Verified",
            evidence_verified=True,
        )

    boss._dispatch_sub_agent = AsyncMock(side_effect=mock_dispatch)

    dispatched_subagent_events = []
    async def phase_cb(event_type, data):
        if event_type == "sub_dispatch":
            dispatched_subagent_events.append((data["sub_mission_id"], data["plane"]))

    report = await boss.run(
        workspace_id="ws-1",
        objective="Execute mixed workload",
        phase_callback=phase_cb,
    )

    assert report.status == "COMPLETE"
    # All 4 sub-missions were executed
    dispatched_ids = [e[0] for e in dispatched_subagent_events]
    assert len(dispatched_ids) == 4
    # The first wave must contain g1, h1, h2 (3 workers: 1 GUI + 2 headless)
    # and g2 must be deferred to the subsequent wave!
    assert dispatched_ids[:3] == ["g1", "h1", "h2"]
    assert dispatched_ids[3] == "g2"


@pytest.mark.asyncio
async def test_boss_reports_inconclusive_judge_as_unverified_not_failed():
    """An INCONCLUSIVE Judge verdict must not be reported as worker FAILED.

    The Judge returns INCONCLUSIVE for GUI work until an evaluated local
    verification head exists. The worker may still have really done the work
    (a live desktop showed the launched window). Reporting that as FAILED
    claimed a failure that never happened; UNVERIFIED is the honest status.
    """
    from sonic.agents.judge import JudgeVerdict

    assert BossAgent._status_for_judge_verdict(JudgeVerdict.FALSIFIED) == "FAILED"
    assert BossAgent._status_for_judge_verdict(JudgeVerdict.INCONCLUSIVE) == "UNVERIFIED"
    assert BossAgent._status_for_judge_verdict(JudgeVerdict.UNVERIFIED) == "UNVERIFIED"
    assert BossAgent._status_for_judge_verdict(JudgeVerdict.BENIGN) == "UNVERIFIED"
