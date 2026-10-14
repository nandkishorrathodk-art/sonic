"""Boss delegates every task to the shared GUI-capable computer-use worker."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.agents.judge import JudgeVerdict
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import SubMission


class _MockProvider:
    async def screenshot(self, *args, **kwargs):
        return b"desktop"

    async def gui_action(self, *args, **kwargs):
        return True


class _MockLLM:
    async def complete(self, request, **kwargs):
        return MagicMock(content="{}")


@pytest.mark.asyncio
async def test_visual_mission_uses_shared_gui_computer_use_worker():
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    boss.judge.evaluate_sub_mission = AsyncMock(
        return_value=SimpleNamespace(
            verdict=JudgeVerdict.CONFIRMED,
            notes="The Judge confirms the visible GUI action receipt.",
        )
    )
    sub_mission = SubMission(
        id="sub-desktop-web",
        goal="Open the provided resource in an installed desktop application and inspect it",
        role="browser",
        plane="browser",
        max_steps=3,
    )
    worker = MagicMock()
    worker.run_mission = AsyncMock(return_value=[
        {
            "step_index": 1,
            "action_type": "GUI_CLICK",
            "target_resource": "visible application control",
            "actual_observation": "Control activated on the shared desktop",
            "status": "SUCCESS",
            "exit_code": 0,
        },
    ])
    worker.mission_summary = "Visible application control activated."

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=worker) as make_worker:
        result = await boss._dispatch_sub_agent("ws-1", sub_mission, sub_agent_num=1)

    make_worker.assert_called_once()
    worker_config = make_worker.call_args.kwargs
    assert worker_config["execution_plane"] == "gui"
    assert worker_config["observe_desktop"] is True
    assert "browser" not in worker_config
    worker.run_mission.assert_awaited_once()
    assert result.success is True
    assert result.status == "SUCCESS"
    assert result.actions_taken == 1


@pytest.mark.asyncio
async def test_visual_worker_heartbeat_is_forwarded_to_boss_callback():
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    events = []

    async def phase_callback(event_type, data):
        events.append((event_type, data))

    worker = MagicMock()
    worker.run_mission = AsyncMock(return_value=[])

    with patch(
        "sonic.computer_use.agent.ComputerUseAgent",
        return_value=worker,
    ):
        await boss._dispatch_sub_agent(
            "ws-1",
            SubMission(
                id="sub-heartbeat",
                name="Desktop worker",
                goal="Inspect the visible desktop",
                plane="gui",
                max_steps=1,
            ),
            phase_callback=phase_callback,
            sub_agent_num=1,
        )

    await worker._heartbeat_callback(
        {
            "step": 2,
            "elapsed_seconds": 4.5,
            "last_observation_version": 7,
        }
    )

    assert events == [
        (
            "sub_heartbeat",
            {
                "sub_mission_id": "sub-heartbeat",
                "agent_id": "sub-agent-1",
                "agent_name": "Desktop worker",
                "role": "subagent",
                "plane": "gui",
                "sub_agent_number": 1,
                "elapsed_seconds": 4.5,
                "step": 2,
                "last_observation_version": 7,
            },
        )
    ]


@pytest.mark.asyncio
async def test_run_mission_forwards_interrupt_check():
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    report = MagicMock()
    interrupt_check = MagicMock(return_value=True)

    with patch.object(boss, "run", new_callable=AsyncMock, return_value=report) as run:
        result = await boss.run_mission(
            workspace_id="ws-1",
            goal="Inspect the desktop",
            interrupt_check=interrupt_check,
        )

    assert result is report
    run.assert_awaited_once_with(
        workspace_id="ws-1",
        objective="Inspect the desktop",
        phase_callback=None,
        requested_plane=None,
        interrupt_check=interrupt_check,
    )


@pytest.mark.asyncio
async def test_subagent_interrupt_callback_includes_external_operator_interrupt():
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    operator_interrupted = [False]
    received = {}
    worker = MagicMock()

    async def run_worker(**kwargs):
        received.update(kwargs)
        assert kwargs["interrupt_check"]() is False
        operator_interrupted[0] = True
        return []

    worker.run_mission = AsyncMock(side_effect=run_worker)

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=worker):
        await boss._dispatch_sub_agent(
            "ws-1",
            SubMission(
                id="sub-interrupt",
                goal="Inspect the desktop",
                plane="gui",
                max_steps=1,
            ),
            sub_agent_num=1,
            interrupt_check=lambda: operator_interrupted[0],
        )

    assert received["interrupt_check"]() is True


@pytest.mark.asyncio
async def test_subagent_self_correction_on_judge_rejection():
    """If Judge rejects initial outcome, worker gets up to 1 self-correction turn with Judge feedback."""
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())

    eval_falsified = SimpleNamespace(
        verdict=JudgeVerdict.FALSIFIED,
        notes="File was not saved with the expected factorial function.",
    )
    eval_confirmed = SimpleNamespace(
        verdict=JudgeVerdict.CONFIRMED,
        notes="The Judge confirms the verified test run.",
    )
    boss.judge.evaluate_sub_mission = AsyncMock(side_effect=[eval_falsified, eval_confirmed])

    worker = MagicMock()
    trace_fail = {
        "step_index": 1,
        "action_type": "TERMINAL_EXEC",
        "target_resource": "cat app.py",
        "actual_observation": "partial code",
        "status": "SUCCESS",
        "exit_code": 0,
    }
    trace_fix = {
        "step_index": 2,
        "action_type": "FILE_WRITE",
        "target_resource": "app.py",
        "actual_observation": "GOAL_COMPLETE",
        "status": "SUCCESS",
        "exit_code": 0,
    }
    worker.run_mission = AsyncMock(side_effect=[[trace_fail], [trace_fix]])
    worker.mission_summary = "Fixed and verified"

    sub_mission = SubMission(
        id="sub-correct",
        goal="Implement factorial function",
        role="coder",
        plane="headless",
        max_steps=2,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=worker):
        result = await boss._dispatch_sub_agent("ws-1", sub_mission, sub_agent_num=1)

    assert worker.run_mission.await_count == 2
    second_call_goal = worker.run_mission.call_args_list[1].kwargs["goal"]
    assert "INDEPENDENT JUDGE FEEDBACK - SELF-CORRECTION" in second_call_goal
    assert "File was not saved with the expected factorial function" in second_call_goal

    assert boss.judge.evaluate_sub_mission.await_count == 2
    assert result.success is True
    assert result.evidence_verified is True
    assert result.status == "SUCCESS"
    assert result.judge_verdict == JudgeVerdict.CONFIRMED.value


@pytest.mark.asyncio
async def test_subagent_self_correction_failure_finalizes_failed_status():
    """If self-correction still fails Judge verification, finalize FAILED status."""
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())

    eval_falsified_1 = SimpleNamespace(
        verdict=JudgeVerdict.FALSIFIED,
        notes="Tests failed on assertion.",
    )
    eval_falsified_2 = SimpleNamespace(
        verdict=JudgeVerdict.FALSIFIED,
        notes="Tests still failing after fix attempt.",
    )
    boss.judge.evaluate_sub_mission = AsyncMock(side_effect=[eval_falsified_1, eval_falsified_2])

    worker = MagicMock()
    trace = {
        "step_index": 1,
        "action_type": "TERMINAL_EXEC",
        "target_resource": "pytest",
        "actual_observation": "1 failed",
        "status": "FAILED",
        "exit_code": 1,
    }
    worker.run_mission = AsyncMock(side_effect=[[trace], [trace]])
    worker.mission_summary = "Fix attempted but failed"

    sub_mission = SubMission(
        id="sub-fail",
        goal="Fix test",
        role="coder",
        plane="headless",
        max_steps=2,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=worker):
        result = await boss._dispatch_sub_agent("ws-1", sub_mission, sub_agent_num=1)

    assert worker.run_mission.await_count == 2
    assert result.success is False
    assert result.evidence_verified is False
    assert result.status == "FAILED"
    assert result.judge_verdict == JudgeVerdict.FALSIFIED.value
    assert "FALSIFIED by Judge" in result.findings_summary


@pytest.mark.asyncio
async def test_subagent_no_self_correction_when_waiting_for_human():
    """Waiting for human state must not trigger automated self-correction loop."""
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())

    worker = MagicMock()
    worker._waiting_for_human = True
    worker._waiting_for_human_reason = "CAPTCHA detected"
    worker.run_mission = AsyncMock(return_value=[{
        "step_index": 1,
        "action_type": "WAIT_FOR_HUMAN",
        "target_resource": "captcha",
        "actual_observation": "CAPTCHA challenge",
        "status": "WAITING_FOR_HUMAN",
    }])
    worker.mission_summary = "Waiting for operator"

    sub_mission = SubMission(
        id="sub-wfh",
        goal="Solve login",
        role="worker",
        plane="gui",
        max_steps=2,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=worker):
        result = await boss._dispatch_sub_agent("ws-1", sub_mission, sub_agent_num=1)

    assert worker.run_mission.await_count == 1
    assert result.waiting_for_human is True
    assert result.status == "WAITING_FOR_HUMAN"
