"""Regression coverage for Boss completion verification on arbitrary GUI tasks."""

from types import SimpleNamespace
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.agents.judge import JudgeAgent, JudgeVerdict
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import ComputerActionType, ComputerDecisionTrace
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult


class _Desktop:
    async def screenshot(self, workspace_id):
        return SimpleNamespace(
            screenshot_base64="dGVzdC1mcmFtZQ==",
            capture_timestamp_ns=time.time_ns(),
            capture_age_ms=0.0,
            desktop_state="INTERACTIVE",
            active_window="Document editor",
        )


class _VisionJudge:
    def __init__(self, content):
        self.content = content
        self.complete = AsyncMock(return_value=SimpleNamespace(content=content))


def _gui_result():
    trace = ComputerDecisionTrace(
        action_type=ComputerActionType.GUI_CLICK,
        target_resource="visible save control",
        predicted_outcome="Save the requested report",
        actual_observation="Clicked the visible save control; desktop refreshed",
        status="SUCCESS",
        exit_code=0,
    )
    return SubMissionResult(
        sub_mission_id="gui-task",
        goal="Prepare and save the requested report in the installed office program",
        role="worker",
        traces=[trace],
        findings_summary="The requested report is visible and saved.",
        success=True,
        actions_taken=1,
    )


@pytest.mark.asyncio
async def test_gui_judge_uses_fresh_screenshot_and_confirms_visible_outcome():
    router = _VisionJudge(
        '{"verdict":"CONFIRMED","visible_evidence":"The report is displayed in the editor."}'
    )
    judge = JudgeAgent(computer_provider=_Desktop(), llm_router=router)
    result = _gui_result()
    sub_mission = SubMission(
        id="gui-task",
        goal=result.goal,
        plane="gui",
    )

    verdict = await judge.evaluate_sub_mission("workspace", sub_mission, result)

    assert verdict.verdict == JudgeVerdict.CONFIRMED
    request = router.complete.await_args.args[0]
    assert request.task_type == "vision"
    assert request.messages[-1].images[0].base64 == "dGVzdC1mcmFtZQ=="


@pytest.mark.asyncio
async def test_gui_judge_does_not_confirm_ambiguous_screenshot():
    router = _VisionJudge(
        '{"verdict":"INCONCLUSIVE","visible_evidence":"The target state is not legible."}'
    )
    judge = JudgeAgent(computer_provider=_Desktop(), llm_router=router)
    result = _gui_result()

    verdict = await judge.evaluate_sub_mission(
        "workspace",
        SubMission(id="gui-task", goal=result.goal, plane="gui"),
        result,
    )

    assert verdict.verdict == JudgeVerdict.INCONCLUSIVE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capture_timestamp_ns", "capture_age_ms"),
    ((None, None), (time.time_ns() - 5_000_000_000, None), (None, 2_001.0)),
)
async def test_gui_judge_rejects_missing_or_stale_screenshot(
    capture_timestamp_ns,
    capture_age_ms,
):
    router = _VisionJudge(
        '{"verdict":"CONFIRMED","visible_evidence":"The requested result is visible."}'
    )
    screen = SimpleNamespace(
        screenshot_base64="dGVzdC1mcmFtZQ==",
        capture_timestamp_ns=capture_timestamp_ns,
        capture_age_ms=capture_age_ms,
        desktop_state="INTERACTIVE",
        active_window="Document editor",
    )
    judge = JudgeAgent(
        computer_provider=SimpleNamespace(screenshot=AsyncMock(return_value=screen)),
        llm_router=router,
    )

    verdict = await judge.evaluate_sub_mission(
        "workspace",
        SubMission(id="gui-task", goal="Save the requested report", plane="gui"),
        _gui_result(),
    )

    assert verdict.verdict == JudgeVerdict.INCONCLUSIVE
    router.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_judge_requires_a_successful_gui_action_receipt():
    router = _VisionJudge(
        '{"verdict":"CONFIRMED","visible_evidence":"The requested result is visible."}'
    )
    judge = JudgeAgent(computer_provider=_Desktop(), llm_router=router)
    result = _gui_result()
    result.traces[0].status = "BLOCKED"

    verdict = await judge.evaluate_sub_mission(
        "workspace",
        SubMission(id="gui-task", goal=result.goal, plane="gui"),
        result,
    )

    assert verdict.verdict == JudgeVerdict.INCONCLUSIVE
    assert "successful GUI interaction" in verdict.notes
    router.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_judge_fails_closed_without_vision_or_screenshot():
    result = _gui_result()
    no_vision = JudgeAgent(computer_provider=_Desktop())
    no_screen = JudgeAgent(
        computer_provider=SimpleNamespace(
            screenshot=AsyncMock(return_value=SimpleNamespace(screenshot_base64=""))
        ),
        llm_router=_VisionJudge(
            '{"verdict":"CONFIRMED","visible_evidence":"claimed"}'
        ),
    )
    sub_mission = SubMission(id="gui-task", goal=result.goal, plane="gui")

    assert (
        await no_vision.evaluate_sub_mission("workspace", sub_mission, result)
    ).verdict == JudgeVerdict.INCONCLUSIVE
    assert (
        await no_screen.evaluate_sub_mission("workspace", sub_mission, result)
    ).verdict == JudgeVerdict.INCONCLUSIVE


@pytest.mark.asyncio
async def test_boss_rejects_worker_success_when_judge_is_unavailable():
    boss = BossAgent(computer_provider=_Desktop(), llm_router=MagicMock())
    boss.judge = None
    worker = MagicMock()
    worker.run_mission = AsyncMock(return_value=[
        ComputerDecisionTrace(
            action_type=ComputerActionType.TERMINAL_EXEC,
            target_resource="inspect",
            predicted_outcome="inspect",
            actual_observation="inspection completed",
            status="SUCCESS",
            exit_code=0,
        )
    ])
    worker.mission_summary = "Inspection completed."

    from unittest.mock import patch

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=worker):
        result = await boss._dispatch_sub_agent(
            "workspace",
            SubMission(id="headless-task", goal="Inspect the current workspace", plane="headless"),
        )

    assert result.success is False
    assert result.evidence_verified is False
    assert result.status == "FAILED"
    assert "independent Judge is not configured" in result.findings_summary


@pytest.mark.asyncio
async def test_completion_checker_does_not_accept_success_trace_as_judge_proof():
    router = _VisionJudge('{"complete":true,"reason":"worker says done"}')
    boss = BossAgent(computer_provider=_Desktop(), llm_router=router)
    result = SubMissionResult(
        sub_mission_id="unjudged",
        goal="Inspect and update the workspace",
        success=True,
        evidence_verified=False,
        traces=[
            ComputerDecisionTrace(
                action_type=ComputerActionType.TERMINAL_EXEC,
                target_resource="inspect",
                predicted_outcome="inspect",
                actual_observation="operation completed",
                status="SUCCESS",
                exit_code=0,
            )
        ],
    )

    complete, reason = await boss._check_objective_complete(
        "Inspect and update the workspace",
        "Worker claims completion",
        [result],
    )

    assert complete is False
    assert "no independently verified" in reason.lower()


@pytest.mark.asyncio
async def test_completion_checker_requires_success_as_well_as_verified_evidence():
    router = _VisionJudge('{"complete":true,"reason":"verification flag is set"}')
    boss = BossAgent(computer_provider=_Desktop(), llm_router=router)
    inconsistent = SubMissionResult(
        sub_mission_id="failed-but-marked-verified",
        goal="Create a document",
        success=False,
        evidence_verified=True,
        findings_summary="The worker failed.",
    )

    complete, reason = await boss._check_objective_complete(
        "Create a document",
        "Worker failed",
        [inconsistent],
    )

    assert complete is False
    assert "no independently verified" in reason.lower()


@pytest.mark.asyncio
async def test_completion_checker_rejects_verified_but_unrelated_subtask():
    router = _VisionJudge('{"complete":true,"reason":"A worker succeeded"}')
    boss = BossAgent(computer_provider=_Desktop(), llm_router=router)
    unrelated = SubMissionResult(
        sub_mission_id="network-check",
        goal="Check network connectivity",
        success=True,
        evidence_verified=True,
        findings_summary="Network connectivity check independently verified.",
    )

    complete, reason = await boss._check_objective_complete(
        "Create a folder for project notes and save report",
        "A worker completed a network check.",
        [unrelated],
    )

    assert complete is False
    assert "do not cover every meaningful part" in reason


@pytest.mark.asyncio
async def test_completion_checker_rejects_partial_verified_objective_coverage():
    router = _VisionJudge('{"complete":true,"reason":"The folder is ready"}')
    boss = BossAgent(computer_provider=_Desktop(), llm_router=router)
    partial = SubMissionResult(
        sub_mission_id="folder-only",
        goal="Create a folder for project notes",
        success=True,
        evidence_verified=True,
        findings_summary="The project notes folder was independently verified.",
    )

    complete, reason = await boss._check_objective_complete(
        "Create a folder for project notes and save report",
        "The folder is ready.",
        [partial],
    )

    assert complete is False
    assert "do not cover every meaningful part" in reason


@pytest.mark.asyncio
async def test_completion_checker_accepts_verified_subtasks_covering_entire_goal():
    router = _VisionJudge('{"complete":true,"reason":"Both results are verified"}')
    boss = BossAgent(computer_provider=_Desktop(), llm_router=router)
    results = [
        SubMissionResult(
            sub_mission_id="folder",
            goal="Create a folder for project notes",
            success=True,
            evidence_verified=True,
            findings_summary="Folder created and verified.",
        ),
        SubMissionResult(
            sub_mission_id="report",
            goal="Save report for project notes",
            success=True,
            evidence_verified=True,
            findings_summary="Report saved and verified.",
        ),
    ]

    complete, reason = await boss._check_objective_complete(
        "Create a folder for project notes and save report",
        "Folder and report verified.",
        results,
    )

    assert complete is True
    assert reason == "Both results are verified"


@pytest.mark.asyncio
async def test_final_boss_report_is_not_complete_without_judge_confirmation():
    boss = BossAgent(
        computer_provider=_Desktop(),
        llm_router=_VisionJudge("Summary based on the available work."),
    )
    boss.phases.append(Phase(
        phase_number=1,
        name="GUI work",
        results=[
            SubMissionResult(
                sub_mission_id="unjudged",
                goal="Update the visible document",
                success=True,
                evidence_verified=False,
                traces=[
                    ComputerDecisionTrace(
                        action_type=ComputerActionType.GUI_TYPE,
                        target_resource="document",
                        predicted_outcome="update",
                        actual_observation="Typed text",
                        status="SUCCESS",
                    )
                ],
            )
        ],
    ))

    report = await boss._synthesize_final_report("Update the visible document", time.perf_counter())

    assert report.status == "PARTIAL"
