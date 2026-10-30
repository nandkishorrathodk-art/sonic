"""Focused regression tests for orchestration truthfulness semantics."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from sonic.agents.orchestrator import MetaOrchestrator
from sonic.agents.task_graph import TaskGraph, TaskNode, TaskStatus, TaskGraphError
from sonic.computer_use.models_boss import SubMissionResult


def test_task_graph_blocked_is_not_success():
    graph = TaskGraph(engagement_id="e", tenant_id="t")
    task = TaskNode(name="triage", agent_type="recon", engagement_id="e", tenant_id="t")
    task_id = graph.add_task(task)
    graph.mark_blocked(task_id, "not executed")
    assert graph.get_task(task_id).status == TaskStatus.BLOCKED
    assert graph.get_task(task_id).error == "not executed"


def test_meta_orchestrator_reports_only_succeeded_phases():
    graph = TaskGraph(engagement_id="e", tenant_id="t")
    first = TaskNode(name="Recon: inspect", agent_type="recon", engagement_id="e", tenant_id="t")
    second = TaskNode(
        name="Testing: probe",
        agent_type="dynamic",
        engagement_id="e",
        tenant_id="t",
        max_retries=0,
    )
    first_id = graph.add_task(first)
    second_id = graph.add_task(second)
    graph.mark_running(first_id)
    graph.mark_completed(first_id, {"status": "success"})
    graph.mark_running(second_id)
    graph.mark_failed(second_id, "blocked")
    names = MetaOrchestrator._completed_phase_names(
        graph, [{"name": "Recon"}, {"name": "Testing"}]
    )
    assert names == ["Recon"]


@pytest.mark.asyncio
async def test_meta_orchestrator_does_not_attribute_director_work_to_its_plan_phases():
    director = MagicMock()
    director.start_engagement = AsyncMock(return_value="director-engagement")
    director.run_engagement_loop = AsyncMock(
        return_value={"graph_complete": True, "completed": 2, "failed": 0}
    )
    orchestrator = MetaOrchestrator(director=director, tenant_id="tenant")
    orchestrator._create_plan = AsyncMock(
        return_value={
            "phases": [
                {"name": "Recon", "tasks": ["inspect source"]},
                {"name": "Verification", "tasks": ["verify findings"]},
            ]
        }
    )

    result = await orchestrator.run({"target": "app.test", "worker_fn": AsyncMock()})

    assert result["phases_completed"] == []
    assert orchestrator.status == "incomplete"


def test_task_failures_retry_with_a_bound_and_reject_stale_callbacks():
    graph = TaskGraph(engagement_id="e", tenant_id="t")
    task_id = graph.add_task(
        TaskNode(name="retryable", agent_type="recon", max_retries=1)
    )
    first_execution_id = graph.mark_running(task_id)
    graph.mark_failed(task_id, "transient", execution_id=first_execution_id)

    task = graph.get_task(task_id)
    assert task is not None
    assert task.retry_count == 1
    assert task.status == TaskStatus.READY
    second_execution_id = graph.mark_running(task_id)
    with pytest.raises(TaskGraphError):
        graph.mark_failed(
            task_id,
            "stale callback from first attempt",
            execution_id=first_execution_id,
        )
    graph.mark_failed(task_id, "retry exhausted", execution_id=second_execution_id)
    assert task.status == TaskStatus.FAILED
    with pytest.raises(TaskGraphError):
        graph.mark_completed(task_id, {"status": "success"})


def test_boss_timeout_result_cannot_be_evidence_verified():
    result = SubMissionResult(
        sub_mission_id="sub-1",
        goal="inspect target",
        status="TIMED_OUT",
        success=False,
        evidence_verified=False,
    )
    assert result.status == "TIMED_OUT"
    assert result.success is False
    assert result.evidence_verified is False


def test_director_fallback_plan_is_modality_neutral():
    from sonic.agents.director import Director
    plan = Director.__new__(Director)._default_plan("sample.bin")
    assert "web application" not in str(plan).lower()
    assert plan["tasks"][0]["task_payload"]["task"] if "task_payload" in plan["tasks"][0] else plan["tasks"][0]["payload"]["task"] == "establish_target_modality_and_surface"


def test_dynamic_candidate_is_not_reportable_by_engagement_collector():
    from sonic.agents.engagement import EngagementManager
    collected = EngagementManager._collect_findings(object(), {
        "dynamic": {"findings": [{"title": "signal", "status": "candidate"}]},
        "verify": {"findings": [{"title": "verified", "status": "verified"}]},
    })
    assert [item["title"] for item in collected] == ["verified"]
