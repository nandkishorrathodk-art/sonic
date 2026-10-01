"""Focused coverage for generic human workflow graph extensions."""

import pytest

from sonic.agents.cognitive_state import CognitiveState
from sonic.agents.replan import ProposedTask, ReplanDecision, ReplanEngine
from sonic.agents.task_graph import (
    CheckpointStatus,
    TaskGraph,
    TaskGraphError,
    TaskNode,
    TaskStatus,
)


def test_human_task_checkpoints_and_state_delta_round_trip():
    graph = TaskGraph("engagement-1", "tenant-1")
    task_id = graph.add_task(
        TaskNode(name="Operator handoff", agent_type="human", requires_human=True)
    )
    checkpoint_id = graph.add_checkpoint(task_id, "Review current state")
    graph.update_state_delta(task_id, {"next_step": "operator review"})
    graph.record_checkpoint(
        task_id,
        checkpoint_id,
        state_delta={"reviewed": True},
        note="Operator acknowledged the handoff",
    )

    task = graph.get_task(task_id)
    assert task is not None
    assert task.checkpoints[0].status == CheckpointStatus.COMPLETED
    assert task.state_delta == {"next_step": "operator review", "reviewed": True}

    restored = TaskGraph.from_dict(graph.to_dict())
    restored_task = restored.get_task(task_id)
    assert restored_task is not None
    assert restored_task.checkpoints[0].status == CheckpointStatus.COMPLETED
    assert restored_task.state_delta["reviewed"] is True


def test_recovery_branch_does_not_depend_on_failed_task():
    graph = TaskGraph("engagement-1", "tenant-1")
    failed_id = graph.add_task(
        TaskNode(name="Primary workflow", agent_type="human_workflow", max_retries=0)
    )
    graph.mark_running(failed_id)
    graph.mark_failed(failed_id, "operator unavailable")

    recovery_id = graph.branch_recovery(
        failed_id,
        [TaskNode(name="Recovery handoff", agent_type="human", requires_human=True)],
    )[0]
    recovery = graph.get_task(recovery_id)
    assert recovery is not None
    assert recovery.recovery_for == failed_id
    assert failed_id not in recovery.depends_on
    assert recovery.status == TaskStatus.READY


def test_replan_accepts_human_workflow_metadata_and_rejects_bad_recovery_source():
    graph = TaskGraph("engagement-1", "tenant-1")
    source_id = graph.add_task(
        TaskNode(name="Primary", agent_type="dynamic", max_retries=0)
    )
    graph.mark_running(source_id)
    graph.mark_failed(source_id, "failed")
    state = CognitiveState(engagement_id="engagement-1", tenant_id="tenant-1")
    engine = ReplanEngine()

    decision = ReplanDecision(
        should_replan=True,
        new_tasks=[
            ProposedTask(
                name="Human recovery",
                agent_type="human",
                requires_human=True,
                recovery_for=source_id,
                checkpoints=[{"name": "Confirm revised approach"}],
                state_delta={"reason": "operator choice"},
            )
        ],
    )
    validation = engine.validate_decision(decision, state, graph)
    assert validation.valid
    changes = engine.apply_decision(decision, graph, state)
    task = graph.get_task(changes["tasks_added"][0])
    assert task is not None
    assert task.requires_human is True
    assert task.recovery_for == source_id
    assert task.checkpoints[0].name == "Confirm revised approach"

    invalid = ReplanDecision(
        should_replan=True,
        new_tasks=[
            ProposedTask(
                name="Invalid recovery",
                agent_type="human",
                recovery_for="missing-task",
            )
        ],
    )
    rejected = engine.validate_decision(invalid, state, graph)
    assert not rejected.valid
    assert any("not found" in error for error in rejected.errors)


@pytest.mark.parametrize("terminal_status", ["failed", "blocked", "cancelled"])
def test_replan_rejects_non_successful_dependencies(terminal_status):
    graph = TaskGraph("engagement-1", "tenant-1")
    source_id = graph.add_task(
        TaskNode(name="Prerequisite", agent_type="recon", max_retries=0)
    )
    source = graph.get_task(source_id)
    if terminal_status == "failed":
        graph.mark_running(source_id)
        graph.mark_failed(source_id, "failed")
    elif terminal_status == "blocked":
        graph.mark_blocked(source_id, "blocked")
    else:
        graph.mark_cancelled(source_id, "cancelled")
    state = CognitiveState(engagement_id="engagement-1", tenant_id="tenant-1")
    decision = ReplanDecision(
        should_replan=True,
        new_tasks=[
            ProposedTask(
                name="Follow-up",
                agent_type="recon",
                depends_on_completed=[source_id],
            )
        ],
    )
    engine = ReplanEngine()

    validation = engine.validate_decision(decision, state, graph)
    changes = engine.apply_decision(decision, graph, state)

    assert source is not None and source.status.value == terminal_status
    assert not validation.valid
    assert not changes["tasks_added"]
    assert any("not succeeded" in error for error in changes["errors"])


def test_director_replan_registry_rejects_graph_only_human_worker_type():
    graph = TaskGraph("engagement-1", "tenant-1")
    state = CognitiveState(engagement_id="engagement-1", tenant_id="tenant-1")
    decision = ReplanDecision(
        should_replan=True,
        new_tasks=[
            ProposedTask(name="Human handoff", agent_type="human_workflow")
        ],
    )
    engine = ReplanEngine(allowed_agent_types={"recon"})

    validation = engine.validate_decision(decision, state, graph)

    assert not validation.valid
    assert any("no registered worker" in error for error in validation.errors)


def test_checkpoint_requires_existing_task_and_checkpoint():
    graph = TaskGraph("engagement-1", "tenant-1")
    with pytest.raises(TaskGraphError):
        graph.add_checkpoint("missing", "checkpoint")
    task_id = graph.add_task(TaskNode(name="Workflow", agent_type="human"))
    with pytest.raises(TaskGraphError):
        graph.record_checkpoint(task_id, "missing")
