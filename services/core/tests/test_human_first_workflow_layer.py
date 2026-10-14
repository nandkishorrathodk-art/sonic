"""Tests for the planning-only human-first engineering workflow."""

import pytest

from sonic.agents.task_graph import TaskGraphError, TaskStatus
from sonic.computer_use.human_workflow import (
    HumanFirstWorkflow,
    WorkflowOutcome,
)


def _finish(workflow, names):
    for name in names:
        workflow.record(name, success=True, observation=f"{name} observed")


def test_workflow_is_a_generic_dependency_chain_and_does_not_execute():
    workflow = HumanFirstWorkflow("improve the service")
    graph = workflow.build()
    names = [task.name for task in graph.tasks.values()]
    assert names == ["inspect", "reproduce", "patch", "test", "launch", "verify"]
    assert graph.get_ready_tasks()[0].name == "inspect"
    assert workflow.computer_agent is None
    assert workflow.report().outcome == WorkflowOutcome.INCOMPLETE

    for task in list(graph.tasks.values())[1:]:
        assert task.depends_on == [workflow.stage_ids[names[names.index(task.name) - 1]]]


def test_failure_is_recorded_and_replan_adds_independent_recovery_branch():
    workflow = HumanFirstWorkflow("repair a failing workflow")
    workflow.build()
    workflow.record("inspect", success=False, error="reproduction disagreed")
    source = workflow.graph.get_task(workflow.stage_ids["inspect"])
    assert source is not None and source.status == TaskStatus.FAILED

    recovery_ids = workflow.replan_after_failure("inspect", "try a different observation")
    recovery = workflow.graph.get_task(recovery_ids[0])
    assert recovery is not None
    assert recovery.recovery_for == source.id
    assert source.id not in recovery.depends_on
    assert recovery.status == TaskStatus.READY


def test_commit_and_publish_are_human_approval_boundaries():
    workflow = HumanFirstWorkflow("ship a small correction")
    workflow.add_release_boundary()
    _finish(workflow, ("inspect", "reproduce", "patch", "test", "launch", "verify"))

    commit = workflow.graph.get_task(workflow.stage_ids["commit"])
    publish = workflow.graph.get_task(workflow.stage_ids["publish"])
    assert commit is not None and commit.requires_human
    assert publish is not None and publish.requires_human
    with pytest.raises(TaskGraphError, match="operator approval"):
        workflow.record("commit", success=True, observation="ready")
    assert workflow.report().outcome == WorkflowOutcome.AWAITING_APPROVAL

    checkpoint = commit.checkpoints[0]
    workflow.graph.record_checkpoint(commit.id, checkpoint.id, note="approved")
    workflow.record("commit", success=True, observation="operator approved")
    assert workflow.graph.get_task(workflow.stage_ids["publish"]).status == TaskStatus.READY


def test_report_never_claims_completion_without_verified_final_stage():
    workflow = HumanFirstWorkflow("make a change")
    workflow.build()
    _finish(workflow, ("inspect", "reproduce", "patch", "test", "launch"))
    report = workflow.report()
    assert report.complete is False
    assert report.outcome == WorkflowOutcome.INCOMPLETE
    assert "verify" in report.pending_stages

