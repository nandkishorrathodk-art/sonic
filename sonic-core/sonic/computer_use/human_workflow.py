"""A planning-only, human-first software engineering workflow.

This module describes ordinary software work without deciding *how* a step is
performed.  A caller may use :class:`ComputerUseAgent` (or another worker) to
perform a ready task and then report the observed result back to this module.
The workflow itself never dispatches commands, browser actions, or commits.

The release boundary is deliberately explicit: commit and publish are not
part of the default execution chain and, when requested, are represented by
human-approval tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from sonic.agents.cognitive_state import CognitiveState
from sonic.agents.replan import ProposedTask, ReplanDecision, ReplanEngine, ReplanTrigger
from sonic.agents.task_graph import TaskGraph, TaskGraphError, TaskNode, TaskStatus

if TYPE_CHECKING:
    from sonic.computer_use.agent import ComputerUseAgent


class WorkflowStage(StrEnum):
    INSPECT = "inspect"
    REPRODUCE = "reproduce"
    PATCH = "patch"
    TEST = "test"
    LAUNCH = "launch"
    VERIFY = "verify"
    COMMIT = "commit"
    PUBLISH = "publish"


class WorkflowOutcome(StrEnum):
    INCOMPLETE = "incomplete"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass(frozen=True)
class WorkflowReport:
    """Truthful projection of graph state; it is not an execution receipt."""

    outcome: WorkflowOutcome
    completed_stages: tuple[str, ...]
    pending_stages: tuple[str, ...]
    failed_stages: tuple[str, ...]
    approval_required: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.outcome == WorkflowOutcome.SUCCEEDED


class HumanFirstWorkflow:
    """Build and track a generic engineering workflow using existing primitives.

    ``computer_agent`` is retained as an integration seam for callers that
    already have a :class:`ComputerUseAgent`; it is never invoked here.
    """

    BASE_STAGES = (
        WorkflowStage.INSPECT,
        WorkflowStage.REPRODUCE,
        WorkflowStage.PATCH,
        WorkflowStage.TEST,
        WorkflowStage.LAUNCH,
        WorkflowStage.VERIFY,
    )

    def __init__(
        self,
        objective: str,
        *,
        engagement_id: str = "software-workflow",
        tenant_id: str = "default",
        computer_agent: ComputerUseAgent | None = None,
        replan_engine: ReplanEngine | None = None,
    ):
        if not objective or not objective.strip():
            raise ValueError("objective must not be empty")
        self.objective = objective.strip()
        self.computer_agent = computer_agent
        self.replan_engine = replan_engine or ReplanEngine()
        self.graph = TaskGraph(engagement_id, tenant_id)
        self._stage_ids: dict[str, str] = {}
        self._built = False

    @property
    def stage_ids(self) -> dict[str, str]:
        return dict(self._stage_ids)

    def build(self) -> TaskGraph:
        """Create the dependency chain without executing any task."""
        if self._built:
            return self.graph
        previous: str | None = None
        for stage in self.BASE_STAGES:
            node = TaskNode(
                name=stage.value,
                agent_type="human_workflow",
                workflow_kind="software_engineering",
                task_payload={"stage": stage.value, "objective": self.objective},
                depends_on=[previous] if previous else [],
                tenant_id=self.graph.tenant_id,
                engagement_id=self.graph.engagement_id,
                expected_observation=f"Observed outcome for {stage.value}",
            )
            task_id = self.graph.add_task(node)
            self._stage_ids[stage.value] = task_id
            previous = task_id
        self._built = True
        return self.graph

    def add_release_boundary(self) -> dict[str, str]:
        """Append explicit, human-approved commit and publish tasks.

        These tasks are opt-in and remain pending until an operator completes
        their approval checkpoints.  No git or deployment operation is called.
        """
        self.build()
        verify_id = self._stage_ids[WorkflowStage.VERIFY.value]
        if WorkflowStage.COMMIT.value in self._stage_ids:
            return dict(self._stage_ids)
        previous = verify_id
        for stage in (WorkflowStage.COMMIT, WorkflowStage.PUBLISH):
            node = TaskNode(
                name=stage.value,
                agent_type="human_workflow",
                workflow_kind="software_engineering",
                task_payload={"stage": stage.value, "objective": self.objective},
                depends_on=[previous],
                requires_human=True,
                checkpoints=[],
                tenant_id=self.graph.tenant_id,
                engagement_id=self.graph.engagement_id,
            )
            task_id = self.graph.add_task(node)
            checkpoint = self.graph.add_checkpoint(
                task_id, f"operator approval: {stage.value}"
            )
            self.graph.tasks[task_id].task_payload["approval_checkpoint"] = checkpoint
            self._stage_ids[stage.value] = task_id
            previous = task_id
        return dict(self._stage_ids)

    def record(self, stage: str | WorkflowStage, *, success: bool,
               observation: str = "", error: str = "") -> None:
        """Record a worker's observed result; never infer success."""
        self.build()
        stage_name = stage.value if isinstance(stage, WorkflowStage) else stage
        task_id = self._stage_ids.get(stage_name)
        if not task_id:
            raise TaskGraphError(f"Unknown workflow stage '{stage_name}'")
        task = self.graph.get_task(task_id)
        assert task is not None
        if task.requires_human and not self.graph.required_checkpoints_complete(task_id):
            raise TaskGraphError(f"Stage '{stage_name}' requires operator approval")
        if task.status == TaskStatus.READY:
            self.graph.mark_running(task_id)
        if task.status != TaskStatus.RUNNING:
            raise TaskGraphError(f"Stage '{stage_name}' is {task.status}, not ready to record")
        if success:
            self.graph.mark_completed(task_id, {"observation": observation})
        else:
            self.graph.mark_failed(task_id, error or observation or "worker reported failure")

    def replan_after_failure(self, stage: str | WorkflowStage, reason: str) -> list[str]:
        """Ask the existing replan validator to add a recovery branch."""
        self.build()
        stage_name = stage.value if isinstance(stage, WorkflowStage) else stage
        failed_id = self._stage_ids.get(stage_name)
        if not failed_id:
            raise TaskGraphError(f"Unknown workflow stage '{stage_name}'")
        source = self.graph.get_task(failed_id)
        if not source or source.status not in (TaskStatus.FAILED, TaskStatus.BLOCKED):
            raise TaskGraphError("replan requires a failed or blocked stage")
        decision = ReplanDecision(
            should_replan=True,
            trigger=ReplanTrigger.AGENT_FAILURE,
            reasoning=reason,
            new_tasks=[ProposedTask(
                name=f"{stage_name} recovery",
                agent_type="human_workflow",
                workflow_kind="software_engineering",
                recovery_for=failed_id,
                task_payload={"stage": stage_name, "objective": self.objective,
                              "recovery": True},
            )],
        )
        state = CognitiveState(
            engagement_id=self.graph.engagement_id, tenant_id=self.graph.tenant_id
        )
        validation = self.replan_engine.validate_decision(decision, state, self.graph)
        if not validation.valid:
            raise TaskGraphError("; ".join(validation.errors))
        changes = self.replan_engine.apply_decision(decision, self.graph, state)
        return list(changes["tasks_added"])

    def report(self) -> WorkflowReport:
        """Return incomplete until the final required task is actually done."""
        self.build()
        tasks = list(self.graph.tasks.values())
        completed = tuple(t.name for t in tasks if t.status == TaskStatus.SUCCEEDED)
        pending = tuple(t.name for t in tasks if t.status in (
            TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.BLOCKED
        ))
        failed = tuple(t.name for t in tasks if t.status == TaskStatus.FAILED)
        approvals = tuple(t.name for t in tasks if t.requires_human and
                          not self.graph.required_checkpoints_complete(t.id))
        if failed:
            outcome = WorkflowOutcome.FAILED
        elif approvals:
            outcome = WorkflowOutcome.AWAITING_APPROVAL
        elif all(t.status == TaskStatus.SUCCEEDED for t in tasks):
            outcome = WorkflowOutcome.SUCCEEDED
        else:
            outcome = WorkflowOutcome.INCOMPLETE
        return WorkflowReport(outcome, completed, pending, failed, approvals)
