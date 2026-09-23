from __future__ import annotations

import asyncio

from sonic.agents.task_graph import TaskGraph, TaskNode
from sonic.computer_use.checkpoint import CheckpointManager, LocalWorkspaceFileStore


def run(awaitable):
    return asyncio.run(awaitable)


def test_checkpoint_restores_explicit_artifacts_and_state_delta(tmp_path):
    store = LocalWorkspaceFileStore(tmp_path)
    run(store.write_file("notes.txt", "before"))
    manager = CheckpointManager(store)
    checkpoint = run(manager.create(
        "before edit",
        state={"branch": "main", "dirty": False},
        artifacts=("notes.txt",),
    ))

    run(store.write_file("notes.txt", "after"))
    applied = {}
    receipt = run(manager.rollback(
        checkpoint.id,
        current_state={"branch": "feature", "dirty": True},
        apply_state=lambda state: applied.update(state),
    ))

    assert receipt.status.value == "SUCCEEDED"
    assert receipt.state_delta.changed["branch"] == "main"
    assert receipt.state_delta.changed["dirty"] is False
    assert run(store.read_file("notes.txt")) == b"before"
    assert applied == {"branch": "main", "dirty": False}
    assert "restored checkpoint" in receipt.evidence


def test_missing_or_tampered_checkpoint_fails_closed(tmp_path):
    store = LocalWorkspaceFileStore(tmp_path)
    manager = CheckpointManager(store)
    missing = run(manager.rollback("not-known", current_state={"value": 1}))
    assert missing.status.value == "BLOCKED"
    assert "unknown" in missing.evidence

    checkpoint = run(manager.create("safe", state={"value": 1}))
    manager._checkpoints[checkpoint.id] = checkpoint.__class__(
        id=checkpoint.id,
        name=checkpoint.name,
        state={"value": 999},
        artifacts=checkpoint.artifacts,
        manifest_digest=checkpoint.manifest_digest,
    )
    corrupt = run(manager.rollback(checkpoint.id, current_state={"value": 2}))
    assert corrupt.status.value == "BLOCKED"
    assert "corrupt" in corrupt.evidence


def test_checkpoint_rejects_escape_and_uses_task_graph_checkpoint(tmp_path):
    store = LocalWorkspaceFileStore(tmp_path)
    graph = TaskGraph("engagement", "tenant")
    task = TaskNode(name="human step", agent_type="human", tenant_id="tenant")
    task_id = graph.add_task(task)
    manager = CheckpointManager(store, task_graph=graph)

    try:
        run(manager.create("unsafe", state={}, artifacts=("../outside",)))
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe artifact path was accepted")

    checkpoint = run(manager.create(
        "approved pause",
        state={"phase": "inspect"},
        task_id=task_id,
    ))
    assert checkpoint.task_id == task_id
    assert len(graph.get_checkpoints(task_id)) == 1
    assert graph.get_checkpoints(task_id)[0].status.value == "completed"
