"""Human-approved, fail-closed checkpoints for workflow state and artifacts.

Checkpoints contain only the state mappings and artifact paths explicitly
provided by the caller.  They never walk a workspace and rollback never
deletes anything other than an explicitly snapshotted path.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sonic.agents.task_graph import TaskGraph
from sonic.computer_use.action_contracts import ActionReceipt, StateDelta
from sonic.computer_use.models import ActionExecutionStatus


class WorkspaceFileStore(Protocol):
    async def read_file(self, path: str) -> bytes | str: ...
    async def write_file(self, path: str, content: bytes | str) -> bool: ...
    async def delete_file(self, path: str) -> bool: ...


class LocalWorkspaceFileStore:
    """A confined local file store used by tests and local workflows."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("artifact path must be non-empty")
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("artifact path must be workspace-relative")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact path escapes workspace") from exc
        return resolved

    async def read_file(self, path: str) -> bytes:
        return await asyncio.to_thread(self._path(path).read_bytes)

    async def write_file(self, path: str, content: bytes | str) -> bool:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        await asyncio.to_thread(target.write_bytes, data)
        return True

    async def delete_file(self, path: str) -> bool:
        target = self._path(path)
        if not target.exists():
            return True
        if target.is_dir():
            raise ValueError("rollback refuses to delete directories")
        await asyncio.to_thread(target.unlink)
        return True


@dataclass(frozen=True)
class ArtifactSnapshot:
    path: str
    content: bytes
    digest: str

    @classmethod
    def from_content(cls, path: str, content: bytes | str) -> ArtifactSnapshot:
        raw = content.encode() if isinstance(content, str) else bytes(content)
        return cls(path, raw, hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class Checkpoint:
    id: str
    name: str
    state: dict[str, Any]
    artifacts: tuple[ArtifactSnapshot, ...]
    manifest_digest: str
    task_id: str | None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CheckpointManager:
    """Create and restore named checkpoints without implicit filesystem scope."""

    def __init__(
        self,
        store: WorkspaceFileStore,
        *,
        task_graph: TaskGraph | None = None,
    ) -> None:
        self.store = store
        self.task_graph = task_graph
        self._checkpoints: dict[str, Checkpoint] = {}

    async def create(
        self,
        name: str,
        *,
        state: Mapping[str, Any],
        artifacts: Mapping[str, bytes | str] | tuple[str, ...] = (),
        task_id: str | None = None,
    ) -> Checkpoint:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("checkpoint name must be non-empty")
        if task_id and (not self.task_graph or not self.task_graph.get_task(task_id)):
            raise ValueError("task_id must identify a task in the supplied graph")
        try:
            state_copy = json.loads(_canonical(dict(state)))
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint state must be JSON-serializable") from exc

        paths = list(artifacts.keys()) if isinstance(artifacts, Mapping) else list(artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("checkpoint artifact paths must be unique")
        snapshots: list[ArtifactSnapshot] = []
        for path in paths:
            self._validate_artifact_path(path)
            if isinstance(artifacts, Mapping):
                content = artifacts[path]
            else:
                content = await self.store.read_file(path)
            snapshots.append(ArtifactSnapshot.from_content(path, content))
        manifest = {
            "name": name.strip(),
            "state": state_copy,
            "artifacts": [(a.path, a.digest) for a in snapshots],
            "task_id": task_id,
        }
        checkpoint = Checkpoint(
            id=f"checkpoint-{uuid.uuid4().hex}",
            name=name.strip(),
            state=state_copy,
            artifacts=tuple(snapshots),
            manifest_digest=hashlib.sha256(_canonical(manifest).encode()).hexdigest(),
            task_id=task_id,
        )
        self._checkpoints[checkpoint.id] = checkpoint
        if task_id and self.task_graph:
            graph_checkpoint = self.task_graph.add_checkpoint(task_id, checkpoint.name)
            self.task_graph.record_checkpoint(
                task_id,
                graph_checkpoint,
                state_delta=dict(state_copy),
                note=f"checkpoint:{checkpoint.id}",
            )
        return checkpoint

    def get(self, checkpoint_id: str) -> Checkpoint | None:
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> tuple[Checkpoint, ...]:
        """Return checkpoints in creation order without exposing mutable state."""
        return tuple(self._checkpoints.values())

    def _valid(self, checkpoint: Checkpoint) -> bool:
        try:
            manifest = {
                "name": checkpoint.name,
                "state": checkpoint.state,
                "artifacts": [(a.path, a.digest) for a in checkpoint.artifacts],
                "task_id": checkpoint.task_id,
            }
            if hashlib.sha256(_canonical(manifest).encode()).hexdigest() != checkpoint.manifest_digest:
                return False
            return all(
                hashlib.sha256(a.content).hexdigest() == a.digest
                and self._is_safe_artifact_path(a.path)
                for a in checkpoint.artifacts
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_safe_artifact_path(path: str) -> bool:
        return (
            isinstance(path, str)
            and bool(path.strip())
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
        )

    @classmethod
    def _validate_artifact_path(cls, path: str) -> None:
        if not cls._is_safe_artifact_path(path):
            raise ValueError("artifact path must be a safe workspace-relative path")

    async def rollback(
        self,
        checkpoint_id: str,
        *,
        current_state: Mapping[str, Any],
        apply_state: Callable[[dict[str, Any]], Any] | None = None,
        read_state: Callable[[], Mapping[str, Any] | Any] | None = None,
    ) -> ActionReceipt:
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return self._failed_receipt("unknown checkpoint", current_state)
        if not self._valid(checkpoint):
            return self._failed_receipt("checkpoint is corrupt", current_state)
        before_state = dict(current_state)
        if checkpoint.state != before_state and apply_state is None:
            return self._failed_receipt(
                "checkpoint state differs but no state applier was provided",
                before_state,
            )

        original_artifacts: dict[str, bytes | str] = {}
        restored_paths: list[str] = []
        state_attempted = False
        try:
            for artifact in checkpoint.artifacts:
                if not isinstance(artifact.content, bytes):
                    raise ValueError("checkpoint artifact content is invalid")
                original = await self.store.read_file(artifact.path)
                if not isinstance(original, (bytes, str)):
                    raise TypeError(f"workspace returned invalid content for {artifact.path}")
                original_artifacts[artifact.path] = original

            for artifact in checkpoint.artifacts:
                if not await self.store.write_file(artifact.path, artifact.content):
                    raise OSError(f"workspace refused restore: {artifact.path}")
                restored_paths.append(artifact.path)
                restored = await self.store.read_file(artifact.path)
                if not isinstance(restored, (bytes, str)):
                    raise TypeError(f"workspace returned invalid read-back for {artifact.path}")
                restored_bytes = restored.encode() if isinstance(restored, str) else restored
                if hashlib.sha256(restored_bytes).hexdigest() != artifact.digest:
                    raise OSError(f"workspace read-back did not match checkpoint: {artifact.path}")

            if apply_state is not None:
                state_attempted = True
                result = apply_state(dict(checkpoint.state))
                if inspect.isawaitable(result):
                    result = await result
                observed_state = read_state() if read_state is not None else result
                if inspect.isawaitable(observed_state):
                    observed_state = await observed_state
                if not isinstance(observed_state, Mapping):
                    raise TypeError("state rollback requires a readable state mapping")
                if dict(observed_state) != checkpoint.state:
                    raise OSError("state read-back did not match checkpoint")

            delta = StateDelta.from_states(dict(current_state), checkpoint.state)
            evidence = f"restored checkpoint {checkpoint.id}; artifacts={len(checkpoint.artifacts)}"
            return ActionReceipt(
                action_type="ROLLBACK",
                status=ActionExecutionStatus.SUCCEEDED,
                state_delta=delta,
                before=dict(current_state),
                after=dict(checkpoint.state),
                evidence=evidence,
            )
        except Exception as exc:
            compensation_errors: list[str] = []
            for path in reversed(restored_paths):
                try:
                    if not await self.store.write_file(path, original_artifacts[path]):
                        raise OSError("workspace refused compensation")
                    restored = await self.store.read_file(path)
                    restored_bytes = restored.encode() if isinstance(restored, str) else restored
                    original = original_artifacts[path]
                    original_bytes = original.encode() if isinstance(original, str) else original
                    if restored_bytes != original_bytes:
                        raise OSError("compensation read-back mismatch")
                except Exception as compensation_exc:
                    compensation_errors.append(
                        f"{path}: {type(compensation_exc).__name__}: {compensation_exc}"
                    )

            if state_attempted and apply_state is not None:
                try:
                    result = apply_state(before_state)
                    if inspect.isawaitable(result):
                        result = await result
                    observed_state = read_state() if read_state is not None else result
                    if inspect.isawaitable(observed_state):
                        observed_state = await observed_state
                    if not isinstance(observed_state, Mapping) or dict(observed_state) != before_state:
                        raise OSError("state compensation read-back mismatch")
                except Exception as compensation_exc:
                    compensation_errors.append(
                        f"state: {type(compensation_exc).__name__}: {compensation_exc}"
                    )

            reason = f"rollback failed: {type(exc).__name__}: {exc}"
            if compensation_errors:
                reason += "; compensation incomplete: " + "; ".join(compensation_errors)
            else:
                reason += "; original state restored"
            return self._failed_receipt(reason, before_state)

    @staticmethod
    def _failed_receipt(reason: str, state: Mapping[str, Any]) -> ActionReceipt:
        snapshot = dict(state)
        return ActionReceipt(
            action_type="ROLLBACK",
            status=ActionExecutionStatus.BLOCKED if reason in {"unknown checkpoint", "checkpoint is corrupt"} else ActionExecutionStatus.FAILED,
            state_delta=StateDelta(),
            before=snapshot,
            after=snapshot,
            evidence=reason,
        )
