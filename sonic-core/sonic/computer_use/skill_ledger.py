"""Durable, evidence-backed computer skill and workflow memory.

This is deliberately generic: skills are learned from observed applications
and successful traces, never from a vendor-specific application map. Workflow
records remain candidates until a replay produces runtime evidence.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonic.computer_use.gui_world_graph import GuiWorldGraph
from sonic.computer_use.models import ActionExecutionStatus, ComputerDecisionTrace


@dataclass(frozen=True)
class VerifiedSkill:
    """A state-conditioned action learned from a real successful trace."""

    skill_id: str
    action_type: str
    target_resource: str
    pre_graph_fingerprint: str
    post_graph_fingerprint: str
    post_graph_delta: str
    evidence: str

    def can_apply(self, graph: GuiWorldGraph) -> bool:
        """Allow reuse only when the live graph matches the learned pre-state."""
        return bool(self.pre_graph_fingerprint) and graph.fingerprint() == self.pre_graph_fingerprint

    def postcondition_holds(self, graph: GuiWorldGraph) -> bool:
        """Verify the learned post-state against a fresh observation."""
        return bool(self.post_graph_fingerprint) and graph.fingerprint() == self.post_graph_fingerprint


@dataclass(frozen=True)
class WorkflowRecord:
    """A replayable workflow whose lifecycle is grounded in observed runs."""

    workflow_id: str
    steps: tuple[dict[str, Any], ...]
    observed_at: str
    replay_attempts: int = 0
    successful_replays: int = 0
    failed_replays: int = 0
    last_replay: dict[str, Any] | None = None
    promoted: bool = False


class ComputerSkillLedger:
    """Persist application capabilities learned from real workstation traces."""

    def __init__(self, tenant_id: str, root: str | None = None) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip() or tenant_id != tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty canonical tenant identifier")
        self.tenant_id = tenant_id
        safe_tenant = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        base = Path(root or "sonic_data/skills")
        self.path = base / f"{safe_tenant}.json"
        self._data: dict[str, Any] = {
            "tenant_id": tenant_id,
            "applications": {},
            "actions": [],
            "verified_skills": [],
            "workflows": {},
        }
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            if raw.get("tenant_id") not in (None, self.tenant_id):
                raise ValueError("skill ledger tenant identity does not match its filename")
            if raw.get("tenant_id") is None and any(raw.get(key) for key in ("applications", "actions", "verified_skills", "workflows")):
                return
            self._data.update(raw)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def observe_applications(self, applications: list[str]) -> None:
        changed = False
        for app in applications:
            name = str(app).strip()
            if not name:
                continue
            record = self._data["applications"].setdefault(
                name, {"observations": 0, "successful_actions": [], "last_seen": ""}
            )
            record["observations"] = int(record.get("observations", 0)) + 1
            record["last_seen"] = datetime.now(UTC).isoformat()
            changed = True
        if changed:
            self._save()

    def record_success(self, application: str, action: str, evidence: str) -> None:
        app = str(application or "").strip()
        act = str(action or "").strip()
        if not app or not act or not evidence.strip():
            return
        record = self._data["applications"].setdefault(
            app, {"observations": 0, "successful_actions": [], "last_seen": ""}
        )
        actions = record.setdefault("successful_actions", [])
        item = {"action": act[:160], "evidence": evidence[:500]}
        if item not in actions:
            actions.append(item)
            actions[:] = actions[-20:]
            self._save()

    def compile_verified_skill(
        self,
        trace: ComputerDecisionTrace,
        before_graph: GuiWorldGraph,
        after_graph: GuiWorldGraph,
    ) -> VerifiedSkill | None:
        """Compile a trace only when execution and observed evidence agree."""
        status = getattr(trace.status, "value", str(trace.status))
        if status not in {
            ActionExecutionStatus.SUCCESS.value,
            ActionExecutionStatus.SUCCEEDED.value,
            ActionExecutionStatus.COMPLETED.value,
            ActionExecutionStatus.VERIFIED.value,
        }:
            return None
        if not trace.actual_observation.strip() or (
            trace.prediction_error is None or trace.prediction_error > 0.0
        ):
            return None
        skill = VerifiedSkill(
            skill_id=f"skill-{trace.action_id}",
            action_type=trace.action_type.value,
            target_resource=trace.target_resource,
            pre_graph_fingerprint=before_graph.fingerprint(),
            post_graph_fingerprint=after_graph.fingerprint(),
            post_graph_delta=trace.graph_delta,
            evidence=trace.actual_observation[:500],
        )
        encoded = asdict(skill)
        skills = self._data.setdefault("verified_skills", [])
        if not any(item.get("skill_id") == skill.skill_id for item in skills):
            skills.append(encoded)
            skills[:] = skills[-100:]
            self._save()
        return skill

    def verified_skills(self) -> list[VerifiedSkill]:
        return [VerifiedSkill(**item) for item in self._data.get("verified_skills", []) if isinstance(item, dict)]

    def record_observed_workflow(
        self,
        workflow_id: str,
        steps: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowRecord | None:
        """Store a workflow only when it contains concrete observed steps.

        This is intentionally a recording operation, not a promotion operation:
        an observed sequence is merely a candidate until a later replay proves
        its postconditions.
        """
        identifier = str(workflow_id or "").strip()
        observed_steps = [
            dict(step)
            for step in steps
            if isinstance(step, dict)
            and str(step.get("action_type") or step.get("action") or "").strip()
            and str(step.get("observation") or step.get("evidence") or "").strip()
        ]
        if not identifier or not observed_steps:
            return None
        workflows = self._data.setdefault("workflows", {})
        existing = workflows.get(identifier, {})
        record = {
            "workflow_id": identifier,
            "steps": observed_steps,
            "observed_at": str(existing.get("observed_at") or datetime.now(UTC).isoformat()),
            "replay_attempts": int(existing.get("replay_attempts", 0)),
            "successful_replays": int(existing.get("successful_replays", 0)),
            "failed_replays": int(existing.get("failed_replays", 0)),
            "last_replay": existing.get("last_replay"),
            "promoted": bool(existing.get("promoted", False)),
        }
        if metadata:
            record["observation_metadata"] = dict(metadata)
        elif "observation_metadata" in existing:
            record["observation_metadata"] = existing["observation_metadata"]
        workflows[identifier] = record
        self._save()
        return self._workflow_record(record)

    def verify_workflow(
        self,
        workflow_id: str,
        replay: dict[str, Any],
        *,
        independent_verifier: Callable[
            [str, tuple[dict[str, Any], ...], dict[str, Any]], bool
        ] | None = None,
    ) -> WorkflowRecord | None:
        """Record one replay, accepting success only with empirical evidence."""
        identifier = str(workflow_id or "").strip()
        workflows = self._data.setdefault("workflows", {})
        record = workflows.get(identifier)
        if not record or not isinstance(replay, dict):
            return None
        status = str(replay.get("status", "")).upper()
        evidence = str(replay.get("evidence") or replay.get("observation") or "").strip()
        metadata = replay.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        # A caller's success flag is not evidence. Require a successful runtime
        # status, non-empty output, and replay metadata identifying the run.
        successful = (
            status in {"SUCCESS", "SUCCEEDED", "COMPLETED", "VERIFIED"}
            and bool(evidence)
            and replay.get("exit_code") == 0
            and bool(str(metadata.get("replay_id") or metadata.get("trace_id") or "").strip())
            and callable(independent_verifier)
            and independent_verifier(identifier, tuple(record.get("steps", [])), replay) is True
        )
        record["replay_attempts"] = int(record.get("replay_attempts", 0)) + 1
        if successful:
            record["successful_replays"] = int(record.get("successful_replays", 0)) + 1
        else:
            record["failed_replays"] = int(record.get("failed_replays", 0)) + 1
        record["last_replay"] = {
            "status": status,
            "evidence": evidence[:500],
            "metadata": dict(metadata),
            "verified": successful,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        self._save()
        return self._workflow_record(record)

    def promote_workflow(self, workflow_id: str) -> WorkflowRecord | None:
        """Promote only a workflow with at least one successful replay."""
        identifier = str(workflow_id or "").strip()
        record = self._data.setdefault("workflows", {}).get(identifier)
        if not record or int(record.get("successful_replays", 0)) < 1:
            return None
        record["promoted"] = True
        self._save()
        return self._workflow_record(record)

    def workflows(self, *, promoted_only: bool = False) -> list[WorkflowRecord]:
        records = [
            self._workflow_record(item)
            for item in self._data.setdefault("workflows", {}).values()
            if isinstance(item, dict)
            and (not promoted_only or bool(item.get("promoted", False)))
        ]
        return records

    @staticmethod
    def _workflow_record(record: dict[str, Any]) -> WorkflowRecord:
        return WorkflowRecord(
            workflow_id=str(record["workflow_id"]),
            steps=tuple(dict(step) for step in record.get("steps", [])),
            observed_at=str(record.get("observed_at", "")),
            replay_attempts=int(record.get("replay_attempts", 0)),
            successful_replays=int(record.get("successful_replays", 0)),
            failed_replays=int(record.get("failed_replays", 0)),
            last_replay=record.get("last_replay"),
            promoted=bool(record.get("promoted", False)),
        )

    def context(self, limit: int = 20) -> str:
        lines: list[str] = []
        for name, record in list(self._data["applications"].items())[:limit]:
            actions = record.get("successful_actions", [])
            learned = ", ".join(str(item.get("action", "")) for item in actions[-5:])
            lines.append(
                f"- {name}: observed {record.get('observations', 0)} times"
                + (f"; verified actions: {learned}" if learned else "")
            )
        for workflow in self.workflows(promoted_only=True)[:limit]:
            lines.append(
                f"- workflow {workflow.workflow_id}: promoted after "
                f"{workflow.successful_replays} successful replay(s)"
            )
        return "\n".join(lines)
