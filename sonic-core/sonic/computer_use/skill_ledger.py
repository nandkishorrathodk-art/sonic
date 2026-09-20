"""Durable, evidence-backed computer skill memory.

This is deliberately generic: skills are learned from observed applications
and successful traces, never from a vendor-specific application map.
"""

from __future__ import annotations

import json
import re
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


class ComputerSkillLedger:
    """Persist application capabilities learned from real workstation traces."""

    def __init__(self, tenant_id: str, root: str | None = None) -> None:
        safe_tenant = re.sub(r"[^A-Za-z0-9_.-]", "_", tenant_id or "default")
        base = Path(root or "sonic_data/skills")
        self.path = base / f"{safe_tenant}.json"
        self._data: dict[str, Any] = {"applications": {}, "actions": [], "verified_skills": []}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
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

    def context(self, limit: int = 20) -> str:
        lines: list[str] = []
        for name, record in list(self._data["applications"].items())[:limit]:
            actions = record.get("successful_actions", [])
            learned = ", ".join(str(item.get("action", "")) for item in actions[-5:])
            lines.append(
                f"- {name}: observed {record.get('observations', 0)} times"
                + (f"; verified actions: {learned}" if learned else "")
            )
        return "\n".join(lines)
