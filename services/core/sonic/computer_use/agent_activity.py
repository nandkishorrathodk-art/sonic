"""Worker activity projection for the operator UI.

The Boss streams lifecycle events for every worker it launches
(``sub_dispatch`` -> ``sub_step`` / ``sub_heartbeat`` -> ``sub_report``). The
operator dashboard needs a *per-worker* view: which workers exist, each one's
stable id, whether it is actually running right now, and what it is doing.

This module is the single source of truth for that projection. It is a pure
function over the persisted worklog plus two facts the route owns
(``mission_status`` and ``mission_task_alive``), so it never invents state and
is trivially testable.

Honesty rules:

* A worker is reported ``RUNNING`` only while the owning background task is
  still alive. Once the task has exited (mission finished, crashed, or the
  control plane restarted), any worker that never produced a terminal report is
  ``INTERRUPTED`` -- never a stale ``RUNNING``.
* Status comes from the real ``sub_report`` event, not from a guess.
* ``latest_action`` / ``latest_thought`` are copied from real step traces.
"""

from __future__ import annotations

from typing import Any

# A worker in one of these states has finished (or is parked) and will not
# produce further steps for the current run.
TERMINAL_WORKER_STATUSES = frozenset(
    {
        "COMPLETED",
        "SUCCESS",
        "FAILED",
        "TIMED_OUT",
        "BLOCKED",
        "WAITING_FOR_HUMAN",
        "INTERRUPTED",
        "SKIPPED",
        "PARTIAL",
        "UNVERIFIED",
    }
)

# Statuses that mean "still working" as far as the event stream reports.
ACTIVE_WORKER_STATUSES = frozenset({"RUNNING", "PENDING", "STARTING"})

_MAX_STEPS_KEPT = 40


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _worker_key(item: dict[str, Any]) -> str | None:
    """Stable identity for a worklog item, preferring the run-unique uid."""
    uid = item.get("agent_uid") or item.get("sub_agent_uid")
    if uid:
        return f"uid:{uid}"
    agent_id = item.get("agent_id") or item.get("sub_agent_id")
    if agent_id:
        return f"agent:{agent_id}"
    mission_id = item.get("sub_mission_id")
    if mission_id:
        return f"mission:{mission_id}"
    number = item.get("sub_agent_number")
    if number is not None:
        return f"number:{number}"
    return None


def _normalize_terminal(status: str, success: Any) -> str:
    """Map a worker's reported terminal status onto a display status."""
    upper = str(status or "").upper()
    if upper in TERMINAL_WORKER_STATUSES:
        return upper
    if success is True:
        return "COMPLETED"
    if success is False:
        return "FAILED"
    return "UNVERIFIED"


def _step_summary(item: dict[str, Any]) -> str:
    action = str(item.get("action_type") or item.get("type") or "Action")
    target = (
        item.get("file")
        or item.get("command")
        or item.get("target_resource")
        or ""
    )
    label = action.replace("_", " ").lower()
    return f"{label} · {target}" if target else label


def collect_worker_activity(
    worklog: list[dict[str, Any]] | None,
    *,
    mission_task_alive: bool = False,
) -> list[dict[str, Any]]:
    """Project the worklog into an ordered list of per-worker activity records.

    ``mission_task_alive`` is the authoritative answer to "is a background
    task still driving this session?". It gates the ``RUNNING`` verdict so a
    dead task can never leave a worker displayed as running forever.
    """
    workers: dict[str, dict[str, Any]] = {}

    for item in worklog or []:
        if not isinstance(item, dict):
            continue
        event = item.get("orchestration_event")

        if event == "sub_dispatch":
            key = _worker_key(item)
            if not key:
                continue
            uid = item.get("agent_uid") or item.get("sub_agent_uid") or ""
            workers[key] = {
                "key": key,
                "agent_uid": uid,
                "agent_id": item.get("agent_id") or "",
                "agent_name": item.get("agent_name") or "",
                "sub_mission_id": item.get("sub_mission_id") or "",
                "sub_agent_number": _as_int(item.get("sub_agent_number")),
                "role": item.get("worker_role") or "subagent",
                "plane": item.get("execution_plane") or "gui",
                "goal": item.get("goal") or "",
                "raw_status": str(item.get("status") or "RUNNING").upper(),
                "success": None,
                "findings_summary": "",
                "key_discoveries": [],
                "judge_verdict": None,
                "judge_notes": None,
                "evidence_verified": None,
                "actions_taken": None,
                "duration_seconds": None,
                "latest_action": "",
                "latest_action_status": "",
                "latest_thought": "",
                "started_at": item.get("timestamp") or "",
                "updated_at": item.get("timestamp") or "",
                "step_count": 0,
                "steps": [],
            }
            continue

        if event == "sub_report":
            key = _worker_key(item)
            worker = workers.get(key) if key else None
            if worker is None:
                continue
            worker["raw_status"] = str(
                item.get("status")
                or ("COMPLETED" if item.get("success") else "FAILED")
            ).upper()
            worker["success"] = item.get("success")
            worker["findings_summary"] = item.get("findings_summary") or ""
            discoveries = item.get("key_discoveries")
            worker["key_discoveries"] = list(discoveries) if isinstance(discoveries, list) else []
            worker["judge_verdict"] = item.get("judge_verdict")
            worker["judge_notes"] = item.get("judge_notes")
            worker["evidence_verified"] = item.get("evidence_verified")
            worker["actions_taken"] = _as_int(item.get("actions_taken"))
            duration = item.get("duration_seconds")
            worker["duration_seconds"] = (
                float(duration)
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else None
            )
            if item.get("agent_name"):
                worker["agent_name"] = item["agent_name"]
            worker["updated_at"] = item.get("timestamp") or worker["updated_at"]
            continue

        # Step-level items carry the worker identity but not the orchestration
        # event marker. Attach them to the owning worker when we can resolve it.
        if item.get("type") == "thought" and item.get("sub_agent_uid"):
            key = _worker_key(item)
            worker = workers.get(key) if key else None
            if worker is not None:
                worker["latest_thought"] = item.get("content") or ""
                worker["updated_at"] = item.get("timestamp") or worker["updated_at"]
            continue

        if item.get("sub_agent_id") or item.get("sub_agent_uid"):
            key = _worker_key(item)
            worker = workers.get(key) if key else None
            if worker is None:
                continue
            summary = _step_summary(item)
            worker["latest_action"] = summary
            worker["latest_action_status"] = str(
                item.get("action_status")
                or ("FAILED" if item.get("type") == "error" else "")
            )
            worker["updated_at"] = item.get("timestamp") or worker["updated_at"]
            worker["step_count"] = _as_int(worker.get("step_count")) or 0
            worker["step_count"] += 1
            steps = worker["steps"]
            steps.append(
                {
                    "step_index": _as_int(item.get("step_index")),
                    "action_type": item.get("action_type") or "",
                    "status": item.get("action_status") or item.get("status") or "",
                    "target": item.get("target_resource") or item.get("file") or "",
                    "thought": item.get("content") if item.get("type") == "thought" else "",
                    "summary": summary,
                    "timestamp": item.get("timestamp") or "",
                }
            )
            if len(steps) > _MAX_STEPS_KEPT:
                del steps[: len(steps) - _MAX_STEPS_KEPT]

    projected: list[dict[str, Any]] = []
    for worker in workers.values():
        raw = str(worker.get("raw_status") or "").upper()
        if raw in TERMINAL_WORKER_STATUSES:
            status = _normalize_terminal(raw, worker.get("success"))
        elif mission_task_alive:
            status = "RUNNING"
        else:
            # No terminal report and no live task driving it: the worker did
            # not finish -- most likely interrupted or the control plane
            # restarted. Report that honestly instead of a stale RUNNING.
            status = "INTERRUPTED"
        worker["status"] = status
        worker["is_running"] = status == "RUNNING"
        projected.append(worker)

    return projected


def find_worker(
    worklog: list[dict[str, Any]] | None,
    worker_id: str,
    *,
    mission_task_alive: bool = False,
) -> dict[str, Any] | None:
    """Resolve a single worker by uid, agent id, sub-mission id, or number."""
    target = str(worker_id or "").strip()
    if not target:
        return None
    workers = collect_worker_activity(
        worklog,
        mission_task_alive=mission_task_alive,
    )
    for worker in workers:
        if target in {
            str(worker.get("agent_uid") or ""),
            str(worker.get("agent_id") or ""),
            str(worker.get("sub_mission_id") or ""),
        }:
            return worker
    # Allow addressing by the numeric sub-agent index too.
    if target.isdigit():
        for worker in workers:
            if worker.get("sub_agent_number") == int(target):
                return worker
    return None
