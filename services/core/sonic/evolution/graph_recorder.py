"""Write verified self-evolution outcomes to persistent graph memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonic.logger import get_logger

logger = get_logger(__name__)


async def record_verified_candidate(
    candidate: dict[str, Any],
    repo_root: str | Path,
) -> str | None:
    """Persist a verified candidate outcome only after active-worktree checks pass."""
    status = candidate.get("status")
    graph_status = {
        "applied": "verified_applied",
        "promoted": "verified_promoted",
        "committed_metadata_failed": "verified_committed_metadata_failed",
    }.get(status)
    if graph_status is None:
        return None

    verification = candidate.get("verification")
    active = verification.get("active_worktree") if isinstance(verification, dict) else None
    if not isinstance(verification, dict) or not isinstance(active, dict):
        raise ValueError("Applied candidate is missing verification receipts")

    required_checks = (
        verification.get("full_test_suite") is True,
        verification.get("security_regression") is True,
        active.get("full_test_suite") is True,
        active.get("security_regression") is True,
    )
    if not all(required_checks):
        raise ValueError("Applied candidate does not have complete passing verification")

    if status in {"promoted", "committed_metadata_failed"} and not candidate.get("commit"):
        raise ValueError("Committed candidate is missing its commit hash")
    if status == "promoted" and (
        not candidate.get("version_after") or candidate.get("journal_recorded") is not True
    ):
        raise ValueError("Promoted candidate is missing version or journal receipts")

    candidate_id = candidate.get("candidate_id")
    tenant_id = candidate.get("tenant_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Applied candidate is missing its candidate ID")
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("Applied candidate is missing its tenant ID")

    from sonic.memory.router import get_smart_memory

    memory = await get_smart_memory(
        sqlite_db_path=str(Path(repo_root).resolve() / "sonic_data.db")
    )
    add_node = getattr(memory, "add_node", None)
    if add_node is None:
        raise RuntimeError("Graph memory backend does not support generic nodes")

    node = {
        "uid": f"self-evolution-{candidate_id}",
        "node_type": "SelfEvolutionCandidate",
        "tenant_id": tenant_id,
        "candidate_id": candidate_id,
        "status": graph_status,
        "repository": str(Path(repo_root).resolve()),
        "base_branch": candidate.get("base_branch", ""),
        "base_commit": candidate.get("base_commit", ""),
        "commit": candidate.get("commit", ""),
        "commit_branch": candidate.get("commit_branch", ""),
        "version_before": candidate.get("version_before", ""),
        "version_after": candidate.get("version_after", ""),
        "journal_recorded": candidate.get("journal_recorded", False),
        "promotion_error": candidate.get("promotion_error", ""),
        "target_file": candidate.get("target_file", ""),
        "description": candidate.get("description", ""),
        "diagnosis": candidate.get("diagnosis", ""),
        "impact": (
            candidate.get("impact", {}).get("assessment", "")
            if isinstance(candidate.get("impact"), dict)
            else str(candidate.get("impact", ""))
        ),
        "files": candidate.get("files", []),
        "full_test_suite": True,
        "security_regression": True,
        "active_worktree_full_test_suite": True,
        "active_worktree_security_regression": True,
        "test_count": verification.get("test_count", 0),
        "passed_tests": verification.get("passed_tests", 0),
        "failed_tests": verification.get("failed_tests", 0),
        "import_check": json.dumps(verification.get("import_check", "")),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    uid = await add_node("SelfEvolutionCandidate", node)
    if not uid:
        raise RuntimeError("Graph memory did not persist the verified candidate")

    logger.info(
        "self_evolution_verified_candidate_recorded",
        candidate_id=candidate_id,
        tenant_id=tenant_id,
        graph_uid=uid,
    )
    return uid
