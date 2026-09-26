"""
SONIC — Self-Coder API (observe SONIC improving its own source)
==============================================================

Exposes isolated self-coder candidates so an operator can:
    * inspect the real signals the coder sees (`GET /self-coder/signals`),
    * generate and test proposals in a detached Git worktree,
    * approve an exact proposal for local commit on a dedicated branch, and
    * reject or clean up unapproved candidates.

Candidate generation never mutates the active checkout, commits, or pushes.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from sonic.auth.middleware import require_operator
from sonic.auth.models import User
from sonic.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_last_reports_by_tenant: dict[str, dict] = {}
_configured_evolution_engine = None


def configure_evolution_engine(engine) -> None:
    """Use the application EvolutionEngine's capability/LLM configuration."""
    global _configured_evolution_engine
    _configured_evolution_engine = engine


def _build_coder(repo_root: Path | str | None = None, *, require_llm: bool = False):
    """Build a coder whose goal director/evolver share the configured engine."""
    from sonic.evolution.codebase_evolver import CodebaseEvolver
    from sonic.evolution.engine import EvolutionEngine
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.pipeline import EvolutionPipeline
    from sonic.evolution.self_coder import SelfCoder
    from sonic.evolution.version_tracker import VersionTracker

    template = _configured_evolution_engine
    llm = getattr(template, "llm_client", None)
    if template is None:
        from sonic.config import CONFIGS_DIR
        from sonic.llm.router import ModelRouter

        try:
            llm = ModelRouter.from_config(CONFIGS_DIR / "models.yaml")
        except (OSError, ValueError, ImportError) as exc:
            logger.error("self_coder_api_llm_resolve_failed", error=str(exc))
            if require_llm:
                raise RuntimeError(f"Self-coder LLM configuration failed: {exc}") from exc
    if require_llm and not getattr(llm, "providers", None):
        raise RuntimeError("Self-coder requires at least one configured LLM provider")

    root = Path(repo_root).resolve() if repo_root else CodebaseEvolver().repo_root
    db_path = root / "sonic_data" / "self_coder_evolution.db"
    pipeline = EvolutionPipeline()
    evolver = CodebaseEvolver(repo_root=root, pipeline=pipeline)
    versions = VersionTracker(db_path=str(db_path), repo_root=root)
    journal = EvolutionJournal(db_path=str(db_path), repo_root=root)
    director = EvolutionGoalDirector(
        repo_root=root,
        db_path=str(db_path),
        evolver=evolver,
        version_tracker=versions,
        journal=journal,
        llm_client=llm,
    )
    engine = EvolutionEngine(
        pipeline=pipeline,
        strategy_engine=getattr(template, "strategy_engine", None),
        method_lab=getattr(template, "method_lab", None),
        toolsmith=getattr(template, "toolsmith", None),
        lessons_ledger=getattr(template, "lessons_ledger", None),
        codebase_evolver=evolver,
        version_tracker=versions,
        journal=journal,
        goal_director=director,
        llm_client=llm,
        queue_self_update_goal=False,
    )
    return SelfCoder(
        repo_root=root,
        llm=engine.llm_client,
        evolver=engine.codebase_evolver,
        director=engine.goal_director,
        evolution_engine=engine,
        max_cycles=1,
        allow_writes=False,
        push=False,
    )


def _candidate_store():
    from sonic.evolution.codebase_evolver import CodebaseEvolver
    from sonic.evolution.self_coder_candidates import SelfCoderCandidateStore

    root = CodebaseEvolver().repo_root
    storage = os.environ.get("SONIC_SELF_CODER_CANDIDATE_DIR")
    return SelfCoderCandidateStore(root, storage_dir=storage)


class SelfCoderRunRequest(BaseModel):
    max_cycles: int = 1
    push: bool = False


class SelfCoderRejectRequest(BaseModel):
    reason: str = ""


def _tenant_id(user: User) -> str:
    return user.tenant_id or "default"


@router.get("/audit")
async def self_coder_audit(
    run_tests: bool = Query(False),
    limit: int = Query(60, ge=1, le=200),
    user: User = Depends(require_operator),
):
    """Read-only codebase audit: health, static defects, optionally the tests."""
    from sonic.evolution.codebase_auditor import CodebaseAuditor
    from sonic.evolution.codebase_report import narrate

    auditor = CodebaseAuditor(max_findings=limit)
    report = await asyncio.to_thread(auditor.audit, run_tests=run_tests)
    return {
        "ok": report.ok,
        "files_scanned": report.files_scanned,
        "counts": report.counts(),
        "health": report.health,
        "tests": report.tests,
        "findings": [f.__dict__ for f in report.sorted_findings()],
        "narrative": narrate(report),
        "text": report.to_text(),
    }


@router.get("/signals")
async def self_coder_signals(
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_operator),
):
    """List the real signals the self-coder would act on (no mutation)."""
    coder = _build_coder(require_llm=False)
    signals = await asyncio.to_thread(coder.discover_signals, limit=limit)
    return {
        "count": len(signals),
        "signals": [
            {"signal": s, "target_file": f, "category": c} for s, f, c in signals
        ],
    }


@router.post("/run")
async def self_coder_run(
    req: SelfCoderRunRequest = SelfCoderRunRequest(),
    user: User = Depends(require_operator),
):
    """Generate, verify, and locally commit candidates on isolated branches."""
    if req.push:
        raise HTTPException(status_code=400, detail="Self-coder candidates never push to a remote")
    if req.max_cycles < 1 or req.max_cycles > 10:
        raise HTTPException(status_code=400, detail="max_cycles must be between 1 and 10")
    try:
        store = _candidate_store()

        def run_isolated():
            async def run():
                return await store.run_candidate(
                    _tenant_id(user),
                    req.max_cycles,
                    lambda root: _build_coder(root, require_llm=True),
                )
            return asyncio.run(run())

        result = await asyncio.to_thread(run_isolated)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.error("self_coder_candidate_run_failed", tenant_id=_tenant_id(user), error=str(exc))
        raise HTTPException(status_code=500, detail=f"Candidate run failed: {exc}") from exc
    report_dict = {
        "candidates": result["candidates"],
        "changed": any(
            item.get("status") == "auto_promoted"
            for item in result["candidates"]
        ),
        "active_branch_changed": False,
        "text": result["report"],
    }
    tenant_id = _tenant_id(user)
    _last_reports_by_tenant[tenant_id] = report_dict
    return report_dict


@router.get("/report")
async def self_coder_report(user: User = Depends(require_operator)):
    """Return the most recent tenant-scoped candidate-run report."""
    tenant_id = _tenant_id(user)
    return _last_reports_by_tenant.get(tenant_id) or {"cycles": [], "changed": False, "text": ""}


@router.get("/status")
async def self_coder_status(user: User = Depends(require_operator)):
    """Report independent background self-coding and optional curiosity status."""
    from sonic.being.life_loop import get_life_loop_status
    from sonic.evolution.self_coder_config import get_self_coder_config
    from sonic.evolution.self_coder_loop import get_self_coder_loop_status

    try:
        config = get_self_coder_config()
    except ValueError as exc:
        logger.error("self_coder_configuration_invalid", error=str(exc))
        return {"enabled": False, "reason": str(exc), "configuration_error": True}
    loop = get_life_loop_status()
    coder_loop = get_self_coder_loop_status()
    if not config.enabled:
        reason = "SONIC_SELF_CODER_ENABLED is false"
    elif not coder_loop["running"]:
        reason = "Background self-coder is not running; inspect backend logs"
    else:
        reason = f"Runs every {config.interval_seconds} seconds"
    return {
        "enabled": config.enabled and coder_loop["running"],
        "reason": reason,
        "configured_enabled": config.enabled,
        "interval_seconds": config.interval_seconds,
        "every_ticks": config.every_ticks,
        "background_loop": coder_loop,
        "being_life_loop": loop,
    }


@router.get("/candidates")
async def self_coder_candidates(user: User = Depends(require_operator)):
    return {"candidates": _candidate_store().list(_tenant_id(user))}


@router.get("/candidates/{candidate_id}")
async def self_coder_candidate(candidate_id: str, user: User = Depends(require_operator)):
    record = _candidate_store().get(candidate_id, _tenant_id(user))
    if record is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return record


@router.post("/candidates/{candidate_id}/approve")
async def self_coder_approve_candidate(candidate_id: str, user: User = Depends(require_operator)):
    try:
        return await asyncio.to_thread(
            _candidate_store().approve, candidate_id, _tenant_id(user)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("self_coder_candidate_approval_failed", candidate_id=candidate_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Candidate approval failed: {exc}") from exc


@router.post("/candidates/{candidate_id}/reject")
async def self_coder_reject_candidate(
    candidate_id: str,
    req: SelfCoderRejectRequest,
    user: User = Depends(require_operator),
):
    try:
        return _candidate_store().reject(candidate_id, _tenant_id(user), req.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/candidates/{candidate_id}")
async def self_coder_cleanup_candidate(candidate_id: str, user: User = Depends(require_operator)):
    try:
        _candidate_store().cleanup(candidate_id, _tenant_id(user))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"candidate_id": candidate_id, "cleaned": True}
