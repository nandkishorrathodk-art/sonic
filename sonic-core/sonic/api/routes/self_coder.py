"""
SONIC — Self-Coder API (observe SONIC improving its own source)
==============================================================

Exposes the autonomous self-coder so an operator can:
    * inspect the real signals the coder sees (`GET /self-coder/signals`),
    * trigger one or more improvement cycles (`POST /self-coder/run`), and
    * read the honest per-cycle report (`GET /self-coder/report`).

The coder writes only non-safety source, commits locally, and returns a summary
— precisely what the operator asked for, with no per-change approval step.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from sonic.auth.middleware import require_operator
from sonic.auth.models import User
from sonic.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_last_report: dict | None = None


def _build_coder():
    from sonic.evolution.self_coder import SelfCoder

    llm = None
    try:
        from sonic.config import CONFIGS_DIR
        from sonic.llm.router import ModelRouter

        router_ = ModelRouter.from_config(CONFIGS_DIR / "models.yaml")
        if getattr(router_, "providers", None):
            llm = router_
    except Exception as exc:
        logger.warning("self_coder_api_llm_resolve_failed", error=str(exc))
    return SelfCoder(llm=llm, max_cycles=1)


class SelfCoderRunRequest(BaseModel):
    max_cycles: int = 1
    push: bool = False


@router.get("/signals")
async def self_coder_signals(
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_operator),
):
    """List the real signals the self-coder would act on (no mutation)."""
    coder = _build_coder()
    signals = coder.discover_signals(limit=limit)
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
    """Run autonomous self-improvement cycles; returns the honest report."""
    global _last_report
    if req.max_cycles < 1 or req.max_cycles > 10:
        raise HTTPException(status_code=400, detail="max_cycles must be between 1 and 10")
    coder = _build_coder()
    coder.push = bool(req.push)
    report = await coder.run(max_cycles=req.max_cycles)
    _last_report = {
        "cycles": [
            {
                "signal": c.signal,
                "target_file": c.target_file,
                "goal_id": c.goal_id,
                "stage": c.stage,
                "reason": c.reason,
                "version_after": c.version_after,
                "commit_hash": c.commit_hash,
                "patch_files": c.patch_files,
                "lines_added": c.lines_added,
                "lines_removed": c.lines_removed,
            }
            for c in report.cycles
        ],
        "changed": report.changed,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "text": report.to_text(),
    }
    return _last_report


@router.get("/report")
async def self_coder_report(user: User = Depends(require_operator)):
    """Return the most recent self-coder report (empty if none has run yet)."""
    return _last_report or {"cycles": [], "changed": False, "text": ""}
