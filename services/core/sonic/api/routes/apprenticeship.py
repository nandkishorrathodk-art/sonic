"""
SONIC — AVCS Apprenticeship API (the eye learning from its own work)
====================================================================

Exposes the eye's self-learning loop so an operator can see what the being has
learned from and trigger a learning round:

    * ``GET  /avcs/apprenticeship/status``  — samples, labels, rounds, deployed accuracy
    * ``POST /avcs/apprenticeship/capture`` — sample the live desktop and store labeled frames
    * ``POST /avcs/apprenticeship/learn``   — train + verify a candidate; promote only on real gain
    * ``GET  /avcs/apprenticeship/labels``  — the label distribution it has collected

Every endpoint reports real state. A learning round that cannot improve the
deployed eye says so instead of claiming success.
"""

from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from sonic.auth.middleware import require_operator
from sonic.auth.models import User
from sonic.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _get_loop():
    """The process-wide apprenticeship loop, bound to the live provider.

    Shares one loop with the awareness monitor and the agent so a learning round
    trains on the same store the being's frames and clicks were written to.
    """
    from sonic.computer_use.apprenticeship import (
        AVCSApprenticeshipLoop,
        get_shared_apprenticeship_loop,
    )

    loop = get_shared_apprenticeship_loop()
    if loop is None:
        # The API exposes an explicit learning surface even when continuous
        # capture is off, so an operator can trigger a round on demand.
        loop = AVCSApprenticeshipLoop()
    if getattr(loop, "_perception_engine", None) is None:
        # Bind a perception engine so a promoted checkpoint is hot-reloaded into
        # the running eye instead of only existing on disk.
        try:
            from sonic.computer_use.avcs_perception import AVCSPerception

            loop._perception_engine = AVCSPerception()
        except Exception:
            pass
    return loop


class CaptureRequest(BaseModel):
    session_id: str = "default"
    samples: int = 8
    interval_s: float = 0.5


class LearnRequest(BaseModel):
    epochs: int = 6


async def _provider_and_workspace(session_id: str):
    from sonic.api.routes.workstation import get_computer

    provider = get_computer()
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="No computer provider is available; apprenticeship capture is fail-closed.",
        )
    workspace_id = await _session_workspace_id_for_default(session_id)
    if not workspace_id:
        raise HTTPException(
            status_code=409,
            detail="No workstation is provisioned for this session.",
        )
    return provider, workspace_id


async def _session_workspace_id_for_default(session_id: str) -> str:
    """Resolve a workspace id for the apprenticeship session.

    Uses the being's persistent home so apprenticeship works without an
    operator-provisioned session, which is what "learns while it works" needs.
    """
    from sonic.api.routes.workstation import get_computer

    provider = get_computer()
    if provider is None:
        return ""
    try:
        home = await provider.get_or_create_home("default")
    except Exception:
        return ""
    return getattr(home, "id", None) or getattr(home, "workspace_id", None) or ""


@router.get("/avcs/apprenticeship/status")
async def apprenticeship_status(user: User = Depends(require_operator)):
    """Current apprenticeship state: samples, labels, rounds, deployed accuracy."""
    loop = _get_loop()
    status = loop.status.to_dict()
    # The awareness monitor owns its own loop instance; the sample count is a
    # property of the shared on-disk store, so read it from the store rather
    # than this instance's possibly-stale counter.
    status["samples"] = loop.store.count()
    status["label_counts"] = loop.store.label_counts()
    status["store_dir"] = str(loop.store.root)
    status["checkpoint"] = str(loop.checkpoint_path)
    status["checkpoint_exists"] = loop.checkpoint_path.is_file()
    return status


@router.get("/avcs/apprenticeship/labels")
async def apprenticeship_labels(user: User = Depends(require_operator)):
    """Distribution of honestly-derived labels the eye has collected."""
    loop = _get_loop()
    return {
        "counts": loop.store.label_counts(),
        "samples": loop.store.count(),
        "label_rule": "freedesktop_categories_from_wm_class",
    }


@router.post("/avcs/apprenticeship/capture")
async def apprenticeship_capture(
    req: CaptureRequest, user: User = Depends(require_operator)
):
    """Sample the live desktop and store frames that carry an honest label."""
    from sonic.computer_use.apprenticeship import (
        build_provider_frame_source,
        build_provider_labeler,
    )

    loop = _get_loop()
    provider, workspace_id = await _provider_and_workspace(req.session_id)
    # Bind the labeler to the provider's own sandbox exec so it can read the
    # real .desktop metadata; a bare labeler resolves nothing and labels nothing.
    loop.labeler = build_provider_labeler(provider)
    source = build_provider_frame_source(provider, workspace_id, loop.labeler)
    stored = await loop.capture_round(
        source,
        max_samples=max(1, min(req.samples, 200)),
        interval_s=max(0.0, req.interval_s),
    )
    return {
        "stored": stored,
        "total_samples": loop.store.count(),
        "label_counts": loop.store.label_counts(),
        "note": "Frames whose window could not be honestly labeled were skipped, not guessed.",
    }


@router.post("/avcs/apprenticeship/learn")
async def apprenticeship_learn(
    req: LearnRequest, user: User = Depends(require_operator)
):
    """Train a candidate on lived experience; promote only on a real held-out gain."""
    loop = _get_loop()
    loop.epochs = max(1, min(req.epochs, 50))
    result = await loop.learn()
    return result.to_dict()


@router.get("/avcs/apprenticeship/affordance")
async def apprenticeship_affordance_status(user: User = Depends(require_operator)):
    """State of the affordance head: collected click outcomes and its training proof.

    Reports the measured outcome counts and whether the current affordance
    checkpoint has a provenance sidecar proving a real, above-chance training
    run. When it does not, the agent reports hotspots as unavailable rather than
    as detected controls.
    """
    loop = _get_loop()
    counts = loop.affordance_sample_counts()
    checkpoint = loop.store.root / "cortex_affordance.pt"
    trained = False
    provenance = None
    try:
        from forge.vision.cortex.training.affordance_provenance import (
            affordance_head_trained,
            read_affordance_provenance,
        )

        trained = bool(checkpoint.is_file()) and affordance_head_trained(checkpoint)
        provenance = read_affordance_provenance(checkpoint)
    except Exception:
        trained = False
    return {
        "outcome_samples": counts,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "affordance_head_trained": trained,
        "provenance": provenance,
        "label_rule": "post_action_screen_change",
        "note": (
            "Hotspots are reported only when a provenance sidecar proves the "
            "dorsal head was trained to an above-chance held-out accuracy."
        ),
    }


@router.post("/avcs/apprenticeship/learn-affordance")
async def apprenticeship_learn_affordance(
    req: LearnRequest, user: User = Depends(require_operator)
):
    """Fine-tune the dorsal head on accumulated click outcomes.

    Trains only the affordance head, verifies it on a held-out split grouped by
    working run, and records the measured accuracy in a provenance sidecar. The
    head is treated as usable only if that measurement clears chance.
    """
    loop = _get_loop()
    loop.epochs = max(1, min(req.epochs, 50))
    return await loop.learn_affordance()
