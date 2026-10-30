"""
SONIC-REDA — LLM Routes
===========================
Test endpoints for the Custom LLM Provider + Model Router.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from sonic.auth.middleware import require_auth
from sonic.auth.models import User
from sonic.config import CONFIGS_DIR
from sonic.llm.laya_engine import _has_sufficient_ram_gb, get_laya_engine
from sonic.llm.router import ModelRouter
from sonic.llm.schemas import ImageContent, LLMRequest, Message, MessageRole

router = APIRouter()

# FastAPI 0.115+ compatibility for route introspection on _IncludedRouter
try:
    import fastapi.routing

    if hasattr(fastapi.routing, "_IncludedRouter") and not hasattr(fastapi.routing._IncludedRouter, "path"):
        def _get_included_router_path(self):
            prefix = getattr(getattr(self, "include_context", None), "prefix", "")
            routes = getattr(getattr(self, "original_router", None), "routes", [])
            sub_paths = [getattr(r, "path", "") for r in routes]
            return f"{prefix} " + " ".join(f"{prefix}{p}" for p in sub_paths)

        fastapi.routing._IncludedRouter.path = property(_get_included_router_path)  # type: ignore[attr-defined]
except Exception:
    pass

# Lazy-loaded router singleton
_model_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    """Get or create the Model Router singleton."""
    global _model_router
    if _model_router is None:
        config_path = CONFIGS_DIR / "models.yaml"
        _model_router = ModelRouter.from_config(config_path)
    return _model_router


class ChatImageInput(BaseModel):
    """Image input for multimodal chat (base64 or URL)."""
    base64: str | None = None
    url: str | None = None
    media_type: str = "image/png"


class ChatRequest(BaseModel):
    """Simple chat request for testing.

    Supports multimodal/vision: pass `images` to send screenshots to vision models.
    """
    message: str
    images: list[ChatImageInput] = Field(default_factory=list)
    provider: str | None = None  # Explicit provider name
    model: str | None = None  # Explicit model override
    task_type: str | None = None  # Routing hint
    temperature: float = 0.7
    max_tokens: int = 2048


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: User = Depends(require_auth),
):
    """
    Send a chat message through the Model Router.
    Useful for testing provider connectivity and routing.
    """
    router_instance = get_model_router()

    if not router_instance.providers:
        raise HTTPException(
            status_code=503,
            detail="No LLM providers configured. Check your API keys in .env",
        )

    llm_request = LLMRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content=request.message,
                images=[
                    ImageContent(base64=img.base64, url=img.url, media_type=img.media_type)
                    for img in request.images
                ],
            ),
        ],
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        task_type=request.task_type,
    )

    try:
        response = await router_instance.complete(
            llm_request,
            task_type=request.task_type,
            provider_name=request.provider,
        )
        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "usage": response.usage.model_dump(),
            "latency_ms": round(response.latency_ms, 1),
            "cost_usd": round(response.cost_usd, 6),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM request failed: {str(e)}",
        )


@router.get("/providers")
async def list_providers(user: User = Depends(require_auth)):
    """List all configured LLM providers and their models."""
    router_instance = get_model_router()
    return {
        "providers": {
            name: {
                "base_url": p.base_url,
                "default_model": p.default_model,
                "capabilities": p.capabilities,
                "stats": p.get_stats(),
            }
            for name, p in router_instance.providers.items()
        },
        "routing_rules": router_instance.routing_rules,
        "default_provider": router_instance.default_provider,
    }


@router.get("/stats")
async def llm_stats(user: User = Depends(require_auth)):
    """Get LLM usage statistics and costs."""
    router_instance = get_model_router()
    return router_instance.get_stats()


# =====================================================================
# Laya System-1 Non-Autoregressive Decision Engine Endpoints
# =====================================================================

LAYA_PRESETS: dict[str, dict[str, Any]] = {
    "plane": {
        "plane": {
            "type": "choice",
            "instructions": "Select the execution plane substrate matching this objective",
            "criteria": {
                "gui": (
                    "gui visual visually screen screenshot desktop window browser click button "
                    "scroll drag ui page website open kholo dikhao ब्राउज़र स्क्रीन विंडो खोलो दिखाओ"
                ),
                "headless": (
                    "headless terminal cli command bash shell script directory mkdir author "
                    "source code file curl netstat ss port ports endpoints background git "
                    "pytest test audit server likho banao टर्मिनल कोड फाइल कमांड"
                ),
            },
        }
    },
    "receipt": {
        "empirical_proof": {
            "type": "noul",
            "instructions": "Verify empirical execution receipt",
            "affirmative_anchor": "FILE_VERIFIED verified success passed 200 OK exit 0 created committed running",
            "negative_anchor": "error traceback exception failed Terminal Ready (0 procs) __sonic_obs_ready__ exit 126 blocked",
        }
    },
    "curiosity": {
        "worth_learning": {
            "type": "noul",
            "instructions": "Evaluate observation information content",
            "affirmative_anchor": "discovered active service port file version configuration capability novel",
            "negative_anchor": "empty idle waiting nothing unchanged error",
        }
    },
}
LAYA_PRESETS["execution_plane"] = LAYA_PRESETS["plane"]
LAYA_PRESETS["verify_receipt"] = LAYA_PRESETS["receipt"]
LAYA_PRESETS["empirical_proof"] = LAYA_PRESETS["receipt"]
LAYA_PRESETS["affect"] = LAYA_PRESETS["curiosity"]
LAYA_PRESETS["worth_learning"] = LAYA_PRESETS["curiosity"]


LAYA_PRESET_ALIASES: dict[str, str] = {
    "plane": "triage",
    "execution_plane": "triage",
    "receipt": "verification",
    "verify_receipt": "verification",
    "empirical_proof": "verification",
    "safety": "guard",
    "risk": "guard",
    "affect": "curiosity",
    "novelty": "curiosity",
}


def _resolve_preset_questions(preset_key: str, state: str | dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Resolve preset questions and adapted state for Laya System-1 prediction."""
    base_questions = LAYA_PRESETS[preset_key]
    if preset_key in ("receipt", "verify_receipt", "empirical_proof") and isinstance(state, dict):
        goal = state.get("goal", "")
        evidence = state.get("evidence", "")
        actual_state = evidence or str(state)
        questions = {
            "empirical_proof": {
                "type": "noul",
                "instructions": "Verify empirical execution receipt",
                "affirmative_anchor": (
                    f"{goal} FILE_VERIFIED verified success passed 200 OK exit 0 created committed running"
                    if goal
                    else "FILE_VERIFIED verified success passed 200 OK exit 0 created committed running"
                ),
                "negative_anchor": "error traceback exception failed Terminal Ready (0 procs) __sonic_obs_ready__ exit 126 blocked",
            }
        }
        return actual_state, questions
    if preset_key in ("curiosity", "affect", "worth_learning") and isinstance(state, dict):
        obs = state.get("observation") or str(state)
        return obs, base_questions
    return state, base_questions


class LayaPredictRequest(BaseModel):
    """Request payload for Laya System-1 single-pass typed decision pass."""
    state: str | dict[str, Any] = ""
    questions: dict[str, Any] | None = None
    preset: str | None = None
    model: str | None = None


@router.get("/laya/status")
async def laya_status(user: User = Depends(require_auth)):
    """
    Get runtime status of the Laya System-1 Non-Autoregressive Decision Engine.
    Returns backend, language support, RAM safety check, and available presets.
    """
    engine = get_laya_engine()
    st: dict[str, Any] = {}
    if hasattr(engine, "status") and callable(engine.status):
        try:
            custom_status = engine.status()
            if isinstance(custom_status, dict):
                st = dict(custom_status)
        except Exception:
            st = {}

    if not st:
        try:
            ram_ok = _has_sufficient_ram_gb(1.5)
        except Exception:
            ram_ok = True

        if engine.api_url:
            backend = "laya_remote"
        elif engine.enable_local_model and ram_ok:
            backend = "laya_local"
        else:
            backend = "laya_vector_projection"

        st = {
            "backend": backend,
            "api_url_configured": bool(engine.api_url),
            "local_model_enabled": bool(engine.enable_local_model),
            "sufficient_ram": ram_ok,
            "supported_presets": ["triage", "guard", "verification", "curiosity"],
        }

    # Ensure all telemetry & alias keys are present for caller compatibility
    if "status" not in st:
        st["status"] = "ready"
    if "api_url" not in st:
        st["api_url"] = engine.api_url or None
    if "language_support" not in st:
        st["language_support"] = ["en", "hi-Deva", "hi-Latn", "multilingual"]
    if "languages" not in st:
        st["languages"] = ["en", "hi-Deva", "hi-Latn", "multilingual"]
    if "ram_check" not in st:
        st["ram_check"] = {
            "sufficient": st.get("sufficient_ram", True),
            "min_free_gb": 1.5,
        }
    if "presets" not in st:
        st["presets"] = st.get("supported_presets", ["triage", "guard", "verification", "curiosity"])
    if "has_ram_headroom" not in st:
        st["has_ram_headroom"] = bool(st.get("sufficient_ram", True))

    return st


@router.post("/laya/predict")
async def laya_predict(
    request: LayaPredictRequest,
    user: User = Depends(require_auth),
):
    """
    Execute a System-1 decision pass over `state` with structured questions or presets.
    """
    engine = get_laya_engine()
    state_val = request.state if request.state is not None else ""

    def predict_sync():
        if request.preset and not request.questions:
            preset_raw = request.preset.strip().lower()
            preset_resolved = LAYA_PRESET_ALIASES.get(preset_raw, preset_raw)

            if isinstance(state_val, dict):
                state_str = str(state_val.get("evidence") or state_val.get("command") or state_val.get("goal") or state_val)
            else:
                state_str = str(state_val)

            preset_questions_map = getattr(engine, "PRESET_QUESTIONS", {})
            if preset_resolved in preset_questions_map:
                if request.model:
                    return engine.predict(
                        state=state_str,
                        questions=preset_questions_map[preset_resolved],
                        model=request.model,
                    )
                if hasattr(engine, "predict_preset") and callable(engine.predict_preset):
                    return engine.predict_preset(state=state_str, preset=preset_resolved)
                return engine.predict(
                    state=state_str,
                    questions=preset_questions_map[preset_resolved],
                )
            if preset_raw in LAYA_PRESETS:
                actual_state, questions = _resolve_preset_questions(preset_raw, state_val)
                return engine.predict(state=actual_state, questions=questions, model=request.model)

            available = sorted(
                set(
                    list(preset_questions_map.keys())
                    + list(LAYA_PRESETS.keys())
                    + list(LAYA_PRESET_ALIASES.keys())
                )
            )
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset '{request.preset}'. Available presets: {available}",
            )
        if request.questions:
            return engine.predict(state=state_val, questions=request.questions, model=request.model)
        raise HTTPException(
            status_code=400,
            detail="Either 'questions' or 'preset' must be provided for Laya prediction.",
        )

    try:
        res = await run_in_threadpool(predict_sync)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Laya prediction failed: {str(e)}",
        )

    answers = getattr(res, "answers", None)
    if answers is None and isinstance(res, dict):
        answers = res.get("answers", {})
    routing = getattr(res, "routing", None)
    if routing is None and isinstance(res, dict):
        routing = res.get("routing", {})

    return {
        "answers": answers if answers is not None else {},
        "routing": routing if routing is not None else {},
    }
