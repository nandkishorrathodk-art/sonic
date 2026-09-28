"""
SONIC-REDA — FastAPI Application Entry Point
================================================
Main API server with auth middleware, CORS, and route mounting.

Run:
    uvicorn sonic.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from sonic import __codename__, __version__
from sonic.api.routes import (
    agents,
    auth,
    checkpoints,
    engagements,
    experiments,
    findings,
    graph,
    health,
    jobs,
    live,
    llm,
    organization,
    security,
    self_coder,
    terminal,
    workstation,
)
from sonic.config import get_settings
from sonic.logger import get_logger
from sonic.memory.graph import get_graph_memory
from sonic.safety.scope import get_scope_checker

logger = get_logger(__name__)


async def _maybe_start_being_life_loop(settings):
    """Re-attach the persistent being and start its always-on curiosity loop.

    Returns the running BeingLifeLoop, or None if it could not start (no home
    workspace, no safety policy, or live infra unavailable). Never raises — a
    missing being loop must not block API boot.
    """
    import os

    # Only run the always-on loop when explicitly enabled. Default off so test
    # boots and headless dev don't spawn a background actor that needs a sandbox.
    if os.environ.get("SONIC_ENABLE_BEING_LIFE_LOOP") != "1":
        logger.info(
            "being_life_loop_disabled",
            reason="SONIC_ENABLE_BEING_LIFE_LOOP!=1",
        )
        return None
    try:
        from sonic.being.identity import get_being_store, get_or_create_being
        from sonic.being.life_loop import BeingLifeLoop
        from sonic.computer_use.agent import ComputerUseAgent
        from sonic.computer_use.curiosity import CuriosityLoop
        from sonic.memory.vector import get_vector_memory
        from sonic.sandbox.virtual_computer import get_sandbox_provider

        tenant_id = os.environ.get("SONIC_BEING_TENANT", "default")
        being = get_or_create_being(tenant_id)

        # Re-attach the home desktop (Phase 2 persistent body).
        # Prefer the native Docker cyber-workstation (real GUI + terminal +
        # browser) when the daemon is reachable; fall back to the generic
        # sandbox provider (container or fail-closed local). Both providers
        # now expose get_or_create_home, so the always-on being has a home instead
        # of aborting at boot (the pre-fix VirtualComputer had no such method).
        provider = None
        try:
            from sonic.computer.docker_computer import DockerComputerProvider
            from sonic.computer.models import ComputerWorkspaceStatus
            candidate = DockerComputerProvider()
            st = await candidate.status(candidate.container_name)
            if st.status not in (ComputerWorkspaceStatus.FAILED, ComputerWorkspaceStatus.DEGRADED):
                provider = candidate
        except Exception as e:
            logger.warning("being_provider_docker_unavailable", error=str(e))
        if provider is None:
            provider = await get_sandbox_provider()
        if not callable(getattr(provider, "screenshot", None)):
            logger.warning(
                "being_life_loop_no_computer_provider",
                provider=type(provider).__name__,
                reason="screen_capture_unavailable",
            )
            return None
        home = None
        if hasattr(provider, "get_or_create_home"):
            home = await provider.get_or_create_home(tenant_id)
        if home is None or getattr(home, "id", None) is None or getattr(home, "id", "") == "":
            logger.warning("being_life_loop_no_home", being_id=being.being_id)
            return None

        from sonic.safety.sealed import seal_default
        safety = seal_default(
            workspace_root="/home/sonic/workspace",
            allow_private_networks=True,
            require_approval_for_intrusive=False,
            max_actions_per_minute=300,
        )
        from pathlib import Path

        from sonic.config import CONFIGS_DIR
        from sonic.llm.router import ModelRouter
        cfg_path = Path("configs/models.yaml")
        if not cfg_path.exists() and (CONFIGS_DIR / "models.yaml").exists():
            cfg_path = CONFIGS_DIR / "models.yaml"
        router = ModelRouter.from_config(cfg_path) if cfg_path.exists() else ModelRouter()
        from sonic.tools.registry import get_default_registry
        security_registry = get_default_registry(provider)
        # Toolsmith loop (Phase A, AIOSR): the being authors NEW tools for
        # observation gaps.
        from sonic.being.craft import BeingCraft
        from sonic.being.toolsmith import ToolsmithLoop
        toolsmith = ToolsmithLoop(
            craft=BeingCraft(being_id=being.being_id),
            llm=router, registry=security_registry,
        )
        # Method Lab loop (Phase B, AIOSR): the being invents NEW attack methods
        # for defensive postures where existing tools fail.
        from sonic.being.method_lab import MethodLab
        method_lab = MethodLab(
            llm=router,
            vector_memory=get_vector_memory(),
            toolsmith=toolsmith,
        )
        # Lessons Ledger (Phase C, AIOSR): the being learns from experience
        # and adapts testing strategies across missions.
        from sonic.being.lessons import LessonsLedger
        lessons_ledger = LessonsLedger(
            tenant_id=tenant_id,
            agent_id=being.being_id,
            semantic_memory=get_vector_memory(),
        )
        # Use the visible X11 desktop (:99) as the single computer surface
        # rather than spawning a hidden headless Playwright instance.
        browser = None
        from sonic.evolution.engine import EvolutionEngine
        from sonic.evolution.strategy import DynamicStrategyEngine
        evolution_engine = EvolutionEngine(
            strategy_engine=DynamicStrategyEngine(),
            method_lab=method_lab,
            toolsmith=toolsmith,
            lessons_ledger=lessons_ledger,
            llm_client=router,
            queue_self_update_goal=False,
        )
        self_coder.configure_evolution_engine(evolution_engine)
        agent = ComputerUseAgent(
            computer_provider=provider, llm_router=router,
            security_tools=security_registry.as_dict(),
            safety=safety, self_host=True, tenant_id=tenant_id, agent_id=being.being_id,
            browser=browser,
            gui_only=False,
            compact_prompt=True,
            observe_desktop=True,
            toolsmith=toolsmith,
            method_lab=method_lab,
            lessons_ledger=lessons_ledger,
            evolution_engine=evolution_engine,
            being_mind=get_being_store().get_mind(being.being_id),
        )
        steps_per_goal = int(os.environ.get("SONIC_BEING_STEPS_PER_CYCLE", "6"))
        curiosity = CuriosityLoop(
            llm_router=router,
            vector_memory=get_vector_memory(),
            max_cycles=1000,
            exploration_steps=steps_per_goal,
        )

        # Stream autonomous BeingLifeLoop goals and actions into the live Workstation UI
        try:
            from sonic.api.routes.workstation import (
                _append_worklog,
                _get_or_create_session,
                _persist_workstation_state,
            )

            async def _on_idle_goal(goal_text: str, steps_budget: int) -> None:
                ws_state = _get_or_create_session(tenant_id, "default")
                ws_state["status"] = "RUNNING"
                ws_state["current_action"] = f"[LifeLoop] {goal_text[:90]}"
                _append_worklog(
                    ws_state,
                    "action",
                    "Autonomous BeingLifeLoop Cycle",
                    f"Proposed self-directed goal ({steps_budget} steps): {goal_text}",
                )
                _persist_workstation_state()

            async def _on_idle_step(trace: Any) -> None:
                ws_state = _get_or_create_session(tenant_id, "default")
                act_type = getattr(trace, "action_type", "")
                act_val = act_type.value if hasattr(act_type, "value") else str(act_type)
                target_res = getattr(trace, "target_resource", "")
                obs_res = (getattr(trace, "actual_observation", "") or "")[:400]
                thought_txt = getattr(trace, "thought", "") or ""
                ws_state["current_action"] = f"[LifeLoop] {act_val} -> {target_res}"
                if thought_txt:
                    _append_worklog(ws_state, "thought", "LifeLoop Thinking", thought_txt)
                _append_worklog(
                    ws_state,
                    "action",
                    f"LifeLoop: {act_val}",
                    f"Target: {target_res}\nResult: {obs_res}",
                )
                _persist_workstation_state()

            agent._idle_goal_callback = _on_idle_goal
            agent._idle_step_callback = _on_idle_step
        except Exception as bridge_err:
            logger.debug("being_life_loop_ui_bridge_skipped", error=str(bridge_err))

        tick_interval = float(os.environ.get("SONIC_BEING_TICK_INTERVAL", "12"))
        loop = BeingLifeLoop(
            being, agent, curiosity, home.id, tick_interval=tick_interval,
        )
        loop.start()
        from sonic.being.life_loop import register_life_loop
        register_life_loop(loop)
        loop.before_tick_hooks: list = getattr(loop, "before_tick_hooks", [])
        loop.before_tick_hooks.append(
            lambda: setattr(agent, "being_mind", get_being_store().get_mind(being.being_id))
        )
        logger.info("being_life_loop_running", being_id=being.being_id, home_id=home.id)
        return loop
    except Exception as e:
        logger.warning("being_life_loop_init_failed", error=str(e))
        return None


async def _maybe_start_self_coder_background_loop():
    """Start periodic codebase maintenance independently of desktop curiosity."""
    import os

    from sonic.evolution.self_coder_config import get_self_coder_config

    try:
        config = get_self_coder_config()
    except ValueError as exc:
        logger.error("self_coder_configuration_invalid", error=str(exc))
        return None
    if not config.enabled:
        logger.info("self_coder_background_loop_disabled")
        return None
    try:
        from sonic.api.routes.self_coder import _build_coder, _candidate_store
        from sonic.evolution.self_coder_loop import SelfCoderBackgroundLoop

        coder = await asyncio.to_thread(_build_coder, require_llm=True)
        loop = SelfCoderBackgroundLoop(
            candidate_store=_candidate_store(),
            coder_factory=lambda root: coder.isolated_copy(root),
            tenant_id=os.environ.get(
                "SONIC_SELF_CODER_TENANT",
                os.environ.get("SONIC_BEING_TENANT", "default"),
            ),
            interval_seconds=config.interval_seconds,
        )
        loop.start()
        logger.info(
            "self_coder_background_loop_configured",
            interval_seconds=config.interval_seconds,
        )
        return loop
    except Exception as exc:
        logger.error("self_coder_background_loop_init_failed", error=str(exc))
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()

    logger.info(
        "sonic_starting",
        version=__version__,
        codename=__codename__,
        env=settings.app_env,
    )

    # Security: refuse to boot in production with an insecure JWT signing secret.
    jwt_ok, jwt_reason = settings.validate_jwt_secret()
    if not jwt_ok:
        logger.error("jwt_secret_invalid", reason=jwt_reason)
        raise RuntimeError(jwt_reason)
    if settings.is_dev and "default" in jwt_reason:
        logger.warning("jwt_secret_weak_dev", reason=jwt_reason)
    logger.info("jwt_secret_validated", production=settings.is_production)

    # Load safety rules (immutable after this point)
    get_scope_checker()
    logger.info("safety_layer_loaded")

    # Connect to Graph Memory (Neo4j) with fast dev fallback to SQLite
    from sonic.memory.router import get_smart_memory, set_active_memory
    graph = get_graph_memory()
    try:
        connected = await asyncio.wait_for(graph.connect(), timeout=2.0)
    except Exception as exc:
        logger.warning("graph_memory_connect_timeout_or_failed", error=str(exc))
        connected = False
    if connected:
        logger.info("graph_memory_connected")
        # Initialize graph schema (constraints + indexes)
        await graph.init_schema()
        set_active_memory(graph)
        logger.info("graph_schema_initialized")
    else:
        smart_mem = await get_smart_memory()
        logger.info("persistent_sqlite_graph_memory_ready", backend=type(smart_mem).__name__)

    # AI-Human layer: re-attach the persistent being for the default tenant and
    # spawn its always-on curiosity life loop. The being's identity + mind are
    # re-resolved from SQLite (survives restart). The loop is ONLY spawned when
    # a home workspace is available and an ActionPolicy is configured — an
    # always-on autonomous being must not act without the safety envelope. In
    # test/headless boots without a sandbox, this is a no-op (logged, not fatal).
    life_loop_task = None
    try:
        life_loop_task = await _maybe_start_being_life_loop(settings)
    except Exception as e:
        logger.warning("being_life_loop_not_started", reason=str(e))
    self_coder_loop = await _maybe_start_self_coder_background_loop()

    yield

    # Shutdown
    if self_coder_loop is not None:
        try:
            await self_coder_loop.stop()
        except Exception as e:
            logger.warning("self_coder_background_loop_stop_failed", error=str(e))
    if life_loop_task is not None:
        try:
            await life_loop_task.stop()
        except Exception as e:
            logger.warning("being_life_loop_stop_failed", error=str(e))
    # Release the shared computer provider client (aiohttp session) if one was created.
    # Name is `_primary_computer_instance` in workstation.py (was `_daytona_provider_instance`
    # during the Daytona era; the stale name crashed clean shutdowns with ImportError).
    from sonic.api.routes.workstation import _primary_computer_instance
    if _primary_computer_instance is not None:
        try:
            await _primary_computer_instance.close()
        except Exception as e:
            logger.warning("daytona_provider_close_failed", error=str(e))
    await graph.disconnect()
    logger.info("sonic_shutdown")


app = FastAPI(
    title="SONIC-REDA API",
    description="Autonomous AI-Human Digital Being & Dual-Plane Computer Workstation Engine",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---- CORS ----
settings = get_settings()
logger.info("cors_configured", origins=settings.cors_origins_list)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Mount Routes ----
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(llm.router, prefix="/llm", tags=["LLM"])
app.include_router(engagements.router, prefix="/engagements", tags=["Engagements"])
app.include_router(agents.router, prefix="/agents", tags=["Agents"])
app.include_router(graph.router, prefix="/graph", tags=["Graph Memory"])
app.include_router(findings.router, prefix="/findings", tags=["Findings"])
app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
app.include_router(terminal.router, prefix="/terminal", tags=["Terminal"])
app.include_router(live.router, prefix="/live", tags=["Live Dashboard"])
app.include_router(jobs.router, prefix="/jobs", tags=["Async Jobs"])
app.include_router(organization.router, prefix="/organization", tags=["Organization"])
app.include_router(workstation.router, tags=["Workstation"])
app.include_router(checkpoints.router, tags=["Workstation Checkpoints"])
app.include_router(security.router, prefix="/security", tags=["Self-Security Lab"])
app.include_router(self_coder.router, prefix="/self-coder", tags=["Self-Coder"])



from fastapi.responses import PlainTextResponse

from sonic.observability.metrics import get_metrics


@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
async def metrics():
    """Prometheus metrics exposition endpoint."""
    return get_metrics().export_prometheus()


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint — system info."""
    return {
        "system": __codename__,
        "version": __version__,
        "status": "operational",
        "docs": "/docs",
    }
