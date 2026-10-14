"""
End-to-End Integration Tests for Director, Event-Driven Scheduling, Replan Loop, and State Recovery.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sonic.agents.director import Director, AgentOutput
from sonic.agents.replan import ReplanDecision
from sonic.agents.state_store import StateStore
from sonic.agents.task_graph import TaskGraphError
from sonic.memory.inmemory import InMemoryGraph
from sonic.safety.scope import ScopeChecker
from sonic.llm.router import ModelRouter
from sonic.llm.schemas import LLMResponse


def get_mock_router():
    router = MagicMock(spec=ModelRouter)
    # Default mock response returning valid JSON plan/replan
    async def _complete(req, task_type="planning"):
        if "replan" in req.messages[-1].content.lower() or "trigger" in req.messages[-1].content.lower():
            return LLMResponse(
                content="""```json
{
    "should_replan": true,
    "reasoning": "New endpoint /api/v2/tokens discovered. Need auth analysis.",
    "new_tasks": [
        {
            "name": "Dynamic: Test /api/v2/tokens",
            "agent_type": "dynamic",
            "task_payload": {"target": "target.com/api/v2/tokens"},
            "depends_on_completed": [],
            "priority": "high"
        }
    ],
    "tasks_to_skip": [],
    "updated_unknowns": [{"question": "Is /api/v2/tokens authenticated?"}],
    "updated_next_action": "Execute token testing",
    "confidence": 0.8
}
```""",
                model="test-model",
            )
        return LLMResponse(
            content="""```json
{
    "assumptions": ["target.com has REST APIs"],
    "unknowns": ["What subdomains exist?"],
    "tasks": [
        {
            "name": "Recon-A: Subdomain Enumeration",
            "agent_type": "recon",
            "priority": "high",
            "depends_on": [],
            "payload": {"target": "target.com", "task": "subdomains"}
        },
        {
            "name": "Recon-B: Port Scan",
            "agent_type": "recon",
            "priority": "medium",
            "depends_on": [],
            "payload": {"target": "target.com", "task": "ports"}
        }
    ]
}
```""",
            model="test-model",
        )
    router.complete = AsyncMock(side_effect=_complete)
    return router


def test_director_deterministic_e2e_flow():
    async def _run():
        mock_router = get_mock_router()
        memory = InMemoryGraph()
        await memory.init_schema()
        scope = ScopeChecker()
        state_store = StateStore(redis_client=None)

        director = Director(
            model_router=mock_router,
            graph_memory=memory,
            scope_checker=scope,
            state_store=state_store,
            max_parallel=2,
        )

        # 1. Start engagement
        engagement_id = await director.start_engagement(
            target="target.com",
            scope={"allowed_domains": ["target.com"]},
            tenant_id="tenant-acme",
            created_by="auditor@acme.org",
        )
        assert engagement_id.startswith("eng-")

        # Verify initial state and graph
        state_summary = director.get_engagement_state(engagement_id)
        assert state_summary is not None
        assert state_summary["unresolved_unknowns_count"] >= 1

        graph = director._graphs[engagement_id]
        assert graph.size == 2

        # 2. Scheduler finds ready tasks
        dispatchable = director.get_dispatchable_tasks(engagement_id)
        assert len(dispatchable) == 2
        task1 = dispatchable[0]
        assert task1["tenant_id"] == "tenant-acme"
        assert task1["engagement_id"] == engagement_id

        # 3. Simulate Recon-A completing with new discovery
        recon_a_output = AgentOutput(
            observations=[
                {"description": "Discovered new endpoint /api/v2/tokens", "raw_data": "HTTP 200 OK"}
            ],
            facts=[
                {"description": "Target hosts /api/v2/tokens on port 443", "evidence_ids": ["ev-1"]}
            ],
            hypotheses=[
                {
                    "title": "Unauthenticated Token Leakage",
                    "description": "Tokens can be fetched without auth header",
                    "vulnerability_class": "Auth Bypass",
                }
            ],
            evidence_refs=[
                {"evidence_type": "request", "description": "GET /api/v2/tokens", "storage_key": "s3://ev-1"}
            ],
        )

        res1 = await director.on_task_completed(
            engagement_id=engagement_id,
            task_id=task1["task_id"],
            result=recon_a_output.to_dict(),
        )

        # Verify replan was triggered and injected new task
        assert res1["facts_ingested"] == 1
        assert res1["hypotheses_ingested"] == 1
        assert "tasks_added" in res1["replan"]
        assert len(res1["replan"]["tasks_added"]) == 1

        # Graph size should now be 3 (2 initial + 1 injected)
        assert graph.size == 3

        # 4. Simulate Recon-B completing
        task2 = dispatchable[1]
        recon_b_output = AgentOutput(
            facts=[{"description": "Ports 80, 443 open"}],
        )
        res2 = await director.on_task_completed(
            engagement_id=engagement_id,
            task_id=task2["task_id"],
            result=recon_b_output.to_dict(),
        )

        # 5. Simulate the injected Dynamic tasks completing until graph is complete
        loop_guard = 0
        while not graph.is_complete() and loop_guard < 10:
            loop_guard += 1
            injected_ready = director.get_dispatchable_tasks(engagement_id)
            if not injected_ready:
                break
            for inj_task in injected_ready:
                dynamic_output = AgentOutput(
                    facts=[{"description": "Token leakage confirmed with HTTP 200"}],
                    findings=[
                        {
                            "title": "Critical Token Leak on /api/v2/tokens",
                            "severity": "critical",
                            "vulnerability_class": "Auth Bypass",
                        }
                    ],
                )
                await director.on_task_completed(
                    engagement_id=engagement_id,
                    task_id=inj_task["task_id"],
                    result=dynamic_output.to_dict(),
                )

        final_summary = director.get_engagement_state(engagement_id)
        assert final_summary["facts_count"] >= 3
        assert final_summary["replan_count"] >= 1
        assert graph.is_complete() is True

    asyncio.run(_run())


def test_pausing_engagement_cancels_active_async_workers_and_resets_tasks():
    async def _run():
        memory = InMemoryGraph()
        await memory.init_schema()
        director = Director(
            model_router=get_mock_router(),
            graph_memory=memory,
            scope_checker=ScopeChecker(),
            state_store=StateStore(redis_client=None),
            max_parallel=2,
        )
        engagement_id = await director.start_engagement(
            target="target.example.test",
            scope={"allowed_domains": ["target.example.test"]},
            tenant_id="pause-tenant",
            created_by="operator@example.test",
        )
        started = asyncio.Event()

        async def worker(_payload):
            started.set()
            await asyncio.Event().wait()

        loop_task = asyncio.create_task(
            director.run_engagement_loop(engagement_id, worker)
        )
        await started.wait()
        assert await director.pause_engagement(engagement_id)
        result = await asyncio.wait_for(loop_task, timeout=1)

        assert result["status"] == "paused"
        assert not any(
            task.status.value == "running"
            for task in director._graphs[engagement_id].tasks.values()
        )
        assert await director.resume_engagement(engagement_id)
        assert not director._pause_events[engagement_id].is_set()

    asyncio.run(_run())


def test_pausing_engagement_drains_non_interruptible_sync_workers():
    async def _run():
        import threading

        memory = InMemoryGraph()
        await memory.init_schema()
        director = Director(
            model_router=get_mock_router(),
            graph_memory=memory,
            scope_checker=ScopeChecker(),
            state_store=StateStore(redis_client=None),
            max_parallel=1,
        )
        engagement_id = await director.start_engagement(
            target="target.example.test",
            scope={"allowed_domains": ["target.example.test"]},
            tenant_id="pause-sync-tenant",
            created_by="operator@example.test",
        )
        started = threading.Event()
        release = threading.Event()

        def worker(_payload):
            started.set()
            release.wait(timeout=2)
            return {"status": "completed"}

        loop_task = asyncio.create_task(
            director.run_engagement_loop(engagement_id, worker)
        )
        assert await asyncio.to_thread(started.wait, 1)
        assert await director.pause_engagement(engagement_id)
        await asyncio.sleep(0)
        assert not loop_task.done()
        release.set()
        result = await asyncio.wait_for(loop_task, timeout=2)

        assert result["status"] == "paused"
        assert result["completed"] == 1
        assert all(
            task.status.value != "running"
            for task in director._graphs[engagement_id].tasks.values()
        )

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_director_rejects_callbacks_from_a_previous_retry_attempt():
    director = Director(
        model_router=get_mock_router(),
        graph_memory=InMemoryGraph(),
        scope_checker=ScopeChecker(),
        state_store=StateStore(redis_client=None),
        max_parallel=1,
    )
    director.replan_engine.evaluate = AsyncMock(
        return_value=ReplanDecision(should_replan=False)
    )
    engagement_id = await director.start_engagement(
        target="target.com",
        scope={},
        tenant_id="tenant-stale-callback",
    )

    first_attempt = director.get_dispatchable_tasks(engagement_id)[0]
    await director.on_task_failed(
        engagement_id,
        first_attempt["task_id"],
        "transient failure",
        execution_id=first_attempt["execution_id"],
    )
    retry_attempt = director.get_dispatchable_tasks(engagement_id)[0]
    assert retry_attempt["task_id"] == first_attempt["task_id"]
    assert retry_attempt["execution_id"] != first_attempt["execution_id"]

    stale_completion = await director.on_task_completed(
        engagement_id,
        retry_attempt["task_id"],
        {"status": "success"},
        execution_id=first_attempt["execution_id"],
    )
    assert "error" in stale_completion

    await director.on_task_completed(
        engagement_id,
        retry_attempt["task_id"],
        {"status": "success"},
        execution_id=retry_attempt["execution_id"],
    )
    with pytest.raises(TaskGraphError):
        await director.on_task_failed(
            engagement_id,
            retry_attempt["task_id"],
            "late failure",
            execution_id=first_attempt["execution_id"],
        )


def test_director_concurrency_limit():
    async def _run():
        mock_router = get_mock_router()
        memory = InMemoryGraph()
        scope = ScopeChecker()
        state_store = StateStore(redis_client=None)

        director = Director(
            model_router=mock_router,
            graph_memory=memory,
            scope_checker=scope,
            state_store=state_store,
            max_parallel=1,  # Max 1 concurrent task
        )

        engagement_id = await director.start_engagement(
            target="target.com",
            scope={},
            tenant_id="tenant-concurrency",
        )

        # Even though 2 tasks are ready, only 1 should be dispatched due to max_parallel=1
        dispatchable = director.get_dispatchable_tasks(engagement_id)
        assert len(dispatchable) == 1

        # Second call should return 0 since 1 is running and max_parallel=1
        dispatchable_2 = director.get_dispatchable_tasks(engagement_id)
        assert len(dispatchable_2) == 0

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_director_rejected_empty_plan_is_not_complete_or_successful():
    """Unsupported worker labels are rejected; an empty graph is not success."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(return_value=LLMResponse(
        content="""{
            "assumptions": [],
            "unknowns": [],
            "tasks": [
                {"name": "Author custom probe", "agent_type": "toolsmith"},
                {"name": "Invent method", "agent_type": "method"}
            ]
        }""",
        model="test-model",
    ))
    director = Director(
        model_router=router,
        graph_memory=InMemoryGraph(),
        scope_checker=ScopeChecker(),
        state_store=StateStore(),
    )
    engagement_id = await director.start_engagement(
        target="target.test",
        scope={},
        tenant_id="tenant-empty-plan",
    )
    graph = director._graphs[engagement_id]
    assert graph.size == 0
    assert graph.is_complete() is False

    result = await director.run_engagement_loop(
        engagement_id,
        worker_fn=AsyncMock(),
    )
    assert result["completed"] == 0
    assert result["graph_complete"] is False
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_director_runs_dispatched_tasks_concurrently_within_limit():
    """A dispatched batch runs concurrently and retains each task's findings."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(return_value=LLMResponse(
        content="""{
            "assumptions": [],
            "unknowns": [],
            "tasks": [
                {"name": "Task one", "agent_type": "recon", "payload": {}},
                {"name": "Task two", "agent_type": "static", "payload": {}},
                {"name": "Task three", "agent_type": "dynamic", "payload": {}}
            ]
        }""",
        model="test-model",
    ))
    director = Director(
        model_router=router,
        graph_memory=InMemoryGraph(),
        scope_checker=ScopeChecker(),
        state_store=StateStore(),
        max_parallel=2,
    )
    director._evaluate_stop_condition = MagicMock(return_value=MagicMock(
        should_stop=False,
        model_dump=lambda: {},
    ))
    director.replan_engine.evaluate = AsyncMock(
        return_value=MagicMock(should_replan=False)
    )
    engagement_id = await director.start_engagement(
        target="target.test",
        scope={},
        tenant_id="tenant-concurrent-loop",
    )

    active = 0
    max_active = 0

    async def worker(_payload):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.02)
            if _payload["agent_type"] == "static":
                raise RuntimeError("static worker failed")
            return {
                "status": "success",
                "findings": [{"title": f"finding-{max_active}"}],
            }
        finally:
            active -= 1

    result = await director.run_engagement_loop(engagement_id, worker)

    assert max_active == 2
    assert result["completed"] == 2
    assert result["failed"] == 3  # initial attempt plus the two configured retries
    assert len(result["findings"]) == 2
    assert result["graph_complete"] is True
    assert result["status"] == "partial"
    failed_task = next(
        task for task in director._graphs[engagement_id].tasks.values()
        if task.agent_type == "static"
    )
    assert failed_task.status.value == "failed"
    assert failed_task.retry_count == failed_task.max_retries == 2
    assert failed_task.error == "static worker failed"
    assert not director._graphs[engagement_id].get_running_tasks()


@pytest.mark.asyncio
async def test_director_rejects_graph_only_human_task_and_its_dependents():
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(return_value=LLMResponse(
    content="""{
        "assumptions": [],
        "unknowns": [],
        "tasks": [
            {"name": "Operator review", "agent_type": "human"},
            {
                "name": "Follow-up",
                "agent_type": "recon",
                "depends_on": ["Operator review"]
            }
        ]
    }""",
    model="test-model",
    ))
    director = Director(
    model_router=router,
    graph_memory=InMemoryGraph(),
    scope_checker=ScopeChecker(),
    state_store=StateStore(redis_client=None),
    available_agent_types={"recon"},
    )

    engagement_id = await director.start_engagement(
    target="target.test",
    scope={},
    tenant_id="tenant-human-graph",
    )

    graph = director._graphs[engagement_id]
    assert graph.size == 0
    assert director.get_dispatchable_tasks(engagement_id) == []


def test_director_pause_and_resume():
    async def _run():
        mock_router = get_mock_router()
        memory = InMemoryGraph()
        scope = ScopeChecker()
        state_store = StateStore(redis_client=None)

        director = Director(
            model_router=mock_router,
            graph_memory=memory,
            scope_checker=scope,
            state_store=state_store,
        )

        engagement_id = await director.start_engagement(
            target="target.com",
            scope={},
            tenant_id="tenant-pause",
        )

        paused = await director.pause_engagement(engagement_id)
        assert paused is True

        resumed = await director.resume_engagement(engagement_id)
        assert resumed is True

    asyncio.run(_run())
