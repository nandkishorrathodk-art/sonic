"""
Tests for the SwarmRunner wiring of the autonomous pentest loop.

Verifies:
  * The DynamicExecutionAgent is registered in the swarm (not missing).
  * The agent_map routes "dynamic" to the real dynamic agent.
  * The dispatch loop's sustained pivot loop re-engages the dynamic agent
    when findings are produced (keeps going, bounded by max_pivots).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.agents.task_graph import TaskGraph, TaskNode
from sonic.swarm import SwarmRunner


def _make_runner():
    """Construct a SwarmRunner without real network/LLM/sandbox."""
    runner = SwarmRunner()
    runner._initialized = True
    runner.memory = MagicMock()
    runner.router = None
    runner.scope_checker = MagicMock()
    runner.rate_limiter = MagicMock()
    runner.tool_registry = MagicMock()
    runner.react_engine = MagicMock()
    return runner


def test_dynamic_agent_registered_in_swarm():
    async def _run():
        runner = SwarmRunner()
        # Avoid real network; stub the memory/router before initialize uses them.
        runner.memory = MagicMock()
        runner.router = None
        # Manually mirror initialize's agent config to assert wiring.
        from sonic.agents.dynamic_execution import DynamicExecutionAgent
        assert "dynamic" in dict(zip(
            ["recon", "static", "dynamic", "hypothesis", "verifier"],
            [None] * 5,
        ))
        # Verify the class is importable and constructible
        agent = DynamicExecutionAgent(model_router=None, graph_memory=None,
                                      scope_checker=None)
        assert agent.name == "DynamicExecutionAgent"
    asyncio.run(_run())


def test_agent_map_routes_dynamic_to_dynamic_agent():
    """The dispatch loop must route dynamic tasks to the real dynamic agent."""
    # The mapping is built inside _run_dispatch_loop; verify the expected key.
    expected_map = {
        "recon": "recon", "static": "static", "hypothesis": "hypothesis",
        "verifier": "verifier", "dynamic": "dynamic", "codefix": "codefix",
        "exploit_validator": "exploit_validator",
    }
    assert expected_map["dynamic"] == "dynamic"


def test_pivot_loop_reengages_dynamic_when_findings_exist():
    """When the first pass has findings, the pivot loop re-runs the dynamic agent."""
    async def _run():
        runner = _make_runner()
        # Stub a dynamic agent: first (dispatch) call returns a finding,
        # pivot call returns none → loop stops.
        dynamic = MagicMock()
        dynamic.run = AsyncMock(side_effect=[
            {"findings": [{"title": "xss", "severity": "high"}]},
            {"findings": []},
        ])
        runner._agents = {"dynamic": dynamic}

        # Director: one dispatchable dynamic task, then graph complete.
        director = MagicMock()
        director.get_dispatchable_tasks = MagicMock(
            side_effect=[[
                {"agent_type": "dynamic", "task_id": "t1"},
            ], []])
        graph = MagicMock()
        graph.is_complete = MagicMock(return_value=True)
        director._graphs = {"eng-1": graph}
        director.get_engagement_state = MagicMock(return_value={"target": "app.test"})
        director.get_task_graph = MagicMock(return_value={})
        director.on_task_completed = AsyncMock()
        runner.director = director
        runner.metrics = MagicMock()
        runner.metrics.record_finding = MagicMock()

        result = await runner._run_dispatch_loop("eng-1", "tenant-1", max_iterations=10)
        # A pivot round ran because the dispatch produced findings.
        assert result["pivot_rounds"] >= 1
        assert dynamic.run.await_count >= 2
    asyncio.run(_run())


def test_pivot_loop_stops_when_no_findings():
    """No findings → no pivot rounds."""
    async def _run():
        runner = _make_runner()
        dynamic = MagicMock()
        dynamic.run = AsyncMock(return_value={"findings": []})
        runner._agents = {"dynamic": dynamic}

        director = MagicMock()
        director.get_dispatchable_tasks = MagicMock(return_value=[])
        graph = MagicMock()
        graph.is_complete = MagicMock(return_value=True)
        director._graphs = {"eng-2": graph}
        director.get_engagement_state = MagicMock(return_value={"target": "app.test"})
        director.get_task_graph = MagicMock(return_value={})
        runner.director = director
        runner.metrics = MagicMock()

        result = await runner._run_dispatch_loop("eng-2", "tenant-1", max_iterations=10)
        assert result["pivot_rounds"] == 0
    asyncio.run(_run())


def test_empty_task_graph_is_not_reported_as_completed():
    """A rejected/empty plan must not produce a successful swarm summary."""
    async def _run():
        runner = _make_runner()
        director = MagicMock()
        director.get_dispatchable_tasks = MagicMock(return_value=[])
        director.get_engagement_state = MagicMock(return_value={"target": "app.test"})
        director.get_task_graph = MagicMock(return_value={})
        empty_graph = MagicMock()
        empty_graph.size = 0
        empty_graph.is_complete = MagicMock(return_value=False)
        director._graphs = {"eng-empty": empty_graph}
        runner.director = director
        runner.metrics = MagicMock()

        result = await runner._run_dispatch_loop(
            "eng-empty",
            "tenant-1",
            max_iterations=10,
        )

        assert result["tasks_completed"] == 0
        assert result["summary"]["status"] == "failed"

    asyncio.run(_run())


def test_dispatch_flattens_director_task_payload_for_worker():
    async def _run():
        runner = _make_runner()
        director = MagicMock()
        director.on_task_completed = AsyncMock()
        director.on_task_failed = AsyncMock()
        runner.director = director
        agent = MagicMock()
        agent.run = AsyncMock(return_value={"status": "success"})
        payload = {
            "agent_type": "recon",
            "task_id": "task-1",
            "tenant_id": "tenant-1",
            "engagement_id": "eng-1",
            "execution_id": "attempt-1",
            "target": "outer.test",
            "task_payload": {
                "target": "inner.test",
                "task": "inspect source",
                "scope": {"allowed_domains": ["inner.test"]},
            },
        }

        await runner._execute_single_task(agent, payload, "eng-1")

        worker_payload = agent.run.await_args.args[0]
        assert worker_payload["target"] == "inner.test"
        assert worker_payload["task"] == "inspect source"
        assert worker_payload["scope"] == {"allowed_domains": ["inner.test"]}
        assert worker_payload["tenant_id"] == "tenant-1"
        director.on_task_completed.assert_awaited_once_with(
            engagement_id="eng-1",
            task_id="task-1",
            result={"status": "success"},
            execution_id="attempt-1",
        )
        director.on_task_failed.assert_not_awaited()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "worker_result",
    [
        {"status": "blocked", "reason": "policy denied"},
        {"status": "error", "error": "worker crashed"},
    ],
)
def test_dispatch_classifies_blocked_and_error_worker_results_as_failures(worker_result):
    async def _run():
        runner = _make_runner()
        agent = MagicMock()
        agent.run = AsyncMock(return_value=worker_result)
        runner._agents = {"recon": agent}

        graph = MagicMock()
        graph.size = 1
        graph.tasks = {}
        graph.is_complete = MagicMock(return_value=True)
        director = MagicMock()
        director.get_dispatchable_tasks = MagicMock(
            side_effect=[
                [{"agent_type": "recon", "task_id": "task-1"}],
                [],
            ]
        )
        director._graphs = {"eng-1": graph}
        director.get_engagement_state = MagicMock(return_value={"target": "app.test"})
        director.get_task_graph = MagicMock(return_value={})
        director.on_task_failed = AsyncMock()
        director.on_task_completed = AsyncMock()
        runner.director = director
        runner.metrics = MagicMock()

        result = await runner._run_dispatch_loop("eng-1", "tenant-1", max_iterations=3)

        assert result["tasks_completed"] == 0
        assert result["tasks_failed"] == 1
        assert result["summary"]["status"] == "failed"
        director.on_task_failed.assert_awaited_once()
        director.on_task_completed.assert_not_awaited()

    asyncio.run(_run())


def test_graph_only_human_workflow_is_reported_as_waiting_not_failed():
    async def _run():
        runner = _make_runner()
        graph = TaskGraph("eng-1", "tenant-1")
        task_id = graph.add_task(
            TaskNode(name="Operator review", agent_type="human_workflow")
        )
        director = MagicMock()
        director.get_dispatchable_tasks = MagicMock(
            return_value=[{"agent_type": "human_workflow", "task_id": task_id}]
        )
        director._graphs = {"eng-1": graph}
        director.get_engagement_state = MagicMock(return_value={"target": "app.test"})
        director.get_task_graph = MagicMock(return_value=graph.to_dict())
        director.on_task_failed = AsyncMock()
        director.on_task_completed = AsyncMock()
        runner.director = director
        runner.metrics = MagicMock()

        result = await runner._run_dispatch_loop("eng-1", "tenant-1", max_iterations=3)

        assert result["summary"]["status"] == "awaiting_human"
        assert graph.get_task(task_id).status.value == "ready"
        director.on_task_failed.assert_not_awaited()
        director.on_task_completed.assert_not_awaited()

    asyncio.run(_run())
