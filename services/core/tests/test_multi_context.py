"""Tests for multi-context retrieval: fusion, tenancy, bounds, honesty."""

import asyncio

import pytest

from sonic.memory.multi_context import (
    ContextItem,
    EpisodicContextSource,
    GraphFindingSource,
    MultiContextAssembler,
    VectorContextSource,
    build_default_sources,
    render_multi_context,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------


class _Vector:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, query, *, top_k, score_threshold, tenant_id):
        self.calls.append((query, top_k, score_threshold, tenant_id))
        return [r for r in self.rows if r["tenant_id"] == tenant_id][:top_k]


def _vrow(doc_id, text, score, tenant_id="t1", citation=None):
    return {
        "doc_id": doc_id,
        "text": text,
        "score": score,
        "citation": citation or f"doc/{doc_id}",
        "metadata": {"kind": "note"},
        "tenant_id": tenant_id,
    }


def test_vector_source_maps_rows_to_cited_items():
    src = VectorContextSource(_Vector([_vrow("a", "open port 22 on host", 0.7)]))
    items = _run(src.fetch("ports", tenant_id="t1", limit=5))
    assert len(items) == 1
    assert items[0].source == "vector_memory"
    assert items[0].citation == "doc/a"
    assert items[0].score == 0.7


def test_vector_source_skips_blank_text():
    src = VectorContextSource(_Vector([_vrow("a", "   ", 0.9)]))
    assert _run(src.fetch("q", tenant_id="t1", limit=5)) == []


class _Graph:
    def __init__(self, rows):
        self.rows = rows

    async def search_findings(self, text, limit=20, tenant_id=None):
        return self.rows[:limit]


def test_graph_source_maps_findings_and_keeps_severity_metadata():
    graph = _Graph([
        {"finding": {"uid": "f1", "title": "SQLi in login", "description": "auth bypass", "severity": "high"}, "score": 1.0},
    ])
    items = _run(GraphFindingSource(graph).fetch("sqli", tenant_id="t1", limit=5))
    assert items[0].source == "graph_memory"
    assert "SQLi in login" in items[0].content
    assert items[0].citation == "graph/finding/f1"
    assert items[0].metadata["severity"] == "high"


def test_graph_source_without_search_findings_is_empty():
    class _NoSearch:
        pass

    assert _run(GraphFindingSource(_NoSearch()).fetch("q", tenant_id="t1", limit=5)) == []


class _SubstringGraph:
    """Mirrors SqliteGraph.search_findings: full-string substring match."""

    def __init__(self, findings):
        self.findings = findings
        self.probes = []

    async def search_findings(self, text, limit=20, tenant_id=None):
        self.probes.append(text)
        out = []
        for f in self.findings:
            hay = f"{f.get('title','')} {f.get('description','')} {f.get('vulnerability_class','')}".lower()
            if text.lower() in hay:
                out.append({"finding": f, "score": 1.0})
        return out[:limit]


def test_graph_source_reaches_findings_from_a_natural_language_goal():
    graph = _SubstringGraph([
        {"uid": "f1", "title": "SQL injection in login", "description": "auth bypass OR 1=1",
         "vulnerability_class": "sqli", "severity": "high"},
        {"uid": "f2", "title": "Unrelated XSS", "description": "reflected", "severity": "low"},
    ])
    items = _run(GraphFindingSource(graph).fetch(
        "scan the web target for SQL injection and auth bypass", tenant_id="t1", limit=5))
    citations = {i.citation for i in items}
    assert "graph/finding/f1" in citations
    assert "graph/finding/f2" not in citations
    # The full goal string is tried first, then distinct keywords.
    assert graph.probes[0] == "scan the web target for SQL injection and auth bypass"
    assert "sqli" in graph.probes or "injection" in graph.probes


def test_graph_source_merges_duplicate_hits_across_probes():
    graph = _SubstringGraph([
        {"uid": "f1", "title": "SQL injection in login", "description": "auth bypass",
         "vulnerability_class": "sqli", "severity": "high"},
    ])
    items = _run(GraphFindingSource(graph).fetch("sqli injection login", tenant_id="t1", limit=5))
    assert len(items) == 1
    assert items[0].citation == "graph/finding/f1"


def test_graph_source_survives_a_probe_error():
    class _Flaky:
        def __init__(self):
            self.calls = 0

        async def search_findings(self, text, limit=20, tenant_id=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("backend blip")
            if text == "login":
                return [{"finding": {"uid": "f9", "title": "login issue"}, "score": 1.0}]
            return []

    items = _run(GraphFindingSource(_Flaky()).fetch("login", tenant_id="t1", limit=5))
    assert [i.citation for i in items] == ["graph/finding/f9"]


class _Episode:
    def __init__(self, eid, target, goal, summary):
        self.episode_id = eid
        self.target = target
        self.goal = goal
        self.summary = summary


class _Episodic:
    def __init__(self, episodes):
        self.episodes = episodes

    def get_all(self, tenant_id="default"):
        return self.episodes


def test_episodic_source_ranks_by_keyword_overlap():
    episodic = _Episodic([
        _Episode("e1", "example.com", "scan web target", "found open admin panel"),
        _Episode("e2", "db.local", "tune database", "index rebuild"),
    ])
    items = _run(EpisodicContextSource(episodic).fetch("scan web target", tenant_id="t1", limit=5))
    assert len(items) == 1
    assert items[0].citation == "episode/e1"
    assert "example.com" in items[0].content


# ---------------------------------------------------------------------------
# Fusion behaviour
# ---------------------------------------------------------------------------


class _StaticSource:
    def __init__(self, name, items):
        self.name = name
        self.items = items

    async def fetch(self, query, *, tenant_id, limit):
        return self.items[:limit]


def test_fusion_ranks_multi_source_hits_higher():
    # "shared fact" appears in both sources; "only here" in one.
    a = _StaticSource("alpha", [
        ContextItem("alpha", "only here", "a/1", 0.9),
        ContextItem("alpha", "shared fact", "a/2", 0.5),
    ])
    b = _StaticSource("beta", [
        ContextItem("beta", "shared fact", "b/1", 0.4),
    ])
    pack = _run(MultiContextAssembler([a, b]).assemble("q", tenant_id="t1"))
    assert pack.items[0].content == "shared fact"
    assert set(pack.items[0].metadata["sources"]) == {"alpha", "beta"}
    assert pack.sources_used == ("alpha", "beta")


def test_dedup_collapses_same_content_across_sources():
    a = _StaticSource("alpha", [ContextItem("alpha", "Duplicated Fact!", "a/1", 0.5)])
    b = _StaticSource("beta", [ContextItem("beta", "duplicated fact", "b/1", 0.5)])
    pack = _run(MultiContextAssembler([a, b]).assemble("q", tenant_id="t1"))
    assert len(pack.items) == 1
    assert pack.items[0].citation == "a/1; b/1"


def test_failing_source_is_recorded_not_fatal():
    class _Boom:
        name = "boom"

        async def fetch(self, query, *, tenant_id, limit):
            raise RuntimeError("backend down")

    good = _StaticSource("good", [ContextItem("good", "real fact", "g/1", 0.5)])
    pack = _run(MultiContextAssembler([_Boom(), good]).assemble("q", tenant_id="t1"))
    assert pack.sources_failed == ("boom",)
    assert pack.sources_used == ("good",)
    assert len(pack.items) == 1


def test_empty_sources_yield_empty_pack_and_no_render():
    pack = _run(MultiContextAssembler([_StaticSource("x", [])]).assemble("q", tenant_id="t1"))
    assert not pack
    assert pack.items == ()
    assert render_multi_context(pack) == ""


def test_bounds_are_enforced():
    many = _StaticSource(
        "big", [ContextItem("big", f"fact number {i}", f"b/{i}", 0.5) for i in range(50)]
    )
    pack = _run(
        MultiContextAssembler([many], max_items=3, max_chars=1000).assemble(
            "q", tenant_id="t1"
        )
    )
    assert len(pack.items) == 3


def test_max_chars_truncates_content():
    src = _StaticSource("big", [ContextItem("big", "x" * 500, "b/1", 0.5)])
    pack = _run(
        MultiContextAssembler([src], max_chars=100).assemble("q", tenant_id="t1")
    )
    assert len(pack.items[0].content) <= 100


def test_invalid_constructor_args_rejected():
    with pytest.raises(ValueError):
        MultiContextAssembler([], max_items=0)
    with pytest.raises(ValueError):
        MultiContextAssembler([], max_chars=0)
    with pytest.raises(ValueError):
        MultiContextAssembler([], per_source_limit=0)


def test_empty_query_and_tenant_rejected():
    asm = MultiContextAssembler([_StaticSource("x", [])])
    with pytest.raises(ValueError):
        _run(asm.assemble("   ", tenant_id="t1"))
    with pytest.raises(ValueError):
        _run(asm.assemble("q", tenant_id="  "))


def test_tenant_is_forwarded_unchanged_to_sources():
    seen = {}

    class _Spy:
        name = "spy"

        async def fetch(self, query, *, tenant_id, limit):
            seen["tenant_id"] = tenant_id
            seen["limit"] = limit
            return []

    _run(MultiContextAssembler([_Spy()], per_source_limit=4).assemble("q", tenant_id="tenant-z"))
    assert seen["tenant_id"] == "tenant-z"
    assert seen["limit"] == 4


def test_render_marks_untrusted_and_lists_sources():
    src = _StaticSource("alpha", [ContextItem("alpha", "a fact", "a/1", 0.5)])
    pack = _run(MultiContextAssembler([src]).assemble("q", tenant_id="t1"))
    rendered = render_multi_context(pack)
    assert "MULTI-CONTEXT" in rendered
    assert "untrusted-evidence" in rendered
    assert "alpha" in rendered


def test_build_default_sources_omits_missing_stores():
    sources = build_default_sources(vector_memory=_Vector([]))
    assert [s.name for s in sources] == ["vector_memory"]
    assert build_default_sources() == []


# ---------------------------------------------------------------------------
# Agent wiring: the block actually reaches the reasoning prompt
# ---------------------------------------------------------------------------


def test_agent_injects_multi_context_into_reasoning_prompt():
    from unittest.mock import MagicMock

    from sonic.computer_use.agent import ComputerUseAgent
    from sonic.memory.multi_context import MultiContextPack

    src = _StaticSource("alpha", [ContextItem("alpha", "cached secret in /tmp/x", "a/1", 0.5)])
    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=None,
        initial_context={},
        multi_context_sources=[src],
    )
    agent._last_screenshot_b64 = ""
    # Simulate the async pre-fetch that _llm_choose_action performs.
    items = _run(src.fetch("goal", tenant_id="default", limit=6))
    pack = MultiContextPack(
        query="goal", items=tuple(items), sources_used=("alpha",), sources_failed=()
    )
    agent._multi_context_pack = pack
    agent._multi_context_block = render_multi_context(pack)
    agent._multi_context_goal = "do the task"
    _, ctx = agent._build_reasoning_context("do the task", MagicMock(), 1, "", "")
    assert "MULTI-CONTEXT" in ctx
    assert "cached secret in /tmp/x" in ctx


def test_agent_omits_multi_context_for_a_different_goal():
    from unittest.mock import MagicMock

    from sonic.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=None,
        initial_context={},
        multi_context_sources=[],
    )
    agent._last_screenshot_b64 = ""
    agent._multi_context_block = "=== MULTI-CONTEXT ... ==="
    agent._multi_context_goal = "some other goal"
    _, ctx = agent._build_reasoning_context("do the task", MagicMock(), 1, "", "")
    assert "MULTI-CONTEXT" not in ctx


def test_agent_with_no_sources_adds_no_multi_context_noise():
    from unittest.mock import MagicMock

    from sonic.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent(
        computer_provider=MagicMock(), llm_router=None, initial_context={}
    )
    agent._last_screenshot_b64 = ""
    _, ctx = agent._build_reasoning_context("goal", MagicMock(), 1, "", "")
    assert "MULTI-CONTEXT" not in ctx


# ---------------------------------------------------------------------------
# Agentic multi-source retrieval (LLM query planning over fused sources)
# ---------------------------------------------------------------------------


class _PlannerRouter:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def complete(self, request, *, task_type):
        self.requests.append(request)
        from types import SimpleNamespace

        return SimpleNamespace(content=self.replies.pop(0) if self.replies else '{"sufficient": true, "queries": []}')


class _QueryAwareSource:
    """Returns different items per query so refinement is observable."""

    def __init__(self, name, by_query):
        self.name = name
        self.by_query = by_query

    async def fetch(self, query, *, tenant_id, limit):
        return list(self.by_query.get(query, []))[:limit]


def test_agentic_multi_context_refines_query_and_merges():
    from sonic.memory.multi_context import AgenticMultiContextRetriever

    src = _QueryAwareSource("alpha", {
        "objective": [ContextItem("alpha", "first partial evidence", "a/1", 0.6)],
        "refined missing concept": [
            ContextItem("alpha", "additional evidence", "a/2", 0.8),
        ],
    })
    router = _PlannerRouter([
        '{"sufficient": false, "queries": ["refined missing concept"]}',
        '{"sufficient": true, "queries": []}',
    ])
    retriever = AgenticMultiContextRetriever(MultiContextAssembler([src]), router)
    pack = _run(retriever.retrieve("objective", tenant_id="t1"))
    contents = {i.content for i in pack.items}
    assert "first partial evidence" in contents
    assert "additional evidence" in contents
    assert len(router.requests) == 2


def test_agentic_multi_context_stops_when_sufficient():
    from sonic.memory.multi_context import AgenticMultiContextRetriever

    src = _QueryAwareSource("alpha", {
        "objective": [ContextItem("alpha", "enough", "a/1", 0.9)],
    })
    router = _PlannerRouter(['{"sufficient": true, "queries": ["ignored"]}'])
    retriever = AgenticMultiContextRetriever(MultiContextAssembler([src]), router)
    pack = _run(retriever.retrieve("objective", tenant_id="t1"))
    assert [i.content for i in pack.items] == ["enough"]
    assert len(router.requests) == 1


def test_agentic_multi_context_keeps_initial_if_planner_fails():
    from sonic.memory.multi_context import AgenticMultiContextRetriever

    src = _QueryAwareSource("alpha", {
        "objective": [ContextItem("alpha", "initial hit", "a/1", 0.9)],
    })
    router = _PlannerRouter(["not json at all"])
    retriever = AgenticMultiContextRetriever(MultiContextAssembler([src]), router)
    pack = _run(retriever.retrieve("objective", tenant_id="t1"))
    assert [i.content for i in pack.items] == ["initial hit"]


def test_agentic_multi_context_bounded_and_rejects_repeats():
    from sonic.memory.multi_context import AgenticMultiContextRetriever

    src = _QueryAwareSource("alpha", {"objective": [ContextItem("alpha", "hit", "a/1", 0.9)]})
    # Always asks for the same query it already ran; must not loop forever.
    router = _PlannerRouter([
        '{"sufficient": false, "queries": ["objective"]}',
        '{"sufficient": false, "queries": ["objective"]}',
    ])
    retriever = AgenticMultiContextRetriever(
        MultiContextAssembler([src]), router, max_rounds=2, max_queries=4
    )
    pack = _run(retriever.retrieve("objective", tenant_id="t1"))
    assert [i.content for i in pack.items] == ["hit"]


def test_agentic_multi_context_plan_parser_rejects_malformed():
    from sonic.memory.multi_context import AgenticMultiContextRetriever

    parse = AgenticMultiContextRetriever._parse_plan
    assert parse("no json") is None
    assert parse('{"queries": ["a"]}') is None  # missing sufficient
    assert parse('{"sufficient": "yes", "queries": []}') is None  # wrong type
    assert parse('{"sufficient": false, "queries": "nope"}') is None
    assert parse('{"sufficient": false, "queries": [1, 2]}') is None
    assert parse('{"sufficient": false, "queries": ["ok"]}') == (False, ["ok"])


def test_agentic_multi_context_constructor_bounds():
    from sonic.memory.multi_context import AgenticMultiContextRetriever

    with pytest.raises(ValueError):
        AgenticMultiContextRetriever(MultiContextAssembler([]), None, max_rounds=9)
    with pytest.raises(ValueError):
        AgenticMultiContextRetriever(MultiContextAssembler([]), None, max_queries=0)


def test_boss_forwards_multi_context_sources_to_worker():
    from unittest.mock import MagicMock

    from sonic.computer_use.boss import BossAgent

    src = _StaticSource("alpha", [])
    boss = BossAgent(
        computer_provider=MagicMock(),
        llm_router=MagicMock(),
        multi_context_sources=[src],
        tenant_id="t1",
    )
    assert boss.multi_context_sources == [src]
