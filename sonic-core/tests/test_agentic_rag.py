"""Tests for bounded, source-grounded agentic retrieval."""

import asyncio
import os
from types import SimpleNamespace

from sonic.being.lessons import (
    Lesson,
    LessonKind,
    LessonsLedger,
    inject_into_context,
)
from sonic.memory.agentic import AgenticRetriever


class _Memory:
    def __init__(self):
        self.calls = []
        self.rows = {
            "initial task query": [
                self._row("lesson-a", "first partial evidence", 0.7, "tenant-a"),
            ],
            "missing follow-up concept": [
                self._row("lesson-b", "additional evidence", 0.9, "tenant-a"),
                self._row("private", "another tenant's evidence", 1.0, "tenant-b"),
            ],
        }

    @staticmethod
    def _row(doc_id, text, score, tenant_id):
        return {
            "doc_id": doc_id,
            "text": text,
            "score": score,
            "citation": f"memory/{doc_id}",
            "metadata": {"lesson_id": doc_id},
            "tenant_id": tenant_id,
        }

    def search(self, query, *, top_k, score_threshold, tenant_id):
        self.calls.append((query, tenant_id))
        return [
            row
            for row in self.rows.get(query, [])
            if row["tenant_id"] == tenant_id and row["score"] >= score_threshold
        ][:top_k]


class _Router:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, request, *, task_type):
        self.requests.append((request, task_type))
        return SimpleNamespace(content=next(self.replies))


def test_agentic_retrieval_refines_query_and_merges_real_cited_sources():
    memory = _Memory()
    router = _Router(
        [
            '{"sufficient": false, "queries": ["missing follow-up concept"]}',
            '{"sufficient": true, "queries": []}',
        ]
    )

    result = asyncio.run(
        AgenticRetriever(memory, router).retrieve(
            "initial task query",
            tenant_id="tenant-a",
            top_k=5,
        )
    )

    assert result.queries == (
        "initial task query",
        "missing follow-up concept",
    )
    assert result.rounds == 2
    assert result.stop_reason == "sufficient_evidence"
    assert result.planner_used is True
    assert [doc["doc_id"] for doc in result.documents] == ["lesson-b", "lesson-a"]
    assert all(doc["tenant_id"] == "tenant-a" for doc in result.documents)
    assert result.documents[0]["citation"] == "memory/lesson-b"
    assert memory.calls == [
        ("initial task query", "tenant-a"),
        ("missing follow-up concept", "tenant-a"),
    ]
    plan_request = router.requests[0][0]
    assert router.requests[0][1] == "reasoning"
    assert plan_request.max_tokens == 256
    assert "<untrusted-memory-evidence>" in plan_request.messages[-1].content


def test_agentic_retrieval_stops_when_evidence_is_sufficient():
    memory = _Memory()
    router = _Router(['{"sufficient": true, "queries": ["should not run"]}'])

    result = asyncio.run(
        AgenticRetriever(memory, router).retrieve(
            "initial task query", tenant_id="tenant-a"
        )
    )

    assert result.stop_reason == "sufficient_evidence"
    assert result.rounds == 1
    assert result.queries == ("initial task query",)
    assert len(memory.calls) == 1


def test_agentic_retrieval_fuses_ranks_across_queries():
    class _RankedMemory:
        def search(self, query, *, top_k, score_threshold, tenant_id):
            if query == "initial task query":
                return [
                    {"doc_id": "single", "text": "one-query match", "score": 0.99},
                    {"doc_id": "recurring", "text": "multi-query match", "score": 0.70},
                ]
            return [
                {"doc_id": "recurring", "text": "multi-query match", "score": 0.60},
                {"doc_id": "single-2", "text": "other one-query match", "score": 0.99},
            ]

    router = _Router(
        [
            '{"sufficient": false, "queries": ["alternate formulation"]}',
        ]
    )
    result = asyncio.run(
        AgenticRetriever(
            _RankedMemory(), router, max_rounds=1, score_threshold=0.0
        ).retrieve("initial task query", tenant_id="tenant-a", top_k=3)
    )

    assert result.documents[0]["doc_id"] == "recurring"
    assert result.documents[0]["matched_queries"] == [
        "initial task query",
        "alternate formulation",
    ]
    assert result.documents[0]["query_ranks"] == {
        "initial task query": 2,
        "alternate formulation": 1,
    }
    assert result.documents[0]["rrf_score"] > result.documents[1]["rrf_score"]


def test_agentic_retrieval_rejects_repeated_or_invalid_queries_and_is_bounded():
    memory = _Memory()
    router = _Router(
        [
            '{"sufficient": false, "queries": '
            '["initial task query", "missing follow-up concept"]}',
            '{"sufficient": false, "queries": ["missing follow-up concept"]}',
        ]
    )

    result = asyncio.run(
        AgenticRetriever(memory, router, max_rounds=2, max_queries=2).retrieve(
            "initial task query", tenant_id="tenant-a"
        )
    )

    assert result.queries == (
        "initial task query",
        "missing follow-up concept",
    )
    assert result.stop_reason == "no_useful_queries"
    assert result.rounds == 2
    assert len(memory.calls) == 2


def test_agentic_retrieval_keeps_initial_results_if_planner_fails():
    class _FailingRouter:
        async def complete(self, request, *, task_type):
            raise RuntimeError("planner unavailable")

    memory = _Memory()
    result = asyncio.run(
        AgenticRetriever(memory, _FailingRouter()).retrieve(
            "initial task query", tenant_id="tenant-a"
        )
    )

    assert [doc["doc_id"] for doc in result.documents] == ["lesson-a"]
    assert result.stop_reason == "planner_unavailable"
    assert result.rounds == 1
    assert result.planner_used is True


def test_agentic_plan_parser_rejects_malformed_or_unbounded_output():
    parsed = AgenticRetriever._parse_plan(
        'prefix ```json\n{"sufficient": false, "queries": '
        '["  broader   query ", "second query", "third query"]}\n```'
    )

    assert parsed == (False, ["broader query", "second query"])
    assert AgenticRetriever._parse_plan(
        '{"sufficient": "yes", "queries": ["search"]}'
    ) is None
    assert AgenticRetriever._parse_plan(
        '{"sufficient": false, "queries": [false]}'
    ) is None


def test_lessons_ledger_uses_agentic_hits_and_keeps_tenant_scope(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SONIC_DATA_DIR", os.fspath(tmp_path))

    class _LessonMemory:
        def __init__(self):
            self.tenant_queries = []

        def search(self, query, *, top_k, score_threshold, tenant_id):
            self.tenant_queries.append(tenant_id)
            return [
                {
                    "doc_id": "lesson-private",
                    "text": "Relevant prior outcome",
                    "score": 0.9,
                    "metadata": {"lesson_id": "lesson-private"},
                    "tenant_id": tenant_id,
                    "citation": "lesson:lesson-private",
                }
            ]

    ledger_memory = _LessonMemory()
    ledger = LessonsLedger(
        tenant_id="tenant-a",
        agent_id="rag-test",
        semantic_memory=ledger_memory,
    )
    relevant = Lesson(
        lesson_id="lesson-private",
        kind=LessonKind.REUSE,
        goal="an unrelated old goal",
        approach="reuse the verified approach",
        evidence="confirmed result",
        created_at="2026-10-08T00:00:00Z",
    )
    ledger._lessons = [relevant]
    router = _Router(['{"sufficient": true, "queries": []}'])

    results = asyncio.run(
        ledger.relevant_agentic("current objective", router, k=1)
    )

    assert results == [relevant]
    assert ledger_memory.tenant_queries == ["tenant-a"]
    assert "[memory:lesson-private]" in inject_into_context(results)
