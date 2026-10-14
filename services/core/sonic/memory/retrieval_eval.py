"""Small, repeatable evaluation set for configured semantic retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sonic.memory.embeddings import configured_embedding_provider
from sonic.memory.vector import VectorMemory


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_doc_ids: frozenset[str]


_DOCUMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "fresh_visual_target",
        "Before a visible desktop action, capture a current screenshot and verify "
        "the screen still matches the image used to choose the target. If pixels "
        "changed, observe again and re-ground the point before clicking.",
        "docs/SONIC_COMPUTER_USER_GOAL.md#fresh-actions",
    ),
    (
        "visual_text_focus",
        "To type in a desktop input, click the visually grounded field, capture a "
        "new screenshot, and confirm a visible caret or focus indicator. Do not "
        "send text when focus is uncertain.",
        "sonic/computer_use/agent.py#GUI_TYPE",
    ),
    (
        "judge_completion",
        "A worker completion proposal is not proof. Verify the requested result "
        "from fresh desktop evidence and action receipts with an independent Judge.",
        "sonic/computer_use/boss.py#Judge",
    ),
    (
        "tenant_vector_isolation",
        "Persistent vector memory searches must filter documents by tenant so one "
        "customer cannot retrieve another customer's stored information.",
        "sonic/memory/vector.py#tenant_id",
    ),
    (
        "embedding_model_migration",
        "Embedding vectors from different models or dimensions are not comparable. "
        "Record the model identity and re-embed stored documents before using a new "
        "embedding model for vector similarity.",
        "sonic/memory/vector.py#reembed_all",
    ),
    (
        "provider_failure",
        "If the configured embedding service is unavailable or returns invalid "
        "vectors, report the failure. Do not silently replace semantic vectors "
        "with a different representation.",
        "sonic/memory/embeddings.py#EmbeddingError",
    ),
)

_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        "What if the screen changes while the model is deciding where to click?",
        frozenset({"fresh_visual_target"}),
    ),
    RetrievalCase(
        "How can SONIC make sure text goes into the intended box?",
        frozenset({"visual_text_focus"}),
    ),
    RetrievalCase(
        "Who independently confirms that the requested desktop task really finished?",
        frozenset({"judge_completion"}),
    ),
    RetrievalCase(
        "How do we prevent one tenant from seeing another tenant's memories?",
        frozenset({"tenant_vector_isolation"}),
    ),
    RetrievalCase(
        "Can I compare saved embeddings after changing the vector model?",
        frozenset({"embedding_model_migration"}),
    ),
    RetrievalCase(
        "Should the system quietly switch to fake vectors when the embedding API errors?",
        frozenset({"provider_failure"}),
    ),
)


def evaluate_retrieval(
    memory: VectorMemory,
    cases: tuple[RetrievalCase, ...] = _CASES,
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Compute Recall@K and MRR against known relevant document IDs."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not cases:
        raise ValueError("at least one retrieval evaluation case is required")

    hits = 0
    reciprocal_ranks: list[float] = []
    details: list[dict[str, Any]] = []
    for case in cases:
        results = memory.search(
            case.query,
            top_k=top_k,
            score_threshold=0.0,
        )
        ids = [str(result["doc_id"]) for result in results]
        first_relevant_rank = next(
            (
                index
                for index, doc_id in enumerate(ids, start=1)
                if doc_id in case.relevant_doc_ids
            ),
            None,
        )
        if first_relevant_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            hits += 1
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        details.append(
            {
                "query": case.query,
                "expected_doc_ids": sorted(case.relevant_doc_ids),
                "retrieved_doc_ids": ids,
                "first_relevant_rank": first_relevant_rank,
            }
        )

    return {
        "cases": len(cases),
        "top_k": top_k,
        "recall_at_k": round(hits / len(cases), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "embedding_model": memory.embedding_model,
        "results": details,
    }


def run_configured_evaluation() -> dict[str, Any]:
    """Index the fixed corpus in memory and evaluate the configured model."""
    provider = configured_embedding_provider()
    if provider is None:
        raise RuntimeError(
            "Configure a real embedding provider before running retrieval evaluation; "
            "legacy hash vectors are not semantic embeddings."
        )
    memory = VectorMemory(persist=False, embedding_provider=provider)
    for doc_id, text, source in _DOCUMENTS:
        memory.index_document(
            doc_id,
            text,
            metadata={"source": source, "evaluation_corpus": "sonic-rag-v1"},
        )
    return evaluate_retrieval(memory)


if __name__ == "__main__":
    print(json.dumps(run_configured_evaluation(), indent=2, ensure_ascii=False))
