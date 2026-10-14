"""Regression tests for model-aware semantic and hybrid retrieval."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sonic.memory.embeddings import (
    EmbeddingError,
    OpenAICompatibleEmbeddingProvider,
    configured_embedding_provider,
)
from sonic.memory.retrieval_eval import RetrievalCase, evaluate_retrieval
from sonic.memory.vector import VectorMemory


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from sonic.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FixedEmbeddingProvider:
    def __init__(self, model_id: str, vectors: dict[str, list[float]]):
        self._model_id = model_id
        self.vectors = vectors

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts):
        return [self.vectors[text] for text in texts]


def test_semantic_retrieval_finds_paraphrase_and_returns_source():
    provider = _FixedEmbeddingProvider(
        "fixture-english-v1",
        {
            "forgot account password": [1.0, 0.0],
            "reset credentials": [1.0, 0.0],
            "weather forecast": [0.0, 1.0],
        },
    )
    memory = VectorMemory(persist=False, embedding_provider=provider)
    memory.index_document(
        "doc-reset",
        "forgot account password",
        {"source": "support/manual.md#reset"},
    )
    memory.index_document("doc-weather", "weather forecast", {})

    results = memory.search("reset credentials", top_k=2, score_threshold=0)

    assert results[0]["doc_id"] == "doc-reset"
    assert results[0]["embedding_model"] == "fixture-english-v1"
    assert results[0]["embedding_dimension"] == 2
    assert results[0]["citation"] == "support/manual.md#reset"
    assert results[0]["retrieval_mode"] == "semantic+lexical"
    assert results[0]["score"] > results[1]["score"]


def test_tenant_filter_is_applied_to_hybrid_retrieval():
    provider = _FixedEmbeddingProvider(
        "fixture-v1",
        {"same query": [1.0, 0.0], "private record": [1.0, 0.0]},
    )
    memory = VectorMemory(persist=False, embedding_provider=provider)
    memory.index_document("tenant-a-doc", "private record", tenant_id="tenant-a")
    memory.index_document("tenant-b-doc", "private record", tenant_id="tenant-b")

    results = memory.search("same query", tenant_id="tenant-a")

    assert [result["doc_id"] for result in results] == ["tenant-a-doc"]


def test_search_does_not_compare_vectors_from_different_models():
    first = _FixedEmbeddingProvider("model-a", {"shared terms": [1.0, 0.0]})
    memory = VectorMemory(persist=False, embedding_provider=first)
    memory.index_document("old", "shared terms")

    second = _FixedEmbeddingProvider("model-b", {"shared terms": [0.0, 1.0]})
    memory.embedding_provider = second
    memory.embedding_model = second.model_id

    result = memory.search("shared terms")[0]

    assert result["doc_id"] == "old"
    assert result["embedding_model"] == "model-a"
    assert result["retrieval_mode"] == "lexical_only"
    assert result["score"] == pytest.approx(0.3)


def test_reembed_all_updates_rows_to_active_model():
    first = _FixedEmbeddingProvider("model-a", {"document text": [1.0, 0.0]})
    memory = VectorMemory(persist=False, embedding_provider=first)
    memory.index_document("doc", "document text")
    second = _FixedEmbeddingProvider("model-b", {"document text": [0.0, 2.0]})
    memory.embedding_provider = second
    memory.embedding_model = second.model_id

    assert memory.reembed_all(batch_size=1) == 1
    assert memory.documents[("default", "doc")].embedding_model == "model-b"
    assert memory.documents[("default", "doc")].vector == [0.0, 1.0]
    assert memory.documents[("default", "doc")].embedding_dimension == 2


def test_reembed_requires_real_embedding_provider():
    memory = VectorMemory(persist=False)
    with pytest.raises(ValueError, match="configured real embedding provider"):
        memory.reembed_all()


def test_persisted_vectors_are_model_scoped_until_reembedded(tmp_path: Path):
    db_path = str(tmp_path / "vector-memory.sqlite3")
    first = _FixedEmbeddingProvider("model-a", {"document text": [1.0, 0.0]})
    original = VectorMemory(db_path=db_path, embedding_provider=first)
    original.index_document("doc", "document text", {"source": "manual.md"})
    original._db.close()

    second = _FixedEmbeddingProvider("model-b", {"document text": [0.0, 1.0]})
    migrated = VectorMemory(db_path=db_path, embedding_provider=second)
    assert migrated.search("document text")[0]["retrieval_mode"] == "lexical_only"

    assert migrated.reembed_all(batch_size=1) == 1
    result = migrated.search("document text")[0]

    assert result["retrieval_mode"] == "semantic+lexical"
    assert result["embedding_model"] == "model-b"
    assert result["embedding_dimension"] == 2
    row = migrated._db.execute(
        "SELECT embedding_model, embedding_dimension FROM vector_documents "
        "WHERE tenant_id = ? AND doc_id = ?",
        ("default", "doc"),
    ).fetchone()
    assert row == ("model-b", 2)
    migrated._db.close()


def test_legacy_database_migrates_and_labels_hash_vectors(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE vector_documents (doc_id TEXT, text TEXT NOT NULL, "
        "metadata_json TEXT NOT NULL DEFAULT '{}', vector_json TEXT NOT NULL DEFAULT '[]', "
        "tenant_id TEXT NOT NULL DEFAULT 'default', PRIMARY KEY (tenant_id, doc_id))"
    )
    connection.execute(
        "INSERT INTO vector_documents VALUES (?, ?, ?, ?, ?)",
        ("legacy-doc", "stored text", "{}", json.dumps([1.0, 0.0, 0.0]), "tenant"),
    )
    connection.commit()
    connection.close()

    memory = VectorMemory(dim=3, db_path=str(db_path))
    result = memory.search("stored text", tenant_id="tenant")[0]

    assert result["embedding_model"] == "legacy-hash-ngram-v1:3"
    assert result["embedding_dimension"] == 3
    assert result["retrieval_mode"] == "legacy_hash+lexical"
    assert result["citation"] == "legacy-doc"
    memory._db.close()


def test_active_model_dimension_mismatch_fails_loudly():
    provider = _FixedEmbeddingProvider(
        "model-v1",
        {"one dimension": [1.0, 0.0], "two dimensions": [1.0, 0.0, 0.0]},
    )
    memory = VectorMemory(persist=False, embedding_provider=provider)
    memory.index_document("doc", "one dimension")
    with pytest.raises(ValueError, match="dimension changed"):
        memory.index_document("doc-2", "two dimensions")


def test_openai_compatible_provider_sorts_and_normalizes_response_vectors():
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.0, 5.0]},
            {"index": 0, "embedding": [3.0, 4.0]},
        ]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = response

    with patch("httpx.Client", return_value=client):
        provider = OpenAICompatibleEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="local-embedding-model",
            api_key="test-token",
        )
        vectors = provider.embed(["first", "second"])

    assert vectors == [[0.6, 0.8], [0.0, 1.0]]
    client.post.assert_called_once_with(
        "http://localhost:11434/v1/embeddings",
        headers={"Authorization": "Bearer test-token"},
        json={"model": "local-embedding-model", "input": ["first", "second"]},
    )


def test_openai_compatible_provider_fails_explicitly_on_invalid_response():
    response = MagicMock()
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.0, 0.0]}]}
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = response

    with patch("httpx.Client", return_value=client):
        provider = OpenAICompatibleEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="local-embedding-model",
        )
        with pytest.raises(EmbeddingError, match="zero vector"):
            provider.embed(["text"])


def test_configured_embedding_provider_supports_local_openai_compatible_service(
    monkeypatch,
):
    monkeypatch.setenv("SONIC_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SONIC_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("SONIC_EMBEDDING_MODEL", "local-embed")
    monkeypatch.delenv("SONIC_EMBEDDING_API_KEY", raising=False)

    provider = configured_embedding_provider()

    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.model_id == "openai-compatible:http://localhost:11434/v1:local-embed"


def test_configured_semantic_provider_does_not_silently_fallback(monkeypatch):
    monkeypatch.setenv("SONIC_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.delenv("SONIC_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("SONIC_EMBEDDING_MODEL", raising=False)

    with pytest.raises(ValueError, match="base URL and model are required"):
        configured_embedding_provider()


def test_retrieval_evaluator_reports_recall_and_mrr():
    provider = _FixedEmbeddingProvider(
        "fixture-eval-v1",
        {
            "reset secret": [1.0, 0.0],
            "weather": [0.0, 1.0],
            "forgot password": [1.0, 0.0],
            "forecast": [0.0, 1.0],
        },
    )
    memory = VectorMemory(persist=False, embedding_provider=provider)
    memory.index_document("password", "forgot password")
    memory.index_document("forecast", "forecast")

    metrics = evaluate_retrieval(
        memory,
        (
            RetrievalCase("reset secret", frozenset({"password"})),
            RetrievalCase("weather", frozenset({"forecast"})),
        ),
        top_k=1,
    )

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["embedding_model"] == "fixture-eval-v1"
