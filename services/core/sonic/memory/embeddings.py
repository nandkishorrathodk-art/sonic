"""Embedding providers for semantic memory retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlsplit


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return valid vectors."""


class EmbeddingProvider(Protocol):
    """Produces comparable vectors for one stable model identity."""

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize_vector(values: object) -> list[float]:
    """Validate and L2-normalize one provider response vector."""
    if not isinstance(values, (list, tuple)) or not values:
        raise EmbeddingError("embedding provider returned an empty or invalid vector")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in values
    ):
        raise EmbeddingError("embedding provider returned non-finite or non-numeric values")
    vector = [float(value) for value in values]
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0:
        raise EmbeddingError("embedding provider returned a zero vector")
    return [value / norm for value in vector]


class OpenAICompatibleEmbeddingProvider:
    """Embedding client for hosted or local OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or not model.strip()
        ):
            raise ValueError("embedding base URL and model are required")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("embedding timeout must be a finite positive number")
        self.base_url = normalized_url
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds

    @property
    def model_id(self) -> str:
        return f"openai-compatible:{self.base_url}:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": list(texts)},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise EmbeddingError(
                f"embedding request failed for model {self.model!r}: {exc}"
            ) from exc

        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise EmbeddingError("embedding response data count did not match input count")
        try:
            ordered_rows = sorted(rows, key=lambda row: row["index"])
            vectors = [normalize_vector(row["embedding"]) for row in ordered_rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"embedding response schema was invalid: {exc}") from exc
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingError("embedding response contained inconsistent vector dimensions")
        return vectors


class SentenceTransformerEmbeddingProvider:
    """Optional in-process local embedding model; imported only when selected."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("local embedding model name is required")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EmbeddingError(
                "local embeddings require the optional 'rag-local' dependency"
            ) from exc
        self.model_name = model_name.strip()
        self._model = SentenceTransformer(self.model_name)

    @property
    def model_id(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [normalize_vector(vector.tolist()) for vector in vectors]


def configured_embedding_provider() -> EmbeddingProvider | None:
    """Build the explicitly configured semantic provider, if one is selected."""
    from sonic.config import get_settings

    settings = get_settings()
    provider = settings.embedding_provider.strip().lower()
    if provider in {"", "legacy_hash"}:
        return None
    if provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(
            settings.embedding_local_model
        )
    raise ValueError(
        "SONIC_EMBEDDING_PROVIDER must be legacy_hash, openai_compatible, "
        "or sentence_transformers"
    )
