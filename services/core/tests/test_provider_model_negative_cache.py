"""Regression tests for the provider model negative-cache.

A decommissioned model id on a provider endpoint (e.g. a retired NIM model that
404s for the account) used to cost a wasted round-trip on EVERY request before
the model-level fallback recovered. The provider now remembers not-found models
and routes straight to a working fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sonic.llm.providers.custom import CustomLLMProvider
from sonic.llm.schemas import LLMRequest, LLMResponse, Message, MessageRole


def _provider(**kwargs) -> CustomLLMProvider:
    kwargs.setdefault("fallback_models", ["nvidia/live-model"])
    return CustomLLMProvider(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="test-key",
        default_model="nvidia/dead-model",
        **kwargs,
    )


def _request(model: str = "nvidia/dead-model") -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        model=model,
    )


def _not_found_error() -> Exception:
    error = Exception("Error code: 404 - model not found")
    error.status_code = 404  # type: ignore[attr-defined]
    return error


def _ok_response(model: str) -> LLMResponse:
    return LLMResponse(
        content="ok",
        model=model,
        provider="nvidia",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )


@pytest.mark.asyncio
async def test_dead_model_is_remembered_and_skipped_next_time():
    provider = _provider()
    live = _ok_response("nvidia/live-model")

    async def complete(request: LLMRequest):
        if request.model == "nvidia/dead-model":
            raise _not_found_error()
        return live

    provider._complete_openai = AsyncMock(side_effect=complete)  # type: ignore[method-assign]
    async def _passthrough(request, call):
        return await call(request)

    provider._with_retry = AsyncMock(side_effect=_passthrough)  # type: ignore[method-assign]

    first = await provider.complete(_request())
    assert first.model == "nvidia/live-model"
    assert "nvidia/dead-model" in provider._unavailable_models

    provider._complete_openai.reset_mock()
    second = await provider.complete(_request())
    assert second.model == "nvidia/live-model"
    # The dead model must not be attempted again.
    for call in provider._complete_openai.await_args_list:
        assert call.args[0].model != "nvidia/dead-model"


@pytest.mark.asyncio
async def test_no_fallback_model_raises_honest_error():
    provider = _provider(fallback_models=[])

    async def complete(request: LLMRequest):
        raise _not_found_error()

    provider._complete_openai = AsyncMock(side_effect=complete)  # type: ignore[method-assign]
    async def _passthrough(request, call):
        return await call(request)

    provider._with_retry = AsyncMock(side_effect=_passthrough)  # type: ignore[method-assign]

    with pytest.raises(Exception) as excinfo:
        await provider.complete(_request())
    assert "404" in str(excinfo.value) or "not found" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_non_not_found_fallback_error_propagates():
    """A fallback failing for a real reason must not be silently skipped."""
    provider = _provider()

    async def complete(request: LLMRequest):
        if request.model == "nvidia/dead-model":
            raise _not_found_error()
        raise RuntimeError("upstream 503")

    provider._complete_openai = AsyncMock(side_effect=complete)  # type: ignore[method-assign]
    async def _passthrough(request, call):
        return await call(request)

    provider._with_retry = AsyncMock(side_effect=_passthrough)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="upstream 503"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_cached_dead_model_with_no_working_fallback_is_honest():
    provider = _provider()
    provider._unavailable_models.add("nvidia/dead-model")

    async def complete(request: LLMRequest):
        raise _not_found_error()

    provider._complete_openai = AsyncMock(side_effect=complete)  # type: ignore[method-assign]
    async def _passthrough(request, call):
        return await call(request)

    provider._with_retry = AsyncMock(side_effect=_passthrough)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="unavailable"):
        await provider.complete(_request())
