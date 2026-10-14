"""Regression tests for configured LLM request and engagement budgets."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from sonic.llm.providers.custom import CustomLLMProvider
from sonic.llm.router import (
    LLMBudgetEstimateUnavailableError,
    LLMBudgetExceededError,
    ModelRouter,
)
from sonic.llm.schemas import (
    ImageContent,
    LLMChunk,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ProviderName,
    TokenUsage,
)


def _provider(
    *,
    input_rate: float = 0.001,
    output_rate: float = 0.001,
) -> CustomLLMProvider:
    return CustomLLMProvider(
        name="openai",
        base_url="https://provider.invalid/v1",
        api_key="test-key",
        default_model="priced-model",
        cost_per_1k_input=input_rate,
        cost_per_1k_output=output_rate,
        max_retries=0,
    )


def _request(*, engagement_id: str | None = "eng-1", max_tokens: int = 800) -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content="short prompt")],
        max_tokens=max_tokens,
        engagement_id=engagement_id,
    )


def _response(cost: float = 0.0008) -> LLMResponse:
    return LLMResponse(
        content="answer",
        model="priced-model",
        provider=ProviderName.OPENAI,
        usage=TokenUsage(prompt_tokens=12, completion_tokens=800, total_tokens=812),
        cost_usd=cost,
    )


def test_configured_yaml_budget_values_are_loaded():
    models_config = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"
    router = ModelRouter.from_config(models_config, env_vars={})

    assert router.max_per_request_usd == 1.0
    assert router.max_per_engagement_usd == 50.0
    assert router.warn_at_percent == 80


@pytest.mark.asyncio
async def test_per_request_cap_rejects_before_provider_call():
    router = ModelRouter()
    router.max_per_request_usd = 0.0005
    provider = _provider()
    provider.complete_with_timing = AsyncMock(return_value=_response())
    router.add_provider(provider)

    with pytest.raises(LLMBudgetExceededError, match="per-request"):
        await router.complete(_request(engagement_id=None), provider_name="openai")

    provider.complete_with_timing.assert_not_awaited()


@pytest.mark.asyncio
async def test_engagement_warning_is_exposed_in_stats_and_log(monkeypatch):
    from sonic.llm import router as router_module

    warning_log = Mock()
    monkeypatch.setattr(router_module.logger, "warning", warning_log)
    router = ModelRouter()
    router.max_per_request_usd = 0.01
    router.max_per_engagement_usd = 0.0013
    router.warn_at_percent = 80
    provider = _provider()
    provider.complete_with_timing = AsyncMock(return_value=_response())
    router.add_provider(provider)

    result = await router.complete(_request(), provider_name="openai")

    assert result.content == "answer"
    assert len(router.budget_warnings) == 1
    assert router.budget_warnings[0]["engagement_id"] == "eng-1"
    assert router.get_stats()["budget"]["warnings"] == router.budget_warnings
    assert any(
        call.args[0] == "llm_engagement_budget_warning"
        for call in warning_log.call_args_list
    )


@pytest.mark.asyncio
async def test_engagement_cap_reservations_prevent_concurrent_overspend():
    router = ModelRouter()
    router.max_per_request_usd = 0.01
    router.max_per_engagement_usd = 0.0013
    provider = _provider()

    async def slow_completion(_request):
        await asyncio.sleep(0.02)
        return _response()

    provider.complete_with_timing = AsyncMock(side_effect=slow_completion)
    router.add_provider(provider)
    outcomes = await asyncio.gather(
        router.complete(_request(), provider_name="openai"),
        router.complete(_request(), provider_name="openai"),
        return_exceptions=True,
    )

    assert sum(isinstance(item, LLMResponse) for item in outcomes) == 1
    rejected = [item for item in outcomes if isinstance(item, LLMBudgetExceededError)]
    assert len(rejected) == 1
    assert rejected[0].budget == "per-engagement"
    assert provider.complete_with_timing.await_count == 1


@pytest.mark.asyncio
async def test_missing_model_prices_fail_closed_and_config_absence_is_legacy():
    priced_router = ModelRouter()
    priced_router.max_per_request_usd = 1.0
    provider = _provider()
    provider.model_pricing.clear()
    provider.complete_with_timing = AsyncMock(return_value=_response())
    priced_router.add_provider(provider)

    with pytest.raises(LLMBudgetEstimateUnavailableError, match="no complete"):
        await priced_router.complete(_request(), provider_name="openai")
    provider.complete_with_timing.assert_not_awaited()

    legacy_router = ModelRouter()
    legacy_provider = _provider(input_rate=0.0, output_rate=0.0)
    legacy_provider.complete_with_timing = AsyncMock(return_value=_response())
    legacy_router.add_provider(legacy_provider)
    await legacy_router.complete(_request(), provider_name="openai")
    legacy_provider.complete_with_timing.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_cost_model_is_valid_for_budget_estimation():
    router = ModelRouter()
    router.max_per_request_usd = 1.0
    router.max_per_engagement_usd = 1.0
    provider = _provider(input_rate=0.0, output_rate=0.0)
    provider.complete_with_timing = AsyncMock(return_value=_response(cost=0.0))
    router.add_provider(provider)

    response = await router.complete(_request(), provider_name="openai")

    assert response.content == "answer"
    provider.complete_with_timing.assert_awaited_once()
    assert router.cost_records[0].cost_usd == 0.0
    assert router.get_stats()["budget"]["reserved_by_engagement_usd"] == {}


@pytest.mark.asyncio
async def test_stream_budget_is_checked_before_provider_stream_starts():
    router = ModelRouter()
    router.max_per_request_usd = 0.0005
    provider = _provider()
    stream_call = Mock()
    provider.stream = stream_call
    router.add_provider(provider)

    with pytest.raises(LLMBudgetExceededError, match="per-request"):
        async for _ in router.stream(_request(engagement_id=None), provider_name="openai"):
            pass

    stream_call.assert_not_called()


@pytest.mark.asyncio
async def test_stream_skips_text_model_and_preserves_images_for_vision_model():
    router = ModelRouter()
    router.max_per_request_usd = 0.05
    router.max_per_engagement_usd = 0.05
    primary = _provider()
    primary.default_model = "text-model"
    primary.model_pricing["text-model"] = {
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.001,
    }
    fallback = CustomLLMProvider(
        name="vision",
        base_url="https://vision.invalid/v1",
        api_key="test-key",
        default_model="vision-model",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.001,
        max_retries=0,
    )
    router.add_provider(primary)
    router.add_provider(fallback)
    router.routing_rules["vision"] = {
        "provider": "openai",
        "model": "text-model",
        "fallback": ["vision/vision-model"],
    }
    primary_requests = []
    fallback_requests = []

    async def primary_stream(request):
        primary_requests.append(request)
        raise ConnectionError("primary unavailable before output")
        yield  # Make this an async generator.

    async def fallback_stream(request):
        fallback_requests.append(request)
        yield LLMChunk(content="described")
        yield LLMChunk(
            is_final=True,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    primary.stream = primary_stream
    fallback.stream = fallback_stream
    request = _request()
    request.messages[0].images = [ImageContent(base64="aW1hZ2U=")]

    chunks = [chunk async for chunk in router.stream(request)]

    assert [chunk.content for chunk in chunks if chunk.content] == ["described"]
    assert primary_requests == []
    assert fallback_requests[0].messages[0].images == request.messages[0].images
    assert request.messages[0].images[0].base64 == "aW1hZ2U="
    assert len(router.cost_records) == 1
    assert router.cost_records[0].model == "vision-model"
    budget = router.get_stats()["budget"]
    assert budget["uncertain_by_engagement_usd"].get("eng-1", 0) == 0
    assert "eng-1" not in budget["reserved_by_engagement_usd"]


@pytest.mark.asyncio
async def test_completion_skips_text_model_for_vision_fallback():
    router = ModelRouter()
    primary = _provider()
    primary.default_model = "text-model"
    fallback = CustomLLMProvider(
        name="vision",
        base_url="https://vision.invalid/v1",
        api_key="test-key",
        default_model="vision-model",
        max_retries=0,
    )
    router.add_provider(primary)
    router.add_provider(fallback)
    router.routing_rules["vision"] = {
        "provider": "openai",
        "model": "text-model",
        "fallback": ["vision/vision-model"],
    }
    primary_requests = []
    fallback_requests = []

    async def fail_primary(request):
        primary_requests.append(request)
        raise ConnectionError("primary unavailable")

    async def complete_fallback(request):
        fallback_requests.append(request)
        return _response()

    primary.complete_with_timing = fail_primary
    fallback.complete_with_timing = complete_fallback
    request = _request()
    request.messages[0].images = [ImageContent(base64="aW1hZ2U=")]

    response = await router.complete(request)

    assert response.content == "answer"
    assert primary_requests == []
    assert fallback_requests[0].messages[0].images == request.messages[0].images
    assert request.messages[0].images[0].base64 == "aW1hZ2U="


@pytest.mark.asyncio
async def test_stream_does_not_fallback_after_emitting_partial_output():
    router = ModelRouter()
    primary = _provider()
    fallback = CustomLLMProvider(
        name="fallback",
        base_url="https://fallback.invalid/v1",
        api_key="test-key",
        default_model="fallback-model",
        max_retries=0,
    )
    router.add_provider(primary)
    router.add_provider(fallback)
    router.routing_rules["reasoning"] = {
        "provider": "openai",
        "fallback": ["fallback/fallback-model"],
    }
    fallback_stream = Mock()

    async def primary_stream(_request):
        yield LLMChunk(content="partial")
        raise ConnectionError("connection lost")

    primary.stream = primary_stream
    fallback.stream = fallback_stream

    generator = router.stream(_request(), task_type="reasoning")
    assert (await generator.__anext__()).content == "partial"
    with pytest.raises(ConnectionError, match="connection lost"):
        await generator.__anext__()
    fallback_stream.assert_not_called()


@pytest.mark.asyncio
async def test_closing_budgeted_stream_settles_reservation():
    router = ModelRouter()
    router.max_per_request_usd = 0.1
    router.max_per_engagement_usd = 0.1
    provider = _provider()
    router.add_provider(provider)

    async def provider_stream(_request):
        yield LLMChunk(content="partial")
        yield LLMChunk(is_final=True)

    provider.stream = provider_stream
    generator = router.stream(_request())
    assert (await generator.__anext__()).content == "partial"
    await generator.aclose()

    budget = router.get_stats()["budget"]
    assert "eng-1" not in budget["reserved_by_engagement_usd"]
    assert budget["uncertain_by_engagement_usd"]["eng-1"] > 0


@pytest.mark.asyncio
async def test_image_request_is_sent_warns_limit_not_guaranteed_and_records_cost(monkeypatch):
    from sonic.llm import router as router_module

    warning_log = Mock()
    monkeypatch.setattr(router_module.logger, "warning", warning_log)
    router = ModelRouter()
    router.max_per_request_usd = 0.01
    router.max_per_engagement_usd = 0.01
    provider = _provider()
    provider.default_model = "priced-vision-model"
    provider.model_pricing[provider.default_model] = {
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.001,
    }
    provider.complete_with_timing = AsyncMock(
        return_value=LLMResponse(
            content="image described",
            request_id="image-request",
            model="priced-model",
            provider=ProviderName.OPENAI,
            usage=TokenUsage(prompt_tokens=120, completion_tokens=20, total_tokens=140),
            cost_usd=0.0,
        )
    )
    router.add_provider(provider)
    request = _request()
    request.messages[0].images = [ImageContent(base64="aW1hZ2U=")]

    response = await router.complete(request, provider_name="openai")

    provider.complete_with_timing.assert_awaited_once()
    assert response.content == "image described"
    assert len(router.cost_records) == 1
    assert router.cost_records[0].cost_usd == pytest.approx(0.00014)
    assert router.cost_records[0].input_tokens == 120
    assert router.cost_records[0].output_tokens == 20
    budget_stats = router.get_stats()["budget"]
    assert len(budget_stats["incomplete_image_estimates"]) == 1
    assert budget_stats["incomplete_image_estimates"][0]["preflight_estimate_includes_images"] is False
    assert budget_stats["incomplete_image_estimates"][0]["hard_cap_guaranteed"] is False
    assert budget_stats["image_preflight_limit_guaranteed"] is False
    events = [call.args[0] for call in warning_log.call_args_list]
    assert "llm_image_budget_estimate_incomplete" in events
    assert "llm_budget_exceeded_after_response" not in events
