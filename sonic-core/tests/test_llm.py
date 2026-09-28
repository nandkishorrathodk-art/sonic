"""
Unit tests for Custom LLM Provider & Model Router.
"""

from unittest.mock import AsyncMock

import pytest
from sonic.llm.providers.custom import CustomLLMProvider
from sonic.llm.router import ModelRouter
from sonic.llm.schemas import LLMRequest, Message, MessageRole


def test_custom_llm_provider_initialization():
    provider = CustomLLMProvider(
        name="grok",
        base_url="https://api.x.ai/v1",
        api_key="mock-key-123",
        default_model="grok-3",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    )

    assert provider.name == "grok"
    assert provider.base_url == "https://api.x.ai/v1"
    assert provider.default_model == "grok-3"
    assert provider.is_anthropic is False

    models = provider.list_models()
    assert len(models) == 1
    assert models[0].id == "grok-3"


def test_anthropic_detection():
    claude_provider = CustomLLMProvider(
        name="claude",
        base_url="https://api.anthropic.com",
        api_key="sk-ant-mock",
        default_model="claude-sonnet-4-20250514",
    )
    assert claude_provider.is_anthropic is True


def test_model_router_registration_and_fallback():
    router = ModelRouter()
    
    p1 = CustomLLMProvider(name="primary", base_url="http://localhost:8001/v1", api_key="k1", default_model="m1")
    p2 = CustomLLMProvider(name="fallback", base_url="http://localhost:8002/v1", api_key="k2", default_model="m2")
    
    router.add_provider(p1)
    router.add_provider(p2)
    
    router.routing_rules["reasoning"] = {
        "provider": "primary",
        "model": "m1",
        "fallback": ["fallback/m2"],
    }
    
    provider, model = router._resolve_provider(task_type="reasoning")
    assert provider.name == "primary"
    assert model == "m1"

    # Explicit override
    provider_exp, _ = router._resolve_provider(provider_name="fallback")
    assert provider_exp.name == "fallback"


def test_vision_route_fails_closed_instead_of_using_local_text_model(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:11434")
    router = ModelRouter()
    router.default_provider = "local"
    router.fallback_chain = ["local"]
    router.add_provider(
        CustomLLMProvider(
            name="nvidia",
            base_url="https://example.invalid/v1",
            api_key="",
            default_model="vision-model",
        )
    )
    router.add_provider(
        CustomLLMProvider(
            name="local",
            base_url="http://localhost:11434/v1",
            api_key="",
            default_model="text-only-model",
        )
    )
    router.routing_rules["vision"] = {
        "provider": "nvidia",
        "model": "vision-model",
        "fallback": [],
    }

    with pytest.raises(RuntimeError, match="No configured vision provider is ready"):
        router._resolve_provider(task_type="vision")


@pytest.mark.asyncio
async def test_vision_request_does_not_fall_back_to_global_text_provider(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:11434")
    router = ModelRouter()
    primary = CustomLLMProvider(
        name="nvidia",
        base_url="https://example.invalid/v1",
        api_key="vision-key",
        default_model="vision-model",
    )
    text_fallback = CustomLLMProvider(
        name="local",
        base_url="http://localhost:11434/v1",
        api_key="",
        default_model="text-only-model",
    )
    text_fallback.complete_with_timing = AsyncMock()
    router.add_provider(primary)
    router.add_provider(text_fallback)
    router.routing_rules["vision"] = {
        "provider": "nvidia",
        "model": "vision-model",
        "fallback": [],
    }
    router.fallback_chain = ["local"]

    with pytest.raises(RuntimeError, match="All configured vision providers failed"):
        await router._try_fallbacks(
            LLMRequest(messages=[Message(role=MessageRole.USER, content="describe image")]),
            failed_provider="nvidia",
            task_type="vision",
        )

    text_fallback.complete_with_timing.assert_not_awaited()
