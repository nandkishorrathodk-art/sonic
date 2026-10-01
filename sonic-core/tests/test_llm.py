"""
Unit tests for Custom LLM Provider & Model Router.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from pathlib import Path

import pytest
from sonic.health import HealthChecker, HealthStatus
from sonic.llm.providers.custom import CustomLLMProvider
from sonic.llm.router import ModelRouter
from sonic.llm.schemas import (
    ImageContent,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ProviderName,
    ToolDefinition,
)


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


@pytest.mark.asyncio
async def test_openai_stream_sends_tools_and_stop_sequences_and_returns_tool_calls():
    async def chunks():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call-1",
                                function=SimpleNamespace(name="lookup", arguments='{"key":'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments='"value"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )

    create = AsyncMock(return_value=chunks())
    provider = CustomLLMProvider(
        name="openai",
        base_url="https://provider.invalid/v1",
        api_key="test-key",
        default_model="model",
        max_retries=0,
    )
    provider._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="look this up")],
        tools=[ToolDefinition(
            name="lookup",
            description="Look up a value",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}},
        )],
        stop_sequences=["END"],
    )

    result = [chunk async for chunk in provider.stream(request)]

    kwargs = create.await_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == "lookup"
    assert kwargs["stop"] == ["END"]
    assert result[-1].finish_reason == "tool_calls"
    assert result[-1].tool_calls[0].name == "lookup"
    assert result[-1].tool_calls[0].arguments == {"key": "value"}


@pytest.mark.asyncio
async def test_anthropic_stream_sends_tools_and_stop_sequences():
    class FakeStream:
        @property
        def text_stream(self):
            async def _text():
                yield "answer"

            return _text()

        async def get_final_message(self):
            return SimpleNamespace(
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=8, output_tokens=4),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="call-1",
                        name="lookup",
                        input={"key": "value"},
                    )
                ],
            )

    class StreamContext:
        async def __aenter__(self):
            return FakeStream()

        async def __aexit__(self, *_args):
            return None

    stream_call = Mock(return_value=StreamContext())
    provider = CustomLLMProvider(
        name="anthropic",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        default_model="claude-sonnet",
        max_retries=0,
    )
    provider._anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(stream=stream_call)
    )
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="look this up")],
        tools=[ToolDefinition(
            name="lookup",
            description="Look up a value",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}},
        )],
        stop_sequences=["END"],
    )

    result = [chunk async for chunk in provider.stream(request)]

    kwargs = stream_call.call_args.kwargs
    assert kwargs["tools"][0]["name"] == "lookup"
    assert kwargs["stop_sequences"] == ["END"]
    assert result[0].content == "answer"
    assert result[-1].tool_calls[0].arguments == {"key": "value"}


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


def test_vision_routes_configure_distinct_multimodal_fallbacks():
    from_config = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
    router = ModelRouter.from_config(from_config, env_vars={})

    assert router.routing_rules["vision"]["fallback"] == [
        "groq/qwen/qwen3.8-27b",
        "openrouter/openai/gpt-4o",
    ]
    assert router.routing_rules["computer_use"]["fallback"] == [
        "groq/qwen/qwen3.8-27b",
        "openrouter/openai/gpt-4o",
    ]
    nvidia = router.providers["nvidia"]
    assert "vision" in nvidia.model_capabilities["meta/llama-3.2-11b-vision-instruct"]
    assert "tool_use" not in nvidia.model_capabilities["meta/llama-3.2-11b-vision-instruct"]
    assert "tool_use" in nvidia.model_capabilities["nvidia/nemotron-3-super-120b-a12b"]


@pytest.mark.asyncio
async def test_multimodal_model_without_tool_use_uses_text_action_fallback():
    model = "meta/llama-3.2-11b-vision-instruct"
    provider = CustomLLMProvider(
        name="nvidia",
        base_url="https://example.invalid/v1",
        api_key="",
        default_model=model,
        capabilities=["vision", "computer_use", "multimodal"],
        model_capabilities={
            model: ["vision", "computer_use", "multimodal"],
        },
        max_retries=0,
    )
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="ACTION: GUI_SCREENSHOT",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )],
        usage=None,
    ))
    provider._openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    request = LLMRequest(
        model=model,
        messages=[
            Message(role=MessageRole.SYSTEM, content="Choose one GUI action."),
            Message(
                role=MessageRole.USER,
                content="Describe this frame and choose an action.",
                images=[ImageContent(base64="synthetic-image", media_type="image/png")],
            ),
        ],
        tools=[ToolDefinition(
            name="computer_action",
            description="Choose one action.",
            parameters={"type": "object", "properties": {"action": {"type": "string"}}},
        )],
    )
    text_request = request.model_copy(deep=True)
    text_request.messages[-1].images = []
    assert provider._tools_for_request(text_request, model) is not None
    provider.model_capabilities[model] = frozenset({"vision", "multimodal", "tool_use"})
    assert provider._tools_for_request(request, model) is not None
    provider.model_capabilities[model] = frozenset({"vision", "computer_use", "multimodal"})

    response = await provider._complete_openai(request)

    kwargs = create.await_args.kwargs
    assert "tools" not in kwargs
    assert kwargs["messages"][0]["content"][0]["type"] == "image_url"
    assert response.content == "ACTION: GUI_SCREENSHOT"


@pytest.mark.asyncio
async def test_vision_fallback_preserves_image_and_reports_provider_status():
    router = ModelRouter()
    primary = CustomLLMProvider(
        name="nvidia",
        base_url="https://example.invalid/v1",
        api_key="vision-key",
        default_model="vision-model",
    )
    fallback = CustomLLMProvider(
        name="groq",
        base_url="https://example.invalid/v1",
        api_key="fallback-key",
        default_model="qwen/qwen3.8-27b",
    )
    primary_error = RuntimeError("private provider response must not be exposed")
    primary_error.status_code = 503
    primary.complete_with_timing = AsyncMock(side_effect=primary_error)
    fallback.complete_with_timing = AsyncMock(return_value=LLMResponse(
        content="READY",
        model="qwen/qwen3.8-27b",
        provider=ProviderName.GROQ,
    ))
    router.add_provider(primary)
    router.add_provider(fallback)
    router.routing_rules["vision"] = {
        "provider": "nvidia",
        "model": "vision-model",
        "fallback": ["groq/qwen/qwen3.8-27b"],
    }
    request = LLMRequest(
        messages=[Message(
            role=MessageRole.USER,
            content="Describe the image.",
            images=[ImageContent(base64="synthetic-image")],
        )],
        task_type="computer_use",
    )

    response = await router.complete(request)

    assert response.content == "READY"
    fallback_request = fallback.complete_with_timing.await_args.args[0]
    assert fallback_request.model == "qwen/qwen3.8-27b"
    assert fallback_request.messages[0].images[0].base64 == "synthetic-image"


@pytest.mark.asyncio
async def test_vision_failure_summary_exposes_status_not_response_body():
    router = ModelRouter()
    primary_error = RuntimeError("sensitive provider response")
    primary_error.status_code = 401

    with pytest.raises(RuntimeError) as error:
        await router._try_fallbacks(
            LLMRequest(messages=[Message(
                role=MessageRole.USER,
                content="Describe the image.",
            )], model="vision-model"),
            failed_provider="nvidia",
            task_type="vision",
            primary_error=primary_error,
        )

    assert "HTTP 401" in str(error.value)
    assert "sensitive provider response" not in str(error.value)


@pytest.mark.asyncio
async def test_llm_health_degrades_when_only_non_vision_provider_is_ready(monkeypatch):
    from types import SimpleNamespace

    router = SimpleNamespace(
        providers={"nvidia": object(), "deepseek": object()},
        routing_rules={
            "vision": {
                "provider": "nvidia",
                "fallback": ["groq/qwen/qwen3.8-27b"],
            },
        },
        _is_provider_ready=lambda name: name == "deepseek",
    )
    monkeypatch.setattr(
        ModelRouter,
        "from_config",
        classmethod(lambda cls: router),
    )

    health = await HealthChecker.check_llm()

    assert health.status == HealthStatus.DEGRADED
    assert "No configured vision provider credentials" in health.message
    assert "deepseek" in health.message


@pytest.mark.asyncio
async def test_llm_health_reports_ready_vision_route(monkeypatch):
    from types import SimpleNamespace

    router = SimpleNamespace(
        providers={"nvidia": object(), "groq": object()},
        routing_rules={
            "vision": {
                "provider": "nvidia",
                "fallback": ["groq/qwen/qwen3.8-27b"],
            },
        },
        _is_provider_ready=lambda name: name in {"nvidia", "groq"},
    )
    monkeypatch.setattr(
        ModelRouter,
        "from_config",
        classmethod(lambda cls: router),
    )

    health = await HealthChecker.check_llm()

    assert health.status == HealthStatus.HEALTHY
    assert health.message == (
        "Configured vision providers: groq, nvidia; remote reachability is not probed."
    )
