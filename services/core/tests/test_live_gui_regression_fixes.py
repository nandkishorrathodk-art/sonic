"""Regression tests for two real bugs found during a live GUI acceptance run.

1. ``ComputerUseAgent`` referenced ``np`` (numpy) in the AVCS learned-cortex
   perception path without importing it, so cortex perception silently failed on
   *every* frame (caught by a broad ``except``). Symptom in logs:
   ``avcs_cortex_perception_failed error="name 'np' is not defined"``.

2. The model router applied the global ``default_model`` (a Groq model id,
   ``qwen/qwen3.8-27b``) to whichever provider was the global default provider.
   When ``DEFAULT_LLM_PROVIDER`` resolved to ``nvidia``, NVIDIA was handed a
   Groq model id that does not exist on its endpoint, so every request 404'd and
   paid a wasted round-trip before the model-level fallback recovered.
"""
from __future__ import annotations

import ast
from pathlib import Path


from sonic.llm.router import ModelRouter

AGENT_PATH = (
    Path(__file__).resolve().parents[1] / "sonic" / "computer_use" / "agent.py"
)


def test_agent_module_has_numpy_import():
    """numpy must be imported because the AVCS cortex path uses np.frombuffer."""
    source = AGENT_PATH.read_text()
    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    assert "np" in imported_names, "agent.py uses np but never imports numpy"


def test_agent_np_usage_is_covered_by_import():
    """Any bare ``np.`` usage must be backed by an ``import numpy as np``."""
    source = AGENT_PATH.read_text()
    uses_np = "np." in source
    has_import = "import numpy as np" in source
    assert not uses_np or has_import


def _router_with_providers(providers: dict, default_provider: str, default_model: str) -> ModelRouter:
    """Build a router by loading a temp YAML config (no real credentials needed)."""
    import tempfile

    import yaml

    config = {
        "default_provider": default_provider,
        "default_model": default_model,
        "providers": providers,
        "routing_rules": {},
        "global_fallback_chain": [],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(config, fh)
        path = fh.name
    return ModelRouter.from_config(config_path=path, env_vars={})


def test_global_default_model_not_leaked_to_other_provider():
    """A global default model id must only apply to a provider that declares it."""
    router = _router_with_providers(
        providers={
            "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "models": [
                    {"id": "qwen/qwen3.8-27b"},
                    {"id": "openai/gpt-oss-120b"},
                ],
            },
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "models": [
                    {"id": "meta/llama-3.2-11b-vision-instruct"},
                    {"id": "openai/gpt-oss-20b"},
                ],
            },
        },
        default_provider="nvidia",
        default_model="qwen/qwen3.8-27b",  # a Groq model id, not an NVIDIA one
    )
    nvidia = router.providers["nvidia"]
    groq = router.providers["groq"]
    # NVIDIA must fall back to its own first declared model, never the Groq id.
    assert nvidia.default_model != "qwen/qwen3.8-27b"
    assert nvidia.default_model == "meta/llama-3.2-11b-vision-instruct"
    # The provider that actually declares the model still gets it.
    assert groq.default_model == "qwen/qwen3.8-27b"


def test_provider_declared_default_model_is_respected():
    """When the global default model *is* declared by the provider, keep it."""
    router = _router_with_providers(
        providers={
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "models": [
                    {"id": "meta/llama-3.2-11b-vision-instruct"},
                    {"id": "openai/gpt-oss-20b"},
                ],
            },
        },
        default_provider="nvidia",
        default_model="openai/gpt-oss-20b",
    )
    assert router.providers["nvidia"].default_model == "openai/gpt-oss-20b"


def test_provider_specific_env_override_still_wins_when_declared():
    """NVIDIA_MODEL=<declared id> must still override the provider default."""
    import tempfile

    import yaml

    config = {
        "default_provider": "nvidia",
        "default_model": "openai/gpt-oss-20b",
        "providers": {
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "models": [
                    {"id": "meta/llama-3.2-11b-vision-instruct"},
                    {"id": "openai/gpt-oss-20b"},
                ],
            },
        },
        "routing_rules": {},
        "global_fallback_chain": [],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(config, fh)
        path = fh.name
    router = ModelRouter.from_config(
        config_path=path,
        env_vars={"NVIDIA_MODEL": "meta/llama-3.2-11b-vision-instruct"},
    )
    assert router.providers["nvidia"].default_model == "meta/llama-3.2-11b-vision-instruct"


def test_undeclared_provider_env_override_falls_back_to_declared():
    """An NVIDIA_MODEL that NVIDIA does not declare must not be used verbatim."""
    import tempfile

    import yaml

    config = {
        "default_provider": "nvidia",
        "default_model": "",
        "providers": {
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "models": [{"id": "meta/llama-3.2-11b-vision-instruct"}],
            },
        },
        "routing_rules": {},
        "global_fallback_chain": [],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(config, fh)
        path = fh.name
    router = ModelRouter.from_config(
        config_path=path,
        env_vars={"NVIDIA_MODEL": "some/nonexistent-model"},
    )
    assert (
        router.providers["nvidia"].default_model
        == "meta/llama-3.2-11b-vision-instruct"
    )
