"""Unit tests for AVCS Canonical Tool Schemas."""
from __future__ import annotations

import json
import pytest

from adapters.tool_schemas import (
    get_canonical_tools,
    get_anthropic_tools,
    get_openai_tools,
    get_ollama_tools,
    get_mcp_tools,
)


def test_canonical_tools_structure():
    tools = get_canonical_tools()
    assert len(tools) >= 6
    names = {t["name"] for t in tools}
    assert "locate" in names
    assert "click" in names
    assert "type_text" in names
    assert "type_into" in names
    assert "wait_stable" in names
    assert "get_affordances" in names
    assert "press_hotkey" in names

    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "parameters" in t
        assert t["parameters"]["type"] == "object"
        assert "properties" in t["parameters"]


def test_anthropic_tools_format():
    anthropic_tools = get_anthropic_tools()
    for t in anthropic_tools:
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t
        # JSON serializable
        json.dumps(t)


def test_openai_and_ollama_tools_format():
    openai_tools = get_openai_tools(strict=True)
    ollama_tools = get_ollama_tools()

    assert len(openai_tools) == len(ollama_tools)
    for t in openai_tools:
        assert t["type"] == "function"
        assert "function" in t
        fn = t["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn.get("strict") is True
        assert fn["parameters"].get("additionalProperties") is False
        # In strict mode, all properties must be in required
        props = fn["parameters"]["properties"]
        assert set(fn["parameters"]["required"]) == set(props.keys())
        # In strict mode, no property may specify a default value
        for prop_name, spec in props.items():
            assert "default" not in spec
        json.dumps(t)


def test_mcp_tools_format():
    mcp_tools = get_mcp_tools()
    for t in mcp_tools:
        assert "name" in t
        assert "description" in t
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"
        json.dumps(t)

