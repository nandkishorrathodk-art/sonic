"""Canonical Tool Schemas for AVCS AI Agent Integrations.

Provides standardized, type-safe schema generators for Anthropic Claude Tools,
OpenAI Function Calling, Ollama Local Models, and the Model Context Protocol (MCP).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

# Master definition of canonical AVCS tools
CANONICAL_TOOLS_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "locate",
        "description": (
            "Locate a visual or semantic UI element on the screen using AVCS gaze foveation. "
            "Returns normalized (x, y) coordinates within the [0.0, 1.0] screen viewport range."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description or text label of the target UI element "
                        "(e.g., 'search input', 'blue submit button', 'profile avatar icon', 'close button')."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "click",
        "description": (
            "Steer the mouse cursor to a target coordinate or query using closed-loop "
            "biological servoing and perform a mouse click."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "Normalized horizontal coordinate [0.0 - 1.0]. Optional if query is specified.",
                },
                "y": {
                    "type": "number",
                    "description": "Normalized vertical coordinate [0.0 - 1.0]. Optional if query is specified.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional element description to foveate and click in a single step.",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "default": "left",
                    "description": "Mouse button to click.",
                },
                "double_click": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, performs a double-click.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type a text string into the active focused control using the native hardware keyboard driver."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The exact text string to type.",
                },
                "clear_first": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, sends Ctrl+A and Backspace to clear existing content before typing.",
                },
                "press_enter": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, presses the Enter/Return key after typing.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "type_into",
        "description": (
            "Atomic action: locates a target input field by visual query, clicks to focus, and types text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Visual or semantic description of the input field.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the field.",
                },
                "clear_first": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to clear existing field content before typing.",
                },
                "press_enter": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to press Enter after typing.",
                },
            },
            "required": ["query", "text"],
        },
    },
    {
        "name": "wait_stable",
        "description": (
            "Wait until screen transitions, page loads, or UI animations settle into visual stability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeout_seconds": {
                    "type": "number",
                    "default": 3.0,
                    "description": "Maximum seconds to wait for visual stability settling.",
                },
                "stability_threshold": {
                    "type": "number",
                    "default": 0.8,
                    "description": "Threshold score (0.0 to 1.0) above which the screen is considered stable.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_affordances",
        "description": (
            "Retrieve real-time actionable UI hotspots and affordance basins across the screen "
            "(e.g., clickable buttons, typeable fields) without sending screenshot pixels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["clickable", "typeable", "scrollable"],
                    "default": "clickable",
                    "description": "Category of affordances to retrieve.",
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum confidence threshold (0.0 to 1.0).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "press_hotkey",
        "description": (
            "Press a sequence of keys simultaneously as a keyboard shortcut (e.g., ['ctrl', 'c'], ['enter'])."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key names to press together.",
                },
            },
            "required": ["keys"],
        },
    },
]


def get_canonical_tools() -> List[Dict[str, Any]]:
    """Returns deep-copied master registry of canonical AVCS tools."""
    return copy.deepcopy(CANONICAL_TOOLS_REGISTRY)


def get_anthropic_tools() -> List[Dict[str, Any]]:
    """Returns tool definitions formatted for Anthropic Claude Messages API (tool_use)."""
    tools = []
    for tool in CANONICAL_TOOLS_REGISTRY:
        tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": copy.deepcopy(tool["parameters"]),
        })
    return tools


def get_openai_tools(strict: bool = False) -> List[Dict[str, Any]]:
    """Returns tool definitions formatted for OpenAI Function Calling (ChatCompletions API)."""
    tools = []
    for tool in CANONICAL_TOOLS_REGISTRY:
        params = copy.deepcopy(tool["parameters"])
        fn_def: Dict[str, Any] = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": params,
        }
        if strict:
            fn_def["strict"] = True
            params["additionalProperties"] = False
            props = params.get("properties", {})
            orig_req = set(params.get("required", []))
            params["required"] = list(props.keys())
            for key, spec in props.items():
                if "default" in spec:
                    del spec["default"]
                if key not in orig_req:
                    curr_type = spec.get("type")
                    if isinstance(curr_type, str):
                        spec["type"] = [curr_type, "null"]
                    if "enum" in spec and None not in spec["enum"]:
                        spec["enum"] = list(spec["enum"]) + [None]
        tools.append({
            "type": "function",
            "function": fn_def,
        })
    return tools


def get_ollama_tools() -> List[Dict[str, Any]]:
    """Returns function calling tool definitions compatible with Ollama API /api/chat."""
    return get_openai_tools(strict=False)


def get_mcp_tools() -> List[Dict[str, Any]]:
    """Returns tool definitions formatted for the Model Context Protocol (MCP) tools/list method."""
    tools = []
    for tool in CANONICAL_TOOLS_REGISTRY:
        tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": copy.deepcopy(tool["parameters"]),
        })
    return tools

