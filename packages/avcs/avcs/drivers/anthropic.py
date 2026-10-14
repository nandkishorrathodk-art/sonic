"""Anthropic Claude Driver for AVCS.

Connects Claude 3.5 / 3.7 Sonnet or Haiku to AVCS with tool_use and visual verification.
Supports mock client mode for hermetic local testing without API keys.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from adapters.tool_schemas import get_anthropic_tools
from forge.vision.cortex.sdk import ScreenEye
from avcs.drivers.base import BaseAVCSDriver, StepResult


class AnthropicAVCSDriver(BaseAVCSDriver):
    """Driver executing Claude tool_use blocks against the AVCS visual cortex."""

    def __init__(
        self,
        eye: ScreenEye,
        api_key: Optional[str] = None,
        model: str = "claude-3-7-sonnet-20250219",
        force_virtual_motor: bool = False,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(eye=eye, force_virtual_motor=force_virtual_motor)
        self.api_key = api_key
        self.model = model
        self.client = client
        self.tools = get_anthropic_tools()

    def step(self, tool_call_block: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an Anthropic tool_use content block to AVCS and returns tool_result block."""
        call_id = tool_call_block.get("id", "call_default")
        name = tool_call_block.get("name", "")
        args = tool_call_block.get("input", {})

        output, success, error = self.execute_tool(name, args)
        if success:
            content_text = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
            return {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content_text,
            }
        else:
            return {
                "type": "tool_result",
                "tool_use_id": call_id,
                "is_error": True,
                "content": error or "Unknown execution error",
            }

