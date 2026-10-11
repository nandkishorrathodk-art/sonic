"""OpenAI Driver for AVCS.

Connects GPT-4o, GPT-4o-mini, and Operator to AVCS via ChatCompletions function calling.
Supports mock client mode for hermetic testing.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from adapters.tool_schemas import get_openai_tools
from forge.vision.cortex.sdk import ScreenEye
from avcs.drivers.base import BaseAVCSDriver, StepResult


class OpenAIAVCSDriver(BaseAVCSDriver):
    """Driver executing OpenAI tool_calls against AVCS."""

    def __init__(
        self,
        eye: ScreenEye,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        force_virtual_motor: bool = False,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(eye=eye, force_virtual_motor=force_virtual_motor)
        self.api_key = api_key
        self.model = model
        self.client = client
        self.tools = get_openai_tools(strict=False)

    def step(self, tool_call_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an OpenAI tool_call and returns role='tool' message dictionary."""
        call_id = tool_call_obj.get("id", "call_default")
        fn = tool_call_obj.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}

        output, success, error = self.execute_tool(name, args)
        content_text = json.dumps(output) if success else json.dumps({"error": error})

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content_text,
        }

