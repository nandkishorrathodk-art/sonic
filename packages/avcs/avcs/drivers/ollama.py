"""Ollama Local Model Driver for AVCS.

Connects local open-source LLMs (Llama 3.2-Vision, Qwen2.5, DeepSeek) to AVCS
via local Ollama REST daemon with zero external pip dependencies.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from adapters.tool_schemas import get_ollama_tools
from forge.vision.cortex.sdk import ScreenEye
from avcs.drivers.base import BaseAVCSDriver, StepResult


class OllamaAVCSDriver(BaseAVCSDriver):
    """Driver connecting local Ollama LLMs to AVCS."""

    def __init__(
        self,
        eye: ScreenEye,
        host: str = "http://localhost:11434",
        model: str = "llama3.2-vision",
        force_virtual_motor: bool = False,
    ) -> None:
        super().__init__(eye=eye, force_virtual_motor=force_virtual_motor)
        self.host = host.rstrip("/")
        self.model = model
        self.tools = get_ollama_tools()

    def step(self, tool_call_obj: Dict[str, Any]) -> Dict[str, Any]:
        fn = tool_call_obj.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        output, success, error = self.execute_tool(name, args)
        content_text = json.dumps(output) if success else json.dumps({"error": error})

        return {
            "role": "tool",
            "content": content_text,
        }

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sends chat request to local Ollama daemon with AVCS tool schemas."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

