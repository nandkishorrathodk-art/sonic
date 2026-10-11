"""LangChain & CrewAI Universal Visual Perception Tools.

Provides plug-and-play Agentic Tools for LangChain, CrewAI, AutoGen, and OpenAI
Function Calling architectures. Allows LLM agents to interact with screens
sensory-first without needing heavy screenshot attachments on every step.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from avcs import ScreenEye


class AVCSScreenTools:
    """Suite of high-level agent tools for LLM frameworks."""

    def __init__(self, eye: Optional[ScreenEye] = None) -> None:
        self.eye = eye or ScreenEye()

    def look_at_element(self, element_description: str) -> str:
        """Directs visual gaze onto a described UI element.

        Args:
            element_description: Natural language description (e.g. 'Submit button', 'Search bar').
        Returns:
            JSON string with normalized (x, y) coordinates and screen category.
        """
        x, y = self.eye.look_at(element_description)
        state = self.eye.current_state or self.eye.perceive_now()
        return json.dumps({
            "target_coordinates": [round(x, 4), round(y, 4)],
            "screen_category": state.primary_category,
            "status": "target_acquired",
        })

    def check_screen_stability(self) -> str:
        """Checks if current screen has finished loading and settled.

        Returns:
            JSON string stating whether screen is stable and safe to interact with.
        """
        is_stable = self.eye.is_stable()
        state = self.eye.current_state or self.eye.perceive_now()
        return json.dumps({
            "is_stable": is_stable,
            "stability_state": state.stability_state,
            "surprise_score": round(state.surprise_score, 4),
        })

    def get_interactive_buttons(self) -> str:
        """Discovers clickable buttons and interactive controls on screen.

        Returns:
            JSON array of top clickable elements with coordinates and confidence.
        """
        buttons = self.eye.clickable_hotspots()
        payload = [
            {"x": round(x, 4), "y": round(y, 4), "confidence": round(conf, 3)}
            for x, y, conf in buttons
        ]
        return json.dumps({"clickable_elements": payload})

    def steer_and_click(self, element_description: str) -> str:
        """Physically steers the mouse cursor to the target element and clicks it.

        Uses smooth biological Hand-Eye servoing with emergency breakout fail-safe.
        """
        gaze = self.eye.look_at(element_description)
        result = self.eye.steer_hand_physical(
            target_gaze=gaze,
            click_on_arrival=True,
            force_virtual=False,
        )
        return json.dumps({
            "action": "click",
            "element": element_description,
            "contact_acquired": result.contact_acquired,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "final_position": [round(result.final_position[0], 4), round(result.final_position[1], 4)],
        })

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Returns standard OpenAI Function Calling schema tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "look_at_element",
                    "description": "Locate a UI element on the screen using natural language visual gaze.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_description": {"type": "string", "description": "Name or visual description of the target element."}
                        },
                        "required": ["element_description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_screen_stability",
                    "description": "Verify if animations or web page loads have finished settling before clicking.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_interactive_buttons",
                    "description": "List top interactive buttons and controls detected by the dorsal visual cortex.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "steer_and_click",
                    "description": "Smoothly move mouse cursor to target element using biological visual servoing and click.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_description": {"type": "string", "description": "The element to steer to and click."}
                        },
                        "required": ["element_description"],
                    },
                },
            },
        ]

