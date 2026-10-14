"""Unified Base LLM Driver for AVCS.

Provides standard execution loops, tool dispatch, visual verification,
and telemetry accounting for Claude, OpenAI, and Ollama agents.
"""
from __future__ import annotations

import abc
import dataclasses
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from adapters.tool_schemas import get_canonical_tools
from forge.vision.cortex.sdk import ScreenEye
from avcs.stream.reflex_controller import GameReflexController


@dataclasses.dataclass
class StepResult:
    """Outcome of a single agent step in the visual perception loop."""
    step_index: int
    tool_name: str
    tool_args: Dict[str, Any]
    output: Any
    elapsed_seconds: float
    screen_category: str
    stability_state: str
    surprise_score: float
    success: bool = True
    error: Optional[str] = None


class BaseAVCSDriver(abc.ABC):
    """Abstract driver providing unified tool execution and verification against AVCS."""

    def __init__(
        self,
        eye: ScreenEye,
        force_virtual_motor: bool = False,
    ) -> None:
        self.eye = eye
        self.force_virtual_motor = force_virtual_motor
        self.reflex = GameReflexController(eye=eye, force_virtual=force_virtual_motor)
        self.step_history: List[StepResult] = []

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Tuple[Any, bool, Optional[str]]:
        """Executes a canonical AVCS action and verifies post-condition visual state."""
        start_t = time.time()
        tool_output: Any = None
        tool_success: bool = False
        tool_error: Optional[str] = None
        try:
            if name == "locate":
                query = args.get("query", "")
                x, y = self.eye.locate(query)
                tool_output = {"target": query, "x": round(float(x), 4), "y": round(float(y), 4)}
                tool_success = True

            elif name == "click":
                query = args.get("query")
                x = args.get("x")
                y = args.get("y")
                button = args.get("button", "left")
                double_click = bool(args.get("double_click", False))

                if query:
                    ok = self.eye.click_target(query=query, button=button, force_virtual=self.force_virtual_motor)
                    tool_output = {"action": "click", "target": query, "success": ok}
                    tool_success = ok

                elif x is not None and y is not None:
                    should_arrival_click = (not double_click and button == "left")
                    res = self.eye.steer_hand_physical(
                        target_gaze=(float(x), float(y)),
                        click_on_arrival=should_arrival_click,
                        force_virtual=self.force_virtual_motor,
                        realtime=False,
                    )
                    if double_click:
                        self.reflex.driver.double_click(button=button)
                    elif button != "left":
                        self.reflex.driver.click(button=button)
                    tool_output = {"action": "click", "x": x, "y": y, "contact": res.contact_acquired}
                    tool_success = True

                else:
                    gaze = self.eye.cortex.retina.current_fixation
                    should_arrival_click = (not double_click and button == "left")
                    res = self.eye.steer_hand_physical(
                        target_gaze=gaze,
                        click_on_arrival=should_arrival_click,
                        force_virtual=self.force_virtual_motor,
                        realtime=False,
                    )
                    if double_click:
                        self.reflex.driver.double_click(button=button)
                    elif button != "left":
                        self.reflex.driver.click(button=button)
                    tool_output = {"action": "click", "fixation": gaze, "contact": res.contact_acquired}
                    tool_success = True

            elif name == "type_text":
                text = args.get("text", "")
                clear_first = bool(args.get("clear_first", False))
                press_enter = bool(args.get("press_enter", False))
                self.eye.type_text(
                    text=text,
                    clear_first=clear_first,
                    press_enter=press_enter,
                    force_virtual=self.force_virtual_motor,
                )
                tool_output = {"action": "type_text", "characters": len(text)}
                tool_success = True

            elif name == "type_into":
                query = args.get("query", "")
                text = args.get("text", "")
                clear_first = bool(args.get("clear_first", True))
                press_enter = bool(args.get("press_enter", True))
                ok = self.eye.type_into(
                    query=query,
                    text=text,
                    clear_first=clear_first,
                    press_enter=press_enter,
                    force_virtual=self.force_virtual_motor,
                )
                tool_output = {"action": "type_into", "target": query, "success": ok}
                tool_success = ok

            elif name == "wait_stable":
                timeout = float(args.get("timeout_seconds", 3.0))
                thresh = float(args.get("stability_threshold", 0.8))
                start_w = time.time()
                stable = False
                while time.time() - start_w < timeout:
                    state = self.eye.perceive_now()
                    if state.stability_state in ("stable", "stable_scene", "nominal") or state.surprise_score < (1.0 - thresh):
                        stable = True
                        break
                    time.sleep(0.05)
                tool_output = {"action": "wait_stable", "stable": stable}
                tool_success = True

            elif name == "get_affordances":
                action_type = args.get("action_type", "clickable")
                min_conf = float(args.get("min_confidence", 0.5))
                spots = self.eye.get_affordances(action_type=action_type)
                filtered = [
                    {"x": round(x, 4), "y": round(y, 4), "confidence": round(conf, 3)}
                    for x, y, conf in spots
                    if conf >= min_conf
                ]
                tool_output = {"action_type": action_type, "hotspots": filtered}
                tool_success = True

            elif name == "press_hotkey":
                keys = args.get("keys", [])
                self.eye.press_hotkey(*keys, force_virtual=self.force_virtual_motor)
                tool_output = {"action": "press_hotkey", "keys": keys}
                tool_success = True

            else:
                tool_output = None
                tool_success = False
                tool_error = f"Unknown tool: '{name}'"

        except Exception as e:
            tool_output = None
            tool_success = False
            tool_error = str(e)
        finally:
            elapsed = time.time() - start_t
            state = self.eye.current_state or self.eye.perceive_now()
            step_res = StepResult(
                step_index=len(self.step_history) + 1,
                tool_name=name,
                tool_args=args,
                output=tool_output,
                elapsed_seconds=elapsed,
                screen_category=state.primary_category,
                stability_state=state.stability_state,
                surprise_score=state.surprise_score,
                success=tool_success,
                error=tool_error,
            )
            self.step_history.append(step_res)

        return tool_output, tool_success, tool_error

    def close(self) -> None:
        self.reflex.close()
        self.eye.close()

