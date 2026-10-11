"""Model Context Protocol (MCP) Server for AVCS.

Exposes AVCS's biological visual cortex and motor steering to Claude Desktop, Cursor,
and any MCP-compliant AI client over standard JSON-RPC 2.0 stdio.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, TextIO

from adapters.tool_schemas import get_mcp_tools
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen

logger = logging.getLogger("avcs.mcp")


class AVCSMCPServer:
    """Standard JSON-RPC 2.0 MCP Server for AVCS."""

    def __init__(
        self,
        eye: Optional[ScreenEye] = None,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
        force_virtual_motor: bool = False,
    ) -> None:
        self.eye = eye
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.force_virtual_motor = force_virtual_motor
        self._running = False

    def _ensure_eye(self) -> ScreenEye:
        if self.eye is None:
            # Default to ScreenEye with desktop capture (or synthetic fallback)
            try:
                self.eye = ScreenEye()
            except Exception:
                self.eye = ScreenEye(source=SyntheticScreen())
        return self.eye

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a single JSON-RPC request and returns the response dictionary."""
        is_notification = ("id" not in req)
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        # Notifications (no response permitted per JSON-RPC 2.0 sec 4.1)
        if is_notification:
            return None

        if not method:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
            }

        # Standard MCP Methods
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                    },
                    "serverInfo": {
                        "name": "avcs-mcp",
                        "version": "0.2.0",
                    },
                },
            }

        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": get_mcp_tools(),
                },
            }

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                result_content = self._execute_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": result_content,
                        "isError": False,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool execution failed: {str(e)}"}],
                        "isError": True,
                    },
                }

        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "resources": [
                        {
                            "uri": "avcs://screen/state",
                            "name": "Screen Perception State",
                            "mimeType": "application/json",
                            "description": "Current visual state: stability, scene category, surprise level.",
                        },
                        {
                            "uri": "avcs://screen/affordances",
                            "name": "Interactive Affordances",
                            "mimeType": "application/json",
                            "description": "Detected clickable and typeable UI hotspots.",
                        },
                    ],
                },
            }

        if method == "resources/read":
            uri = params.get("uri")
            eye = self._ensure_eye()
            state = eye.perceive_now()
            if uri == "avcs://screen/state":
                data = {
                    "stability_state": state.stability_state,
                    "surprise_score": float(state.surprise_score),
                    "scene_category": state.primary_category,
                    "fixation": list(state.fixation),
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "contents": [
                            {"uri": uri, "mimeType": "application/json", "text": json.dumps(data, indent=2)}
                        ]
                    },
                }
            elif uri == "avcs://screen/affordances":
                hotspots = [
                    {"x": round(x, 4), "y": round(y, 4), "action": act, "confidence": round(conf, 3)}
                    for x, y, act, conf in (state.action_hotspots or [])
                ]
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "contents": [
                            {"uri": uri, "mimeType": "application/json", "text": json.dumps(hotspots, indent=2)}
                        ]
                    },
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Resource not found: {uri}"},
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: '{method}'"},
        }

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        eye = self._ensure_eye()

        if name == "locate":
            query = args.get("query", "")
            x, y = eye.locate(query)
            res = {
                "target": query,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "normalized": True,
            }
            return [{"type": "text", "text": json.dumps(res)}]

        elif name == "click":
            button = args.get("button", "left")
            double_click = bool(args.get("double_click", False))
            query = args.get("query")
            x = args.get("x")
            y = args.get("y")

            if query:
                ok = eye.click_target(
                    query=query,
                    button=button,
                    force_virtual=self.force_virtual_motor,
                )
                msg = f"Foveated and clicked '{query}': {'success' if ok else 'failed'}"
                return [{"type": "text", "text": msg}]

            should_arrival_click = (not double_click and button == "left")
            if x is not None and y is not None:
                x_f = float(x)
                y_f = float(y)
                import math
                if not (0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0 and math.isfinite(x_f) and math.isfinite(y_f)):
                    raise ValueError(f"Normalized coordinates ({x}, {y}) must be finite numbers within [0.0, 1.0]")
                res = eye.steer_hand_physical(
                    target_gaze=(x_f, y_f),
                    click_on_arrival=should_arrival_click,
                    force_virtual=self.force_virtual_motor,
                    realtime=False,
                )
                from forge.vision.cortex.motor import get_default_driver
                driver = get_default_driver(force_virtual=self.force_virtual_motor)
                if double_click:
                    driver.double_click(button=button)
                elif button != "left":
                    driver.click(button=button)
                msg = f"Steered to ({x_f:.4f}, {y_f:.4f}) and clicked {button}. Contact: {res.contact_acquired}"
                return [{"type": "text", "text": msg}]

            # Fallback: click current gaze fixation
            gaze = eye.cortex.retina.current_fixation
            res = eye.steer_hand_physical(
                target_gaze=gaze,
                click_on_arrival=should_arrival_click,
                force_virtual=self.force_virtual_motor,
                realtime=False,
            )
            from forge.vision.cortex.motor import get_default_driver
            driver = get_default_driver(force_virtual=self.force_virtual_motor)
            if double_click:
                driver.double_click(button=button)
            elif button != "left":
                driver.click(button=button)
            return [{"type": "text", "text": f"Clicked {button} at gaze fixation ({gaze[0]:.4f}, {gaze[1]:.4f})"}]

        elif name == "type_text":
            text = args.get("text", "")
            clear_first = bool(args.get("clear_first", False))
            press_enter = bool(args.get("press_enter", False))
            eye.type_text(
                text=text,
                clear_first=clear_first,
                press_enter=press_enter,
                force_virtual=self.force_virtual_motor,
            )
            return [{"type": "text", "text": f"Typed {len(text)} characters (clear={clear_first}, enter={press_enter})"}]

        elif name == "type_into":
            query = args.get("query", "")
            text = args.get("text", "")
            clear_first = bool(args.get("clear_first", True))
            press_enter = bool(args.get("press_enter", True))
            ok = eye.type_into(
                query=query,
                text=text,
                clear_first=clear_first,
                press_enter=press_enter,
                force_virtual=self.force_virtual_motor,
            )
            msg = f"Typed into '{query}': {'success' if ok else 'failed'}"
            return [{"type": "text", "text": msg}]

        elif name == "wait_stable":
            timeout = float(args.get("timeout_seconds", 3.0))
            thresh = float(args.get("stability_threshold", 0.8))
            start_t = time.time()
            stable = False
            while time.time() - start_t < timeout:
                state = eye.perceive_now()
                if state.stability_state in ("stable", "stable_scene", "nominal") or state.surprise_score < (1.0 - thresh):
                    stable = True
                    break
                time.sleep(0.05)
            elapsed = time.time() - start_t
            return [{"type": "text", "text": f"Stability reached: {stable} (elapsed: {elapsed:.2f}s)"}]

        elif name == "get_affordances":
            action_type = args.get("action_type", "clickable")
            min_conf = float(args.get("min_confidence", 0.5))
            raw_spots = eye.get_affordances(action_type=action_type)
            filtered = [
                {"x": round(x, 4), "y": round(y, 4), "confidence": round(conf, 3)}
                for x, y, conf in raw_spots
                if conf >= min_conf
            ]
            return [{"type": "text", "text": json.dumps({"action_type": action_type, "hotspots": filtered})}]

        elif name == "press_hotkey":
            keys = args.get("keys", [])
            eye.press_hotkey(*keys, force_virtual=self.force_virtual_motor)
            return [{"type": "text", "text": f"Pressed hotkey: {'+'.join(keys)}"}]

        else:
            raise ValueError(f"Unknown tool name: '{name}'")

    def run(self) -> None:
        """Runs the JSON-RPC stdio event loop until EOF."""
        self._running = True
        while self._running:
            line = self.stdin.readline()
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            try:
                req = json.loads(line_str)
            except Exception as e:
                resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                }
                self._send_response(resp)
                continue

            resp = self.handle_request(req)
            if resp is not None:
                self._send_response(resp)

    def _send_response(self, resp: Dict[str, Any]) -> None:
        line = json.dumps(resp) + "\n"
        self.stdout.write(line)
        self.stdout.flush()

    def close(self) -> None:
        self._running = False
        if self.eye is not None:
            self.eye.close()


def run_mcp_server(eye: Optional[ScreenEye] = None, force_virtual: bool = False) -> None:
    """Entry point helper to run AVCS MCP server on stdio."""
    server = AVCSMCPServer(eye=eye, force_virtual_motor=force_virtual)
    try:
        server.run()
    finally:
        server.close()

