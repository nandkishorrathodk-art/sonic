"""Unit tests for AVCS Model Context Protocol (MCP) Server."""
from __future__ import annotations

import io
import json
import pytest

from avcs.mcp.server import AVCSMCPServer
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


@pytest.fixture
def test_server():
    eye = ScreenEye(source=SyntheticScreen(width=320, height=240), device="cpu")
    server = AVCSMCPServer(eye=eye, force_virtual_motor=True)
    yield server
    server.close()


def test_mcp_initialize(test_server):
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = test_server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    res = resp["result"]
    assert res["protocolVersion"] == "2024-11-05"
    assert res["serverInfo"]["name"] == "avcs-mcp"
    assert "tools" in res["capabilities"]


def test_mcp_ping(test_server):
    req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    resp = test_server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 2
    assert resp["result"] == {}


def test_mcp_tools_list(test_server):
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = test_server.handle_request(req)
    assert resp["id"] == 3
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "locate" in tool_names
    assert "click" in tool_names
    assert "type_text" in tool_names
    assert "type_into" in tool_names
    assert "wait_stable" in tool_names
    assert "get_affordances" in tool_names


def test_mcp_tools_call_locate(test_server):
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "locate",
            "arguments": {"query": "Submit button"},
        },
    }
    resp = test_server.handle_request(req)
    assert resp["id"] == 4
    assert "result" in resp
    content = resp["result"]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    parsed = json.loads(content[0]["text"])
    assert parsed["target"] == "Submit button"
    assert 0.0 <= parsed["x"] <= 1.0
    assert 0.0 <= parsed["y"] <= 1.0


def test_mcp_tools_call_click(test_server):
    # Click with query
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "click",
            "arguments": {"query": "Search bar", "button": "left"},
        },
    }
    resp = test_server.handle_request(req)
    assert resp["id"] == 5
    assert "result" in resp
    assert "Clicked" in resp["result"]["content"][0]["text"] or "Foveated" in resp["result"]["content"][0]["text"]

    # Click with normalized coords
    req2 = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "click",
            "arguments": {"x": 0.5, "y": 0.5, "button": "left", "double_click": True},
        },
    }
    resp2 = test_server.handle_request(req2)
    assert resp2["id"] == 6
    assert "Steered to" in resp2["result"]["content"][0]["text"]


def test_mcp_tools_call_typing(test_server):
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "type_text",
            "arguments": {"text": "hello world", "clear_first": True, "press_enter": True},
        },
    }
    resp = test_server.handle_request(req)
    assert resp["id"] == 7
    assert "Typed 11 characters" in resp["result"]["content"][0]["text"]


def test_mcp_tools_call_type_into(test_server):
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "type_into",
            "arguments": {"query": "Username field", "text": "alice"},
        },
    }
    resp = test_server.handle_request(req)
    assert resp["id"] == 8
    assert "Typed into 'Username field'" in resp["result"]["content"][0]["text"]


def test_mcp_tools_call_affordances_and_stability(test_server):
    req1 = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {"name": "get_affordances", "arguments": {"action_type": "clickable"}},
    }
    resp1 = test_server.handle_request(req1)
    assert resp1["id"] == 9
    parsed_aff = json.loads(resp1["result"]["content"][0]["text"])
    assert parsed_aff["action_type"] == "clickable"

    req2 = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "wait_stable", "arguments": {"timeout_seconds": 0.1}},
    }
    resp2 = test_server.handle_request(req2)
    assert resp2["id"] == 10
    assert "Stability reached" in resp2["result"]["content"][0]["text"]


def test_mcp_resources(test_server):
    req1 = {"jsonrpc": "2.0", "id": 11, "method": "resources/list"}
    resp1 = test_server.handle_request(req1)
    uris = [r["uri"] for r in resp1["result"]["resources"]]
    assert "avcs://screen/state" in uris
    assert "avcs://screen/affordances" in uris

    req2 = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "resources/read",
        "params": {"uri": "avcs://screen/state"},
    }
    resp2 = test_server.handle_request(req2)
    data = json.loads(resp2["result"]["contents"][0]["text"])
    assert "stability_state" in data
    assert "surprise_score" in data


def test_mcp_stdio_stream_loop():
    input_text = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
        "",
    ])
    in_stream = io.StringIO(input_text)
    out_stream = io.StringIO()

    eye = ScreenEye(source=SyntheticScreen(width=320, height=240), device="cpu")
    server = AVCSMCPServer(eye=eye, stdin=in_stream, stdout=out_stream, force_virtual_motor=True)
    server.run()
    server.close()

    lines = [l for l in out_stream.getvalue().split("\n") if l.strip()]
    assert len(lines) == 2
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["id"] == 1
    assert r2["id"] == 2


def test_mcp_error_handling_and_notifications(test_server):
    # 1. Notification suppression (no "id" member)
    notif = {"jsonrpc": "2.0", "method": "notifications/custom_event", "params": {}}
    resp = test_server.handle_request(notif)
    assert resp is None  # MUST NOT reply to notifications per JSON-RPC 2.0

    # 2. Invalid coordinates in click tool returns isError: true inside CallToolResult
    bad_req = {
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/call",
        "params": {
            "name": "click",
            "arguments": {"x": 2.5, "y": -0.5},
        },
    }
    resp = test_server.handle_request(bad_req)
    assert resp["id"] == 99
    assert resp["result"]["isError"] is True
    assert "must be finite numbers within [0.0, 1.0]" in resp["result"]["content"][0]["text"]

    # 3. wait_stable returns True on synthetic screen
    wait_req = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {
            "name": "wait_stable",
            "arguments": {"timeout_seconds": 1.0},
        },
    }
    resp_wait = test_server.handle_request(wait_req)
    assert resp_wait["id"] == 100
    assert resp_wait["result"]["isError"] is False
    assert "Stability reached: True" in resp_wait["result"]["content"][0]["text"]

