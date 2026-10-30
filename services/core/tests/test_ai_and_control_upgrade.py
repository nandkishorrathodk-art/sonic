"""
Tests for the AI + control upgrades:

AI:
  1. LLM provider transient-error retry/backoff (429/5xx) and no-retry for
     auth/model-not-found.
  2. Robust tool-call argument parsing (malformed/noisy JSON).
  3. ReAct engine native function-calling path with PARALLEL tool execution.

Control:
  4. ComputerUseAgent screen-dimension tracking + coordinate bounds validation
     (off-screen / negative coordinates are BLOCKED and never reach the
     provider).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from sonic.agents.react_engine import ReActEngine, ToolCategory, ToolDefinition, create_default_tool_registry
from sonic.computer.models import (
    ComputerState, FileEntry, GitStatusInfo, ScreenObservation,
)
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType, ComputerDecisionTrace, ComputerWorldObservation
from sonic.llm.providers.custom import (
    CustomLLMProvider,
    _extract_json_object,
    _is_transient_error,
    parse_tool_arguments,
)
from sonic.llm.schemas import LLMRequest, LLMResponse, Message, MessageRole, ToolCall
from sonic.sandbox.provider import ExecResult


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _png_frame(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), "black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 1. Robust JSON parsing
# ---------------------------------------------------------------------------

def test_parse_tool_arguments_valid_json():
    assert parse_tool_arguments('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_tool_arguments_extracts_from_prose():
    raw = 'I will scan with: {"target": "example.com", "port": 443} now.'
    assert parse_tool_arguments(raw) == {"target": "example.com", "port": 443}


def test_parse_tool_arguments_trailing_comma_fixed():
    assert parse_tool_arguments('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_parse_tool_arguments_single_quotes_fixed():
    assert parse_tool_arguments("{'cmd': 'ls'}") == {"cmd": "ls"}


def test_parse_tool_arguments_none_and_empty():
    assert parse_tool_arguments(None) == {}
    assert parse_tool_arguments("") == {}


def test_parse_tool_arguments_bare_value_wrapped():
    # Non-JSON bare string is wrapped so dict-expecting callers still get text.
    assert parse_tool_arguments("nmap -sV target.com") == {"value": "nmap -sV target.com"}


def test_extract_json_object_returns_none_when_no_object():
    assert _extract_json_object("just words, no braces") is None
    assert _extract_json_object("") is None


# ---------------------------------------------------------------------------
# 2. Transient-error classification
# ---------------------------------------------------------------------------

def test_is_transient_rate_limit():
    assert _is_transient_error(RuntimeError("429 Too Many Requests")) is True
    assert _is_transient_error(RuntimeError("Rate limit exceeded")) is True


def test_is_transient_server_error():
    assert _is_transient_error(RuntimeError("503 Service Unavailable")) is True
    assert _is_transient_error(RuntimeError("502 Bad Gateway")) is True


def test_is_transient_transport_error():
    assert _is_transient_error(ConnectionError("Connection timeout")) is True


def test_non_transient_errors_not_flagged():
    # Auth / bad request / model-not-found must NOT be retried.
    assert _is_transient_error(RuntimeError("401 Unauthorized")) is False
    oversized = RuntimeError("Rate limit exceeded: input token limit")
    oversized.status_code = 413
    assert _is_transient_error(oversized) is False
    assert _is_transient_error(RuntimeError("model not found: foo-3")) is False
    assert _is_transient_error(ValueError("invalid api key")) is False


# ---------------------------------------------------------------------------
# 3. LLM provider retry/backoff
# ---------------------------------------------------------------------------

class _ScriptedClient:
    """A fake OpenAI-style SDK client that replays a scripted call sequence.

    ``responses`` is a list of either:
      - an LLMResponse-like object (returned on success), or
      - an Exception instance (raised).
    Each ``chat.completions.create`` call consumes the next item.
    """

    class _Choice:
        def __init__(self, content, tool_calls=None, finish="stop"):
            self.message = type("M", (), {
                "content": content,
                "tool_calls": tool_calls or None,
            })()
            self.finish_reason = finish

    class _Resp:
        def __init__(self, choice, usage=None):
            self.choices = [choice]
            self.usage = type("U", (), {
                "prompt_tokens": (usage or (0, 0, 0))[0],
                "completion_tokens": (usage or (0, 0, 0))[1],
                "total_tokens": (usage or (0, 0, 0))[2],
            })()

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        # Mirror the OpenAI SDK shape: provider._openai_client.chat.completions.create
        self.chat = type("C", (), {"completions": self})()

    async def create(self, **kwargs):
        item = self.responses[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _make_provider(monkeypatch_sleep, max_retries=3):
    """Build a CustomLLMProvider whose SDK client + sleep are injected for tests."""
    provider = CustomLLMProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        default_model="gpt-4o",
        max_retries=max_retries,
        retry_base_delay=0.0,  # no real waiting in tests
        retry_max_delay=0.0,
    )
    return provider


def test_provider_retries_on_rate_limit_then_succeeds(monkeypatch):
    # First call 429s, second succeeds.
    good = _ScriptedClient._Resp(_ScriptedClient._Choice("ok"))
    client = _ScriptedClient([RuntimeError("429 Too Many Requests"), good])
    provider = _make_provider(monkeypatch)
    provider._openai_client = client

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])
    resp = _run(provider.complete(req))
    assert resp.content == "ok"
    assert client.calls == 2  # one retry


def test_provider_does_not_retry_auth_error(monkeypatch):
    bad = RuntimeError("401 Unauthorized")
    client = _ScriptedClient([bad])
    provider = _make_provider(monkeypatch)
    provider._openai_client = client

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])
    with pytest.raises(RuntimeError, match="401"):
        _run(provider.complete(req))
    assert client.calls == 1  # no retry


def test_provider_retries_exhausted(monkeypatch):
    err = RuntimeError("503 Service Unavailable")
    client = _ScriptedClient([err, err, err, err])
    provider = _make_provider(monkeypatch, max_retries=3)
    provider._openai_client = client

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])
    with pytest.raises(RuntimeError, match="503"):
        _run(provider.complete(req))
    # initial attempt (1) + 3 retries = 4 total calls.
    assert client.calls == 4


# ---------------------------------------------------------------------------
# 4. ReAct native function-calling + parallel tool execution
# ---------------------------------------------------------------------------

class _ToolLLM:
    """Returns a sequence of LLMResponses with tool_calls / final answer."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []
        self._i = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        resp = self.responses[self._i]
        self._i += 1
        return resp


def _tc(name: str, arguments: dict, idx: int = 1) -> ToolCall:
    return ToolCall(id=f"call-{idx}-{name}", name=name, arguments=arguments)


def test_react_execute_with_tools_runs_parallel_tool_calls():
    """Multiple tool calls in one step are executed concurrently."""
    registry = create_default_tool_registry()
    # Replace handlers with counters + an ordering barrier to prove parallelism.
    barrier = asyncio.Event()

    seen = {"order": []}

    async def slow_tool(args):
        seen["order"].append(("tool_a_start", args))
        # Force a context switch so parallel execution overlaps.
        await asyncio.sleep(0)
        seen["order"].append(("tool_a_end", args))
        return "tool_a output"

    async def fast_tool(args):
        seen["order"].append(("tool_b_start", args))
        seen["order"].append(("tool_b_end", args))
        return "tool_b output"

    registry.register(ToolDefinition(
        name="tool_a", description="Test tool A", category=ToolCategory.UTILITY,
        parameters={"args": "arguments"}, handler=slow_tool,
    ))
    registry.register(ToolDefinition(
        name="tool_b", description="Test tool B", category=ToolCategory.UTILITY,
        parameters={"args": "arguments"}, handler=fast_tool,
    ))

    engine = ReActEngine(registry, max_iterations=5)

    step1 = LLMResponse(
        content="Running two tools in parallel.",
        tool_calls=[_tc("tool_a", {"args": "198.51.100.1"}), _tc("tool_b", {"args": "-sV"})],
    )
    step2 = LLMResponse(content="done", tool_calls=[])
    llm = _ToolLLM([step1, step2])

    result = _run(engine.execute_with_tools("recon target", llm.complete))
    assert result["success"] is True
    assert result["steps"] == 2
    # Both tool calls recorded as observations in step 1.
    step1_obs = [o for o in result["observations"] if o["step"] == 1 and o["action_type"] == "tool_call"]
    assert len(step1_obs) == 2
    names = {o["action_input"].split("[")[0] for o in step1_obs}
    assert names == {"tool_a", "tool_b"}
    # Parallel proof: the second tool started before the first ended.
    assert seen["order"].index(("tool_b_start", "-sV")) < seen["order"].index(("tool_a_end", "198.51.100.1"))


def test_react_execute_with_tools_tool_schemas_sent():
    """The model request carries real tool definitions, not text parsing hints."""
    registry = create_default_tool_registry()
    async def noop(args): return "ok"
    for name in ("probe", "scanner", "fuzzer"):
        registry.register(ToolDefinition(
            name=name, description=f"Test {name}", category=ToolCategory.UTILITY,
            parameters={"args": "arguments"}, handler=noop,
        ))
    engine = ReActEngine(registry, max_iterations=2)
    llm = _ToolLLM([LLMResponse(content="answer", tool_calls=[])])
    _run(engine.execute_with_tools("task", llm.complete))
    assert llm.requests, "think_fn should have been called"
    tools = llm.requests[0].tools
    assert tools and len(tools) == 3
    tool_names = {t.name for t in tools}
    assert {"probe", "scanner", "fuzzer"} == tool_names


def test_react_execute_with_tools_dict_args_coerced():
    """Dict arguments from function-calling reach handlers as their positional string."""
    registry = create_default_tool_registry()
    received = []

    async def capture(args):
        received.append(args)
        return "ok"

    registry.register(ToolDefinition(
        name="probe", description="Test probe", category=ToolCategory.UTILITY,
        parameters={"args": "arguments"}, handler=capture,
    ))
    engine = ReActEngine(registry, max_iterations=2)
    step1 = LLMResponse(
        content="",
        tool_calls=[_tc("probe", {"args": "198.51.100.1"})],
    )

    step2 = LLMResponse(content="final", tool_calls=[])
    llm = _ToolLLM([step1, step2])
    result = _run(engine.execute_with_tools("task", llm.complete))
    assert result["success"] is True
    assert received == ["198.51.100.1"]


def test_react_execute_with_tools_max_iterations():
    registry = create_default_tool_registry()
    engine = ReActEngine(registry, max_iterations=2)
    # Always requests one tool, never answers.
    llm = _ToolLLM([
        LLMResponse(content="", tool_calls=[_tc("curl", {"args": "x"})]),
        LLMResponse(content="", tool_calls=[_tc("curl", {"args": "y"})]),
    ])
    result = _run(engine.execute_with_tools("task", llm.complete))
    assert result["success"] is False
    assert result["steps"] == 2


# ---------------------------------------------------------------------------
# 5. ComputerUseAgent coordinate bounds validation
# ---------------------------------------------------------------------------

class _StubComputer(ComputerProvider):
    """Minimal computer that records GUI actions and reports a fixed screen size."""

    def __init__(self, width=1920, height=1080):
        self.workspace_id = "ws"
        self.width = width
        self.height = height
        self.gui_calls: list[tuple[str, int, int]] = []
        self.terminal_calls: list[str] = []

    async def create(self, *a, **k): return type("W", (), {"id": self.workspace_id})()
    async def destroy(self, *a, **k): return True
    async def status(self, workspace_id):
        return ComputerState(
            workspace_id=workspace_id,
            tenant_id="t",
            active_application="x",
        )
    async def screenshot(self, workspace_id):
        return ScreenObservation(
            screenshot_base64=_png_frame(self.width, self.height),
            width=self.width,
            height=self.height,
            visible_text="ui",
            active_window="x",
            capture_age_ms=0,
            controls_age_ms=0,
            desktop_state="INTERACTIVE",
        )
    async def gui_action(self, workspace_id, action, actor="operator"):
        self.gui_calls.append((action.action, action.x or 0, action.y or 0))
        return ScreenObservation(
            screenshot_base64=_png_frame(self.width, self.height),
            width=self.width,
            height=self.height,
            active_window="x",
            capture_age_ms=0,
            controls_age_ms=0,
            desktop_state="INTERACTIVE",
        )
    async def terminal(self, workspace_id, command, timeout=60, actor="operator"):
        self.terminal_calls.append(command)
        return ExecResult(command=command, exit_code=0, stdout="", stderr="")
    async def read_file(self, *a, **k): return ""
    async def write_file(self, *a, **k): return True
    async def list_files(self, *a, **k): return [FileEntry(name="f", path="f")]
    async def git_action(self, *a, **k): return GitStatusInfo(branch="main", is_clean=True)
    async def process_list(self, *a, **k): raise NotImplementedError
    async def application_list(self, *a, **k): raise NotImplementedError
    async def launch_application(self, *a, **k): raise NotImplementedError
    async def close_application(self, *a, **k): raise NotImplementedError
    async def install_application(self, *a, **k): raise NotImplementedError
    async def uninstall_application(self, *a, **k): raise NotImplementedError
    async def service_action(self, *a, **k): raise NotImplementedError
    async def snapshot(self, *a, **k): raise NotImplementedError
    async def restore_snapshot(self, *a, **k): raise NotImplementedError


def test_coordinate_validation_tracks_screen_from_observe():
    comp = _StubComputer(width=1366, height=768)
    agent = ComputerUseAgent(computer_provider=comp)
    assert agent._screen_width == 1920  # default before observe
    _run(agent.observe(comp.workspace_id))
    assert agent._screen_width == 1366
    assert agent._screen_height == 768


def test_sandbox_agent_observation_does_not_capture_desktop():
    comp = _StubComputer()
    agent = ComputerUseAgent(computer_provider=comp, observe_desktop=False)
    _run(agent.observe(comp.workspace_id))
    assert agent._last_screenshot_b64 == ""


def test_off_screen_click_is_blocked_not_executed():
    comp = _StubComputer(width=800, height=600)
    agent = ComputerUseAgent(computer_provider=comp)
    _run(agent.observe(comp.workspace_id))  # learn real screen size
    trace = _run(agent.execute_action(
        comp.workspace_id, ComputerActionType.GUI_CLICK, "900,500",
        {"x": 900, "y": 500}, "click button",
    ))
    assert trace.status == "BLOCKED"
    assert "out of bounds" in trace.actual_observation.lower() or "outside screen" in trace.actual_observation.lower()
    assert comp.gui_calls == []  # provider never touched


def test_negative_coordinates_blocked():
    comp = _StubComputer()
    agent = ComputerUseAgent(computer_provider=comp)
    trace = _run(agent.execute_action(
        comp.workspace_id, ComputerActionType.GUI_CLICK, "-10,20",
        {"x": -10, "y": 20}, "click",
    ))
    assert trace.status == "BLOCKED"
    assert comp.gui_calls == []


def test_in_bounds_click_executed():
    comp = _StubComputer(width=800, height=600)
    agent = ComputerUseAgent(computer_provider=comp)
    _run(agent.observe(comp.workspace_id))
    trace = _run(agent.execute_action(
        comp.workspace_id, ComputerActionType.GUI_CLICK, "100,200",
        {"x": 100, "y": 200}, "click",
    ))
    assert trace.status != "BLOCKED"
    assert len(comp.gui_calls) == 1
    assert comp.gui_calls[0][1] == 100 and comp.gui_calls[0][2] == 200


def test_non_coordinate_action_not_validated():
    """Terminal exec must never be blocked by coordinate validation."""
    comp = _StubComputer()
    agent = ComputerUseAgent(computer_provider=comp)
    trace = _run(agent.execute_action(
        comp.workspace_id, ComputerActionType.TERMINAL_EXEC, "term",
        {"command": "echo hi"}, "run",
    ))
    assert trace.status != "BLOCKED"
    assert "out of bounds" not in trace.actual_observation.lower()


def test_drag_off_screen_target_blocked():
    comp = _StubComputer(width=800, height=600)
    agent = ComputerUseAgent(computer_provider=comp)
    _run(agent.observe(comp.workspace_id))
    # Source in-bounds, drag target off-screen.
    trace = _run(agent.execute_action(
        comp.workspace_id, ComputerActionType.GUI_DRAG, "100,100",
        {"x": 100, "y": 100, "x2": 900, "y2": 100}, "drag",
    ))
    assert trace.status == "BLOCKED"
    assert comp.gui_calls == []


def test_coordinate_at_exact_edge_blocked():
    """x == width is out of bounds (0-indexed)."""
    comp = _StubComputer(width=800, height=600)
    agent = ComputerUseAgent(computer_provider=comp)
    _run(agent.observe(comp.workspace_id))
    trace = _run(agent.execute_action(
        comp.workspace_id, ComputerActionType.GUI_CLICK, "800,599",
        {"x": 800, "y": 599}, "click",
    ))
    assert trace.status == "BLOCKED"


# ---------------------------------------------------------------------------
# 5. Target Extraction & Self-Targeting Guard Tests
# ---------------------------------------------------------------------------

def test_extract_targets_strips_prior_discoveries():
    goal = "Perform assessment on https://target.internal:8080/api [Prior Discoveries: Discovered host 10.0.0.1 on port 80 and url https://old-target.internal]"
    targets = ComputerUseAgent._extract_targets_from_goal(goal)
    assert targets["urls"] == ["https://target.internal:8080/api"]
    assert "10.0.0.1" not in targets["ips"]
    assert targets["primary_target"] == "https://target.internal:8080/api"


def test_extract_targets_strips_prior_discoveries_no_colon():
    goal = "Scan http://prod.corp.com [Prior Discoveries found 192.168.1.100]"
    targets = ComputerUseAgent._extract_targets_from_goal(goal)
    assert targets["urls"] == ["http://prod.corp.com"]
    assert "192.168.1.100" not in targets["ips"]
    assert targets["primary_target"] == "http://prod.corp.com"


def test_extract_targets_guards_against_localhost_and_loopback():
    goal = "Run audit against 127.0.0.1"
    targets = ComputerUseAgent._extract_targets_from_goal(goal)
    assert "127.0.0.1" in targets["ips"]
    assert targets["primary_target"] == ""

    goal_zero = "Check port 443 on 0.0.0.0"
    targets_zero = ComputerUseAgent._extract_targets_from_goal(goal_zero)
    assert "0.0.0.0" in targets_zero["ips"]
    assert targets_zero["primary_target"] == ""


def test_extract_targets_guards_against_egress_tagged_ip():
    goal = "Scan target 10.0.0.5 with egress IP 198.51.100.1"
    targets = ComputerUseAgent._extract_targets_from_goal(goal)
    assert "10.0.0.5" in targets["ips"]
    assert "198.51.100.1" in targets["ips"]
    assert targets["primary_target"] == "http://10.0.0.5"

    goal_egress_only = "Security audit (egress: 198.51.100.2)"
    targets_egress = ComputerUseAgent._extract_targets_from_goal(goal_egress_only)
    assert targets_egress["primary_target"] == ""


def test_extract_targets_prioritizes_hostnames_and_urls_over_ips():
    goal = "Check target.company.com and secondary 192.168.1.50"
    targets = ComputerUseAgent._extract_targets_from_goal(goal)
    assert targets["primary_target"] == "https://target.company.com"

    goal2 = "Check https://app.example.com and ip 10.0.0.2"
    targets2 = ComputerUseAgent._extract_targets_from_goal(goal2)
    assert targets2["primary_target"] == "https://app.example.com"


# ---------------------------------------------------------------------------
# 6. LLM Safety Refusal Auto-Recovery Tests
# ---------------------------------------------------------------------------

class _StubRouter:
    def __init__(self, response_content: str = ""):
        self.response_content = response_content

    async def complete(self, request):
        return LLMResponse(content=self.response_content, model="test-model")


class _RecordingRouter(_StubRouter):
    def __init__(self):
        super().__init__("ACTION: GOAL_COMPLETE\nTARGET: goal_complete\nPAYLOAD: {}\nEXPECTED: Done")
        self.request = None

    async def complete(self, request):
        self.request = request
        return await super().complete(request)


@pytest.mark.asyncio
async def test_heartbeat_callback_failure_does_not_discard_llm_action(monkeypatch):
    real_sleep = asyncio.sleep

    async def quick_sleep(_delay: float):
        await real_sleep(0.01)

    monkeypatch.setattr("sonic.computer_use.agent.asyncio.sleep", quick_sleep)

    class DelayedRouter(_StubRouter):
        async def complete(self, request):
            await real_sleep(0.04)
            return LLMResponse(
                content=(
                    "ACTION: TERMINAL_EXEC\n"
                    "TARGET: hostname\n"
                    "PAYLOAD: {\"command\": \"hostname\"}\n"
                    "EXPECTED: Hostname output"
                ),
                model="test-model",
            )

    agent = ComputerUseAgent(
        computer_provider=_StubComputer(),
        llm_router=DelayedRouter(),
        execution_plane="headless",
    )

    def broken_heartbeat(_event):
        raise RuntimeError("worklog persistence failed")

    agent._heartbeat_callback = broken_heartbeat
    action_type, target, payload, _ = await agent._llm_choose_action(
        goal="Check hostname",
        observation=ComputerWorldObservation(),
        step_index=1,
        primary_file="",
        test_file="",
    )

    assert action_type == ComputerActionType.TERMINAL_EXEC
    assert target == "hostname"
    assert payload["command"] == "hostname"


@pytest.mark.asyncio
async def test_gui_reasoning_timeout_returns_explicit_blocked_result(monkeypatch):
    import sonic.computer_use.agent as agent_module

    monkeypatch.setattr(agent_module, "_GUI_REASONING_TIMEOUT_SECONDS", 0.01)

    class HangingRouter(_StubRouter):
        async def complete(self, request):
            await asyncio.Event().wait()

    agent = ComputerUseAgent(
        computer_provider=_StubComputer(),
        llm_router=HangingRouter(),
        execution_plane="gui",
    )
    observation = await agent.observe(agent.computer.workspace_id)

    action_type, target, _, expected = await agent._llm_choose_action(
        goal="Use the visible desktop",
        observation=observation,
        step_index=1,
        primary_file="",
        test_file="",
    )

    assert action_type == ActionExecutionStatus.BLOCKED
    assert target == "reasoning-unavailable"
    assert "timed out after 0.01 seconds" in expected
    assert "no action was dispatched" in expected


@pytest.mark.asyncio
async def test_computer_use_attaches_fresh_screenshot_to_vision_request():
    router = _RecordingRouter()
    agent = ComputerUseAgent(
        computer_provider=_StubComputer(),
        llm_router=router,
        execution_plane="gui",
    )
    observation = await agent.observe(agent.computer.workspace_id)
    await agent._llm_choose_action(
        goal="Inspect the visible application",
        observation=observation,
        step_index=1,
        primary_file="",
        test_file="",
    )
    assert router.request is not None
    assert any(message.images for message in router.request.messages)


@pytest.mark.parametrize(
    "refusal_text",
    [
        "I can't help with that request as it involves security assessments.",
        "I cannot assist with this task. Assisting with unauthorized testing is against safety policies.",
        "I must refuse as this appears to be unethical and illegal.",
        "I cannot fulfill this request. Creating malware or attacking systems violates our policies.",
    ],
)
def test_parse_llm_action_refusal_does_not_return_empty_no_op(refusal_text):
    action_type, target, payload, expected = ComputerUseAgent._parse_llm_action(
        refusal_text, "app.py"
    )
    assert action_type == ComputerActionType.TERMINAL_EXEC
    assert payload == {}
    assert expected == "GOAL_COMPLETE"


def test_llm_choose_action_recovers_from_refusal_with_target():
    router = _StubRouter(
        response_content="I cannot assist with this request because it is against safety policies."
    )
    comp = _StubComputer()
    agent = ComputerUseAgent(
        computer_provider=comp,
        llm_router=router,
        execution_plane="headless",
    )
    obs = ComputerWorldObservation()
    action_type, target, payload, expected = _run(
        agent._llm_choose_action(
            goal="Scan vulnerabilities on https://test-target.org/api",
            observation=obs,
            step_index=1,
            primary_file="test.py",
            test_file="test.py",
        )
    )
    # Rule 2: refusal is now recorded as BLOCKED status (no canned shell command)
    assert action_type == ActionExecutionStatus.BLOCKED
    assert target == "refusal-recorded"


def test_llm_choose_action_recovers_from_refusal_without_target():
    router = _StubRouter(
        response_content="I can't help with that. Generating malware is illegal."
    )
    comp = _StubComputer()
    agent = ComputerUseAgent(
        computer_provider=comp,
        llm_router=router,
        execution_plane="headless",
    )
    obs = ComputerWorldObservation()
    action_type, target, payload, expected = _run(
        agent._llm_choose_action(
            goal="Analyze sample in local test environment",
            observation=obs,
            step_index=1,
            primary_file="",
            test_file="",
        )
    )
    # Rule 2: refusal is now recorded as BLOCKED status (no canned shell command)
    assert action_type == ActionExecutionStatus.BLOCKED
    assert payload == {}
    assert expected == "Record model refusal and trigger replan"


def test_execute_action_handles_blocked_refusal_action_type():
    """Verify that execute_action safely handles ActionExecutionStatus.BLOCKED without validation errors."""
    comp = _StubComputer()
    agent = ComputerUseAgent(computer_provider=comp)
    trace = _run(
        agent.execute_action(
            workspace_id="ws-test",
            action_type=ActionExecutionStatus.BLOCKED,
            target_resource="refusal-recorded",
            payload={},
            predicted_outcome="Record model refusal and trigger replan",
        )
    )
    assert trace.status == ActionExecutionStatus.BLOCKED
    assert trace.exit_code == 126
    assert "Action blocked: refusal-recorded" in trace.actual_observation
    # A blocked proposal executes nothing, so it must not be recorded as a
    # terminal command (the old TERMINAL_EXEC label made blocked GUI workers
    # show up in the operator UI as "terminal exec").
    assert trace.action_type != ComputerActionType.TERMINAL_EXEC


def test_computer_decision_trace_coerces_blocked_status_into_valid_action_type():
    """Verify ComputerDecisionTrace Pydantic model coerces ActionExecutionStatus.BLOCKED."""
    trace = ComputerDecisionTrace(
        action_type=ActionExecutionStatus.BLOCKED,
        target_resource="test-target",
        predicted_outcome="test outcome",
        actual_observation="test observation",
    )
    assert trace.action_type == ComputerActionType.TERMINAL_EXEC
    assert trace.target_resource == "test-target"


def test_action_parser_prevents_exec_command_not_found():
    """Verify _parse_llm_action never sets raw_cmd to '_EXEC' when ACTION is TERMINAL_EXEC."""
    from sonic.computer_use.agent import ComputerUseAgent
    
    # When model outputs ACTION: TERMINAL_EXEC with target as command and empty payload
    text = (
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: mkdir -p /home/sonic/workspace/password_security_suite\n"
        "PAYLOAD: {}\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "/home/sonic/workspace/README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") == "mkdir -p /home/sonic/workspace/password_security_suite"
    assert "_EXEC" not in payload.get("command", "")


def test_clean_terminal_command_strips_exec():
    """Verify wrapper markers are removed without inventing successful commands."""
    from sonic.computer_use.agent import ComputerUseAgent

    assert ComputerUseAgent._clean_terminal_command("_EXEC") == ""
    assert ComputerUseAgent._clean_terminal_command("_EXEC ls -la") == "ls -la"
    assert ComputerUseAgent._clean_terminal_command("EXEC ps aux") == "ps aux"


def test_clean_terminal_command_preserves_shell_arguments_and_trailing_operators():
    from sonic.computer_use.agent import ComputerUseAgent

    command = "printf '%s\\n' 'two words';"
    assert ComputerUseAgent._clean_terminal_command(command) == command
    assert ComputerUseAgent._clean_terminal_command(
        json.dumps({"command": command})
    ) == command
    assert ComputerUseAgent._clean_terminal_command(None) == ""


def test_clean_terminal_command_removes_explanatory_parenthetical_and_following_error():
    from sonic.computer_use.agent import ComputerUseAgent

    command = (
        "whoami (or we can put the command in the payload): "
        "bash: -c: line 1: syntax error"
    )
    assert ComputerUseAgent._clean_terminal_command(command) == "whoami"
    quoted = "printf '%s\\n' 'two words (or more)'"
    assert ComputerUseAgent._clean_terminal_command(quoted) == quoted


@pytest.mark.asyncio
async def test_gui_reasoning_unavailable_stops_mission_without_dispatch(monkeypatch):
    comp = _StubComputer()
    agent = ComputerUseAgent(computer_provider=comp, execution_plane="gui")
    choose_action = AsyncMock(return_value=(
        ActionExecutionStatus.BLOCKED,
        "reasoning-unavailable",
        {},
        "GUI reasoning timed out; no action was dispatched.",
    ))
    monkeypatch.setattr(agent, "choose_action", choose_action)

    traces = await agent.run_mission(comp.workspace_id, "Open a visible application", steps=3)

    choose_action.assert_awaited_once()
    assert len(traces) == 1
    assert comp.gui_calls == []
    assert comp.terminal_calls == []
