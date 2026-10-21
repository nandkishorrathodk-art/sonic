"""Focused coverage for execution evidence exposed on computer-use traces."""

import asyncio
import base64
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw

from sonic.computer.models import (
    ComputerState,
    GitStatusInfo,
    GUIAction,
    GUIActionType,
    ScreenObservation,
)
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.action_contracts import ActionReceipt, GUIActionIntent
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
    ComputerWorldObservation,
)
from sonic.safety.action_policy import ActionPolicy
from sonic.sandbox.provider import ExecResult

_FRAME_BUFFER = BytesIO()
Image.new("RGB", (1280, 800), color=(20, 30, 40)).save(_FRAME_BUFFER, format="PNG")
_FRAME_BASE64 = base64.b64encode(_FRAME_BUFFER.getvalue()).decode("ascii")


class _ReceiptComputer(ComputerProvider):
    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        self.gui_calls = 0
        self.commands: list[str] = []
        self.git_actions: list[tuple[str, dict]] = []

    async def create(self, *args, **kwargs): return type("Workspace", (), {"id": "workspace"})()
    async def destroy(self, *args, **kwargs): return True
    async def status(self, workspace_id):
        return ComputerState(
            workspace_id=workspace_id,
            tenant_id="test",
            active_application="Desktop",
        )

    async def screenshot(self, workspace_id):
        return _avcs_screen(
            frame_index=1,
            scene_hash="before-screen",
        )
    async def list_files(self, *args, **kwargs): return []
    async def read_file(self, *args, **kwargs): return ""
    async def write_file(self, *args, **kwargs): return True
    async def git_action(self, workspace_id, action, **kwargs):
        self.git_actions.append((action, kwargs))
        return GitStatusInfo(branch="main", is_clean=True)
    async def application_list(self, *args, **kwargs): return []
    async def process_list(self, *args, **kwargs): return []
    async def launch_application(self, *args, **kwargs): return True
    async def close_application(self, *args, **kwargs): return True
    async def install_application(self, *args, **kwargs): return True, ""
    async def uninstall_application(self, *args, **kwargs): return True
    async def service_action(self, *args, **kwargs): return None

    async def terminal(self, workspace_id, command, timeout=60, actor="operator"):
        self.commands.append(command)
        return ExecResult(
            command=command,
            exit_code=self.exit_code,
            stdout="real output" if self.exit_code == 0 else "",
            stderr="" if self.exit_code == 0 else "real failure",
        )

    async def gui_action(self, workspace_id, action, actor="operator"):
        self.gui_calls += 1
        return _avcs_screen(
            frame_index=2,
            scene_hash="after-screen",
            active_window="observed window",
        )


def _avcs_screen(
    screenshot: str | None = None,
    *,
    frame_index: int = 1,
    scene_hash: str | None = None,
    active_window: str = "Desktop",
    **kwargs,
) -> ScreenObservation:
    if screenshot is None:
        from hashlib import sha256

        seed = scene_hash or "before-screen"
        image = Image.new("RGB", (1280, 800), color=(30, 40, 50))
        draw = ImageDraw.Draw(image)
        for row in range(8):
            for column in range(8):
                cell = sha256(f"{seed}:{row}:{column}".encode()).digest()
                color = tuple(48 + value % 192 for value in cell[:3])
                draw.rectangle(
                    (
                        column * 160,
                        row * 100,
                        (column + 1) * 160 - 1,
                        (row + 1) * 100 - 1,
                    ),
                    fill=color,
                )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        screenshot = base64.b64encode(buffer.getvalue()).decode("ascii")

    from sonic.computer_use.avcs_perception import AVCSPerception

    avcs_state = AVCSPerception().perceive_png(
        base64.b64decode(screenshot, validate=True)
    )
    avcs_state["frame_index"] = frame_index
    return ScreenObservation(
        screenshot_base64=screenshot,
        avcs_state=avcs_state,
        perception_version=frame_index,
        width=1280,
        height=800,
        active_window=active_window,
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
        **kwargs,
    )


def _run(coro):
    return asyncio.run(coro)


def _execute_gui_move_from_stale_observation(
    computer,
    screen,
    perception_version=0,
    observation_age_seconds=3,
):
    agent = ComputerUseAgent(computer_provider=computer)

    async def run():
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=perception_version,
            observation_age_ms=3_000,
        )
        agent._last_screen_observation_received_at = (
            asyncio.get_running_loop().time() - observation_age_seconds
        )
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_MOVE,
            "640,400",
            {"x": 640, "y": 400},
            "cursor reaches requested position",
        )

    return _run(run())


def test_terminal_trace_exposes_receipt_with_exit_code_and_delta():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.TERMINAL_EXEC,
            "inspect",
            {"command": "inspect"},
            "real output",
        )
    )

    assert isinstance(trace.action_receipt, ActionReceipt)
    assert trace.action_receipt.exit_code == 0
    assert trace.action_receipt.status == trace.status
    assert trace.action_receipt.evidence == "real output"
    assert trace.action_receipt.state_delta.changed["exit_code"] == 0


def test_failed_terminal_trace_receipt_is_not_successful():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer(exit_code=1))

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.TERMINAL_EXEC,
            "inspect",
            {"command": "inspect"},
            "real output",
        )
    )

    assert trace.status == ActionExecutionStatus.RECOVERED
    assert trace.action_receipt.status == ActionExecutionStatus.RECOVERED
    assert trace.action_receipt.exit_code == 1
    assert trace.action_receipt.after["exit_code"] == 1


def test_native_kernel_deny_blocks_provider_dispatch(monkeypatch):
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    monkeypatch.setattr(
        agent,
        "_native_govern_action",
        lambda *_: {
            "executable": True,
            "native_verdict": "Deny",
            "reason": "target is blocked",
        },
    )

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.TERMINAL_EXEC,
            "inspect target",
            {"command": "inspect"},
            "inspection output",
        )
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert trace.exit_code == 126
    assert computer.commands == []


def test_native_kernel_approval_required_blocks_without_operator_approval(monkeypatch):
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    monkeypatch.setattr(
        agent,
        "_native_govern_action",
        lambda *_: {
            "executable": True,
            "native_verdict": "RequireApproval",
            "reason": "operator approval is required",
        },
    )

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.TERMINAL_EXEC,
            "publish artifact",
            {"command": "publish artifact"},
            "published",
        )
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert trace.exit_code == 126
    assert "operator approval" in trace.actual_observation
    assert computer.commands == []


def test_git_commit_requires_explicit_operator_approval(monkeypatch):
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    monkeypatch.setattr(
        agent,
        "_native_govern_action",
        lambda *_: {
            "executable": True,
            "native_verdict": "Allow",
            "reason": "allowed",
        },
    )

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.GIT_COMMIT,
            "commit",
            {"message": "Save verified work"},
            "commit recorded",
        )
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert trace.exit_code == 126
    assert computer.git_actions == []


def test_git_commit_dispatches_only_after_operator_approval(monkeypatch):
    computer = _ReceiptComputer()
    approvals = []

    async def approve(action_type, target, reason):
        approvals.append((action_type, target, reason))
        return True

    agent = ComputerUseAgent(
        computer_provider=computer,
        operator_approval_callback=approve,
    )
    monkeypatch.setattr(
        agent,
        "_native_govern_action",
        lambda *_: {
            "executable": True,
            "native_verdict": "RequireApproval",
            "reason": "native policy requires approval",
        },
    )

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.GIT_COMMIT,
            "commit",
            {"message": "Save verified work"},
            "commit recorded",
        )
    )

    assert approvals == [
        ("GIT_COMMIT", "commit", "GIT_COMMIT is a consequential operation")
    ]
    assert len(computer.git_actions) == 1
    assert computer.git_actions[0][0] == "commit"
    assert trace.status == ActionExecutionStatus.COMPLETED


def test_external_gui_control_requires_operator_approval(monkeypatch):
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(
        computer_provider=computer,
        operator_approval_callback=lambda *_: False,
    )
    monkeypatch.setattr(
        agent,
        "_native_govern_action",
        lambda *_: {
            "executable": True,
            "native_verdict": "Allow",
            "reason": "allowed",
        },
    )

    async def run_click():
        screen = await computer.screenshot("workspace")
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=1,
            observation_age_ms=0,
            observation_received_monotonic_ns=time.perf_counter_ns(),
        )
        agent._last_observed_screen = screen
        agent._last_screen_observation_received_at = (
            asyncio.get_running_loop().time()
        )
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_CLICK,
            "Send message",
            {"x": 20, "y": 30},
            "message sent",
        )

    trace = _run(run_click())

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert "operator approval" in trace.actual_observation
    assert computer.gui_calls == 0


def test_gui_trace_receipt_uses_observed_screen_transition():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)

    async def run_click():
        screen = await computer.screenshot("workspace")
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=1,
            observation_age_ms=0,
            observation_received_monotonic_ns=time.perf_counter_ns(),
        )
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_observed_screen = screen
        agent._last_gui_dispatch_evidence = (
            screen,
            asyncio.get_running_loop().time(),
        )
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_CLICK,
            "10,10",
            {"x": 10, "y": 10},
            "screen changes",
        )

    trace = _run(run_click())

    assert trace.status != ActionExecutionStatus.BLOCKED, trace.actual_observation
    assert computer.gui_calls == 1
    before_hash = trace.action_receipt.before["screen_hash"]
    after_hash = trace.action_receipt.after["screen_hash"]
    assert before_hash
    assert after_hash
    assert before_hash != after_hash
    assert "screen_hash" in trace.action_receipt.state_delta.changed


def test_configured_avcs_grounder_binds_semantic_target_to_current_frame(monkeypatch):
    class _GroundingComputer(_ReceiptComputer):
        def __init__(self):
            super().__init__()
            self.gui_actions = []
            self.frame = ""

    computer = _GroundingComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    image = Image.new("RGB", (1280, 800), color=(20, 30, 40))
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    screenshot = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    computer.frame = screenshot

    from sonic.computer_use.avcs_perception import AVCSPerception

    grounder = AVCSPerception()
    grounder.request = None

    def ground_target_png(png_bytes, query, width, height, window_geometry=None):
        grounder.request = (png_bytes, query, width, height, window_geometry)
        return {
            "x": 321,
            "y": 123,
            "held_out_point_accuracy": 0.91,
            "query_score": 0.95,
            "query_score_threshold": 0.8,
            "engine": "avcs_goal_grounder",
            "frame_index": grounder._frame_index,
            "scene_hash": grounder._last_scene_hash,
            "checkpoint_metadata": {
                "query_confidence_calibration": {
                    "evaluation": {
                        "metric": "empirical_precision_at_threshold",
                        "passed": True,
                        "samples": 20,
                        "threshold": 0.8,
                        "precision": 0.95,
                        "required_precision": 0.9,
                    },
                },
            },
            "latency_ms": 0.25,
        }

    grounder.ground_target_png = ground_target_png
    agent._avcs_perception = grounder
    monkeypatch.setenv("SONIC_AVCS_GROUNDING_CHECKPOINT", "evaluated-checkpoint.pt")

    async def run_click():
        screen = agent._ensure_avcs_state(
            _avcs_screen(screenshot, scene_hash=screenshot)
        )
        agent._last_world_observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=screen.perception_version,
            observation_age_ms=0,
            observation_received_monotonic_ns=time.perf_counter_ns(),
        )
        agent._last_observed_screen = screen
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_screenshot_b64 = screenshot
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        agent._last_observed_screen.active_window_geometry = (100, 80, 900, 600)
        return await agent._ground_target_with_avcs("Save button", screen=screen)

    result = _run(run_click())

    assert result["engine"] == "avcs_goal_grounder"
    assert (result["x"], result["y"]) == (321, 123)
    assert result["frame_index"] == agent._last_observed_screen.avcs_state["frame_index"]
    assert result["scene_hash"] == agent._last_observed_screen.avcs_state["scene_hash"]
    assert grounder.request == (
        image_buffer.getvalue(),
        "Save button",
        1280,
        800,
        (100, 80, 900, 600),
    )


def test_agent_png_perception_binds_avcs_state_to_provider_frame_id():
    agent = ComputerUseAgent(
        computer_provider=_ReceiptComputer(),
        execution_plane="gui",
    )
    screen = _avcs_screen()
    screen.frame_id = 42

    processed = agent._ensure_avcs_state(screen)

    assert processed.avcs_state["source_frame_id"] == 42
    assert ComputerUseAgent._fresh_avcs_reflex_summary(processed) != "unavailable"


def test_agent_processes_raw_rgb_observation_locally_and_keeps_frame_private():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer())
    raw_rgb = bytes((24, 48, 72)) * (64 * 48)
    screen = ScreenObservation(
        raw_rgb=raw_rgb,
        frame_id=17,
        width=64,
        height=48,
        capture_age_ms=0,
    )

    observed = agent._ensure_avcs_state(screen)

    assert observed.avcs_state["source_frame_id"] == 17
    assert observed.avcs_state["frame_index"] > 0
    assert observed.avcs_state["width"] == 64
    assert observed.avcs_state["height"] == 48
    assert "raw_rgb" not in observed.model_dump()
    assert "raw_rgb" not in observed.model_dump_json()


def test_gui_dispatch_accepts_raw_avcs_frame_without_png_serialization():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    before = agent._ensure_avcs_state(
        ScreenObservation(
            raw_rgb=bytes((24, 48, 72)) * (64 * 48),
            frame_id=20,
            width=64,
            height=48,
            capture_age_ms=0,
            desktop_state="INTERACTIVE",
        )
    )
    after = ScreenObservation(
        raw_rgb=bytes((90, 40, 10)) * (64 * 48),
        frame_id=21,
        width=64,
        height=48,
        capture_age_ms=0,
        desktop_state="INTERACTIVE",
    )

    async def gui_action(*_args, **_kwargs):
        return after

    computer.gui_action = gui_action
    agent._last_observed_screen = before
    agent._last_gui_action_observation_valid = False
    agent._last_gui_state_observation = None
    agent._last_screen_has_pixels = True
    agent._last_screen_frame_id = before.frame_id
    agent._last_screenshot_b64 = ""

    async def dispatch():
        return await agent._dispatch_gui_action(
            "workspace",
            GUIAction(action=GUIActionType.CLICK, x=10, y=10),
        )

    result = asyncio.run(dispatch())

    assert result.frame_id == 21
    assert result.screenshot_base64 == ""
    assert result.avcs_state["source_frame_id"] == 21
    assert agent._last_gui_action_observation_valid is True
    assert agent._last_screen_frame_id == 21
    assert agent._last_screenshot_b64 == ""


def test_dashboard_snapshot_encodes_raw_frame_only_at_explicit_api_boundary():
    from sonic.api.routes.workstation import _dashboard_screen_payload

    screen = ScreenObservation(
        raw_rgb=bytes((24, 48, 72)) * (8 * 6),
        frame_id=8,
        width=8,
        height=6,
        desktop_state="INTERACTIVE",
    )

    payload = _dashboard_screen_payload(screen)

    assert payload["screenshot_base64"]
    assert "raw_rgb" not in payload
    image_bytes = base64.b64decode(payload["screenshot_base64"], validate=True)
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_gui_action_intent_is_checked_under_display_lease_before_dispatch():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer)
    screen = _avcs_screen()

    async def run_click():
        observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=1,
            observation_age_ms=0,
            observation_received_monotonic_ns=time.perf_counter_ns(),
        )
        agent._last_world_observation = observation
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_observed_screen = screen
        agent._last_gui_dispatch_evidence = (
            screen,
            asyncio.get_running_loop().time(),
        )
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        agent._pending_gui_action_intent = GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="10,10",
            payload={"x": 10, "y": 10},
            expected_outcome="screen changes",
            observation=observation,
        )
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_CLICK,
            "10,10",
            {"x": 10, "y": 10},
            "screen changes",
        )

    trace = _run(run_click())

    assert trace.status != ActionExecutionStatus.BLOCKED, trace.actual_observation
    assert computer.gui_calls == 1
    assert trace.action_receipt.intent["source_perception_version"] == 1
    assert trace.action_receipt.intent["source_active_window"] == "Desktop"
    assert not hasattr(agent, "_pending_gui_action_intent")


@pytest.mark.parametrize(
    ("fresh_pixels", "should_block"),
    [("before-screen", False), ("changed-screen", True)],
)
def test_expired_gui_action_intent_renews_only_after_unchanged_state_revalidation(
    fresh_pixels,
    should_block,
):
    computer = _ReceiptComputer()

    async def screenshot(_workspace_id):
        return _avcs_screen(
            frame_index=2,
            scene_hash=(
                "changed-scene" if fresh_pixels == "changed-screen" else "before-screen"
            ),
        )

    computer.screenshot = screenshot
    agent = ComputerUseAgent(computer_provider=computer, gui_only=True)
    screen = _avcs_screen()

    async def run_expired_click():
        observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=1,
            observation_age_ms=0,
            observation_received_monotonic_ns=time.perf_counter_ns(),
        )
        agent._last_world_observation = observation
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        agent._pending_gui_action_intent = GUIActionIntent.create(
            action_type="GUI_CLICK",
            target="10,10",
            payload={"x": 10, "y": 10},
            expected_outcome="screen changes",
            observation=observation,
            ttl_ms=100,
        )
        await asyncio.sleep(0.15)
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_CLICK,
            "10,10",
            {"x": 10, "y": 10},
            "screen changes",
        )

    trace = _run(run_expired_click())

    if should_block:
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert "AVCS scene changed" in trace.actual_observation
        assert computer.gui_calls == 0
    else:
        assert trace.status != ActionExecutionStatus.BLOCKED, trace.actual_observation
        assert computer.gui_calls == 1


def test_run_mission_creates_a_bounded_intent_for_grounded_gui_actions():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer, execution_plane="gui")
    screen = _avcs_screen()
    observation = ComputerWorldObservation(
        screen=screen,
        active_application="Desktop",
        perception_version=1,
        observation_age_ms=0,
        observation_received_monotonic_ns=time.perf_counter_ns(),
    )
    observed_intents = []

    async def observe(_workspace_id):
        agent._last_world_observation = observation
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        return observation

    async def choose_action(*_args):
        return (
            ComputerActionType.GUI_CLICK,
            "10,10",
            {
                "x": 10,
                "y": 10,
                "postcondition": {"active_window": "observed window"},
            },
            "screen changes",
        )

    original_execute_action = agent.execute_action

    async def capture_intent(*args):
        observed_intents.append(getattr(agent, "_pending_gui_action_intent", None))
        return await original_execute_action(*args)

    agent.observe = observe
    agent.choose_action = choose_action
    agent.execute_action = capture_intent

    traces = _run(agent.run_mission("workspace", "click the observed control", steps=1))

    assert computer.gui_calls == 1
    assert len(observed_intents) == 1
    assert isinstance(observed_intents[0], GUIActionIntent)
    assert observed_intents[0].expected_outcome == "screen changes"
    assert traces[0].status != ActionExecutionStatus.BLOCKED, traces[0].actual_observation


def test_run_mission_dispatches_gui_action_from_raw_avcs_rgb_observation():
    raw_rgb = bytes((24, 48, 72)) * (128 * 80)

    class RawFrameComputer(_ReceiptComputer):
        def __init__(self):
            super().__init__()
            self.frame_id = 1

        async def screenshot(self, workspace_id):
            self.frame_id += 1
            return ScreenObservation(
                raw_rgb=raw_rgb,
                frame_id=self.frame_id,
                width=128,
                height=80,
                active_window="Desktop",
                desktop_state="INTERACTIVE",
                capture_age_ms=0,
            )

        async def gui_action(self, workspace_id, action, actor="operator"):
            self.gui_calls += 1
            self.frame_id += 1
            return ScreenObservation(
                raw_rgb=raw_rgb,
                frame_id=self.frame_id,
                width=128,
                height=80,
                active_window="observed window",
                desktop_state="INTERACTIVE",
                capture_age_ms=0,
            )

    computer = RawFrameComputer()
    agent = ComputerUseAgent(computer_provider=computer, execution_plane="gui")
    initial_screen = agent._ensure_avcs_state(
        ScreenObservation(
            raw_rgb=raw_rgb,
            frame_id=1,
            width=128,
            height=80,
            active_window="Desktop",
            desktop_state="INTERACTIVE",
            capture_age_ms=0,
        )
    )
    observation = ComputerWorldObservation(
        screen=initial_screen,
        active_application="Desktop",
        perception_version=1,
        observation_age_ms=0,
        observation_received_monotonic_ns=time.perf_counter_ns(),
    )

    async def observe(_workspace_id):
        agent._last_world_observation = observation
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_screenshot_b64 = ""
        agent._screen_width = initial_screen.width
        agent._screen_height = initial_screen.height
        return observation

    async def choose_action(*_args):
        return (
            ComputerActionType.GUI_CLICK,
            "10,10",
            {
                "x": 10,
                "y": 10,
                "postcondition": {"active_window": "observed window"},
            },
            "the expected window is active",
        )

    agent.observe = observe
    agent.choose_action = choose_action

    traces = _run(agent.run_mission("workspace", "activate the observed window", steps=1))

    assert computer.gui_calls == 1
    assert traces
    assert traces[0].status != ActionExecutionStatus.BLOCKED, traces[0].actual_observation
    assert traces[0].action_receipt is not None
    assert traces[0].action_receipt.intent["source_perception_version"] > 0


def test_stale_screenshot_intent_refresh_uses_only_a_fresh_screenshot():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(computer_provider=computer, execution_plane="gui")
    source_screen = _avcs_screen()
    source = ComputerWorldObservation(
        screen=source_screen,
        active_application="Desktop",
        perception_version=4,
        observation_age_ms=3_000,
        observation_received_monotonic_ns=time.perf_counter_ns(),
    )

    async def fail_if_full_observe_is_used(_workspace_id):
        raise AssertionError("GUI intent refresh must not run the full observation pipeline")

    agent.observe = fail_if_full_observe_is_used
    intent = _run(
        agent._create_gui_action_intent(
            "workspace",
            action_type=ComputerActionType.GUI_CLICK,
            target="observed control",
            payload={"x": 10, "y": 10},
            expected_outcome="screen changes",
            observation=source,
            ttl_ms=2_000,
        )
    )

    assert intent.source_perception_version > 0
    assert intent.source_active_window == "Desktop"
    assert (
        agent._last_gui_dispatch_evidence[0].screenshot_base64
        == source_screen.screenshot_base64
    )


def test_slow_gui_reasoning_revalidates_unchanged_screenshot_before_dispatch():
    computer = _ReceiptComputer()
    screen = _avcs_screen()

    trace = _execute_gui_move_from_stale_observation(computer, screen)

    assert trace.status != ActionExecutionStatus.BLOCKED, trace.actual_observation
    assert computer.gui_calls == 1


def test_slow_gui_reasoning_fails_closed_when_screenshot_changed():
    class ChangedScreenComputer(_ReceiptComputer):
        async def screenshot(self, workspace_id):
            return _avcs_screen(
                frame_index=2,
                scene_hash="changed-scene",
            )

    computer = ChangedScreenComputer()
    screen = _avcs_screen()

    trace = _execute_gui_move_from_stale_observation(computer, screen)

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert "AVCS scene changed" in trace.actual_observation
    assert computer.gui_calls == 0


@pytest.mark.asyncio
async def test_refreshed_gui_intent_keeps_grounding_frame_identity_in_sync():
    rgb = bytes((30, 40, 50)) * (8 * 6)
    state = {
        "engine": "avcs_reflex",
        "frame_index": 1,
        "source_frame_id": 1,
        "scene_hash": "stable-scene",
        "width": 8,
        "height": 6,
    }
    source_screen = ScreenObservation(
        raw_rgb=rgb,
        frame_id=1,
        avcs_state=state,
        perception_version=1,
        width=8,
        height=6,
        active_window="Desktop",
        capture_age_ms=3_000,
        desktop_state="INTERACTIVE",
    )
    refreshed_screen = ScreenObservation(
        raw_rgb=rgb,
        frame_id=2,
        avcs_state={**state, "frame_index": 2, "source_frame_id": 2},
        perception_version=2,
        width=8,
        height=6,
        active_window="Desktop",
        capture_age_ms=0,
        desktop_state="INTERACTIVE",
    )
    computer = _ReceiptComputer()
    computer.screenshot = AsyncMock(return_value=refreshed_screen)
    agent = ComputerUseAgent(computer_provider=computer, execution_plane="gui")
    agent._ensure_avcs_state = lambda screen: screen
    observation = ComputerWorldObservation(
        screen=source_screen,
        active_application="Desktop",
        perception_version=1,
        observation_age_ms=3_000,
        observation_received_monotonic_ns=time.perf_counter_ns(),
    )

    await agent._create_gui_action_intent(
        "workspace",
        action_type=ComputerActionType.GUI_CLICK,
        target="Applications menu",
        payload={"x": 1, "y": 1},
        expected_outcome="Menu opens",
        observation=observation,
        ttl_ms=2_000,
    )

    assert agent._last_observed_screen is refreshed_screen
    assert agent._last_screen_has_pixels is True
    assert agent._last_screen_frame_id == 2

    class _Grounder:
        def ground_target_rgb_bytes(
            self, frame, query, width, height, geometry, source_frame_id
        ):
            assert source_frame_id == 2
            return None

    agent._avcs_perception = _Grounder()
    assert await agent._ground_target_with_avcs("Applications menu") is None


@pytest.mark.asyncio
async def test_run_mission_stops_after_missing_avcs_grounder_blocks_gui_action():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer(), execution_plane="gui")
    observation = ComputerWorldObservation(
        screen=_avcs_screen(),
        active_application="Desktop",
        perception_version=1,
        observation_age_ms=0,
        observation_received_monotonic_ns=time.perf_counter_ns(),
    )
    agent.observe = AsyncMock(return_value=observation)
    agent.choose_action = AsyncMock(return_value=(
        ComputerActionType.GUI_CLICK,
        "Applications menu button",
        {"postcondition": {"scene_changed": True}},
        "The Applications menu is open",
    ))
    agent._single_gui_postcondition_error = lambda *_args: None
    agent._create_gui_action_intent = AsyncMock(return_value=object())
    blocked_trace = ComputerDecisionTrace(
        step_index=1,
        action_type=ComputerActionType.GUI_CLICK,
        target_resource="Applications menu button",
        payload="{}",
        predicted_outcome="",
        actual_observation=(
            "GUI action blocked: the local AVCS grounder did not resolve the "
            "target 'Applications menu button'. "
            "SONIC_AVCS_GROUNDING_CHECKPOINT is not configured"
        ),
        expected_observation="",
        info_gain=0.0,
        recovery_attempted=False,
        status=ActionExecutionStatus.BLOCKED,
        exit_code=126,
    )

    async def execute_blocked(*_args, **_kwargs):
        agent.traces.append(blocked_trace)
        return blocked_trace

    agent.execute_action = AsyncMock(side_effect=execute_blocked)

    await agent.run_mission("workspace", "Open the Applications menu", steps=5)

    assert agent.observe.await_count == 1
    assert agent.choose_action.await_count == 1
    assert agent.execute_action.await_count == 1
    assert any(item["action"] == "GUI_CAPABILITY_UNAVAILABLE" for item in agent.history)


@pytest.mark.asyncio
async def test_run_mission_stops_before_repeating_unverified_gui_key():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer(), execution_plane="gui")
    observation = ComputerWorldObservation(
        screen=_avcs_screen(),
        active_application="Desktop",
        perception_version=1,
        observation_age_ms=0,
        observation_received_monotonic_ns=time.perf_counter_ns(),
    )
    agent.observe = AsyncMock(return_value=observation)
    agent.choose_action = AsyncMock(side_effect=[
        (
            ComputerActionType.GUI_KEYPRESS,
            "keyboard",
            {"key": "Win"},
            "The menu opens",
        ),
        (
            ComputerActionType.GUI_HOTKEY,
            "keyboard",
            {"keys": "Win"},
            "The menu opens",
        ),
    ])
    agent._single_gui_postcondition_error = lambda *_args: None
    agent._create_gui_action_intent = AsyncMock(return_value=object())

    async def execute_unverified_key(_workspace_id, action_type, target, payload, *_args):
        agent._recent_action_signatures.append(
            (action_type.value, target.lower(), str(payload))
        )
        agent._consecutive_unchanged_screens = 1
        trace = ComputerDecisionTrace(
            step_index=len(agent.traces) + 1,
            action_type=action_type,
            target_resource=target,
            payload=str(payload),
            predicted_outcome="",
            actual_observation="AVCS saw no pixel-scene change.",
            expected_observation="",
            info_gain=0.0,
            recovery_attempted=False,
            status=ActionExecutionStatus.UNVERIFIED,
        )
        agent.traces.append(trace)
        return trace

    agent.execute_action = AsyncMock(side_effect=execute_unverified_key)

    await agent.run_mission("workspace", "Open the Applications menu", steps=5)

    assert agent.choose_action.await_count == 2
    assert agent.execute_action.await_count == 1
    assert any(item["action"] == "WATCHDOG_ABORT" for item in agent.history)


def test_recent_gui_observation_fails_closed_when_screenshot_changed():
    class ChangedScreenComputer(_ReceiptComputer):
        async def screenshot(self, workspace_id):
            return _avcs_screen(
                frame_index=2,
                scene_hash="changed-scene",
            )

    computer = ChangedScreenComputer()
    screen = _avcs_screen()

    trace = _execute_gui_move_from_stale_observation(
        computer,
        screen,
        observation_age_seconds=0,
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert "AVCS scene changed" in trace.actual_observation
    assert computer.gui_calls == 0


class _PostconditionComputer(_ReceiptComputer):
    def __init__(self, *, screen_changed: bool):
        super().__init__()
        self.screen_changed = screen_changed

    async def gui_action(self, workspace_id, action, actor="operator"):
        self.gui_calls += 1
        return _avcs_screen(
            frame_index=2,
            scene_hash="after-screen" if self.screen_changed else "before-screen",
        )


def _execute_click_with_postcondition(computer, postcondition):
    agent = ComputerUseAgent(computer_provider=computer, execution_plane="gui")
    payload = {"x": 100, "y": 100, "postcondition": postcondition}
    screen = _avcs_screen()

    async def run():
        observation = ComputerWorldObservation(
            screen=screen,
            active_application="Desktop",
            perception_version=4,
            observation_age_ms=0,
            observation_received_monotonic_ns=time.perf_counter_ns(),
        )
        agent._last_world_observation = observation
        agent._last_observed_screen = screen
        agent._last_screen_observation_received_at = asyncio.get_running_loop().time()
        agent._last_gui_dispatch_evidence = (
            screen,
            asyncio.get_running_loop().time(),
        )
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        agent._pending_gui_action_intent = GUIActionIntent.create(
            action_type=ComputerActionType.GUI_CLICK.value,
            target="100,100",
            payload=payload,
            expected_outcome="The create dialog appears",
            observation=observation,
        )
        return await agent.execute_action(
            "workspace",
            ComputerActionType.GUI_CLICK,
            "100,100",
            payload,
            "The create dialog appears",
        )

    return agent, _run(run())


def test_gui_action_is_verified_only_when_fresh_postcondition_matches():
    agent, trace = _execute_click_with_postcondition(
        _PostconditionComputer(screen_changed=True),
        {"screen_changed": True},
    )

    assert trace.status == ActionExecutionStatus.VERIFIED
    assert "postcondition observed" in trace.verification_evidence
    assert agent.computer.gui_calls == 1


def test_gui_action_with_unobserved_postcondition_is_unverified():
    agent, trace = _execute_click_with_postcondition(
        _PostconditionComputer(screen_changed=False),
        {"screen_changed": True},
    )

    assert trace.status == ActionExecutionStatus.UNVERIFIED, trace.actual_observation
    assert "POSTCONDITION UNVERIFIED" in trace.actual_observation
    assert agent.computer.gui_calls == 1


def test_gui_postcondition_rejects_screen_changed_false_when_screen_changed():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer())

    passed, message = agent._burst_postcondition_ok(
        {"screen_changed": False},
        object(),
        changed=True,
    )

    assert passed is False
    assert "did not match" in message


@pytest.mark.asyncio
async def test_reasoning_fallback_stores_refreshed_frame_for_target_grounding():
    refreshed_screen = ScreenObservation(
        raw_rgb=bytes((1, 2, 3)) * (8 * 6),
        frame_id=2,
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 2,
            "source_frame_id": 2,
            "scene_hash": "fresh-scene",
            "width": 8,
            "height": 6,
        },
        capture_age_ms=0,
        perception_version=2,
        width=8,
        height=6,
        desktop_state="INTERACTIVE",
    )

    class _Router:
        async def complete(self, request):
            return SimpleNamespace(
                content=(
                    '{"action":"GUI_SCREENSHOT","target":"visible-desktop",'
                    '"payload":{},"expected":"Inspect the current desktop"}'
                )
            )

    class _Computer:
        async def screenshot(self, workspace_id):
            return refreshed_screen

    agent = ComputerUseAgent(
        computer_provider=_Computer(),
        llm_router=_Router(),
        execution_plane="gui",
    )
    agent._active_workspace_id = "workspace"
    agent._ensure_avcs_state = lambda screen: screen
    agent._build_reasoning_context = lambda *args: ("system", "observation")
    observation = ComputerWorldObservation(
        screen=ScreenObservation(
            width=8,
            height=6,
            capture_age_ms=2_001,
            desktop_state="INTERACTIVE",
        )
    )

    selected = await agent.choose_action("Inspect the desktop", observation)

    assert selected[0] == ComputerActionType.GUI_SCREENSHOT
    assert agent._last_observed_screen is refreshed_screen
    assert observation.screen is refreshed_screen


def test_mission_gui_postcondition_requires_structured_observable_condition():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer())

    assert "structured observable postcondition" in (
        agent._single_gui_postcondition_error(
            ComputerActionType.GUI_CLICK, {"x": 10, "y": 10}
        )
    )
    assert agent._single_gui_postcondition_error(
        ComputerActionType.GUI_CLICK,
        {"x": 10, "y": 10, "postcondition": {"screen_changed": True}},
    ) is None
    assert agent._single_gui_postcondition_error(
        ComputerActionType.GUI_CLICK,
        {"x": 10, "y": 10, "postcondition": {"control_present": "Save"}},
    ) is not None
    assert agent._single_gui_postcondition_error(
        ComputerActionType.GUI_CLICK,
        {"x": 10, "y": 10, "postcondition": {"unknown": "value"}},
    ) is not None


def test_run_mission_blocks_gui_action_without_postcondition():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(
        computer_provider=computer,
        execution_plane="gui",
    )
    screen = _avcs_screen(scene_hash="fresh-frame")
    observation = ComputerWorldObservation(
        screen=screen,
        active_application="Desktop",
        perception_version=3,
        observation_age_ms=0,
        observation_received_monotonic_ns=time.perf_counter_ns(),
    )

    async def observe(_workspace_id):
        agent._last_world_observation = observation
        agent._last_screen_observation_received_at = (
            asyncio.get_running_loop().time()
        )
        agent._last_gui_dispatch_evidence = (
            screen,
            asyncio.get_running_loop().time(),
        )
        agent._last_screenshot_b64 = screen.screenshot_base64
        agent._screen_width = screen.width
        agent._screen_height = screen.height
        return observation

    async def choose_action(*_args):
        return (
            ComputerActionType.GUI_CLICK,
            "Create",
            {"x": 100, "y": 100},
            "The dialog appears",
        )

    agent.observe = observe
    agent.choose_action = choose_action
    traces = _run(agent.run_mission("workspace", "create a folder", steps=1))

    assert traces[0].status == ActionExecutionStatus.BLOCKED
    assert traces[0].target_resource == "gui-postcondition-required"
    assert computer.gui_calls == 0


def test_file_write_trace_exposes_receipt_without_fabricating_exit_code():
    agent = ComputerUseAgent(computer_provider=_ReceiptComputer())

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.FILE_WRITE,
            "notes.txt",
            {"path": "notes.txt", "content": "observed"},
            "file written",
        )
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert isinstance(trace.receipt, ActionReceipt)
    assert trace.receipt.exit_code is None
    assert trace.state_delta is trace.receipt.state_delta
    assert trace.receipt.evidence == "Wrote patch (8 bytes) to notes.txt"


def test_verify_goal_runs_typed_probe_through_action_receipt():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(
        computer_provider=computer,
        safety=ActionPolicy(workspace_root="/home/sonic/workspace"),
    )

    verified, evidence = _run(agent.verify_goal("workspace", "run pytest tests"))

    assert verified is True, evidence
    assert computer.commands == ["python -m pytest -q --tb=short"]
    trace = agent.traces[-1]
    assert trace.action_type == ComputerActionType.VERIFY_PROBE
    assert trace.exit_code == 0
    assert trace.action_receipt.exit_code == 0
    assert "exit_code=0" in evidence


def test_invalid_verification_probe_is_blocked_before_terminal():
    computer = _ReceiptComputer()
    agent = ComputerUseAgent(
        computer_provider=computer,
        safety=ActionPolicy(workspace_root="/home/sonic/workspace"),
    )

    trace = _run(
        agent.execute_action(
            "workspace",
            ComputerActionType.VERIFY_PROBE,
            "pytest",
            {"probe": {"kind": "pytest", "target": "; touch /tmp/pwned"}},
            "verification",
        )
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert trace.exit_code == 126
    assert computer.commands == []


def test_llm_verification_cannot_supply_a_shell_command():
    computer = _ReceiptComputer()
    router = type(
        "Judge",
        (),
        {"complete": staticmethod(
            lambda request: _async_response("YES")
        )},
    )()
    agent = ComputerUseAgent(
        computer_provider=computer,
        llm_router=router,
        safety=ActionPolicy(workspace_root="/home/sonic/workspace"),
    )

    verified, _ = _run(agent.verify_goal("workspace", "run pytest", use_llm=True))

    assert verified is True
    assert computer.commands == ["python -m pytest -q --tb=short"]


async def _async_response(content: str):
    return type("Response", (), {"content": content})()
