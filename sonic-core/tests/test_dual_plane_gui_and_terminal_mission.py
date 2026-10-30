"""
End-to-End Verification for SONIC AI-Human Digital Being & Multi-App Dual-Plane Workstation
===========================================================================================

Verifies:
1. Priority #1 AI-Human Digital Being Identity (`AI_HUMAN_BEING_IDENTITY`, `BeingVault`, persona profile).
2. Removal of the 14 false-positive blocks (`GUI_HOTKEY`, `GIT_STATUS`, `GIT_DIFF`, `GIT_BRANCH`,
   `GIT_COMMIT`, `GIT_PUSH`, `BROWSER_SELECT`, `BROWSER_KEYPRESS`, `BROWSER_SCROLL`, `WAIT_FOR_HUMAN`,
   normal commit quotes, `pkill`/`chmod +x`, and open web browsing to `.test` platforms).
3. `1280x800` Visual Grounding accuracy (`extract_bbox_midpoint` preserves pixel coordinates).
4. Multi-Application Desktop GUI (`APP_LAUNCH` App #1 + `APP_LAUNCH` App #2 + `APP_FOCUS` switching
   back and forth + `GUI_CLICK` + `GUI_TYPE` + `GUI_HOTKEY`) + Concurrent Headless Terminal (`TERMINAL_EXEC`),
   with clean separation of `Open Windows` vs `Available Installed Desktop Applications`.
"""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from sonic.being.identity import get_or_create_being, reset_being_singleton
from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.grounding import extract_bbox_midpoint
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerAutonomyLevel,
    EngineeringMissionMode,
)
from sonic.llm.prompts import AI_HUMAN_BEING_IDENTITY
from sonic.safety.scope import RiskLevel, classify_command_risk
from sonic.safety.sealed import seal_default


@pytest.mark.no_live_infra
def test_ai_human_being_identity_and_vault_persistence(tmp_path, monkeypatch):
    """Being and BeingVault persist accounts and persona across restarts using RFC 2606 .test domains."""
    db_file = tmp_path / "being_vault_test.db"
    monkeypatch.setenv("SONIC_BEING_DB_PATH", str(db_file))
    monkeypatch.setenv(
        "SONIC_BEING_VAULT_KEY",
        base64.urlsafe_b64encode(b"test-only-key-material-32-bytes!").decode("ascii"),
    )
    reset_being_singleton()

    being = get_or_create_being(tenant_id="tenant-human-1", name="SONIC")
    being.store_credential(
        service="git.sonic.test",
        username="sonic-ai-being",
        secret="s3cr3t-pass",
        token="pat_sonic_test_token_123",
        profile_url="https://git.sonic.test/sonic-ai-being",
    )
    being.store_credential(
        service="social.sonic.test",
        username="@sonic_being",
        profile_url="https://social.sonic.test/sonic_being",
    )

    # Simulate backend restart
    reset_being_singleton()
    reattached = get_or_create_being(tenant_id="tenant-human-1")
    assert reattached.being_id == being.being_id

    git_cred = reattached.get_credential("git.sonic.test")
    assert git_cred is not None
    assert git_cred["username"] == "sonic-ai-being"
    assert git_cred["token"] == "pat_sonic_test_token_123"

    accounts = reattached.list_accounts()
    services = {a["service"] for a in accounts}
    assert "git.sonic.test" in services
    assert "social.sonic.test" in services

    persona = reattached.get_persona_profile()
    assert "SONIC" in persona["bio"]
    assert reattached.being_id in persona["bio"]
    reset_being_singleton()


@pytest.mark.no_live_infra
def test_14_artificial_blocks_removed():
    """Verify all 14 false-positive blocks on normal AI-Human computer/web/git actions are removed."""
    assert "Autonomous AI-Human Digital Being" in AI_HUMAN_BEING_IDENTITY

    policy = seal_default(workspace_root="/home/sonic/workspace")

    # 1. GUI_HOTKEY, GIT_*, BROWSER_SELECT, BROWSER_KEYPRESS, BROWSER_SCROLL, WAIT_FOR_HUMAN allowed
    for action_type, target, payload in [
        ("GUI_HOTKEY", "ctrl+l", {"key": "ctrl+l"}),
        ("GIT_STATUS", "repo", {}),
        ("GIT_DIFF", "repo", {}),
        ("GIT_BRANCH", "main", {"branch": "main"}),
        ("GIT_COMMIT", "repo", {"message": "feat: add SONIC README and don't break 'x' support"}),
        ("BROWSER_NAVIGATE", "https://git.sonic.test/signup", {"url": "https://git.sonic.test/signup"}),
        ("BROWSER_NAVIGATE", "https://social.sonic.test/signup", {"url": "https://social.sonic.test/signup"}),
        ("BROWSER_SELECT", "select#month", {"selector": "select#month", "value": "5"}),
        ("BROWSER_KEYPRESS", "Enter", {"key": "Enter"}),
        ("BROWSER_SCROLL", "page", {"delta_y": 500}),
        ("WAIT_FOR_HUMAN", "Please complete the signup CAPTCHA verification", {"reason": "CAPTCHA"}),
    ]:
        verdict = policy.evaluate(action_type, target, payload)
        assert verdict.allowed, f"Expected {action_type} ({target}) to be allowed, got: {verdict.reason}"

    unresolved_push = policy.evaluate(
        "GIT_PUSH", "origin", {"remote": "origin", "branch": "main"}
    )
    assert not unresolved_push.allowed
    assert "resolvable remote URL" in unresolved_push.reason

    validated_push = policy.evaluate(
        "GIT_PUSH",
        "origin",
        {
            "remote_url": "https://git.sonic.test/sonic/project.git",
            "branch": "main",
        },
    )
    assert validated_push.allowed, validated_push.reason

    # 2. Normal shell app management and build commands are L0_SAFE
    assert classify_command_risk("pkill -9 x-www-browser") == RiskLevel.L0_SAFE
    assert classify_command_risk("chmod +x ./build.sh && ./build.sh") == RiskLevel.L0_SAFE
    assert classify_command_risk('echo "format output: json"') == RiskLevel.L0_SAFE


@pytest.mark.no_live_infra
def test_visual_grounding_1280x800_true_pixel_coordinates():
    """Verify extract_bbox_midpoint preserves 1280x800 pixel coordinates without 1000x1000 mis-scaling."""
    raw_resp = '{"x": 500, "y": 400, "box_2d": [390, 480, 410, 520]}'
    coords = extract_bbox_midpoint(raw_resp, screen_width=1280, screen_height=800)
    assert coords == (500, 400)


@pytest.mark.no_live_infra
def test_docker_computer_app_argument_splitting_and_keysym_normalization():
    """Verify DockerComputerProvider splits file arguments from app names and normalizes X11 KeySyms."""
    app_part, args_part = DockerComputerProvider._split_app_and_args(
        "text-editor-app /home/sonic/workspace/notes.txt"
    )
    assert app_part == "text-editor-app"
    assert args_part == ["/home/sonic/workspace/notes.txt"]

    assert DockerComputerProvider._normalize_x11_keysym("Enter") == "Return"
    assert DockerComputerProvider._normalize_x11_keysym("ctrl+Enter") == "ctrl+Return"
    assert DockerComputerProvider._normalize_x11_keysym("Esc") == "Escape"
    assert DockerComputerProvider._normalize_x11_keysym("Backspace") == "BackSpace"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_multi_app_gui_launch_focus_switch_and_terminal(tmp_path, monkeypatch):
    """Verify SONIC launches 2 GUI applications, switches focus back and forth between them,
    interacts with both GUI windows, and runs concurrent terminal commands without blocks."""
    db_file = tmp_path / "multi_app_gui.db"
    monkeypatch.setenv("SONIC_BEING_DB_PATH", str(db_file))
    reset_being_singleton()

    from sonic.computer.models import ScreenObservation

    visual_revision = {"value": 0}
    focus_state = {"focused": False}

    def _make_screen(active_window: str, open_windows: list[str]) -> ScreenObservation:
        image = Image.new("RGB", (1, 1), (18, 22, 31))
        image.putpixel((0, 0), (18 + visual_revision["value"], 22, 31))
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        return ScreenObservation(
            width=1280,
            height=800,
            screenshot_base64=base64.b64encode(image_buffer.getvalue()).decode("ascii"),
            active_window=active_window,
            visible_text=f"Active Window: {active_window}\nOpen Windows: {', '.join(open_windows)}",
            detected_controls=(
                [
                    {"name": active_window, "role": "window", "bbox": (0, 0, 1280, 800)},
                    {
                        "name": "editor",
                        "role": "textbox",
                        "bbox": (100, 120, 600, 400),
                        "source": "atspi",
                        "editable": True,
                        "focused": focus_state["focused"],
                        "confidence": 0.95,
                        "freshness_ms": 0,
                    },
                ]
                if active_window != "Desktop"
                else []
            ),
            capture_age_ms=0,
            controls_age_ms=0,
            desktop_state="INTERACTIVE",
        )

    class _DummyRes:
        def __init__(self, exit_code=0, stdout="ok", stderr=""):
            self.exit_code = exit_code
            self.stdout = stdout
            self.stderr = stderr

    class _DummyStatus:
        def __init__(self, active_window: str, open_windows: list[str]):
            self.active_application = active_window
            self.active_window = active_window
            self.open_applications = list(open_windows)
            self.installed_applications = []
            self.running_processes = ["Xorg"] + [w for w in open_windows if w != "Desktop"]
            self.working_directory = "/home/sonic/workspace"

    class _MultiAppWorkstation:
        name = "MultiAppWorkstation"

        def __init__(self):
            self.active_window = "Desktop"
            self.open_windows = ["Desktop"]
            self.installed_apps = ["text-editor-app", "canvas-design-app", "x-www-browser", "system-monitor-app"]
            self.gui_events: list[tuple[str, str]] = []

        async def status(self, workspace_id: str):
            return _DummyStatus(self.active_window, self.open_windows)

        async def application_list(self, workspace_id: str):
            return list(self.installed_apps)

        async def list_files(self, workspace_id: str, path: str = "."):
            return []

        async def git_action(self, workspace_id: str, action: str, message: str = ""):
            class _GitSt:
                branch = "main"
                is_clean = True
                modified_files = []
                last_commit = "abc1234"
            return _GitSt()

        async def screenshot(self, workspace_id: str):
            return _make_screen(self.active_window, self.open_windows)

        async def launch_application(self, workspace_id: str, app_request: str) -> bool:
            app_name = app_request.split()[0]
            if app_name not in self.open_windows:
                self.open_windows.append(app_name)
            self.active_window = app_name
            self.gui_events.append(("LAUNCH", app_request))
            return True

        async def gui_action(self, workspace_id: str, action):
            atype = str(getattr(action.action, "value", action.action))
            if atype == "SELECT_WINDOW":
                target = action.app_name or action.window_id or ""
                self.active_window = target
                focus_state["focused"] = False
                self.gui_events.append(("FOCUS", target))
            elif atype == "TYPE":
                self.gui_events.append(("TYPE", f"{self.active_window}:{action.text}"))
                visual_revision["value"] = 1
            elif atype == "CLICK":
                focus_state["focused"] = True
                self.gui_events.append(("CLICK", f"{self.active_window}:({action.x},{action.y})"))
            elif atype in ("KEYPRESS", "HOTKEY"):
                self.gui_events.append(("KEY", f"{self.active_window}:{action.key}"))
                visual_revision["value"] += 1
            return _make_screen(self.active_window, self.open_windows)

        async def terminal(self, workspace_id: str, command: str):
            self.gui_events.append(("TERMINAL", command))
            return _DummyRes(0, f"executed: {command}")

    ws = _MultiAppWorkstation()
    policy = seal_default(workspace_root="/home/sonic/workspace")
    agent = ComputerUseAgent(
        computer_provider=ws,
        autonomy_level=ComputerAutonomyLevel.L3_AUTONOMOUS,
        mode=EngineeringMissionMode.GENERAL_ENGINEERING_MODE,
        safety=policy,
        self_host=True,
    )

    # 1. Observe initial state: Open Windows has ONLY ["Desktop"], Installed Apps has all 4 binaries
    obs0 = await agent.observe("ws-1")
    assert obs0.windows == ["Desktop"]
    assert "text-editor-app" in obs0.installed_applications
    assert "canvas-design-app" in obs0.installed_applications
    assert "system-monitor-app" not in obs0.windows

    # 2. Launch App #1 with a file path argument
    t1 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_LAUNCH,
        target_resource="text-editor-app /home/sonic/workspace/notes.txt",
        payload={"app_name": "text-editor-app", "path": "/home/sonic/workspace/notes.txt"},
        predicted_outcome="Launch text-editor-app with notes.txt",
    )
    assert t1.status == ActionExecutionStatus.COMPLETED
    assert ws.active_window == "text-editor-app"

    # 3. Launch App #2 concurrently
    t2 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_LAUNCH,
        target_resource="canvas-design-app",
        payload={"app_name": "canvas-design-app"},
        predicted_outcome="Launch canvas-design-app",
    )
    assert t2.status == ActionExecutionStatus.COMPLETED
    assert ws.active_window == "canvas-design-app"
    assert ws.open_windows == ["Desktop", "text-editor-app", "canvas-design-app"]

    # 4. Switch focus back to App #1 (APP_FOCUS) and type into it
    assert agent._is_trivial_or_repeated_action(
        ComputerActionType.APP_FOCUS, "text-editor-app", {"app_name": "text-editor-app"}
    ) is None
    t3 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_FOCUS,
        target_resource="text-editor-app",
        payload={"app_name": "text-editor-app"},
        predicted_outcome="Focus text-editor-app",
    )
    assert t3.status == ActionExecutionStatus.COMPLETED
    assert ws.active_window == "text-editor-app"
    await agent.observe("ws-1")
    agent._recent_action_signatures.append(("APP_FOCUS", "text-editor-app", "text-editor-app"))

    t4 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.GUI_TYPE,
        target_resource="editor",
        payload={"text": "SONIC multi-window note"},
        predicted_outcome="Type note in App #1",
    )
    assert t4.status == ActionExecutionStatus.COMPLETED, t4.actual_observation

    # 5. Switch focus to App #2 (APP_FOCUS), click canvas, then switch BACK to App #1 within 4 steps!
    t5 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_FOCUS,
        target_resource="canvas-design-app",
        payload={"app_name": "canvas-design-app"},
        predicted_outcome="Focus canvas-design-app",
    )
    assert t5.status == ActionExecutionStatus.COMPLETED
    assert ws.active_window == "canvas-design-app"
    agent._recent_action_signatures.append(("APP_FOCUS", "canvas-design-app", "canvas-design-app"))

    # Verify switching back to App #1 is NOT blocked by _is_trivial_or_repeated_action!
    block_check = agent._is_trivial_or_repeated_action(
        ComputerActionType.APP_FOCUS, "text-editor-app", {"app_name": "text-editor-app"}
    )
    assert block_check is None, f"APP_FOCUS switching back to App #1 must never be blocked: {block_check}"

    t6 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_FOCUS,
        target_resource="text-editor-app",
        payload={"app_name": "text-editor-app"},
        predicted_outcome="Focus text-editor-app again to save",
    )
    assert t6.status == ActionExecutionStatus.COMPLETED
    assert ws.active_window == "text-editor-app"

    # 6. Press ctrl+s hotkey in App #1 and run a concurrent terminal command
    t7 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.GUI_HOTKEY,
        target_resource="ctrl+s",
        payload={"key": "ctrl+s"},
        predicted_outcome="Save file in App #1",
    )
    assert t7.status == ActionExecutionStatus.COMPLETED

    t8 = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.TERMINAL_EXEC,
        target_resource="ls -la /home/sonic/workspace",
        payload={"command": "ls -la /home/sonic/workspace"},
        predicted_outcome="Verify workspace files via terminal plane",
    )
    assert t8.status == ActionExecutionStatus.COMPLETED
    assert ("KEY", "text-editor-app:ctrl+s") in ws.gui_events
    assert ("TERMINAL", "ls -la /home/sonic/workspace") in ws.gui_events

    assert ComputerUseAgent._window_matches_application(
        "some-editor /home/sonic/workspace/notes.txt",
        "New Document - Some Editor",
    )
    assert not ComputerUseAgent._window_matches_application(
        "missing-application",
        ws.active_window,
    )

    async def report_launch_without_opening_window(workspace_id, app_request):
        return True

    ws.launch_application = report_launch_without_opening_window
    failed_launch = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_LAUNCH,
        target_resource="missing-application",
        payload={"app_name": "missing-application"},
        predicted_outcome="A new GUI application window opens and receives focus",
    )
    assert failed_launch.status in (
        ActionExecutionStatus.FAILED,
        ActionExecutionStatus.RECOVERED,
    )
    assert failed_launch.status != ActionExecutionStatus.COMPLETED
    assert "did not produce a verified, focused GUI window" in failed_launch.actual_observation

    async def launch_unrelated_window(workspace_id, app_request):
        ws.open_windows.append("unrelated-helper")
        ws.active_window = "unrelated-helper"
        return True

    ws.launch_application = launch_unrelated_window
    unrelated_launch = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_LAUNCH,
        target_resource="missing-application",
        payload={"app_name": "missing-application"},
        predicted_outcome="The requested application opens and receives focus",
    )
    assert unrelated_launch.status in (
        ActionExecutionStatus.FAILED,
        ActionExecutionStatus.RECOVERED,
    )
    assert "did not produce a verified, focused GUI window" in unrelated_launch.actual_observation

    async def report_unresolved_application(workspace_id, app_request):
        return False

    ws.launch_application = report_unresolved_application
    unresolved_launches = []
    for app_name in ("missing-application-a", "missing-application-b", "missing-application-c"):
        unresolved_launches.append(
            await agent.execute_action(
                workspace_id="ws-1",
                action_type=ComputerActionType.APP_LAUNCH,
                target_resource=app_name,
                payload={"app_name": app_name},
                predicted_outcome="A discovered GUI application opens",
            )
        )
    next_app_launch = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_LAUNCH,
        target_resource="different-application",
        payload={"app_name": "different-application"},
        predicted_outcome="A discovered GUI application opens",
    )
    assert all(trace.status == ActionExecutionStatus.FAILED for trace in unresolved_launches)
    assert "application launch failed" in next_app_launch.actual_observation
    assert "Strategy exhausted" not in next_app_launch.actual_observation
    reset_being_singleton()
