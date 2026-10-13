"""Legacy action aliases must operate the shared visible desktop."""

from __future__ import annotations

from sonic.computer.models import (
    ComputerState,
    FileEntry,
    GitStatusInfo,
    GUIAction,
    GUIActionType,
    ScreenObservation,
)
from sonic.computer.provider import ComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
from tests.avcs_test_helpers import png_base64


class _VisibleDesktop(ComputerProvider):
    def __init__(self):
        self.actions: list[GUIAction] = []
        self.capture_count = 0
        self.focused = False
        self.screen_color = "white"

    def _screen_image(self):
        return png_base64(color=self.screen_color)

    async def create(self, tenant_id, engagement_id, **kw):
        return type("Workspace", (), {"id": "ws-desktop"})()

    async def destroy(self, workspace_id):
        return True

    async def status(self, workspace_id):
        return ComputerState(
            workspace_id=workspace_id,
            tenant_id="tenant",
            active_application="Generic Desktop Application",
            active_window="Generic Desktop Application",
            open_applications=["Generic Desktop Application"],
            running_processes=[],
        )

    async def screenshot(self, workspace_id):
        self.capture_count += 1
        return ScreenObservation(
            screenshot_base64=self._screen_image(),
            width=1280,
            height=800,
            desktop_state="INTERACTIVE",
            active_window="Generic Desktop Application",
            visible_text="Application content",
            capture_age_ms=0,
        )

    async def gui_action(self, workspace_id, action, **kwargs):
        if isinstance(action, GUIAction):
            self.actions.append(action)
            if action.action == GUIActionType.CLICK:
                self.focused = True
            elif action.action in {
                GUIActionType.TYPE,
                GUIActionType.KEYPRESS,
                GUIActionType.SCROLL,
            }:
                self.screen_color = "black"
        return await self.screenshot(workspace_id)

    async def terminal(self, workspace_id, command, timeout=60, actor="operator"):
        raise AssertionError("GUI navigation must not use the terminal")

    async def read_file(self, workspace_id, path):
        return ""

    async def write_file(self, workspace_id, path, content, actor="operator"):
        return True

    async def list_files(self, workspace_id, path="."):
        return [FileEntry(name="document.txt", path="document.txt")]

    async def git_action(self, workspace_id, action, **kw):
        return GitStatusInfo()

    async def process_list(self, *args, **kwargs):
        return []

    async def application_list(self, *args, **kwargs):
        return []

    async def launch_application(self, *args, **kwargs):
        return True

    async def close_application(self, *args, **kwargs):
        return True

    async def install_application(self, *args, **kwargs):
        return True

    async def uninstall_application(self, *args, **kwargs):
        return True

    async def service_action(self, *args, **kwargs):
        return True

    async def snapshot(self, *args, **kwargs):
        return {}

    async def restore_snapshot(self, *args, **kwargs):
        return True


async def test_navigation_alias_uses_live_shared_desktop():
    computer = _VisibleDesktop()
    agent = ComputerUseAgent(computer_provider=computer)

    trace = await agent.execute_action(
        "ws-desktop",
        ComputerActionType.BROWSER_NAVIGATE,
        "https://operator-provided.invalid",
        {"url": "https://operator-provided.invalid"},
        "Navigate in the visible application",
    )

    assert trace.status == ActionExecutionStatus.BLOCKED
    assert "focus-verification" in trace.actual_observation.lower()
    assert computer.actions == []
    assert agent._last_gui_navigation_target == ""


async def test_legacy_browser_screenshot_captures_desktop_not_hidden_page():
    computer = _VisibleDesktop()
    agent = ComputerUseAgent(computer_provider=computer)

    trace = await agent.execute_action(
        "ws-desktop",
        ComputerActionType.BROWSER_SCREENSHOT,
        "current application",
        {},
        "Inspect the current visible application",
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert computer.capture_count == 1
    assert agent._last_screenshot_b64 == computer._screen_image()


async def test_legacy_keyboard_alias_dispatches_and_unresolved_scroll_fails_closed():
    computer = _VisibleDesktop()
    agent = ComputerUseAgent(computer_provider=computer)
    await agent.observe("ws-desktop")

    key_trace = await agent.execute_action(
        "ws-desktop",
        ComputerActionType.BROWSER_KEYPRESS,
        "Return",
        {"key": "Return"},
        "Press the focused application's confirmation key",
    )
    scroll_trace = await agent.execute_action(
        "ws-desktop",
        ComputerActionType.BROWSER_SCROLL,
        "visible content",
        {"delta_y": 500, "x": 500, "y": 400},
        "Scroll the focused application",
    )

    assert key_trace.status == ActionExecutionStatus.UNVERIFIED
    assert scroll_trace.status == ActionExecutionStatus.UNVERIFIED
    assert [action.action.value for action in computer.actions] == ["KEYPRESS", "SCROLL"]
    unresolved_scroll = await agent.execute_action(
        "ws-desktop",
        ComputerActionType.BROWSER_SCROLL,
        "",
        {"delta_y": 500},
        "Scroll the focused application",
    )
    assert unresolved_scroll.status == ActionExecutionStatus.BLOCKED
    assert [action.action.value for action in computer.actions] == ["KEYPRESS", "SCROLL"]


async def test_wait_refreshes_gui_but_does_not_claim_selector_verified():
    computer = _VisibleDesktop()
    agent = ComputerUseAgent(computer_provider=computer)

    trace = await agent.execute_action(
        "ws-desktop",
        ComputerActionType.BROWSER_WAIT,
        "Continue button",
        {"selector": "Continue button", "seconds": 0},
        "Wait for the desktop to update",
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert "Waited 0s; screen refreshed" in trace.actual_observation
    assert computer.capture_count == 1


async def test_download_alias_fails_honestly_without_gui_download_verification():
    computer = _VisibleDesktop()
    agent = ComputerUseAgent(computer_provider=computer)

    trace = await agent.execute_action(
        "ws-desktop",
        ComputerActionType.BROWSER_DOWNLOAD,
        "Download button",
        {"selector": "Download button", "save_path": "application.zip"},
        "Save the downloaded file",
    )

    assert trace.status == ActionExecutionStatus.FAILED
    assert "visible GUI actions" in trace.actual_observation
    assert computer.actions == []
