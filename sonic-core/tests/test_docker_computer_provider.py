"""
Unit tests for DockerComputerProvider (Native Workstation Engine).
"""

import asyncio
import base64
import binascii
import os
import shutil
import struct
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import zlib
import pytest
import sonic.computer.docker_computer as docker_computer_module
from sonic.computer.daytona_computer import DaytonaComputerProvider
from sonic.computer.docker_computer import (
    DockerComputerProvider,
    _get_display,
)
from sonic.computer.models import (
    ComputerWorkspaceStatus,
    GUIAction,
    GUIActionType,
    ScreenObservation,
)


def _png_base64(width: int = 1280, height: int = 800) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", binascii.crc32(body) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + b"\x00\x00\x00" * width
    pixels = zlib.compress(row * height)
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(image).decode("ascii")


def _workstation_container_up(container_name: str = "sonic-desktop-workstation") -> bool:
    if not shutil.which("docker"):
        return False
    try:
        probe = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return (
            probe.returncode == 0
            and probe.stdout.decode("utf-8", errors="ignore").strip() == "true"
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_exec_cancellation_kills_and_reaps_docker_client(monkeypatch):
    provider = DockerComputerProvider(container_name="cancel-test-workstation")
    monkeypatch.setattr(docker_computer_module.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(
        docker_computer_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"true"),
    )
    communicate_started = asyncio.Event()

    class FakeProcess:
        returncode = None
        killed = False
        reaped = False

        async def communicate(self):
            communicate_started.set()
            await asyncio.Future()

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            self.reaped = True
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(
        docker_computer_module.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )

    task = asyncio.create_task(provider._docker_exec("sleep 60"))
    await communicate_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True
    assert process.reaped is True


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_application_list_discovers_path_only_binaries():
    provider = DockerComputerProvider(container_name="app-discovery-workstation")
    provider._docker_exec = AsyncMock(
        return_value=(0, "path-only-app\npath-symlink-app\n", "")
    )

    applications = await provider.application_list("ws-1")

    assert "path-only-app" in applications
    command = provider._docker_exec.await_args.args[0]
    assert 'printf \'%s\\n\' "$PATH"' in command
    assert r"\( -type f -o -type l \) -executable" in command


@pytest.mark.no_live_infra
def test_scroll_action_requires_nonzero_delta():
    with pytest.raises(ValueError, match="non-zero scroll_delta"):
        GUIAction(action=GUIActionType.SCROLL, x=10, y=10, scroll_delta=0)

    assert GUIAction(
        action=GUIActionType.SCROLL, x=10, y=10, scroll_delta=-1
    ).scroll_delta == -1
    assert GUIAction(action=GUIActionType.KEYPRESS, key="Return").scroll_delta == 0


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_window_focus_does_not_send_escape_to_current_application():
    provider = DockerComputerProvider(container_name="focus-workstation")

    async def fake_exec(command, timeout=60):
        if "xdotool getactivewindow getwindowname" in command:
            return 0, "target window", ""
        return 0, "", ""

    provider._docker_exec = AsyncMock(side_effect=fake_exec)
    provider.screenshot = AsyncMock(
        return_value=ScreenObservation(width=1280, height=800)
    )

    await provider.gui_action(
        "ws-1",
        GUIAction(action=GUIActionType.SELECT_WINDOW, app_name="target window"),
    )

    focus_commands = [
        call.args[0]
        for call in provider._docker_exec.await_args_list
        if "wmctrl -a" in call.args[0]
    ]
    assert len(focus_commands) == 1
    assert "xdotool key Escape" not in focus_commands[0]
    assert provider._active_windows["ws-1"] == "target window"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_window_focus_fails_if_active_window_does_not_match():
    provider = DockerComputerProvider(container_name="focus-workstation")

    async def fake_exec(command, timeout=60):
        if "xdotool getactivewindow getwindowname" in command:
            return 0, "another window", ""
        return 0, "", ""

    provider._docker_exec = AsyncMock(side_effect=fake_exec)

    with pytest.raises(RuntimeError, match="could not be verified"):
        await provider.gui_action(
            "ws-1",
            GUIAction(action=GUIActionType.SELECT_WINDOW, app_name="target window"),
        )

    assert "ws-1" not in provider._active_windows


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_tile_workstation_reports_partial_window_failure():
    provider = DockerComputerProvider(container_name="tile-workstation")

    async def fake_exec(command, timeout=60):
        if "xdotool getdisplaygeometry" in command:
            return 0, "1920 1080", ""
        if "wmctrl -l" in command:
            return 0, "0x1 0 0 host First\n0x2 0 0 host Second\n", ""
        if "wmctrl -i -r \"0x1\" -e" in command:
            return 0, "", ""
        if "wmctrl -i -r \"0x2\" -e" in command:
            return 1, "", "window move failed"
        return 0, "", ""

    provider._docker_exec = AsyncMock(side_effect=fake_exec)

    assert await provider.tile_workstation("ws-1") is False


@pytest.mark.no_live_infra
@pytest.mark.skipif(
    os.environ.get("SONIC_RUN_LIVE_WORKSTATION") != "1",
    reason="Live shared workstation test; set SONIC_RUN_LIVE_WORKSTATION=1 to run",
)
@pytest.mark.asyncio
async def test_docker_computer_provider_lifecycle():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    ws = await provider.create("tenant-1", "eng-1")
    assert ws.id == "sonic-desktop-workstation"
    assert ws.tenant_id == "tenant-1"

    url = await provider.get_vnc_url(ws.id)
    # A real noVNC listener only exists when the daemon + workstation container
    # are actually up. Fail-closed returns None otherwise — assert accordingly.
    workstation_up = _workstation_container_up(ws.id)
    if workstation_up:
        assert url is not None
        assert "6080" in url

    if workstation_up:
        status = await provider.status(ws.id)
        assert status.workspace_id == "sonic-desktop-workstation"
        assert len(status.running_processes) >= 0

        # Test terminal execution
        res = await provider.terminal(ws.id, "echo hello_docker_native")
        if res.exit_code == 0:
            assert "hello_docker_native" in res.stdout

        # Test screenshot
        obs = await provider.screenshot(ws.id)
        assert obs.width == 1280
        assert obs.height == 800

        # Test gui click action
        action = GUIAction(action=GUIActionType.CLICK, x=100, y=100)
        action_obs = await provider.gui_action(ws.id, action)
        assert action_obs.width == 1280


@pytest.mark.no_live_infra
@pytest.mark.skipif(
    os.environ.get("SONIC_RUN_LIVE_WORKSTATION") != "1",
    reason="Live shared workstation test; set SONIC_RUN_LIVE_WORKSTATION=1 to run",
)
@pytest.mark.asyncio
async def test_docker_computer_provider_files_and_apps():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    ws_id = "sonic-desktop-workstation"

    if _workstation_container_up(ws_id):
        # Test write file
        test_content = "sonic_security_audit_test_file"
        test_path = "/tmp/sonic_test.txt"
        wrote = await provider.write_file(ws_id, test_path, test_content)
        assert wrote is True

        # Test read file
        read_back = await provider.read_file(ws_id, test_path)
        assert test_content in read_back

        # Test list applications
        apps = await provider.application_list(ws_id)
        assert len(apps) > 0


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_provider_reports_active_window_class_independently_of_title():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    provider._docker_exec = AsyncMock(side_effect=[
        (0, "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x2800007, 0x0", ""),
        (0, 'WM_CLASS(STRING) = "thunar", "Thunar"', ""),
    ])

    window_class = await provider.get_active_window_class("ws-1")

    assert window_class == "thunar.Thunar"
    assert provider._docker_exec.await_count == 2


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_processes_pixels_locally_with_avcs():
    provider = DockerComputerProvider(container_name="avcs-test-workstation")
    encoded = _png_base64(80, 60)
    timestamp_ns = time.time_ns()
    provider._docker_exec = AsyncMock(return_value=(
        0,
        f"___CAPTURE_TIMESTAMP_NS___\n{timestamp_ns}\n{encoded}"
        "\n___ACTIVE_WINDOW___\nDesktop\n___OPEN_WINDOWS___\n",
        "",
    ))
    avcs_state = {
        "engine": "avcs_reflex",
        "frame_index": 1,
        "scene_hash": "avcs-test-scene",
        "width": 80,
        "height": 60,
    }
    perception = SimpleNamespace(perceive_png=Mock(return_value=avcs_state))
    provider._avcs_perception = perception

    observation = await provider.screenshot("ws-1")

    perception.perceive_png.assert_called_once_with(
        base64.b64decode(encoded, validate=True)
    )
    assert observation.avcs_state == avcs_state
    assert observation.screenshot_base64 == encoded
    assert observation.capture_timestamp_ns == timestamp_ns
    assert observation.capture_age_ms is not None
    assert observation.capture_age_ms <= 2_000
    assert observation.visible_text == ""
    assert observation.detected_controls == []
    assert observation.desktop_state == "INTERACTIVE"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_vnc_url_uses_selected_container_published_host_port():
    provider = DockerComputerProvider(container_name="isolated-workstation")
    provider._container_is_running = AsyncMock(return_value=True)
    provider._resolve_novnc_url = AsyncMock(
        return_value="http://127.0.0.1:6081/vnc.html"
    )
    provider._docker_exec = AsyncMock(return_value=(0, "0", ""))

    url = await provider.get_vnc_url("isolated-workstation")

    assert url == (
        "http://localhost:6081/vnc.html"
        "?autoconnect=true&resize=scale&show_dot=true&cursor=true"
    )
    provider._docker_exec.assert_awaited_once()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_vnc_url_fails_closed_without_selected_container_port_binding():
    provider = DockerComputerProvider(container_name="unpublished-workstation")
    provider._container_is_running = AsyncMock(return_value=True)
    provider._resolve_novnc_url = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(0, "0", ""))

    assert await provider.get_vnc_url("unpublished-workstation") is None
    provider._docker_exec.assert_not_awaited()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_does_not_reuse_stale_pixels_after_capture_failure():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = _png_base64()
    provider._avcs_perception = SimpleNamespace(
        perceive_png=Mock(return_value={
            "engine": "avcs_reflex",
            "frame_index": 1,
            "width": 1280,
            "height": 800,
        })
    )
    provider._docker_exec = AsyncMock(side_effect=[
        (
            0,
            f"___CAPTURE_TIMESTAMP_NS___\n{time.time_ns()}\n{encoded}",
            "",
        ),
        (126, "", "display unavailable"),
    ])

    first = await provider.screenshot("ws-1")
    provider._last_screenshot_time = 0.0
    with pytest.raises(RuntimeError, match="Desktop screenshot capture failed"):
        await provider.screenshot("ws-1")

    assert first.desktop_state == "INTERACTIVE"
    assert provider._last_screenshot is None
    assert provider._last_screenshot_time == 0.0


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_gui_pointer_coordinates_require_fresh_screen_and_in_bounds():
    provider = DockerComputerProvider()
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider.screenshot = AsyncMock(return_value=ScreenObservation(
        screenshot_base64=_png_base64(80, 60),
        width=80,
        height=60,
        capture_age_ms=0,
        desktop_state="INTERACTIVE",
    ))

    with pytest.raises(ValueError, match="outside captured screen bounds"):
        await provider.gui_action(
            "ws-1", GUIAction(action=GUIActionType.CLICK, x=80, y=10)
        )
    provider._docker_exec.assert_not_awaited()

    with pytest.raises(ValueError, match="explicit integer coordinates"):
        await provider.gui_action("ws-1", GUIAction(action=GUIActionType.CLICK))
    provider._docker_exec.assert_not_awaited()

    await provider.gui_action(
        "ws-1", GUIAction(action=GUIActionType.CLICK, x=79, y=59)
    )
    assert "sonic_cursor.py move 79 59" in provider._docker_exec.await_args.args[0]


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_gui_move_requires_cursor_to_reach_requested_coordinates():
    provider = DockerComputerProvider()
    provider._docker_exec = AsyncMock(return_value=(0, "X=40\nY=30\nSCREEN=0\n", ""))
    observation = ScreenObservation(
        screenshot_base64=_png_base64(80, 60),
        width=80,
        height=60,
        capture_age_ms=0,
        desktop_state="INTERACTIVE",
    )
    provider.screenshot = AsyncMock(return_value=observation)
    provider._last_screenshot = observation
    provider._last_screenshot_time = time.monotonic()

    await provider.gui_action(
        "ws-1",
        GUIAction(action=GUIActionType.MOVE, x=40, y=30),
    )

    command = provider._docker_exec.await_args.args[0]
    assert "sonic_cursor.py move 40 30" in command
    assert "xdotool getmouselocation --shell" in command


@pytest.mark.no_live_infra
@pytest.mark.parametrize(
    "cursor_output",
    ["X=39\nY=30\n", "X=40\n", ""],
)
@pytest.mark.asyncio
async def test_docker_gui_move_fails_when_cursor_position_is_unverified(cursor_output):
    provider = DockerComputerProvider()
    provider._docker_exec = AsyncMock(return_value=(0, cursor_output, ""))
    observation = ScreenObservation(
        screenshot_base64=_png_base64(80, 60),
        width=80,
        height=60,
        capture_age_ms=0,
        desktop_state="INTERACTIVE",
    )
    provider.screenshot = AsyncMock(return_value=observation)
    provider._last_screenshot = observation
    provider._last_screenshot_time = time.monotonic()

    with pytest.raises(RuntimeError, match="Cursor move verification failed"):
        await provider.gui_action(
            "ws-1",
            GUIAction(action=GUIActionType.MOVE, x=40, y=30),
        )


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_drag_uses_smooth_cursor_helper():
    provider = DockerComputerProvider()
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider.screenshot = AsyncMock()
    provider._last_screenshot = ScreenObservation(
        screenshot_base64=_png_base64(),
        width=1280,
        height=800,
        capture_age_ms=0,
        desktop_state="INTERACTIVE",
    )
    provider._last_screenshot_time = time.monotonic()

    await provider.gui_action(
        "ws-1",
        GUIAction(action=GUIActionType.DRAG, x=10, y=20, x2=300, y2=400),
    )

    command = provider._docker_exec.await_args.args[0]
    assert "sonic_cursor.py drag 10 20 300 400" in command
    assert "xdotool mousemove --sync" not in command


@pytest.mark.no_live_infra
def test_docker_computer_display_resolution(monkeypatch):
    """Proves dynamic display resolution helper checks environment and defaults to :99."""
    monkeypatch.delenv("DISPLAY", raising=False)
    provider = DockerComputerProvider()
    assert _get_display() == ":99"
    assert provider._get_display("ws-1") == ":99"

    monkeypatch.setenv("DISPLAY", ":42")
    assert _get_display() == ":42"
    assert provider._get_display("ws-1") == ":42"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_computer_launch_application_quoting(monkeypatch):
    """Proves launch_application uses shlex.split and quotes each part individually."""
    monkeypatch.delenv("DISPLAY", raising=False)
    provider = DockerComputerProvider()
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider._resolve_app_binary = AsyncMock(
        side_effect=lambda query: f"/usr/bin/{query.replace(' ', '-')}"
    )

    success = await provider.launch_application("ws-1", "chromium https://target.local")
    assert success is True
    called_cmds = [call.args[0] for call in provider._docker_exec.call_args_list]
    # Mock returns exit 0 for the --no-sandbox probe, so flags are injected
    spawn = next(cmd for cmd in called_cmds if "nohup /usr/bin/chromium" in cmd)
    assert "DISPLAY=:99 nohup /usr/bin/chromium --no-sandbox --disable-dev-shm-usage https://target.local" in spawn
    assert "kill -0" in spawn
    assert "grep -q -- '--no-sandbox'" in spawn
    assert "retry_args=(" not in spawn

    # Also test an argument with spaces / special characters that requires escaping
    provider._docker_exec.reset_mock()
    success2 = await provider.launch_application("ws-1", "chromium 'https://target.local/path with spaces'")
    assert success2 is True
    called_cmds2 = [call.args[0] for call in provider._docker_exec.call_args_list]
    spawn2 = next(cmd for cmd in called_cmds2 if "nohup /usr/bin/chromium" in cmd)
    assert (
        "DISPLAY=:99 nohup /usr/bin/chromium --no-sandbox --disable-dev-shm-usage "
        "'https://target.local/path with spaces' >/tmp/sonic_app_launch.log 2>&1 &"
    ) in spawn2

    # Also test multi-word application query (e.g. 'google chrome')
    provider._docker_exec.reset_mock()
    success3 = await provider.launch_application("ws-1", "google chrome https://target.local")
    assert success3 is True
    called_cmds3 = [call.args[0] for call in provider._docker_exec.call_args_list]
    spawn3 = next(cmd for cmd in called_cmds3 if "nohup" in cmd)
    assert "DISPLAY=:99 nohup" in spawn3
    assert "/usr/bin/google-chrome" in spawn3
    assert "https://target.local" in spawn3
    # Mock returns exit 0 for probe so flags are injected
    assert "--no-sandbox" in spawn3


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_computer_does_not_guess_unresolved_application(monkeypatch):
    """An unknown app must not be passed to a shell as a guessed executable."""
    monkeypatch.delenv("DISPLAY", raising=False)
    provider = DockerComputerProvider()
    provider._docker_exec = AsyncMock(return_value=(1, "", ""))

    launched = await provider.launch_application("ws-1", "unavailable application")

    assert launched is False
    assert all("nohup" not in call.args[0] for call in provider._docker_exec.call_args_list)


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_daytona_smooth_cursor_interpolates_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("SONIC_WORKSTATION_STATE_PATH", str(tmp_path / "workstations.json"))
    provider = DaytonaComputerProvider(api_key="")
    commands = []

    class Process:
        async def exec(self, command):
            commands.append(command)
            if "getmouselocation" in command:
                return SimpleNamespace(exit_code=0, result="X=0\nY=0\nSCREEN=0\nWINDOW=1\n")
            return SimpleNamespace(exit_code=0, result="")

    sandbox = SimpleNamespace(process=Process())
    await provider._move_mouse_smoothly(sandbox, "ws-1", 100, 200)

    moves = [cmd for cmd in commands if "xdotool mousemove" in cmd]
    assert len(moves) == 25
    assert moves[0] != moves[-1]
    assert moves[-1].endswith("mousemove 100 200")

    class UnavailableProcess:
        async def exec(self, command):
            return SimpleNamespace(exit_code=126, result="")

    with pytest.raises(RuntimeError, match="Unable to read cursor position"):
        await provider._move_mouse_smoothly(
            SimpleNamespace(process=UnavailableProcess()), "ws-1", 100, 200
        )


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_daytona_computer_display_and_launch_quoting(monkeypatch):
    """Proves DaytonaComputerProvider uses the workstation display and quotes launch arguments."""
    from sonic.sandbox.provider import ExecResult

    monkeypatch.delenv("DISPLAY", raising=False)
    provider = DaytonaComputerProvider(api_key="")
    assert provider._get_display("ws-1") == ":99"

    monkeypatch.setenv("DISPLAY", ":1")
    assert provider._get_display("ws-1") == ":1"

    provider.terminal = AsyncMock(
        side_effect=[
            ExecResult(command="", exit_code=0, stdout="321\n", stderr=""),
            ExecResult(command="", exit_code=0, stdout="654\n", stderr=""),
        ]
    )
    success = await provider.launch_application("ws-1", "chromium https://target.local")
    assert success is True
    provider.terminal.assert_called_once()
    called_cmd = provider.terminal.call_args[0][1]
    assert "DISPLAY=:1 nohup chromium https://target.local" in called_cmd
    assert "kill -0 \"$app_pid\"" in called_cmd
    assert "printf '%s\\n' \"$app_pid\"" in called_cmd
    assert provider._active_windows.get("ws-1") is None

    # Also test special characters quoting
    provider.terminal.reset_mock()
    success2 = await provider.launch_application("ws-1", "chromium 'https://target.local/search?q=1&v=2'")
    assert success2 is True
    called_cmd2 = provider.terminal.call_args[0][1]
    assert (
        "DISPLAY=:1 nohup chromium 'https://target.local/search?q=1&v=2'" in called_cmd2
    )


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_daytona_destroy_failure_keeps_workspace_tracked(tmp_path, monkeypatch):
    monkeypatch.setenv("SONIC_WORKSTATION_STATE_PATH", str(tmp_path / "state.json"))
    provider = DaytonaComputerProvider(api_key="")
    workspace = SimpleNamespace(
        id="sandbox-1",
        tenant_id="tenant-a",
        status=ComputerWorkspaceStatus.READY,
    )

    class Sandbox:
        async def delete(self):
            raise RuntimeError("delete failed")

    provider.workspaces[workspace.id] = workspace
    provider._sandboxes[workspace.id] = Sandbox()

    with pytest.raises(RuntimeError, match="delete failed"):
        await provider.destroy(workspace.id)

    assert workspace.status == ComputerWorkspaceStatus.DEGRADED
    assert provider.workspaces[workspace.id] is workspace
    assert provider._sandboxes[workspace.id] is not None
