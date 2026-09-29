"""
Unit tests for DockerComputerProvider (Native Workstation Engine).
"""

import base64
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from sonic.computer.daytona_computer import DaytonaComputerProvider
from sonic.computer.docker_computer import DockerComputerProvider, _get_display
from sonic.computer.models import GUIAction, GUIActionType


def _docker_daemon_up() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            ).returncode
            == 0
        )
    except Exception:
        return False


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_computer_provider_lifecycle():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    ws = await provider.create("tenant-1", "eng-1")
    assert ws.id == "sonic-desktop-workstation"
    assert ws.tenant_id == "tenant-1"

    url = await provider.get_vnc_url(ws.id)
    # A real noVNC listener only exists when the daemon + workstation container
    # are actually up. Fail-closed returns None otherwise — assert accordingly.
    if _docker_daemon_up():
        assert url is not None
        assert "6080" in url

    if _docker_daemon_up():
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
@pytest.mark.asyncio
async def test_docker_computer_provider_files_and_apps():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    ws_id = "sonic-desktop-workstation"

    if _docker_daemon_up():
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
async def test_docker_screenshot_reports_source_frame_age_from_nsi_timestamp():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = base64.b64encode(b"pixel-data" * 20).decode()
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": encoded,
        "capture_timestamp_ns": time.time_ns() - 250_000_000,
    })

    observation = await provider.screenshot("ws-1")

    assert observation.capture_timestamp_ns is not None
    assert observation.capture_age_ms is not None
    assert observation.capture_age_ms >= 250


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_keeps_capture_age_unknown_for_invalid_clock_data():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = base64.b64encode(b"pixel-data" * 20).decode()
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": encoded,
        "capture_timestamp_ns": time.time_ns() + 1_000_000_000,
    })

    observation = await provider.screenshot("ws-1")

    assert observation.capture_timestamp_ns is not None
    assert observation.capture_age_ms is None


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_parses_capture_timestamp_from_fallback_capture():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = base64.b64encode(b"pixel-data" * 20).decode()
    capture_timestamp_ns = time.time_ns() - 100_000_000
    provider._nsi_request = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(
        0,
        (
            f"___CAPTURE_TIMESTAMP_NS___\n{capture_timestamp_ns}\n{encoded}"
            "\n___OCR_TSV___\n"
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "___ACTIVE_WINDOW___\nDesktop\n___OPEN_WINDOWS___\n"
        ),
        "",
    ))

    observation = await provider.screenshot("ws-1")

    assert observation.screenshot_base64 == encoded
    assert observation.capture_timestamp_ns == capture_timestamp_ns
    assert observation.capture_age_ms is not None
    assert observation.capture_age_ms >= 100


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_does_not_reuse_stale_pixels_after_capture_failure():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = base64.b64encode(b"pixel-data" * 20).decode()
    provider._docker_exec = AsyncMock(side_effect=[
        (0, encoded, ""),
        (126, "", "display unavailable"),
    ])

    first = await provider.screenshot("ws-1")
    provider._last_screenshot_time = 0.0
    second = await provider.screenshot("ws-1")

    assert first.desktop_state == "INTERACTIVE"
    assert second.desktop_state == "NO_DISPLAY"
    assert second.screenshot_base64 == ""


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

    success = await provider.launch_application("ws-1", "chromium https://target.local")
    assert success is True
    called_cmds = [call.args[0] for call in provider._docker_exec.call_args_list]
    # Mock returns exit 0 for the --no-sandbox probe, so flags are injected
    spawn = next(cmd for cmd in called_cmds if "nohup chromium" in cmd)
    assert "DISPLAY=:99 nohup chromium --no-sandbox --disable-dev-shm-usage https://target.local" in spawn
    assert "kill -0" in spawn
    assert "grep -q -- '--no-sandbox'" in spawn
    assert "retry_args=(" not in spawn

    # Also test an argument with spaces / special characters that requires escaping
    provider._docker_exec.reset_mock()
    success2 = await provider.launch_application("ws-1", "chromium 'https://target.local/path with spaces'")
    assert success2 is True
    called_cmds2 = [call.args[0] for call in provider._docker_exec.call_args_list]
    spawn2 = next(cmd for cmd in called_cmds2 if "nohup chromium" in cmd)
    assert (
        "DISPLAY=:99 nohup chromium --no-sandbox --disable-dev-shm-usage "
        "'https://target.local/path with spaces' >/tmp/sonic_app_launch.log 2>&1 &"
    ) in spawn2

    # Also test multi-word application query (e.g. 'google chrome')
    provider._docker_exec.reset_mock()
    success3 = await provider.launch_application("ws-1", "google chrome https://target.local")
    assert success3 is True
    called_cmds3 = [call.args[0] for call in provider._docker_exec.call_args_list]
    spawn3 = next(cmd for cmd in called_cmds3 if "nohup" in cmd)
    assert "DISPLAY=:99 nohup" in spawn3
    assert "https://target.local" in spawn3
    # Mock returns exit 0 for probe so flags are injected
    assert "--no-sandbox" in spawn3


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_drag_uses_smooth_cursor_helper():
    provider = DockerComputerProvider()
    provider._nsi_request = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider.screenshot = AsyncMock()

    await provider.gui_action(
        "ws-1",
        GUIAction(action=GUIActionType.DRAG, x=10, y=20, x2=300, y2=400),
    )

    command = provider._docker_exec.await_args.args[0]
    assert "sonic_cursor.py drag 10 20 300 400" in command
    assert "xdotool mousemove --sync" not in command


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

    provider.terminal = AsyncMock(return_value=ExecResult(command="", exit_code=0, stdout="", stderr=""))
    success = await provider.launch_application("ws-1", "chromium https://target.local")
    assert success is True
    provider.terminal.assert_called_once()
    called_cmd = provider.terminal.call_args[0][1]
    expected_spawn = "DISPLAY=:1 nohup chromium https://target.local >/dev/null 2>&1 &"
    assert called_cmd == expected_spawn

    # Also test special characters quoting
    provider.terminal.reset_mock()
    success2 = await provider.launch_application("ws-1", "chromium 'https://target.local/search?q=1&v=2'")
    assert success2 is True
    called_cmd2 = provider.terminal.call_args[0][1]
    expected_spawn2 = "DISPLAY=:1 nohup chromium 'https://target.local/search?q=1&v=2' >/dev/null 2>&1 &"
    assert called_cmd2 == expected_spawn2
