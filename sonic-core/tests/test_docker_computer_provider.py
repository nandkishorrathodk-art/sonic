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
from sonic.computer.docker_computer import (
    DockerComputerProvider,
    _get_display,
    _has_actionable_nsi_controls,
)
from sonic.computer.models import GUIAction, GUIActionType


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
async def test_docker_screenshot_reports_source_frame_age_from_nsi_timestamp():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = base64.b64encode(b"pixel-data" * 20).decode()
    controls = [
        {
            "name": source,
            "role": "button",
            "bbox": [100, 200, 180, 230],
            "bbox_format": "xyxy",
            "source": source,
        }
        for source in ("ocr", "atspi", "x11")
    ]
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": encoded,
        "capture_timestamp_ns": time.time_ns() - 250_000_000,
        "detected_controls": controls,
    })

    observation = await provider.screenshot("ws-1")

    assert observation.capture_timestamp_ns is not None
    assert observation.capture_age_ms is not None
    assert observation.capture_age_ms >= 250
    assert observation.detected_controls == controls


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_keeps_capture_age_unknown_for_invalid_clock_data():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = base64.b64encode(b"pixel-data" * 20).decode()
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": encoded,
        "capture_timestamp_ns": time.time_ns() + 1_000_000_000,
        "detected_controls": [
            {
                "name": "Submit",
                "role": "button",
                "bbox": (10, 10, 40, 30),
                "source": "atspi",
            }
        ],
    })

    observation = await provider.screenshot("ws-1")

    assert observation.capture_timestamp_ns is not None
    assert observation.capture_age_ms is None


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_falls_back_to_ocr_when_nsi_has_no_controls():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    nsi_frame = base64.b64encode(b"nsi-frame").decode()
    fallback_frame = base64.b64encode(b"fallback-frame").decode()
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": nsi_frame,
        "active_window": "workspace",
        "visible_text": "Active Window: workspace",
        "open_windows": ["workspace"],
        "detected_controls": [
            {
                "name": "workspace",
                "role": "window",
                "bbox": [0, 30, 650, 540],
                "bbox_format": "xyxy",
                "source": "x11",
            },
            {
                "name": "Na)",
                "role": "text",
                "bbox": [184, 167, 231, 180],
                "bbox_format": "xyxy",
                "source": "ocr",
                "confidence": 31.0,
            },
        ],
    })
    provider._docker_exec = AsyncMock(return_value=(
        0,
        (
            f"___CAPTURE_TIMESTAMP_NS___\n{time.time_ns()}\n{fallback_frame}"
            "\n___OCR_TSV___\n"
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t100\t200\t40\t18\t95.0\tFile\n"
            "___ACTIVE_WINDOW___\nworkspace\n___OPEN_WINDOWS___\n"
            "0x01 0 0 0 workspace\n"
        ),
        "",
    ))

    observation = await provider.screenshot("ws-1")

    assert observation.screenshot_base64 == fallback_frame
    assert observation.active_window == "workspace"
    assert "Screen OCR Text: File" in observation.visible_text
    assert observation.detected_controls == [
        {
            "name": "workspace",
            "role": "window",
            "bbox": [0, 30, 650, 540],
            "bbox_format": "xyxy",
            "source": "x11",
        },
        {
            "name": "File",
            "role": "text",
            "bbox": (100, 200, 140, 218),
            "bbox_format": "xyxy",
            "source": "ocr",
            "confidence": 95.0,
        }
    ]
    provider._docker_exec.assert_awaited_once()


@pytest.mark.no_live_infra
def test_nsi_actionable_controls_require_reliable_ocr_or_atspi():
    assert not _has_actionable_nsi_controls([
        {
            "name": "Na)",
            "role": "text",
            "bbox": [184, 167, 231, 180],
            "source": "ocr",
            "confidence": 31.0,
        }
    ])
    assert not _has_actionable_nsi_controls([
        {
            "name": "workspace",
            "role": "window",
            "bbox": [0, 30, 650, 540],
            "source": "x11",
        }
    ])
    assert _has_actionable_nsi_controls([
        {
            "name": "Submit",
            "role": "button",
            "bbox": [10, 10, 40, 30],
            "source": "atspi",
        }
    ])
    assert _has_actionable_nsi_controls([
        {
            "name": "File",
            "role": "text",
            "bbox": [10, 10, 40, 30],
            "source": "ocr",
            "confidence": 82.0,
        }
    ])


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_provider_resolves_nsi_port_for_its_own_container(monkeypatch):
    provider = DockerComputerProvider(
        container_name="sonic-desktop-workstation-ci-smoke-20260929"
    )
    inspect_result = SimpleNamespace(
        returncode=0,
        stdout=b'{"6080/tcp":[{"HostIp":"127.0.0.1","HostPort":"6081"}]}',
        stderr=b"",
    )
    to_thread = AsyncMock(return_value=inspect_result)
    monkeypatch.setattr("sonic.computer.docker_computer.asyncio.to_thread", to_thread)

    endpoint = await provider._resolve_nsi_frame_url()

    assert endpoint == "http://127.0.0.1:6081/nsi_frame.json"
    command = to_thread.await_args.args[1]
    assert command[-1] == "sonic-desktop-workstation-ci-smoke-20260929"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_provider_does_not_use_an_unrelated_nsi_port_when_unpublished(monkeypatch):
    provider = DockerComputerProvider(container_name="unpublished-workstation")
    inspect_result = SimpleNamespace(
        returncode=0,
        stdout=b'{"6080/tcp":null}',
        stderr=b"",
    )
    monkeypatch.setattr(
        "sonic.computer.docker_computer.asyncio.to_thread",
        AsyncMock(return_value=inspect_result),
    )

    endpoint = await provider._resolve_nsi_frame_url()

    assert endpoint is None


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_nsi_daemon_boot_does_not_kill_its_own_shell():
    class TestProvider(DockerComputerProvider):
        boot_command = ""

        async def _docker_exec(self, cmd, timeout=60):
            self.boot_command = cmd
            return 0, "", ""

    provider = TestProvider(container_name="isolated-workstation")
    endpoint = "http://127.0.0.1:6092/nsi_frame.json"
    frame = {
        "ok": True,
        "b64": base64.b64encode(b"fresh-frame").decode(),
        "detected_controls": [],
    }

    class NsiHttp:
        calls = 0

        async def get(self, _url):
            self.calls += 1
            if self.calls == 1:
                raise OSError("NSI endpoint is not up yet")
            return SimpleNamespace(status_code=200, json=lambda: frame)

    provider._nsi_http = NsiHttp()
    provider._nsi_booted = False
    provider._resolve_nsi_frame_url = AsyncMock(return_value=endpoint)

    result = await provider._nsi_request({"cmd": "frame"})

    assert result == frame
    boot_command = provider.boot_command
    assert "nohup python3 /tmp/sonic_nsi_daemon.py" in boot_command
    assert "pkill -f" not in boot_command
    assert "/tmp/sonic_nsi_daemon.pid" in boot_command


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
            "5\t1\t1\t1\t1\t1\t100\t200\t80\t30\t90\tSubmit\n"
            "___ACTIVE_WINDOW___\nDesktop\n___OPEN_WINDOWS___\n"
        ),
        "",
    ))

    observation = await provider.screenshot("ws-1")

    assert observation.screenshot_base64 == encoded
    assert observation.capture_timestamp_ns == capture_timestamp_ns
    assert observation.capture_age_ms is not None
    assert observation.capture_age_ms >= 100
    assert observation.detected_controls == [
        {
            "name": "Submit",
            "role": "text",
            "bbox": (100, 200, 180, 230),
            "bbox_format": "xyxy",
            "source": "ocr",
            "confidence": 90.0,
        }
    ]


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_merges_tesseract_layout_passes_without_duplicate_controls():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = base64.b64encode(b"pixel-data" * 20).decode()
    tsv_header = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
        "width\theight\tconf\ttext\n"
    )
    provider._nsi_request = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(
        0,
        (
            f"___CAPTURE_TIMESTAMP_NS___\n{time.time_ns()}\n{encoded}"
            "\n___OCR_TSV___\n"
            f"{tsv_header}"
            "5\t1\t1\t1\t1\t1\t100\t200\t80\t30\t62\tWorkspace\n"
            f"{tsv_header}"
            "5\t1\t1\t1\t1\t1\t102\t201\t78\t29\t91\tWorkspace\n"
            "___OCR_EXIT___\n0,0\n"
            "___ACTIVE_WINDOW___\nFile Manager\n___OPEN_WINDOWS___\n"
        ),
        "",
    ))

    observation = await provider.screenshot("ws-1")

    assert observation.detected_controls == [
        {
            "name": "Workspace",
            "role": "text",
            "bbox": (102, 201, 180, 230),
            "bbox_format": "xyxy",
            "source": "ocr",
            "confidence": 91.0,
        }
    ]
    assert "Screen OCR Text: Workspace" in observation.visible_text


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
