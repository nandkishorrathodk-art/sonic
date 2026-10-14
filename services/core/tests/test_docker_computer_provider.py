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
import sys
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


def _seed_raw_frame(
    provider: DockerComputerProvider,
    *,
    frame_id: int = 1,
    width: int = 80,
    height: int = 60,
    timestamp_ns: int | None = None,
    avcs_state: dict | None = None,
) -> bytes:
    raw_rgb = bytes((40, 80, 120)) * (width * height)
    provider._latest_processed_frame = SimpleNamespace(
        frame_id=frame_id,
        capture_timestamp_ns=timestamp_ns or time.time_ns(),
        width=width,
        height=height,
        raw_rgb=raw_rgb,
        avcs_state=avcs_state
        or {
            "engine": "avcs_reflex",
            "frame_index": frame_id,
            "source_frame_id": frame_id,
            "scene_hash": f"avcs-scene-{frame_id}",
            "width": width,
            "height": height,
        },
    )
    provider._start_raw_frame_stream = AsyncMock()
    provider._docker_exec = AsyncMock(return_value=(
        0,
        "___ACTIVE_WINDOW___\nDesktop\n___OPEN_WINDOWS___\n",
        "",
    ))
    return raw_rgb


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
async def test_docker_screenshot_returns_fresh_raw_rgb_without_png_encoding():
    provider = DockerComputerProvider(container_name="avcs-test-workstation")
    raw_rgb = _seed_raw_frame(provider)

    observation = await provider.screenshot("ws-1")

    assert observation.raw_rgb == raw_rgb
    assert observation.frame_id == 1
    assert observation.avcs_state["source_frame_id"] == 1
    assert observation.screenshot_base64 == ""
    assert "raw_rgb" not in observation.model_dump()
    assert observation.capture_age_ms is not None
    assert observation.capture_age_ms <= 2_000
    assert observation.visible_text == ""
    assert observation.detected_controls == []
    assert observation.desktop_state == "INTERACTIVE"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_rejects_frame_that_ages_during_avcs(monkeypatch):
    provider = DockerComputerProvider(container_name="stale-avcs-test-workstation")
    timestamp_ns = time.time_ns()
    _seed_raw_frame(provider, timestamp_ns=timestamp_ns)
    monotonic_wall_clock = iter((timestamp_ns, timestamp_ns + 2_001_000_000))
    monkeypatch.setattr(
        docker_computer_module.time,
        "time_ns",
        lambda: next(monotonic_wall_clock),
    )

    observation = await provider.screenshot("ws-1")

    assert observation.desktop_state == "NO_DISPLAY"
    assert observation.screenshot_base64 == ""
    assert observation.capture_age_ms == pytest.approx(2_001)
    assert provider._last_screenshot is None
    assert provider._last_screenshot_time == 0.0


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_frame_processor_initializes_avcs_for_raw_stream(monkeypatch):
    provider = DockerComputerProvider(container_name="avcs-lazy-init-workstation")
    raw_rgb = bytes((10, 20, 30)) * (80 * 60)
    avcs_state = {
        "engine": "avcs_reflex",
        "frame_index": 1,
        "source_frame_id": 1,
        "scene_hash": "locally-processed-scene",
        "width": 80,
        "height": 60,
    }
    perception = SimpleNamespace(perceive_rgb_bytes=Mock(return_value=avcs_state))
    avcs_factory = Mock(return_value=perception)
    monkeypatch.setitem(
        sys.modules,
        "sonic.computer_use.avcs_perception",
        SimpleNamespace(AVCSPerception=avcs_factory),
    )
    provider._frame_stream_queue.put_nowait(
        (1, time.time_ns(), 80, 60, raw_rgb)
    )
    processor = asyncio.create_task(provider._process_raw_frame_stream())
    await asyncio.wait_for(provider._frame_ready.wait(), timeout=1)
    processor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await processor

    avcs_factory.assert_called_once_with()
    perception.perceive_rgb_bytes.assert_called_once_with(raw_rgb, 80, 60, 1)
    assert provider._latest_processed_frame.raw_rgb == raw_rgb
    assert provider._latest_processed_frame.avcs_state["source_frame_id"] == 1


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_raw_stream_reader_keeps_ids_monotonic_across_worker_restarts():
    provider = DockerComputerProvider(container_name="avcs-sequence-workstation")
    provider._stream_frame_id_offset = 7
    provider._frame_stream_queue = asyncio.Queue(maxsize=1)

    class Process:
        returncode = None

        def __init__(self, stdout):
            self.stdout = stdout

        def terminate(self):
            self.returncode = -15

    def process_for(source_id: int) -> Process:
        stdout = asyncio.StreamReader()
        header = docker_computer_module._RAW_FRAME_HEADER.pack(
            docker_computer_module._RAW_FRAME_MAGIC,
            source_id,
            time.time_ns(),
            2,
            2,
        )
        stdout.feed_data(header + bytes((1, 2, 3)) * 4)
        stdout.feed_eof()
        return Process(stdout)

    await provider._read_raw_frame_stream(process_for(1))

    global_id, *_ = provider._frame_stream_queue.get_nowait()
    assert global_id == 8
    assert provider._last_stream_frame_id == 8

    provider._stream_frame_id_offset = provider._last_stream_frame_id
    provider._last_stream_source_frame_id = 0
    provider._frame_stream_error = None
    provider._frame_stream_queue = asyncio.Queue(maxsize=1)
    await provider._read_raw_frame_stream(process_for(1))

    restarted_global_id, *_ = provider._frame_stream_queue.get_nowait()
    assert restarted_global_id == 9
    assert provider._last_stream_frame_id == 9


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_raw_stream_reader_rejects_invalid_magic_and_records_failure():
    provider = DockerComputerProvider(container_name="avcs-protocol-workstation")
    stdout = asyncio.StreamReader()
    stdout.feed_data(
        docker_computer_module._RAW_FRAME_HEADER.pack(
            b"NOPE", 1, time.time_ns(), 2, 2
        )
    )
    stdout.feed_eof()

    class Process:
        returncode = None

        def __init__(self):
            self.stdout = stdout
            self.terminated = False

        def terminate(self):
            self.terminated = True

    process = Process()
    await provider._read_raw_frame_stream(process)

    assert provider._frame_stream_error == "AVCS stream emitted invalid frame metadata"
    assert provider._latest_processed_frame is None
    assert process.terminated is True


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_raw_frame_stream_uses_open_stdin_until_provider_closes_it(monkeypatch):
    provider = DockerComputerProvider(container_name="avcs-stdin-workstation")
    provider._container_is_running = AsyncMock(return_value=True)
    monkeypatch.setattr(docker_computer_module.shutil, "which", lambda _: "docker")

    class FakeStdin:
        closed = False

        def is_closing(self):
            return self.closed

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.stdin = FakeStdin()
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()

        async def wait(self):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    process = FakeProcess()
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(
        docker_computer_module.asyncio,
        "create_subprocess_exec",
        spawn,
    )

    await provider._start_raw_frame_stream("ws-1")

    assert spawn.await_args.kwargs["stdin"] is asyncio.subprocess.PIPE
    await provider.close()
    assert process.stdin.closed is True


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
    _seed_raw_frame(provider)
    provider._start_raw_frame_stream = AsyncMock(
        side_effect=RuntimeError("display unavailable")
    )

    observation = await provider.screenshot("ws-1")

    assert observation.desktop_state == "NO_DISPLAY"
    assert observation.raw_rgb == b""
    assert provider._latest_processed_frame is None
    assert provider._last_screenshot is None
    assert provider._last_screenshot_time == 0.0


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_frame_processor_invalidates_cached_frame_when_avcs_fails():
    provider = DockerComputerProvider(container_name="avcs-error-workstation")
    provider._latest_processed_frame = SimpleNamespace(frame_id=0)
    provider._avcs_perception = SimpleNamespace(perceive_rgb_bytes=Mock(
        side_effect=RuntimeError("perception unavailable")
    ))
    provider._frame_stream_queue.put_nowait(
        (1, time.time_ns(), 4, 3, bytes((1, 2, 3)) * 12)
    )
    processor = asyncio.create_task(provider._process_raw_frame_stream())
    await asyncio.wait_for(provider._frame_ready.wait(), timeout=1)
    processor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await processor

    assert provider._latest_processed_frame is None
    assert "perception unavailable" in provider._frame_stream_error


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_frame_processor_rejects_invalid_avcs_state():
    provider = DockerComputerProvider(container_name="invalid-avcs-workstation")
    provider._avcs_perception = SimpleNamespace(
        perceive_rgb_bytes=Mock(return_value={
            "engine": "avcs_reflex",
            "frame_index": 1,
            "source_frame_id": 1,
            "scene_hash": "wrong-frame-dimensions",
            "width": 1280,
            "height": 800,
        })
    )
    provider._frame_stream_queue.put_nowait(
        (1, time.time_ns(), 80, 60, bytes((1, 2, 3)) * (80 * 60))
    )
    processor = asyncio.create_task(provider._process_raw_frame_stream())
    await asyncio.wait_for(provider._frame_ready.wait(), timeout=1)
    processor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await processor

    assert provider._latest_processed_frame is None
    assert provider._frame_stream_error == "AVCS returned invalid raw-frame state"


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
async def test_gui_pointer_bounds_accept_fresh_raw_rgb_stream_frame(monkeypatch):
    provider = DockerComputerProvider(container_name="avcs-raw-pointer-workstation")
    raw_rgb = bytes((1, 2, 3)) * (80 * 60)
    observation = ScreenObservation(
        raw_rgb=raw_rgb,
        frame_id=1,
        width=80,
        height=60,
        capture_age_ms=0,
        desktop_state="INTERACTIVE",
    )
    provider.screenshot = AsyncMock(side_effect=(observation, observation))
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(docker_computer_module.asyncio, "sleep", AsyncMock())

    result = await provider.gui_action(
        "ws-1", GUIAction(action=GUIActionType.CLICK, x=79, y=59)
    )

    assert result is observation
    assert provider._docker_exec.await_count == 1
    assert provider.screenshot.await_count == 2


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_destroying_last_workspace_closes_raw_frame_stream():
    provider = DockerComputerProvider()
    provider.workspaces["ws-1"] = SimpleNamespace()
    provider.close = AsyncMock()

    await provider.destroy("ws-1")

    provider.close.assert_awaited_once()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_destroying_workspace_preserves_stream_when_another_workspace_remains():
    provider = DockerComputerProvider()
    provider.workspaces["ws-1"] = SimpleNamespace()
    provider.workspaces["ws-2"] = SimpleNamespace()
    provider.close = AsyncMock()

    await provider.destroy("ws-1")

    provider.close.assert_not_awaited()


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
