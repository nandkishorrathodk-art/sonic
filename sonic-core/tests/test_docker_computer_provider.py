"""
Unit tests for DockerComputerProvider (Native Workstation Engine).
"""

import ast
import base64
import binascii
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import zlib
import pytest
import sonic.computer.docker_computer as docker_computer_module
from sonic.computer.daytona_computer import DaytonaComputerProvider
from sonic.computer.docker_computer import (
    DockerComputerProvider,
    _get_display,
    _has_actionable_nsi_controls,
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


@pytest.mark.no_live_infra
def test_workstation_entrypoint_allows_missing_optional_ocr():
    docker_dir = Path(__file__).resolve().parents[2] / "docker"
    entrypoint = (docker_dir / "entrypoint-workstation.sh").read_text(
        encoding="utf-8"
    )
    dockerfile = (docker_dir / "Dockerfile.workstation").read_text(
        encoding="utf-8"
    )
    preflight = entrypoint.split("# Remove any stale X locks", maxsplit=1)[0]

    assert "tesseract-ocr is unavailable" in preflight
    assert "exit 1" not in preflight
    assert "python3-gi" in dockerfile
    assert "gir1.2-atspi-2.0" in dockerfile
    assert "at-spi2-core" in dockerfile


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
async def test_vnc_url_uses_selected_container_published_host_port():
    provider = DockerComputerProvider(container_name="isolated-workstation")
    provider._container_is_running = AsyncMock(return_value=True)
    provider._resolve_nsi_frame_url = AsyncMock(
        return_value="http://127.0.0.1:6081/nsi_frame.json"
    )
    provider._docker_exec = AsyncMock(return_value=(0, "0", ""))

    url = await provider.get_vnc_url("isolated-workstation")

    assert url == "http://localhost:6081/vnc.html"
    provider._docker_exec.assert_awaited_once()
    assert "/6080" in provider._docker_exec.await_args.args[0]


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_vnc_url_fails_closed_without_selected_container_port_binding():
    provider = DockerComputerProvider(container_name="unpublished-workstation")
    provider._container_is_running = AsyncMock(return_value=True)
    provider._resolve_nsi_frame_url = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(0, "0", ""))

    assert await provider.get_vnc_url("unpublished-workstation") is None
    provider._docker_exec.assert_not_awaited()


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
async def test_docker_screenshot_reports_source_frame_age_from_nsi_timestamp():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = _png_base64()
    controls = [
        {
            "name": source,
            "role": "button",
            "bbox": [100, 200, 180, 230],
            "bbox_format": "xyxy",
            "source": source,
            "freshness_ms": 0,
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
async def test_docker_observe_state_returns_compact_fresh_nsi_state():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    capture_timestamp_ns = time.time_ns() - 30_000_000
    controls_timestamp_ns = time.time_ns() - 20_000_000
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "version": 42,
        "capture_timestamp_ns": capture_timestamp_ns,
        "controls_timestamp_ns": controls_timestamp_ns,
        "width": 1280,
        "height": 800,
        "active_window": "Document",
        "visible_text": "Document - Untitled",
        "detected_controls": [
            {
                "name": "Save",
                "role": "button",
                "bbox": [30, 40, 90, 70],
                "bbox_format": "xyxy",
                "source": "atspi",
            }
        ],
        "changed_region": [10, 20, 120, 90],
    })

    observation = await provider.observe_state("ws-1")

    assert observation is not None
    assert observation.screenshot_base64 == ""
    assert observation.desktop_state == "STATE_ONLY"
    assert observation.perception_version == 42
    assert observation.width == 1280
    assert observation.height == 800
    assert observation.active_window == "Document"
    assert observation.controls_age_ms is not None
    assert observation.controls_age_ms < 250
    assert observation.changed_region == (10, 20, 120, 90)
    assert provider._nsi_request.await_args.args[0] == {"cmd": "state"}


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_provider_resolves_state_feed_on_selected_workstation_port():
    provider = DockerComputerProvider(container_name="isolated-workstation")
    provider._resolve_nsi_frame_url = AsyncMock(
        return_value="http://127.0.0.1:6092/nsi_frame.json"
    )

    assert await provider._resolve_nsi_state_url() == (
        "http://127.0.0.1:6092/nsi_state.json"
    )
    assert "nsi_state.json" in provider._NSI_DAEMON_SCRIPT
    assert 'if key != "b64"' in provider._NSI_DAEMON_SCRIPT
    compile(provider._NSI_DAEMON_SCRIPT, "<embedded-nsi-daemon>", "exec")


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_observe_state_rejects_stale_or_invalid_nsi_state():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "version": 42,
        "capture_timestamp_ns": time.time_ns() - 50_000_000,
        "controls_timestamp_ns": time.time_ns() - 3_000_000_000,
        "width": 1280,
        "height": 800,
        "active_window": "Document",
        "detected_controls": [],
    })

    assert await provider.observe_state("ws-1") is None

    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "version": True,
        "capture_timestamp_ns": time.time_ns(),
        "controls_timestamp_ns": time.time_ns(),
        "width": 1280,
        "height": 800,
        "active_window": "Document",
    })
    assert await provider.observe_state("ws-1") is None


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_pointer_action_uses_fresh_state_bounds_without_pre_screenshot():
    encoded = _png_base64()

    class StateFeedProvider(DockerComputerProvider):
        frame_reads = 0

        async def _nsi_request(self, payload, timeout=2.5):
            if payload["cmd"] == "state":
                now_ns = time.time_ns()
                return {
                    "ok": True,
                    "version": 7,
                    "capture_timestamp_ns": now_ns,
                    "controls_timestamp_ns": now_ns,
                    "width": 1280,
                    "height": 800,
                    "active_window": "Document",
                    "detected_controls": [],
                }
            if payload["cmd"] == "frame":
                self.frame_reads += 1
                return {
                    "ok": True,
                    "version": 8,
                    "b64": encoded,
                    "capture_timestamp_ns": time.time_ns(),
                    "width": 1280,
                    "height": 800,
                    "active_window": "Document",
                    "detected_controls": [
                        {
                            "name": "Save",
                            "role": "button",
                            "bbox": [30, 40, 90, 70],
                            "bbox_format": "xyxy",
                            "source": "atspi",
                        }
                    ],
                }
            return None

        async def _docker_exec(self, cmd, timeout=60):
            return 0, "", ""

    provider = StateFeedProvider(container_name="state-feed-test")
    await provider.observe_state("ws-1")

    await provider.gui_action(
        "ws-1",
        GUIAction(action=GUIActionType.CLICK, x=200, y=200),
    )

    assert provider.frame_reads == 1


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_gui_action_state_uses_new_nsi_state_without_frame_capture():
    provider = DockerComputerProvider(container_name="state-receipt-test")
    versions = iter((10, 11))
    frame_reads = 0

    async def nsi_request(payload, timeout=2.5):
        nonlocal frame_reads
        if payload["cmd"] == "state":
            version = next(versions)
            now_ns = time.time_ns()
            return {
                "ok": True,
                "version": version,
                "capture_timestamp_ns": now_ns,
                "controls_timestamp_ns": now_ns,
                "width": 1280,
                "height": 800,
                "active_window": "Document",
                "visible_text": "Document" if version == 10 else "Document - Saved",
                "detected_controls": [],
            }
        if payload["cmd"] == "x11":
            return {"exit_code": 0, "stderr": ""}
        if payload["cmd"] == "frame":
            frame_reads += 1
        return None

    provider._nsi_request = AsyncMock(side_effect=nsi_request)
    await provider.observe_state("ws-1")

    observation = await provider.gui_action_state(
        "ws-1",
        GUIAction(action=GUIActionType.KEYPRESS, key="Return"),
    )

    assert observation.desktop_state == "STATE_ONLY"
    assert observation.perception_version == 11
    assert observation.screenshot_base64 == ""
    assert frame_reads == 0


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_gui_action_state_falls_back_when_nsi_version_did_not_advance():
    provider = DockerComputerProvider(container_name="stale-state-receipt-test")
    now_ns = time.time_ns()
    provider._nsi_request = AsyncMock(side_effect=[
        {
            "ok": True,
            "version": 10,
            "capture_timestamp_ns": now_ns,
            "controls_timestamp_ns": now_ns,
            "width": 1280,
            "height": 800,
            "active_window": "Document",
            "visible_text": "Document",
            "detected_controls": [],
        },
        {"exit_code": 0, "stderr": ""},
        {
            "ok": True,
            "version": 10,
            "capture_timestamp_ns": time.time_ns(),
            "controls_timestamp_ns": time.time_ns(),
            "width": 1280,
            "height": 800,
            "active_window": "Document",
            "visible_text": "Document",
            "detected_controls": [],
        },
    ])
    fallback_frame = ScreenObservation(
        screenshot_base64=_png_base64(),
        width=1280,
        height=800,
        active_window="Document",
    )
    provider.screenshot = AsyncMock(return_value=fallback_frame)
    await provider.observe_state("ws-1")

    observation = await provider.gui_action_state(
        "ws-1",
        GUIAction(action=GUIActionType.KEYPRESS, key="Return"),
    )

    assert observation is fallback_frame
    provider.screenshot.assert_awaited_once_with("ws-1")


@pytest.mark.no_live_infra
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_timestamp_ns",
    [None, time.time_ns() + 60_000_000_000],
)
async def test_docker_screenshot_falls_back_for_invalid_capture_timestamp(
    invalid_timestamp_ns,
):
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = _png_base64()
    fallback_frame = _png_base64(640, 480)
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": encoded,
        "capture_timestamp_ns": invalid_timestamp_ns,
        "detected_controls": [
            {
                "name": "Submit",
                "role": "button",
                "bbox": (10, 10, 40, 30),
                "source": "atspi",
                "freshness_ms": 0,
            }
        ],
    })
    provider._docker_exec = AsyncMock(return_value=(
        0,
        f"___CAPTURE_TIMESTAMP_NS___\n{time.time_ns()}\n{fallback_frame}"
        "\n___OCR_TSV___\n"
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "___ACTIVE_WINDOW___\nFresh fallback\n___OPEN_WINDOWS___\n",
        "",
    ))

    observation = await provider.screenshot("ws-1")

    assert observation.screenshot_base64 == fallback_frame
    assert observation.active_window == "Fresh fallback"
    assert observation.capture_age_ms is not None
    assert observation.capture_age_ms >= 0
    provider._docker_exec.assert_awaited_once()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_falls_back_to_ocr_when_nsi_has_no_controls(monkeypatch):
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    nsi_frame = _png_base64()
    fallback_frame = _png_base64(640, 480)
    warnings = []
    monkeypatch.setattr(
        docker_computer_module,
        "logger",
        SimpleNamespace(
            info=lambda event, **fields: None,
            warning=lambda event, **fields: warnings.append((event, fields))
        ),
    )
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": nsi_frame,
        "active_window": "workspace",
        "visible_text": "Active Window: workspace",
        "open_windows": ["workspace"],
        "ocr_error": "Tesseract timed out",
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
            "name": "File",
            "role": "text",
            "bbox": (100, 200, 140, 218),
            "bbox_format": "xyxy",
            "source": "ocr",
            "confidence": 95.0,
        }
    ]
    assert (
        "nsi_frame_ocr_failed",
        {
            "container": "sonic-desktop-workstation",
            "error": "Tesseract timed out",
        },
    ) in warnings
    provider._docker_exec.assert_awaited_once()


@pytest.mark.no_live_infra
def test_nsi_actionable_controls_require_fresh_in_bounds_reliable_controls():
    assert not _has_actionable_nsi_controls([
        {
            "name": "Na)",
            "role": "text",
            "bbox": [184, 167, 231, 180],
            "source": "ocr",
            "confidence": 31.0,
            "freshness_ms": 0,
        }
    ], width=1280, height=800)
    assert not _has_actionable_nsi_controls([
        {
            "name": "workspace",
            "role": "window",
            "bbox": [0, 30, 650, 540],
            "source": "x11",
            "freshness_ms": 0,
        }
    ], width=1280, height=800)
    assert _has_actionable_nsi_controls([
        {
            "name": "Submit",
            "role": "button",
            "bbox": [10, 10, 40, 30],
            "source": "atspi",
            "freshness_ms": 0,
        }
    ], width=1280, height=800)
    assert _has_actionable_nsi_controls([
        {
            "name": "File",
            "role": "text",
            "bbox": [10, 10, 40, 30],
            "source": "ocr",
            "confidence": 82.0,
            "freshness_ms": 0,
        }
    ], width=1280, height=800)
    assert not _has_actionable_nsi_controls([
        {
            "name": "Stale control",
            "role": "button",
            "bbox": [10, 10, 40, 30],
            "source": "atspi",
            "freshness_ms": 251,
        }
    ], width=1280, height=800)
    assert not _has_actionable_nsi_controls([
        {
            "name": "Offscreen control",
            "role": "button",
            "bbox": [1270, 790, 1290, 810],
            "source": "atspi",
            "freshness_ms": 0,
        }
    ], width=1280, height=800)
    assert not _has_actionable_nsi_controls([
        {
            "name": "Unknown-age control",
            "role": "button",
            "bbox": [10, 10, 40, 30],
            "source": "atspi",
        }
    ], width=1280, height=800)


@pytest.mark.no_live_infra
def test_nsi_ocr_timeout_kills_process_group_and_reaps_shell():
    daemon = ast.parse(DockerComputerProvider._NSI_DAEMON_SCRIPT)
    run_function = next(
        node
        for node in daemon.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    kill_process_group = Mock()

    class TimedOutProcess:
        pid = 123

        def __init__(self):
            self.communication_timeouts = []

        def communicate(self, timeout=None):
            self.communication_timeouts.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("tesseract", timeout)
            return "", "OCR child terminated"

    process = TimedOutProcess()
    popen = Mock(return_value=process)
    runner_namespace = {
        "os": SimpleNamespace(killpg=kill_process_group),
        "signal": SimpleNamespace(SIGKILL=9),
        "subprocess": SimpleNamespace(
            Popen=popen,
            PIPE=subprocess.PIPE,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    }
    module = ast.Module(body=[run_function], type_ignores=[])
    exec(compile(module, "<nsi-daemon-runner>", "exec"), runner_namespace)

    exit_code, stdout, stderr = runner_namespace["_run"]("tesseract frame.png", 0.1)

    assert exit_code == 124
    assert stdout == ""
    assert stderr == "OCR child terminated"
    assert popen.call_args.kwargs["start_new_session"] is True
    kill_process_group.assert_called_once_with(123, 9)
    assert process.communication_timeouts == [0.1, None]


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
    assert provider._nsi_booted is True
    boot_command = provider.boot_command
    assert "nohup python3 /tmp/sonic_nsi_daemon.py" in boot_command
    assert "pkill -f" not in boot_command
    assert "/tmp/sonic_nsi_daemon.pid" in boot_command


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_nsi_daemon_boot_retries_after_initial_failure():
    class TestProvider(DockerComputerProvider):
        boot_attempts = 0

        async def _docker_exec(self, cmd, timeout=60):
            self.boot_attempts += 1
            if self.boot_attempts == 1:
                return 1, "", "startup failed"
            return 0, "", ""

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
            if self.calls <= 2:
                return SimpleNamespace(status_code=503, json=lambda: {})
            return SimpleNamespace(status_code=200, json=lambda: frame)

    provider = TestProvider(container_name="isolated-workstation")
    provider._nsi_http = NsiHttp()
    provider._nsi_booted = False
    provider._resolve_nsi_frame_url = AsyncMock(return_value=endpoint)

    assert await provider._nsi_request({"cmd": "frame"}) is None
    assert provider._nsi_booted is False

    assert await provider._nsi_request({"cmd": "frame"}) == frame
    assert provider.boot_attempts == 2
    assert provider._nsi_booted is True


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_docker_screenshot_parses_capture_timestamp_from_fallback_capture():
    provider = DockerComputerProvider(container_name="sonic-desktop-workstation")
    encoded = _png_base64()
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
    encoded = _png_base64()
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
    encoded = _png_base64()
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
    second = await provider.screenshot("ws-1")

    assert first.desktop_state == "INTERACTIVE"
    assert second.desktop_state == "NO_DISPLAY"
    assert second.screenshot_base64 == ""
    assert "fallback" in second.visible_text.lower()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_stale_actionable_nsi_frame_uses_fallback_frame_and_controls():
    provider = DockerComputerProvider()
    nsi_frame = _png_base64()
    fallback_frame = _png_base64(640, 480)
    stale_controls = [{
        "name": "Old Submit",
        "role": "button",
        "bbox": [10, 10, 90, 40],
        "bbox_format": "xyxy",
        "source": "atspi",
    }]
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": nsi_frame,
        "capture_timestamp_ns": time.time_ns() - 3_000_000_000,
        "detected_controls": stale_controls,
    })
    provider._docker_exec = AsyncMock(return_value=(
        0,
        (
            f"___CAPTURE_TIMESTAMP_NS___\n{time.time_ns()}\n{fallback_frame}"
            "\n___OCR_TSV___\n"
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t100\t200\t40\t18\t95.0\tCurrent\n"
            "___ACTIVE_WINDOW___\nCurrent Window\n___OPEN_WINDOWS___\n"
        ),
        "",
    ))

    observation = await provider.screenshot("ws-1")

    assert observation.screenshot_base64 == fallback_frame
    assert observation.capture_age_ms < 2_000
    assert observation.active_window == "Current Window"
    assert observation.detected_controls == [{
        "name": "Current",
        "role": "text",
        "bbox": (100, 200, 140, 218),
        "bbox_format": "xyxy",
        "source": "ocr",
        "confidence": 95.0,
    }]


@pytest.mark.no_live_infra
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "controls",
    [
        [],
        [{
            "name": "Old Submit",
            "role": "button",
            "bbox": [10, 10, 90, 40],
            "source": "ocr",
            "confidence": 95,
            "freshness_ms": 1_500,
        }],
    ],
    ids=["no-controls", "stale-controls"],
)
async def test_fresh_nsi_pixels_do_not_require_fresh_controls(controls):
    provider = DockerComputerProvider()
    nsi_frame = _png_base64()
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": nsi_frame,
        "capture_timestamp_ns": time.time_ns(),
        "capture_age_ms": 25,
        "active_window": "Current Window",
        "visible_text": "Current screen",
        "detected_controls": controls,
    })
    provider._docker_exec = AsyncMock()

    observation = await provider.screenshot("ws-1")

    assert observation.desktop_state == "INTERACTIVE"
    assert observation.screenshot_base64 == nsi_frame
    assert observation.capture_age_ms < 2_000
    assert observation.active_window == "Current Window"
    assert observation.visible_text == "Current screen"
    assert observation.detected_controls == []
    provider._docker_exec.assert_not_awaited()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_stale_nsi_capture_age_value_forces_refresh_even_with_recent_timestamp():
    provider = DockerComputerProvider()
    fallback_frame = _png_base64(640, 480)
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": _png_base64(),
        "capture_timestamp_ns": time.time_ns(),
        "capture_age_ms": 3_000,
        "detected_controls": [{
            "name": "Old frame button",
            "role": "button",
            "bbox": [10, 10, 90, 40],
            "source": "atspi",
        }],
    })
    provider._docker_exec = AsyncMock(return_value=(
        0,
        f"___CAPTURE_TIMESTAMP_NS___\n{time.time_ns()}\n{fallback_frame}"
        "\n___ACTIVE_WINDOW___\nFresh frame\n___OPEN_WINDOWS___\n",
        "",
    ))

    observation = await provider.screenshot("ws-1")

    assert observation.screenshot_base64 == fallback_frame
    assert observation.active_window == "Fresh frame"
    assert observation.detected_controls == []


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_stale_nsi_pixels_are_not_returned_when_fallback_fails():
    provider = DockerComputerProvider()
    nsi_frame = _png_base64()
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": nsi_frame,
        "capture_timestamp_ns": time.time_ns() - 3_000_000_000,
        "detected_controls": [{
            "name": "Submit",
            "role": "button",
            "bbox": [10, 10, 90, 40],
            "source": "atspi",
        }],
    })
    provider._docker_exec = AsyncMock(return_value=(126, "", "display unavailable"))

    observation = await provider.screenshot("ws-1")

    assert observation.desktop_state == "NO_DISPLAY"
    assert observation.screenshot_base64 == ""
    assert observation.detected_controls == []
    assert "valid PNG" in observation.visible_text


@pytest.mark.no_live_infra
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_frame",
    ["bm90LWEtcG5n", "%%%not-base64%%%", "iVBORw0KGgo="],
)
async def test_malformed_nsi_and_fallback_images_are_not_interactive(invalid_frame):
    provider = DockerComputerProvider()
    provider._nsi_request = AsyncMock(return_value={
        "ok": True,
        "b64": invalid_frame,
        "detected_controls": [{
            "name": "Submit",
            "role": "button",
            "bbox": [10, 10, 90, 40],
            "source": "atspi",
        }],
    })
    provider._docker_exec = AsyncMock(return_value=(0, invalid_frame, ""))

    observation = await provider.screenshot("ws-1")

    assert observation.desktop_state == "NO_DISPLAY"
    assert observation.screenshot_base64 == ""
    assert observation.detected_controls == []


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_gui_pointer_coordinates_require_fresh_screen_and_in_bounds():
    provider = DockerComputerProvider()
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider.screenshot = AsyncMock(return_value=ScreenObservation(
        screenshot_base64=_png_base64(80, 60),
        width=80,
        height=60,
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
    provider._nsi_request = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(0, "X=40\nY=30\nSCREEN=0\n", ""))
    observation = ScreenObservation(
        screenshot_base64=_png_base64(80, 60),
        width=80,
        height=60,
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
    [
        "X=39\nY=30\n",
        "X=40\n",
        "",
    ],
)
@pytest.mark.asyncio
async def test_docker_gui_move_fails_when_cursor_position_is_unverified(cursor_output):
    provider = DockerComputerProvider()
    provider._nsi_request = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(0, cursor_output, ""))
    observation = ScreenObservation(
        screenshot_base64=_png_base64(80, 60),
        width=80,
        height=60,
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
    provider._last_screenshot = ScreenObservation(
        screenshot_base64=_png_base64(),
        width=1280,
        height=800,
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
