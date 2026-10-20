"""
SONIC A-SEA — Docker Cyber Workstation Provider
================================================
First-class native Linux Workstation provider operating inside Docker
(sonic-desktop-workstation). Replaces cloud sandboxes (Daytona/E2B)
with a full, unrestricted, self-contained cyber environment:
- Desktop GUI: XFCE4, Xvfb :99 (1920x1080x24), xdotool, wmctrl, ImageMagick
- Interactive Streaming: x11vnc (:5900) + noVNC websockify (:6080)
- Graphical desktop applications and an isolated operator workspace
- Zero cloud rate limits or connection resets.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any
import zlib

from sonic.computer.models import (
    ApplicationPolicy,
    ComputerAuditEvent,
    ComputerProfile,
    ComputerState,
    ComputerWorkspace,
    ComputerWorkspaceStatus,
    ComputerWorkspaceType,
    FileEntry,
    GitStatusInfo,
    GUIAction,
    GUIActionType,
    ProcessInfo,
    ScreenObservation,
    ServiceInfo,
)
from sonic.computer.provider import ComputerProvider
from sonic.logger import get_logger
from sonic.sandbox.provider import ExecResult

logger = get_logger(__name__)

_SCREEN_CAPTURE_FRESHNESS_BUDGET_MS = 2_000
_RAW_FRAME_HEADER = struct.Struct("!4sQQII")
_RAW_FRAME_MAGIC = b"AVCS"
_RAW_FRAME_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class _RawDesktopFrame:
    frame_id: int
    capture_timestamp_ns: int
    width: int
    height: int
    raw_rgb: bytes
    avcs_state: dict[str, Any]
    capture_age_ms: float


def _get_display(workspace_id: str | None = None) -> str:
    """Check if environment has DISPLAY, default to :99 if running docker Xvfb."""
    return os.environ.get("DISPLAY") or ":99"


def _docker_binary() -> str | None:
    """Resolve Docker even when the backend is launched without the user PATH."""
    configured = os.environ.get("SONIC_DOCKER_BIN", "").strip()
    if configured:
        return configured if Path(configured).exists() else None
    found = shutil.which("docker")
    if found:
        return found
    if os.name == "nt":
        for candidate in (
            r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
            r"C:\ProgramData\DockerDesktop\version-bin\docker.exe",
        ):
            if Path(candidate).exists():
                return candidate
    return None


def _validated_png_dimensions(encoded: Any) -> tuple[int, int] | None:
    """Return PNG dimensions only for a complete, structurally valid PNG."""
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        image = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError, TypeError):
        return None
    if len(image) < 57 or image[:8] != b"\x89PNG\r\n\x1a\n":
        return None

    offset = 8
    dimensions: tuple[int, int] | None = None
    row_bytes: int | None = None
    height: int | None = None
    image_data = bytearray()
    found_idat = False
    found_iend = False
    while offset + 12 <= len(image):
        chunk_length = struct.unpack_from(">I", image, offset)[0]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(image):
            return None
        chunk_type = image[offset + 4:offset + 8]
        chunk_data = image[offset + 8:offset + 8 + chunk_length]
        chunk_crc = struct.unpack_from(">I", image, offset + 8 + chunk_length)[0]
        if (binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF) != chunk_crc:
            return None

        if offset == 8:
            if chunk_type != b"IHDR" or chunk_length != 13:
                return None
            width, height = struct.unpack_from(">II", chunk_data)
            if width <= 0 or height <= 0 or width > 16_384 or height > 16_384:
                return None
            bit_depth, color_type, _, _, interlace = chunk_data[8:13]
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
            if (
                channels is None
                or bit_depth not in (1, 2, 4, 8, 16)
                or (color_type == 3 and bit_depth not in (1, 2, 4, 8))
                or (color_type == 2 and bit_depth not in (8, 16))
                or (color_type == 4 and bit_depth not in (8, 16))
                or (color_type == 6 and bit_depth not in (8, 16))
                or chunk_data[10:12] != b"\x00\x00"
                or interlace != 0
            ):
                return None
            row_bytes = (width * channels * bit_depth + 7) // 8
            if (row_bytes + 1) * height > 64 * 1024 * 1024:
                return None
            dimensions = (width, height)
        elif chunk_type == b"IHDR":
            return None

        if chunk_type == b"IDAT":
            found_idat = True
            image_data.extend(chunk_data)
        if chunk_type == b"IEND":
            if chunk_length != 0 or not found_idat or chunk_end != len(image):
                return None
            found_iend = True
            break
        offset = chunk_end

    if not dimensions or row_bytes is None or height is None or not found_iend:
        return None
    expected_size = (row_bytes + 1) * height
    try:
        decompressor = zlib.decompressobj()
        pixels = decompressor.decompress(bytes(image_data), expected_size + 1)
        if (
            len(pixels) != expected_size
            or not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            return None
    except zlib.error:
        return None
    if any(pixels[row * (row_bytes + 1)] > 4 for row in range(height)):
        return None
    return dimensions


class DockerComputerProvider(ComputerProvider):
    """Native Docker-based Workstation Provider for SONIC A-SEA."""

    def _get_display(self, workspace_id: str | None = None) -> str:
        """Check if environment has DISPLAY, default to :99 if running docker Xvfb."""
        return _get_display(workspace_id)

    def __init__(
        self,
        container_name: str = "sonic-desktop-workstation",
        app_policy: ApplicationPolicy | None = None,
    ):
        self.container_name = os.environ.get("SONIC_DOCKER_WORKSTATION_CONTAINER", container_name).strip()
        self.app_policy = app_policy or ApplicationPolicy()
        self.workspaces: dict[str, ComputerWorkspace] = {}
        self._active_windows: dict[str, str] = {}
        self.audit_log: list[ComputerAuditEvent] = []
        self._last_exec: tuple[str, int, str, str] | None = None  # (cmd, exit, stdout, stderr)
        self._daemon_checked: bool | None = None
        self._daemon_checked_at: float | None = None
        self._container_running_cache: bool = False
        self._container_checked_at: float | None = None
        self._default_workspace_id = self.container_name
        # A single Docker desktop is not a tenant-isolated resource.  Refuse
        # cross-tenant reuse rather than silently moving ownership on every
        # create() call.
        self._workspace_owner: str | None = None
        self._exec_semaphore = asyncio.Semaphore(10)
        self._gui_lock = asyncio.Lock()
        self._screenshot_lock = asyncio.Lock()
        self._last_screenshot: ScreenObservation | None = None
        self._last_screenshot_time: float = 0.0
        self._novnc_url: str | None = None
        self._novnc_url_checked_at: float = 0.0
        self._avcs_perception = None
        self._frame_stream_process: asyncio.subprocess.Process | None = None
        self._frame_stream_reader_task: asyncio.Task[None] | None = None
        self._frame_stream_stderr_task: asyncio.Task[None] | None = None
        self._frame_stream_processor_task: asyncio.Task[None] | None = None
        self._frame_stream_queue: asyncio.Queue[tuple[int, int, int, int, bytes]] = (
            asyncio.Queue(maxsize=1)
        )
        self._frame_ready = asyncio.Event()
        self._frame_stream_error: str | None = None
        self._latest_processed_frame: _RawDesktopFrame | None = None
        self._last_stream_frame_id = 0
        self._last_stream_source_frame_id = 0
        self._stream_frame_id_offset = 0
        self._last_returned_frame_id = 0
        self._stream_frames_dropped = 0
        self._capture_times_ns: deque[int] = deque(maxlen=30)
        self._processing_times: deque[float] = deque(maxlen=30)
        self._frame_stream_lock = asyncio.Lock()

    def _ensure_default_workspace(self, tenant_id: str = "default", engagement_id: str = "default") -> ComputerWorkspace:
        if self._default_workspace_id not in self.workspaces:
            ws = ComputerWorkspace(
                id=self._default_workspace_id,
                tenant_id=tenant_id,
                engagement_id=engagement_id,
                workspace_type=ComputerWorkspaceType.MISSION_COMPUTER,
                profile=ComputerProfile.DEBIAN_ENGINEERING,
                provider_type="DockerComputerProvider",
                image="sonic-workstation:latest",
                status=ComputerWorkspaceStatus.READY,
            )
            self.workspaces[self._default_workspace_id] = ws
        return self.workspaces[self._default_workspace_id]

    @property
    def name(self) -> str:
        return "DockerComputerProvider"

    async def _container_is_running(self) -> bool:
        """Probe the real Docker container state (fail-closed, short TTL cache)."""
        docker = _docker_binary()
        if not docker:
            return False
        now = asyncio.get_event_loop().time()
        if self._container_checked_at is not None and (now - self._container_checked_at) < 1.5:
            return self._container_running_cache
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [docker, "inspect", "-f", "{{.State.Running}}", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            running = probe.stdout.decode("utf-8", errors="ignore").strip() == "true"
        except Exception:
            running = False
        self._container_running_cache = running
        self._container_checked_at = now
        return running

    async def _provision_workstation(self) -> bool:
        """Try to bring up the workstation container from repo compose files.

        Gated bye SONIC_AUTO_PROVISION_WORKSTATION (default off) — a full
        XFCE desktop image build can take minutes and should never silently
        trigger from a polling/status code path.
        """
        if os.environ.get("SONIC_AUTO_PROVISION_WORKSTATION", "0").strip() != "1":
            return False
        docker = _docker_binary()
        if not docker:
            return False

        candidates = [
            Path.cwd() / "docker-compose.yml",
            Path.cwd() / "docker-compose.prod.yml",
            Path.cwd().parent / "docker-compose.yml",
            Path(__file__).resolve().parents[4] / "docker-compose.yml",
        ]
        compose_file = next((p for p in candidates if p.exists()), None)
        if compose_file is None:
            logger.error("workstation_auto_provision_no_compose_file")
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                docker, "compose", "-f", str(compose_file), "up", "-d", "workstation",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
            if proc.returncode == 0:
                logger.info("workstation_auto_provisioned", container=self.container_name)
                return True
            logger.error("workstation_auto_provision_failed", error=stderr.decode("utf-8", errors="replace").strip()[:400])
            return False
        except Exception as e:
            logger.error("workstation_auto_provision_exception", error=str(e))
            return False

    async def _docker_exec(self, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
        """Execute a bash command inside the docker container."""
        docker = _docker_binary()
        if not docker:
            return 127, "", "docker binary not found on host"
        if not await self._container_is_running():
            return 125, "", f"container {self.container_name} is not running; command blocked fail-closed"
        now = asyncio.get_event_loop().time()
        if (
            self._daemon_checked is None
            or not self._daemon_checked
            or (self._daemon_checked_at is not None and (now - self._daemon_checked_at) > 10.0)
        ):
            try:
                probe = await asyncio.to_thread(
                    subprocess.run,
                    [docker, "info", "--format", "{{.ServerVersion}}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
                self._daemon_checked = probe.returncode == 0
                self._daemon_checked_at = now
            except Exception:
                self._daemon_checked = False
                self._daemon_checked_at = now
        if not self._daemon_checked:
            return 126, "", "docker daemon is unreachable; command execution failed-closed"
        async with self._exec_semaphore:
            try:
                disp = self._get_display("") if hasattr(self, "_get_display") else ":99"
                ws_dir = os.environ.get("SONIC_WORKSPACE_ROOT", "/home/sonic/workspace")
                exec_args = [docker, "exec", "-e", f"DISPLAY={disp}", "-w", ws_dir, self.container_name, "bash", "-c", cmd]
                proc = await asyncio.create_subprocess_exec(
                    *exec_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout if timeout > 0 else 60,
                )
                exit_code = proc.returncode if proc.returncode is not None else 0
                stdout_str = stdout_data.decode("utf-8", errors="replace")
                stderr_str = stderr_data.decode("utf-8", errors="replace")
                self._last_exec = (cmd, exit_code, stdout_str, stderr_str)
                return exit_code, stdout_str, stderr_str
            except asyncio.CancelledError:
                if "proc" in locals() and proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                raise
            except TimeoutError:
                try:
                    proc.kill()
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
                return 124, "", f"Command timed out after {timeout} seconds"
            except Exception as e:
                try:
                    proc.kill()
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
                return 126, "", str(e)

    # -------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------

    async def create(
        self,
        tenant_id: str,
        engagement_id: str,
        workspace_type: ComputerWorkspaceType = ComputerWorkspaceType.MISSION_COMPUTER,
        profile: ComputerProfile = ComputerProfile.DEBIAN_ENGINEERING,
    ) -> ComputerWorkspace:
        if self._workspace_owner and self._workspace_owner != tenant_id:
            raise RuntimeError(
                "shared Docker workstation is already owned by another tenant; "
                "provision a dedicated workstation"
            )
        ws = self._ensure_default_workspace(tenant_id)
        self._workspace_owner = tenant_id
        ws.tenant_id = tenant_id
        ws.engagement_id = engagement_id
        ws.profile = profile
        if not await self._container_is_running():
            provisioned = await self._provision_workstation()
            if not provisioned:
                # Try to start existing container if it exists but is stopped
                docker = _docker_binary()
                if not docker:
                    ws.status = ComputerWorkspaceStatus.FAILED
                    return ws
                try:
                    start_proc = await asyncio.to_thread(
                        subprocess.run,
                        [docker, "start", self.container_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
                    if start_proc.returncode != 0:
                        ws.status = ComputerWorkspaceStatus.STOPPED
                        return ws
                    # Give container time to start and verify liveness
                    await asyncio.sleep(1.5)
                    self._container_running_cache = await self._container_is_running()
                    self._container_checked_at = asyncio.get_event_loop().time()
                    if not self._container_running_cache:
                        ws.status = ComputerWorkspaceStatus.STOPPED
                        return ws
                except Exception:
                    # Fail-closed: never report a desktop that was not actually provisioned.
                    ws.status = ComputerWorkspaceStatus.STOPPED if _docker_binary() else ComputerWorkspaceStatus.FAILED
                    return ws
            else:
                self._container_running_cache = True
                self._container_checked_at = asyncio.get_event_loop().time()
        ws.status = ComputerWorkspaceStatus.RUNNING
        return ws

    async def get_or_create_home(self, tenant_id: str) -> ComputerWorkspace:
        """Return the tenant's long-lived MISSION_COMPUTER home (persistent body)."""
        if self._workspace_owner and self._workspace_owner != tenant_id:
            raise RuntimeError(
                "shared Docker workstation is already owned by another tenant"
            )
        for ws in self.workspaces.values():
            if (
                ws.tenant_id == tenant_id
                and ws.workspace_type == ComputerWorkspaceType.MISSION_COMPUTER
                and ws.status not in (ComputerWorkspaceStatus.DESTROYED, ComputerWorkspaceStatus.FAILED)
            ):
                logger.info(
                    "home_workstation_reused",
                    workspace_id=ws.id,
                    tenant_id=tenant_id,
                )
                return ws
        ws = ComputerWorkspace(
            id=f"home-{tenant_id}",
            tenant_id=tenant_id,
            engagement_id=f"home-{tenant_id}",
            workspace_type=ComputerWorkspaceType.MISSION_COMPUTER,
            status=ComputerWorkspaceStatus.RUNNING,
        )
        self.workspaces[ws.id] = ws
        self._workspace_owner = tenant_id
        return ws

    async def destroy(self, workspace_id: str) -> bool:
        if workspace_id in self.workspaces:
            del self.workspaces[workspace_id]
        if not self.workspaces:
            await self.close()
        if (
            workspace_id == self._default_workspace_id
            or workspace_id.startswith("home-")
            or not self.workspaces
        ):
            self._workspace_owner = None
        return True

    async def close(self) -> None:
        """Stop the local raw-frame stream and release its pipe readers."""
        process = self._frame_stream_process
        if process is not None and process.returncode is None:
            stdin = process.stdin
            if stdin is not None and not stdin.is_closing():
                stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        for task in (
            self._frame_stream_reader_task,
            self._frame_stream_stderr_task,
            self._frame_stream_processor_task,
        ):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (
                    self._frame_stream_reader_task,
                    self._frame_stream_stderr_task,
                    self._frame_stream_processor_task,
                )
                if task is not None
            ),
            return_exceptions=True,
        )
        self._frame_stream_process = None
        self._frame_stream_reader_task = None
        self._frame_stream_stderr_task = None
        self._frame_stream_processor_task = None
        self._latest_processed_frame = None
        self._last_returned_frame_id = 0
        self._frame_ready.clear()

    async def get_vnc_url(self, workspace_id: str) -> str | None:
        # Fail-closed: a stream URL is only real when the container actually
        # runs, its noVNC listener is reachable, and that container's published
        # port can be resolved without crossing into another desktop session.
        if not await self._container_is_running():
            return None
        stream_url = await self._resolve_novnc_url()
        port_match = re.search(r":(\d+)/vnc\.html$", stream_url or "")
        if port_match is None:
            return None
        published_port = port_match.group(1)
        code, out, _ = await self._docker_exec(
            "if (exec 3<>/dev/tcp/127.0.0.1/6080) 2>/dev/null; then echo 0; else echo 1; fi",
            timeout=8,
        )
        if code == 0 and out.strip() == "0":
            return (
                f"http://localhost:{published_port}/vnc.html"
                "?autoconnect=true&resize=scale&show_dot=true&cursor=true"
            )
        return None

    async def get_stream_url(self, workspace_id: str) -> str | None:
        return await self.get_vnc_url(workspace_id)

    # -------------------------------------------------------------
    # Status & Telemetry
    # -------------------------------------------------------------

    async def status(self, workspace_id: str) -> ComputerState:
        ws = self._ensure_default_workspace()
        if not await self._container_is_running():
            # Fail-closed: no synthetic RUNNING/apps/windows when the container
            # does not actually exist. Report DEGRADED and empty telemetry.

            ws.status = ComputerWorkspaceStatus.DEGRADED
            return ComputerState(
                workspace_id=workspace_id or self.container_name,
                tenant_id=ws.tenant_id,
                status=ComputerWorkspaceStatus.DEGRADED,
                active_application="",
                open_applications=[],
                active_window="",
                working_directory="/home/sonic/workspace",
                running_processes=[],
                installed_applications=[],
                current_project="",
                git_branch="",
                resource_usage={},
            )

        ws.status = ComputerWorkspaceStatus.RUNNING
        disp = self._get_display(workspace_id)
        active_window = "Desktop"
        open_windows: list[str] = ["Desktop"]

        # Check real open windows via wmctrl
        code, out, _ = await self._docker_exec(f"DISPLAY={disp} wmctrl -l 2>/dev/null", timeout=5)
        if code == 0 and out.strip():
            for line in out.splitlines():
                parts = line.split(maxsplit=3)
                if len(parts) >= 4:
                    title = parts[3].strip()
                    if title and title not in ("xfce4-panel", "Desktop") and title not in open_windows:
                        open_windows.append(title)

        # Always ensure Desktop is in the list for running containers
        if "Desktop" not in open_windows:
            open_windows.insert(0, "Desktop")

        # Check active window via xdotool
        code, out, _ = await self._docker_exec(f"DISPLAY={disp} xdotool getactivewindow getwindowname 2>/dev/null", timeout=5)
        if code == 0 and out.strip():
            active_window = out.strip()
        else:
            active_window = open_windows[-1] if len(open_windows) > 1 else "Desktop"

        # Check running processes
        procs = await self.process_list(workspace_id)
        proc_names = [p.name for p in procs]
        installed = await self.application_list(workspace_id)

        return ComputerState(
            workspace_id=workspace_id or self.container_name,
            tenant_id=ws.tenant_id,
            status=ComputerWorkspaceStatus.RUNNING,
            active_application=active_window,
            open_applications=open_windows,
            active_window=active_window,
            working_directory="/home/sonic/workspace",
            running_processes=proc_names,
            installed_applications=installed,
            current_project="sonic-repo",
            git_branch="main",
            resource_usage={"cpu_pct": 5.0, "memory_mb": 512.0},
        )

    async def get_active_window_class(self, workspace_id: str) -> str:
        """Return the WM_CLASS identity of the active window, independent of its title."""
        disp = self._get_display(workspace_id)
        code, output, error = await self._docker_exec(
            f"DISPLAY={disp} xprop -root _NET_ACTIVE_WINDOW",
            timeout=5,
        )
        window_id_match = re.search(r"window id # (0x[0-9a-fA-F]+)", output)
        if code != 0 or window_id_match is None:
            logger.warning(
                "active_window_id_unavailable",
                exit_code=code,
                error=error.strip(),
            )
            return ""

        code, output, error = await self._docker_exec(
            f"DISPLAY={disp} xprop -id {window_id_match.group(1)} WM_CLASS",
            timeout=5,
        )
        class_match = re.search(
            r'WM_CLASS\(STRING\)\s*=\s*"([^"]+)"\s*,\s*"([^"]+)"',
            output,
        )
        if code != 0 or class_match is None:
            logger.warning(
                "active_window_class_unavailable",
                exit_code=code,
                error=error.strip(),
            )
            return ""
        return ".".join(class_match.groups())

    # -------------------------------------------------------------
    async def _resolve_novnc_url(self) -> str | None:
        """Resolve this container's locally published noVNC URL."""
        now = asyncio.get_running_loop().time()
        if now - getattr(self, "_novnc_url_checked_at", 0.0) < 5.0:
            return getattr(self, "_novnc_url", None)
        self._novnc_url_checked_at = now
        docker = _docker_binary()
        if not docker:
            logger.warning("novnc_url_unavailable", reason="docker_binary_not_found")
            self._novnc_url = None
            return None
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [docker, "inspect", "--format", "{{json .NetworkSettings.Ports}}", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("novnc_port_lookup_failed", container=self.container_name, error=str(exc))
            self._novnc_url = None
            return None
        if result.returncode != 0:
            logger.warning(
                "novnc_port_lookup_failed",
                container=self.container_name,
                exit_code=result.returncode,
                error=result.stderr.decode("utf-8", errors="replace").strip()[:300],
            )
            self._novnc_url = None
            return None
        try:
            ports = json.loads(result.stdout.decode("utf-8", errors="replace"))
            bindings = ports.get("6080/tcp") if isinstance(ports, dict) else None
            binding = next(
                (
                    item for item in (bindings or [])
                    if isinstance(item, dict)
                    and str(item.get("HostPort", "")).isdigit()
                    and item.get("HostIp") in {"127.0.0.1", "0.0.0.0", "::", ""}
                ),
                None,
            )
            if binding is None:
                logger.warning("novnc_port_unavailable", container=self.container_name)
                self._novnc_url = None
                return None
            self._novnc_url = f"http://127.0.0.1:{binding['HostPort']}/vnc.html"
            return self._novnc_url
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("novnc_port_lookup_invalid", container=self.container_name, error=str(exc))
            self._novnc_url = None
            return None

    async def screenshot(self, workspace_id: str) -> ScreenObservation:
        """Return the newest fresh frame from the in-memory AVCS stream."""
        async with self._screenshot_lock:
            self._last_screenshot = None
            self._last_screenshot_time = 0.0
            try:
                return await self._capture_screenshot(workspace_id)
            except BaseException:
                self._last_screenshot = None
                self._last_screenshot_time = 0.0
                self._latest_processed_frame = None
                self._frame_ready.clear()
                raise

    async def _start_raw_frame_stream(self, workspace_id: str) -> None:
        async with self._frame_stream_lock:
            process = self._frame_stream_process
            reader = self._frame_stream_reader_task
            if (
                process is not None
                and process.returncode is None
                and reader is not None
                and not reader.done()
            ):
                return

            await self.close()
            docker = _docker_binary()
            if not docker:
                raise RuntimeError("Docker binary is unavailable for AVCS frame streaming")
            if not await self._container_is_running():
                raise RuntimeError("Workstation container is unavailable for AVCS frame streaming")

            try:
                fps = float(os.environ.get("SONIC_AVCS_CAPTURE_FPS", "30"))
            except ValueError as exc:
                raise ValueError("SONIC_AVCS_CAPTURE_FPS must be between 1 and 60") from exc
            if not math.isfinite(fps) or not 1 <= fps <= 60:
                raise ValueError("SONIC_AVCS_CAPTURE_FPS must be between 1 and 60")

            process = await asyncio.create_subprocess_exec(
                docker,
                "exec",
                "-i",
                "-e",
                f"DISPLAY={self._get_display(workspace_id)}",
                self.container_name,
                "/opt/sonic/venv/bin/python",
                "/opt/sonic/avcs_frame_stream.py",
                "--fps",
                str(fps),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_RAW_FRAME_MAX_BYTES,
            )
            self._frame_stream_process = process
            self._frame_stream_error = None
            self._latest_processed_frame = None
            self._stream_frame_id_offset = self._last_stream_frame_id
            self._last_stream_source_frame_id = 0
            self._last_returned_frame_id = 0
            self._stream_frames_dropped = 0
            self._capture_times_ns.clear()
            self._processing_times.clear()
            self._frame_ready.clear()
            self._frame_stream_queue = asyncio.Queue(maxsize=1)
            if self._avcs_perception is None:
                from sonic.computer_use.avcs_perception import AVCSPerception

                self._avcs_perception = AVCSPerception()
            self._frame_stream_reader_task = asyncio.create_task(
                self._read_raw_frame_stream(process)
            )
            self._frame_stream_stderr_task = asyncio.create_task(
                self._drain_raw_frame_stream_stderr(process)
            )
            self._frame_stream_processor_task = asyncio.create_task(
                self._process_raw_frame_stream()
            )

    async def _read_raw_frame_stream(
        self, process: asyncio.subprocess.Process
    ) -> None:
        assert process.stdout is not None
        try:
            while True:
                header = await process.stdout.readexactly(_RAW_FRAME_HEADER.size)
                magic, frame_id, timestamp_ns, width, height = _RAW_FRAME_HEADER.unpack(
                    header
                )
                frame_size = width * height * 3
                if (
                    magic != _RAW_FRAME_MAGIC
                    or frame_id <= self._last_stream_source_frame_id
                    or timestamp_ns <= 0
                    or width <= 0
                    or height <= 0
                    or width * height > 64_000_000
                    or frame_size > _RAW_FRAME_MAX_BYTES
                ):
                    raise ValueError("AVCS stream emitted invalid frame metadata")
                raw_rgb = await process.stdout.readexactly(frame_size)
                frame_id += self._stream_frame_id_offset
                self._last_stream_source_frame_id = (
                    frame_id - self._stream_frame_id_offset
                )
                self._last_stream_frame_id = frame_id
                self._capture_times_ns.append(timestamp_ns)
                if self._frame_stream_queue.full():
                    try:
                        self._frame_stream_queue.get_nowait()
                        self._stream_frames_dropped += 1
                    except asyncio.QueueEmpty:
                        pass
                self._frame_stream_queue.put_nowait(
                    (frame_id, timestamp_ns, width, height, raw_rgb)
                )
        except asyncio.CancelledError:
            raise
        except (asyncio.IncompleteReadError, OSError, ValueError) as exc:
            self._frame_stream_error = str(exc)
            self._latest_processed_frame = None
            self._frame_ready.set()
            if process.returncode is None:
                process.terminate()
            logger.error(
                "avcs_raw_frame_stream_failed",
                container=self.container_name,
                error=str(exc),
            )

    async def _drain_raw_frame_stream_stderr(
        self, process: asyncio.subprocess.Process
    ) -> None:
        assert process.stderr is not None
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            logger.warning(
                "avcs_raw_frame_stream_stderr",
                container=self.container_name,
                message=line.decode("utf-8", errors="replace").strip()[:300],
            )

    async def _process_raw_frame_stream(self) -> None:
        while True:
            frame_id, timestamp_ns, width, height, raw_rgb = (
                await self._frame_stream_queue.get()
            )
            started = time.perf_counter()
            try:
                if self._avcs_perception is None:
                    from sonic.computer_use.avcs_perception import AVCSPerception

                    self._avcs_perception = AVCSPerception()
                state = await asyncio.to_thread(
                    self._avcs_perception.perceive_rgb_bytes,
                    raw_rgb,
                    width,
                    height,
                    frame_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._frame_stream_error = f"AVCS raw-frame processing failed: {exc}"
                self._latest_processed_frame = None
                self._frame_ready.set()
                logger.error(
                    "avcs_raw_frame_processing_failed",
                    container=self.container_name,
                    frame_id=frame_id,
                    error=str(exc),
                )
                continue

            if (
                not isinstance(state, dict)
                or state.get("engine") != "avcs_reflex"
                or state.get("source_frame_id") != frame_id
                or state.get("width") != width
                or state.get("height") != height
                or not isinstance(state.get("scene_hash"), str)
                or not state["scene_hash"]
            ):
                self._frame_stream_error = "AVCS returned invalid raw-frame state"
                self._latest_processed_frame = None
                self._frame_ready.set()
                logger.error(
                    "avcs_raw_frame_invalid_state",
                    container=self.container_name,
                    frame_id=frame_id,
                )
                continue

            processing_finished = time.perf_counter()
            self._processing_times.append(processing_finished)
            capture_age_ms = (time.time_ns() - timestamp_ns) / 1_000_000
            if (
                not math.isfinite(capture_age_ms)
                or capture_age_ms < 0
                or capture_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
            ):
                logger.warning(
                    "avcs_raw_frame_stale_after_processing",
                    container=self.container_name,
                    frame_id=frame_id,
                    capture_age_ms=capture_age_ms,
                )
                self._frame_ready.set()
                continue

            state = dict(state)
            state["capture_fps"] = self._measured_fps(
                self._capture_times_ns, timestamps=True
            )
            state["processing_fps"] = self._measured_fps(
                self._processing_times, timestamps=False
            )
            state["dropped_frames"] = self._stream_frames_dropped
            self._latest_processed_frame = _RawDesktopFrame(
                frame_id=frame_id,
                capture_timestamp_ns=timestamp_ns,
                width=width,
                height=height,
                raw_rgb=raw_rgb,
                avcs_state=state,
                capture_age_ms=capture_age_ms,
            )
            self._frame_stream_error = None
            self._frame_ready.set()
            logger.debug(
                "avcs_raw_frame_processed",
                frame_id=frame_id,
                capture_age_ms=round(capture_age_ms, 2),
                process_ms=round((processing_finished - started) * 1_000, 2),
                capture_fps=state["capture_fps"],
                processing_fps=state["processing_fps"],
                dropped_frames=state["dropped_frames"],
            )

    @staticmethod
    def _measured_fps(samples: deque[int] | deque[float], *, timestamps: bool) -> float:
        if len(samples) < 2:
            return 0.0
        elapsed = (
            (samples[-1] - samples[0]) / 1_000_000_000
            if timestamps
            else float(samples[-1] - samples[0])
        )
        return round((len(samples) - 1) / elapsed, 2) if elapsed > 0 else 0.0

    async def _capture_screenshot(self, workspace_id: str) -> ScreenObservation:
        code, output, error = await self._docker_exec(
            "printf '%s\\n' '___ACTIVE_WINDOW___'; "
            "xdotool getactivewindow getwindowname; "
            "printf '%s\\n' '___OPEN_WINDOWS___'; wmctrl -l",
            timeout=5,
        )
        active_window = ""
        windows: list[str] = []
        if code == 0:
            active_part, separator, windows_part = output.partition(
                "___OPEN_WINDOWS___"
            )
            active_window = (
                active_part.split("___ACTIVE_WINDOW___", 1)[-1].strip()
                if "___ACTIVE_WINDOW___" in active_part
                else ""
            )
            if separator:
                for line in windows_part.splitlines():
                    fields = line.strip().split(maxsplit=3)
                    if len(fields) == 4 and fields[3] not in {
                        "xfce4-panel",
                        "Desktop",
                        "desktop",
                    }:
                        windows.append(fields[3])
        else:
            logger.warning(
                "avcs_stream_window_metadata_unavailable",
                workspace_id=workspace_id,
                exit_code=code,
                error=error.strip()[:300],
            )

        try:
            await self._start_raw_frame_stream(workspace_id)
            deadline = asyncio.get_running_loop().time() + 5
            while True:
                current = self._latest_processed_frame
                if (
                    current is not None
                    and current.frame_id > self._last_returned_frame_id
                ):
                    break
                self._frame_ready.clear()
                current = self._latest_processed_frame
                if (
                    current is not None
                    and current.frame_id > self._last_returned_frame_id
                ):
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a new processed AVCS frame")
                await asyncio.wait_for(self._frame_ready.wait(), timeout=remaining)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            self._frame_stream_error = str(exc)
            self._latest_processed_frame = None
            self._frame_ready.clear()
            logger.error(
                "avcs_raw_frame_stream_unavailable",
                workspace_id=workspace_id,
                error=str(exc),
            )
            return ScreenObservation(
                width=0,
                height=0,
                visible_text=f"AVCS raw frame stream unavailable: {exc}",
                desktop_state="NO_DISPLAY",
            )

        frame = self._latest_processed_frame
        if frame is None or frame.frame_id <= self._last_returned_frame_id:
            return ScreenObservation(
                width=0,
                height=0,
                visible_text=(
                    self._frame_stream_error
                    or "AVCS raw frame stream has not produced a processed frame."
                ),
                desktop_state="NO_DISPLAY",
            )

        age_ms = (time.time_ns() - frame.capture_timestamp_ns) / 1_000_000
        if (
            not math.isfinite(age_ms)
            or age_ms < 0
            or age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
        ):
            self._latest_processed_frame = None
            self._frame_ready.clear()
            return ScreenObservation(
                frame_id=frame.frame_id,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                capture_age_ms=age_ms,
                width=frame.width,
                height=frame.height,
                visible_text="AVCS raw frame stream returned stale pixels.",
                desktop_state="NO_DISPLAY",
            )

        age_ms = (time.time_ns() - frame.capture_timestamp_ns) / 1_000_000
        if age_ms < 0 or age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS:
            return ScreenObservation(
                frame_id=frame.frame_id,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                capture_age_ms=age_ms,
                width=frame.width,
                height=frame.height,
                visible_text="AVCS raw frame became stale while reading window state.",
                desktop_state="NO_DISPLAY",
            )
        observation = ScreenObservation(
            raw_rgb=frame.raw_rgb,
            frame_id=frame.frame_id,
            avcs_state=frame.avcs_state,
            capture_timestamp_ns=frame.capture_timestamp_ns,
            capture_age_ms=age_ms,
            perception_version=int(frame.avcs_state["frame_index"]),
            width=frame.width,
            height=frame.height,
            active_window=active_window,
            visible_text="",
            detected_controls=[],
            desktop_state="INTERACTIVE",
        )
        self._last_screenshot = observation
        self._last_screenshot_time = asyncio.get_running_loop().time()
        self._last_returned_frame_id = frame.frame_id
        return observation

    # Graphical Actions (Human Takeover & Agent GUI Actions)
    # -------------------------------------------------------------

    @staticmethod
    def _normalize_x11_keysym(raw_key: str) -> str:
        """Normalize common key aliases into valid X11 keysyms for xdotool."""
        key_map = {
            "enter": "Return",
            "return": "Return",
            "esc": "Escape",
            "escape": "Escape",
            "backspace": "BackSpace",
            "space": "space",
            "tab": "Tab",
            "del": "Delete",
            "delete": "Delete",
            "pgup": "Prior",
            "pageup": "Prior",
            "pgdn": "Next",
            "pagedown": "Next",
            "up": "Up",
            "down": "Down",
            "left": "Left",
            "right": "Right",
        }
        tokens = [t.strip() for t in (raw_key or "").split("+") if t.strip()]
        normalized = [key_map.get(t.lower(), t) for t in tokens]
        return "+".join(normalized) if normalized else raw_key

    async def gui_action(
        self,
        workspace_id: str,
        action: GUIAction,
        actor: str = "operator",
    ) -> ScreenObservation:
        async with self._gui_lock:
            atype = action.action
            disp = self._get_display(workspace_id)

            async def exec_gui(command: str) -> str:
                code, stdout, err = await self._docker_exec(command)
                if code != 0:
                    raise RuntimeError(f"GUI action failed with exit code {code}: {err.strip()}")
                return stdout

            async def _validate_pointer_coords(
                *points: tuple[int | None, int | None],
            ) -> None:
                if any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for point in points
                    for value in point
                ):
                    raise ValueError("Pointer actions require explicit integer coordinates")

                now = asyncio.get_running_loop().time()
                observation = self._last_screenshot
                cache_age_ms = (
                    max(0.0, now - self._last_screenshot_time) * 1000
                    if observation is not None and self._last_screenshot_time > 0
                    else float("inf")
                )
                if (
                    observation is None
                    or observation.desktop_state != "INTERACTIVE"
                    or cache_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                    or observation.capture_age_ms is None
                    or observation.capture_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                ):
                    observation = await self.screenshot(workspace_id)
                dimensions = (
                    (
                        (observation.width, observation.height)
                        if observation.raw_rgb
                        and isinstance(observation.width, int)
                        and not isinstance(observation.width, bool)
                        and isinstance(observation.height, int)
                        and not isinstance(observation.height, bool)
                        and observation.width > 0
                        and observation.height > 0
                        and len(observation.raw_rgb)
                        == observation.width * observation.height * 3
                        else _validated_png_dimensions(observation.screenshot_base64)
                    )
                    if observation.desktop_state == "INTERACTIVE"
                    and observation.capture_age_ms is not None
                    and observation.capture_age_ms <= _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                    else None
                )
                if dimensions is None:
                    raise RuntimeError("Pointer action blocked: no fresh AVCS screen bounds")
                width, height = dimensions
                for cx, cy in points:
                    if cx is None or cy is None:
                        raise ValueError(f"GUI coordinates cannot be None: ({cx}, {cy})")
                    if cx < 0 or cy < 0 or cx >= width or cy >= height:
                        raise ValueError(
                            f"GUI coordinates ({cx}, {cy}) outside captured screen bounds "
                            f"{width}x{height}"
                        )

            pointer_points: tuple[tuple[int | None, int | None], ...] = ()
            if atype in (
                GUIActionType.CLICK,
                GUIActionType.DOUBLE_CLICK,
                GUIActionType.RIGHT_CLICK,
                GUIActionType.MOVE,
                GUIActionType.SCROLL,
            ):
                pointer_points = ((action.x, action.y),)
            elif atype == GUIActionType.DRAG:
                if (
                    action.x is None
                    or action.y is None
                    or action.x2 is None
                    or action.y2 is None
                ):
                    raise ValueError(
                        "GUIActionType.DRAG requires start (x, y) and destination (x2, y2) coordinates"
                    )
                pointer_points = (
                    (action.x, action.y),
                    (action.x2, action.y2),
                )
            if pointer_points:
                await _validate_pointer_coords(*pointer_points)
            elif atype == GUIActionType.TYPE and (
                action.x is not None or action.y is not None
            ):
                if action.x is None or action.y is None:
                    raise ValueError("Typing focus coordinates require both x and y")
                await _validate_pointer_coords((action.x, action.y))
            elif atype == GUIActionType.TYPE and actor == "sonic-agent":
                raise ValueError(
                    "Agent typing requires focus coordinates verified from the current desktop frame"
                )

            if atype in (GUIActionType.CLICK, GUIActionType.DOUBLE_CLICK):
                if action.button is not None and action.button not in (1, 2, 3):
                    raise ValueError(f"Unsupported mouse button: {action.button}")
                button = action.button if action.button in (1, 2, 3) else 1
                repeat = 2 if atype == GUIActionType.DOUBLE_CLICK else 1
                move_cmd = f"test -x /usr/local/bin/sonic_cursor.py || exit 126; python3 /usr/local/bin/sonic_cursor.py move {action.x} {action.y}"
                await exec_gui(f"export DISPLAY={disp}; {move_cmd} && xdotool click --repeat {repeat} {button}")

            elif atype == GUIActionType.RIGHT_CLICK:
                move_cmd = f"test -x /usr/local/bin/sonic_cursor.py || exit 126; python3 /usr/local/bin/sonic_cursor.py move {action.x} {action.y}"
                await exec_gui(f"export DISPLAY={disp}; {move_cmd} && xdotool click 3")

            elif atype == GUIActionType.MOVE:
                move_cmd = f"test -x /usr/local/bin/sonic_cursor.py || exit 126; python3 /usr/local/bin/sonic_cursor.py move {action.x} {action.y}"
                location = await exec_gui(
                    f"export DISPLAY={disp}; {move_cmd} && "
                    f"xdotool getmouselocation --shell"
                )
                actual_x = re.search(r"(?m)^X=(-?\d+)$", location)
                actual_y = re.search(r"(?m)^Y=(-?\d+)$", location)
                if (
                    actual_x is None
                    or actual_y is None
                    or (int(actual_x.group(1)), int(actual_y.group(1)))
                    != (action.x, action.y)
                ):
                    raise RuntimeError(
                        f"Cursor move verification failed: requested ({action.x}, {action.y}), "
                        f"observed ({actual_x.group(1) if actual_x else 'None'}, {actual_y.group(1) if actual_y else 'None'})"
                    )

            elif atype == GUIActionType.DRAG:
                sx, sy = action.x, action.y
                dx, dy = action.x2, action.y2
                cursor = "test -x /usr/local/bin/sonic_cursor.py || exit 126; python3 /usr/local/bin/sonic_cursor.py"
                await exec_gui(
                    f"export DISPLAY={disp}; {cursor} drag {sx} {sy} {dx} {dy}"
                )

            elif atype == GUIActionType.TYPE:
                if not action.text:
                    raise ValueError("GUIActionType.TYPE requires non-empty text")
                delay = max(0, int(action.delay_ms)) if action.delay_ms is not None else 25
                safe_text = shlex.quote(action.text)
                focus_prefix = ""
                if (
                    actor != "sonic-agent"
                    and action.x is not None
                    and action.y is not None
                ):
                    focus_prefix = f"test -x /usr/local/bin/sonic_cursor.py || exit 126; python3 /usr/local/bin/sonic_cursor.py click {action.x} {action.y} && sleep 0.05 && "
                await exec_gui(f"export DISPLAY={disp}; {focus_prefix}xdotool type --clearmodifiers --delay {delay} -- {safe_text}")

            elif atype in (GUIActionType.KEYPRESS, GUIActionType.HOTKEY) and (action.key or action.text):
                norm_key = self._normalize_x11_keysym(action.key or action.text or "")
                safe_key = shlex.quote(norm_key)
                await exec_gui(f"DISPLAY={disp} xdotool key --clearmodifiers {safe_key}")

            elif atype == GUIActionType.SCROLL:
                btn = 5 if action.scroll_delta < 0 else 4
                times = abs(action.scroll_delta) if action.scroll_delta != 0 else 3
                move_prefix = f"test -x /usr/local/bin/sonic_cursor.py || exit 126; python3 /usr/local/bin/sonic_cursor.py move {action.x} {action.y} && "
                await exec_gui(f"export DISPLAY={disp}; {move_prefix}xdotool click --repeat {times} {btn}")

            elif atype in (GUIActionType.OPEN_APP, GUIActionType.LAUNCH_APP) and action.app_name:
                launched = await self.launch_application(workspace_id, action.app_name, actor=actor)
                if not launched:
                    raise RuntimeError(f"GUI launch_application failed for '{action.app_name}'")

            elif atype == GUIActionType.CLOSE_WINDOW:
                target = action.window_id or action.app_name
                if target:
                    safe_target = shlex.quote(target)
                    await exec_gui(
                        f"export DISPLAY={disp}; "
                        f"wmctrl -c {safe_target} 2>/dev/null || "
                        f"wmctrl -ic {safe_target} 2>/dev/null || "
                        f"xdotool search --onlyvisible --name {safe_target} windowclose 2>/dev/null"
                    )
                else:
                    await exec_gui(f"DISPLAY={disp} xdotool getactivewindow windowclose")

            elif atype == GUIActionType.CLOSE_APP and action.app_name:
                await exec_gui(
                    f"export DISPLAY={disp}; "
                    f"wmctrl -c {shlex.quote(action.app_name)}"
                )

            elif atype == GUIActionType.INSTALL_APP and (action.app_name or action.text):
                pkg = action.app_name or action.text or ""
                ok, err = await self.install_application(workspace_id, pkg, actor=actor)
                if not ok:
                    raise RuntimeError(f"GUI install_application failed for '{pkg}': {err}")

            elif atype == GUIActionType.WAIT:
                await asyncio.sleep(max(0.1, (action.delay_ms or 500) / 1000.0))

            elif atype == GUIActionType.SELECT_WINDOW and (action.window_id or action.app_name):
                target = (action.window_id or action.app_name or "").strip()
                if target:
                    if target.lower().startswith("0x"):
                        focus_cmd = (
                            f"export DISPLAY={disp}; "
                            f"wmctrl -i -a {shlex.quote(target)} 2>/dev/null || "
                            f"xdotool windowactivate --sync {shlex.quote(target)} 2>/dev/null"
                        )
                    else:
                        focus_cmd = (
                            f"export DISPLAY={disp}; "
                            f"(wmctrl -a {shlex.quote(target)} 2>/dev/null || "
                            f"wmctrl -x -a {shlex.quote(target)} 2>/dev/null || "
                            f"xdotool search --onlyvisible --name {shlex.quote(target)} windowactivate 2>/dev/null || "
                            f"xdotool search --onlyvisible --class {shlex.quote(target)} windowactivate 2>/dev/null)"
                        )
                    focus_code, _, focus_err = await self._docker_exec(focus_cmd)
                    if focus_code != 0:
                        raise RuntimeError(
                            f"window focus failed for '{target}': {focus_err.strip()}"
                        )
                    if target.lower().startswith("0x"):
                        active_code, active_id, active_err = await self._docker_exec(
                            f"export DISPLAY={disp}; xdotool getactivewindow"
                        )
                        try:
                            focused = (
                                active_code == 0
                                and int(active_id.strip(), 0) == int(target, 16)
                            )
                        except ValueError:
                            focused = False
                    else:
                        active_code, active_title, active_err = await self._docker_exec(
                            f"export DISPLAY={disp}; "
                            "xdotool getactivewindow getwindowname"
                        )
                        focused = (
                            active_code == 0
                            and target.casefold() in active_title.strip().casefold()
                        )
                    if not focused:
                        raise RuntimeError(
                            f"window focus could not be verified for '{target}': "
                            f"{active_err.strip()}"
                        )
                    self._active_windows[workspace_id] = target

            elif atype in (GUIActionType.APP_MAXIMIZE, GUIActionType.WINDOW_MAXIMIZE):
                target = (action.window_id or action.app_name or "").strip()
                if target and target != ":ACTIVE:":
                    if target.lower().startswith("0x"):
                        max_cmd = f"export DISPLAY={disp}; wmctrl -i -r {shlex.quote(target)} -b add,maximized_vert,maximized_horz"
                    else:
                        max_cmd = (
                            f"export DISPLAY={disp}; "
                            f"wmctrl -r {shlex.quote(target)} -b add,maximized_vert,maximized_horz 2>/dev/null || "
                            f"wmctrl -x -r {shlex.quote(target)} -b add,maximized_vert,maximized_horz 2>/dev/null || "
                            f"wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz"
                        )
                else:
                    max_cmd = f"export DISPLAY={disp}; wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz"
                await exec_gui(max_cmd)
                if target and not target.lower().startswith("0x"):
                    self._active_windows[workspace_id] = target

            elif atype == GUIActionType.WINDOW_TILE:
                await self.tile_workstation(workspace_id)

            else:
                raise ValueError(f"Unsupported or incomplete GUI action: {action.action}")

            # Invalidate 2-second cache so fresh post-action screen is captured
            self._last_screenshot_time = 0.0
            await asyncio.sleep(0.3)
            return await self.screenshot(workspace_id)

    async def gui_action_state(
        self,
        workspace_id: str,
        action: GUIAction,
        actor: str = "operator",
    ) -> ScreenObservation:
        """Return a fresh AVCS-processed screenshot after GUI dispatch."""
        return await self.gui_action(
            workspace_id,
            action,
            actor=actor,
        )

    async def maximize_window(self, workspace_id: str, target: str | None = None) -> bool:
        """Maximize the active or targeted window using wmctrl."""
        disp = self._get_display(workspace_id)
        if target and target.strip() and target.strip() != ":ACTIVE:":
            safe = target.strip()
            if safe.lower().startswith("0x"):
                cmd = f"DISPLAY={disp} wmctrl -i -r {shlex.quote(safe)} -b add,maximized_vert,maximized_horz"
            else:
                cmd = (
                    f"DISPLAY={disp} wmctrl -r {shlex.quote(safe)} -b add,maximized_vert,maximized_horz 2>/dev/null || "
                    f"DISPLAY={disp} wmctrl -x -r {shlex.quote(safe)} -b add,maximized_vert,maximized_horz 2>/dev/null || "
                    f"DISPLAY={disp} wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz"
                )
        else:
            cmd = f"DISPLAY={disp} wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz"
        code, _, _ = await self._docker_exec(cmd, timeout=5)
        return code == 0

    async def tile_workstation(self, workspace_id: str) -> bool:
        """
        Executes wmctrl commands to tile windows side-by-side:
        - First discovered application window: left half
        - Second discovered application window: right half
        Discovers open windows dynamically via wmctrl while maintaining compatibility
        with direct wmctrl targeting.
        """
        disp = self._get_display(workspace_id)
        size_code, size_out, _ = await self._docker_exec(
            f"DISPLAY={disp} xdotool getdisplaygeometry", timeout=5
        )
        try:
            width, height = (int(value) for value in size_out.split())
        except (TypeError, ValueError):
            return False
        if size_code != 0 or width <= 1 or height <= 0:
            return False

        windows: list[tuple[str, str]] = []
        code_l, out_l, _ = await self._docker_exec(f"DISPLAY={disp} wmctrl -l 2>/dev/null", timeout=5)
        if code_l == 0 and out_l.strip():
            for line in out_l.splitlines():
                parts = line.split(maxsplit=3)
                if len(parts) >= 4:
                    win_id = parts[0]
                    title = parts[3].strip()
                    if title and title not in ("xfce4-panel", "Desktop"):
                        windows.append((win_id, title))

        code1 = 1
        code2 = 1
        client_windows = [w for w in windows if w[1] not in ("xfce4-panel", "Desktop", "desktop")]
        if client_windows:
            left_win = client_windows[0]
            unmax_code, _, _ = await self._docker_exec(
                f'DISPLAY={disp} wmctrl -i -r "{left_win[0]}" -b remove,maximized_vert,maximized_horz'
            )
            if unmax_code != 0:
                return False
            left_width = width // 2
            code1, _, _ = await self._docker_exec(
                f'DISPLAY={disp} wmctrl -i -r "{left_win[0]}" -e 0,0,0,{left_width},{height}'
            )
            if len(client_windows) > 1:
                right_win = client_windows[1]
                unmax_code, _, _ = await self._docker_exec(
                    f'DISPLAY={disp} wmctrl -i -r "{right_win[0]}" -b remove,maximized_vert,maximized_horz'
                )
                if unmax_code != 0:
                    return False
                right_width = width - left_width
                code2, _, _ = await self._docker_exec(
                    f'DISPLAY={disp} wmctrl -i -r "{right_win[0]}" '
                    f'-e 0,{left_width},0,{right_width},{height}'
                )

        if not client_windows:
            return False
        return code1 == 0 and (len(client_windows) < 2 or code2 == 0)

    async def settle_screen(
        self,
        workspace_id: str,
        max_wait: float = 2.0,
        interval: float = 0.3,
    ) -> ScreenObservation:
        """
        Captures screenshots with interval. When two consecutive screenshot
        hashes match (screen is static / loaded), returns the settled observation.
        """
        def _calc_hash(obs: ScreenObservation) -> str:
            if not obs.screenshot_base64:
                raise RuntimeError("Cannot settle the desktop without a valid pixel frame")
            payload = obs.screenshot_base64.encode("utf-8")
            return hashlib.md5(payload).hexdigest()

        start = asyncio.get_event_loop().time()
        self._last_screenshot_time = 0.0
        last_obs = await self.screenshot(workspace_id)
        last_hash = _calc_hash(last_obs)

        while (asyncio.get_event_loop().time() - start) < max_wait:
            await asyncio.sleep(interval)
            self._last_screenshot_time = 0.0
            curr_obs = await self.screenshot(workspace_id)
            curr_hash = _calc_hash(curr_obs)
            if curr_hash == last_hash:
                return curr_obs
            last_obs = curr_obs
            last_hash = curr_hash

        return last_obs

    # -------------------------------------------------------------
    # Terminal PTY Execution
    # -------------------------------------------------------------

    async def terminal(
        self,
        workspace_id: str,
        command: str,
        timeout: int = 60,
        actor: str = "operator",
    ) -> ExecResult:
        code, stdout, stderr = await self._docker_exec(command, timeout=timeout)
        # Auto-heal: If mkdir failed because a 0-byte empty file previously blocked directory creation
        if code != 0 and "cannot create directory" in (stderr or "") and "File exists" in (stderr or ""):
            m = re.search(r"cannot create directory ['\"]?([^'\":\n]+)['\"]?: File exists", stderr)
            if m:
                conflict_path = m.group(1).strip()
                cleanup_cmd = f"[ -f {shlex.quote(conflict_path)} ] && [ ! -s {shlex.quote(conflict_path)} ] && rm -f {shlex.quote(conflict_path)}"
                c_code, _, _ = await self._docker_exec(cleanup_cmd, timeout=10)
                if c_code == 0:
                    code, stdout, stderr = await self._docker_exec(command, timeout=timeout)
        return ExecResult(
            command=command,
            exit_code=code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=0.0,
            timed_out=(code == 124),
            sandbox_id=self.container_name,
        )

    async def execute(
        self,
        workspace_id: str,
        command: str | list[str],
        timeout: int = 60,
        actor: str = "operator",
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ExecResult:
        """Executes a command inside the container (ComputeProvider interface for SecurityTool)."""
        if isinstance(command, list):
            cmd_str = " ".join(shlex.quote(c) for c in command)
        else:
            cmd_str = str(command)
        if cwd:
            cmd_str = f"cd {shlex.quote(cwd)} && {cmd_str}"
        return await self.terminal(workspace_id, cmd_str, timeout=timeout, actor=actor)

    # -------------------------------------------------------------
    # Filesystem & Git Operations
    # -------------------------------------------------------------

    async def read_file(self, workspace_id: str, path: str) -> str:
        code, out, _ = await self._docker_exec(f"cat {shlex.quote(path)}", timeout=10)
        return out if code == 0 else ""

    async def write_file(self, workspace_id: str, path: str, content: str, actor: str = "operator") -> bool:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            await self._docker_exec(f"mkdir -p {shlex.quote(parent_dir)}")
        ws_dir = os.environ.get("SONIC_WORKSPACE_ROOT", "/home/sonic/workspace")
        docker_bin = _docker_binary() or "docker"
        proc = await asyncio.create_subprocess_exec(
            docker_bin, "exec", "-i", "-w", ws_dir, self.container_name, "sh", "-c", f"cat > {shlex.quote(path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=content.encode("utf-8"))
        return proc.returncode == 0

    async def list_files(self, workspace_id: str, path: str = ".") -> list[FileEntry]:
        code, out, _ = await self._docker_exec(f"ls -la {shlex.quote(path)} 2>/dev/null", timeout=10)
        entries: list[FileEntry] = []
        if code == 0 and out.strip():
            for line in out.splitlines()[1:]:  # skip 'total'
                parts = line.split(maxsplit=8)
                if len(parts) >= 9:
                    fname = parts[8].strip()
                    if fname not in (".", ".."):
                        is_dir = parts[0].startswith("d")
                        entries.append(
                            FileEntry(
                                path=f"{path}/{fname}".replace("//", "/"),
                                name=fname,
                                is_dir=is_dir,
                                size_bytes=int(parts[4]) if parts[4].isdigit() else 0,
                            )
                        )
        return entries

    async def git_action(
        self,
        workspace_id: str,
        action: str,
        **kwargs: Any,
    ) -> Any:
        if action == "status":
            code, out, _ = await self._docker_exec("git -C /home/sonic/workspace status --porcelain 2>/dev/null", timeout=15)
            bcode, bout, _ = await self._docker_exec("git -C /home/sonic/workspace branch --show-current 2>/dev/null", timeout=15)
            branch = bout.strip() if bcode == 0 and bout.strip() else ("uninitialized" if code != 0 else "main")
            clean = (code == 0) and (len(out.strip()) == 0)
            untracked = [line[3:] for line in out.splitlines() if line.startswith("??")] if code == 0 else []
            modified = [line[3:] for line in out.splitlines() if not line.startswith("??")] if code == 0 else []
            return GitStatusInfo(
                branch=branch,
                is_clean=clean,
                untracked_files=untracked,
                modified_files=modified,
                staged_files=[],
            )
        elif action == "commit":
            msg = kwargs.get("message", "chore: automated commit")
            code, _, _ = await self._docker_exec(f"git -C /home/sonic/workspace add -A && git -C /home/sonic/workspace commit -m {shlex.quote(str(msg))}", timeout=15)
            return code == 0
        elif action in ["branch", "checkout"]:
            branch_name = str(kwargs.get("branch_name") or kwargs.get("branch") or "main").lstrip("-") or "main"
            code, _, _ = await self._docker_exec(f"git -C /home/sonic/workspace checkout -B {shlex.quote(branch_name)}", timeout=15)
            return code == 0
        elif action == "diff":
            bcode, bout, _ = await self._docker_exec("git -C /home/sonic/workspace branch --show-current 2>/dev/null", timeout=15)
            branch = bout.strip() if bcode == 0 and bout.strip() else "HEAD"
            code, out, _ = await self._docker_exec(f"git -C /home/sonic/workspace diff {branch}", timeout=15)
            return out or "Working tree clean."
        args = kwargs.get("args") or []
        cmd = f"git -C /home/sonic/workspace {action} " + " ".join(shlex.quote(str(a)) for a in args)
        code, out, _ = await self._docker_exec(cmd, timeout=15)
        clean = "nothing to commit" in out.lower()
        branch = "main"
        if "on branch" in out.lower():
            for line in out.splitlines():
                if "on branch" in line.lower():
                    branch = line.split()[-1]
                    break
        return GitStatusInfo(branch=branch, is_clean=clean, modified_files=[])

    async def process_list(self, workspace_id: str) -> list[ProcessInfo]:
        code, out, _ = await self._docker_exec("ps -eo pid,user,%cpu,%mem,comm --no-headers 2>/dev/null | head -30", timeout=5)
        procs: list[ProcessInfo] = []
        if code == 0 and out.strip():
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    procs.append(
                        ProcessInfo(
                            pid=int(parts[0]) if parts[0].isdigit() else 0,
                            name=parts[4],
                            cpu_pct=float(parts[2]) if parts[2].replace(".", "").isdigit() else 0.0,
                            memory_mb=float(parts[3]) if parts[3].replace(".", "").isdigit() else 0.0,
                        )
                    )
        return procs

    async def install_application(self, workspace_id: str, package_name: str, actor: str = "operator") -> tuple[bool, str]:
        allowed, reason = self.app_policy.is_package_allowed(package_name)
        if not allowed:
            return False, reason
        self._cached_app_list = None
        code, out, err = await self._docker_exec(f"DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y {shlex.quote(package_name)}", timeout=180)
        return (code == 0), out if code == 0 else err

    async def uninstall_application(self, workspace_id: str, package_name: str, actor: str = "operator") -> bool:
        self._cached_app_list = None
        code, _, _ = await self._docker_exec(f"DEBIAN_FRONTEND=noninteractive apt-get remove -y {shlex.quote(package_name)}", timeout=120)
        return code == 0

    async def application_list(self, workspace_id: str) -> list[str]:
        now = asyncio.get_event_loop().time()
        cached = getattr(self, "_cached_app_list", None)
        if cached is not None and (now - cached[0]) < 60.0:
            return list(cached[1])

        std_utils = list(dict.fromkeys(self.app_policy.allowed_packages))
        utils_str = " ".join(std_utils)
        discovery_cmd = (
            "find /root/Desktop /home/sonic/Desktop /usr/share/applications /usr/local/share/applications ~/.local/share/applications -name '*.desktop' 2>/dev/null | while read -r f; do "
            "[ -f \"$f\" ] || continue; "
            "grep -qi '^NoDisplay=true' \"$f\" 2>/dev/null && continue; "
            "grep -qi '^Terminal=true' \"$f\" 2>/dev/null && continue; "
            "grep -qi 'TerminalEmulator' \"$f\" 2>/dev/null && continue; "
            "ex=$(grep -m1 -E '^Exec=' \"$f\" 2>/dev/null | cut -d= -f2- | awk '{print $1}'); "
            "[ -n \"$ex\" ] && basename \"$ex\"; "
            "b=$(basename \"$f\" .desktop); echo \"$b\"; "
            "done; "
            "printf '%s\\n' \"$PATH\" | tr ':' '\\n' | while read -r d; do "
            "[ -d \"$d\" ] || continue; "
            "find \"$d\" -maxdepth 1 \\( -type f -o -type l \\) -executable -printf '%f\\n' 2>/dev/null; "
            "done; "
            f"for b in {utils_str}; do command -v \"$b\" 2>/dev/null; done; true"
        )
        code, out, _ = await self._docker_exec(discovery_cmd, timeout=10)
        apps: list[str] = []
        if code == 0 and out.strip():
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                app = line.rsplit("/", 1)[-1].strip()
                if app.endswith(".desktop"):
                    app = app[:-8]
                app = app.strip("\"' ")
                if app and app not in apps:
                    apps.append(app)
        if apps:
            self._cached_app_list = (now, list(apps))
        return apps

    @staticmethod
    def _split_app_and_args(clean_app: str) -> tuple[str, list[str]]:
        """Separates an application command string into application query and arguments.
        Handles flags, URLs, file paths, and multi-word application names safely."""
        try:
            parts = shlex.split(clean_app)
        except Exception:
            parts = clean_app.split()
        if not parts:
            return "", []

        # Find the index of the first argument (flag, URL, or file path)
        arg_idx = None
        for idx, part in enumerate(parts):
            if idx == 0:
                continue
            if (
                part.startswith("-")
                or "://" in part
                or part.startswith("www.")
                or part.startswith("/")
                or part.startswith("./")
                or part.startswith("../")
                or part.startswith("~")
                or ("." in part and not part.endswith(".desktop"))
            ):
                arg_idx = idx
                break

        if arg_idx is not None and arg_idx > 0:
            app_part = " ".join(parts[:arg_idx])
            args_part = parts[arg_idx:]
            return app_part, args_part
        return clean_app, []


    async def _resolve_app_binary(self, app_query: str) -> str:
        """Dynamically resolve an application query to its executable binary on the system.
        Follows AGENTS.md Rule 1: Zero hardcoding. Resolves via PATH, POSIX alternatives,
        and .desktop entries discovered dynamically from the OS. Returns an empty string
        when discovery cannot verify a matching executable; it never guesses a binary."""
        app_clean = app_query.strip()
        if not app_clean:
            return ""

        # 1. If it's a direct executable on PATH or full path (must be >20 bytes so empty stubs are ignored)
        code, out, _ = await self._docker_exec(
            f"bin=$(which {shlex.quote(app_clean)} 2>/dev/null | head -n1); "
            f"if [ -n \"$bin\" ] && [ -f \"$bin\" ] && [ $(wc -c < \"$bin\" 2>/dev/null || echo 100) -gt 20 ]; then echo \"$bin\"; exit 0; fi; exit 1"
        )
        if code == 0 and out.strip():
            return out.strip().splitlines()[0]

        # 2. Try hyphenated version for multi-word display names
        if " " in app_clean:
            hyphenated = app_clean.replace(" ", "-").lower()
            code, out, _ = await self._docker_exec(f"which {shlex.quote(hyphenated)} 2>/dev/null")
            if code == 0 and out.strip():
                return out.strip().splitlines()[0]

        # 3. Dynamic lookup in .desktop files (/root/Desktop, /usr/share/applications, ~/.local/share/applications)
        search_value = shlex.quote(app_clean.lower())
        desktop_lookup_cmd = (
            f"for d in /root/Desktop/*.desktop /usr/share/applications/*.desktop ~/.local/share/applications/*.desktop; do "
            f"  [ -f \"$d\" ] || continue; "
            f"  grep -qi '^NoDisplay=true' \"$d\" 2>/dev/null && continue; "
            f"  grep -qi '^Terminal=true' \"$d\" 2>/dev/null && continue; "
            f"  name=$(grep -i '^Name=' \"$d\" 2>/dev/null | head -n1 | cut -d= -f2- | tr '[:upper:]' '[:lower:]'); "
            f"  gname=$(grep -i '^GenericName=' \"$d\" 2>/dev/null | head -n1 | cut -d= -f2- | tr '[:upper:]' '[:lower:]'); "
            f"  cats=$(grep -i '^Categories=' \"$d\" 2>/dev/null | head -n1 | cut -d= -f2- | tr '[:upper:]' '[:lower:]'); "
            f"  base=$(basename \"$d\" .desktop 2>/dev/null | tr '[:upper:]' '[:lower:]'); "
            f"  if echo \"$name $gname $cats $base\" | grep -Fqi -- {search_value}; then "
            f"    exec_line=$(grep '^Exec=' \"$d\" 2>/dev/null | head -n1 | cut -d= -f2-); "
            f"    bin=$(echo \"$exec_line\" | awk '{{print $1}}'); "
            f"    if which \"$bin\" >/dev/null 2>&1; then which \"$bin\"; exit 0; fi; "
            f"  fi; "
            f"done; "
            f"exit 1"
        )
        code, out, _ = await self._docker_exec(desktop_lookup_cmd)
        if code == 0 and out.strip():
            return out.strip().splitlines()[0]

        # 4. Fallback to POSIX standard alternatives if querying a generic role (browser, editor, terminal)
        role_lower = app_clean.lower()
        if any(role in role_lower for role in ("browser", "web")):
            for alt in ("x-www-browser", "sensible-browser", "xdg-open"):
                code, out, _ = await self._docker_exec(f"which {alt} 2>/dev/null")
                if code == 0 and out.strip():
                    return out.strip().splitlines()[0]
        elif any(role in role_lower for role in ("terminal", "console", "shell")):
            for alt in ("x-terminal-emulator", "sensible-terminal"):
                code, out, _ = await self._docker_exec(f"which {alt} 2>/dev/null")
                if code == 0 and out.strip():
                    return out.strip().splitlines()[0]
        elif any(role in role_lower for role in ("editor", "text", "notepad")):
            code, out, _ = await self._docker_exec(
                "for d in /usr/share/applications/*.desktop; do "
                "  [ -f \"$d\" ] || continue; "
                "  grep -qi '^Terminal=true' \"$d\" 2>/dev/null && continue; "
                "  if grep -qi 'TextEditor' \"$d\" 2>/dev/null; then "
                "    bin=$(grep '^Exec=' \"$d\" | head -n1 | cut -d= -f2- | awk '{print $1}'); "
                "    if which \"$bin\" >/dev/null 2>&1; then which \"$bin\"; exit 0; fi; "
                "  fi; "
                "done; exit 1"
            )
            if code == 0 and out.strip():
                return out.strip().splitlines()[0]

        return ""

    async def launch_application(self, workspace_id: str, app_name: str, actor: str = "operator") -> bool:
        clean_app = (app_name or "").strip()
        if not clean_app:
            return False

        app_query, args = self._split_app_and_args(clean_app)
        resolved_bin = await self._resolve_app_binary(app_query)
        if not resolved_bin:
            logger.info("application_launch_unresolved", app_query=app_query)
            return False

        if hasattr(self, "app_policy") and self.app_policy:
            allowed, _ = self.app_policy.is_package_allowed(resolved_bin)
            if not allowed:
                return False

        # In root container environments, applications with embedded rendering engines
        # require container flags to prevent unprivileged user namespace crashes.
        # H7 audit fix: No hardcoded application names (Rule 1). Dynamically probe
        # whether the resolved binary supports --no-sandbox.
        if "--no-sandbox" not in args:
            chk_cmd = (
                f"{shlex.quote(resolved_bin)} --help 2>&1 | grep -q -- '--no-sandbox'"
            )
            chk_code, _, _ = await self._docker_exec(chk_cmd)
            if chk_code == 0:
                args = ["--no-sandbox", "--disable-dev-shm-usage"] + args

        parts = [resolved_bin] + args
        disp = self._get_display(workspace_id)
        bin_base_str = os.path.basename(resolved_bin)
        bin_base = shlex.quote(bin_base_str)

        # If application is already open and no specific arguments/files were requested,
        # reuse and focus the existing window instead of spawning duplicate instances
        if not args:
            focus_existing = (
                f"DISPLAY={disp} wmctrl -x -a {bin_base} 2>/dev/null || "
                f"DISPLAY={disp} wmctrl -a {bin_base} 2>/dev/null"
            )
            fcode, _, _ = await self._docker_exec(focus_existing)
            if fcode == 0:
                self._active_windows[workspace_id] = bin_base_str
                self._last_screenshot = None
                self._last_screenshot_time = 0.0
                return True

        retry_args = list(args)
        for flag in reversed(("--no-sandbox", "--disable-dev-shm-usage")):
            if flag not in retry_args:
                retry_args.insert(0, flag)
        retry_spawn = (
            f"DISPLAY={disp} nohup {shlex.quote(resolved_bin)} "
            f"{' '.join(shlex.quote(arg) for arg in retry_args)} "
            ">/tmp/sonic_app_launch.log 2>&1 & "
        )
        spawn = (
            f"DISPLAY={disp} nohup {' '.join(shlex.quote(p) for p in parts)} >/tmp/sonic_app_launch.log 2>&1 & "
            f"app_pid=$!; sleep 0.4; "
            f"if ! kill -0 \"$app_pid\" 2>/dev/null && grep -q -- '--no-sandbox' /tmp/sonic_app_launch.log 2>/dev/null; then "
            f"  {retry_spawn}"
            f"  app_pid=$!; sleep 0.4; "
            f"fi; "
            f"kill -0 \"$app_pid\" 2>/dev/null || pgrep -f -- {bin_base} >/dev/null 2>&1"
        )
        code, _, _ = await self._docker_exec(spawn)
        if code == 0:
            self._active_windows[workspace_id] = os.path.basename(resolved_bin)
            self._last_screenshot = None
            self._last_screenshot_time = 0.0
            return True
        return False

    async def cleanup_desktop_windows(self, workspace_id: str) -> None:
        """Closes open application windows to restore desktop hygiene without killing panels or desktop."""
        disp = self._get_display(workspace_id)
        clean_cmd = (
            f"for wid in $(DISPLAY={disp} wmctrl -l 2>/dev/null | grep -v ' -1 ' | cut -d' ' -f1); do "
            f"  DISPLAY={disp} wmctrl -ic \"$wid\" 2>/dev/null; "
            f"done"
        )
        await self._docker_exec(clean_cmd)
        self._active_windows[workspace_id] = "Desktop"
        self._last_screenshot = None
        self._last_screenshot_time = 0.0

    async def close_application(self, workspace_id: str, app_name: str, actor: str = "operator") -> bool:
        action = GUIAction(action=GUIActionType.CLOSE_APP, app_name=app_name)
        await self.gui_action(workspace_id, action, actor=actor)
        return True

    async def service_action(self, workspace_id: str, service_name: str, action: str, actor: str = "operator") -> ServiceInfo:
        code, out, err = await self._docker_exec(f"service {shlex.quote(service_name)} {shlex.quote(action)} 2>&1", timeout=30)
        st = "RUNNING" if code == 0 and action in ("start", "restart") else action.upper()
        return ServiceInfo(
            name=service_name,
            status=st,
            port=0,
            logs=(out + err).splitlines(),
        )
