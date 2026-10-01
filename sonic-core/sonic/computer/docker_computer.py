"""
SONIC A-SEA — Docker Cyber Workstation Provider
================================================
First-class native Linux Workstation provider operating inside Docker
(sonic-desktop-workstation). Replaces cloud sandboxes (Daytona/E2B)
with a full, unrestricted, self-contained cyber environment:
- Desktop GUI: XFCE4, Xvfb :99 (1280x800x24), xdotool, wmctrl, ImageMagick
- Interactive Streaming: x11vnc (:5900) + noVNC websockify (:6080)
- Graphical desktop applications and an isolated operator workspace
- Zero cloud rate limits or connection resets.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any
import uuid
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


def _has_actionable_nsi_controls(controls: list[Any]) -> bool:
    for control in controls:
        if not isinstance(control, dict):
            continue
        name = str(control.get("name", "")).strip()
        bbox = control.get("bbox")
        if (
            not name
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        if control.get("source") == "atspi":
            return True
        if control.get("source") != "ocr" or control.get("role") == "window":
            continue
        try:
            confidence = float(control.get("confidence", 0))
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if (
            confidence >= 60
            and sum(character.isalpha() for character in name) >= 3
            and x2 > x1
            and y2 > y1
        ):
            return True
    return False


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
        self._last_screenshot: ScreenObservation | None = None
        self._last_screenshot_time: float = 0.0
        self._last_state_observation: ScreenObservation | None = None
        self._last_state_observed_at: float = 0.0
        self._nsi_frame_url: str | None = None
        self._nsi_frame_url_checked_at: float = 0.0

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
            Path(__file__).resolve().parents[3] / "docker-compose.yml",
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
        if (
            workspace_id == self._default_workspace_id
            or workspace_id.startswith("home-")
            or not self.workspaces
        ):
            self._workspace_owner = None
        return True

    async def close(self) -> None:
        """Release provider resources (no-op for statelessdocker CLI wrapper)."""

        return None

    async def get_vnc_url(self, workspace_id: str) -> str | None:
        # Fail-closed: a stream URL is only real when the container actually
        # runs and a VNC/noVNC listener is reachable on the configured port.
        if not await self._container_is_running():
            return None
        port = os.environ.get("SONIC_WORKSTATION_VNC_PORT", "6080")
        code, out, _ = await self._docker_exec(
            f"if (exec 3<>/dev/tcp/127.0.0.1/{port}) 2>/dev/null; then echo 0; else echo 1; fi",
            timeout=8,
        )
        if code == 0 and out.strip() == "0":
            return f"http://localhost:{port}/vnc.html"
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
    # NSI (Neural-Symbolic Instant Eye) Persistent 8-15ms Retina Pipe
    # -------------------------------------------------------------

    _NSI_DAEMON_SCRIPT = r"""
import base64, hashlib, io, json, os, signal, subprocess, sys, threading, time, urllib.request
def _detect_display():
    d = os.environ.get("DISPLAY", "").strip()
    if d:
        return d
    try:
        xs = sorted(f[1:] for f in os.listdir("/tmp/.X11-unix") if f.startswith("X") and f[1:].isdigit())
        if xs:
            return ":" + xs[-1]
    except Exception:
        pass
    return ":0"
_DISP = _detect_display()
os.environ["DISPLAY"] = _DISP
try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.b64 = ""
        self.active_window = "Desktop"
        self.open_windows = []
        self.visible_text = ""
        self.detected_controls = []
        self.files = []
        self.running_processes = []
        self.installed_applications = []
        self.working_directory = "/home/sonic/workspace"
        self.capture_timestamp_ns = 0
        self.controls_timestamp_ns = 0
        self.changed_region = None
        self.version = 0
        self.tile_hashes = {}
        self.cached_ocr_controls = []
        self.cached_ocr_lines = []
        self.ocr_error = ""
        self.ocr_retry_after = 0.0
        self.ocr_retry_delay = 1.0
        self.last_env_ts = 0.0
        self.last_apps_ts = 0.0

S = _State()

def _run(cmd, timeout=2.0):
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            return 124, stdout, stderr or f"Command timed out after {timeout} seconds"
        return 124, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
        return 1, "", str(e)

def _grab():
    if _HAS_MSS and _HAS_PIL:
        try:
            with mss.MSS() if hasattr(mss, "MSS") else mss.mss() as sct:
                raw = sct.grab(sct.monitors[0])
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                buf = io.BytesIO()
                img.save(buf, format="PNG", compress_level=1)
                return img, buf.getvalue()
        except Exception:
            pass
    p = "/dev/shm/sonic_nsi_live.png"
    _run(f"DISPLAY={_DISP} scrot -p -o {p} 2>/dev/null", timeout=1.5)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                b = f.read()
            return (Image.open(io.BytesIO(b)).convert("RGB") if _HAS_PIL else None), b
        except Exception:
            pass
    return None, b""

def _dirty_bbox(img):
    if img is None:
        return True, (0, 0, 1280, 800), {}
    w, h = img.size
    small = img.resize((w // 4, h // 4))
    sw, sh = small.size
    raw = small.tobytes()
    stride = sw * 3
    nh, dxs, dys = {}, [], []
    for gy in range(0, sh, 16):
        for gx in range(0, sw, 16):
            i1 = gy * stride + gx * 3
            i2 = min(len(raw) - 12, (gy + 8) * stride + gx * 3)
            th = hashlib.md5(raw[i1:i1 + 48] + raw[i2:i2 + 48]).digest()
            k = (gx // 16, gy // 16)
            nh[k] = th
            if S.tile_hashes.get(k) != th:
                tx, ty = k[0] * 64, k[1] * 64
                dxs.extend([tx, min(w, tx + 64)])
                dys.extend([ty, min(h, ty + 64)])
    if not dxs:
        return False, None, nh
    return True, (max(0, min(dxs) - 16), max(0, min(dys) - 16), min(w, max(dxs) + 16), min(h, max(dys) + 16)), nh

def _delta_ocr(img, png_bytes, bbox):
    if bbox is None:
        return S.cached_ocr_controls, S.cached_ocr_lines, True, ""
    x1, y1, x2, y2 = bbox
    cp = "/dev/shm/sonic_nsi_crop.png"
    try:
        if img is not None and (x2 - x1 < img.size[0] or y2 - y1 < img.size[1]):
            img.crop((x1, y1, x2, y2)).save(cp, format="PNG", compress_level=1)
        else:
            x1, y1 = 0, 0
            with open(cp, "wb") as f:
                f.write(png_bytes)
    except Exception as exc:
        return S.cached_ocr_controls, S.cached_ocr_lines, False, str(exc)
    code, out, err = _run(f"tesseract {cp} stdout --psm 11 tsv", timeout=5.0)
    succeeded = code == 0 and bool(out.strip())
    error = "" if succeeded else (
        err.strip() or f"Tesseract exited with status {code} without TSV output"
    )
    kept = [c for c in S.cached_ocr_controls if not (x1 <= (c["bbox"][0] + c["bbox"][2]) // 2 <= x2 and y1 <= (c["bbox"][1] + c["bbox"][3]) // 2 <= y2)]
    added = []
    if succeeded:
        for row in out.splitlines()[1:]:
            cols = row.split("\t")
            if len(cols) >= 12 and cols[11].strip():
                try:
                    conf, wx, wy, ww, wh = float(cols[10]), int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9])
                    if conf >= 25.0 and ww > 2 and wh > 2:
                        added.append({
                            "name": cols[11].strip(),
                            "role": "text",
                            "bbox": [x1 + wx, y1 + wy, x1 + wx + ww, y1 + wy + wh],
                            "bbox_format": "xyxy",
                            "source": "ocr",
                            "confidence": conf,
                        })
                except Exception:
                    pass
    merged = kept + added
    merged.sort(key=lambda c: (c["bbox"][1] // 18, c["bbox"][0]))
    lmap = {}
    for c in merged:
        lmap.setdefault(c["bbox"][1] // 18, []).append(c["name"])
    lines = [" ".join(w) for _, w in sorted(lmap.items())[:45]]
    return merged, lines, succeeded, error

def _atspi_and_cdp():
    ctrls, page_txt = [], ""
    try:
        tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=0.12).read().decode("utf-8", errors="replace"))
        for t in tabs:
            if t.get("type") == "page" and not str(t.get("url", "")).startswith("devtools://"):
                page_txt = f"Browser Tab: {t.get('title', '')} ({t.get('url', '')})"
                break
    except Exception:
        pass
    try:
        bus_num = _DISP.lstrip(":").split(".")[0]
        bus_sock = f"/root/.cache/at-spi/bus_{bus_num}"
        if not os.path.exists(bus_sock):
            _run(f"DISPLAY={_DISP} /usr/libexec/at-spi-bus-launcher --launch-immediately >/dev/null 2>&1 &", timeout=0.4)
        if os.path.exists(bus_sock):
            import gi
            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi
            d = Atspi.get_desktop(0)
            for i in range(min(d.get_child_count(), 10)):
                app = d.get_child_at_index(i)
                if not app: continue
                st, vis = [(app, 0)], 0
                while st and vis < 100:
                    n, dp = st.pop()
                    vis += 1
                    if dp > 6 or not n: continue
                    try:
                        rn, nm = (n.get_role_name() or "").lower(), (n.get_name() or "").strip()
                        if nm and any(r in rn for r in ("button", "menu", "entry", "text", "tab", "link")):
                            cp = n.get_component_iface()
                            if cp:
                                rc = cp.get_extents(Atspi.CoordType.SCREEN)
                                if rc and rc.width > 4 and rc.height > 4 and rc.x >= 0 and rc.y >= 0:
                                    ctrls.append({"name": nm, "role": rn, "bbox": [int(rc.x), int(rc.y), int(rc.x + rc.width), int(rc.y + rc.height)], "bbox_format": "xyxy", "source": "atspi"})
                        for ci in range(min(n.get_child_count(), 16)):
                            ch = n.get_child_at_index(ci)
                            if ch: st.append((ch, dp + 1))
                    except Exception:
                        pass
    except Exception:
        pass
    return ctrls, page_txt

def _fast_retina_worker():
    while True:
        try:
            img, png_b = _grab()
            if png_b:
                S.capture_timestamp_ns = time.time_ns()
            _, wout, _ = _run(f"(DISPLAY={_DISP} xdotool getactivewindow getwindowname 2>/dev/null || echo Desktop); echo '___W___'; DISPLAY={_DISP} wmctrl -lG 2>/dev/null", timeout=0.8)
            aw, ows, wctrls = "Desktop", [], []
            if "___W___" in wout:
                ap, op = wout.split("___W___", 1)
                aw = ap.strip() or "Desktop"
                for ln in op.splitlines():
                    lp = ln.strip().split(maxsplit=7)
                    if len(lp) >= 8 and lp[7] not in ("xfce4-panel", "Desktop", "desktop"):
                        ows.append(lp[7])
                        try:
                            wx, wy, ww, wh = int(lp[2]), int(lp[3]), int(lp[4]), int(lp[5])
                            x0, y0 = max(0, wx), max(0, wy)
                            wctrls.append({"name": lp[7], "role": "window", "bbox": [x0, y0, max(0, wx + ww), max(0, wy + wh)], "bbox_format": "xyxy", "source": "x11"})
                        except Exception:
                            pass
            b64 = base64.b64encode(png_b).decode("ascii") if png_b else ""
            S.latest_img = img
            S.latest_png = png_b
            with S.lock:
                if b64:
                    S.b64 = b64
                S.active_window = aw
                S.open_windows = ows
                S.wctrls = wctrls
                octrls = list(S.cached_ocr_controls)
                olines = list(S.cached_ocr_lines)
                actrls = list(getattr(S, "cached_atspi_controls", []))
                ptxt = getattr(S, "cached_page_txt", "")
                vparts = []
                if aw and aw != "Desktop": vparts.append(f"Active Window: {aw}")
                if ows: vparts.append(f"Open Windows: {', '.join(ows)}")
                if ptxt: vparts.append(ptxt)
                if olines: vparts.append("Screen OCR Text: " + " | ".join(olines))
                S.visible_text = "\n".join(vparts)
                S.detected_controls = actrls + wctrls + octrls
                S.controls_timestamp_ns = time.time_ns()
                S.version += 1
                width, height = img.size if img is not None else (0, 0)
                pkt = {
                    "ok": True,
                    "version": S.version,
                    "ts": time.time(),
                    "capture_timestamp_ns": S.capture_timestamp_ns or None,
                    "controls_timestamp_ns": S.controls_timestamp_ns or None,
                    "capture_age_ms": (
                        max(0.0, (time.time_ns() - S.capture_timestamp_ns) / 1_000_000)
                        if S.capture_timestamp_ns
                        else None
                    ),
                    "controls_age_ms": (
                        max(0.0, (time.time_ns() - S.controls_timestamp_ns) / 1_000_000)
                        if S.controls_timestamp_ns
                        else None
                    ),
                    "width": width,
                    "height": height,
                    "changed_region": S.changed_region,
                    "b64": S.b64,
                    "active_window": S.active_window,
                    "open_windows": S.open_windows,
                    "visible_text": S.visible_text,
                    "detected_controls": S.detected_controls,
                    "ocr_error": S.ocr_error or None,
                    "files": S.files,
                    "running_processes": S.running_processes,
                    "installed_applications": getattr(S, "installed_applications", []),
                    "working_directory": S.working_directory,
                }
                state_pkt = {key: value for key, value in pkt.items() if key != "b64"}
            tmp_f = "/dev/shm/sonic_nsi_frame.tmp"
            shm_f = "/dev/shm/sonic_nsi_frame.json"
            novnc_f = "/usr/share/novnc/nsi_frame.json"
            state_tmp_f = "/dev/shm/sonic_nsi_state.tmp"
            state_shm_f = "/dev/shm/sonic_nsi_state.json"
            state_novnc_f = "/usr/share/novnc/nsi_state.json"
            try:
                with open(tmp_f, "w") as f:
                    json.dump(pkt, f)
                os.replace(tmp_f, shm_f)
                with open(state_tmp_f, "w") as f:
                    json.dump(state_pkt, f, separators=(",", ":"))
                os.replace(state_tmp_f, state_shm_f)
                if os.path.isdir("/usr/share/novnc") and not os.path.islink(novnc_f):
                    try:
                        if os.path.exists(novnc_f):
                            os.remove(novnc_f)
                        os.symlink(shm_f, novnc_f)
                    except Exception:
                        pass
                if os.path.isdir("/usr/share/novnc") and not os.path.islink(state_novnc_f):
                    try:
                        if os.path.exists(state_novnc_f):
                            os.remove(state_novnc_f)
                        os.symlink(state_shm_f, state_novnc_f)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        time.sleep(0.05)

def _ocr_and_semantic_worker():
    while True:
        try:
            img = getattr(S, "latest_img", None)
            png_b = getattr(S, "latest_png", b"")
            if png_b:
                chg, dbox, nh = _dirty_bbox(img)
                with S.lock:
                    S.changed_region = list(dbox) if chg and dbox is not None else None
                if chg and time.monotonic() >= S.ocr_retry_after:
                    octrls, olines, ocr_succeeded, ocr_error = _delta_ocr(
                        img, png_b, dbox
                    )
                    with S.lock:
                        S.cached_ocr_controls = octrls
                        S.cached_ocr_lines = olines
                        S.ocr_error = ocr_error
                        if ocr_succeeded:
                            S.tile_hashes = nh
                            S.ocr_retry_after = 0.0
                            S.ocr_retry_delay = 1.0
                        else:
                            S.ocr_retry_after = (
                                time.monotonic() + S.ocr_retry_delay
                            )
                            S.ocr_retry_delay = min(S.ocr_retry_delay * 2, 15.0)
            actrls, ptxt = _atspi_and_cdp()
            with S.lock:
                S.cached_atspi_controls = actrls
                S.cached_page_txt = ptxt
            now = time.time()
            if now - S.last_env_ts > 1.5:
                S.last_env_ts = now
                wd = "/home/sonic/workspace" if os.path.exists("/home/sonic/workspace") else "/workspace"
                try: S.files = sorted(os.listdir(wd))[:80]
                except Exception: S.files = []
                _, pout, _ = _run("ps -eo comm --no-headers 2>/dev/null | sort -u | head -35", timeout=1.0)
                S.running_processes = [p.strip() for p in pout.splitlines() if p.strip()]
                S.working_directory = wd
            if now - S.last_apps_ts > 60.0 or not getattr(S, "installed_applications", None):
                S.last_apps_ts = now
                _, aout, _ = _run(
                    "for f in /usr/share/applications/*.desktop; do "
                    "[ -f \"$f\" ] || continue; "
                    "grep -qi '^NoDisplay=true' \"$f\" 2>/dev/null && continue; "
                    "grep -qi '^Terminal=true' \"$f\" 2>/dev/null && continue; "
                    "grep -qi 'TerminalEmulator' \"$f\" 2>/dev/null && continue; "
                    "basename \"$f\" .desktop; done",
                    timeout=2.0,
                )
                S.installed_applications = list(dict.fromkeys([a.strip() for a in aout.splitlines() if a.strip()]))
        except Exception:
            pass
        time.sleep(0.08)

threading.Thread(target=_fast_retina_worker, daemon=True).start()
threading.Thread(target=_ocr_and_semantic_worker, daemon=True).start()
while True:
    time.sleep(10.0)
"""

    async def _resolve_nsi_frame_url(self) -> str | None:
        """Resolve this container's published noVNC port without crossing desktop sessions."""
        now = asyncio.get_event_loop().time()
        if now - self._nsi_frame_url_checked_at < 5.0:
            return self._nsi_frame_url

        self._nsi_frame_url_checked_at = now
        docker = _docker_binary()
        if not docker:
            logger.warning("nsi_frame_unavailable", reason="docker_binary_not_found")
            return None
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    docker,
                    "inspect",
                    "--format",
                    "{{json .NetworkSettings.Ports}}",
                    self.container_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "nsi_frame_port_lookup_failed",
                container=self.container_name,
                error=str(exc),
            )
            return None

        if result.returncode != 0:
            logger.warning(
                "nsi_frame_port_lookup_failed",
                container=self.container_name,
                exit_code=result.returncode,
                error=result.stderr.decode("utf-8", errors="replace").strip()[:300],
            )
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
                logger.warning(
                    "nsi_frame_port_unavailable",
                    container=self.container_name,
                    reason="no_localhost_6080_binding",
                )
                self._nsi_frame_url = None
                return None
            self._nsi_frame_url = (
                f"http://127.0.0.1:{binding['HostPort']}/nsi_frame.json"
            )
            return self._nsi_frame_url
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "nsi_frame_port_lookup_invalid",
                container=self.container_name,
                error=str(exc),
            )
            return None

    async def _resolve_nsi_state_url(self) -> str | None:
        frame_url = await self._resolve_nsi_frame_url()
        if frame_url is None:
            return None
        return frame_url.rsplit("/", 1)[0] + "/nsi_state.json"

    async def _nsi_request(self, payload: dict[str, Any], timeout: float = 2.5) -> dict[str, Any] | None:
        """Read the requested frame, state, or X11 receipt from this workstation."""
        if "_docker_exec" in self.__dict__:
            return None
        command = payload.get("cmd")
        if command not in {"frame", "state"}:
            return None
        nsi_url = (
            await self._resolve_nsi_state_url()
            if command == "state"
            else await self._resolve_nsi_frame_url()
        )
        if nsi_url is None:
            return None
        if not hasattr(self, "_nsi_http"):
            import httpx
            self._nsi_http = httpx.AsyncClient(timeout=timeout, trust_env=False)
            self._nsi_booted = False
        try:
            resp = await self._nsi_http.get(nsi_url)
            if resp.status_code == 200:
                data = resp.json()
                valid_payload = data.get("ok") and (
                    (
                        command == "state"
                        and isinstance(data.get("version"), int)
                        and not isinstance(data.get("version"), bool)
                    )
                    or (command == "frame" and bool(data.get("b64")))
                )
                if valid_payload:
                    cache_name = "_last_nsi_state" if command == "state" else "_last_nsi_frame"
                    setattr(self, cache_name, (asyncio.get_event_loop().time(), data))
                    return data
                logger.warning(
                    "nsi_frame_response_invalid",
                    container=self.container_name,
                    status_code=resp.status_code,
                )
        except Exception as exc:
            logger.debug(
                "nsi_frame_request_failed",
                container=self.container_name,
                error=str(exc),
            )

        # If daemon isn't running yet inside container, start it once in background
        if not getattr(self, "_nsi_booted", False):
            self._nsi_booted = True
            try:
                import base64
                b64_script = base64.b64encode(self._NSI_DAEMON_SCRIPT.encode("utf-8")).decode("ascii")
                boot_cmd = (
                    f"printf '%s' '{b64_script}' | base64 -d > /tmp/sonic_nsi_daemon.py && "
                    "daemon_pid=$(cat /tmp/sonic_nsi_daemon.pid 2>/dev/null || printf '') && "
                    "if [ -z \"$daemon_pid\" ] || ! kill -0 \"$daemon_pid\" 2>/dev/null; then "
                    "rm -f /dev/shm/sonic_nsi_frame.json /dev/shm/sonic_nsi_state.json; "
                    "nohup python3 /tmp/sonic_nsi_daemon.py "
                    "</dev/null >/tmp/sonic_nsi.log 2>&1 & "
                    "printf '%s\\n' \"$!\" > /tmp/sonic_nsi_daemon.pid; "
                    "fi; "
                    "for _ in $(seq 1 40); do "
                    "[ -s /dev/shm/sonic_nsi_frame.json ] && break; sleep 0.05; done; "
                    "[ -s /dev/shm/sonic_nsi_frame.json ]"
                )
                boot_code, _, boot_err = await self._docker_exec(boot_cmd, timeout=6)
                if boot_code != 0:
                    logger.warning(
                        "nsi_daemon_boot_failed",
                        container=self.container_name,
                        exit_code=boot_code,
                        error=boot_err[:300],
                    )
                    return None
                nsi_url = (
                    await self._resolve_nsi_state_url()
                    if command == "state"
                    else await self._resolve_nsi_frame_url()
                )
                if nsi_url is None:
                    return None
                resp = await self._nsi_http.get(nsi_url)
                if resp.status_code == 200:
                    data = resp.json()
                    valid_payload = data.get("ok") and (
                        (
                            command == "state"
                            and isinstance(data.get("version"), int)
                            and not isinstance(data.get("version"), bool)
                        )
                        or (command == "frame" and bool(data.get("b64")))
                    )
                    if valid_payload:
                        cache_name = "_last_nsi_state" if command == "state" else "_last_nsi_frame"
                        setattr(self, cache_name, (asyncio.get_event_loop().time(), data))
                        return data
            except Exception:
                pass
        return None

    # -------------------------------------------------------------
    # Multi-Modal Perception: Screen Observation
    # -------------------------------------------------------------

    async def observe_state(self, workspace_id: str) -> ScreenObservation | None:
        """Read the live compact NSI state without transferring screenshot pixels."""
        data = await self._nsi_request({"cmd": "state"}, timeout=0.5)
        if not data or not data.get("ok"):
            return None
        version = data.get("version")
        width, height = data.get("width"), data.get("height")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version <= 0
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            logger.warning(
                "nsi_state_response_invalid",
                container=self.container_name,
                version=version,
                width=width,
                height=height,
            )
            return None

        capture_timestamp_ns = data.get("capture_timestamp_ns")
        if (
            not isinstance(capture_timestamp_ns, int)
            or isinstance(capture_timestamp_ns, bool)
            or capture_timestamp_ns <= 0
        ):
            logger.warning(
                "nsi_state_timestamp_invalid",
                container=self.container_name,
                version=version,
            )
            return None

        now_ns = time.time_ns()
        capture_age_ms = (now_ns - capture_timestamp_ns) / 1_000_000
        controls_timestamp_ns = data.get("controls_timestamp_ns")
        controls_age_ms = (
            (now_ns - controls_timestamp_ns) / 1_000_000
            if isinstance(controls_timestamp_ns, int)
            and not isinstance(controls_timestamp_ns, bool)
            and controls_timestamp_ns > 0
            else None
        )
        if (
            capture_age_ms < 0
            or capture_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
            or controls_age_ms is None
            or controls_age_ms < 0
            or controls_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
        ):
            logger.warning(
                "nsi_state_stale",
                container=self.container_name,
                version=version,
                capture_age_ms=round(capture_age_ms, 1),
                controls_age_ms=(
                    round(controls_age_ms, 1) if controls_age_ms is not None else None
                ),
            )
            return None

        raw_changed_region = data.get("changed_region")
        changed_region = None
        if isinstance(raw_changed_region, (list, tuple)) and len(raw_changed_region) == 4:
            try:
                candidate = tuple(int(value) for value in raw_changed_region)
                x1, y1, x2, y2 = candidate
                if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
                    changed_region = candidate
            except (TypeError, ValueError):
                logger.warning(
                    "nsi_state_changed_region_invalid",
                    container=self.container_name,
                    version=version,
                )

        observation = ScreenObservation(
            screenshot_base64="",
            capture_timestamp_ns=capture_timestamp_ns,
            capture_age_ms=capture_age_ms,
            controls_age_ms=controls_age_ms,
            perception_version=version,
            changed_region=changed_region,
            width=width,
            height=height,
            active_window=str(data.get("active_window") or "Desktop"),
            visible_text=str(data.get("visible_text") or ""),
            detected_controls=list(data.get("detected_controls") or []),
            desktop_state="STATE_ONLY",
        )
        self._last_state_observation = observation
        self._last_state_observed_at = asyncio.get_event_loop().time()
        return observation

    async def screenshot(self, workspace_id: str) -> ScreenObservation:
        now = asyncio.get_event_loop().time()
        nsi_data = await self._nsi_request({"cmd": "frame"}, timeout=2.5)
        nsi_b64 = (nsi_data or {}).get("b64")
        nsi_dimensions = _validated_png_dimensions(nsi_b64)
        nsi_controls = (
            list(nsi_data.get("detected_controls") or [])
            if nsi_data and nsi_data.get("ok") and nsi_dimensions is not None
            else []
        )
        has_actionable_nsi_controls = _has_actionable_nsi_controls(nsi_controls)
        if nsi_data and nsi_data.get("ok") and nsi_dimensions is not None:
            nsi_ocr_error = nsi_data.get("ocr_error")
            if nsi_ocr_error:
                logger.warning(
                    "nsi_frame_ocr_failed",
                    container=self.container_name,
                    error=str(nsi_ocr_error)[:300],
                )
            capture_timestamp_ns = nsi_data.get("capture_timestamp_ns")
            if (
                not isinstance(capture_timestamp_ns, int)
                or isinstance(capture_timestamp_ns, bool)
                or capture_timestamp_ns < 0
            ):
                capture_timestamp_ns = None
            capture_age_ms = nsi_data.get("capture_age_ms")
            if isinstance(capture_timestamp_ns, int) and not isinstance(capture_timestamp_ns, bool):
                age_ns = time.time_ns() - capture_timestamp_ns
                capture_age_ms = (
                    max(capture_age_ms, age_ns / 1_000_000)
                    if isinstance(capture_age_ms, (int, float))
                    and not isinstance(capture_age_ms, bool)
                    and capture_age_ms >= 0
                    and age_ns >= 0
                    else age_ns / 1_000_000 if age_ns >= 0 else None
                )
            elif (
                not isinstance(capture_age_ms, (int, float))
                or isinstance(capture_age_ms, bool)
                or capture_age_ms < 0
            ):
                capture_age_ms = None
            nsi_frame_is_fresh = (
                capture_age_ms is None
                or capture_age_ms <= _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
            )
            if has_actionable_nsi_controls and nsi_frame_is_fresh:
                obs = ScreenObservation(
                    screenshot_base64=nsi_b64,
                    capture_timestamp_ns=capture_timestamp_ns,
                    capture_age_ms=capture_age_ms,
                    width=nsi_dimensions[0],
                    height=nsi_dimensions[1],
                    active_window=nsi_data.get("active_window") or "Desktop",
                    visible_text=nsi_data.get("visible_text") or "",
                    detected_controls=nsi_controls,
                    desktop_state="INTERACTIVE",
                )
                self._last_screenshot = obs
                self._last_screenshot_time = now
                return obs
            logger.info(
                "nsi_frame_using_ocr_fallback",
                active_window=nsi_data.get("active_window") or "Desktop",
                detected_control_count=len(nsi_controls),
                actionable_controls_found=has_actionable_nsi_controls,
                capture_age_ms=capture_age_ms,
                capture_fresh=nsi_frame_is_fresh,
            )
        elif nsi_data:
            logger.warning(
                "nsi_frame_invalid_image_using_ocr_fallback",
                active_window=nsi_data.get("active_window") or "Desktop",
            )

        disp = self._get_display(workspace_id)
        tmp_id = uuid.uuid4().hex[:8]
        tmp_path = f"/tmp/sonic_screen_{tmp_id}.png"
        scr_cmd = (
            f"DISPLAY={disp} scrot -p -o {tmp_path} 2>/dev/null || "
            f"DISPLAY={disp} import -window root {tmp_path} 2>/dev/null; "
            "echo '___CAPTURE_TIMESTAMP_NS___'; date +%s%N; "
            f"base64 -w0 {tmp_path} 2>/dev/null && echo '___OCR_TSV___'; "
            f"tesseract {tmp_path} stdout --psm 11 tsv; ocr_exit_11=$?; "
            f"tesseract {tmp_path} stdout --psm 6 tsv; ocr_exit_6=$?; "
            "echo '___OCR_EXIT___'; printf '%s,%s\\n' \"$ocr_exit_11\" \"$ocr_exit_6\"; "
            f"rm -f {tmp_path}; "
            f"echo '___ACTIVE_WINDOW___' && "
            f"(DISPLAY={disp} xdotool getactivewindow getwindowname 2>/dev/null || echo 'Desktop') && "
            f"echo '___OPEN_WINDOWS___' && "
            f"DISPLAY={disp} wmctrl -l 2>/dev/null"
        )
        code, out, err = await self._docker_exec(scr_cmd, timeout=15)
        if code != 0:
            logger.warning(
                "desktop_screenshot_capture_command_failed",
                exit_code=code,
            )
        b64 = ""
        ocr_tsvs: list[str] = []
        ocr_exit_codes: list[int] = []
        capture_timestamp_ns = None
        active_win = "Desktop"
        open_wins: list[str] = []
        if out:
            parts = out.split("___ACTIVE_WINDOW___", 1)
            first_chunk = parts[0]
            if "___CAPTURE_TIMESTAMP_NS___" in first_chunk:
                _, timestamp_tail = first_chunk.split("___CAPTURE_TIMESTAMP_NS___", 1)
                timestamp_text, _, image_text = timestamp_tail.lstrip("\r\n").partition("\n")
                try:
                    capture_timestamp_ns = int(timestamp_text.strip())
                except ValueError:
                    capture_timestamp_ns = None
                if capture_timestamp_ns is not None and capture_timestamp_ns < 0:
                    capture_timestamp_ns = None
                first_chunk = image_text
            if "___OCR_TSV___" in first_chunk:
                b64_part, ocr_output = first_chunk.split("___OCR_TSV___", 1)
                b64 = b64_part.strip()
            else:
                ocr_output = ""
                b64 = first_chunk.strip()
            if "___OCR_EXIT___" in ocr_output:
                ocr_output, ocr_status = ocr_output.split("___OCR_EXIT___", 1)
                try:
                    ocr_exit_codes = [
                        int(value)
                        for value in ocr_status.strip().splitlines()[0].split(",")
                    ]
                except (ValueError, IndexError):
                    ocr_exit_codes = []
            ocr_tsvs = [ocr_output.strip()] if ocr_output.strip() else []
            rest = parts[1] if len(parts) > 1 else ""
            if "___OPEN_WINDOWS___" in rest:
                win_parts = rest.split("___OPEN_WINDOWS___", 1)
                active_win = win_parts[0].strip() or "Desktop"
                for line in win_parts[1].splitlines():
                    lp = line.strip().split(maxsplit=3)
                    if len(lp) >= 4 and lp[3] not in ("xfce4-panel", "Desktop", "desktop"):
                        open_wins.append(lp[3])
            else:
                active_win = rest.strip() or "Desktop"

        if ocr_exit_codes and all(code != 0 for code in ocr_exit_codes):
            logger.warning(
                "desktop_screenshot_ocr_failed",
                exit_codes=ocr_exit_codes,
                error=err.strip(),
            )
        elif any(code != 0 for code in ocr_exit_codes):
            logger.warning(
                "desktop_screenshot_ocr_partial_failure",
                exit_codes=ocr_exit_codes,
                error=err.strip(),
            )

        detected_controls: list[dict[str, Any]] = []
        ocr_candidates: list[dict[str, Any]] = []
        for ocr_tsv in ocr_tsvs:
            for row in ocr_tsv.splitlines()[1:]:
                cols = row.split("\t")
                if len(cols) >= 12:
                    word = cols[11].strip()
                    if not word:
                        continue
                    try:
                        conf = float(cols[10])
                        x, y, w, h = int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9])
                    except (ValueError, TypeError):
                        continue
                    if conf >= 25.0 and w > 2 and h > 2:
                        ocr_candidates.append({
                            "name": word,
                            "role": "text",
                            "bbox": (x, y, x + w, y + h),
                            "bbox_format": "xyxy",
                            "source": "ocr",
                            "confidence": conf,
                        })
        unique_ocr: list[dict[str, Any]] = []
        for candidate in sorted(
            ocr_candidates,
            key=lambda control: float(control["confidence"]),
            reverse=True,
        ):
            name = str(candidate["name"]).casefold()
            x1, y1, x2, y2 = candidate["bbox"]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            duplicate = any(
                str(existing["name"]).casefold() == name
                and abs(
                    center_x
                    - (existing["bbox"][0] + existing["bbox"][2]) / 2
                ) <= max(8, min(x2 - x1, existing["bbox"][2] - existing["bbox"][0]) // 2)
                and abs(
                    center_y
                    - (existing["bbox"][1] + existing["bbox"][3]) / 2
                ) <= max(8, min(y2 - y1, existing["bbox"][3] - existing["bbox"][1]) // 2)
                for existing in unique_ocr
            )
            if not duplicate:
                unique_ocr.append(candidate)
        ocr_lines_map: dict[int, list[str]] = {}
        for control in unique_ocr:
            y = control["bbox"][1]
            line = y // 18
            words = ocr_lines_map.setdefault(line, [])
            if str(control["name"]).casefold() not in {
                word.casefold() for word in words
            }:
                words.append(str(control["name"]))
            detected_controls.append(control)
        unique_controls: list[dict[str, Any]] = []
        control_signatures: set[tuple[str, str, str]] = set()
        for control in detected_controls:
            signature = (
                str(control.get("source", "")),
                " ".join(str(control.get("name", "")).casefold().split()),
                str(control.get("bbox", "")),
            )
            if signature not in control_signatures:
                control_signatures.add(signature)
                unique_controls.append(control)
        detected_controls = unique_controls

        vis_lines = []
        if active_win and active_win != "Desktop":
            vis_lines.append(f"Active Window: {active_win}")
        if open_wins:
            vis_lines.append(f"Open Windows: {', '.join(open_wins)}")
        if ocr_lines_map:
            ocr_text_summary = " | ".join(" ".join(words) for words in list(ocr_lines_map.values())[:40])
            vis_lines.append(f"Screen OCR Text: {ocr_text_summary}")
        fallback_dimensions = _validated_png_dimensions(b64) if code == 0 else None
        if fallback_dimensions is None:
            b64 = ""
            detected_controls = []
            visible_text = "Desktop capture unavailable: fallback did not produce a valid PNG frame."
            capture_timestamp_ns = None
        else:
            visible_text = "\n".join(vis_lines)

        capture_age_ms = None
        if (
            fallback_dimensions is not None
            and isinstance(capture_timestamp_ns, int)
            and not isinstance(capture_timestamp_ns, bool)
        ):
            age_ns = time.time_ns() - capture_timestamp_ns
            if age_ns >= 0:
                capture_age_ms = age_ns / 1_000_000

        obs = ScreenObservation(
            screenshot_base64=b64,
            capture_timestamp_ns=capture_timestamp_ns,
            capture_age_ms=capture_age_ms,
            width=fallback_dimensions[0] if fallback_dimensions else 1280,
            height=fallback_dimensions[1] if fallback_dimensions else 800,
            active_window=active_win or "Desktop",
            visible_text=visible_text,
            detected_controls=detected_controls,
            desktop_state=(
                "INTERACTIVE"
                if fallback_dimensions is not None
                else "NO_DISPLAY"
            ),
        )
        if obs.screenshot_base64:
            self._last_screenshot = obs
            self._last_screenshot_time = now
        else:
            self._last_screenshot = None
            self._last_screenshot_time = 0.0
        return obs

    # -------------------------------------------------------------
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
        *,
        capture_screenshot: bool = True,
    ) -> ScreenObservation:
        async with self._gui_lock:
            previous_state_version = getattr(
                getattr(self, "_last_state_observation", None),
                "perception_version",
                0,
            )
            atype = action.action
            disp = self._get_display(workspace_id)

            async def exec_gui(command: str) -> None:
                nsi_res = await self._nsi_request({"cmd": "x11", "shell": command, "timeout": 3.0}, timeout=3.5)
                if nsi_res is not None and "exit_code" in nsi_res:
                    code = int(nsi_res.get("exit_code", 1))
                    err = str(nsi_res.get("stderr", ""))
                else:
                    code, _, err = await self._docker_exec(command)
                if code != 0:
                    raise RuntimeError(f"GUI action failed with exit code {code}: {err.strip()}")

            async def _validate_pointer_coords(
                *points: tuple[int | None, int | None],
            ) -> None:
                if any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for point in points
                    for value in point
                ):
                    raise ValueError("Pointer actions require explicit integer coordinates")

                now = asyncio.get_event_loop().time()
                state_observation = self._last_state_observation
                state_age_ms = (
                    (
                        state_observation.capture_age_ms
                        + max(0.0, now - self._last_state_observed_at) * 1000
                    )
                    if state_observation is not None
                    and state_observation.capture_age_ms is not None
                    and self._last_state_observed_at > 0
                    else float("inf")
                )
                state_dimensions = (
                    (state_observation.width, state_observation.height)
                    if state_observation is not None
                    and state_observation.desktop_state == "STATE_ONLY"
                    and state_observation.width > 0
                    and state_observation.height > 0
                    and state_age_ms <= _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                    else None
                )
                observation = self._last_screenshot
                cache_age_ms = (
                    max(0.0, now - self._last_screenshot_time) * 1000
                    if observation is not None and self._last_screenshot_time > 0
                    else float("inf")
                )
                cached_dimensions = (
                    _validated_png_dimensions(observation.screenshot_base64)
                    if observation is not None
                    and observation.desktop_state == "INTERACTIVE"
                    else None
                )
                if state_dimensions is not None:
                    dimensions = state_dimensions
                else:
                    if (
                        observation is None
                        or cached_dimensions is None
                        or cache_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                        or (
                            observation.capture_age_ms is not None
                            and observation.capture_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                        )
                    ):
                        observation = await self.screenshot(workspace_id)

                    dimensions = (
                        _validated_png_dimensions(observation.screenshot_base64)
                        if observation is not None
                        and observation.desktop_state == "INTERACTIVE"
                        and (
                            observation.capture_age_ms is None
                            or observation.capture_age_ms
                            <= _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                        )
                        else None
                    )
                if dimensions is None:
                    raise RuntimeError("Pointer action blocked: no current captured screen bounds")
                width, height = dimensions
                for cx, cy in points:
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
                await exec_gui(f"export DISPLAY={disp}; {move_cmd}")

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
                if action.x is not None and action.y is not None:
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
                        f"xdotool search --onlyvisible --name {safe_target} windowclose 2>/dev/null || "
                        f"pkill -f -- {safe_target}"
                    )
                else:
                    await exec_gui(f"DISPLAY={disp} xdotool getactivewindow windowclose")

            elif atype == GUIActionType.CLOSE_APP and action.app_name:
                await exec_gui(f"pkill -f -- {shlex.quote(action.app_name)}")

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
                            f"export DISPLAY={disp}; xdotool key Escape 2>/dev/null; "
                            f"wmctrl -i -a {shlex.quote(target)} 2>/dev/null || "
                            f"xdotool windowactivate --sync {shlex.quote(target)} 2>/dev/null"
                        )
                    else:
                        focus_cmd = (
                            f"export DISPLAY={disp}; xdotool key Escape 2>/dev/null; "
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
            self._last_state_observed_at = 0.0
            await asyncio.sleep(0.3)
            if not capture_screenshot:
                state_observation = await self.observe_state(workspace_id)
                state_version = getattr(
                    state_observation, "perception_version", 0
                )
                if (
                    state_observation is not None
                    and state_observation.desktop_state == "STATE_ONLY"
                    and state_version > previous_state_version
                ):
                    return state_observation
                logger.info(
                    "gui_action_state_fell_back_to_screenshot",
                    workspace_id=workspace_id,
                    action=atype.value,
                    previous_version=previous_state_version,
                    observed_version=state_version,
                    reason=(
                        "state_unavailable"
                        if state_observation is None
                        else "state_not_newer_than_pre_action"
                    ),
                )
            return await self.screenshot(workspace_id)

    async def gui_action_state(
        self,
        workspace_id: str,
        action: GUIAction,
        actor: str = "operator",
    ) -> ScreenObservation:
        """Prefer a fresh NSI state receipt after dispatch; preserve image fallback."""
        return await self.gui_action(
            workspace_id,
            action,
            actor=actor,
            capture_screenshot=False,
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
            await self._docker_exec(f'DISPLAY={disp} wmctrl -i -r "{left_win[0]}" -b remove,maximized_vert,maximized_horz')
            code1, _, _ = await self._docker_exec(f'DISPLAY={disp} wmctrl -i -r "{left_win[0]}" -e 0,0,0,640,800')
            if len(client_windows) > 1:
                right_win = client_windows[1]
                await self._docker_exec(f'DISPLAY={disp} wmctrl -i -r "{right_win[0]}" -b remove,maximized_vert,maximized_horz')
                code2, _, _ = await self._docker_exec(f'DISPLAY={disp} wmctrl -i -r "{right_win[0]}" -e 0,640,0,640,800')

        return (not windows and code_l == 0) or code1 == 0 or code2 == 0

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
            payload = (obs.screenshot_base64 or f"{obs.active_window}:{obs.visible_text}" or "empty").encode("utf-8")
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
            "find /usr/local/bin -maxdepth 1 -type f 2>/dev/null | while read -r p; do [ -x \"$p\" ] && basename \"$p\"; done; "
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
        and .desktop entries discovered dynamically from the OS."""
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
        search_esc = app_clean.replace('"', '\\"').lower()
        desktop_lookup_cmd = (
            f"for d in /root/Desktop/*.desktop /usr/share/applications/*.desktop ~/.local/share/applications/*.desktop; do "
            f"  [ -f \"$d\" ] || continue; "
            f"  grep -qi '^NoDisplay=true' \"$d\" 2>/dev/null && continue; "
            f"  grep -qi '^Terminal=true' \"$d\" 2>/dev/null && continue; "
            f"  name=$(grep -i '^Name=' \"$d\" 2>/dev/null | head -n1 | cut -d= -f2- | tr '[:upper:]' '[:lower:]'); "
            f"  gname=$(grep -i '^GenericName=' \"$d\" 2>/dev/null | head -n1 | cut -d= -f2- | tr '[:upper:]' '[:lower:]'); "
            f"  cats=$(grep -i '^Categories=' \"$d\" 2>/dev/null | head -n1 | cut -d= -f2- | tr '[:upper:]' '[:lower:]'); "
            f"  base=$(basename \"$d\" .desktop 2>/dev/null | tr '[:upper:]' '[:lower:]'); "
            f"  if echo \"$name $gname $cats $base\" | grep -q \"{search_esc}\"; then "
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

        # 5. Default fallback: hyphenated if multi-word, else first token
        try:
            parts = shlex.split(app_clean)
            if len(parts) > 1:
                return "-".join(parts)
            return parts[0] if parts else app_clean
        except Exception:
            parts = app_clean.split()
            if len(parts) > 1:
                return "-".join(parts)
            return parts[0] if parts else app_clean

    async def launch_application(self, workspace_id: str, app_name: str, actor: str = "operator") -> bool:
        clean_app = (app_name or "").strip()
        if not clean_app:
            return False

        app_query, args = self._split_app_and_args(clean_app)
        resolved_bin = await self._resolve_app_binary(app_query)
        if not resolved_bin:
            try:
                resolved_bin = shlex.split(clean_app)[0]
            except Exception:
                resolved_bin = clean_app.split()[0]

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
