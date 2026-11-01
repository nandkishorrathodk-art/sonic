"""macOS desktop provider backed by Docker-OSX (QEMU + KVM).

Docker-OSX (``sickcodes/docker-osx``) runs macOS inside a QEMU virtual machine
that is itself packaged as a Docker container.  That is a fundamentally
different body from the Linux XFCE workstation:

* There is **no X11 display inside the guest**, so ``xdotool``/``scrot``/
  ``wmctrl`` cannot be used.  The desktop is reached through the QEMU
  framebuffer over **VNC (RFB)** and a shell over **SSH**.
* It requires ``/dev/kvm`` on the host.  Without hardware virtualisation the VM
  cannot boot at all, so this provider fails closed rather than pretending.

This provider keeps the same ``ComputerProvider`` contract as
``DockerComputerProvider`` (screenshots, GUI actions, terminal, files, git) so
the rest of SONIC — the AVCS perception loop, the agent, the workstation routes
— does not need to know which body it is driving.  It is opt-in via
``SONIC_WORKSTATION_PROVIDER=macos_docker_osx``.

Honest capability boundary: pointer, keyboard and screenshot work through RFB.
File/terminal/process/git work through SSH.  Window-management actions that
have no RFB equivalent use AppleScript over SSH, and anything that cannot be
expressed that way fails closed with an explicit reason instead of silently
doing nothing.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer.models import (
    ComputerProfile,
    ComputerState,
    ComputerWorkspace,
    ComputerWorkspaceStatus,
    ComputerWorkspaceType,
    FileEntry,
    GUIAction,
    GUIActionType,
    ProcessInfo,
    ScreenObservation,
    ServiceInfo,
)
from sonic.computer.rfb import (
    POINTER_BUTTON_LEFT,
    POINTER_BUTTON_MIDDLE,
    POINTER_BUTTON_RIGHT,
    POINTER_WHEEL_DOWN,
    POINTER_WHEEL_LEFT,
    POINTER_WHEEL_RIGHT,
    POINTER_WHEEL_UP,
    RFBClient,
    RFBError,
)
from sonic.logger import get_logger
from sonic.sandbox.provider import ExecResult

logger = get_logger(__name__)

# macOS "Command" is the X11 Super/Meta key; Option is Alt.
_DEFAULT_WORKSPACE_PATH = "/Users/sonic/workspace"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class MacOSDockerComputerProvider(DockerComputerProvider):
    """Drive a Docker-OSX macOS guest through VNC (GUI) and SSH (execution)."""

    host_workspace_id = "macos-docker-osx-desktop"
    reattach_persisted_workspace = True

    def __init__(
        self,
        *,
        container_name: str | None = None,
        vnc_host: str | None = None,
        vnc_port: int | None = None,
        vnc_password: str | None = None,
        ssh_host: str | None = None,
        ssh_port: int | None = None,
        ssh_user: str | None = None,
        ssh_key_path: str | None = None,
        ssh_password: str | None = None,
        perception: Any | None = None,
        rfb_factory: Callable[..., RFBClient] | None = None,
        ssh_runner: Callable[[str, int], Awaitable[tuple[int, str, str]]] | None = None,
    ) -> None:
        super().__init__(container_name=container_name or _env("SONIC_MACOS_CONTAINER", "sonic-macos-desktop"))

        self.vnc_host = vnc_host or _env("SONIC_MACOS_VNC_HOST", "127.0.0.1")
        self.vnc_port = int(vnc_port if vnc_port is not None else _env("SONIC_MACOS_VNC_PORT", "5901"))
        self.vnc_password = vnc_password if vnc_password is not None else _env("SONIC_MACOS_VNC_PASSWORD")
        self.ssh_host = ssh_host or _env("SONIC_MACOS_SSH_HOST", self.vnc_host)
        self.ssh_port = int(ssh_port if ssh_port is not None else _env("SONIC_MACOS_SSH_PORT", "50922"))
        self.ssh_user = ssh_user or _env("SONIC_MACOS_SSH_USER", "sonic")
        self.ssh_key_path = ssh_key_path or _env("SONIC_MACOS_SSH_KEY")
        self.ssh_password = ssh_password if ssh_password is not None else _env("SONIC_MACOS_SSH_PASSWORD")
        self.workspace_path = _env("SONIC_MACOS_WORKSPACE", _DEFAULT_WORKSPACE_PATH)

        self._perception = perception
        self._rfb_factory = rfb_factory or RFBClient
        self._ssh_runner = ssh_runner

        self._rfb: RFBClient | None = None
        self._rfb_lock = asyncio.Lock()
        self._gui_lock = asyncio.Lock()
        self._frame_id = 0
        self._workspace: ComputerWorkspace | None = None
        self._workspace_owner: str | None = None
        self._last_observation: ScreenObservation | None = None
        self._cached_app_list: tuple[float, list[str]] | None = None

    # -------------------------------------------------------------
    # Capability / provisioning
    # -------------------------------------------------------------

    @property
    def name(self) -> str:
        return "MacOSDockerComputerProvider"

    @staticmethod
    def kvm_available() -> bool:
        """Docker-OSX cannot boot without KVM; report the real host capability."""
        return Path("/dev/kvm").exists()

    def _perception_engine(self) -> Any:
        if self._perception is None:
            from sonic.computer_use.avcs_perception import AVCSPerception

            self._perception = AVCSPerception()
        return self._perception

    async def create(
        self,
        tenant_id: str,
        engagement_id: str,
        workspace_type: ComputerWorkspaceType = ComputerWorkspaceType.MISSION_COMPUTER,
        profile: ComputerProfile = ComputerProfile.GENERIC_LINUX,
    ) -> ComputerWorkspace:
        if workspace_type != ComputerWorkspaceType.MISSION_COMPUTER:
            return await super().create(
                tenant_id=tenant_id,
                engagement_id=engagement_id,
                workspace_type=workspace_type,
                profile=profile,
            )
        if not self.kvm_available():
            raise RuntimeError(
                "Docker-OSX requires hardware virtualisation (/dev/kvm), which is "
                "not available on this host. The macOS guest cannot boot."
            )
        if self._workspace_owner not in (None, tenant_id):
            raise RuntimeError(
                "The shared macOS desktop is already owned by another tenant."
            )
        self._workspace_owner = tenant_id
        workspace = ComputerWorkspace(
            id=self.host_workspace_id,
            tenant_id=tenant_id,
            engagement_id=engagement_id,
            workspace_type=workspace_type,
            profile=ComputerProfile.GENERIC_LINUX,
            provider_type=type(self).__name__,
            image="sickcodes/docker-osx:latest",
            status=ComputerWorkspaceStatus.RUNNING,
            workspace_path=self.workspace_path,
            capabilities=["desktop", "vnc", "ssh_terminal", "filesystem", "git", "applications"],
        )
        self._workspace = workspace
        return workspace

    async def destroy(self, workspace_id: str) -> bool:
        if workspace_id == self.host_workspace_id:
            self._workspace = None
            self._workspace_owner = None
            self._last_observation = None
            await self._disconnect_rfb()
            return True
        return await super().destroy(workspace_id)

    async def get_or_create_home(self, tenant_id: str) -> ComputerWorkspace:
        """Return the tenant's long-lived macOS home, provisioning it for real.

        The parent implementation fabricates a RUNNING workspace without
        provisioning anything; that would make the being loop believe a macOS
        desktop exists when the VM is not up. This override provisions through
        ``create()`` (KVM-checked, fail-closed) instead.
        """
        if self._workspace_owner and self._workspace_owner != tenant_id:
            raise RuntimeError("The shared macOS desktop is already owned by another tenant.")
        workspace = self._workspace
        if (
            workspace is not None
            and workspace.tenant_id == tenant_id
            and workspace.status
            not in (ComputerWorkspaceStatus.DESTROYED, ComputerWorkspaceStatus.FAILED)
        ):
            return workspace
        return await self.create(tenant_id, f"home-{tenant_id}")

    def _require_workspace(self, workspace_id: str) -> ComputerWorkspace:
        workspace = self._workspace
        if (
            workspace_id != self.host_workspace_id
            or workspace is None
            or self._workspace_owner != workspace.tenant_id
        ):
            raise RuntimeError("The macOS desktop is not provisioned for this session.")
        return workspace

    # -------------------------------------------------------------
    # SSH execution transport
    # -------------------------------------------------------------

    def _ssh_command(self, command: str) -> list[str]:
        args = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=8",
            "-o", "LogLevel=ERROR",
            "-p", str(self.ssh_port),
        ]
        if self.ssh_key_path:
            args += ["-o", "BatchMode=yes", "-i", self.ssh_key_path]
        elif self.ssh_password:
            # Password auth needs a helper; fail closed if it is unavailable.
            if not _which("sshpass"):
                raise RuntimeError(
                    "macOS SSH password auth requires `sshpass`, which is not "
                    "installed. Provide SONIC_MACOS_SSH_KEY instead."
                )
            args = ["sshpass", "-e", *args, "-o", "BatchMode=no"]
        else:
            args += ["-o", "BatchMode=yes"]
        args.append(f"{self.ssh_user}@{self.ssh_host}")
        args.append(command)
        return args

    async def _ssh_exec(self, command: str, timeout: int = 60) -> tuple[int, str, str]:
        if self._ssh_runner is not None:
            return await self._ssh_runner(command, timeout)

        argv = self._ssh_command(command)
        env = dict(os.environ)
        if self.ssh_password:
            env["SSHPASS"] = self.ssh_password
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"macOS SSH transport unavailable: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"ssh command timed out after {timeout}s"
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def terminal(
        self,
        workspace_id: str,
        command: str,
        timeout: int = 60,
        actor: str = "operator",
    ) -> ExecResult:
        code, stdout, stderr = await self._ssh_exec(command, timeout=timeout)
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
        if isinstance(command, list):
            cmd_str = " ".join(shlex.quote(c) for c in command)
        else:
            cmd_str = str(command)
        if cwd:
            cmd_str = f"cd {shlex.quote(cwd)} && {cmd_str}"
        return await self.terminal(workspace_id, cmd_str, timeout=timeout, actor=actor)

    # -------------------------------------------------------------
    # VNC (RFB) transport
    # -------------------------------------------------------------

    async def _connect_rfb(self) -> RFBClient:
        if self._rfb is not None and self._rfb.connected:
            return self._rfb
        client = self._rfb_factory(
            self.vnc_host,
            self.vnc_port,
            password=self.vnc_password or None,
        )
        await asyncio.to_thread(client.connect)
        self._rfb = client
        return client

    async def _disconnect_rfb(self) -> None:
        client, self._rfb = self._rfb, None
        if client is not None:
            await asyncio.to_thread(client.close)

    def _capture_frame_sync(self) -> tuple[Any, int, int]:
        assert self._rfb is not None
        frame = self._rfb.framebuffer()
        return frame, int(frame.shape[1]), int(frame.shape[0])

    async def screenshot(self, workspace_id: str) -> ScreenObservation:
        self._require_workspace(workspace_id)
        async with self._rfb_lock:
            try:
                await self._connect_rfb()
                frame, width, height = await asyncio.to_thread(self._capture_frame_sync)
            except (RFBError, OSError) as exc:
                # Drop a possibly-desynced connection so the next call reconnects.
                await self._disconnect_rfb()
                logger.error("macos_vnc_capture_failed", error=str(exc))
                return ScreenObservation(
                    width=0,
                    height=0,
                    visible_text=f"macOS VNC capture unavailable: {exc}",
                    desktop_state="NO_DISPLAY",
                )
            self._frame_id += 1
            capture_timestamp_ns = time.time_ns()
            avcs_state = self._perception_engine().perceive_rgb(
                frame, source_frame_id=self._frame_id
            )
            observation = ScreenObservation(
                raw_rgb=frame.tobytes(),
                frame_id=self._frame_id,
                avcs_state=avcs_state,
                capture_timestamp_ns=capture_timestamp_ns,
                capture_age_ms=0.0,
                perception_version=int(avcs_state.get("frame_index", 0)),
                width=width,
                height=height,
                active_window="macOS desktop",
                desktop_state="INTERACTIVE",
                timestamp=datetime.now(UTC).isoformat(),
            )
            self._last_observation = observation
            return observation

    # -------------------------------------------------------------
    # GUI actions (RFB pointer + keyboard)
    # -------------------------------------------------------------

    @staticmethod
    def _button_mask(action: GUIAction) -> int:
        if action.action == GUIActionType.RIGHT_CLICK or action.button == 3:
            return POINTER_BUTTON_RIGHT
        if action.button == 2:
            return POINTER_BUTTON_MIDDLE
        return POINTER_BUTTON_LEFT

    @staticmethod
    def _pointer_point(action: GUIAction, *, destination: bool = False) -> tuple[int, int]:
        x = action.x2 if destination else action.x
        y = action.y2 if destination else action.y
        if x is None or y is None:
            raise RuntimeError(f"{action.action} requires explicit screen pixel coordinates")
        return int(x), int(y)

    def _dispatch_gui_sync(self, action: GUIAction) -> None:
        client = self._rfb
        if client is None or not client.connected:
            raise RFBError("VNC session is not connected")

        atype = action.action
        if atype == GUIActionType.MOVE:
            client.pointer(*self._pointer_point(action), 0)
        elif atype in (GUIActionType.CLICK, GUIActionType.DOUBLE_CLICK, GUIActionType.RIGHT_CLICK):
            x, y = self._pointer_point(action)
            mask = self._button_mask(action)
            presses = 2 if atype == GUIActionType.DOUBLE_CLICK else 1
            for _ in range(presses):
                client.pointer(x, y, mask)
                client.pointer(x, y, 0)
        elif atype == GUIActionType.DRAG:
            sx, sy = self._pointer_point(action)
            dx, dy = self._pointer_point(action, destination=True)
            client.pointer(sx, sy, POINTER_BUTTON_LEFT)
            client.pointer(dx, dy, POINTER_BUTTON_LEFT)
            client.pointer(dx, dy, 0)
        elif atype == GUIActionType.SCROLL:
            x, y = self._pointer_point(action)
            horizontal = action.scroll_axis == "horizontal"
            if horizontal:
                button = POINTER_WHEEL_RIGHT if action.scroll_delta > 0 else POINTER_WHEEL_LEFT
            else:
                button = POINTER_WHEEL_DOWN if action.scroll_delta > 0 else POINTER_WHEEL_UP
            for _ in range(max(1, abs(int(action.scroll_delta)))):
                client.pointer(x, y, button)
                client.pointer(x, y, 0)
        elif atype == GUIActionType.TYPE:
            if not action.text:
                raise RuntimeError("GUI_TYPE requires explicit text")
            client.type_text(action.text)
        elif atype == GUIActionType.KEYPRESS:
            if not action.key:
                raise RuntimeError("KEYPRESS requires a key")
            client.key_press(action.key)
        elif atype == GUIActionType.HOTKEY:
            if not action.key:
                raise RuntimeError("HOTKEY requires a key chord")
            client.chord([part for part in action.key.split("+") if part.strip()])
        elif atype == GUIActionType.WAIT:
            time.sleep(max(0.0, min(30.0, action.delay_ms / 1000.0)))
        else:
            raise RuntimeError(
                f"{atype} has no VNC equivalent; use SSH/AppleScript for window "
                "management on the macOS guest."
            )

    async def gui_action(
        self,
        workspace_id: str,
        action: GUIAction,
        actor: str = "operator",
    ) -> ScreenObservation:
        self._require_workspace(workspace_id)
        async with self._gui_lock:
            async with self._rfb_lock:
                try:
                    await self._connect_rfb()
                    await asyncio.to_thread(self._dispatch_gui_sync, action)
                except (RFBError, OSError) as exc:
                    await self._disconnect_rfb()
                    raise RuntimeError(f"macOS GUI action failed: {exc}") from exc
        # Settle briefly, then return a fresh observation so verification sees
        # the effect of the action rather than the pre-action frame.
        await asyncio.sleep(min(1.0, max(0.15, action.delay_ms / 1000.0)))
        return await self.screenshot(workspace_id)

    # -------------------------------------------------------------
    # Window management via AppleScript (SSH)
    # -------------------------------------------------------------

    async def maximize_window(self, workspace_id: str, target: str | None = None) -> bool:
        if not target:
            return False
        script = (
            'tell application "System Events" to tell process '
            f'"{target}" to set position of window 1 to {{0, 0}}'
        )
        code, _, _ = await self._ssh_exec(f"osascript -e {shlex.quote(script)}", timeout=15)
        return code == 0

    async def tile_workstation(self, workspace_id: str) -> bool:
        # macOS has no wmctrl; window tiling is intentionally not claimed.
        return False

    # -------------------------------------------------------------
    # Status / filesystem / git / applications (SSH)
    # -------------------------------------------------------------

    async def status(self, workspace_id: str) -> ComputerState:
        ws = self._workspace or self._ensure_default_workspace()
        if not await self._container_is_running():
            ws.status = ComputerWorkspaceStatus.DEGRADED
            return ComputerState(
                workspace_id=workspace_id or self.host_workspace_id,
                tenant_id=ws.tenant_id,
                status=ComputerWorkspaceStatus.DEGRADED,
                active_application="",
                open_applications=[],
                active_window="",
                working_directory=self.workspace_path,
                running_processes=[],
                installed_applications=[],
                current_project="",
                git_branch="",
                resource_usage={},
            )

        procs_coro = self.process_list(workspace_id)
        apps_coro = self.application_list(workspace_id)
        procs, installed = await asyncio.gather(procs_coro, apps_coro)
        return ComputerState(
            workspace_id=workspace_id or self.host_workspace_id,
            tenant_id=ws.tenant_id,
            status=ComputerWorkspaceStatus.RUNNING,
            active_application="macOS desktop",
            open_applications=["macOS desktop"],
            active_window="macOS desktop",
            working_directory=self.workspace_path,
            running_processes=[p.name for p in procs],
            installed_applications=installed,
            current_project="sonic",
            git_branch="main",
            resource_usage={},
        )

    async def read_file(self, workspace_id: str, path: str) -> str:
        code, out, _ = await self._ssh_exec(f"cat {shlex.quote(path)}", timeout=15)
        return out if code == 0 else ""

    async def write_file(self, workspace_id: str, path: str, content: str, actor: str = "operator") -> bool:
        parent = os.path.dirname(path)
        if parent:
            await self._ssh_exec(f"mkdir -p {shlex.quote(parent)}", timeout=15)
        # Base64 keeps arbitrary content intact through the SSH command line.
        import base64

        payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = f"printf '%s' {shlex.quote(payload)} | base64 -d > {shlex.quote(path)}"
        code, _, _ = await self._ssh_exec(cmd, timeout=30)
        return code == 0

    async def list_files(self, workspace_id: str, path: str = ".") -> list[FileEntry]:
        code, out, _ = await self._ssh_exec(
            f"ls -la {shlex.quote(path)} 2>/dev/null", timeout=15
        )
        entries: list[FileEntry] = []
        if code == 0 and out.strip():
            for line in out.splitlines()[1:]:
                parts = line.split(maxsplit=8)
                if len(parts) >= 9:
                    fname = parts[8].strip()
                    if fname in (".", ".."):
                        continue
                    entries.append(
                        FileEntry(
                            path=f"{path}/{fname}".replace("//", "/"),
                            name=fname,
                            is_dir=parts[0].startswith("d"),
                            size_bytes=int(parts[4]) if parts[4].isdigit() else 0,
                        )
                    )
        return entries

    async def process_list(self, workspace_id: str) -> list[ProcessInfo]:
        code, out, _ = await self._ssh_exec(
            "ps -Ao pid,%cpu,%mem,comm | head -40", timeout=10
        )
        procs: list[ProcessInfo] = []
        if code == 0 and out.strip():
            for line in out.splitlines()[1:]:
                parts = line.split(maxsplit=3)
                if len(parts) >= 4 and parts[0].isdigit():
                    procs.append(
                        ProcessInfo(
                            pid=int(parts[0]),
                            name=parts[3].strip().rsplit("/", 1)[-1],
                            cpu_pct=_to_float(parts[1]),
                            memory_mb=_to_float(parts[2]),
                        )
                    )
        return procs

    async def git_action(self, workspace_id: str, action: str, **kwargs: Any) -> Any:
        ws = shlex.quote(self.workspace_path)
        if action == "status":
            code, out, _ = await self._ssh_exec(f"git -C {ws} status --porcelain 2>/dev/null", timeout=20)
            bcode, bout, _ = await self._ssh_exec(f"git -C {ws} branch --show-current 2>/dev/null", timeout=20)
            branch = bout.strip() if bcode == 0 and bout.strip() else ("uninitialized" if code != 0 else "main")
            clean = (code == 0) and not out.strip()
            untracked = [line[3:] for line in out.splitlines() if line.startswith("??")] if code == 0 else []
            modified = [line[3:] for line in out.splitlines() if not line.startswith("??")] if code == 0 else []
            from sonic.computer.models import GitStatusInfo

            return GitStatusInfo(branch=branch, is_clean=clean, untracked_files=untracked, modified_files=modified)
        if action == "commit":
            msg = kwargs.get("message", "chore: automated commit")
            code, _, _ = await self._ssh_exec(
                f"git -C {ws} add -A && git -C {ws} commit -m {shlex.quote(str(msg))}", timeout=30
            )
            return code == 0
        if action in ("branch", "checkout"):
            branch_name = str(kwargs.get("branch_name") or kwargs.get("branch") or "main").lstrip("-") or "main"
            code, _, _ = await self._ssh_exec(
                f"git -C {ws} checkout -B {shlex.quote(branch_name)}", timeout=30
            )
            return code == 0
        if action == "diff":
            code, out, _ = await self._ssh_exec(f"git -C {ws} diff", timeout=30)
            return out or "Working tree clean."
        args = kwargs.get("args") or []
        cmd = f"git -C {ws} {shlex.quote(action)} " + " ".join(shlex.quote(str(a)) for a in args)
        code, out, _ = await self._ssh_exec(cmd, timeout=30)
        return out

    async def application_list(self, workspace_id: str) -> list[str]:
        now = asyncio.get_event_loop().time()
        if self._cached_app_list is not None and (now - self._cached_app_list[0]) < 60.0:
            return list(self._cached_app_list[1])
        code, out, _ = await self._ssh_exec(
            "ls -1 /Applications /System/Applications 2>/dev/null", timeout=15
        )
        apps: list[str] = []
        if code == 0:
            for line in out.splitlines():
                name = line.strip()
                if name.endswith(".app"):
                    name = name[:-4]
                if name and name not in apps:
                    apps.append(name)
        if apps:
            self._cached_app_list = (now, list(apps))
        return apps

    async def launch_application(self, workspace_id: str, app_name: str, actor: str = "operator") -> bool:
        name = (app_name or "").strip().strip("\"'")
        if not name:
            return False
        code, _, err = await self._ssh_exec(f"open -a {shlex.quote(name)}", timeout=30)
        if code != 0:
            logger.warning("macos_application_launch_failed", application=name, error=err.strip()[:200])
            return False
        return True

    async def close_application(self, workspace_id: str, app_name: str, actor: str = "operator") -> bool:
        name = (app_name or "").strip().strip("\"'")
        if not name:
            return False
        script = f'tell application {shlex.quote(name)} to quit'
        code, _, _ = await self._ssh_exec(f"osascript -e {shlex.quote(script)}", timeout=20)
        return code == 0

    async def install_application(self, workspace_id: str, package_name: str, actor: str = "operator") -> tuple[bool, str]:
        allowed, reason = self.app_policy.is_package_allowed(package_name)
        if not allowed:
            return False, reason
        code, out, err = await self._ssh_exec(
            f"command -v brew >/dev/null && brew install {shlex.quote(package_name)}", timeout=300
        )
        if code != 0:
            return False, err or "Homebrew is unavailable in the macOS guest."
        self._cached_app_list = None
        return True, out

    async def uninstall_application(self, workspace_id: str, package_name: str, actor: str = "operator") -> bool:
        code, _, _ = await self._ssh_exec(
            f"command -v brew >/dev/null && brew uninstall {shlex.quote(package_name)}", timeout=180
        )
        self._cached_app_list = None
        return code == 0

    async def service_action(self, workspace_id: str, service_name: str, action: str, actor: str = "operator") -> ServiceInfo:
        code, out, _ = await self._ssh_exec(
            f"launchctl list | grep -i {shlex.quote(service_name)} || true", timeout=15
        )
        running = code == 0 and bool(out.strip())
        return ServiceInfo(
            name=service_name,
            status="RUNNING" if running else "UNKNOWN",
            logs=[out.strip()] if out.strip() else [],
        )

    async def get_vnc_url(self, workspace_id: str) -> str | None:
        if workspace_id != self.host_workspace_id:
            return await super().get_vnc_url(workspace_id)
        return f"vnc://{self.vnc_host}:{self.vnc_port}"

    async def close(self) -> None:
        await self._disconnect_rfb()
        await super().close()


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _which(binary: str) -> str | None:
    from shutil import which

    return which(binary)
