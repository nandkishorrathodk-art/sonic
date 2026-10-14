"""
SONIC-REDA — Daytona Real Graphical Workstation & Computer Provider
====================================================================
Integrates Daytona Cloud Sandboxes with full graphical Linux workstation capabilities:
    - Real Xvfb (:99) + XFCE4 desktop session
    - Real VNC / noVNC live streaming bridge
    - Real Daytona SDK Computer Use (AsyncMouse, AsyncKeyboard, AsyncScreenshot)
    - Real remote PTY terminal (/bin/bash)
    - Real remote Filesystem & Git operations
    - Multi-tenant authenticated scoping
    - FAIL-CLOSED security invariant: Zero host OS execution.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import os
import re
import shlex
import time
from io import BytesIO
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()

from sonic.computer.docker_sandbox import DockerContainerSandbox
from sonic.computer.models import (
    ApplicationPolicy,
    ComputerAuditEvent,
    ComputerProfile,
    ComputerRiskLevel,
    ComputerSession,
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
    _new_id,
    _now,
)
from sonic.computer.provider import ComputerProvider
from sonic.logger import get_logger
from sonic.sandbox.provider import ExecResult

logger = get_logger(__name__)

_SCREEN_CAPTURE_FRESHNESS_BUDGET_MS = 2_000


def _get_display(workspace_id: str | None = None) -> str:
    """Check if environment has DISPLAY, default to :99 for Daytona/Docker Xvfb."""
    return os.environ.get("DISPLAY") or ":99"


def _validated_png_dimensions(png_bytes: bytes) -> tuple[int, int] | None:
    """Return dimensions only when Pillow can fully verify a real PNG frame."""
    if not png_bytes:
        return None
    try:
        from PIL import Image

        with Image.open(BytesIO(png_bytes)) as image:
            if image.format != "PNG":
                return None
            image.verify()
        with Image.open(BytesIO(png_bytes)) as image:
            width, height = image.size
        return (width, height) if width > 0 and height > 0 else None
    except Exception:
        return None


class DaytonaComputerProvider(ComputerProvider):
    """
    Daytona-backed Graphical Computer Provider.
    Authoritative remote Linux workstation powered by Daytona Cloud.
    """

    def _get_display(self, workspace_id: str | None = None) -> str:
        """Check if environment has DISPLAY, default to :99 for Daytona/Docker Xvfb."""
        return _get_display(workspace_id)

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        app_policy: ApplicationPolicy | None = None,
    ):
        # Distinguish "not provided" (None -> read from env) from "explicitly
        # empty" ("" -> run offline with no cloud sandbox). Tests pass api_key=""
        # to exercise the no-sandbox path; the workstation API constructs with
        # the default (None) so it picks up DAYTONA_API_KEY from the environment.
        self.api_key = os.environ.get("DAYTONA_API_KEY", "") if api_key is None else api_key
        self.api_url = api_url if api_url is not None else os.environ.get("DAYTONA_API_URL")
        self.target = target if target is not None else os.environ.get("DAYTONA_TARGET", "us")
        self.app_policy = app_policy or ApplicationPolicy()

        self._client = None
        self._sandboxes: dict[str, Any] = {}
        self.workspaces: dict[str, ComputerWorkspace] = {}
        self.sessions: dict[str, ComputerSession] = {}
        self.audit_log: list[ComputerAuditEvent] = []
        self._active_windows: dict[str, str] = {}
        self._application_pids: dict[str, dict[str, set[int]]] = {}
        self._last_screenshot: ScreenObservation | None = None
        self._last_screenshot_time: float = 0.0
        self._last_screenshot_workspace_id: str | None = None
        self._avcs_perception = None
        # Persistent Body (PLAN Phase 2): the home desktop ID survives a
        # backend restart so _resolve_sandbox can re-attach via client.get(id).
        self._state_path = os.environ.get(
            "SONIC_WORKSTATION_STATE_PATH",
            str(Path(os.environ.get("SONIC_DATA_DIR", "sonic_data")) / "workstations.json"),
        )
        self._load_state()

    # -------------------------------------------------------------
    # Persistent Body state (PLAN Phase 2.0)
    # -------------------------------------------------------------

    def _load_state(self) -> None:
        """Reload persisted workspace records on init (simulates restart re-attach)."""
        try:
            raw = Path(self._state_path).read_text()
        except (FileNotFoundError, OSError):
            return
        try:
            records = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("workstation_state_corrupt_ignored", path=self._state_path)
            return
        for ws_id, rec in (records or {}).items():
            try:
                ws = ComputerWorkspace(
                    id=ws_id,
                    tenant_id=rec.get("tenant_id", "default"),
                    engagement_id=rec.get("engagement_id", ""),
                    workspace_type=ComputerWorkspaceType(rec.get("workspace_type", ComputerWorkspaceType.MISSION_COMPUTER.value)),
                    profile=ComputerProfile(rec.get("profile", ComputerProfile.DEBIAN_ENGINEERING.value)),
                    provider_type=rec.get("provider_type", "DaytonaComputerProvider"),
                    image=rec.get("image", ""),
                    status=ComputerWorkspaceStatus(rec.get("status", ComputerWorkspaceStatus.READY.value)),
                    created_at=rec.get("created_at", _now()),
                    last_active_at=rec.get("last_active_at", _now()),
                )
                self.workspaces[ws_id] = ws
            except Exception as exc:
                logger.warning("workstation_state_record_skipped", workspace_id=ws_id, error=str(exc))
        if self.workspaces:
            logger.info("workstation_state_loaded", workstations=list(self.workspaces.keys()))

    def _persist_state(self) -> None:
        """Write the workspace index to disk atomically so IDs survive a restart."""
        try:
            target = Path(self._state_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            records = {
                ws_id: {
                    "tenant_id": ws.tenant_id,
                    "engagement_id": ws.engagement_id,
                    "workspace_type": ws.workspace_type.value,
                    "profile": ws.profile.value,
                    "provider_type": ws.provider_type,
                    "image": ws.image,
                    "status": ws.status.value,
                    "created_at": ws.created_at,
                    "last_active_at": ws.last_active_at,
                }
                for ws_id, ws in self.workspaces.items()
            }
            tmp_path = target.with_suffix(f".tmp.{os.getpid()}")
            tmp_path.write_text(json.dumps(records, indent=2))
            tmp_path.replace(target)
        except Exception as exc:
            logger.warning("workstation_state_persist_failed", error=str(exc))

    async def get_or_create_home(self, tenant_id: str) -> ComputerWorkspace:
        """Return the tenant's long-lived MISSION_COMPUTER home, provisioning once.

        Engagements reuse the home; only research-lab/target sandboxes are
        disposable. If a home already exists (in memory or persisted), it is
        returned and its sandbox re-attached lazily via ``_resolve_sandbox``.
        """
        for ws in self.workspaces.values():
            if (
                ws.tenant_id == tenant_id
                and ws.workspace_type == ComputerWorkspaceType.MISSION_COMPUTER
                and ws.status not in (ComputerWorkspaceStatus.DESTROYED, ComputerWorkspaceStatus.FAILED)
            ):
                # Re-attach the live sandbox by its persisted ID.
                await self._resolve_sandbox(ws.id)
                logger.info("home_workstation_reused", workspace_id=ws.id, tenant_id=tenant_id)
                return ws
        ws = await self.create(
            tenant_id=tenant_id,
            engagement_id=f"home-{tenant_id}",
            workspace_type=ComputerWorkspaceType.MISSION_COMPUTER,
            profile=ComputerProfile.DEBIAN_ENGINEERING,
        )
        self._persist_state()
        logger.info("home_workstation_provisioned", workspace_id=ws.id, tenant_id=tenant_id)
        return ws

    def _get_client(self):
        """Initializes and returns the official AsyncDaytona SDK client."""
        if self._client is None:
            try:
                from daytona import AsyncDaytona, DaytonaConfig

                config_kwargs = {}
                if self.api_key:
                    config_kwargs["api_key"] = self.api_key
                if self.api_url:
                    # Current Daytona SDK uses api_url; server_url is kept
                    # only for older SDKs and emits a deprecation warning.
                    config_kwargs["api_url"] = self.api_url
                if self.target:
                    config_kwargs["target"] = self.target

                config = DaytonaConfig(**config_kwargs) if config_kwargs else None
                self._client = AsyncDaytona(config=config)
                logger.info("daytona_computer_sdk_initialized", has_api_key=bool(self.api_key))
            except Exception as e:
                logger.error("daytona_computer_sdk_init_failed", error=str(e))
                self._client = None
        return self._client

    @staticmethod
    def _normalize_sandbox_state(state: Any) -> str:
        """Normalize a Daytona sandbox state into an upper-case token.

        Daytona's ``SandboxState`` is an enum whose ``str()`` renders as
        ``"<SandboxState.STARTED: 'started'>"``. Comparing that directly never
        matches the simple ``STARTED`` / ``STOPPED`` checks we need, so a freshly
        created sandbox would be misreported as STOPPED. This helper extracts
        the bare canonical token from an enum, its ``.value``, or a plain string.
        """
        if state is None:
            return ""
        # Enum members expose .value (e.g. "started") and .name (e.g. "STARTED")
        value = getattr(state, "value", None)
        name = getattr(state, "name", None)
        token = name or value or str(state)
        return str(token).upper()

    async def _resolve_sandbox(self, workspace_id: str) -> Any | None:
        """Resolves the live AsyncDaytona sandbox instance for workspace or environment sandbox."""
        if workspace_id and workspace_id in self._sandboxes:
            return self._sandboxes[workspace_id]

        env_id = os.environ.get("DAYTONA_SANDBOX_ID", "")
        # Persisted workspace IDs are Daytona sandbox IDs. Older in-memory
        # records used a synthetic ws-* ID; prefer attached env sandbox if available.
        if workspace_id and not workspace_id.startswith("ws-") and workspace_id != "default":
            target_id = workspace_id
        else:
            target_id = env_id or workspace_id or ""

        if target_id:
            if target_id in self._sandboxes:
                return self._sandboxes[target_id]

            # Docker desktop container fallback / direct resolution
            if target_id in ("sonic-desktop-workstation", "docker-workstation") or target_id.startswith("docker:"):
                cname = "sonic-desktop-workstation" if target_id in ("sonic-desktop-workstation", "docker-workstation") else target_id.split(":", 1)[1]
                docker_sb = DockerContainerSandbox(cname)
                self._sandboxes[target_id] = docker_sb
                if workspace_id:
                    self._sandboxes[workspace_id] = docker_sb
                return docker_sb

            client = self._get_client()
            if client:
                try:
                    sandbox = await client.get(target_id)
                    if sandbox:
                        state_token = self._normalize_sandbox_state(getattr(sandbox, "state", None))
                        if state_token in ("STOPPED", "ARCHIVED", "PAUSED"):
                            try:
                                logger.info("daytona_starting_inactive_sandbox", target_id=target_id, state=state_token)
                                await client.start(sandbox)
                            except Exception as start_err:
                                logger.warning("daytona_sandbox_start_failed", error=str(start_err))

                        if workspace_id:
                            self._sandboxes[workspace_id] = sandbox
                        self._sandboxes[target_id] = sandbox
                        # Ensure computer_use VNC stack is started
                        if hasattr(sandbox, "computer_use"):
                            try:
                                await sandbox.computer_use.start()
                            except Exception as e:
                                logger.warning(
                                    "daytona_computer_use_start_failed",
                                    target_id=target_id,
                                    error=str(e),
                                )
                        return sandbox
                except Exception as e:
                    logger.warning("daytona_resolve_sandbox_failed", target_id=target_id, error=str(e))
                    err_msg = str(e).lower()
                    is_not_found = (
                        getattr(e, "status_code", None) == 404
                        or "404" in err_msg
                        or "not found" in err_msg
                        or "does not exist" in err_msg
                        or "expired" in err_msg
                        or "deleted" in err_msg
                    )
                    if not is_not_found:
                        return None
                    # Auto-heal: If sandbox was deleted/expired on Daytona Cloud, check existing or auto-provision a fresh one
                    try:
                        logger.info("daytona_auto_healing_checking_existing_sandboxes")
                        found_sb = None
                        try:
                            async for existing_sb in client.list():
                                found_sb = existing_sb
                                break
                        except Exception:
                            pass

                        if found_sb:
                            logger.info("daytona_auto_healing_reusing_existing_sandbox", sandbox_id=found_sb.id)
                            state_token = self._normalize_sandbox_state(getattr(found_sb, "state", None))
                            if state_token in ("STOPPED", "ARCHIVED", "PAUSED"):
                                try:
                                    await client.start(found_sb)
                                except Exception as start_err:
                                    logger.warning("daytona_auto_heal_start_failed", error=str(start_err))
                            if hasattr(found_sb, "computer_use"):
                                try:
                                    await found_sb.computer_use.start()
                                except Exception:
                                    pass
                            self._sandboxes[found_sb.id] = found_sb
                            if workspace_id:
                                self._sandboxes[workspace_id] = found_sb
                            return found_sb

                        logger.info("daytona_auto_healing_provisioning_fresh_sandbox")
                        sandbox = await client.create()
                        if sandbox:
                            if hasattr(sandbox, "computer_use"):
                                try:
                                    await sandbox.computer_use.start()
                                except Exception:
                                    pass
                            self._sandboxes[sandbox.id] = sandbox
                            if workspace_id:
                                self._sandboxes[workspace_id] = sandbox
                            return sandbox
                    except Exception as heal_err:
                        logger.error("daytona_auto_heal_failed", error=str(heal_err))
        return None

    # -------------------------------------------------------------
    # 1. Lifecycle (Create, Start, Stop, Destroy)
    # -------------------------------------------------------------

    async def create(
        self,
        tenant_id: str,
        engagement_id: str,
        workspace_type: ComputerWorkspaceType = ComputerWorkspaceType.MISSION_COMPUTER,
        profile: ComputerProfile = ComputerProfile.DEBIAN_ENGINEERING,
    ) -> ComputerWorkspace:
        """Provisions an authorized Daytona graphical workstation."""
        workspace_id = _new_id("ws-daytona")
        client = self._get_client()

        # Mission desktops are persistent engineering workstations. Research
        # labs are intentionally disposable and may use a hardened security
        # image configured separately by the operator.
        image_env = {
            ComputerWorkspaceType.RESEARCH_LAB: "DAYTONA_RESEARCH_IMAGE",
            ComputerWorkspaceType.TARGET_SANDBOX: "DAYTONA_TARGET_IMAGE",
            ComputerWorkspaceType.MISSION_COMPUTER: "DAYTONA_IMAGE",
        }[workspace_type]
        configured_image = os.environ.get(image_env, "daytonaio/sandbox:0.9.0")
        ws = ComputerWorkspace(
            id=workspace_id,
            tenant_id=tenant_id,
            engagement_id=engagement_id,
            workspace_type=workspace_type,
            profile=profile,
            provider_type="DaytonaComputerProvider",
            image=configured_image,
            status=ComputerWorkspaceStatus.CREATING,
            capabilities=["desktop", "terminal", "filesystem", "ide", "browser", "git", "computer_use"],
        )
        self.workspaces[workspace_id] = ws

        if not client or not self.api_key:
            # OFFLINE / UNCONFIGURED MODE: no Daytona credentials are present.
            # Return a local workspace object so callers can drive the
            # no-sandbox code paths (screenshot -> NO_DISPLAY, terminal ->
            # fail-closed, vnc_url -> None). The workstation API still enforces
            # its own fail-closed at the /command boundary, so this never opens
            # a host-execution path.
            ws.status = ComputerWorkspaceStatus.READY
            self._active_windows[workspace_id] = "None"
            logger.warning(
                "daytona_offline_mode",
                workspace_id=workspace_id,
                note="No DAYTONA_API_KEY configured; cloud features degrade to no-sandbox state.",
            )
            self._record_audit(
                session_id="system",
                workspace_id=workspace_id,
                tenant_id=tenant_id,
                actor="DaytonaComputerProvider",
                action="CREATE_OFFLINE_WORKSTATION",
                resource=workspace_id,
                result="SUCCESS",
            )
            return ws

        try:
            env_sandbox_id = os.environ.get("DAYTONA_SANDBOX_ID")
            configured_tenant = os.environ.get("DAYTONA_SANDBOX_TENANT_ID")
            if env_sandbox_id and configured_tenant == tenant_id:
                sandbox = await client.get(env_sandbox_id)
                if not sandbox:
                    raise RuntimeError(f"Configured Daytona sandbox '{env_sandbox_id}' was not found")
                self._sandboxes[workspace_id] = sandbox
                logger.info("daytona_sandbox_attached", workspace_id=workspace_id, sandbox_id=env_sandbox_id)
            else:
                from daytona import CreateSandboxFromImageParams

                params = CreateSandboxFromImageParams(
                    name=workspace_id,
                    image=ws.image,
                    labels={
                        "tenant_id": tenant_id,
                        "engagement_id": engagement_id,
                        "managed_by": "sonic-reda",
                        "workstation": "graphical_desktop",
                        "workspace_type": workspace_type.value,
                    },
                    auto_stop_interval=30,
                )
                sandbox = await client.create(params, timeout=120)
                if not sandbox:
                    raise RuntimeError("Daytona returned no sandbox for the create request")
                self._sandboxes[workspace_id] = sandbox

            # Use Daytona's authoritative sandbox ID as the workspace ID so
            # the control plane can reconnect after a process restart.
            remote_id = str(getattr(sandbox, "id", "") or env_sandbox_id or "")
            if remote_id and remote_id != workspace_id:
                self._sandboxes[remote_id] = sandbox
                self.workspaces.pop(workspace_id, None)
                self._active_windows.pop(workspace_id, None)
                ws.id = remote_id
                self.workspaces[remote_id] = ws
                workspace_id = remote_id

            if not hasattr(sandbox, "computer_use"):
                raise RuntimeError("Daytona sandbox does not expose the computer_use API")
            await sandbox.computer_use.start()
            ws.status = ComputerWorkspaceStatus.READY
            logger.info("daytona_computer_use_started", workspace_id=workspace_id)
        except Exception as e:
            ws.status = ComputerWorkspaceStatus.FAILED
            self._sandboxes.pop(workspace_id, None)
            self.workspaces.pop(workspace_id, None)
            logger.error("daytona_cloud_provision_failed", error=str(e), workspace_id=workspace_id)
            raise

        self._active_windows[workspace_id] = "XFCE Desktop"
        self._persist_state()
        self._record_audit(
            session_id="system",
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            actor="DaytonaComputerProvider",
            action="CREATE_GRAPHICAL_WORKSTATION",
            resource=workspace_id,
            result="SUCCESS",
        )
        return ws

    async def destroy(self, workspace_id: str) -> bool:
        """Terminates and destroys the Daytona cloud sandbox.

        Uses _resolve_sandbox (cloud lookup) so destruction succeeds even after
        a process restart when the in-memory workspace dict is empty but the
        real cloud sandbox still exists.
        """
        ws = self.workspaces.get(workspace_id)
        if ws:
            ws.status = ComputerWorkspaceStatus.DESTROYING

        try:
            sandbox = self._sandboxes.get(workspace_id)
            if sandbox is None:
                env_id = os.environ.get("DAYTONA_SANDBOX_ID", "")
                target_id = (
                    workspace_id
                    if workspace_id and not workspace_id.startswith("ws-") and workspace_id != "default"
                    else env_id or workspace_id
                )
                if target_id in (
                    "sonic-desktop-workstation",
                    "docker-workstation",
                ) or (target_id or "").startswith("docker:"):
                    container_name = (
                        "sonic-desktop-workstation"
                        if target_id in ("sonic-desktop-workstation", "docker-workstation")
                        else target_id.split(":", 1)[1]
                    )
                    sandbox = DockerContainerSandbox(container_name)
                elif target_id and (client := self._get_client()) is not None:
                    sandbox = await client.get(target_id)
            if sandbox is None:
                if ws:
                    ws.status = ComputerWorkspaceStatus.DEGRADED
                return False
            await sandbox.delete()
        except Exception:
            if ws:
                ws.status = ComputerWorkspaceStatus.DEGRADED
            logger.exception("daytona_sandbox_delete_failed", workspace_id=workspace_id)
            raise

        self.workspaces.pop(workspace_id, None)
        self._sandboxes.pop(workspace_id, None)
        self._active_windows.pop(workspace_id, None)
        self._application_pids.pop(workspace_id, None)
        self._persist_state()

        self._record_audit(
            session_id="system",
            workspace_id=workspace_id,
            tenant_id=ws.tenant_id if ws else "unknown",
            actor="DaytonaComputerProvider",
            action="DESTROY_GRAPHICAL_WORKSTATION",
            resource=workspace_id,
            result="SUCCESS",
        )
        return True

    async def get_stream_url(self, workspace_id: str) -> str | None:
        """Obtains the Daytona preview/public URL for the noVNC port (6080).

        Uses the Daytona SDK get_preview_link API. Returns None if unavailable.
        """
        sandbox = await self._resolve_sandbox(workspace_id)
        if sandbox and hasattr(sandbox, "get_preview_link"):
            try:
                preview = await sandbox.get_preview_link(6080)
                url = getattr(preview, "url", None)
                token = getattr(preview, "token", None)
                if url:
                    # Encode private-sandbox tokens before returning the URL.
                    # Tokens may contain `+`, `/`, or `=`; concatenating them
                    # raw causes browsers to send a different value and the
                    # Daytona proxy then rejects its /callback state check.
                    if token:
                        parts = urlsplit(str(url))
                        query = dict(parse_qsl(parts.query, keep_blank_values=True))
                        query["token"] = str(token)
                        return urlunsplit((
                            parts.scheme,
                            parts.netloc,
                            parts.path,
                            urlencode(query),
                            parts.fragment,
                        ))
                    return str(url)
            except Exception as e:
                logger.warning("daytona_vnc_preview_url_failed", error=str(e))
        return None

    async def get_vnc_url(self, workspace_id: str) -> str | None:
        """Alias for get_stream_url."""
        return await self.get_stream_url(workspace_id)

    async def status(self, workspace_id: str) -> ComputerState:
        """Returns the real-time operational state of the graphical desktop."""
        ws = self.workspaces.get(workspace_id)
        tenant_id = ws.tenant_id if ws else ""
        # Query real running processes from sandbox
        real_processes = []
        resource_usage = {"cpu_pct": 0.0, "memory_mb": 0.0}
        sandbox = await self._resolve_sandbox(workspace_id)

        # Query real open windows using wmctrl
        disp = self._get_display(workspace_id)
        open_windows: list[str] = []
        try:
            wm_res = await self.terminal(workspace_id, f"DISPLAY={disp} wmctrl -l")
            if wm_res.exit_code == 0 and wm_res.stdout:
                for line in wm_res.stdout.splitlines():
                    parts = line.split(maxsplit=3)
                    if len(parts) >= 4:
                        win_title = parts[3].strip()
                        if win_title not in ("xfce4-panel", "Desktop") and win_title not in open_windows:
                            open_windows.append(win_title)
        except Exception:
            pass

        active_app = "Desktop"
        try:
            act_res = await self.terminal(workspace_id, f"DISPLAY={disp} xdotool getactivewindow getwindowname 2>/dev/null")
            if act_res.exit_code == 0 and act_res.stdout and act_res.stdout.strip():
                active_app = act_res.stdout.strip()
            else:
                active_app = "UNKNOWN"
        except Exception:
            active_app = "UNKNOWN"
        # Derive real workspace status from the sandbox state.
        # Daytona returns a SandboxState enum whose str() includes the
        # enum name (e.g. "<SandboxState.STARTED: 'started'>"); normalize to
        # the bare value so a freshly provisioned sandbox reports RUNNING
        # instead of falling through to the default STOPPED.
        ws_status = ComputerWorkspaceStatus.STOPPED
        if sandbox:
            raw_state = self._normalize_sandbox_state(getattr(sandbox, "state", None))
            if raw_state in ("STARTED", "RUNNING"):
                ws_status = ComputerWorkspaceStatus.RUNNING
            elif raw_state == "STOPPED":
                ws_status = ComputerWorkspaceStatus.STOPPED
            elif raw_state in ("CREATING", "BUILDING"):
                ws_status = ComputerWorkspaceStatus.CREATING
            elif raw_state in ("ERROR", "FAILED"):
                ws_status = ComputerWorkspaceStatus.FAILED
            if hasattr(sandbox, "computer_use"):
                try:
                    cu_status = await sandbox.computer_use.get_status()
                    if cu_status and hasattr(cu_status, "status"):
                        real_processes = [str(cu_status.status)]
                except Exception:
                    pass
                try:
                    metrics = await sandbox.get_metrics_latest()
                    if metrics:
                        resource_usage = {
                            "cpu_pct": getattr(metrics, "cpu_usage_percent", 0.0) or 0.0,
                            "memory_mb": getattr(metrics, "mem_usage_bytes", 0) / (1024 * 1024) if getattr(metrics, "mem_usage_bytes", None) else 0.0,
                        }
                except Exception:
                    pass

        # Query installed applications in the sandbox (empty if sandbox unavailable)
        installed_apps = await self.application_list(workspace_id)

        # Query real git branch and working directory if repository is present.
        # When the sandbox is offline (terminal fail-closes), fall back to the
        # workspace's declared workspace_path rather than a hardcoded path so the
        # reported working directory reflects the configured environment.
        default_workdir = ws.workspace_path if ws else "/home/sonic/workspace"
        git_res = await self.terminal(workspace_id, "git branch --show-current 2>/dev/null")
        git_branch = git_res.stdout.strip() if (git_res.exit_code == 0 and git_res.stdout.strip()) else ""
        pwd_res = await self.terminal(workspace_id, "pwd")
        working_dir = pwd_res.stdout.strip() if (pwd_res.exit_code == 0 and pwd_res.stdout.strip()) else default_workdir

        return ComputerState(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            status=ws_status,
            active_application=active_app,
            open_applications=open_windows or ([active_app] if active_app != "None" else []),
            active_window=active_app,
            working_directory=working_dir,
            running_processes=real_processes,
            installed_applications=installed_apps,
            current_project="sonic",
            git_branch=git_branch,
            resource_usage=resource_usage,
        )

    # -------------------------------------------------------------
    # 2. Real Graphical Screen & Vision (Zero Fabricated Frames)
    # -------------------------------------------------------------

    async def screenshot(self, workspace_id: str) -> ScreenObservation:
        """
        Capture a fresh screenshot and derive the same pixel-only AVCS state as Docker.
        Unavailable, malformed, stale, or unprocessable frames never reuse cached pixels.
        """
        self._last_screenshot = None
        self._last_screenshot_time = 0.0
        self._last_screenshot_workspace_id = None
        sandbox = await self._resolve_sandbox(workspace_id)
        if sandbox:
            b64 = ""
            timestamp_ns: int | None = None
            if hasattr(sandbox, "computer_use"):
                try:
                    capture_started_ns = time.time_ns()
                    response = await sandbox.computer_use.screenshot.take_full_screen()
                    b64 = getattr(response, "screenshot", None) or ""
                    response_timestamp = getattr(response, "capture_timestamp_ns", None)
                    timestamp_ns = (
                        response_timestamp
                        if isinstance(response_timestamp, int)
                        and not isinstance(response_timestamp, bool)
                        and response_timestamp > 0
                        else capture_started_ns
                    )
                except Exception as e:
                    logger.warning("daytona_direct_screenshot_failed", error=str(e))

            if not b64:
                try:
                    disp = self._get_display(workspace_id)
                    scr_cmd = (
                        "printf '%s\\n' '___CAPTURE_TIMESTAMP_NS___'; date +%s%N; "
                        f"DISPLAY={shlex.quote(disp)} import -window root png:- | base64 -w0"
                    )
                    res = await sandbox.process.exec(scr_cmd)
                    if res.exit_code == 0 and res.result and len(res.result.strip()) > 100:
                        marker, _, capture_result = str(res.result).partition(
                            "___CAPTURE_TIMESTAMP_NS___"
                        )
                        if marker:
                            timestamp_text, separator, encoded = capture_result.lstrip(
                                "\r\n"
                            ).partition("\n")
                            if separator:
                                try:
                                    timestamp_ns = int(timestamp_text.strip())
                                except ValueError:
                                    timestamp_ns = None
                                b64 = encoded.strip()
                except Exception as e:
                    logger.warning("daytona_x11_screenshot_failed", error=str(e))

            try:
                png_bytes = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError, TypeError):
                png_bytes = b""
            dimensions = _validated_png_dimensions(png_bytes)
            if dimensions is not None and timestamp_ns is not None and timestamp_ns > 0:
                age_ns = time.time_ns() - timestamp_ns
                capture_age_ms = age_ns / 1_000_000 if age_ns >= 0 else None
                if (
                    capture_age_ms is not None
                    and math.isfinite(capture_age_ms)
                    and capture_age_ms <= _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                ):
                    try:
                        if self._avcs_perception is None:
                            from sonic.computer_use.avcs_perception import AVCSPerception

                            self._avcs_perception = AVCSPerception()
                        avcs_state = self._avcs_perception.perceive_png(png_bytes)
                    except Exception as e:
                        logger.error(
                            "daytona_avcs_perception_failed",
                            workspace_id=workspace_id,
                            error=str(e),
                        )
                        raise RuntimeError(
                            "AVCS perception failed for the Daytona workstation frame"
                        ) from e
                    if (
                        not isinstance(avcs_state, dict)
                        or avcs_state.get("engine") != "avcs_reflex"
                        or isinstance(avcs_state.get("frame_index"), bool)
                        or not isinstance(avcs_state.get("frame_index"), int)
                        or avcs_state["frame_index"] <= 0
                        or avcs_state.get("width") != dimensions[0]
                        or avcs_state.get("height") != dimensions[1]
                        or not isinstance(avcs_state.get("scene_hash"), str)
                        or not avcs_state["scene_hash"]
                    ):
                        logger.error(
                            "daytona_avcs_invalid_state",
                            workspace_id=workspace_id,
                        )
                        raise RuntimeError(
                            "AVCS returned invalid state for the Daytona workstation frame"
                        )

                    active_window = "UNKNOWN"
                    try:
                        active_result = await sandbox.process.exec(
                            f"DISPLAY={self._get_display(workspace_id)} "
                            "xdotool getactivewindow getwindowname"
                        )
                        if (
                            getattr(active_result, "exit_code", 1) == 0
                            and str(getattr(active_result, "result", "") or "").strip()
                        ):
                            active_window = str(active_result.result).strip()
                    except Exception as e:
                        logger.debug("daytona_active_window_unavailable", error=str(e))

                    final_age_ns = time.time_ns() - timestamp_ns
                    final_capture_age_ms = (
                        final_age_ns / 1_000_000 if final_age_ns >= 0 else None
                    )
                    if (
                        final_capture_age_ms is None
                        or not math.isfinite(final_capture_age_ms)
                        or final_capture_age_ms > _SCREEN_CAPTURE_FRESHNESS_BUDGET_MS
                    ):
                        logger.warning(
                            "daytona_screenshot_became_stale_during_perception",
                            workspace_id=workspace_id,
                            capture_age_ms=final_capture_age_ms,
                        )
                        return ScreenObservation(
                            screenshot_base64="",
                            capture_timestamp_ns=timestamp_ns,
                            capture_age_ms=final_capture_age_ms,
                            width=dimensions[0],
                            height=dimensions[1],
                            active_window=active_window,
                            visible_text=(
                                "Desktop capture unavailable: frame freshness could not be "
                                "verified after AVCS processing."
                            ),
                            detected_controls=[],
                            desktop_state="NO_DISPLAY",
                        )
                    capture_age_ms = final_capture_age_ms

                    observation = ScreenObservation(
                        screenshot_base64=b64,
                        avcs_state=avcs_state,
                        capture_timestamp_ns=timestamp_ns,
                        capture_age_ms=capture_age_ms,
                        perception_version=int(avcs_state["frame_index"]),
                        width=dimensions[0],
                        height=dimensions[1],
                        active_window=active_window,
                        visible_text="",
                        detected_controls=[],
                        desktop_state="INTERACTIVE",
                    )
                    self._last_screenshot = observation
                    self._last_screenshot_time = asyncio.get_running_loop().time()
                    self._last_screenshot_workspace_id = workspace_id
                    return observation

            logger.warning(
                "daytona_screenshot_invalid_or_stale",
                workspace_id=workspace_id,
                dimensions_valid=dimensions is not None,
                timestamp_valid=timestamp_ns is not None and timestamp_ns > 0,
            )

        # NO_DISPLAY: no live desktop frame exists — never fabricate a pixel
        return ScreenObservation(
            screenshot_base64="",
            width=1280,
            height=800,
            active_window="UNKNOWN",
            visible_text="Desktop capture unavailable: a valid fresh AVCS frame could not be verified.",
            detected_controls=[],
            desktop_state="NO_DISPLAY",
        )

    # -------------------------------------------------------------
    # 3. Real GUI Action Dispatch (Mouse, Keyboard, Window Management)
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

    async def _move_mouse_smoothly(self, sandbox: Any, workspace_id: str, x: int, y: int) -> None:
        """Move the visible pointer along an eased path using the active sandbox."""
        display = self._get_display(workspace_id)
        position = await sandbox.process.exec(
            f"DISPLAY={display} xdotool getmouselocation --shell"
        )
        if getattr(position, "exit_code", 1) != 0:
            raise RuntimeError("Unable to read cursor position for smooth movement")
        output = str(getattr(position, "result", "") or "")
        x_match = re.search(r"(?m)^X=(\d+)$", output)
        y_match = re.search(r"(?m)^Y=(\d+)$", output)
        if x_match is None or y_match is None:
            raise RuntimeError("Cursor position response was incomplete")

        start_x, start_y = int(x_match.group(1)), int(y_match.group(1))
        for step in range(1, 26):
            progress = step / 25
            eased = 0.5 - 0.5 * math.cos(progress * math.pi)
            next_x = round(start_x + (x - start_x) * eased)
            next_y = round(start_y + (y - start_y) * eased)
            if hasattr(sandbox, "computer_use"):
                await sandbox.computer_use.mouse.move(next_x, next_y)
            else:
                movement = await sandbox.process.exec(
                    f"DISPLAY={display} xdotool mousemove {next_x} {next_y}"
                )
                if getattr(movement, "exit_code", 1) != 0:
                    raise RuntimeError("Cursor movement failed during smooth interpolation")
            await asyncio.sleep(0.01)

    async def gui_action(
        self,
        workspace_id: str,
        action: GUIAction,
        actor: str = "operator",
    ) -> ScreenObservation:
        """
        Dispatches authentic mouse and keyboard events directly into the remote X11 desktop.
        """
        sandbox = await self._resolve_sandbox(workspace_id)
        if not sandbox:
            raise RuntimeError("GUI action cannot execute because no live sandbox is attached")
        action_type = action.action
        pointer_points: tuple[tuple[int | None, int | None], ...] = ()
        if action_type in (
            GUIActionType.CLICK,
            GUIActionType.DOUBLE_CLICK,
            GUIActionType.RIGHT_CLICK,
            GUIActionType.MOVE,
            GUIActionType.SCROLL,
        ):
            pointer_points = ((action.x, action.y),)
        elif action_type == GUIActionType.DRAG:
            pointer_points = ((action.x, action.y), (action.x2, action.y2))
        if pointer_points:
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for point in pointer_points
                for value in point
            ):
                raise ValueError("Pointer actions require explicit integer coordinates")
            now = asyncio.get_running_loop().time()
            observation = self._last_screenshot
            cache_age_ms = (
                max(0.0, now - self._last_screenshot_time) * 1000
                if (
                    observation is not None
                    and self._last_screenshot_time > 0
                    and self._last_screenshot_workspace_id == workspace_id
                )
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
            if observation.desktop_state != "INTERACTIVE" or observation.capture_age_ms is None:
                raise RuntimeError("Pointer action blocked: no fresh AVCS screen bounds")
            for cx, cy in pointer_points:
                if (
                    cx is None
                    or cy is None
                    or cx < 0
                    or cy < 0
                    or cx >= observation.width
                    or cy >= observation.height
                ):
                    raise ValueError(
                        f"GUI coordinates ({cx}, {cy}) outside captured screen bounds "
                        f"{observation.width}x{observation.height}"
                    )

        self._last_screenshot = None
        self._last_screenshot_time = 0.0
        self._last_screenshot_workspace_id = None

        supported_actions = {
            GUIActionType.CLICK,
            GUIActionType.DOUBLE_CLICK,
            GUIActionType.RIGHT_CLICK,
            GUIActionType.TYPE,
            GUIActionType.KEYPRESS,
            GUIActionType.HOTKEY,
            GUIActionType.MOVE,
            GUIActionType.SCROLL,
            GUIActionType.DRAG,
            GUIActionType.SELECT_WINDOW,
            GUIActionType.APP_MAXIMIZE,
            GUIActionType.WINDOW_MAXIMIZE,
            GUIActionType.WINDOW_TILE,
            GUIActionType.OPEN_APP,
            GUIActionType.LAUNCH_APP,
            GUIActionType.CLOSE_APP,
            GUIActionType.CLOSE_WINDOW,
            GUIActionType.INSTALL_APP,
            GUIActionType.WAIT,
        }
        if action_type not in supported_actions:
            raise RuntimeError(f"Daytona GUI action {action_type.value} is not supported by this provider")

        if action_type in [GUIActionType.CLICK, GUIActionType.DOUBLE_CLICK, GUIActionType.RIGHT_CLICK]:
            if action.x is None or action.y is None:
                raise ValueError(f"{action_type.value} requires explicit x and y coordinates")

        try:
            if hasattr(sandbox, "computer_use"):
                cu = sandbox.computer_use
                if action_type in [GUIActionType.CLICK, GUIActionType.DOUBLE_CLICK]:
                    await self._move_mouse_smoothly(sandbox, workspace_id, action.x, action.y)
                    mouse_button = {1: "left", 2: "middle", 3: "right"}.get(action.button or 1)
                    if mouse_button is None:
                        raise ValueError(f"Unsupported mouse button: {action.button}")
                    await cu.mouse.click(
                        action.x,
                        action.y,
                        button=mouse_button,
                        double=(action_type == GUIActionType.DOUBLE_CLICK),
                    )
                elif action_type == GUIActionType.RIGHT_CLICK:
                    await self._move_mouse_smoothly(sandbox, workspace_id, action.x, action.y)
                    await cu.mouse.click(action.x, action.y, button="right")
                elif action_type == GUIActionType.MOVE:
                    await self._move_mouse_smoothly(sandbox, workspace_id, action.x, action.y)
                elif action_type == GUIActionType.DRAG:
                    sx, sy = action.x, action.y
                    dx, dy = action.x2, action.y2
                    if sx is not None and sy is not None and dx is not None and dy is not None:
                        await self._move_mouse_smoothly(sandbox, workspace_id, sx, sy)
                        await cu.mouse.down()
                        try:
                            await self._move_mouse_smoothly(sandbox, workspace_id, dx, dy)
                        finally:
                            await cu.mouse.up()
                elif action_type == GUIActionType.TYPE and action.text:
                    if action.x is not None and action.y is not None:
                        await self._move_mouse_smoothly(
                            sandbox, workspace_id, action.x, action.y
                        )
                        await cu.mouse.click(action.x, action.y, button="left")
                    await cu.keyboard.type(action.text)
                elif action_type in (GUIActionType.KEYPRESS, GUIActionType.HOTKEY) and (action.key or action.text):
                    await cu.keyboard.press(action.key or action.text)
            else:
                # Direct X11 dispatch via xdotool on DISPLAY={disp}
                disp = self._get_display(workspace_id)
                if action_type in [GUIActionType.CLICK, GUIActionType.DOUBLE_CLICK]:
                    await self._move_mouse_smoothly(sandbox, workspace_id, action.x, action.y)
                    repeat = " --repeat 2" if action_type == GUIActionType.DOUBLE_CLICK else ""
                    clicked = await sandbox.process.exec(f"DISPLAY={disp} xdotool click{repeat} 1")
                    if getattr(clicked, "exit_code", 1) != 0:
                        raise RuntimeError("Mouse click failed after smooth movement")
                elif action_type == GUIActionType.RIGHT_CLICK:
                    await self._move_mouse_smoothly(sandbox, workspace_id, action.x, action.y)
                    clicked = await sandbox.process.exec(f"DISPLAY={disp} xdotool click 3")
                    if getattr(clicked, "exit_code", 1) != 0:
                        raise RuntimeError("Right-click failed after smooth movement")
                elif action_type == GUIActionType.MOVE:
                    await self._move_mouse_smoothly(sandbox, workspace_id, action.x, action.y)
                elif action_type == GUIActionType.DRAG:
                    sx, sy, dx, dy = action.x, action.y, action.x2, action.y2
                    if sx is not None and sy is not None and dx is not None and dy is not None:
                        await self._move_mouse_smoothly(sandbox, workspace_id, sx, sy)
                        pressed = await sandbox.process.exec(f"DISPLAY={disp} xdotool mousedown 1")
                        if getattr(pressed, "exit_code", 1) != 0:
                            raise RuntimeError("Mouse press failed before drag")
                        try:
                            await self._move_mouse_smoothly(sandbox, workspace_id, dx, dy)
                        finally:
                            released = await sandbox.process.exec(f"DISPLAY={disp} xdotool mouseup 1")
                            if getattr(released, "exit_code", 1) != 0:
                                raise RuntimeError("Mouse release failed after drag")
                elif action_type == GUIActionType.TYPE and action.text:
                    safe_text = shlex.quote(action.text)
                    focus = ""
                    if action.x is not None and action.y is not None:
                        focus = (
                            f"DISPLAY={disp} xdotool click 1 && "
                        )
                        await self._move_mouse_smoothly(
                            sandbox, workspace_id, int(action.x), int(action.y)
                        )
                    typed = await sandbox.process.exec(
                        f"{focus}DISPLAY={disp} xdotool type --delay {int(action.delay_ms)} "
                        f"--clearmodifiers -- {safe_text}"
                    )
                    if getattr(typed, "exit_code", 1) != 0:
                        raise RuntimeError("Text entry command failed")
                elif action_type in (GUIActionType.KEYPRESS, GUIActionType.HOTKEY) and (action.key or action.text):
                    norm_key = self._normalize_x11_keysym(action.key or action.text or "")
                    safe_key = shlex.quote(norm_key)
                    pressed = await sandbox.process.exec(
                        f"DISPLAY={disp} xdotool key --clearmodifiers {safe_key}"
                    )
                    if getattr(pressed, "exit_code", 1) != 0:
                        raise RuntimeError("Keyboard action failed")

            # Common handlers for SCROLL, APPS, WINDOWS
            disp = self._get_display(workspace_id)
            if action_type == GUIActionType.SCROLL:
                delta = getattr(action, 'scroll_delta', -3)
                button = 4 if delta > 0 else 5
                clicks = abs(delta)
                cmd_parts = []
                if getattr(action, "x", None) is not None and getattr(action, "y", None) is not None:
                    await self._move_mouse_smoothly(
                        sandbox,
                        workspace_id,
                        int(action.x),
                        int(action.y),
                    )
                for _ in range(clicks):
                    cmd_parts.append(f"DISPLAY={disp} xdotool click {button}")
                scroll_cmd = " && ".join(cmd_parts)
                scrolled = await sandbox.process.exec(scroll_cmd)
                if getattr(scrolled, "exit_code", 1) != 0:
                    raise RuntimeError("Scroll action failed")

            elif action_type in (GUIActionType.OPEN_APP, GUIActionType.LAUNCH_APP) and action.app_name:
                raw_name = action.app_name.strip(" *_\n\r\t`\"'")
                if "\n" in raw_name:
                    raw_name = raw_name.split("\n")[0].strip(" *_\n\r\t`\"'")
                clean_app = raw_name.strip()
                if clean_app:
                    parts = shlex.split(clean_app)
                    spawn_cmd = (
                        f"DISPLAY={disp} nohup {shlex.join(parts)} "
                        ">/dev/null 2>&1 </dev/null & app_pid=$!; "
                        "printf '%s\\n' \"$app_pid\""
                    )
                    launch = await sandbox.process.exec(spawn_cmd)
                    if getattr(launch, "exit_code", 1) != 0:
                        raise RuntimeError("Application launch command failed")
                    pid_lines = str(getattr(launch, "result", "") or "").strip().splitlines()
                    try:
                        app_pid = int(pid_lines[-1]) if pid_lines else 0
                    except ValueError:
                        app_pid = 0
                    if app_pid <= 0:
                        raise RuntimeError("Application launch returned no verifiable process ID")
                    check = await sandbox.process.exec(f"ps -p {app_pid} -o pid=")
                    if (
                        getattr(check, "exit_code", 1) != 0
                        or str(getattr(check, "result", "") or "").strip() != str(app_pid)
                    ):
                        raise RuntimeError("Launched application process is not running")
                    self._application_pids.setdefault(workspace_id, {}).setdefault(
                        parts[0], set()
                    ).add(app_pid)

            elif action_type == GUIActionType.CLOSE_WINDOW:
                target = action.window_id or action.app_name
                if target:
                    safe_target = shlex.quote(target)
                    closed = await sandbox.process.exec(
                        f"export DISPLAY={disp}; wmctrl -c {safe_target}"
                    )
                    if getattr(closed, "exit_code", 1) != 0:
                        raise RuntimeError(f"Could not close window {target!r}")
                else:
                    closed = await sandbox.process.exec(
                        f"DISPLAY={disp} xdotool getactivewindow windowclose"
                    )
                    if getattr(closed, "exit_code", 1) != 0:
                        raise RuntimeError("Could not close the active window")

            elif action_type == GUIActionType.CLOSE_APP and action.app_name:
                process_map = self._application_pids.get(workspace_id, {})
                app_parts = shlex.split(action.app_name)
                app_key = app_parts[0] if app_parts else ""
                process_ids = set(process_map.get(app_key, set()))
                if not process_ids:
                    raise RuntimeError("No tracked process is available to close this application")
                for process_id in process_ids:
                    stopped = await sandbox.process.exec(f"kill -TERM -- {process_id}")
                    if getattr(stopped, "exit_code", 1) != 0:
                        check = await sandbox.process.exec(f"ps -p {process_id} -o pid=")
                        if (
                            getattr(check, "exit_code", 1) == 0
                            and str(getattr(check, "result", "") or "").strip()
                        ):
                            raise RuntimeError(f"Could not stop process {process_id}")
                    for _ in range(20):
                        check = await sandbox.process.exec(f"ps -p {process_id} -o pid=")
                        if (
                            getattr(check, "exit_code", 1) != 0
                            or not str(getattr(check, "result", "") or "").strip()
                        ):
                            break
                        await asyncio.sleep(0.1)
                    else:
                        raise RuntimeError(f"Process {process_id} is still running")
                process_map.pop(app_key, None)

            elif action_type == GUIActionType.INSTALL_APP and (action.app_name or action.text):
                pkg = action.app_name or action.text or ""
                ok, err = await self.install_application(workspace_id, pkg, actor=actor)
                if not ok:
                    raise RuntimeError(f"Daytona GUI install_application failed for '{pkg}': {err}")

            elif action_type == GUIActionType.WAIT:
                await asyncio.sleep(max(0.1, (action.delay_ms or 500) / 1000.0))

            elif action_type == GUIActionType.SELECT_WINDOW:
                title = action.app_name or action.window_id or ""
                if title:
                    safe = shlex.quote(title)
                    selected = await sandbox.process.exec(
                        f"export DISPLAY={disp}; wmctrl -a {safe}"
                    )
                    if getattr(selected, "exit_code", 1) != 0:
                        raise RuntimeError(f"Could not focus window {title!r}")
                    active = await sandbox.process.exec(
                        f"DISPLAY={disp} xdotool getactivewindow getwindowname"
                    )
                    actual_title = str(getattr(active, "result", "") or "").strip()
                    if (
                        getattr(active, "exit_code", 1) != 0
                        or title.casefold() not in actual_title.casefold()
                    ):
                        raise RuntimeError(f"Window focus could not be verified for {title!r}")

            elif action_type in (GUIActionType.APP_MAXIMIZE, GUIActionType.WINDOW_MAXIMIZE):
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
                maximized = await sandbox.process.exec(max_cmd)
                if getattr(maximized, "exit_code", 1) != 0:
                    raise RuntimeError("Window maximize command failed")

            elif action_type == GUIActionType.WINDOW_TILE:
                if not await self.tile_workstation(workspace_id):
                    raise RuntimeError("Window tiling command failed")

        except Exception as e:
            logger.error("daytona_gui_action_dispatch_error", error=str(e))
            raise RuntimeError(f"Daytona GUI action failed: {e}") from e

        ws = self.workspaces.get(workspace_id)
        self._record_audit(
            session_id=ws.engagement_id if ws else "unknown",
            workspace_id=workspace_id,
            tenant_id=ws.tenant_id if ws else actor,
            actor=actor,
            action=f"GUI_{action_type.value}",
            resource=action.app_name or "screen",
            result="DISPATCHED",
            details=action.model_dump(),
        )
        observation = await self.screenshot(workspace_id)
        if observation.active_window != "UNKNOWN":
            self._active_windows[workspace_id] = observation.active_window
        return observation

    # -------------------------------------------------------------
    # 4. Real PTY Terminal & Command Execution (Sandbox-Bound)
    # -------------------------------------------------------------

    async def terminal(
        self,
        workspace_id: str,
        command: str,
        timeout: int = 60,
        actor: str = "operator",
    ) -> ExecResult:
        """
        Executes bash commands strictly inside the remote sandbox container.
        FAIL-CLOSED: Host execution is strictly forbidden.
        """
        sandbox = await self._resolve_sandbox(workspace_id)
        start_time = datetime.now(UTC)

        # 1. Try Daytona SDK execution
        if sandbox and hasattr(sandbox, "process"):
            try:
                ts = int(start_time.timestamp() * 1000)
                out_marker = f"/tmp/.sonic_stdout_{ts}"
                err_marker = f"/tmp/.sonic_stderr_{ts}"
                delim = "___SONIC_STDERR_DELIM___"
                wrapped = (
                    f"{{ {command} ; }} > {out_marker} 2> {err_marker}; "
                    f"__sonic_ec=$?; cat {out_marker} 2>/dev/null; "
                    f"echo '{delim}'; cat {err_marker} 2>/dev/null; "
                    f"rm -f {out_marker} {err_marker}; exit $__sonic_ec"
                )
                res = await sandbox.process.exec(wrapped, timeout=timeout)
                duration = (datetime.now(UTC) - start_time).total_seconds()
                raw_result = res.result or ""
                exit_code = getattr(res, "exit_code", 0)

                if delim in raw_result:
                    parts = raw_result.split(delim, 1)
                    stdout = parts[0]
                    stderr = parts[1].lstrip("\r\n")
                else:
                    stdout = raw_result
                    stderr = ""

                return ExecResult(
                    command=command,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    sandbox_id=workspace_id,
                )
            except Exception as e:
                logger.warning("daytona_process_exec_failed", error=str(e))

        # FAIL CLOSED
        return ExecResult(
            command=command,
            exit_code=126,
            stdout="",
            stderr="FAIL-CLOSED: Dedicated sandbox container is unreachable. Host execution is strictly prohibited.",
            duration_seconds=0.0,
            sandbox_id=workspace_id,
        )

    # -------------------------------------------------------------
    # 5. Real Remote Filesystem Operations
    # -------------------------------------------------------------

    async def read_file(self, workspace_id: str, path: str) -> str:
        """Reads a file from the sandbox container filesystem."""
        sandbox = await self._resolve_sandbox(workspace_id)
        if sandbox and hasattr(sandbox, "fs") and hasattr(sandbox.fs, "download_file"):
            try:
                content = await sandbox.fs.download_file(path)
                return content.decode("utf-8") if isinstance(content, bytes) else str(content)
            except Exception as e:
                logger.warning("daytona_fs_read_failed", error=str(e), path=path)

        res = await self.terminal(workspace_id, f"cat -- {shlex.quote(path)}")
        if res.exit_code == 0:
            return res.stdout
        return f"# Error reading file {path} from sandbox"

    async def write_file(self, workspace_id: str, path: str, content: str, actor: str = "operator") -> bool:
        """Writes a file to the sandbox container filesystem."""
        sandbox = await self._resolve_sandbox(workspace_id)
        if sandbox and hasattr(sandbox, "fs") and hasattr(sandbox.fs, "upload_file"):
            try:
                await sandbox.fs.upload_file(src=content.encode("utf-8"), dst=path)
                return True
            except Exception as e:
                logger.warning("daytona_fs_write_failed", error=str(e), path=path)

        # No alternate-container or host fallback.  A Daytona workstation must
        # expose its filesystem API for writes to be considered successful.
        return False

    async def list_files(self, workspace_id: str, path: str = ".") -> list[FileEntry]:
        """Lists files inside the sandbox directory."""
        res = await self.terminal(workspace_id, f"ls -la -- {shlex.quote(path)}")
        entries = []
        if res.exit_code == 0:
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 9 and parts[8] not in [".", ".."]:
                    name = parts[8]
                    is_dir = line.startswith("d")
                    size = int(parts[4]) if parts[4].isdigit() else 0
                    entries.append(
                        FileEntry(
                            name=name,
                            path=f"{path.rstrip('/')}/{name}",
                            is_dir=is_dir,
                            size_bytes=size,
                            modified_at=_now(),
                            permissions=parts[0],
                        )
                    )
        return entries

    async def process_list(self, workspace_id: str) -> list[ProcessInfo]:
        """Lists running processes inside the sandbox by querying real process state."""
        processes: list[ProcessInfo] = []
        sandbox = await self._resolve_sandbox(workspace_id)

        # Query real processes via computer_use status API
        if sandbox and hasattr(sandbox, "computer_use"):
            try:
                for proc_name in ["xvfb", "xfce4", "x11vnc", "novnc"]:
                    try:
                        pstatus = await sandbox.computer_use.get_process_status(proc_name)
                        if pstatus and hasattr(pstatus, "status"):
                            processes.append(
                                ProcessInfo(
                                    pid=getattr(pstatus, "pid", 0) or 0,
                                    name=proc_name,
                                    cpu_pct=0.0,
                                    memory_mb=0.0,
                                    status=str(pstatus.status),
                                )
                            )
                    except Exception:
                        pass
                if processes:
                    return processes
            except Exception:
                pass

        # Fallback: query via ps command inside sandbox
        if sandbox and hasattr(sandbox, "process"):
            try:
                res = await sandbox.process.exec("ps aux --no-headers 2>/dev/null | head -20")
                stdout = getattr(res, "result", "") or ""
                for line in stdout.splitlines():
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        try:
                            processes.append(
                                ProcessInfo(
                                    pid=int(parts[1]),
                                    name=parts[10].split()[0].split("/")[-1],
                                    cpu_pct=float(parts[2]),
                                    memory_mb=float(parts[5]) / 1024.0 if parts[5].isdigit() else 0.0,
                                    status="RUNNING",
                                )
                            )
                        except (ValueError, IndexError):
                            pass
            except Exception:
                pass

        return processes

    async def application_list(self, workspace_id: str) -> list[str]:
        """Lists installed applications in the sandbox by dynamically discovering desktop entries and real binaries."""
        apps: list[str] = []
        std_utils = list(dict.fromkeys(self.app_policy.allowed_packages))
        utils_str = " ".join(std_utils)
        discovery_cmd = (
            "find /usr/share/applications /usr/local/share/applications ~/.local/share/applications -name '*.desktop' 2>/dev/null | while read -r f; do "
            "[ -f \"$f\" ] || continue; "
            "b=$(basename \"$f\" .desktop); echo \"$b\"; "
            "ex=$(grep -m1 -E '^Exec=' \"$f\" 2>/dev/null | cut -d= -f2- | awk '{print $1}'); "
            "[ -n \"$ex\" ] && basename \"$ex\"; "
            "done; "
            "find /usr/local/bin -maxdepth 1 -type f 2>/dev/null | while read -r p; do [ -x \"$p\" ] && basename \"$p\"; done; "
            f"for b in {utils_str}; do command -v \"$b\" 2>/dev/null; done; true"
        )

        def _parse_apps(raw: str) -> list[str]:
            found: list[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                app_name = line.rsplit("/", 1)[-1].strip()
                if app_name.endswith(".desktop"):
                    app_name = app_name[:-8]
                app_name = app_name.strip("\"' ")
                if app_name and app_name not in found:
                    found.append(app_name)
            return found

        sandbox = await self._resolve_sandbox(workspace_id)
        if sandbox and hasattr(sandbox, "process"):
            try:
                res = await sandbox.process.exec(discovery_cmd)
                stdout = getattr(res, "result", "") or ""
                apps = _parse_apps(stdout)
                if apps:
                    return apps
            except Exception:
                pass

        try:
            res = await self.terminal(workspace_id, discovery_cmd)
            if res.exit_code == 0 and res.stdout:
                apps = _parse_apps(res.stdout)
                if apps:
                    return apps
        except Exception:
            pass

        return apps

    async def launch_application(self, workspace_id: str, app_name: str, actor: str = "operator") -> bool:
        """Launches a GUI application."""
        clean_app = (app_name or "").strip()
        if not clean_app:
            return False
        parts = shlex.split(clean_app)
        if not parts:
            return False
        if hasattr(self, "app_policy") and self.app_policy:
            allowed, _ = self.app_policy.is_package_allowed(parts[0])
            if not allowed:
                return False
        disp = self._get_display(workspace_id)
        spawn = (
            f"DISPLAY={disp} nohup {shlex.join(parts)} "
            ">/dev/null 2>&1 </dev/null & app_pid=$!; "
            "sleep 0.1; kill -0 \"$app_pid\" || exit 1; "
            "printf '%s\\n' \"$app_pid\""
        )
        result = await self.terminal(workspace_id, spawn)
        pid_lines = result.stdout.strip().splitlines()
        try:
            app_pid = int(pid_lines[-1]) if pid_lines else 0
        except ValueError:
            app_pid = 0
        if result.exit_code != 0 or app_pid <= 0:
            return False
        self._application_pids.setdefault(workspace_id, {}).setdefault(
            parts[0], set()
        ).add(app_pid)
        return True

    async def tile_workstation(self, workspace_id: str) -> bool:
        """Executes wmctrl commands to tile windows side-by-side."""
        disp = self._get_display(workspace_id)
        result = await self.terminal(
            workspace_id,
            f"DISPLAY={disp} wmctrl -l 2>/dev/null | "
            "awk 'BEGIN {i=0} {if (NF >= 4) {x=(i++ % 2) * 640; "
            "printf \"wmctrl -i -r %s -e 0,%d,0,640,800; \", $1, x}}' | "
            "sh",
        )
        return result.exit_code == 0

    async def close(self) -> None:
        """Close the underlying Daytona SDK client (releases its aiohttp session)."""
        if self._client is not None:
            try:
                close = getattr(self._client, "close", None)
                if close is not None:
                    res = close()
                    if hasattr(res, "__await__"):
                        await res
            except Exception as e:
                logger.warning("daytona_client_close_failed", error=str(e))
            finally:
                self._client = None

    async def close_application(self, workspace_id: str, app_name: str, actor: str = "operator") -> bool:
        """Closes a running desktop application."""
        process_map = self._application_pids.get(workspace_id, {})
        app_parts = shlex.split(app_name)
        app_key = app_parts[0] if app_parts else ""
        process_ids = set(process_map.get(app_key, set()))
        if not process_ids:
            return False
        for process_id in process_ids:
            stopped = await self.terminal(
                workspace_id, f"kill -TERM -- {process_id}"
            )
            for _ in range(20):
                check = await self.terminal(workspace_id, f"ps -p {process_id} -o pid=")
                if check.exit_code != 0 or not check.stdout.strip():
                    break
                await asyncio.sleep(0.1)
            else:
                return False
            if stopped.exit_code != 0 and check.stdout.strip():
                return False
        process_map.pop(app_key, None)
        return True

    async def install_application(self, workspace_id: str, package_name: str, actor: str = "operator") -> tuple[bool, str]:
        """Installs an application inside the sandbox."""
        allowed, reason = self.app_policy.is_package_allowed(package_name)
        if not allowed:
            return False, reason
        safe_package = shlex.quote(package_name.strip())
        res = await self.terminal(workspace_id, f"sudo apt-get update && sudo apt-get install -y -- {safe_package}", actor=actor)
        return (res.exit_code == 0, res.stdout or res.stderr)

    async def uninstall_application(self, workspace_id: str, package_name: str, actor: str = "operator") -> bool:
        """Uninstalls a package inside the sandbox."""
        res = await self.terminal(workspace_id, f"apt-get remove -y {package_name}")
        return res.exit_code == 0

    async def service_action(self, workspace_id: str, service_name: str, action: str, actor: str = "operator") -> ServiceInfo:
        """Controls system services inside the sandbox."""
        res = await self.terminal(workspace_id, f"service {shlex.quote(service_name)} {shlex.quote(action)}")
        status_value = "RUNNING" if res.exit_code == 0 and action in {"start", "restart"} else action.upper()
        return ServiceInfo(
            name=service_name,
            status=status_value,
            port=0,
            logs=(res.stdout + res.stderr).splitlines(),
        )

    async def git_action(self, workspace_id: str, action: str, **kwargs: Any) -> Any:
        """Executes Git operations against the sandbox workspace."""
        if action == "status":
            res = await self.terminal(workspace_id, "git status --porcelain")
            branch_res = await self.terminal(workspace_id, "git branch --show-current")
            branch = branch_res.stdout.strip() if (branch_res.exit_code == 0 and branch_res.stdout.strip()) else ""
            return GitStatusInfo(
                branch=branch,
                is_clean=(len(res.stdout.strip()) == 0),
                untracked_files=[line[3:] for line in res.stdout.splitlines() if line.startswith("??")],
                modified_files=[line[3:] for line in res.stdout.splitlines() if not line.startswith("??")],
                staged_files=[],
            )
        elif action == "commit":
            msg = kwargs.get("message", "chore: automated commit")
            safe_msg = msg.replace("'", "'\\''")
            res = await self.terminal(workspace_id, f"git add -A && git commit -m '{safe_msg}'")
            return res.exit_code == 0
        elif action in ["branch", "checkout"]:
            raw_branch = str(kwargs.get("branch_name") or kwargs.get("branch") or "main").strip()
            safe_branch = shlex.quote(raw_branch)
            res = await self.terminal(workspace_id, f"git checkout -B {safe_branch}")
            return res.exit_code == 0
        elif action == "diff":
            branch_res = await self.terminal(workspace_id, "git branch --show-current")
            branch = branch_res.stdout.strip() if (branch_res.exit_code == 0 and branch_res.stdout.strip()) else "HEAD"
            safe_diff_branch = shlex.quote(branch)
            res = await self.terminal(workspace_id, f"git diff {safe_diff_branch}")
            return res.stdout or "Working tree clean."
        return ""

    def _record_audit(
        self,
        session_id: str,
        workspace_id: str,
        tenant_id: str,
        actor: str,
        action: str,
        resource: str,
        result: str,
        details: dict[str, Any] | None = None,
        risk_level: ComputerRiskLevel = ComputerRiskLevel.LOW,
    ) -> None:
        event = ComputerAuditEvent(
            session_id=session_id,
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            resource=resource,
            result=result,
            risk_level=risk_level,
            details=details or {},
        )
        self.audit_log.append(event)
