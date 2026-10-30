"""
SONIC-REDA — Autonomous Computer & Engineering Workspace Models (Phase 13)
=============================================================================
Data models for persistent computer workspaces, sessions, GUI actions,
screen observations, application policies, processes, files, git status,
services, and audit events.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str = "comp") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ============================================
# Enums
# ============================================

class ComputerWorkspaceStatus(StrEnum):
    CREATING = "CREATING"
    STARTING = "STARTING"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    DEGRADED = "DEGRADED"
    DESTROYING = "DESTROYING"
    DESTROYED = "DESTROYED"
    FAILED = "FAILED"


class ComputerWorkspaceType(StrEnum):
    MISSION_COMPUTER = "MISSION_COMPUTER"  # Persistent long-term environment
    RESEARCH_LAB = "RESEARCH_LAB"          # Disposable single-task sandbox
    TARGET_SANDBOX = "TARGET_SANDBOX"      # Disposable authorized target environment


class ComputerProfile(StrEnum):
    DEBIAN_ENGINEERING = "DEBIAN_ENGINEERING"
    KALI_SECURITY = "KALI_SECURITY"
    GENERIC_LINUX = "GENERIC_LINUX"


class ComputerSessionMode(StrEnum):
    AUTONOMOUS = "AUTONOMOUS"
    ASSISTED = "ASSISTED"
    MANUAL = "MANUAL"


class GUIActionType(StrEnum):
    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    TYPE = "TYPE"
    KEYPRESS = "KEYPRESS"
    HOTKEY = "HOTKEY"
    SCROLL = "SCROLL"
    MOVE = "MOVE"
    DRAG = "DRAG"
    SELECT_WINDOW = "SELECT_WINDOW"
    APP_MAXIMIZE = "APP_MAXIMIZE"
    WINDOW_MAXIMIZE = "WINDOW_MAXIMIZE"
    WINDOW_TILE = "WINDOW_TILE"
    OPEN_APP = "OPEN_APP"
    LAUNCH_APP = "LAUNCH_APP"
    CLOSE_APP = "CLOSE_APP"
    CLOSE_WINDOW = "CLOSE_WINDOW"
    INSTALL_APP = "INSTALL_APP"
    WAIT = "WAIT"


class ComputerRiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================
# Models
# ============================================

class ComputerWorkspace(BaseModel):
    """A persistent or disposable engineering computer sandbox workspace."""
    id: str = Field(default_factory=lambda: _new_id("ws"))
    tenant_id: str
    engagement_id: str
    workspace_type: ComputerWorkspaceType = ComputerWorkspaceType.MISSION_COMPUTER
    profile: ComputerProfile = ComputerProfile.DEBIAN_ENGINEERING
    provider_type: str = "DockerProvider"  # DockerProvider, DaytonaProvider, LocalDevProvider
    image: str = "sonic-workstation:latest"
    status: ComputerWorkspaceStatus = ComputerWorkspaceStatus.READY
    workspace_path: str = "/home/sonic/workspace"
    capabilities: list[str] = Field(default_factory=lambda: [
        "desktop", "terminal", "filesystem", "ide", "browser", "git", "services", "package_manager"
    ])
    resource_profile: dict[str, Any] = Field(default_factory=lambda: {
        "cpu_cores": 4, "memory_gb": 8, "disk_gb": 32
    })
    created_at: str = Field(default_factory=_now)
    last_active_at: str = Field(default_factory=_now)

    @property
    def workspace_id(self) -> str:
        return self.id


class ComputerSession(BaseModel):
    """An active user or agent interaction session attached to a ComputerWorkspace."""
    id: str = Field(default_factory=lambda: _new_id("sess"))
    workspace_id: str
    user_id: str
    tenant_id: str
    mode: ComputerSessionMode = ComputerSessionMode.AUTONOMOUS
    permissions: list[str] = Field(default_factory=lambda: ["read", "write", "execute", "gui"])
    started_at: str = Field(default_factory=_now)
    last_activity: str = Field(default_factory=_now)


class GUIAction(BaseModel):
    """A structured graphical desktop action."""
    action: GUIActionType
    x: int | None = None
    y: int | None = None
    target_query: str | None = None
    # Destination coordinates for DRAG (source is x,y). Lets the agent move a
    # file onto a folder, slide a wizard control, or rearrange windows — like a
    # human press-move-release gesture.
    x2: int | None = None
    y2: int | None = None
    button: int = 1
    text: str | None = None
    key: str | None = None
    window_id: str | None = None
    app_name: str | None = None
    scroll_delta: int = 0
    # X11 delivers vertical wheel events on buttons 4/5 and horizontal ones on
    # 6/7. Without an axis, the agent can only scroll up/down and cannot reach
    # content off to the side (wide pages, spreadsheets, code, timelines).
    scroll_axis: str = "vertical"
    delay_ms: int = 25

    @model_validator(mode="after")
    def validate_scroll_delta(self) -> "GUIAction":
        if self.action == GUIActionType.SCROLL and self.scroll_delta == 0:
            raise ValueError("GUIActionType.SCROLL requires a non-zero scroll_delta")
        normalized = str(self.scroll_axis).strip().lower()
        if normalized in ("horizontal", "x", "left", "right"):
            self.scroll_axis = "horizontal"
        elif normalized in ("vertical", "y", "up", "down", ""):
            self.scroll_axis = "vertical"
        else:
            raise ValueError("scroll_axis must be 'vertical' or 'horizontal'")
        return self


class ScreenObservation(BaseModel):
    """Visual and state perception of the computer desktop screen."""
    screenshot_base64: str = ""
    # ``repr=False`` is load-bearing, not cosmetic: a full 1920x1080 frame is
    # ~6 MB, and any accidental f-string/logger interpolation of this object
    # would dump the raw bytes and flood logs (observed: 81 MB from one run).
    raw_rgb: bytes = Field(default=b"", exclude=True, repr=False)
    frame_id: int | None = Field(default=None, ge=1)
    avcs_state: dict[str, Any] = Field(default_factory=dict)
    capture_timestamp_ns: int | None = Field(default=None, ge=0)
    capture_age_ms: float | None = Field(default=None, ge=0)
    controls_age_ms: float | None = Field(default=None, ge=0)
    perception_version: int = Field(default=0, ge=0)
    changed_region: tuple[int, int, int, int] | None = None
    width: int = 1920
    height: int = 1080
    active_window: str = "Desktop"
    active_window_geometry: tuple[int, int, int, int] | None = None
    visible_text: str = ""
    detected_controls: list[Any] = Field(default_factory=list)
    desktop_state: str = "INTERACTIVE"
    timestamp: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def validate_raw_frame(self) -> "ScreenObservation":
        if not self.raw_rgb:
            return self
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Raw RGB frames require positive screen dimensions")
        if len(self.raw_rgb) != self.width * self.height * 3:
            raise ValueError("Raw RGB frame byte length does not match screen dimensions")
        if self.frame_id is None:
            raise ValueError("Raw RGB frames require a frame_id")
        return self

    def __init__(self, **data: Any):
        if "active_window_geom" in data and "active_window_geometry" not in data:
            data["active_window_geometry"] = data.pop("active_window_geom")
        if "active_window_geometry" in data and data["active_window_geometry"] is not None:
            raw_geom = data["active_window_geometry"]
            if hasattr(raw_geom, "bbox"):
                b = raw_geom.bbox
                data["active_window_geometry"] = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            elif hasattr(raw_geom, "x") and hasattr(raw_geom, "y") and hasattr(raw_geom, "width") and hasattr(raw_geom, "height"):
                data["active_window_geometry"] = (int(raw_geom.x), int(raw_geom.y), int(raw_geom.width), int(raw_geom.height))
            elif isinstance(raw_geom, dict):
                x = raw_geom.get("x", 0)
                y = raw_geom.get("y", 0)
                w = raw_geom.get("w", raw_geom.get("width", 0))
                h = raw_geom.get("h", raw_geom.get("height", 0))
                data["active_window_geometry"] = (int(x), int(y), int(w), int(h))
            elif isinstance(raw_geom, (list, tuple)) and len(raw_geom) >= 4:
                data["active_window_geometry"] = (int(raw_geom[0]), int(raw_geom[1]), int(raw_geom[2]), int(raw_geom[3]))
        super().__init__(**data)

    @property
    def active_window_geom(self) -> tuple[int, int, int, int] | None:
        return self.active_window_geometry



class ComputerState(BaseModel):
    """Complete persistent state representation of the SONIC Computer."""
    workspace_id: str
    tenant_id: str
    status: ComputerWorkspaceStatus = ComputerWorkspaceStatus.READY
    active_application: str = "Desktop"
    open_applications: list[str] = Field(default_factory=lambda: ["Desktop", "Terminal"])
    active_window: str = "Terminal"
    working_directory: str = "/home/sonic/workspace"
    running_processes: list[str] = Field(default_factory=list)
    installed_applications: list[str] = Field(default_factory=list)
    current_project: str = "sonic-repo"
    git_branch: str = "main"
    resource_usage: dict[str, float] = Field(default_factory=lambda: {"cpu_pct": 12.5, "memory_mb": 1024.0})
    last_observation: ScreenObservation | None = None


class ApplicationPolicy(BaseModel):
    """Security rules governing application and package installations.

    The default policy deliberately has no preferred application or tool list.
    A target-driven agent may use an already-installed capability, author a
    focused probe, or request an explicitly approved package when evidence
    shows it is useful.
    """
    allowed_packages: list[str] = Field(default_factory=list)
    forbidden_packages: list[str] = Field(default_factory=lambda: [
        "privileged-packet-sniffer-root", "tor-relay", "cryptominer", "kernel-mod", "ddos-bot"
    ])
    max_install_size_mb: int = 2048
    require_approval: bool = False

    def is_package_allowed(self, package_name: str) -> tuple[bool, str]:
        pkg = package_name.lower().strip()
        if any(f in pkg for f in self.forbidden_packages):
            return False, f"Package '{pkg}' is strictly prohibited by security policy."
        if pkg in self.allowed_packages:
            return True, f"Package '{pkg}' is explicitly allowed by this workspace policy."
        return True, f"Package '{pkg}' permitted under developer engineering policy."


class ComputerAuditEvent(BaseModel):
    """Immutable audit record of a computer operation."""
    id: str = Field(default_factory=lambda: _new_id("audit"))
    session_id: str
    workspace_id: str
    tenant_id: str
    actor: str
    action: str  # "LAUNCH_APP", "WRITE_FILE", "EXECUTE_COMMAND", "GIT_COMMIT", etc.
    application: str | None = None
    resource: str
    result: str = "SUCCESS"
    risk_level: ComputerRiskLevel = ComputerRiskLevel.LOW
    timestamp: str = Field(default_factory=_now)


class ProcessInfo(BaseModel):
    """Running process metadata."""
    pid: int
    name: str
    cpu_pct: float = 0.0
    memory_mb: float = 0.0
    status: str = "RUNNING"


class FileEntry(BaseModel):
    """Filesystem directory entry."""
    name: str
    path: str
    is_dir: bool = False
    size_bytes: int = 0
    modified_at: str = Field(default_factory=_now)


class GitStatusInfo(BaseModel):
    """Git repository status inside the computer workspace."""
    branch: str = "main"
    is_clean: bool = True
    modified_files: list[str] = Field(default_factory=list)
    untracked_files: list[str] = Field(default_factory=list)
    ahead: int = 0
    behind: int = 0


class ServiceInfo(BaseModel):
    """Managed service running inside the computer."""
    name: str
    status: str = "RUNNING"
    port: int | None = None
    logs: list[str] = Field(default_factory=list)
