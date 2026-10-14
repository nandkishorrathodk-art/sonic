"""Windows host desktop provider using local AVCS pixels and guarded input."""

from __future__ import annotations

import asyncio
import hashlib
import math
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer.models import (
    ComputerProfile,
    ComputerState,
    ComputerWorkspace,
    ComputerWorkspaceStatus,
    ComputerWorkspaceType,
    GUIAction,
    GUIActionType,
    ScreenObservation,
)


class WindowsHostAVCSComputerProvider(DockerComputerProvider):
    """Use the Windows desktop for GUI while keeping execution in Docker.

    This provider is opt-in. Terminal, filesystem, and git operations continue
    to use the isolated Docker execution plane; only AVCS screen capture and
    the explicitly supported pointer actions address the physical desktop.
    """

    host_workspace_id = "windows-host-desktop"
    reattach_persisted_workspace = True

    def __init__(
        self,
        *,
        capture_factory: Callable[[], Any] | None = None,
        perception: Any | None = None,
        cursor: Any | None = None,
    ) -> None:
        super().__init__()
        if sys.platform != "win32" and (
            capture_factory is None or perception is None or cursor is None
        ):
            raise RuntimeError("The Windows host desktop provider requires Windows.")
        self._capture_factory = capture_factory
        self._perception = perception
        self._cursor = cursor
        self._host_workspace: ComputerWorkspace | None = None
        self._host_owner: str | None = None
        self._capture_lock = asyncio.Lock()
        self._gui_lock = asyncio.Lock()
        self._frame_id = 0
        self._last_observation: ScreenObservation | None = None

    def _load_host_adapters(self) -> None:
        if self._capture_factory is None:
            from avcs.capture_adapter import Win32GDICaptureAdapter

            self._capture_factory = Win32GDICaptureAdapter
        if self._perception is None:
            from sonic.computer_use.avcs_perception import AVCSPerception

            self._perception = AVCSPerception()
        if self._cursor is None:
            from forge.vision.cortex.motor.win32 import Win32CursorDriver

            self._cursor = Win32CursorDriver(failsafe=True)

    async def create(
        self,
        tenant_id: str,
        engagement_id: str,
        workspace_type: ComputerWorkspaceType = ComputerWorkspaceType.MISSION_COMPUTER,
        profile: ComputerProfile = ComputerProfile.DEBIAN_ENGINEERING,
    ) -> ComputerWorkspace:
        if workspace_type != ComputerWorkspaceType.MISSION_COMPUTER:
            return await super().create(
                tenant_id=tenant_id,
                engagement_id=engagement_id,
                workspace_type=workspace_type,
                profile=profile,
            )
        if sys.platform != "win32" and self._capture_factory is None:
            raise RuntimeError("Windows host desktop mode is only available on Windows.")
        if self._host_owner not in (None, tenant_id):
            raise RuntimeError(
                "The shared Windows host desktop is already owned by another tenant."
            )
        self._load_host_adapters()
        self._host_owner = tenant_id
        workspace = ComputerWorkspace(
            id=self.host_workspace_id,
            tenant_id=tenant_id,
            engagement_id=engagement_id,
            workspace_type=workspace_type,
            profile=profile,
            provider_type=type(self).__name__,
            image="windows-host",
            status=ComputerWorkspaceStatus.RUNNING,
            workspace_path="/home/sonic/workspace",
            capabilities=["desktop", "avcs", "isolated_terminal", "filesystem", "git"],
        )
        self._host_workspace = workspace
        return workspace

    async def destroy(self, workspace_id: str) -> bool:
        if workspace_id == self.host_workspace_id:
            self._host_workspace = None
            self._host_owner = None
            self._last_observation = None
            return True
        return await super().destroy(workspace_id)

    def _require_host_workspace(self, workspace_id: str) -> ComputerWorkspace:
        workspace = self._host_workspace
        if (
            workspace_id != self.host_workspace_id
            or workspace is None
            or self._host_owner != workspace.tenant_id
        ):
            raise RuntimeError("Windows host desktop is not provisioned for this session.")
        return workspace

    @staticmethod
    def _active_window_title() -> str:
        if sys.platform != "win32":
            return "Windows host desktop"
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "Windows host desktop"
        title = ctypes.create_unicode_buffer(512)
        length = user32.GetWindowTextW(hwnd, title, len(title))
        return title.value[:length] if length > 0 else "Windows host desktop"

    @staticmethod
    def _active_window_geometry() -> tuple[int, int, int, int] | None:
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return None
        rect = wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return None
        return rect.left, rect.top, width, height

    async def status(self, workspace_id: str) -> ComputerState:
        if workspace_id != self.host_workspace_id:
            return await super().status(workspace_id)
        workspace = self._require_host_workspace(workspace_id)
        title = await asyncio.to_thread(self._active_window_title)
        return ComputerState(
            workspace_id=workspace.id,
            tenant_id=workspace.tenant_id,
            status=ComputerWorkspaceStatus.RUNNING,
            active_application="Windows host desktop",
            open_applications=["Windows host desktop"],
            active_window=title,
            working_directory=workspace.workspace_path,
            running_processes=[],
            installed_applications=[],
        )

    async def get_vnc_url(self, workspace_id: str) -> str | None:
        if workspace_id == self.host_workspace_id:
            self._require_host_workspace(workspace_id)
            return None
        return await super().get_vnc_url(workspace_id)

    def _capture_and_perceive(self) -> ScreenObservation:
        assert self._capture_factory is not None
        assert self._perception is not None
        with self._capture_factory() as adapter:
            frame = adapter.grab()
            capture_timestamp_ns = time.time_ns()
            geometry = adapter.get_geometry()
        height, width, channels = frame.shape
        if channels != 3 or (width, height) != (
            geometry.virtual_width,
            geometry.virtual_height,
        ):
            raise RuntimeError("AVCS host frame dimensions do not match Windows display geometry.")
        self._frame_id += 1
        avcs_state = self._perception.perceive_rgb(
            frame,
            source_frame_id=self._frame_id,
        )
        raw_rgb = frame.tobytes()
        return ScreenObservation(
            raw_rgb=raw_rgb,
            frame_id=self._frame_id,
            avcs_state=avcs_state,
            perception_version=avcs_state.get("frame_index", 0),
            capture_timestamp_ns=capture_timestamp_ns,
            capture_age_ms=max(0.0, (time.time_ns() - capture_timestamp_ns) / 1_000_000),
            width=width,
            height=height,
            active_window=self._active_window_title(),
            active_window_geometry=self._active_window_geometry(),
            desktop_state="INTERACTIVE",
            timestamp=datetime.now(UTC).isoformat(),
        )

    async def screenshot(self, workspace_id: str) -> ScreenObservation:
        self._require_host_workspace(workspace_id)
        async with self._capture_lock:
            self._load_host_adapters()
            observation = await asyncio.to_thread(self._capture_and_perceive)
            self._last_observation = observation
            return observation

    @staticmethod
    def _validate_pointer_points(
        observation: ScreenObservation,
        points: tuple[tuple[int | None, int | None], ...],
    ) -> None:
        if (
            observation.desktop_state != "INTERACTIVE"
            or observation.capture_age_ms is None
            or observation.capture_age_ms > 2_000
            or not observation.raw_rgb
            or len(observation.raw_rgb) != observation.width * observation.height * 3
        ):
            raise RuntimeError("Host pointer action blocked: fresh AVCS pixels are unavailable.")
        if (
            not isinstance(observation.avcs_state, dict)
            or observation.avcs_state.get("engine") != "avcs_reflex"
        ):
            raise RuntimeError(
                "Host pointer action blocked: AVCS did not process the current frame."
            )
        for x, y in points:
            if (
                isinstance(x, bool)
                or not isinstance(x, int)
                or isinstance(y, bool)
                or not isinstance(y, int)
                or x < 0
                or y < 0
                or x >= observation.width
                or y >= observation.height
            ):
                raise ValueError("Host pointer coordinates must be in-bounds screen pixels.")

    async def _validate_grounded_target(
        self, observation: ScreenObservation, action: GUIAction
    ) -> None:
        query = action.target_query
        if not isinstance(query, str) or not query.strip():
            raise RuntimeError(
                "Host pointer action blocked: semantic target query and a qualified "
                "AVCS grounder are required; raw coordinates are not grounding evidence."
            )
        if len(query.encode("utf-8")) > 96:
            raise RuntimeError("Host pointer action blocked: AVCS target query is too long.")
        ground = getattr(self._perception, "ground_target_rgb_bytes", None)
        if not callable(ground):
            raise RuntimeError("Host pointer action blocked: local AVCS target grounding is unavailable.")
        result = await asyncio.to_thread(
            ground,
            observation.raw_rgb,
            query.strip(),
            observation.width,
            observation.height,
            observation.active_window_geometry,
            observation.frame_id,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                "Host pointer action blocked: AVCS did not resolve this target on the current frame."
            )
        state = observation.avcs_state
        metadata = result.get("checkpoint_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "Host pointer action blocked: AVCS checkpoint evaluation metadata is unavailable."
            )
        evaluation = metadata.get("evaluation") if isinstance(metadata, dict) else None
        calibration = (
            metadata.get("query_confidence_calibration")
            if isinstance(metadata, dict)
            else None
        )
        calibration_evaluation = (
            calibration.get("evaluation") if isinstance(calibration, dict) else None
        )
        accuracy = evaluation.get("value") if isinstance(evaluation, dict) else None
        precision = (
            calibration_evaluation.get("precision")
            if isinstance(calibration_evaluation, dict)
            else None
        )
        required_precision = (
            calibration_evaluation.get("required_precision")
            if isinstance(calibration_evaluation, dict)
            else None
        )
        threshold = (
            calibration_evaluation.get("threshold")
            if isinstance(calibration_evaluation, dict)
            else None
        )
        calibration_samples = (
            calibration_evaluation.get("samples")
            if isinstance(calibration_evaluation, dict)
            else None
        )
        accepted_samples = (
            calibration_evaluation.get("accepted_samples")
            if isinstance(calibration_evaluation, dict)
            else None
        )
        score = result.get("query_score")
        if (
            result.get("engine") != "avcs_goal_grounder"
            or result.get("source_frame_id") != observation.frame_id
            or result.get("scene_hash") != state.get("scene_hash")
            or result.get("frame_index") != state.get("frame_index")
            or metadata.get("schema_version") != 2
            or metadata.get("model_type") != "avcs_goal_conditioned_grounder"
            or metadata.get("dataset_provenance")
            != "recorded_ui_click_demonstrations"
            or metadata.get("coordinate_frame")
            not in ("cropped_target_window", "full_screen")
            or any(
                isinstance(metadata.get(name), bool)
                or not isinstance(metadata.get(name), int)
                or metadata[name] <= 0
                for name in (
                    "training_samples",
                    "validation_samples",
                    "held_out_test_samples",
                )
            )
            or not isinstance(evaluation, dict)
            or evaluation.get("metric")
            != "top1_point_accuracy_within_0.02_diagonal"
            or evaluation.get("passed") is not True
            or evaluation.get("samples") != metadata.get("held_out_test_samples")
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(float(accuracy))
            or float(accuracy) < 0.85
            or not isinstance(calibration_evaluation, dict)
            or calibration_evaluation.get("metric")
            != "empirical_precision_at_threshold"
            or calibration_evaluation.get("passed") is not True
            or isinstance(precision, bool)
            or not isinstance(precision, (int, float))
            or not math.isfinite(float(precision))
            or isinstance(required_precision, bool)
            or not isinstance(required_precision, (int, float))
            or not math.isfinite(float(required_precision))
            or float(required_precision) < 0.8
            or float(precision) < float(required_precision)
            or isinstance(calibration_samples, bool)
            or not isinstance(calibration_samples, int)
            or calibration_samples != metadata.get("held_out_test_samples")
            or isinstance(accepted_samples, bool)
            or not isinstance(accepted_samples, int)
            or accepted_samples < 10
            or accepted_samples > calibration_samples
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(score) < float(threshold)
        ):
            raise RuntimeError(
                "Host pointer action blocked: AVCS grounding lacks valid, "
                "current-frame evaluation or confidence-calibration evidence."
            )

        x, y = result.get("x"), result.get("y")
        if (
            isinstance(x, bool)
            or not isinstance(x, int)
            or isinstance(y, bool)
            or not isinstance(y, int)
            or action.x != x
            or action.y != y
        ):
            raise RuntimeError(
                "Host pointer action blocked: requested coordinates do not match "
                "the qualified AVCS target location."
            )

    async def _move_smoothly(self, x: int, y: int) -> None:
        assert self._cursor is not None
        assert self._last_observation is not None
        rect = self._cursor.virtual_screen_rect
        screen_x, screen_y, screen_width, screen_height = rect
        observation = self._last_observation
        if (
            observation is None
            or screen_x != 0
            or screen_y != 0
            or (screen_width, screen_height)
            != (
                observation.width,
                observation.height,
            )
            or self._cursor.screen_size != (observation.width, observation.height)
        ):
            raise RuntimeError("Host pointer control requires a single primary display at origin.")
        start_x, start_y = self._cursor.get_position_pixels()
        steps = 12
        duration = 0.24
        for step in range(1, steps + 1):
            progress = (1 - math.cos(math.pi * step / steps)) / 2
            next_x = round(start_x + (x - start_x) * progress)
            next_y = round(start_y + (y - start_y) * progress)
            await asyncio.to_thread(self._cursor.move_to_pixels, next_x, next_y)
            if step < steps:
                await asyncio.sleep(duration / steps)

    async def _dispatch_pointer_action(self, action: GUIAction) -> None:
        assert self._cursor is not None
        if action.action == GUIActionType.MOVE:
            assert action.x is not None and action.y is not None
            await self._move_smoothly(action.x, action.y)
            return
        if action.action in (
            GUIActionType.CLICK,
            GUIActionType.DOUBLE_CLICK,
            GUIActionType.RIGHT_CLICK,
        ):
            button = "right" if action.action == GUIActionType.RIGHT_CLICK else "left"
            await asyncio.to_thread(self._cursor.click, button=button)
            if action.action == GUIActionType.DOUBLE_CLICK:
                await asyncio.sleep(0.08)
                await asyncio.to_thread(self._cursor.click, button=button)
            return
        if action.action == GUIActionType.DRAG:
            assert action.x2 is not None and action.y2 is not None
            await asyncio.to_thread(self._cursor.mouse_down)
            try:
                await self._move_smoothly(action.x2, action.y2)
            finally:
                await asyncio.to_thread(self._cursor.mouse_up)
            return
        raise RuntimeError(f"Host desktop action {action.action.value} is not supported.")

    async def gui_action(
        self,
        workspace_id: str,
        action: GUIAction,
        actor: str = "operator",
    ) -> ScreenObservation:
        self._require_host_workspace(workspace_id)
        if action.action not in {
            GUIActionType.CLICK,
            GUIActionType.DOUBLE_CLICK,
            GUIActionType.RIGHT_CLICK,
            GUIActionType.MOVE,
        }:
            raise RuntimeError(
                f"Windows host action {action.action.value} is blocked: "
                "keyboard focus verification and this action's safety contract are unavailable."
            )
        async with self._gui_lock:
            previous = self._last_observation
            if previous is None:
                raise RuntimeError(
                    "Host pointer action blocked: observe the desktop before acting."
                )
            current = await self.screenshot(workspace_id)
            if (
                hashlib.sha256(current.raw_rgb).digest()
                != hashlib.sha256(previous.raw_rgb).digest()
            ):
                raise RuntimeError(
                    "Host pointer action blocked: desktop pixels changed; re-observe before acting."
                )
            points = (
                ((action.x, action.y), (action.x2, action.y2))
                if action.action == GUIActionType.DRAG
                else ((action.x, action.y),)
            )
            self._validate_pointer_points(current, points)
            await self._validate_grounded_target(current, action)
            assert action.x is not None and action.y is not None
            if action.action != GUIActionType.MOVE:
                await self._move_smoothly(action.x, action.y)
            await self._dispatch_pointer_action(action)
            return await self.screenshot(workspace_id)

    async def application_list(self, workspace_id: str) -> list[str]:
        self._require_host_workspace(workspace_id)
        return []

    async def launch_application(
        self, workspace_id: str, app_name: str, actor: str = "operator"
    ) -> bool:
        self._require_host_workspace(workspace_id)
        raise RuntimeError(
            "Launching host applications is not enabled in Windows host desktop mode."
        )

    async def close_application(
        self, workspace_id: str, app_name: str, actor: str = "operator"
    ) -> bool:
        self._require_host_workspace(workspace_id)
        raise RuntimeError("Closing host applications is not enabled in Windows host desktop mode.")

    async def install_application(
        self, workspace_id: str, package_name: str, actor: str = "operator"
    ) -> tuple[bool, str]:
        self._require_host_workspace(workspace_id)
        raise RuntimeError("Installing software on the Windows host is not enabled.")

    async def uninstall_application(
        self, workspace_id: str, package_name: str, actor: str = "operator"
    ) -> bool:
        self._require_host_workspace(workspace_id)
        raise RuntimeError("Uninstalling software from the Windows host is not enabled.")
