from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from sonic.api.routes.workstation import (
    _fresh_avcs_pixel_evidence,
    get_computer,
    get_desktop_screenshot,
)
from sonic.auth.models import User, UserRole
from sonic.computer.models import (
    ComputerProfile,
    ComputerWorkspaceType,
    GUIAction,
    GUIActionType,
)
from sonic.computer.windows_host import WindowsHostAVCSComputerProvider


@dataclass(frozen=True)
class _Geometry:
    virtual_left: int = 0
    virtual_top: int = 0
    virtual_width: int = 5
    virtual_height: int = 4


class _Capture(AbstractContextManager):
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def grab(self):
        return self.frame.copy()

    def get_geometry(self):
        return _Geometry()


class _Cursor:
    virtual_screen_rect = (0, 0, 5, 4)
    screen_size = (5, 4)

    def __init__(self):
        self.position = (0, 0)
        self.actions: list[tuple] = []

    def get_position_pixels(self):
        return self.position

    def move_to_pixels(self, x: int, y: int):
        self.position = (x, y)
        self.actions.append(("move", x, y))

    def click(self, button="left"):
        self.actions.append(("click", button))

    def mouse_down(self):
        self.actions.append(("down",))

    def mouse_up(self):
        self.actions.append(("up",))


class _Perception:
    def __init__(self):
        self.source_ids: list[int] = []
        self.states: dict[int, dict] = {}

    def perceive_rgb(self, frame: np.ndarray, *, source_frame_id: int):
        self.source_ids.append(source_frame_id)
        state = {
            "engine": "avcs_reflex",
            "frame_index": source_frame_id,
            "source_frame_id": source_frame_id,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "scene_hash": f"test-scene-{source_frame_id}",
        }
        self.states[source_frame_id] = state
        return state


class _QualifiedPerception(_Perception):
    def __init__(self, *, accuracy=0.86, source_offset=0):
        super().__init__()
        self.accuracy = accuracy
        self.source_offset = source_offset

    def ground_target_rgb_bytes(
        self, _frame, _query, _width, _height, _window_geometry, source_frame_id
    ):
        state = self.states[source_frame_id]
        return {
            "engine": "avcs_goal_grounder",
            "x": 3,
            "y": 2,
            "source_frame_id": source_frame_id + self.source_offset,
            "frame_index": state["frame_index"],
            "scene_hash": state["scene_hash"],
            "query_score": 0.95,
            "checkpoint_metadata": {
                "schema_version": 2,
                "model_type": "avcs_goal_conditioned_grounder",
                "dataset_provenance": "recorded_ui_click_demonstrations",
                "coordinate_frame": "full_screen",
                "training_samples": 30,
                "validation_samples": 10,
                "held_out_test_samples": 20,
                "evaluation": {
                    "metric": "top1_point_accuracy_within_0.02_diagonal",
                    "passed": True,
                    "value": self.accuracy,
                    "samples": 20,
                },
                "query_confidence_calibration": {
                    "evaluation": {
                        "metric": "empirical_precision_at_threshold",
                        "passed": True,
                        "precision": 0.9,
                        "required_precision": 0.9,
                        "threshold": 0.9,
                        "samples": 20,
                        "accepted_samples": 10,
                    }
                },
            },
        }


def _provider(frame_factory=None):
    frame = np.ones((4, 5, 3), dtype=np.uint8)
    cursor = _Cursor()
    perception = _Perception()
    captures = iter(frame_factory or [frame] * 10)
    provider = WindowsHostAVCSComputerProvider(
        capture_factory=lambda: _Capture(next(captures)),
        perception=perception,
        cursor=cursor,
    )
    return provider, cursor, perception


@pytest.mark.asyncio
async def test_host_provider_captures_and_processes_pixels_through_avcs():
    provider, cursor, perception = _provider()
    workspace = await provider.create("tenant-a", "session-a")

    observation = await provider.screenshot(workspace.id)

    assert workspace.provider_type == "WindowsHostAVCSComputerProvider"
    assert observation.raw_rgb == bytes([1]) * (5 * 4 * 3)
    assert observation.avcs_state["engine"] == "avcs_reflex"
    assert observation.width == 5 and observation.height == 4
    assert observation.capture_age_ms is not None
    assert _fresh_avcs_pixel_evidence(observation) is not None
    assert perception.source_ids == [1]
    assert cursor.actions == []


@pytest.mark.asyncio
async def test_host_provider_revalidates_identical_fresh_pixels_before_pointer_input():
    provider, cursor, perception = _provider()
    workspace = await provider.create("tenant-a", "session-a")
    await provider.screenshot(workspace.id)

    with pytest.raises(RuntimeError, match="semantic target query"):
        await provider.gui_action(
            workspace.id,
            GUIAction(action=GUIActionType.CLICK, x=3, y=2),
            actor="operator",
        )

    assert cursor.actions == []
    assert perception.source_ids == [1, 2]


@pytest.mark.asyncio
async def test_host_provider_blocks_semantic_click_when_local_grounder_is_unavailable():
    provider, cursor, _ = _provider()
    workspace = await provider.create("tenant-a", "session-a")
    await provider.screenshot(workspace.id)

    with pytest.raises(RuntimeError, match="target grounding is unavailable"):
        await provider.gui_action(
            workspace.id,
            GUIAction(
                action=GUIActionType.CLICK,
                x=3,
                y=2,
                target_query="the confirm control",
            ),
        )

    assert cursor.actions == []


@pytest.mark.asyncio
async def test_host_provider_dispatches_only_current_qualified_avcs_grounding():
    frame = np.ones((4, 5, 3), dtype=np.uint8)
    cursor = _Cursor()
    perception = _QualifiedPerception()
    provider = WindowsHostAVCSComputerProvider(
        capture_factory=lambda: _Capture(frame),
        perception=perception,
        cursor=cursor,
    )
    workspace = await provider.create("tenant-a", "session-a")
    await provider.screenshot(workspace.id)

    result = await provider.gui_action(
        workspace.id,
        GUIAction(
            action=GUIActionType.CLICK,
            x=3,
            y=2,
            target_query="the confirm control",
        ),
        actor="operator",
    )

    assert result.frame_id == 3
    assert cursor.position == (3, 2)
    assert ("click", "left") in cursor.actions
    assert perception.source_ids == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accuracy", "source_offset", "message"),
    [
        (0.79, 0, "current-frame evaluation"),
        (0.86, -1, "current-frame evaluation"),
    ],
)
async def test_host_provider_blocks_unqualified_or_stale_grounding(
    accuracy, source_offset, message
):
    frame = np.ones((4, 5, 3), dtype=np.uint8)
    cursor = _Cursor()
    perception = _QualifiedPerception(
        accuracy=accuracy,
        source_offset=source_offset,
    )
    provider = WindowsHostAVCSComputerProvider(
        capture_factory=lambda: _Capture(frame),
        perception=perception,
        cursor=cursor,
    )
    workspace = await provider.create("tenant-a", "session-a")
    await provider.screenshot(workspace.id)

    with pytest.raises(RuntimeError, match=message):
        await provider.gui_action(
            workspace.id,
            GUIAction(
                action=GUIActionType.CLICK,
                x=3,
                y=2,
                target_query="the confirm control",
            ),
        )

    assert cursor.actions == []


@pytest.mark.asyncio
async def test_host_provider_blocks_when_pixels_change_before_dispatch():
    frames = [
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.ones((4, 5, 3), dtype=np.uint8),
    ]
    provider, cursor, _ = _provider(frames)
    workspace = await provider.create("tenant-a", "session-a")
    await provider.screenshot(workspace.id)

    with pytest.raises(RuntimeError, match="pixels changed"):
        await provider.gui_action(
            workspace.id,
            GUIAction(action=GUIActionType.CLICK, x=3, y=2),
        )

    assert cursor.actions == []


@pytest.mark.asyncio
async def test_host_provider_blocks_out_of_bounds_and_unverified_keyboard_actions():
    provider, cursor, _ = _provider()
    workspace = await provider.create("tenant-a", "session-a")
    await provider.screenshot(workspace.id)

    with pytest.raises(ValueError, match="in-bounds"):
        await provider.gui_action(
            workspace.id,
            GUIAction(action=GUIActionType.CLICK, x=5, y=2),
        )
    with pytest.raises(RuntimeError, match="focus verification"):
        await provider.gui_action(
            workspace.id,
            GUIAction(action=GUIActionType.TYPE, x=2, y=2, text="not dispatched"),
        )

    assert cursor.actions == []


@pytest.mark.asyncio
async def test_host_provider_refuses_cross_tenant_desktop_reassignment():
    provider, _, _ = _provider()
    workspace = await provider.create(
        "tenant-a",
        "session-a",
        ComputerWorkspaceType.MISSION_COMPUTER,
        ComputerProfile.DEBIAN_ENGINEERING,
    )

    with pytest.raises(RuntimeError, match="another tenant"):
        await provider.create("tenant-b", "session-b")
    assert await provider.destroy(workspace.id)
    assert (await provider.create("tenant-b", "session-b")).tenant_id == "tenant-b"


@pytest.mark.asyncio
async def test_focus_verification_fails_closed_without_an_evaluated_checkpoint(monkeypatch):
    """AVCS is an optional reflex tool: focus verification must fail closed when
    no evaluated local focus checkpoint is configured, and an unqualified
    boolean 'success' must never be accepted as focus evidence."""
    from sonic.computer_use.avcs_perception import AVCSPerception

    monkeypatch.delenv("SONIC_AVCS_FOCUS_CHECKPOINT", raising=False)
    perception = AVCSPerception()
    assert perception.focus_verifier_configured is False
    assert perception.focus_verifier_ready is False

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="SONIC_AVCS_FOCUS_CHECKPOINT is not configured"):
        perception.verify_focus_rgb_bytes(frame.tobytes(), 4, 4, 2, 1, "text field")


@pytest.mark.asyncio
async def test_focus_metadata_validation_rejects_unqualified_verifier():
    """A verifier without demonstration provenance / held-out evidence must be
    rejected — a bare `focused=True` is not qualifying evidence."""
    from sonic.computer_use.avcs_perception import AVCSPerception

    unqualified = SimpleNamespace(
        is_trained=True,
        checkpoint_metadata={
            "dataset_provenance": "something_else",
            "evaluation": {"metric": "focus_verification_precision", "passed": True},
        },
    )
    with pytest.raises(ValueError):
        AVCSPerception._validated_focus_metadata(unqualified)


@pytest.mark.asyncio
async def test_desktop_screenshot_route_reattaches_host_for_authenticated_tenant(monkeypatch):
    user = User(
        email="operator@example.test",
        name="Operator",
        role=UserRole.OPERATOR,
        tenant_id="host-route-tenant",
    )
    screen = SimpleNamespace(
        model_dump=lambda: {
            "screenshot_base64": "existing-explicit-snapshot",
            "width": 5,
            "height": 4,
            "desktop_state": "INTERACTIVE",
        },
        raw_rgb=b"",
        width=5,
        height=4,
    )
    computer = SimpleNamespace(
        screenshot=AsyncMock(return_value=screen),
        host_workspace_id="windows-host-desktop",
        reattach_persisted_workspace=True,
    )
    monkeypatch.setattr(
        "sonic.api.routes.workstation._session_workspace_id",
        lambda _user, _session_id: "windows-host-desktop",
    )
    monkeypatch.setattr("sonic.api.routes.workstation.get_daytona_computer", lambda: computer)
    reattach_call = AsyncMock()
    monkeypatch.setattr(
        "sonic.api.routes.workstation._reattach_persisted_host_desktop",
        reattach_call,
    )

    response = await get_desktop_screenshot(session_id="host-session", user=user)

    assert response["screenshot_base64"] == "existing-explicit-snapshot"
    computer.screenshot.assert_awaited_once_with(workspace_id="windows-host-desktop")
    reattach_call.assert_awaited_once_with(
        computer, "host-route-tenant", "host-session", "windows-host-desktop"
    )


def test_host_workstation_provider_is_explicitly_opt_in(monkeypatch):
    monkeypatch.setattr("sonic.api.routes.workstation._primary_computer_instance", None)
    monkeypatch.setattr("sonic.api.routes.workstation._daytona_provider_instance", None)
    monkeypatch.setenv("SONIC_WORKSTATION_PROVIDER", "windows_host_avcs")

    computer = get_computer()

    assert isinstance(computer, WindowsHostAVCSComputerProvider)


def test_docker_remains_the_default_workstation_provider(monkeypatch):
    from sonic.computer.docker_computer import DockerComputerProvider

    monkeypatch.setattr("sonic.api.routes.workstation._primary_computer_instance", None)
    monkeypatch.setattr("sonic.api.routes.workstation._daytona_provider_instance", None)
    monkeypatch.delenv("SONIC_WORKSTATION_PROVIDER", raising=False)
    monkeypatch.delenv("SONIC_USE_DAYTONA_CLOUD", raising=False)

    assert isinstance(get_computer(), DockerComputerProvider)
