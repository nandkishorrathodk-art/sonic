"""Closed-loop eye-hand servo: model convergence, human profile, fail-closed.

These tests exercise real code paths (the pure NumPy reach model and the async
servo loop) with fake desktops, not mocks of the code under test. A fake desktop
simulates a pointer that does not land exactly where it was asked -- the exact
condition the closed loop exists to correct.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AVCS_ROOT = _REPO_ROOT / "packages" / "avcs"
if str(_AVCS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AVCS_ROOT))

from forge.vision.servo import (  # noqa: E402
    HumanReachModel,
    ServoError,
    locate_cursor,
)
from sonic.computer_use.visual_servo import (  # noqa: E402
    ServoUnavailable,
    VisualServo,
)


class _LaggingDesktop:
    """A pointer that only travels part of the requested distance.

    This is the honest failure mode the loop is built for: the motor command and
    the observed result disagree. A closed loop converges anyway; an open loop
    would stop short and click the wrong place.
    """

    def __init__(self, start: tuple[int, int], lag: float = 0.4) -> None:
        self.x, self.y = start
        self.lag = lag
        self.moves: list[tuple[int, int]] = []

    async def read_cursor(self) -> tuple[int, int]:
        return self.x, self.y

    async def move_cursor(self, x: int, y: int) -> None:
        self.moves.append((x, y))
        self.x = int(round(self.x + (x - self.x) * (1.0 - self.lag)))
        self.y = int(round(self.y + (y - self.y) * (1.0 - self.lag)))


# ---------------------------------------------------------------- model unit


def test_reach_model_converges_on_a_target():
    model = HumanReachModel()
    steps = model.reach((0.8, 0.7), (0.1, 0.1))
    assert steps, "a reach must produce at least one submovement"
    assert steps[-1].contact_acquired is True
    assert steps[-1].distance_to_target <= model.foveal_contact_radius


def test_opening_throw_is_ballistic_then_corrective():
    """A person throws once, then corrects -- not a long sequence of tiny steps."""
    model = HumanReachModel()
    steps = model.reach((0.9, 0.9), (0.1, 0.1), max_iterations=12)
    first = steps[0]
    # The first submovement covers most of the distance...
    assert first.speed > 0.5
    # ...and any later corrections are strictly smaller (decelerating).
    later = [s.speed for s in steps[1:] if s.speed > 0]
    if later:
        assert max(later) <= first.speed


def test_contact_requires_no_movement_when_already_on_target():
    model = HumanReachModel(foveal_contact_radius=0.02)
    move = model.next_submovement((0.5, 0.5), (0.505, 0.505))
    assert move.contact_acquired is True
    assert move.dx == 0.0 and move.dy == 0.0


def test_model_never_overshoots_the_target():
    model = HumanReachModel()
    for _ in range(40):
        move = model.next_submovement((0.5, 0.5), (0.4, 0.4))
        assert abs(move.dx) <= 0.1 + 1e-9
        assert abs(move.dy) <= 0.1 + 1e-9


def test_invalid_model_parameters_are_rejected():
    with pytest.raises(ValueError):
        HumanReachModel(foveal_contact_radius=0.0)
    with pytest.raises(ValueError):
        HumanReachModel(damping=0.0)
    with pytest.raises(ValueError):
        HumanReachModel(noise=-1.0)


def test_cursor_detector_finds_a_blob_and_rejects_a_flat_frame():
    flat = np.full((200, 300, 3), 240, dtype=np.uint8)
    assert locate_cursor(flat) is None

    frame = np.full((200, 300, 3), 240, dtype=np.uint8)
    frame[100:112, 150:162] = 10  # a small dark blob (the pointer)
    located = locate_cursor(frame)
    assert located is not None
    x, y, confidence = located
    assert 0.45 < x < 0.55 and 0.45 < y < 0.55
    assert 0.0 <= confidence <= 1.0


def test_cursor_detector_rejects_a_bad_shape():
    with pytest.raises(ServoError):
        locate_cursor(np.zeros((10, 10), dtype=np.uint8))


# ------------------------------------------------------------ loop behaviour


def test_loop_converges_despite_a_lagging_pointer():
    desktop = _LaggingDesktop(start=(100, 100))
    servo = VisualServo(
        read_cursor=desktop.read_cursor,
        move_cursor=desktop.move_cursor,
        screen_size=lambda: (1920, 1080),
        tolerance_px=6,
    )
    receipt = asyncio.run(servo.reach("ws", (1400, 900)))
    assert receipt.converged is True
    assert receipt.final_residual_px <= receipt.tolerance_px
    # A lagging pointer needs corrections, so more than one move was issued.
    assert receipt.iterations >= 2
    assert len(desktop.moves) == receipt.iterations
    assert receipt.latency_ms >= 0.0


def test_loop_reports_non_convergence_honestly():
    """A pointer that never moves must NOT be reported as a successful reach."""
    desktop = _LaggingDesktop(start=(100, 100), lag=1.0)  # 100% loss: no movement
    servo = VisualServo(
        read_cursor=desktop.read_cursor,
        move_cursor=desktop.move_cursor,
        screen_size=lambda: (1920, 1080),
        tolerance_px=6,
        max_iterations=3,
    )
    receipt = asyncio.run(servo.reach("ws", (1400, 900)))
    assert receipt.converged is False
    assert receipt.final_residual_px > receipt.tolerance_px
    assert receipt.reason


def test_loop_fails_closed_without_a_pointer_reading():
    async def no_reading() -> None:
        return None

    async def noop(_x: int, _y: int) -> None:
        return None

    servo = VisualServo(
        read_cursor=no_reading,
        move_cursor=noop,
        screen_size=lambda: (1920, 1080),
    )
    with pytest.raises(ServoUnavailable):
        asyncio.run(servo.reach("ws", (500, 500)))


def test_loop_rejects_a_target_outside_the_screen():
    desktop = _LaggingDesktop(start=(10, 10))
    servo = VisualServo(
        read_cursor=desktop.read_cursor,
        move_cursor=desktop.move_cursor,
        screen_size=lambda: (800, 600),
    )
    with pytest.raises(ServoUnavailable):
        asyncio.run(servo.reach("ws", (2000, 2000)))
    # Fail-closed must mean "did not move", not "moved then complained".
    assert desktop.moves == []


def test_loop_stops_when_interrupted():
    desktop = _LaggingDesktop(start=(100, 100), lag=0.9)
    servo = VisualServo(
        read_cursor=desktop.read_cursor,
        move_cursor=desktop.move_cursor,
        screen_size=lambda: (1920, 1080),
        tolerance_px=1,
        max_iterations=20,
    )
    receipt = asyncio.run(servo.reach("ws", (1500, 1000), should_stop=lambda: True))
    assert receipt.converged is False
    assert receipt.reason == "interrupted"
    assert desktop.moves == []


def test_loop_uses_vision_when_the_provider_cannot_report_the_pointer():
    """The pixel-blob fallback must actually drive the loop, not be decorative."""
    frame = np.full((1080, 1920, 3), 240, dtype=np.uint8)

    class _Observation:
        width = 1920
        height = 1080

        @property
        def raw_rgb(self) -> bytes:
            return frame.tobytes()

    moved: list[tuple[int, int]] = []

    async def no_provider_reading():
        return None

    async def move_cursor(x: int, y: int) -> None:
        moved.append((x, y))
        # Move the blob in the frame so the next read sees the change.
        frame[:, :, :] = 240
        frame[max(0, y - 6):y + 6, max(0, x - 6):x + 6] = 10

    async def read_frame():
        return _Observation()

    servo = VisualServo(
        read_cursor=no_provider_reading,
        move_cursor=move_cursor,
        read_frame=read_frame,
        screen_size=lambda: (1920, 1080),
        tolerance_px=12,
        max_iterations=4,
    )
    # Seed a visible pointer blob so the first visual read is non-empty.
    frame[500:512, 500:512] = 10
    receipt = asyncio.run(servo.reach("ws", (1500, 900)))
    assert receipt.cursor_source == "vision"
    assert moved, "the vision path must have issued a move"


# ------------------------------------------------------- agent wiring


class _ServoProvider:
    """Minimal real provider stub exposing the pointer sensor and a lagging move.

    Deliberately not an AsyncMock of the agent: the point is to prove the agent
    drives a real closed loop whose pointer genuinely lags the command.
    """

    def __init__(self, start: tuple[int, int] = (100, 100), lag: float = 0.4) -> None:
        self.x, self.y = start
        self.lag = lag
        self.moves: list[tuple[int, int]] = []

    async def cursor_position(self, _workspace_id: str) -> tuple[int, int]:
        return self.x, self.y

    async def gui_action(self, _workspace_id: str, action) -> object:
        x = int(action.x or 0)
        y = int(action.y or 0)
        self.moves.append((x, y))
        self.x = int(round(self.x + (x - self.x) * (1.0 - self.lag)))
        self.y = int(round(self.y + (y - self.y) * (1.0 - self.lag)))
        return None


def test_agent_closed_loop_reach_converges_and_records_a_receipt():
    """The agent must actually run the loop and surface its measured receipt."""
    from sonic.computer_use.agent import ComputerUseAgent
    from sonic.safety.action_policy import ActionPolicy

    provider = _ServoProvider(start=(120, 140))
    agent = ComputerUseAgent(
        computer_provider=provider,
        execution_plane="gui",
        safety=ActionPolicy(),
        self_host=True,
    )
    agent._screen_width, agent._screen_height = 1920, 1080

    receipt = asyncio.run(agent._servo_to_pixel_target("ws", 1300, 850))

    assert isinstance(receipt, dict)
    assert receipt["converged"] is True
    assert receipt["iterations"] >= 2  # the lagging pointer forced corrections
    assert len(provider.moves) == receipt["iterations"]
    # The recorded receipt is the same one the agent keeps for its trace.
    assert agent._last_servo_receipt == receipt


def test_agent_servo_is_disabled_when_the_provider_has_no_pointer_sensor():
    """No cursor readback -> no closed loop; the agent keeps open-loop control."""
    from sonic.computer_use.agent import ComputerUseAgent

    class _NoSensor:
        pass

    agent = ComputerUseAgent(computer_provider=_NoSensor(), execution_plane="gui")
    assert agent._get_visual_servo("ws") is None
    assert asyncio.run(agent._servo_to_pixel_target("ws", 10, 10)) is None


def test_agent_servo_moves_are_rejected_by_the_safety_envelope():
    """The loop must not become a side channel that skips the policy."""
    from sonic.computer_use.agent import ComputerUseAgent

    class _DenyingPolicy:
        def evaluate(self, *_args, **_kwargs):
            class _Verdict:
                allowed = False
                needs_approval = False
                reason = "denied by test policy"

            return _Verdict()

    provider = _ServoProvider()
    agent = ComputerUseAgent(
        computer_provider=provider,
        execution_plane="gui",
        safety=_DenyingPolicy(),
    )
    agent._screen_width, agent._screen_height = 1920, 1080
    receipt = asyncio.run(agent._servo_to_pixel_target("ws", 1300, 850))
    # The reach cannot proceed: every corrective move is refused, so nothing moved
    # and the servo reports non-convergence rather than pretending success.
    assert provider.moves == []
    assert receipt is None or receipt["converged"] is False


# --------------------------------------------- every pointer action is closed-loop


def _frame_base64(color: int = 0, width: int = 1280, height: int = 800) -> str:
    import base64
    import binascii
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", binascii.crc32(body) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = (
        b"\x00" + bytes([color if y >= height // 2 else 0]) * (width * 3)
        for y in range(height)
    )
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(image).decode("ascii")


class _PointerProvider:
    """Provider stub with a pointer sensor, status, and a lagging move.

    ``cursor_position`` makes the closed loop available; ``gui_action`` records
    every dispatch and models a pointer that only travels part of the way, so a
    servo-corrected action is visibly distinguished from an open-loop one by the
    MOVE dispatches it makes.
    """

    def __init__(self, start: tuple[int, int] = (100, 100), lag: float = 0.5) -> None:
        self.x, self.y = start
        self.lag = lag
        self.moves: list[tuple[int, int]] = []
        self.actions: list[str] = []
        self._frame = _frame_base64()
        self._after = _frame_base64(200)

    async def cursor_position(self, _workspace_id: str) -> tuple[int, int]:
        return self.x, self.y

    async def status(self, _workspace_id: str) -> object:
        from unittest.mock import MagicMock

        return MagicMock(
            active_application="Desktop",
            active_window="Desktop",
            open_applications=["Desktop"],
            running_processes=[],
        )

    async def screenshot(self, _workspace_id: str) -> object:
        from sonic.computer.models import ScreenObservation

        return ScreenObservation(
            width=1280,
            height=800,
            screenshot_base64=self._frame,
            capture_age_ms=0,
            controls_age_ms=0,
            active_window="Desktop",
        )

    async def gui_action(self, _workspace_id: str, action) -> object:
        from sonic.computer.models import ScreenObservation

        name = str(getattr(action.action, "value", action.action))
        self.actions.append(name)
        x = int(action.x or 0)
        y = int(action.y or 0)
        if name == "MOVE":
            self.moves.append((x, y))
        self.x = int(round(self.x + (x - self.x) * (1.0 - self.lag)))
        self.y = int(round(self.y + (y - self.y) * (1.0 - self.lag)))
        return ScreenObservation(
            width=1280,
            height=800,
            screenshot_base64=self._after,
            capture_age_ms=0,
            controls_age_ms=0,
            active_window="Desktop",
        )


def _closed_loop_agent(provider):
    import time

    from sonic.computer.models import ScreenObservation
    from sonic.computer_use.agent import ComputerUseAgent
    from sonic.computer_use.models import ComputerWorldObservation
    from sonic.safety.action_policy import ActionPolicy

    agent = ComputerUseAgent(
        computer_provider=provider,
        execution_plane="gui",
        safety=ActionPolicy(),
    )
    screen = ScreenObservation(
        width=1280,
        height=800,
        screenshot_base64=provider._frame,
        capture_age_ms=0,
        controls_age_ms=0,
        active_window="Desktop",
    )
    screen = agent._ensure_avcs_state(screen)
    agent._last_world_observation = ComputerWorldObservation(
        screen=screen,
        active_application="Desktop",
        perception_version=screen.perception_version,
        observation_age_ms=0,
    )
    agent._last_gui_state_observation = screen
    agent._last_screen_observation_received_at = time.monotonic()
    agent._last_screenshot_b64 = screen.screenshot_base64
    agent._screen_width = 1280
    agent._screen_height = 800
    return agent


@pytest.mark.parametrize(
    ("action_type", "payload", "expected_actions"),
    [
        ("GUI_DOUBLE_CLICK", {"x": 900, "y": 700}, {"DOUBLE_CLICK"}),
        ("GUI_RIGHT_CLICK", {"x": 900, "y": 700}, {"RIGHT_CLICK"}),
        ("GUI_DRAG", {"x": 900, "y": 700, "x2": 1200, "y2": 300}, {"DRAG"}),
        ("GUI_SCROLL", {"x": 900, "y": 700, "delta": -3}, {"SCROLL"}),
    ],
)
def test_pointer_actions_reach_before_acting(action_type, payload, expected_actions):
    """Every pointer action must converge the hand before it acts.

    The reach is what makes a click land where it was aimed; a double-click,
    right-click, drag or scroll that skips it is exactly as open-loop as a raw
    coordinate jump. Each action must therefore dispatch at least one MOVE
    (the closed-loop reach) before its own provider action.
    """
    from sonic.computer_use.models import ComputerActionType

    provider = _PointerProvider(start=(100, 100), lag=0.5)
    agent = _closed_loop_agent(provider)

    asyncio.run(
        agent.execute_action(
            "ws",
            getattr(ComputerActionType, action_type),
            f"{payload['x']},{payload['y']}",
            payload,
            "reach then act",
        )
    )

    assert expected_actions.issubset(set(provider.actions)), provider.actions
    assert provider.moves, f"{action_type} did not run a closed-loop reach"


def test_double_click_does_not_reach_when_the_provider_has_no_pointer_sensor():
    """No cursor readback -> the action keeps its previous open-loop behaviour."""
    from sonic.computer_use.models import ComputerActionType

    provider = _PointerProvider()
    provider.cursor_position = None  # type: ignore[assignment]
    agent = _closed_loop_agent(provider)

    asyncio.run(
        agent.execute_action(
            "ws",
            ComputerActionType.GUI_DOUBLE_CLICK,
            "900,700",
            {"x": 900, "y": 700},
            "double-click",
        )
    )

    assert provider.moves == []
    assert "DOUBLE_CLICK" in provider.actions
