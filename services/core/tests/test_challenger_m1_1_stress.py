"""
Empirical Challenger M1-1 Stress Test Suite
============================================
Adversarial stress harness challenging Worker M1's Computer & Interaction fixes:
1. Cursor move verification rejection when coordinate readback is actually wrong vs right.
2. DRAG action rejection when x2 or y2 is None vs valid bounds vs type invariants.
3. Explicit pixel-target resolution through sync and async utility entry points.
"""

import asyncio
import base64
import binascii
import random
import struct
import time
import zlib
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer.models import (
    GUIAction,
    GUIActionType,
    ScreenObservation,
)
from sonic.computer_use.grounding import (
    resolve_ui_target,
    resolve_ui_target_async,
)

pytestmark = pytest.mark.no_live_infra

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800


def _png_base64(width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT) -> str:
    """Generate minimal valid PNG base64 string for test fixtures."""
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


def _mock_docker_provider(
    cursor_output: str = "X=100\nY=200\n",
    width: int = DISPLAY_WIDTH,
    height: int = DISPLAY_HEIGHT,
) -> DockerComputerProvider:
    """Construct DockerComputerProvider fixture with mocked exec and screenshot."""
    provider = DockerComputerProvider()
    provider._nsi_request = AsyncMock(return_value=None)
    provider._docker_exec = AsyncMock(return_value=(0, cursor_output, ""))
    observation = ScreenObservation(
        screenshot_base64=_png_base64(width, height),
        width=width,
        height=height,
        desktop_state="INTERACTIVE",
        capture_age_ms=0,
    )
    provider.screenshot = AsyncMock(return_value=observation)
    provider._last_screenshot = observation
    provider._last_screenshot_time = time.monotonic()
    return provider


# ==============================================================================
# SECTION 1: Cursor Move Verification Stress Harness
# ==============================================================================

@pytest.mark.asyncio
async def test_stress_cursor_move_exact_match_generator():
    """Generator testing 25 valid coordinate pairs: matching readback MUST succeed."""
    rng = random.Random(42)
    provider = _mock_docker_provider()

    for _ in range(25):
        target_x = rng.randint(0, DISPLAY_WIDTH - 1)
        target_y = rng.randint(0, DISPLAY_HEIGHT - 1)

        simulated_output = (
            f"X={target_x}\n"
            f"Y={target_y}\n"
            f"SCREEN=0\n"
            f"WINDOW=33554434\n"
        )
        provider._docker_exec = AsyncMock(return_value=(0, simulated_output, ""))

        action = GUIAction(action=GUIActionType.MOVE, x=target_x, y=target_y)
        res = await provider.gui_action("ws-1", action)

        assert res is not None
        provider._docker_exec.assert_awaited()
        last_cmd = provider._docker_exec.await_args.args[0]
        assert f"sonic_cursor.py move {target_x} {target_y}" in last_cmd
        assert "xdotool getmouselocation --shell" in last_cmd


@pytest.mark.asyncio
async def test_stress_cursor_move_adversarial_drift_oracle():
    """Oracle testing coordinate drifts: ANY mismatch MUST raise RuntimeError."""
    rng = random.Random(1337)
    provider = _mock_docker_provider()

    drift_vectors = [
        (1, 0),     # Single-axis X + 1
        (-1, 0),    # Single-axis X - 1
        (0, 1),     # Single-axis Y + 1
        (0, -1),    # Single-axis Y - 1
        (10, 10),   # Diagonal drift
        (-5, 12),   # Arbitrary offset
        (100, -50), # Large jump
    ]

    for req_x, req_y in [(100, 200), (0, 0), (640, 400), (1279, 799)]:
        for dx, dy in drift_vectors:
            obs_x = req_x + dx
            obs_y = req_y + dy

            simulated_output = f"X={obs_x}\nY={obs_y}\nSCREEN=0\n"
            provider._docker_exec = AsyncMock(return_value=(0, simulated_output, ""))

            action = GUIAction(action=GUIActionType.MOVE, x=req_x, y=req_y)
            with pytest.raises(RuntimeError, match="Cursor move verification failed") as exc_info:
                await provider.gui_action("ws-1", action)

            err_msg = str(exc_info.value)
            assert f"requested ({req_x}, {req_y})" in err_msg
            assert f"observed ({obs_x}, {obs_y})" in err_msg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt_output,expected_obs_x,expected_obs_y",
    [
        ("Y=200\n", "None", "200"),
        ("X=100\n", "100", "None"),
        ("", "None", "None"),
        ("xdotool: command not found\n", "None", "None"),
        ("FAKE_X=100\nFAKE_Y=200\n", "None", "None"),
        ("X=-25\nY=-30\n", "-25", "-30"),
        ("X=100 \nY=200\n", "None", "200"),  # Trailing space breaks strict ^X=(-?\d+)$
    ],
)
async def test_stress_cursor_move_corrupt_or_partial_telemetry(
    corrupt_output, expected_obs_x, expected_obs_y
):
    """Stress test readback corruption, incomplete outputs, and formatting traps."""
    provider = _mock_docker_provider()
    provider._docker_exec = AsyncMock(return_value=(0, corrupt_output, ""))

    action = GUIAction(action=GUIActionType.MOVE, x=100, y=200)
    with pytest.raises(RuntimeError, match="Cursor move verification failed") as exc_info:
        await provider.gui_action("ws-1", action)

    err_msg = str(exc_info.value)
    assert f"observed ({expected_obs_x}, {expected_obs_y})" in err_msg


@pytest.mark.asyncio
async def test_stress_cursor_move_display_boundary_corners():
    """Verify cursor move at exact display borders and corners."""
    provider = _mock_docker_provider()
    corners = [
        (0, 0),
        (DISPLAY_WIDTH - 1, 0),
        (0, DISPLAY_HEIGHT - 1),
        (DISPLAY_WIDTH - 1, DISPLAY_HEIGHT - 1),
    ]

    for cx, cy in corners:
        provider._docker_exec = AsyncMock(return_value=(0, f"X={cx}\nY={cy}\n", ""))
        action = GUIAction(action=GUIActionType.MOVE, x=cx, y=cy)
        res = await provider.gui_action("ws-1", action)
        assert res is not None


# ==============================================================================
# SECTION 2: DRAG Action Boundary & Invariant Stress Harness
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "x,y,x2,y2",
    [
        (None, 100, 200, 300),
        (100, None, 200, 300),
        (100, 100, None, 300),
        (100, 100, 200, None),
        (None, None, 200, 300),
        (100, 100, None, None),
        (None, None, None, None),
    ],
)
async def test_stress_drag_missing_coordinates_raises_explicit_value_error(x, y, x2, y2):
    """DEFECT-CI-08: Prove None destination/source coordinates raise ValueError, NOT unhandled TypeError."""
    provider = _mock_docker_provider()
    action = GUIAction(action=GUIActionType.DRAG, x=x, y=y, x2=x2, y2=y2)

    with pytest.raises(ValueError) as exc_info:
        await provider.gui_action("ws-1", action)

    assert "DRAG requires start (x, y) and destination (x2, y2) coordinates" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "x,y,x2,y2",
    [
        (True, 100, 200, 200),
        (100, False, 200, 200),
        (100, 100, True, 200),
        (100, 100, 200, False),
    ],
)
async def test_stress_drag_boolean_coordinates_rejected_by_pointer_validator(x, y, x2, y2):
    """Prove booleans constructed into GUIAction are rejected by _validate_pointer_coords."""
    provider = _mock_docker_provider()
    action = GUIAction.model_construct(
        action=GUIActionType.DRAG,
        x=x,
        y=y,
        x2=x2,
        y2=y2,
    )

    with pytest.raises(ValueError, match="Pointer actions require explicit integer coordinates"):
        await provider.gui_action("ws-1", action)


@pytest.mark.parametrize(
    "x,y,x2,y2",
    [
        (100.5, 100, 200, 200),
        (100, 100.5, 200, 200),
        (100, 100, 200.5, 200),
        (100, 100, 200, 200.5),
    ],
)
def test_stress_drag_float_coordinates_rejected_at_model_boundary(x, y, x2, y2):
    """Prove floats with fractional parts are rejected at the GUIAction model boundary."""
    with pytest.raises(ValidationError):
        GUIAction(action=GUIActionType.DRAG, x=x, y=y, x2=x2, y2=y2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "x,y,x2,y2,expected_violator",
    [
        (-1, 100, 200, 200, "(-1, 100)"),
        (100, -1, 200, 200, "(100, -1)"),
        (DISPLAY_WIDTH, 100, 200, 200, f"({DISPLAY_WIDTH}, 100)"),
        (100, DISPLAY_HEIGHT, 200, 200, f"(100, {DISPLAY_HEIGHT})"),
        (100, 100, -1, 200, "(-1, 200)"),
        (100, 100, 200, -1, "(200, -1)"),
        (100, 100, DISPLAY_WIDTH, 200, f"({DISPLAY_WIDTH}, 200)"),
        (100, 100, 200, DISPLAY_HEIGHT, f"(200, {DISPLAY_HEIGHT})"),
        (100, 100, 9999, 9999, "(9999, 9999)"),
    ],
)
async def test_stress_drag_screen_bounds_enforcement(x, y, x2, y2, expected_violator):
    """Prove out-of-bounds start or destination coordinates trigger boundary ValueError."""
    provider = _mock_docker_provider()
    action = GUIAction(action=GUIActionType.DRAG, x=x, y=y, x2=x2, y2=y2)

    with pytest.raises(ValueError, match="outside captured screen bounds") as exc_info:
        await provider.gui_action("ws-1", action)

    assert expected_violator in str(exc_info.value)


@pytest.mark.asyncio
async def test_stress_drag_valid_boundaries_and_command_generation():
    """Verify valid DRAG actions dispatch smooth cursor drag command."""
    provider = _mock_docker_provider()

    valid_drags = [
        (0, 0, DISPLAY_WIDTH - 1, DISPLAY_HEIGHT - 1),
        (DISPLAY_WIDTH - 1, DISPLAY_HEIGHT - 1, 0, 0),
        (0, DISPLAY_HEIGHT - 1, DISPLAY_WIDTH - 1, 0),
        (500, 300, 500, 300),  # Zero-distance drag
    ]

    for sx, sy, dx, dy in valid_drags:
        provider._docker_exec = AsyncMock(return_value=(0, "", ""))
        action = GUIAction(action=GUIActionType.DRAG, x=sx, y=sy, x2=dx, y2=dy)
        await provider.gui_action("ws-1", action)

        last_cmd = provider._docker_exec.await_args.args[0]
        assert f"sonic_cursor.py drag {sx} {sy} {dx} {dy}" in last_cmd


# ==============================================================================
# SECTION 3: Target Resolution & Contracts Regression Harness
# ==============================================================================

def test_stress_direct_numeric_coords_sync_and_async():
    """Explicit pixel targets remain unchanged in both resolver entry points."""
    fake_screenshot = _png_base64(80, 60)

    res_sync = resolve_ui_target("300, 200", screenshot_b64=fake_screenshot, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
    assert res_sync == (300, 200)

    res_async = asyncio.run(
        resolve_ui_target_async("300, 200", screenshot_b64=fake_screenshot, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
    )
    assert res_async == (300, 200)
