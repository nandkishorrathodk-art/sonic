"""Unit coverage for Daytona's local, AVCS-backed screenshot contract."""

import asyncio
import base64
import time
from io import BytesIO

import pytest
from PIL import Image

from sonic.computer.daytona_computer import DaytonaComputerProvider


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (320, 200), color=(24, 48, 96)).save(buffer, format="PNG")
    return buffer.getvalue()


class _ScreenshotService:
    def __init__(self, png_bytes: bytes, timestamp_ns: int | None = None):
        self.png_bytes = png_bytes
        self.timestamp_ns = timestamp_ns

    async def take_full_screen(self):
        return type(
            "ScreenshotResponse",
            (),
            {
                "screenshot": base64.b64encode(self.png_bytes).decode("ascii"),
                "capture_timestamp_ns": self.timestamp_ns,
            },
        )()


class _AVCS:
    def __init__(self, state=None, error=None):
        self.state = state or {
            "engine": "avcs_reflex",
            "frame_index": 1,
            "width": 320,
            "height": 200,
            "scene_hash": "frame-hash",
        }
        self.error = error
        self.frames = []

    def perceive_png(self, frame):
        self.frames.append(frame)
        if self.error:
            raise self.error
        return self.state


def _provider(monkeypatch, service, avcs):
    monkeypatch.setattr(DaytonaComputerProvider, "_load_state", lambda _self: None)
    provider = DaytonaComputerProvider(api_key="")

    async def resolve(_workspace_id):
        return type(
            "Sandbox",
            (),
            {
                "computer_use": type(
                    "ComputerUse",
                    (),
                    {"screenshot": service},
                )(),
            },
        )()

    monkeypatch.setattr(provider, "_resolve_sandbox", resolve)
    provider._avcs_perception = avcs
    return provider


@pytest.mark.no_live_infra
def test_daytona_screenshot_processes_pixels_locally_with_avcs(monkeypatch):
    png_bytes = _png_bytes()
    avcs = _AVCS()
    provider = _provider(
        monkeypatch,
        _ScreenshotService(png_bytes, time.time_ns()),
        avcs,
    )

    observation = asyncio.run(provider.screenshot("workspace-a"))

    assert avcs.frames == [png_bytes]
    assert observation.screenshot_base64 == base64.b64encode(png_bytes).decode("ascii")
    assert observation.avcs_state == avcs.state
    assert observation.perception_version == 1
    assert (observation.width, observation.height) == (320, 200)
    assert observation.visible_text == ""
    assert observation.detected_controls == []
    assert observation.desktop_state == "INTERACTIVE"


@pytest.mark.no_live_infra
def test_daytona_screenshot_rejects_stale_frame_without_cached_pixels(monkeypatch):
    avcs = _AVCS()
    provider = _provider(
        monkeypatch,
        _ScreenshotService(_png_bytes(), time.time_ns() - 5_000_000_000),
        avcs,
    )
    provider._last_screenshot = object()
    provider._last_screenshot_time = 10.0
    provider._last_screenshot_workspace_id = "workspace-a"

    observation = asyncio.run(provider.screenshot("workspace-a"))

    assert observation.desktop_state == "NO_DISPLAY"
    assert observation.screenshot_base64 == ""
    assert observation.avcs_state == {}
    assert avcs.frames == []
    assert provider._last_screenshot is None
    assert provider._last_screenshot_time == 0.0
    assert provider._last_screenshot_workspace_id is None


@pytest.mark.no_live_infra
def test_daytona_screenshot_rechecks_freshness_after_avcs(monkeypatch):
    captured_at_ns = time.time_ns()
    clock = iter((captured_at_ns + 1, captured_at_ns + 2, captured_at_ns + 3_000_000_002))
    monkeypatch.setattr(
        "sonic.computer.daytona_computer.time.time_ns",
        lambda: next(clock),
    )
    avcs = _AVCS()
    provider = _provider(
        monkeypatch,
        _ScreenshotService(_png_bytes(), captured_at_ns),
        avcs,
    )

    observation = asyncio.run(provider.screenshot("workspace-a"))

    assert avcs.frames
    assert observation.desktop_state == "NO_DISPLAY"
    assert observation.screenshot_base64 == ""
    assert observation.capture_age_ms > 2_000
    assert provider._last_screenshot is None


@pytest.mark.no_live_infra
def test_daytona_avcs_error_does_not_retain_previous_frame(monkeypatch):
    png_bytes = _png_bytes()
    service = _ScreenshotService(png_bytes, time.time_ns())
    avcs = _AVCS()
    provider = _provider(monkeypatch, service, avcs)

    asyncio.run(provider.screenshot("workspace-a"))
    assert provider._last_screenshot is not None

    avcs.error = ValueError("perception failed")
    service.timestamp_ns = time.time_ns()
    with pytest.raises(RuntimeError, match="AVCS perception failed"):
        asyncio.run(provider.screenshot("workspace-a"))

    assert provider._last_screenshot is None
    assert provider._last_screenshot_time == 0.0
    assert provider._last_screenshot_workspace_id is None


@pytest.mark.no_live_infra
def test_daytona_rejects_invalid_avcs_state(monkeypatch):
    invalid_state = {
        "engine": "avcs_reflex",
        "frame_index": 0,
        "width": 320,
        "height": 200,
        "scene_hash": "frame-hash",
    }
    provider = _provider(
        monkeypatch,
        _ScreenshotService(_png_bytes(), time.time_ns()),
        _AVCS(state=invalid_state),
    )

    with pytest.raises(RuntimeError, match="AVCS returned invalid state"):
        asyncio.run(provider.screenshot("workspace-a"))

    assert provider._last_screenshot is None
    assert provider._last_screenshot_time == 0.0
