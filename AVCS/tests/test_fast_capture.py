"""Unit Tests for Win32FastCapture and ThreadedScreenCapture."""
from __future__ import annotations

import sys
import time
import numpy as np
import pytest

from forge.vision.capture import (
    ScreenCapture,
    SyntheticScreen,
    ThreadedScreenCapture,
    Win32FastCapture,
    open_capture,
)


def test_win32_fast_capture_on_available_display():
    """Verify Win32FastCapture returns valid RGB uint8 frames if display is available."""
    if not Win32FastCapture.available():
        pytest.skip("Win32FastCapture not available on this platform/session")

    cap = Win32FastCapture()
    with cap:
        frame = cap.grab()
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.shape[0] > 0 and frame.shape[1] > 0


def test_win32_fast_capture_with_target_size():
    """Verify hardware StretchBlt resizing to explicit target dimensions."""
    if not Win32FastCapture.available():
        pytest.skip("Win32FastCapture not available on this platform/session")

    cap = Win32FastCapture(target_size=(320, 180))
    with cap:
        frame = cap.grab()
        assert frame.shape == (180, 320, 3)
        assert frame.dtype == np.uint8


def test_threaded_screen_capture_with_synthetic():
    """Verify ThreadedScreenCapture runs asynchronously and provides non-blocking grabs."""
    syn = SyntheticScreen(width=160, height=120)
    threaded = ThreadedScreenCapture(source=syn, target_fps=100.0)

    with threaded:
        # Initial grab should be available immediately
        f1 = threaded.grab()
        assert f1.shape == (120, 160, 3)

        # Allow worker thread to produce several frames
        time.sleep(0.08)
        seq1 = threaded.sequence_number
        assert seq1 > 0

        # Grab should be non-blocking (<1ms)
        t0 = time.perf_counter()
        f2 = threaded.grab()
        grab_time = (time.perf_counter() - t0) * 1000.0
        assert grab_time < 5.0
        assert f2.shape == (120, 160, 3)


def test_open_capture_threaded_flag():
    """Verify open_capture correctly wraps sources in ThreadedScreenCapture when requested."""
    cap = open_capture(prefer_real=False, threaded=True)
    assert isinstance(cap, ThreadedScreenCapture)
    with cap:
        frame = cap.grab()
        assert frame.ndim == 3 and frame.shape[2] == 3

