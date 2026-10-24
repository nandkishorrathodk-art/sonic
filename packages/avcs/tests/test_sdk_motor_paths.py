"""Regression tests for the HumanEyeClient motor paths.

``HumanEyeClient.click`` and ``type_and_verify`` used to call
``motor_driver.move_to(...)`` unconditionally. Neither ``VirtualCursorDriver``
nor ``OSCursorDriver`` defines ``move_to``; both expose
``move_to_normalized``. Every pointer action therefore raised
``AttributeError: 'VirtualCursorDriver' object has no attribute 'move_to'``,
and the defect went unnoticed because no test exercised the click path.

These tests pin the real driver capability probe so the bug cannot return.
"""
from __future__ import annotations

import pytest

from avcs.sdk import HumanEyeClient


@pytest.fixture()
def client():
    c = HumanEyeClient(source="synthetic", fps=30, force_virtual_motor=True)
    c.start()
    try:
        c.perceive()
        yield c
    finally:
        c.stop()


def test_real_driver_has_no_move_to(client):
    """Guard the premise: the drivers expose move_to_normalized, not move_to."""
    driver = client.motor_driver
    assert hasattr(driver, "move_to_normalized")
    assert not hasattr(driver, "move_to")


def test_click_with_pixel_target_succeeds(client):
    assert client.click((0.5, 0.5)) is True


def test_click_with_language_target_succeeds(client):
    assert client.click("submit button") is True


def test_double_click_succeeds(client):
    assert client.click((0.4, 0.4), double_click=True) is True


def test_type_into_succeeds(client):
    assert client.type_into((0.4, 0.4), "hello") is True


def test_type_and_verify_clears_field_then_verifies(client):
    result = client.type_and_verify((0.4, 0.4), "hi")
    assert result is not None
    assert getattr(result, "status", None) is not None


def test_point_at_raises_for_incapable_driver(client):
    """A driver with neither method must fail loudly, not silently no-op."""

    class _Incapable:
        pass

    original = client.motor_driver
    client.motor_driver = _Incapable()
    try:
        with pytest.raises(RuntimeError):
            client._point_at(0.5, 0.5)
    finally:
        client.motor_driver = original


def test_point_at_prefers_normalized_over_move_to(client):
    """When both exist, the normalized method wins (no ambiguous pixel scaling)."""
    calls: list[tuple] = []

    class _Both:
        def move_to_normalized(self, x, y):
            calls.append(("normalized", x, y))

        def move_to(self, x, y, realtime=False):
            calls.append(("pixel", x, y))

    original = client.motor_driver
    client.motor_driver = _Both()
    try:
        client._point_at(0.25, 0.75)
    finally:
        client.motor_driver = original
    assert calls == [("normalized", 0.25, 0.75)]
