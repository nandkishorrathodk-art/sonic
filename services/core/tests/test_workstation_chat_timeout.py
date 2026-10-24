"""Regression tests for the workstation chat reasoning timeout budget.

The workstation chat path previously hardcoded ``asyncio.wait_for(..., timeout=25)``.
Measured provider reasoning latency for this deployment exceeded 25 s on the slow
tail, which surfaced as intermittent "LLM Response Unavailable" worklog entries
with no code error. The budget is now env-configurable and bounded.
"""

from __future__ import annotations

import pytest

from sonic.api.routes.workstation import _workstation_chat_timeout_seconds


def test_default_timeout_is_25_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SONIC_WORKSTATION_CHAT_TIMEOUT_SECONDS", raising=False)
    assert _workstation_chat_timeout_seconds() == 25.0


def test_env_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONIC_WORKSTATION_CHAT_TIMEOUT_SECONDS", "90")
    assert _workstation_chat_timeout_seconds() == 90.0


def test_floor_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONIC_WORKSTATION_CHAT_TIMEOUT_SECONDS", "1")
    assert _workstation_chat_timeout_seconds() == 10.0


def test_ceiling_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONIC_WORKSTATION_CHAT_TIMEOUT_SECONDS", "9999")
    assert _workstation_chat_timeout_seconds() == 300.0


def test_invalid_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONIC_WORKSTATION_CHAT_TIMEOUT_SECONDS", "not-a-number")
    assert _workstation_chat_timeout_seconds() == 25.0


def test_blank_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONIC_WORKSTATION_CHAT_TIMEOUT_SECONDS", "   ")
    assert _workstation_chat_timeout_seconds() == 25.0


def test_call_site_uses_the_helper_not_a_literal() -> None:
    """Guard against the hardcoded 25 s ceiling being reintroduced."""
    import inspect

    from sonic.api.routes import workstation as workstation_module

    source = inspect.getsource(workstation_module)
    assert "timeout=_workstation_chat_timeout_seconds()" in source
    assert "timeout=25," not in source
