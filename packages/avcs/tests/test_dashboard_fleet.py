"""Tests for the public AVCS dashboard entry point."""
from __future__ import annotations

from avcs.dashboard import main, run_hud


def test_dashboard_entrypoint_exports_callable_commands() -> None:
    assert callable(main)
    assert callable(run_hud)
