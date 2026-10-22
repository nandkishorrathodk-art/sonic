"""Regression: a specific application name must never be silently substituted.

Before this fix ``_resolve_app_binary`` applied the generic POSIX-role fallback
to ANY query containing a role word, so ``gnome-terminal`` (not installed)
resolved to ``x-terminal-emulator``/``exo-open`` and ``launch_application``
returned ``True`` while opening a *different* application. That let the agent
believe it launched what it asked for, hiding a missing-dependency problem.

The fallback must apply only to a genuinely generic role query ("terminal",
"web browser"), never to a product name.
"""
from __future__ import annotations

import pytest

from sonic.computer.docker_computer import DockerComputerProvider


class _FakeExecProvider(DockerComputerProvider):
    """Resolves against a canned fake filesystem instead of a live container."""

    def __init__(self, installed: dict[str, str], generic_alts: dict[str, str]):
        # Bypass __init__ (no docker/daemon needed for resolution logic).
        self._installed = dict(installed)
        self._generic_alts = dict(generic_alts)

    async def _docker_exec(self, cmd: str, timeout: int = 60):  # type: ignore[override]
        # 1. direct `which <app>` / PATH probe
        for name, path in self._installed.items():
            if f"which {name}" in cmd or f"which '{name}'" in cmd:
                return (0, path + "\n", "")
        # 2. POSIX alternative probes (x-terminal-emulator, xdg-open, ...)
        for name, path in self._generic_alts.items():
            if f"which {name}" in cmd:
                return (0, path + "\n", "")
        # 3. .desktop lookup / anything else: not found
        return (1, "", "")


@pytest.mark.asyncio
async def test_specific_missing_app_is_not_substituted():
    provider = _FakeExecProvider(
        installed={"xfce4-terminal": "/usr/bin/xfce4-terminal"},
        generic_alts={"x-terminal-emulator": "/usr/bin/x-terminal-emulator"},
    )
    # A specific, uninstalled product name must resolve to nothing.
    assert await provider._resolve_app_binary("gnome-terminal") == ""
    assert await provider._resolve_app_binary("konsole") == ""


@pytest.mark.asyncio
async def test_generic_role_query_still_resolves():
    provider = _FakeExecProvider(
        installed={},
        generic_alts={"x-terminal-emulator": "/usr/bin/x-terminal-emulator"},
    )
    # A bare role query may be satisfied by the POSIX alternative.
    assert await provider._resolve_app_binary("terminal") == "/usr/bin/x-terminal-emulator"


@pytest.mark.asyncio
async def test_installed_specific_app_resolves_directly():
    provider = _FakeExecProvider(
        installed={"xfce4-terminal": "/usr/bin/xfce4-terminal"},
        generic_alts={},
    )
    assert (
        await provider._resolve_app_binary("xfce4-terminal")
        == "/usr/bin/xfce4-terminal"
    )


@pytest.mark.asyncio
async def test_launch_missing_app_returns_false():
    provider = _FakeExecProvider(
        installed={"xfce4-terminal": "/usr/bin/xfce4-terminal"},
        generic_alts={"x-terminal-emulator": "/usr/bin/x-terminal-emulator"},
    )
    assert await provider.launch_application("ws", "gnome-terminal") is False


def test_workstation_image_declares_a_terminal_emulator():
    """The workstation image must ship a terminal emulator the agent can launch."""
    from pathlib import Path

    dockerfile = (
        Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.workstation"
    )
    text = dockerfile.read_text(encoding="utf-8")
    assert "xfce4-terminal" in text, (
        "workstation image must install a terminal emulator; without one the "
        "agent cannot open a terminal on the desktop"
    )
