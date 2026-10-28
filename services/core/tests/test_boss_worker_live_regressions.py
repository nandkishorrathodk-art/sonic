"""Regressions for live Boss worker dispatch, identity, and GUI worker progress.

These cover four defects reproduced end-to-end against a live workstation:

1. The GUI-only system prompt told the model to supply a ``postcondition`` but
   never showed a concrete shape, and the accompanying action instructions
   forbade the explicit pixel form the same prompt asked for. A capable model
   still omitted the key, so every pointer action was rejected before dispatch
   with ``gui-postcondition-required`` and the worker produced no real work.
2. The GUI reasoning budget was a hardcoded 60s constant. A slow-but-working
   vision route (large fallback model) exceeded it and the worker died with
   ``reasoning-unavailable`` instead of finishing.
3. ``APP_LAUNCH`` targets of the form ``application:mousepad`` (a role
   qualifier the model prepends) were resolved literally, so a genuinely
   installed application failed to launch.
4. The worker board must expose a stable per-worker id, a live status, and the
   latest real activity so the dashboard can render a clickable live view.
"""

from __future__ import annotations

import inspect

from sonic.computer.provider import ComputerProvider
from sonic.computer_use.agent import (
    ComputerUseAgent,
    _gui_reasoning_timeout_seconds,
)


class _StubComputer(ComputerProvider):
    """Minimal provider; only construction is needed to build a prompt."""

    async def create(self, *args, **kwargs):
        return type("Workspace", (), {"id": "workspace"})()

    async def destroy(self, *args, **kwargs):
        return True

    async def status(self, workspace_id):  # pragma: no cover - unused here
        raise NotImplementedError

    async def screenshot(self, workspace_id):  # pragma: no cover - unused here
        raise NotImplementedError

    async def terminal(self, *args, **kwargs):  # pragma: no cover - unused here
        raise NotImplementedError

    async def list_files(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def read_file(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def write_file(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def git_action(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def gui_action(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def launch_application(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def close_application(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def install_application(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def uninstall_application(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def application_list(self, *args, **kwargs):  # pragma: no cover
        return []

    async def process_list(self, *args, **kwargs):  # pragma: no cover
        return []

    async def service_action(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _gui_system_prompt() -> str:
    agent = ComputerUseAgent(computer_provider=_StubComputer())
    return agent._gui_only_system_prompt()


def test_gui_system_prompt_shows_concrete_postcondition_example():
    prompt = _gui_system_prompt()
    # The model needs the literal key and an example value, not just prose.
    assert "postcondition" in prompt
    assert "screen_changed" in prompt
    assert '"action"' in prompt and '"payload"' in prompt


def test_gui_system_prompt_states_postcondition_is_mandatory():
    prompt = _gui_system_prompt()
    assert "mandatory" in prompt.lower()


def test_gui_action_instructions_allow_explicit_pixel_targets():
    """The prompt must not forbid the pixel form it also asks for."""
    source = inspect.getsource(ComputerUseAgent)
    # The old contradictory instruction told the model to "never invent pixel
    # coordinates"; the replacement offers pixels as a first-class option.
    assert "never invent pixel coordinates or claim a target is grounded" not in source
    assert "give an explicit integer screen-pixel point" in source


def test_gui_reasoning_timeout_defaults_to_60(monkeypatch):
    monkeypatch.delenv("SONIC_GUI_REASONING_TIMEOUT_SECONDS", raising=False)
    assert _gui_reasoning_timeout_seconds() == 60.0


def test_gui_reasoning_timeout_honours_env_override(monkeypatch):
    monkeypatch.setenv("SONIC_GUI_REASONING_TIMEOUT_SECONDS", "180")
    assert _gui_reasoning_timeout_seconds() == 180.0


def test_gui_reasoning_timeout_rejects_out_of_range_and_garbage(monkeypatch):
    for value in ("5", "100000", "not-a-number", ""):
        monkeypatch.setenv("SONIC_GUI_REASONING_TIMEOUT_SECONDS", value)
        assert _gui_reasoning_timeout_seconds() == 60.0


def test_normalize_app_name_strips_role_qualifier():
    # The exact failure seen live: the model prefixed the role.
    assert ComputerUseAgent._normalize_app_name("application:mousepad") == "mousepad"
    assert ComputerUseAgent._normalize_app_name("app: mousepad") == "mousepad"
    assert ComputerUseAgent._normalize_app_name("binary=firefox") == "firefox"
    assert ComputerUseAgent._normalize_app_name("program:thunar") == "thunar"
    # A real name that merely contains a colon must survive intact.
    assert ComputerUseAgent._normalize_app_name("foo:bar") == "foo:bar"


def test_normalize_app_name_unchanged_for_plain_names():
    assert ComputerUseAgent._normalize_app_name("mousepad") == "mousepad"
    assert ComputerUseAgent._normalize_app_name("the mousepad") == "mousepad"
