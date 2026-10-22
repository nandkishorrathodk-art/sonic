"""Tests for AVCS-capability-aware GUI reasoning prompts.

When no evaluated local AVCS grounding checkpoint is configured, semantic
``GUI_CLICK`` targets fail closed. The GUI-only system prompt must say so and
steer the model toward routes that need no semantic grounding (launching an
observed installed application, keyboard actions) instead of letting it burn the
whole step budget on a target that can never resolve. It must also forbid
inventing an application name that is not installed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from sonic.computer_use.agent import ComputerUseAgent


def _agent(*, grounder_ready: bool) -> ComputerUseAgent:
    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=MagicMock(),
        tenant_id="prompt-check",
        execution_plane="gui",
    )
    # Stub the capability probe so the test does not need AVCS on disk.
    agent._avcs_grounder_ready = lambda: grounder_ready  # type: ignore[method-assign]
    return agent


def test_prompt_without_grounder_forbids_semantic_clicks():
    prompt = _agent(grounder_ready=False)._gui_only_system_prompt()
    assert "No AVCS grounding checkpoint is configured" in prompt
    assert "fail closed" in prompt
    # Must still offer a workable non-semantic route.
    assert "APP_LAUNCH" in prompt
    assert "GUI_HOTKEY" in prompt


def test_prompt_without_grounder_forbids_inventing_apps():
    prompt = _agent(grounder_ready=False)._gui_only_system_prompt()
    assert "Available Installed Desktop Applications" in prompt
    assert "do not invent a binary name" in prompt


def test_prompt_with_grounder_allows_semantic_targets():
    prompt = _agent(grounder_ready=True)._gui_only_system_prompt()
    assert "No AVCS grounding checkpoint is configured" not in prompt
    assert "AVCS grounding checkpoint resolves it to pixels" in prompt


def test_grounder_readiness_probe_fails_closed_without_checkpoint(monkeypatch):
    """The real probe must report False when no checkpoint env var is set."""
    monkeypatch.delenv("SONIC_AVCS_GROUNDING_CHECKPOINT", raising=False)
    agent = ComputerUseAgent(
        computer_provider=MagicMock(),
        llm_router=MagicMock(),
        tenant_id="probe-check",
        execution_plane="gui",
    )
    assert agent._avcs_grounder_ready() is False


def test_gui_prompt_is_json_action_contract():
    prompt = _agent(grounder_ready=False)._gui_only_system_prompt()
    assert "exactly one JSON object" in prompt
    assert "GOAL_COMPLETE" in prompt


def test_gui_only_observation_surfaces_installed_apps():
    """A GUI-only worker must see the installed-application list, otherwise it
    cannot know which binary to launch and guesses an uninstalled one."""
    import inspect

    from sonic.computer_use import agent as agent_mod

    src = inspect.getsource(agent_mod.ComputerUseAgent._build_reasoning_context)
    # In the GUI-only observation block (which starts with the ASSIGNED PLANE
    # banner) the installed-application inventory must appear before the AVCS
    # pixel summary, so the model can pick a real binary instead of guessing.
    start = src.index("ASSIGNED PLANE: visible desktop GUI only.")
    end = src.index("CURRENT AVCS PIXEL STATE SUMMARY", start)
    gui_branch = src[start:end]
    assert "Available Installed Desktop Applications" in gui_branch
