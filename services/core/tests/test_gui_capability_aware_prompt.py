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


def test_prompt_without_grounder_blocks_semantic_but_allows_pixel_clicks():
    """No grounder blocks *semantic* clicks only, never the mouse.

    Live regression: the old prompt told the model to avoid GUI_CLICK entirely
    when no grounder was configured, so workers never moved the cursor (it
    stayed at 0,0) and silently launched apps instead of using the pointer.
    Pixel clicks read from the attached screenshot do not need the grounder.
    """
    prompt = _agent(grounder_ready=False)._gui_only_system_prompt()
    assert "No AVCS grounding checkpoint is configured" in prompt
    assert "fail closed" in prompt
    # The pointer must remain available via explicit pixel coordinates.
    assert '"x":<int>,"y":<int>' in prompt
    assert "Pixel clicks do not need the grounder" in prompt
    # Non-pointer fallbacks must still be offered.
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


def test_goal_relevant_apps_are_ranked_first():
    """The goal's application must survive the app-list truncation.

    Regression: a real desktop exposes dozens of settings/utility entries
    before the app the task needs, and the prompt truncated the list, so the
    model concluded the app was "not installed". Ranking by goal-token overlap
    (generic, no vendor names) must promote the requested app to the front.
    """
    apps = ["xfce4-terminal", "thunar", "xfce4-settings-editor", "mousepad"]
    ranked = ComputerUseAgent._rank_apps_for_goal(
        apps, "Open the mousepad text editor and save a note"
    )
    assert ranked[0] == "mousepad"
    assert set(ranked) == set(apps)


def test_app_ranking_is_name_agnostic_and_stable():
    """No hardcoded app names: a non-matching goal keeps discovery order, and
    an unrelated app is not promoted."""
    apps = ["alpha-app", "beta-tool", "gamma-editor"]
    # Goal mentions none of the apps -> original order preserved.
    assert ComputerUseAgent._rank_apps_for_goal(apps, "do something generic") == apps
    # Goal mentions one -> only that one is promoted, order of the rest kept.
    ranked = ComputerUseAgent._rank_apps_for_goal(apps, "please use beta-tool now")
    assert ranked[0] == "beta-tool"
    assert ranked[1:] == ["alpha-app", "gamma-editor"]
    # Empty inputs are handled without raising.
    assert ComputerUseAgent._rank_apps_for_goal([], "goal") == []
    assert ComputerUseAgent._rank_apps_for_goal(apps, "") == apps
