"""Tests for screenshot-grounded GUI reasoning prompts."""
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


def test_prompt_without_grounder_uses_attached_screenshot_coordinates():
    prompt = _agent(grounder_ready=False)._gui_only_system_prompt()
    assert "attached fresh workstation screenshot" in prompt
    assert "screen-pixel coordinates" in prompt
    assert "coordinate_space to screen_pixels" in prompt
    assert "Do not use AVCS to identify controls" in prompt
    assert "APP_LAUNCH" in prompt
    assert "GUI_HOTKEY" in prompt


def test_prompt_without_grounder_forbids_inventing_apps():
    prompt = _agent(grounder_ready=False)._gui_only_system_prompt()
    assert "Available Installed Desktop Applications" in prompt
    assert "do not invent a binary name" in prompt


def test_prompt_does_not_depend_on_grounder_readiness():
    prompt = _agent(grounder_ready=True)._gui_only_system_prompt()
    assert "No AVCS grounding checkpoint is configured" not in prompt
    assert "attached fresh workstation screenshot" in prompt
    assert "screen-pixel coordinates" in prompt


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
    end = src.index("OPTIONAL AVCS_REFLEX TOOL OUTPUT", start)
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
