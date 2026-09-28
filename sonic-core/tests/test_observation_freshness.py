import time
from types import SimpleNamespace

import pytest

from sonic.computer.models import ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.perception_bus import PerceptionBus


class _ObservationProvider:
    _last_nsi_frame = None

    def __init__(self, capture_age_ms: float | None):
        self.capture_age_ms = capture_age_ms

    async def screenshot(self, _workspace_id: str) -> ScreenObservation:
        return ScreenObservation(
            screenshot_base64="frame",
            capture_age_ms=self.capture_age_ms,
            width=1280,
            height=800,
            desktop_state="INTERACTIVE",
        )

    async def status(self, _workspace_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            active_application="Desktop",
            active_window="Desktop",
            open_applications=["Desktop"],
            running_processes=[],
            installed_applications=[],
            working_directory="/workspace",
        )

    async def list_files(self, _workspace_id: str, _path: str) -> list[object]:
        return []

    async def git_action(self, _workspace_id: str, _action: str) -> SimpleNamespace:
        return SimpleNamespace(branch="main", is_clean=True)

    async def terminal(self, _workspace_id: str, _command: str) -> SimpleNamespace:
        return SimpleNamespace(stdout="__sonic_obs_ready__", exit_code=0)


def _agent(provider: _ObservationProvider) -> ComputerUseAgent:
    agent = ComputerUseAgent.__new__(ComputerUseAgent)
    agent.computer = provider
    agent.execution_plane = "dual"
    agent.observe_desktop = True
    agent.gui_only = False
    agent._screen_width = 1280
    agent._screen_height = 800
    agent._last_screenshot_b64 = ""
    agent._last_action_output = {}
    agent._last_gui_action_result = {}
    agent._last_navigated_url = ""
    agent._recent_action_signatures = []
    agent.llm_router = None
    agent.browser = None
    agent.perception_bus = PerceptionBus()
    return agent


@pytest.mark.asyncio
async def test_agent_reports_observation_age_at_decision_boundary():
    agent = _agent(_ObservationProvider(80.0))
    observation = await agent.observe("ws-1")

    assert observation.observation_age_ms is not None
    assert observation.observation_age_ms >= 80.0
    assert observation.screen.capture_age_ms == 80.0

    observation.observation_received_monotonic_ns = time.perf_counter_ns() - 50_000_000
    await agent.choose_action("Observe the desktop", observation)
    assert observation.observation_age_ms is not None
    assert observation.observation_age_ms >= 130.0


@pytest.mark.asyncio
async def test_agent_keeps_observation_age_unavailable_without_capture_timestamp():
    observation = await _agent(_ObservationProvider(None)).observe("ws-1")

    assert observation.observation_age_ms is None
