"""Boss delegates every task to the shared GUI-capable computer-use worker."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.agents.judge import JudgeVerdict
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import SubMission


class _MockProvider:
    async def screenshot(self, *args, **kwargs):
        return b"desktop"

    async def gui_action(self, *args, **kwargs):
        return True


class _MockLLM:
    async def complete(self, request, **kwargs):
        return MagicMock(content="{}")


@pytest.mark.asyncio
async def test_visual_mission_uses_shared_gui_computer_use_worker():
    boss = BossAgent(computer_provider=_MockProvider(), llm_router=_MockLLM())
    boss.judge.evaluate_sub_mission = AsyncMock(
        return_value=SimpleNamespace(
            verdict=JudgeVerdict.CONFIRMED,
            notes="The Judge confirms the visible GUI action receipt.",
        )
    )
    sub_mission = SubMission(
        id="sub-desktop-web",
        goal="Open the provided resource in an installed desktop application and inspect it",
        role="browser",
        plane="browser",
        max_steps=3,
    )
    worker = MagicMock()
    worker.run_mission = AsyncMock(return_value=[
        {
            "step_index": 1,
            "action_type": "GUI_CLICK",
            "target_resource": "visible application control",
            "actual_observation": "Control activated on the shared desktop",
            "status": "SUCCESS",
            "exit_code": 0,
        },
    ])
    worker.mission_summary = "Visible application control activated."

    with patch("sonic.computer_use.agent.ComputerUseAgent", return_value=worker) as make_worker:
        result = await boss._dispatch_sub_agent("ws-1", sub_mission, sub_agent_num=1)

    make_worker.assert_called_once()
    worker_config = make_worker.call_args.kwargs
    assert worker_config["execution_plane"] == "gui"
    assert worker_config["observe_desktop"] is True
    assert "browser" not in worker_config
    worker.run_mission.assert_awaited_once()
    assert result.success is True
    assert result.status == "SUCCESS"
    assert result.actions_taken == 1
