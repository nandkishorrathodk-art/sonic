"""
Tests for Multi-Window Management, Window Sizing & Structured Workflow:
1. ComputerActionType & GUIActionType have APP_MAXIMIZE, WINDOW_MAXIMIZE, and WINDOW_TILE.
2. DockerComputerProvider window maximize and tiling handlers.
3. ComputerUseAgent action parsing and execution dispatch for maximize and tiling.
4. BossAgent structured task decomposition (enable_llm_decomposition=True) for multi-step GUI missions.
5. Workstation prompts document APP_MAXIMIZE and WINDOW_TILE.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer.models import GUIAction, GUIActionType, ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerWorldObservation,
    SubGoalChecklist,
)
from sonic.computer_use.models_boss import SubMission
from sonic.llm.prompts import (
    COMPUTER_USE_SYSTEM_PROMPT,
    COMPUTER_USE_SYSTEM_PROMPT_COMPACT,
    EXECUTION_GROUND_TRUTH_CLAUSE,
)
from sonic.safety.action_policy import ActionPolicy


# =========================================================================
# 1. Models & Action Types Tests
# =========================================================================

def test_action_type_enums_contain_window_management():
    """Verify ComputerActionType and GUIActionType contain maximize and tile actions."""
    assert hasattr(ComputerActionType, "APP_MAXIMIZE")
    assert ComputerActionType.APP_MAXIMIZE == "APP_MAXIMIZE"
    assert hasattr(ComputerActionType, "WINDOW_MAXIMIZE")
    assert ComputerActionType.WINDOW_MAXIMIZE == "WINDOW_MAXIMIZE"
    assert hasattr(ComputerActionType, "WINDOW_TILE")
    assert ComputerActionType.WINDOW_TILE == "WINDOW_TILE"

    assert hasattr(GUIActionType, "APP_MAXIMIZE")
    assert GUIActionType.APP_MAXIMIZE == "APP_MAXIMIZE"
    assert hasattr(GUIActionType, "WINDOW_MAXIMIZE")
    assert GUIActionType.WINDOW_MAXIMIZE == "WINDOW_MAXIMIZE"
    assert hasattr(GUIActionType, "WINDOW_TILE")
    assert GUIActionType.WINDOW_TILE == "WINDOW_TILE"


def test_safety_policy_allows_window_management_actions():
    """Verify ActionPolicy default allowed types include window maximize and tile."""
    policy = ActionPolicy(tenant_id="test-tenant")
    for action in ("APP_MAXIMIZE", "WINDOW_MAXIMIZE", "WINDOW_TILE"):
        assert action in policy.allowed_types, f"{action} must be allowed under default ActionPolicy"


# =========================================================================
# 2. DockerComputerProvider Execution Handlers
# =========================================================================

@pytest.mark.asyncio
async def test_docker_computer_gui_action_maximize_active_window():
    """Verify DockerComputerProvider maximizes the active window via wmctrl -r :ACTIVE:."""
    provider = DockerComputerProvider(container_name="test-workstation")
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider.screenshot = AsyncMock(return_value=ScreenObservation(active_window="TextEditor"))

    action = GUIAction(action=GUIActionType.APP_MAXIMIZE)
    obs = await provider.gui_action("ws-1", action)

    assert obs.active_window == "TextEditor"
    executed = [c[0][0] for c in provider._docker_exec.call_args_list]
    assert any("wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz" in cmd for cmd in executed)


@pytest.mark.asyncio
async def test_docker_computer_gui_action_maximize_hex_window_id():
    """Verify DockerComputerProvider maximizes by hex window id via wmctrl -i -r <wid>."""
    provider = DockerComputerProvider(container_name="test-workstation")
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider.screenshot = AsyncMock(return_value=ScreenObservation(active_window="0x02a00003"))

    action = GUIAction(action=GUIActionType.WINDOW_MAXIMIZE, window_id="0x02a00003")
    obs = await provider.gui_action("ws-1", action)

    executed = [c[0][0] for c in provider._docker_exec.call_args_list]
    assert any("wmctrl -i -r 0x02a00003 -b add,maximized_vert,maximized_horz" in cmd for cmd in executed)


@pytest.mark.asyncio
async def test_docker_computer_gui_action_maximize_named_window():
    """Verify DockerComputerProvider maximizes by window name."""
    provider = DockerComputerProvider(container_name="test-workstation")
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))
    provider.screenshot = AsyncMock(return_value=ScreenObservation(active_window="Browser"))

    action = GUIAction(action=GUIActionType.APP_MAXIMIZE, app_name="Browser")
    obs = await provider.gui_action("ws-1", action)

    executed = [c[0][0] for c in provider._docker_exec.call_args_list]
    assert any("wmctrl -r Browser -b add,maximized_vert,maximized_horz" in cmd for cmd in executed)


@pytest.mark.asyncio
async def test_docker_computer_gui_action_window_tile():
    """Verify DockerComputerProvider tiles windows when GUIActionType.WINDOW_TILE is dispatched."""
    provider = DockerComputerProvider(container_name="test-workstation")
    provider._docker_exec = AsyncMock(return_value=(0, "0x01 0 host App1\n0x02 0 host App2\n", ""))
    provider.screenshot = AsyncMock(return_value=ScreenObservation(active_window="App1"))

    action = GUIAction(action=GUIActionType.WINDOW_TILE)
    obs = await provider.gui_action("ws-1", action)

    executed = [c[0][0] for c in provider._docker_exec.call_args_list]
    assert any("wmctrl -l" in cmd for cmd in executed)
    assert any("wmctrl -i -r \"0x01\" -e 0,0,0,640,800" in cmd for cmd in executed)
    assert any("wmctrl -i -r \"0x02\" -e 0,640,0,640,800" in cmd for cmd in executed)


@pytest.mark.asyncio
async def test_docker_computer_maximize_window_method():
    """Verify DockerComputerProvider.maximize_window method."""
    provider = DockerComputerProvider(container_name="test-workstation")
    provider._docker_exec = AsyncMock(return_value=(0, "", ""))

    ok = await provider.maximize_window("ws-1")
    assert ok is True
    executed = [c[0][0] for c in provider._docker_exec.call_args_list]
    assert any("wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz" in cmd for cmd in executed)


# =========================================================================
# 3. ComputerUseAgent Action Parsing & Execution
# =========================================================================

def test_agent_parses_maximize_and_tile_actions():
    """Verify ComputerUseAgent._parse_llm_action correctly parses APP_MAXIMIZE and WINDOW_TILE."""
    # Test APP_MAXIMIZE
    text_max = (
        "THOUGHT: Maximize the active text editor for full canvas view\n"
        "ACTION: APP_MAXIMIZE\n"
        "TARGET: text-editor\n"
        "PAYLOAD: {\"app_name\": \"text-editor\"}\n"
    )
    act_type, target, payload, _ = ComputerUseAgent._parse_llm_action(text_max, "README.md")
    assert act_type == ComputerActionType.APP_MAXIMIZE
    assert target == "text-editor"

    # Test WINDOW_MAXIMIZE alias
    text_wmax = (
        "THOUGHT: Maximize the active window\n"
        "ACTION: WINDOW_MAXIMIZE\n"
        "TARGET: :ACTIVE:\n"
        "PAYLOAD: {}\n"
    )
    act_type2, target2, _, _ = ComputerUseAgent._parse_llm_action(text_wmax, "README.md")
    assert act_type2 == ComputerActionType.WINDOW_MAXIMIZE

    # Test WINDOW_TILE
    text_tile = (
        "THOUGHT: Tile open windows side-by-side\n"
        "ACTION: WINDOW_TILE\n"
        "TARGET:\n"
        "PAYLOAD: {}\n"
    )
    act_type3, _, _, _ = ComputerUseAgent._parse_llm_action(text_tile, "README.md")
    assert act_type3 == ComputerActionType.WINDOW_TILE


@pytest.mark.asyncio
async def test_agent_executes_maximize_action():
    """Verify ComputerUseAgent.execute_action dispatches APP_MAXIMIZE to provider.gui_action."""
    provider = MagicMock()
    provider.screenshot = AsyncMock(return_value=ScreenObservation(active_window="mousepad", width=1280, height=800))
    provider.gui_action = AsyncMock(return_value=ScreenObservation(active_window="mousepad", width=1280, height=800))

    agent = ComputerUseAgent(computer_provider=provider, llm_router=None)

    trace = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.APP_MAXIMIZE,
        target_resource="mousepad",
        payload={"app_name": "mousepad"},
        predicted_outcome="mousepad window maximized",
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert "Maximized window" in trace.actual_observation
    provider.gui_action.assert_called_once()
    gui_act = provider.gui_action.call_args[0][1]
    assert gui_act.action == GUIActionType.APP_MAXIMIZE
    assert gui_act.app_name == "mousepad"


@pytest.mark.asyncio
async def test_agent_executes_window_tile_action():
    """Verify ComputerUseAgent.execute_action dispatches WINDOW_TILE to provider.gui_action."""
    provider = MagicMock()
    provider.screenshot = AsyncMock(return_value=ScreenObservation(width=1280, height=800))
    provider.gui_action = AsyncMock(return_value=ScreenObservation(width=1280, height=800))

    agent = ComputerUseAgent(computer_provider=provider, llm_router=None)

    trace = await agent.execute_action(
        workspace_id="ws-1",
        action_type=ComputerActionType.WINDOW_TILE,
        target_resource="",
        payload={},
        predicted_outcome="windows tiled side by side",
    )

    assert trace.status == ActionExecutionStatus.COMPLETED
    assert "Tiled workstation windows" in trace.actual_observation
    provider.gui_action.assert_called_once()
    gui_act = provider.gui_action.call_args[0][1]
    assert gui_act.action == GUIActionType.WINDOW_TILE


# =========================================================================
# 4. BossAgent Structured Task Decomposition for Multi-Step GUI Sub-Missions
# =========================================================================

@pytest.mark.asyncio
async def test_boss_enables_llm_decomposition_for_multi_step_gui_submission():
    """Verify BossAgent creates ComputerUseAgent with enable_llm_decomposition=True
    for multi-step GUI sub-missions (e.g. Focus -> Maximize -> Type -> Save)."""
    provider = MagicMock()
    provider.screenshot = AsyncMock(return_value=ScreenObservation(width=1280, height=800))

    boss = BossAgent(
        computer_provider=provider,
        llm_router=None,
        tenant_id="test-tenant",
    )

    sub_m = SubMission(
        goal="Focus editor, maximize window, type code, and save file",
        plane="gui",
        max_steps=5,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run_mission = AsyncMock(return_value={"status": "COMPLETED", "traces": []})
        mock_instance.history = []
        mock_instance.action_counter = 3
        mock_instance.metrics = MagicMock()
        mock_instance.metrics.model_dump.return_value = {}
        MockAgent.return_value = mock_instance

        await boss._execute_sub_mission("ws-1", sub_m)

        MockAgent.assert_called_once()
        kwargs = MockAgent.call_args[1]
        assert kwargs["enable_llm_decomposition"] is True, "Multi-step GUI sub-mission must enable LLM decomposition"


@pytest.mark.asyncio
async def test_boss_respects_explicit_sub_mission_decomposition_flag():
    """Verify BossAgent respects explicit enable_llm_decomposition=False on sub-mission."""
    provider = MagicMock()
    boss = BossAgent(computer_provider=provider, llm_router=None, tenant_id="test-tenant")

    sub_m = SubMission(
        goal="Quick single click",
        plane="gui",
        max_steps=5,
        enable_llm_decomposition=False,
    )

    with patch("sonic.computer_use.agent.ComputerUseAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run_mission = AsyncMock(return_value={"status": "COMPLETED", "traces": []})
        mock_instance.history = []
        mock_instance.action_counter = 1
        mock_instance.metrics = MagicMock()
        mock_instance.metrics.model_dump.return_value = {}
        MockAgent.return_value = mock_instance

        await boss._execute_sub_mission("ws-1", sub_m)

        MockAgent.assert_called_once()
        kwargs = MockAgent.call_args[1]
        assert kwargs["enable_llm_decomposition"] is False


# =========================================================================
# 5. Workstation Prompts Documentation
# =========================================================================

def test_prompts_document_app_maximize_and_window_tile():
    """Verify workstation prompts document APP_MAXIMIZE and WINDOW_TILE."""
    # Check COMPUTER_USE_SYSTEM_PROMPT
    assert "APP_MAXIMIZE" in COMPUTER_USE_SYSTEM_PROMPT
    assert "WINDOW_TILE" in COMPUTER_USE_SYSTEM_PROMPT
    assert "WINDOW_MAXIMIZE" in COMPUTER_USE_SYSTEM_PROMPT

    # Check COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "APP_MAXIMIZE" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "WINDOW_TILE" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT

    # Check EXECUTION_GROUND_TRUTH_CLAUSE
    assert "APP_MAXIMIZE" in EXECUTION_GROUND_TRUTH_CLAUSE
    assert "WINDOW_TILE" in EXECUTION_GROUND_TRUTH_CLAUSE
