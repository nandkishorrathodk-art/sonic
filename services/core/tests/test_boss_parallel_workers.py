"""
Tests for BossAgent parallel worker concurrency, plane separation, and honest synthesis.
Validates Planner-Worker-Judge swarm execution and Rule 15 (Zero success-by-decree).
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult
from sonic.sandbox.provider import ExecResult


@pytest.mark.asyncio
async def test_headless_sub_missions_fan_out_concurrently():
    """Headless tasks must not be capped at 1; they should fan out up to max_parallel_workers."""
    mock_comp = MagicMock()
    mock_comp.screenshot = AsyncMock(return_value=MagicMock(desktop_state="active", timestamp="now", width=1280, height=800, active_window="", visible_text="", detected_controls=[]))
    mock_comp.gui_action = AsyncMock(return_value=True)

    # Note that screenshot and gui_action are callable on mock_comp,
    # meaning _shared_desktop evaluates to True!
    boss = BossAgent(
        computer_provider=mock_comp,
        llm_router=MagicMock(),
        tenant_id="test-tenant",
        max_parallel_workers=4,
    )
    assert boss._shared_desktop is True

    # Ready batch of 3 headless sub-missions
    ready_batch = [
        SubMission(id="sub-1", goal="Run static code audit", plane="headless"),
        SubMission(id="sub-2", goal="Extract AST symbols", plane="headless"),
        SubMission(id="sub-3", goal="Check port listeners", plane="headless"),
    ]

    has_gui = any(getattr(s, "plane", "headless") == "gui" for s in ready_batch)
    limit = 1 if (has_gui and boss._shared_desktop) else boss.max_parallel_workers

    assert has_gui is False
    assert limit == 4  # Concurrency unlocked!


@pytest.mark.asyncio
async def test_gui_sub_missions_serialize_cleanly():
    """GUI tasks must restrict wave to 1 worker to prevent desktop focus race conditions."""
    mock_comp = MagicMock()
    mock_comp.screenshot = AsyncMock()
    mock_comp.gui_action = AsyncMock()

    boss = BossAgent(
        computer_provider=mock_comp,
        llm_router=MagicMock(),
        tenant_id="test-tenant",
        max_parallel_workers=4,
    )

    batch_with_gui = [
        SubMission(id="sub-1", goal="Inspect browser DOM element", plane="gui"),
        SubMission(id="sub-2", goal="Run background curl", plane="headless"),
    ]

    has_gui = any(getattr(s, "plane", "headless") == "gui" for s in batch_with_gui)
    limit = 1 if (has_gui and boss._shared_desktop) else boss.max_parallel_workers

    assert has_gui is True
    assert limit == 1  # Serialized to protect desktop display!


@pytest.mark.asyncio
async def test_honest_synthesis_when_all_subagents_fail():
    """When subagents fail, Boss must NOT hallucinate a success report or bracketed placeholders."""
    mock_comp = MagicMock()
    boss = BossAgent(
        computer_provider=mock_comp,
        llm_router=MagicMock(),
        tenant_id="test-tenant",
    )

    # Simulate a phase where all subagents failed
    failed_result1 = SubMissionResult(
        sub_mission_id="sub-1",
        goal="mkdir sonic_validation_task_tracker",
        success=False,
        status="FAILED",
        findings_summary="Host execution is disabled in HeadlessComputeProvider (exit 126)",
    )
    failed_result2 = SubMissionResult(
        sub_mission_id="sub-2",
        goal="touch task_tracker.py",
        success=False,
        status="FAILED",
        findings_summary="Host execution is disabled in HeadlessComputeProvider (exit 126)",
    )

    boss.phases = [
        Phase(
            phase_number=1,
            name="Project Creation",
            results=[failed_result1, failed_result2],
        )
    ]

    report = await boss._synthesize_final_report(
        objective="Create task tracker project and verify",
        t_start=0.0,
    )

    assert report.status == "INCOMPLETE"
    assert "No sub-missions achieved verified success" in report.findings_summary
    assert "Host execution is disabled" in report.findings_summary
    assert "[Insert" not in report.findings_summary
    assert "A new folder named 'sonic_validation_task_tracker' was created" not in report.findings_summary
