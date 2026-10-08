from __future__ import annotations

import asyncio

import pytest

from sonic.computer.models import GUIAction, GUIActionType, ScreenObservation
from sonic.computer_use.motor import MotorReflexes


class _Computer:
    def __init__(self, failures: set[int] | None = None):
        self.calls: list[GUIAction] = []
        self.failures = failures or set()

    async def gui_action(self, workspace_id: str, action: GUIAction):
        index = len(self.calls)
        self.calls.append(action)
        if index in self.failures:
            raise RuntimeError("dispatch failed")
        return ScreenObservation()


def _actions(count: int = 3) -> list[GUIAction]:
    return [GUIAction(action=GUIActionType.KEYPRESS, key=f"F{i}") for i in range(count)]


@pytest.mark.asyncio
async def test_execute_burst_success_and_cadence(monkeypatch):
    computer = _Computer()
    sleeps: list[float] = []

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    result = await MotorReflexes(computer).execute_burst(
        "workspace", _actions(), interval_seconds=0.02
    )
    assert result.success is True
    assert result.completed == 3
    assert result.stopped is False
    assert len(computer.calls) == 3
    assert sleeps == [0.02, 0.02]


@pytest.mark.asyncio
async def test_execute_burst_stops_on_dispatch_failure():
    computer = _Computer({1})
    result = await MotorReflexes(computer).execute_burst("workspace", _actions())
    assert result.success is False
    assert result.stopped is True
    assert result.completed == 1
    assert len(result.actions) == 2
    assert len(computer.calls) == 2
    assert "execution failed" in (result.stop_reason or "")


@pytest.mark.parametrize(
    "receipt",
    [
        {"success": False},
        {"ok": False},
    ],
)
@pytest.mark.asyncio
async def test_execute_burst_stops_on_negative_provider_receipt(receipt):
    class ReceiptComputer(_Computer):
        async def gui_action(self, workspace_id: str, action: GUIAction):
            self.calls.append(action)
            return receipt

    computer = ReceiptComputer()
    result = await MotorReflexes(computer).execute_burst("workspace", _actions())

    assert result.success is False
    assert result.completed == 0
    assert len(computer.calls) == 1
    assert "false" in (result.stop_reason or "")


@pytest.mark.asyncio
async def test_execute_burst_verifier_interrupts():
    computer = _Computer()
    seen = 0

    async def verifier(action, observation):
        nonlocal seen
        seen += 1
        return seen == 1

    result = await MotorReflexes(computer).execute_burst(
        "workspace", _actions(), verifier=verifier
    )
    assert result.success is False
    assert result.completed == 1
    assert len(computer.calls) == 2
    assert result.stop_reason == "verifier interrupted burst"


@pytest.mark.asyncio
async def test_execute_burst_interrupt_cancels_in_flight_action_and_stops():
    started = asyncio.Event()
    interrupted = asyncio.Event()

    class BlockingComputer(_Computer):
        async def gui_action(self, workspace_id: str, action: GUIAction):
            self.calls.append(action)
            started.set()
            await asyncio.Event().wait()

    computer = BlockingComputer()
    task = asyncio.create_task(
        MotorReflexes(computer).execute_burst(
            "workspace",
            _actions(),
            interrupt_event=interrupted,
        )
    )
    await started.wait()
    interrupted.set()
    result = await task

    assert result.success is False
    assert result.stopped is True
    assert result.stop_reason == "motor burst interrupted during dispatch"
    assert len(computer.calls) == 1


def test_execute_burst_validates_configuration():
    motor = MotorReflexes(_Computer())
    actions = _actions(1)

    with pytest.raises(ValueError):
        asyncio.run(motor.execute_burst("workspace", actions, interval_seconds=-0.01))
    with pytest.raises(ValueError):
        asyncio.run(motor.execute_burst("workspace", actions, interval_seconds=1.01))
    with pytest.raises(TypeError):
        asyncio.run(motor.execute_burst("workspace", [object()]))
