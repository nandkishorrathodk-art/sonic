"""Tests for the Boss dispatch policy: WHEN to fan out vs work alone."""
from __future__ import annotations

from sonic.computer_use.dispatch_policy import (
    MODE_DIRECT,
    MODE_PARALLEL,
    MODE_SINGLE,
    Workstream,
    decide_dispatch,
    workstreams_from_sub_missions,
)


def _ws(ws_id, plane="headless", paths=(), depends=()):
    return Workstream(id=ws_id, plane=plane, owned_paths=tuple(paths), depends_on=tuple(depends))


def test_empty_objective_is_direct():
    decision = decide_dispatch([])
    assert decision.mode == MODE_DIRECT
    assert decision.max_concurrent == 0


def test_single_workstream_is_single_worker():
    decision = decide_dispatch([_ws("a")])
    assert decision.mode == MODE_SINGLE
    assert decision.max_concurrent == 0


def test_independent_headless_workstreams_fan_out():
    decision = decide_dispatch(
        [_ws("a"), _ws("b"), _ws("c")],
        max_parallel=4,
        shared_desktop=True,
    )
    assert decision.mode == MODE_PARALLEL
    assert decision.headless_parallel == 3
    assert decision.gui_parallel == 0
    assert decision.max_concurrent == 3


def test_dependent_workstreams_do_not_fan_out():
    decision = decide_dispatch(
        [_ws("a"), _ws("b", depends=("a",))],
        shared_desktop=False,
    )
    assert decision.mode == MODE_SINGLE
    assert decision.collapsed is True
    assert "depend" in decision.reason.lower() or "race" in decision.reason.lower()


def test_overlapping_owned_paths_do_not_fan_out():
    decision = decide_dispatch(
        [_ws("a", paths=("services/core/app.py",)), _ws("b", paths=("Services/Core/App.py",))],
        shared_desktop=False,
    )
    assert decision.mode == MODE_SINGLE
    assert decision.collapsed is True


def test_shared_desktop_serializes_gui_workers_but_allows_headless():
    decision = decide_dispatch(
        [_ws("gui-1", plane="gui"), _ws("gui-2", plane="gui"), _ws("h-1"), _ws("h-2")],
        max_parallel=4,
        shared_desktop=True,
    )
    # Both headless workers fan out; only ONE GUI worker joins them.
    assert decision.mode == MODE_PARALLEL
    assert decision.headless_parallel == 2
    assert decision.gui_parallel == 1
    assert decision.serialized is True
    assert "shared desktop" in decision.reason.lower()


def test_all_gui_workstreams_serialize_on_shared_desktop():
    decision = decide_dispatch(
        [_ws("gui-1", plane="gui"), _ws("gui-2", plane="gui")],
        shared_desktop=True,
    )
    assert decision.mode == MODE_SINGLE
    assert decision.serialized is True
    assert decision.collapsed is True


def test_independent_gui_workstreams_fan_out_without_shared_desktop():
    decision = decide_dispatch(
        [_ws("gui-1", plane="gui"), _ws("gui-2", plane="gui")],
        shared_desktop=False,
    )
    assert decision.mode == MODE_PARALLEL
    assert decision.gui_parallel == 2
    assert decision.serialized is False


def test_concurrency_ceiling_is_respected():
    decision = decide_dispatch(
        [_ws(f"h-{i}") for i in range(10)],
        max_parallel=3,
        shared_desktop=True,
    )
    assert decision.mode == MODE_PARALLEL
    assert decision.max_concurrent == 3


def test_decision_is_reportable_and_serializable():
    decision = decide_dispatch([_ws("a"), _ws("b")], shared_desktop=True)
    payload = decision.to_dict()
    assert payload["mode"] == MODE_PARALLEL
    assert isinstance(payload["reason"], str) and payload["reason"]
    assert payload["groups"] == [["a"], ["b"]]


def test_workstreams_from_sub_missions_is_duck_typed():
    class _SM:
        def __init__(self, sid, plane, paths, depends):
            self.id = sid
            self.plane = plane
            self.owned_paths = paths
            self.depends_on = depends

    streams = workstreams_from_sub_missions([
        _SM("s1", "headless", ["a/b.py"], []),
        _SM("s2", "gui", [], ["s1"]),
    ])
    assert [s.id for s in streams] == ["s1", "s2"]
    assert streams[0].owned_paths == ("a/b.py",)
    assert streams[1].depends_on == ("s1",)


# ---------------------------------------------------------------------------
# Integration: the decision must be applied inside boss.run(), not just exist.
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402

from sonic.computer_use.boss import BossAgent  # noqa: E402
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult  # noqa: E402


class _StubComputer:
    """Shared-desktop provider stub (has screenshot/gui_action)."""

    async def screenshot(self, *a, **k):  # pragma: no cover - never reached
        raise AssertionError("dispatch should be stubbed out")

    async def gui_action(self, *a, **k):  # pragma: no cover - never reached
        raise AssertionError("dispatch should be stubbed out")


def _run_boss_with(phase: Phase) -> tuple:
    boss = BossAgent(
        computer_provider=_StubComputer(),
        llm_router=None,
        tenant_id="t1",
        max_parallel_workers=4,
    )
    events: list[tuple[str, dict]] = []
    dispatched: list[str] = []

    async def _fake_decompose(_objective):
        return phase

    async def _fake_dispatch(workspace_id, sub_m, phase_callback=None, **kwargs):
        dispatched.append(sub_m.id)
        return SubMissionResult(
            sub_mission_id=sub_m.id,
            goal=sub_m.goal,
            role="worker",
            success=True,
            status="SUCCESS",
            findings_summary="ok",
        )

    boss._strategic_decomposition = _fake_decompose  # type: ignore[assignment]
    boss._dispatch_sub_agent = _fake_dispatch  # type: ignore[assignment]

    async def _cb(event_type, data):
        events.append((event_type, data))

    report = asyncio.run(
        boss.run("ws-1", "audit the target", phase_callback=_cb)
    )
    return boss, events, dispatched, report


def test_run_emits_dispatch_decision_and_fans_out_headless():
    phase = Phase(
        phase_number=1,
        name="recon",
        thinking="split into independent recon tasks",
        sub_missions=[
            SubMission(id="sub-1", goal="gather headers", plane="headless"),
            SubMission(id="sub-2", goal="enumerate routes", plane="headless"),
        ],
    )
    boss, events, dispatched, report = _run_boss_with(phase)
    decision_events = [d for (t, d) in events if t == "dispatch_decision"]
    assert decision_events, "boss.run must stream its dispatch decision"
    assert decision_events[0]["mode"] == "parallel"
    assert decision_events[0]["headless_parallel"] == 2
    # Both independent headless workers were actually dispatched.
    assert set(dispatched) == {"sub-1", "sub-2"}
    # The decision is durable on the report, not just a transient event.
    assert report.dispatch_decision["mode"] == "parallel"


def test_run_serializes_gui_workers_on_shared_desktop():
    phase = Phase(
        phase_number=1,
        name="gui-work",
        thinking="two GUI tasks on one desktop",
        sub_missions=[
            SubMission(id="g1", goal="edit doc", plane="gui"),
            SubMission(id="g2", goal="browse web", plane="gui"),
        ],
    )
    boss, events, dispatched, report = _run_boss_with(phase)
    decision_events = [d for (t, d) in events if t == "dispatch_decision"]
    assert decision_events
    # A single shared desktop cannot run two GUI workers at once.
    assert decision_events[0]["mode"] == "single"
    assert decision_events[0]["serialized"] is True
    assert decision_events[0]["gui_parallel"] <= 1
    # Work is still executed (sequentially), never dropped.
    assert set(dispatched) == {"g1", "g2"}


def test_run_single_workstream_stays_single_worker():
    phase = Phase(
        phase_number=1,
        name="one",
        thinking="one cohesive outcome",
        sub_missions=[SubMission(id="only", goal="do the thing", plane="headless")],
    )
    boss, events, dispatched, report = _run_boss_with(phase)
    decision_events = [d for (t, d) in events if t == "dispatch_decision"]
    assert decision_events
    assert decision_events[0]["mode"] == "single"
    assert dispatched == ["only"]

