"""Boss dispatch policy: decide WHEN to fan out to parallel sub-agents.

The Boss already *has* the machinery to run workers concurrently (dependency
waves + ``asyncio.gather``), but the decision of *whether fan-out is even
appropriate* was implicit. That made it easy to launch a swarm of workers for
a task one worker should own, and impossible for an operator to see why.

This module makes the decision explicit, structural, and reportable:

* ``DIRECT``   - a single cohesive outcome; no delegation needed.
* ``SINGLE``   - one bounded adaptive worker owns the whole objective.
* ``PARALLEL`` - N genuinely independent workstreams dispatched concurrently.

"Independent" is *structural*, never keyword-based. Two workstreams may run
concurrently only when ALL hold:

1. neither depends on the other (no dependency edge in either direction),
2. their owned paths do not overlap (no shared mutable file surface),
3. they are not two GUI-capable workers racing a single shared physical
   desktop (focus, windows, cursor, and typed input are shared mutable state).

A single workstation therefore parallelizes headless work freely but must
serialize GUI work. That is a real constraint, not a bug, and the decision
records it honestly instead of pretending parallelism it cannot deliver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

MODE_DIRECT = "direct"
MODE_SINGLE = "single"
MODE_PARALLEL = "parallel"


@dataclass(frozen=True)
class Workstream:
    """A declared unit of work the Boss is considering dispatching."""

    id: str
    plane: str = "gui"
    owned_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()

    @property
    def is_gui(self) -> bool:
        return str(self.plane).lower().strip() != "headless"


@dataclass(frozen=True)
class DispatchDecision:
    """The Boss's orchestration verdict for one phase."""

    mode: str
    reason: str
    headless_parallel: int = 0
    gui_parallel: int = 0
    serialized: bool = False
    collapsed: bool = False
    groups: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    @property
    def max_concurrent(self) -> int:
        return self.headless_parallel + self.gui_parallel

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "headless_parallel": self.headless_parallel,
            "gui_parallel": self.gui_parallel,
            "max_concurrent": self.max_concurrent,
            "serialized": self.serialized,
            "collapsed": self.collapsed,
            "groups": [list(group) for group in self.groups],
        }


def _norm_paths(paths: Iterable[str]) -> set[str]:
    return {str(p).strip().lower() for p in paths if str(p).strip()}


def _conflicts(a: Workstream, b: Workstream, *, shared_desktop: bool) -> bool:
    """True when two workstreams must NOT run concurrently."""
    if a.id == b.id:
        return True
    if a.id in b.depends_on or b.id in a.depends_on:
        return True
    if _norm_paths(a.owned_paths) & _norm_paths(b.owned_paths):
        return True
    if shared_desktop and a.is_gui and b.is_gui:
        return True
    return False


def _independent_parallel_set(
    candidates: Sequence[Workstream], *, shared_desktop: bool
) -> list[Workstream]:
    """Greedily pick the largest mutually non-conflicting set (stable order)."""
    chosen: list[Workstream] = []
    for candidate in candidates:
        if all(not _conflicts(candidate, picked, shared_desktop=shared_desktop) for picked in chosen):
            chosen.append(candidate)
    return chosen


def decide_dispatch(
    workstreams: Sequence[Workstream],
    *,
    max_parallel: int = 4,
    shared_desktop: bool = True,
) -> DispatchDecision:
    """Decide how the Boss should dispatch a phase's workstreams.

    ``max_parallel`` is the operator-configured concurrency ceiling; the policy
    never exceeds it. ``shared_desktop`` is True when the workers share one
    physical display (the normal workstation case).
    """
    max_parallel = max(1, int(max_parallel))
    streams = list(workstreams)

    if not streams:
        return DispatchDecision(mode=MODE_DIRECT, reason="No workstreams declared.")

    if len(streams) == 1:
        return DispatchDecision(
            mode=MODE_SINGLE,
            reason="Single cohesive workstream; one adaptive worker owns it.",
        )

    # A GUI-capable worker on a shared desktop excludes every other GUI worker,
    # so it can never contribute to a concurrent set. Prefer headless work for
    # the parallel set and keep GUI work for the serialized lane.
    parallel_eligible = _independent_parallel_set(streams, shared_desktop=shared_desktop)
    headless = [w for w in parallel_eligible if not w.is_gui]
    gui = [w for w in parallel_eligible if w.is_gui]
    if not shared_desktop:
        # Independent desktops (or none) can run GUI work concurrently too.
        concurrent = parallel_eligible
        headless_n, gui_n = len(headless), len(gui)
    else:
        # At most one GUI worker runs at a time; headless workers still fan out.
        concurrent = headless + gui[:1]
        headless_n, gui_n = len(headless), min(len(gui), 1)

    # Respect the concurrency ceiling by trimming the (stable-ordered) set.
    if len(concurrent) > max_parallel:
        concurrent = concurrent[:max_parallel]
        headless_n = sum(1 for w in concurrent if not w.is_gui)
        gui_n = sum(1 for w in concurrent if w.is_gui)

    gui_total = sum(1 for w in streams if w.is_gui)
    serialized = shared_desktop and gui_total > 1

    if len(concurrent) >= 2:
        parts = []
        if headless_n:
            parts.append(f"{headless_n} independent headless workstream(s)")
        if gui_n:
            parts.append(f"{gui_n} GUI workstream on the shared desktop")
        reason = "Fanning out: " + " + ".join(parts) + "."
        if serialized:
            reason += (
                f" {gui_total} GUI workstreams share one physical desktop and "
                "will run one at a time."
            )
        return DispatchDecision(
            mode=MODE_PARALLEL,
            reason=reason,
            headless_parallel=headless_n,
            gui_parallel=gui_n,
            serialized=serialized,
            groups=tuple((w.id,) for w in concurrent),
        )

    # No two workstreams can safely run together. Do not fabricate parallelism:
    # one bounded adaptive worker owns the objective, and we say why.
    if serialized:
        reason = (
            f"{gui_total} GUI workstreams share one physical desktop, so "
            "concurrent execution is not possible; Boss runs one adaptive "
            "worker at a time."
        )
    else:
        reason = (
            "Workstreams depend on each other or share owned paths, so "
            "concurrent execution would race; one adaptive worker owns the "
            "objective."
        )
    return DispatchDecision(
        mode=MODE_SINGLE,
        reason=reason,
        serialized=serialized,
        collapsed=True,
        groups=tuple((w.id,) for w in streams),
    )


def workstreams_from_sub_missions(sub_missions: Sequence[object]) -> list[Workstream]:
    """Build policy inputs from ``SubMission`` records (duck-typed)."""
    streams: list[Workstream] = []
    for sm in sub_missions:
        depends = getattr(sm, "depends_on", None) or []
        paths = getattr(sm, "owned_paths", None) or []
        streams.append(
            Workstream(
                id=str(getattr(sm, "id", "") or f"sub-{len(streams)}"),
                plane=str(getattr(sm, "plane", "gui") or "gui"),
                owned_paths=tuple(str(p) for p in paths),
                depends_on=tuple(str(d) for d in depends),
            )
        )
    return streams
