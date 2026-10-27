"""Regression tests for the operator worker-board visibility contract.

These pin two real defects found while investigating "clicking a Boss
subagent's id/name shows nothing":

1. ``ScreenObservation.raw_rgb`` held a full ~6 MB RGB frame and was rendered
   by ``repr()``. Any accidental f-string/logger interpolation of the model
   dumped the raw pixels (observed: 81 MB of logs from one run) and leaked
   workstation pixels into the log stream.
2. ``_record_step_trace`` in the workstation route hardcoded the worker
   identity fields to ``""`` on the trace-object branch, so a step that
   arrived as a ``ComputerDecisionTrace`` lost the worker it belonged to and
   could never be drilled into from the operator board.
"""

from sonic.computer.models import ScreenObservation
from sonic.computer_use.models import ComputerActionType, ComputerDecisionTrace


def _frame() -> ScreenObservation:
    return ScreenObservation(
        raw_rgb=b"\x01" * (1920 * 1080 * 3),
        frame_id=7,
        width=1920,
        height=1080,
    )


class TestRawFrameIsNotLogged:
    def test_repr_does_not_contain_raw_pixels(self):
        observation = _frame()
        rendered = repr(observation)
        # A 6 MB byte string would make repr enormous; the field must be omitted.
        assert len(rendered) < 1000
        assert "raw_rgb" not in rendered
        assert "\\x01" not in rendered

    def test_repr_stays_bounded_for_log_interpolation(self):
        # An f-string/logger call interpolates the same way repr() does.
        observation = _frame()
        assert len(f"{observation}") < 1000

    def test_model_dump_still_excludes_raw_pixels(self):
        assert "raw_rgb" not in _frame().model_dump()

    def test_raw_pixels_remain_available_to_callers(self):
        observation = _frame()
        # repr/exclude are presentation concerns only; the frame is intact.
        assert len(observation.raw_rgb) == 1920 * 1080 * 3
        assert observation.frame_id == 7


class TestObjectTraceKeepsWorkerIdentity:
    """The object branch of ``_record_step_trace`` must not erase identity."""

    def test_route_object_branch_preserves_identity_when_present(self):
        """A trace object that carries identity must round-trip it.

        ``ComputerDecisionTrace`` itself has no identity fields, so a plain
        trace is safely empty. A worker-aware trace (or subclass) that does
        carry them must not be flattened to ``""`` by the route.
        """
        from sonic.computer_use.models import ComputerDecisionTrace as Trace

        class WorkerTrace(Trace):
            sub_agent_id: str = "sub-agent-2"
            sub_agent_uid: str = "worker-2-deadbe"
            sub_agent_name: str = "Form Filler"
            sub_mission_id: str = "sub-2"
            worker_role: str = "gui_operator"
            execution_plane: str = "gui"

        trace = WorkerTrace(
            step_index=1,
            action_type=ComputerActionType.GUI_CLICK,
            target_resource="Save button",
            predicted_outcome="click lands",
            actual_observation="clicked",
            status="COMPLETED",
        )
        # Exactly the reads the route's object branch performs.
        assert str(getattr(trace, "sub_agent_uid", "") or "") == "worker-2-deadbe"
        assert str(getattr(trace, "sub_agent_name", "") or "") == "Form Filler"
        assert str(getattr(trace, "sub_mission_id", "") or "") == "sub-2"
        assert str(getattr(trace, "worker_role", "") or "") == "gui_operator"
        assert str(getattr(trace, "execution_plane", "") or "") == "gui"

    def test_missing_identity_defaults_to_empty_not_error(self):
        trace = ComputerDecisionTrace(
            step_index=1,
            action_type=ComputerActionType.TERMINAL_EXEC,
            target_resource="pwd",
            predicted_outcome="path",
            actual_observation="/root",
            status="COMPLETED",
        )
        # getattr fallback keeps a plain trace safe for the object branch.
        assert str(getattr(trace, "sub_agent_uid", "") or "") == ""
        assert str(getattr(trace, "worker_role", "") or "") == ""

    def test_route_object_branch_reads_identity_with_getattr(self):
        """Source contract: the object branch must not hardcode identity to ''."""
        import inspect

        from sonic.api.routes import workstation

        source = inspect.getsource(workstation)
        assert 'getattr(trace, "sub_agent_uid", "")' in source
        assert 'getattr(trace, "sub_agent_name", "")' in source
        assert 'getattr(trace, "worker_role", "")' in source
        assert 'getattr(trace, "execution_plane", "")' in source


class TestWorklogIsBoundedWithoutLosingWorkerIdentity:
    """A long mission must not grow the worklog without limit.

    Before the cap, every worker step appended an item and every append
    re-serialized the entire tenant/session store to disk (O(n^2) over a
    mission). The cap must trim *step* noise while keeping the
    ``sub_dispatch``/``sub_report`` markers, because the worker projection
    reconstructs each operator card from that pair -- dropping them would make
    an active worker vanish from the board.
    """

    def test_worklog_is_capped_under_sustained_appends(self):
        from sonic.api.routes import workstation

        state: dict = {"worklog": []}
        for i in range(workstation._WORKLOG_MAX_ITEMS * 3):
            workstation._append_worklog(state, "event", f"step {i}", "noise")
        assert len(state["worklog"]) <= workstation._WORKLOG_MAX_ITEMS

    def test_dispatch_and_report_markers_survive_trimming(self):
        from sonic.api.routes import workstation

        state: dict = {"worklog": []}
        workstation._append_worklog(
            state,
            "event",
            "SubAgent dispatched",
            "worker A",
            orchestration_event="sub_dispatch",
            agent_uid="worker-1-abcdef",
            agent_id="sub-agent-1",
            agent_name="File Manager Observer",
            sub_mission_id="sub-1",
            execution_plane="gui",
        )
        workstation._append_worklog(
            state,
            "event",
            "SubAgent finished",
            "worker A done",
            orchestration_event="sub_report",
            agent_uid="worker-1-abcdef",
            status="COMPLETED",
            success=True,
        )
        for i in range(workstation._WORKLOG_MAX_ITEMS * 3):
            workstation._append_worklog(state, "event", f"step {i}", "noise")

        events = [
            item.get("orchestration_event")
            for item in state["worklog"]
            if item.get("orchestration_event")
        ]
        assert "sub_dispatch" in events
        assert "sub_report" in events

    def test_objective_marker_survives_so_timeline_stays_scoped(self):
        from sonic.api.routes import workstation

        state: dict = {"worklog": []}
        workstation._append_worklog(state, "response", "Objective Received", "do the thing")
        for i in range(workstation._WORKLOG_MAX_ITEMS * 3):
            workstation._append_worklog(state, "event", f"step {i}", "noise")
        titles = [item.get("title") for item in state["worklog"]]
        assert "Objective Received" in titles

    def test_trim_preserves_chronological_order(self):
        from sonic.api.routes import workstation

        state: dict = {"worklog": []}
        for i in range(workstation._WORKLOG_MAX_ITEMS * 2):
            workstation._append_worklog(state, "event", f"step {i}", "noise")
        timestamps = [item["timestamp"] for item in state["worklog"]]
        assert timestamps == sorted(timestamps)

    def test_short_worklog_is_untouched(self):
        from sonic.api.routes import workstation

        state: dict = {"worklog": []}
        for i in range(5):
            workstation._append_worklog(state, "event", f"step {i}", "noise")
        assert len(state["worklog"]) == 5
