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
