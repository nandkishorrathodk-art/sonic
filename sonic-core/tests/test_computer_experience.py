from sonic.computer_use.experience import compare_prediction, state_signature


def test_prediction_error_is_high_when_action_produces_no_evidence():
    before = state_signature(active_window="workstation")
    record = compare_prediction(
        action="GUI_CLICK",
        predicted_outcome="a new panel appears",
        before=before,
        after=before,
        evidence="",
    )
    assert record.state_changed is False
    assert record.prediction_error == 1.0


def test_prediction_error_is_low_for_real_observed_transition():
    before = state_signature(active_window="workstation")
    after = state_signature(active_window="workstation", visible_text="panel rendered")
    record = compare_prediction(
        action="GUI_CLICK",
        predicted_outcome="a new panel appears",
        before=before,
        after=after,
        evidence="panel rendered",
    )
    assert record.state_changed is True
    assert record.prediction_error == 0.0


def test_state_signature_is_application_agnostic():
    signature = state_signature(
        active_window="arbitrary application",
        windows=["second window", "first window"],
        processes=["worker", "app"],
    )
    assert signature.windows == ("first window", "second window")
    assert signature.processes == ("app", "worker")


def test_terminal_output_is_part_of_observed_state():
    before = state_signature()
    after = state_signature(terminal_output="real sandbox output", exit_code=0)
    record = compare_prediction(
        action="TERMINAL_EXEC",
        predicted_outcome="command returns output",
        before=before,
        after=after,
        evidence="real sandbox output",
    )
    assert record.state_changed is True
    assert record.prediction_error == 0.0


def test_graph_delta_is_recorded_as_observation_evidence():
    before = state_signature(graph_hash="before")
    after = state_signature(graph_hash="after")
    record = compare_prediction(
        action="GUI_CLICK",
        predicted_outcome="a dialog appears",
        before=before,
        after=after,
        evidence="dialog visible",
        graph_delta="added=('window:dialog',); removed=()",
    )
    assert record.graph_delta.startswith("added=")
