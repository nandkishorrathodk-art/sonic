from sonic.orchestration.dual_process import DualProcessController, ProcessMode


def test_known_skill_and_low_error_choose_fast_mode():
    decision = DualProcessController.decide_from_observation(
        prediction_error=0.0,
        graph_changed=True,
        known_skill_available=True,
    )
    assert decision.mode == ProcessMode.FAST


def test_high_prediction_error_chooses_deep_mode():
    decision = DualProcessController.decide_from_observation(
        prediction_error=0.8,
        graph_changed=False,
        known_skill_available=True,
    )
    assert decision.mode == ProcessMode.DEEP


def test_unknown_static_state_without_skill_chooses_deep_mode():
    decision = DualProcessController.decide_from_observation(
        prediction_error=0.0,
        graph_changed=False,
        known_skill_available=False,
    )
    assert decision.mode == ProcessMode.DEEP
