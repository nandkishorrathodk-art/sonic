from types import SimpleNamespace

from sonic.computer_use.replay import evaluate_traces


def _trace(**kwargs):
    defaults = {
        "status": "VERIFIED",
        "action_type": "GUI_CLICK",
        "target_resource": "observed control",
        "actual_observation": "state changed",
        "verification_evidence": "fresh observation confirms the result",
        "prediction_error": 0.0,
        "state_changed": True,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_replay_metrics_measure_verified_evidence():
    metrics = evaluate_traces([_trace()])
    assert metrics.total_actions == 1
    assert metrics.verified_actions == 1
    assert metrics.verification_rate == 1.0
    assert metrics.composite_score == 100.0


def test_replay_metrics_expose_repetition_and_prediction_error():
    traces = [
        _trace(status="FAILED", prediction_error=1.0, verification_evidence=""),
        _trace(status="FAILED", prediction_error=1.0, verification_evidence=""),
    ]
    metrics = evaluate_traces(traces)
    assert metrics.failed_actions == 2
    assert metrics.repeated_action_rate == 0.5
    assert metrics.prediction_accuracy == 0.0
    assert metrics.composite_score < 50


def test_replay_never_treats_safety_block_as_success():
    metrics = evaluate_traces([
        _trace(
            status="BLOCKED",
            actual_observation="Safety blocked: target is outside scope",
            verification_evidence="",
        )
    ])
    assert metrics.safety_blocks == 1
    assert metrics.successful_actions == 0
