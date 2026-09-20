from sonic.computer_use.replay import ReplayMetrics, promotion_gate


def _metrics(**changes):
    values = {
        "total_actions": 2,
        "verified_actions": 1,
        "successful_actions": 1,
        "failed_actions": 0,
        "safety_blocks": 0,
        "evidence_complete_actions": 2,
        "prediction_accuracy": 1.0,
        "verification_rate": 0.5,
        "repeated_action_rate": 0.0,
        "evidence_completeness": 1.0,
        "composite_score": 90.0,
    }
    values.update(changes)
    return ReplayMetrics(**values)


def test_promotion_gate_accepts_equal_or_better_candidate():
    decision = promotion_gate(_metrics(), _metrics(composite_score=92.0))
    assert decision.allowed is True


def test_promotion_gate_rejects_evidence_regression():
    decision = promotion_gate(_metrics(), _metrics(evidence_completeness=0.5))
    assert decision.allowed is False
    assert "evidence" in decision.reason


def test_promotion_gate_rejects_new_safety_blocks():
    decision = promotion_gate(_metrics(), _metrics(safety_blocks=1))
    assert decision.allowed is False
    assert "safety" in decision.reason
