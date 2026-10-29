"""Verify SONIC accepts the real Laya answer contract.

These are the exact answer shapes captured from laya 0.4.17
(convaiinnovations/laya). The previous validator read ``probability``, which
the real model never emits, so every valid answer was rejected and silently
replaced by the keyword fallback.
"""

import pytest

from sonic.llm.laya_engine import LayaSystem1Engine

# Captured verbatim from the real model.
REAL_CHOICE = {
    "type": "choice",
    "choice": "gui",
    "probabilities": {"gui": 0.8204, "headless": 0.1796},
    "confidence": 0.3208,
    "answer_confidence": 0.8204,
    "action": {"act_probability": 1.0},
}
REAL_SCORE = {
    "type": "score",
    "score": 1.1508,
    "legend": {"0": "low", "1": "medium", "2": "high"},
    "probabilities": {"0": 0.163, "1": 0.5231, "2": 0.3138},
    "confidence": 0.0913,
    "answer_confidence": 0.5231,
}
REAL_NOUL = {
    "type": "noul",
    "noul": 0.6521,
    "confidence": 0.6521,
    "answer_confidence": 0.6521,
}

CHOICE_Q = {
    "plane": {
        "type": "choice",
        "instructions": "Which execution plane?",
        "criteria": {"gui": "desktop", "headless": "terminal"},
    }
}
SCORE_Q = {
    "urgency": {
        "type": "score",
        "instructions": "Urgency",
        "criteria": ["low", "medium", "high"],
    }
}
NOUL_Q = {
    "proven": {
        "type": "noul",
        "instructions": "Is there proof it ran?",
        "affirmative_anchor": "output shows completion",
        "negative_anchor": "no evidence",
    }
}


def test_real_choice_answer_is_accepted():
    result = LayaSystem1Engine._validate_backend_result(
        {"answers": {"plane": REAL_CHOICE}, "routing": {}}, CHOICE_Q, "english", "laya_local"
    )
    assert result is not None, "real Laya choice answer must not be rejected"
    assert result.get_choice("plane") == "gui"
    assert result.answers["plane"]["probability"] == pytest.approx(0.8204)


def test_real_score_above_one_is_normalised_not_rejected():
    result = LayaSystem1Engine._validate_backend_result(
        {"answers": {"urgency": REAL_SCORE}, "routing": {}}, SCORE_Q, "english", "laya_local"
    )
    assert result is not None, "real Laya score answer must not be rejected"
    score = result.get_score("urgency")
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.1508 / 2)


def test_real_noul_answer_is_accepted():
    result = LayaSystem1Engine._validate_backend_result(
        {"answers": {"proven": REAL_NOUL}, "routing": {}}, NOUL_Q, "english", "laya_local"
    )
    assert result is not None
    assert result.get_noul("proven") == pytest.approx(0.6521)


def test_fallback_shape_still_accepted():
    answer = {"choice": "headless", "probability": 0.7, "probabilities": {"headless": 0.7}}
    result = LayaSystem1Engine._validate_backend_result(
        {"answers": {"plane": answer}, "routing": {}}, CHOICE_Q, "english", "laya_vector_projection"
    )
    assert result is not None
    assert result.get_choice("plane") == "headless"


def test_choice_outside_criteria_is_rejected():
    bad = dict(REAL_CHOICE, choice="browser")
    assert (
        LayaSystem1Engine._validate_backend_result(
            {"answers": {"plane": bad}, "routing": {}}, CHOICE_Q, "english", "laya_local"
        )
        is None
    )


def test_missing_probability_is_rejected():
    bad = {"type": "choice", "choice": "gui"}
    assert (
        LayaSystem1Engine._validate_backend_result(
            {"answers": {"plane": bad}, "routing": {}}, CHOICE_Q, "english", "laya_local"
        )
        is None
    )


def test_nan_noul_is_rejected():
    bad = dict(REAL_NOUL, noul=float("nan"))
    assert (
        LayaSystem1Engine._validate_backend_result(
            {"answers": {"proven": bad}, "routing": {}}, NOUL_Q, "english", "laya_local"
        )
        is None
    )


def test_missing_answer_key_is_rejected():
    assert (
        LayaSystem1Engine._validate_backend_result(
            {"answers": {}, "routing": {}}, CHOICE_Q, "english", "laya_local"
        )
        is None
    )
