from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import SubMission


def test_boss_selects_deep_for_high_prediction_error():
    mode = BossAgent._select_process_mode(
        {"prediction_error": 0.9, "graph_changed": False, "known_skill_available": True}
    )
    assert mode == "deep"


def test_boss_selects_fast_only_for_known_low_error_state():
    mode = BossAgent._select_process_mode(
        {"prediction_error": 0.0, "graph_changed": True, "known_skill_available": True}
    )
    assert mode == "fast"


def test_sub_mission_carries_scoped_evidence_context():
    mission = SubMission(
        goal="inspect the current live state",
        process_mode="deep",
        evidence_context={"graph_changed": False, "prediction_error": 0.8},
    )
    assert mission.evidence_context["prediction_error"] == 0.8
    assert mission.process_mode == "deep"
