from avcs_suite_a.catalog import CASE_BY_ID
from avcs_suite_a.evaluator import rect_iou, score_case, state_matches, summarize
from avcs_suite_a.models import Rect


def test_iou_identical_and_disjoint_rects():
    a = Rect(10, 10, 20, 20)
    assert rect_iou(a, a) == 1.0
    assert rect_iou(a, Rect(100, 100, 10, 10)) == 0.0


def test_state_match_checks_only_expected_fields():
    assert state_matches({"saved": True}, {"saved": True, "unrelated": 5})
    assert not state_matches({"saved": True}, {"saved": False})


def test_false_confirmation_is_scored_against_oracle_state():
    case = CASE_BY_ID["button_exact"]
    before = {"state": {"saved": False}}
    after = {"state": {"saved": False}}
    run = {"status": "confirmed", "actions": []}
    score = score_case(case, before, after, Rect(0, 0, 20, 20), run)
    assert not score.task_success
    assert score.false_confirmation


def test_blocked_negative_control_requires_blocked_status_and_unchanged_state():
    case = CASE_BY_ID["disabled_control"]
    before = {"state": {"continued": False}}
    unchanged = {"state": {"continued": False}}
    good = score_case(case, before, unchanged, None, {"status": "blocked"})
    bad = score_case(case, before, unchanged, None, {"status": "confirmed"})
    changed = score_case(case, before, {"state": {"continued": True}}, None, {"status": "blocked"})
    assert good.task_success
    assert not bad.task_success
    assert not changed.task_success


def test_target_scoring_uses_iou_not_agent_claimed_label():
    case = CASE_BY_ID["button_exact"]
    target = Rect(100, 100, 100, 40)
    good = score_case(case, {}, {"state": {"saved": True}}, target,
                      {"status": "confirmed", "actions": [{"target_bbox": target.to_dict()}]})
    bad = score_case(case, {}, {"state": {"saved": True}}, target,
                     {"status": "confirmed", "actions": [{"target_bbox": Rect(500, 500, 100, 40).to_dict()}]})
    assert good.target_match is True
    assert bad.target_match is False


def test_summary_keeps_undefined_denominators_null():
    summary = summarize([])
    assert summary["task_completion_rate"] is None
    assert summary["false_confirmation_rate"] is None
    assert summary["target_precision_at_iou_0_5"] is None


def test_ambiguous_is_success_for_fixture_that_never_reaches_requested_completion():
    case = CASE_BY_ID["no_false_success"]
    before = {"state": {"refresh_requested": False, "preview_refreshed": False}}
    after = {"state": {"refresh_requested": True, "preview_refreshed": False}}
    honest = score_case(case, before, after, Rect(100, 100, 80, 30), {"status": "ambiguous"})
    overclaim = score_case(case, before, after, Rect(100, 100, 80, 30), {"status": "confirmed"})
    assert honest.task_success
    assert not honest.false_confirmation
    assert not overclaim.task_success
    assert overclaim.false_confirmation


def test_suite_case_pass_rate_does_not_let_negative_controls_boost_task_completion():
    from avcs_suite_a.models import CaseScore
    from avcs_suite_a.evaluator import summarize

    scores = [
        CaseScore("positive", "x", "confirmed", "failed", False, False, False, None, None, False, 1.0),
        CaseScore("negative", "x", "blocked", "blocked", True, True, False, None, None, False, 1.0),
    ]
    summary = summarize(scores)
    assert summary["suite_case_pass_rate"] == 0.5
    assert summary["task_completion_rate"] == 0.0
    assert summary["positive_task_cases"] == 1
