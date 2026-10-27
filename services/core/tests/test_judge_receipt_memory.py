"""Judge receipts must gate being-memory credit (requirement: Judge receipts -> memory).

Before this seam was closed, the workstation route credited a mission to the
being's cross-session memory as a *verified success* whenever the Boss reported
``status == COMPLETE`` -- even though the route never passed a
``verification_receipt_id``. ``BeingMind.record_mission_outcome`` therefore
computed ``verified_success = False`` and the confirmed work earned no satiety /
curiosity credit. These tests pin the honest contract:

* a receipt is minted only from real Judge CONFIRMED verdicts,
* an empty receipt is produced when nothing was independently confirmed,
* the receipt gates the affect credit in ``record_mission_outcome``.
"""
from __future__ import annotations

from sonic.being.identity import BeingMind
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models_boss import Phase, SubMissionResult


def _boss_with_results(results: list[SubMissionResult]) -> BossAgent:
    boss = BossAgent.__new__(BossAgent)
    boss.phases = [Phase(phase_number=1, name="test", results=results)]
    return boss


def _result(verdict: str, verified: bool, uid: str = "worker-1-aaaa") -> SubMissionResult:
    return SubMissionResult(
        sub_mission_id="sm-1",
        agent_uid=uid,
        goal="do the thing",
        success=verified,
        evidence_verified=verified,
        judge_verdict=verdict,
        status="SUCCESS" if verified else "FAILED",
    )


def test_receipt_minted_from_confirmed_judge_verdicts():
    boss = _boss_with_results(
        [
            _result("CONFIRMED", True, "worker-1-aaaa"),
            _result("CONFIRMED", True, "worker-2-bbbb"),
        ]
    )
    receipt = boss._build_judge_receipt()
    assert receipt["receipt_id"].startswith("judge-")
    assert receipt["confirmed_sub_missions"] == 2
    assert receipt["source"] == "independent_judge"


def test_receipt_is_empty_without_a_confirmed_verdict():
    boss = _boss_with_results(
        [
            _result("INCONCLUSIVE", False, "worker-1-aaaa"),
            _result("FALSIFIED", False, "worker-2-bbbb"),
        ]
    )
    receipt = boss._build_judge_receipt()
    assert receipt["receipt_id"] == ""
    assert receipt["confirmed_sub_missions"] == 0


def test_receipt_ignores_verdict_without_verified_evidence():
    # A CONFIRMED string alone is not enough: evidence_verified must be True.
    boss = _boss_with_results([_result("CONFIRMED", False, "worker-1-aaaa")])
    receipt = boss._build_judge_receipt()
    assert receipt["receipt_id"] == ""


def test_receipt_is_deterministic_for_the_same_work():
    results = [_result("CONFIRMED", True, "worker-1-aaaa")]
    first = _boss_with_results(results)._build_judge_receipt()
    second = _boss_with_results(results)._build_judge_receipt()
    assert first["receipt_id"] == second["receipt_id"]


def test_record_mission_outcome_credits_verified_success_only_with_receipt():
    mind = BeingMind(being_id="being-test", _store=None)
    before_satiety = mind.satiety
    mind.record_mission_outcome(
        goal="verified goal",
        success=True,
        summary="judge confirmed the result",
        verification_receipt_id="judge-abc123",
    )
    assert mind.goals_completed == 1
    assert mind.satiety > before_satiety


def test_record_mission_outcome_does_not_credit_success_without_receipt():
    mind = BeingMind(being_id="being-test", _store=None)
    before_satiety = mind.satiety
    before_completed = mind.goals_completed
    mind.record_mission_outcome(
        goal="unverified goal",
        success=True,
        summary="boss said complete but nothing was independently verified",
        verification_receipt_id="",
    )
    # Pursued, but NOT completed: no receipt means no verified success.
    assert mind.goals_completed == before_completed
    assert mind.satiety <= before_satiety
