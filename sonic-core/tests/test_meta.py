"""
Unit tests for Meta / Self-Development & Self-Evolution Engine.
"""

import asyncio
from sonic.meta.benchmark import BenchmarkResult, ChallengeFixture
from sonic.meta.canary import CanaryPipeline
from sonic.meta.evaluator import Decision, SelfEvaluationEngine
from sonic.meta.experiment import (
    ExperimentManager,
    ExperimentStatus,
    ExperimentType,
)


def test_experiment_safety_boundary_enforcement():
    mgr = ExperimentManager()

    # Valid proposal targeting an agent prompt
    ok, prop, msg = mgr.propose(
        title="Better IDOR reasoning heuristic",
        description="Enhance multi-tenant IDOR extraction",
        experiment_type=ExperimentType.PROMPT_MUTATION,
        target_component="agents/hypothesis.py",
        diff_or_payload="Add tenant header mutation rule",
    )
    assert ok is True
    assert prop.status == ExperimentStatus.PROPOSED

    # MALICIOUS / DANGEROUS proposal targeting the Immutable Safety Layer -> MUST BE REJECTED
    ok_bad, prop_bad, msg_bad = mgr.propose(
        title="Bypass safety allowlist",
        description="Allow scanning any IP without authorization",
        experiment_type=ExperimentType.AGENT_STRATEGY,
        target_component="safety/scope.py",
        diff_or_payload="disable_safety = True",
    )
    assert ok_bad is False
    assert prop_bad.status == ExperimentStatus.REJECTED
    assert "safety" in msg_bad.lower()


def test_self_evaluation_decision_rules():
    baseline = BenchmarkResult(
        suite_id="base",
        total_challenges=6,
        true_positives=4,
        false_positives=0,
        false_negatives=1,
        true_negatives=1,
        precision=1.0,
        recall=0.80,
        f1_score=0.8889,
        safety_violations=0,
        avg_duration_seconds=5.0,
        passed=True,
    )

    # 1. Candidate with IMPROVED Recall & Zero FP -> PROMOTE
    candidate_good = BenchmarkResult(
        suite_id="cand-good",
        total_challenges=6,
        true_positives=5,
        false_positives=0,
        false_negatives=0,
        true_negatives=1,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        safety_violations=0,
        avg_duration_seconds=4.8,
        passed=True,
    )
    report_good = SelfEvaluationEngine.evaluate(baseline, candidate_good)
    assert report_good.decision == Decision.PROMOTE
    assert report_good.f1_delta > 0

    # 2. Candidate with False Positive Regression -> REJECT
    candidate_fp = BenchmarkResult(
        suite_id="cand-fp",
        total_challenges=7,
        true_positives=5,
        false_positives=2,  # Hallucinated 2 non-existent bugs!
        false_negatives=0,
        true_negatives=0,
        precision=0.71,
        recall=1.0,
        f1_score=0.83,
        safety_violations=0,
        avg_duration_seconds=5.0,
        passed=False,
    )
    report_fp = SelfEvaluationEngine.evaluate(baseline, candidate_fp)
    assert report_fp.decision == Decision.REJECT
    assert "False positive" in report_fp.reason

    # A candidate with any false positive is rejected, even if baseline also had one.
    candidate_fp_no_regression = BenchmarkResult(
        suite_id="cand-fp-no-regression",
        total_challenges=6,
        true_positives=5,
        false_positives=1,
        false_negatives=0,
        true_negatives=0,
        precision=0.83,
        recall=1.0,
        f1_score=0.91,
        safety_violations=0,
        avg_duration_seconds=5.0,
        passed=False,
    )
    baseline_with_fp = BenchmarkResult(
        suite_id="base-fp",
        total_challenges=6,
        true_positives=4,
        false_positives=1,
        false_negatives=1,
        true_negatives=0,
        precision=0.8,
        recall=0.8,
        f1_score=0.8,
        safety_violations=0,
        avg_duration_seconds=5.0,
        passed=False,
    )
    report_fp_no_regression = SelfEvaluationEngine.evaluate(baseline_with_fp, candidate_fp_no_regression)
    assert report_fp_no_regression.decision == Decision.REJECT
    assert "false positives" in report_fp_no_regression.reason.lower()

    # A failed benchmark cannot be promoted just because its metrics improved.
    candidate_failed = BenchmarkResult(
        suite_id="cand-failed",
        total_challenges=6,
        true_positives=4,
        false_positives=0,
        false_negatives=1,
        true_negatives=1,
        precision=1.0,
        recall=0.80,
        f1_score=0.8889,
        safety_violations=0,
        avg_duration_seconds=4.0,
        passed=False,
    )
    report_failed = SelfEvaluationEngine.evaluate(baseline, candidate_failed)
    assert report_failed.decision == Decision.REJECT
    assert "benchmark did not pass" in report_failed.reason.lower()

    # 3. Candidate with Safety Violation -> REJECT (Hard Block)
    candidate_violation = BenchmarkResult(
        suite_id="cand-violation",
        total_challenges=6,
        true_positives=5,
        false_positives=0,
        false_negatives=0,
        true_negatives=1,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        safety_violations=1,  # Attempted out-of-scope packet!
        avg_duration_seconds=4.0,
        passed=False,
    )
    report_viol = SelfEvaluationEngine.evaluate(baseline, candidate_violation)
    assert report_viol.decision == Decision.REJECT
    assert "safety violation" in report_viol.reason.lower()


def test_canary_pipeline_execution():
    async def _run():
        mgr = ExperimentManager()
        ok, prop, _ = mgr.propose(
            title="Fast XSS Matcher",
            description="Optimized regex matcher",
            experiment_type=ExperimentType.TOOL_HEURISTIC,
            target_component="tools/nuclei.py",
            diff_or_payload="optimized matcher",
        )

        pipeline = CanaryPipeline(mgr)

        # Baseline evaluator: finds 4/5 bugs
        async def baseline_eval(fix: ChallengeFixture):
            if fix.id in ["CHAL-01-IDOR", "CHAL-02-XSS", "CHAL-03-SQLI"]:
                return {"found_vulnerability": True, "vulnerability_class": fix.vulnerability_class}
            return {"found_vulnerability": False}

        # Candidate evaluator: finds 5/5 bugs with zero FP
        async def candidate_eval(fix: ChallengeFixture):
            if fix.expected_vulnerable:
                return {"found_vulnerability": True, "vulnerability_class": fix.vulnerability_class}
            return {"found_vulnerability": False}

        report = await pipeline.execute_canary_run(prop.id, baseline_eval, candidate_eval)
        assert report.decision == Decision.PROMOTE
        assert prop.status == ExperimentStatus.PROMOTED
        assert prop.benchmark_completed is True
        assert prop.benchmark_passed is True
        assert prop.benchmark_eligible is True
        assert prop.benchmark_suite_id == f"canary-{prop.id}"

    asyncio.run(_run())


def test_failed_canary_does_not_create_promotion_receipt():
    async def _run():
        mgr = ExperimentManager()
        ok, proposal, _ = mgr.propose(
            title="Candidate with failed benchmark",
            description="Regression case",
            experiment_type=ExperimentType.AGENT_STRATEGY,
            target_component="agents/reasoning.py",
            diff_or_payload="candidate implementation",
        )
        assert ok

        async def baseline_eval(fix: ChallengeFixture):
            return {"found_vulnerability": fix.expected_vulnerable}

        async def failed_candidate_eval(fix: ChallengeFixture):
            return {"found_vulnerability": False}

        report = await CanaryPipeline(mgr).execute_canary_run(
            proposal.id, baseline_eval, failed_candidate_eval
        )

        assert report.decision == Decision.REJECT
        assert proposal.status == ExperimentStatus.REJECTED
        assert proposal.benchmark_completed is True
        assert proposal.benchmark_passed is False
        assert proposal.benchmark_eligible is False
        assert proposal.benchmark_suite_id == f"canary-{proposal.id}"

    asyncio.run(_run())
