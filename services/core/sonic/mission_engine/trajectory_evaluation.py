"""Offline baseline metrics for privacy-reviewed SONIC decision examples.

This evaluator is deliberately pure: it performs no persistence, model calls,
action dispatch, or promotion. Empty input and missing measurements remain
explicitly unavailable rather than being reported as zero or synthetic scores.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

from sonic.llm.trajectory import (
    ActionOutcome,
    MissionOutcome,
    SafetyVerdict,
    TrainingExample,
    VerificationStatus,
)


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int
    value: float | None
    confidence_95: tuple[float, float] | None


@dataclass(frozen=True)
class NumericSummary:
    count: int
    mean: float | None


@dataclass(frozen=True)
class TrajectoryBaseline:
    example_count: int
    episode_count: int
    action_outcomes: dict[str, int]
    mission_outcomes: dict[str, int]
    action_success_rate: Rate
    verified_mission_success_rate: Rate
    independent_verification_coverage: Rate
    correction_rate: Rate
    policy_block_rate: Rate
    prediction_error: NumericSummary
    model_latency_ms: NumericSummary
    model_cost_usd: NumericSummary


def _rate(numerator: int, denominator: int) -> Rate:
    interval = None
    if denominator:
        z = 1.959963984540054
        proportion = numerator / denominator
        z_squared = z * z
        denominator_adjusted = 1 + z_squared / denominator
        center = (proportion + z_squared / (2 * denominator)) / denominator_adjusted
        margin = (
            z
            * (
                proportion * (1 - proportion) / denominator
                + z_squared / (4 * denominator * denominator)
            )
            ** 0.5
            / denominator_adjusted
        )
        interval = (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))
    return Rate(
        numerator=numerator,
        denominator=denominator,
        value=round(numerator / denominator, 4) if denominator else None,
        confidence_95=interval,
    )


def evaluate_training_examples(
    examples: Sequence[TrainingExample],
) -> TrajectoryBaseline:
    """Summarize validated records without implying causal model improvement.

    Mission-level rates use unique episodes, not decision rows. Every episode
    must report one consistent final mission outcome; otherwise the input is
    not a sound baseline and is rejected.
    """
    if not examples:
        empty_rate = _rate(0, 0)
        unavailable = NumericSummary(count=0, mean=None)
        return TrajectoryBaseline(
            example_count=0,
            episode_count=0,
            action_outcomes={},
            mission_outcomes={},
            action_success_rate=empty_rate,
            verified_mission_success_rate=empty_rate,
            independent_verification_coverage=empty_rate,
            correction_rate=empty_rate,
            policy_block_rate=empty_rate,
            prediction_error=unavailable,
            model_latency_ms=unavailable,
            model_cost_usd=unavailable,
        )

    outcomes_by_episode: dict[str, set[MissionOutcome]] = {}
    tenants_by_episode: dict[str, set[str]] = {}
    verified_episodes: set[str] = set()
    independently_checked_episodes: set[str] = set()
    for example in examples:
        outcomes_by_episode.setdefault(example.episode_id, set()).add(example.mission_outcome)
        tenants_by_episode.setdefault(example.episode_id, set()).add(example.tenant_ref)
        verification = example.verification
        if (
            verification.independent
            and verification.verifier_id
            and verification.status != VerificationStatus.NOT_RUN
            and verification.evidence_ref is not None
        ):
            independently_checked_episodes.add(example.episode_id)
            if verification.status == VerificationStatus.PASSED:
                verified_episodes.add(example.episode_id)

    inconsistent = [
        episode_id
        for episode_id, outcomes in outcomes_by_episode.items()
        if len(outcomes) != 1
    ]
    if inconsistent:
        raise ValueError(
            "inconsistent mission_outcome within episodes: "
            + ", ".join(sorted(inconsistent))
        )
    cross_tenant_episodes = [
        episode_id
        for episode_id, tenant_refs in tenants_by_episode.items()
        if len(tenant_refs) != 1
    ]
    if cross_tenant_episodes:
        raise ValueError(
            "episode_id must belong to exactly one tenant: "
            + ", ".join(sorted(cross_tenant_episodes))
        )

    episode_outcomes = {
        episode_id: next(iter(outcomes))
        for episode_id, outcomes in outcomes_by_episode.items()
    }
    action_outcomes = Counter(example.receipt.outcome.value for example in examples)
    mission_outcomes = Counter(outcome.value for outcome in episode_outcomes.values())
    verified_successes = sum(
        outcome == MissionOutcome.SUCCESS and episode_id in verified_episodes
        for episode_id, outcome in episode_outcomes.items()
    )
    independent_verifications = len(independently_checked_episodes)
    prediction_errors = [
        example.receipt.prediction_error
        for example in examples
        if example.receipt.prediction_error is not None
    ]
    latencies = [
        example.model.latency_ms
        for example in examples
        if example.model.latency_ms is not None
    ]
    costs = [
        example.model.cost_usd
        for example in examples
        if example.model.cost_usd is not None
    ]

    def _summary(values: list[float]) -> NumericSummary:
        return NumericSummary(
            count=len(values),
            mean=round(fmean(values), 4) if values else None,
        )

    total = len(examples)
    episodes = len(episode_outcomes)
    return TrajectoryBaseline(
        example_count=total,
        episode_count=episodes,
        action_outcomes=dict(sorted(action_outcomes.items())),
        mission_outcomes=dict(sorted(mission_outcomes.items())),
        action_success_rate=_rate(action_outcomes[ActionOutcome.SUCCEEDED.value], total),
        verified_mission_success_rate=_rate(verified_successes, episodes),
        independent_verification_coverage=_rate(independent_verifications, episodes),
        correction_rate=_rate(
            sum(bool(example.corrections) for example in examples),
            total,
        ),
        policy_block_rate=_rate(
            sum(
                example.receipt.safety_verdict
                in {SafetyVerdict.BLOCK, SafetyVerdict.REQUIRE_APPROVAL}
                for example in examples
            ),
            total,
        ),
        prediction_error=_summary(prediction_errors),
        model_latency_ms=_summary(latencies),
        model_cost_usd=_summary(costs),
    )
