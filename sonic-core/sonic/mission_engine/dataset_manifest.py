"""In-memory, tenant-isolated freeze manifest for reviewed trajectory examples.

This module does not collect, persist, train on, or execute examples. It accepts
already-validated examples only and refuses to freeze below the agreed sample
floor or without complete redaction-review attestations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from sonic.llm.trajectory import TrainingExample
from sonic.mission_engine.privacy_gate import (
    PrivacyReviewVerifier,
    canonical_review_payload,
)
from sonic.mission_engine.trajectory_evaluation import evaluate_training_examples

MIN_EPISODES = 300
MIN_TENANTS = 3
MIN_EPISODES_PER_SPLIT = {
    "train": 150,
    "validation": 75,
    "holdout": 75,
}


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


@dataclass(frozen=True)
class EpisodeAssignment:
    episode_id: str
    split: DatasetSplit


@dataclass(frozen=True)
class FrozenDatasetManifest:
    schema_version: str
    seed: str
    example_count: int
    episode_count: int
    tenant_count: int
    episode_counts: dict[str, int]
    tenant_counts: dict[str, int]
    dataset_sha256: str
    split_sha256: str
    assignments: tuple[EpisodeAssignment, ...]


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def freeze_dataset(
    examples: Sequence[TrainingExample],
    *,
    seed: str,
    privacy_review_verifier: PrivacyReviewVerifier,
) -> FrozenDatasetManifest:
    """Validate and deterministically split reviewed examples by tenant.

    All episodes for one HMAC-pseudonymized tenant stay in exactly one split.
    Sample floors count unique episodes, not action rows. The manifest and its
    digests are returned in memory only; callers must explicitly persist any
    approved artifacts in a future collection implementation.
    """
    if not seed or not seed.strip():
        raise ValueError("a non-empty split seed is required")
    if not examples:
        raise ValueError("cannot freeze an empty dataset")

    decisions: dict[str, TrainingExample] = {}
    episodes: dict[str, list[TrainingExample]] = {}
    tenants: dict[str, set[str]] = {}
    for example in examples:
        review = example.privacy_review
        if not privacy_review_verifier.verify(
            canonical_review_payload(review),
            review.signature_sha256,
        ):
            raise ValueError(f"invalid privacy review signature for decision {example.decision_id}")
        if example.decision_id in decisions:
            raise ValueError(f"duplicate decision_id: {example.decision_id}")
        decisions[example.decision_id] = example
        episodes.setdefault(example.episode_id, []).append(example)
        tenants.setdefault(example.tenant_ref, set()).add(example.episode_id)

    episode_count = len(episodes)
    if episode_count < MIN_EPISODES:
        raise ValueError(
            f"dataset requires at least {MIN_EPISODES} unique episodes; received {episode_count}"
        )
    if len(tenants) < MIN_TENANTS:
        raise ValueError(f"dataset requires at least {MIN_TENANTS} distinct tenant groups")

    evaluate_training_examples(examples)

    # Assign whole tenant groups, largest first. Hash ordering makes ties stable
    # across runs while the target-ratio greedy assignment balances split sizes.
    total = episode_count
    targets = {
        split: total * share
        for split, share in (
            (DatasetSplit.TRAIN, 0.5),
            (DatasetSplit.VALIDATION, 0.25),
            (DatasetSplit.HOLDOUT, 0.25),
        )
    }
    tenant_groups = sorted(
        tenants.items(),
        key=lambda item: (
            -len(item[1]),
            _stable_digest(f"{seed}:{item[0]}"),
        ),
    )
    assigned: dict[DatasetSplit, list[str]] = {split: [] for split in DatasetSplit}
    episode_counts = {split: 0 for split in DatasetSplit}

    for tenant_ref, tenant_episodes in tenant_groups:
        split = min(
            DatasetSplit,
            key=lambda candidate: (
                episode_counts[candidate] / targets[candidate],
                candidate.value,
            ),
        )
        assigned[split].extend(tenant_episodes)
        episode_counts[split] += len(tenant_episodes)

    for split, minimum in MIN_EPISODES_PER_SPLIT.items():
        if episode_counts[DatasetSplit(split)] < minimum:
            raise ValueError(
                f"tenant-grouped {split} split has {episode_counts[DatasetSplit(split)]} episodes; "
                f"requires at least {minimum}"
            )

    split_by_episode = {
        episode_id: split
        for split, episode_ids in assigned.items()
        for episode_id in episode_ids
    }
    tenant_splits: dict[str, DatasetSplit] = {}
    for tenant_ref, tenant_episodes in tenants.items():
        observed_splits = {split_by_episode[episode_id] for episode_id in tenant_episodes}
        if len(observed_splits) != 1:
            raise ValueError(f"tenant leakage across splits for {tenant_ref}")
        tenant_splits[tenant_ref] = next(iter(observed_splits))
    if len(tenant_splits) != len(tenants):
        raise ValueError("failed to assign every tenant to exactly one split")

    assignments = tuple(
        sorted(
            (
                EpisodeAssignment(episode_id=episode_id, split=split)
                for split, episode_ids in assigned.items()
                for episode_id in episode_ids
            ),
            key=lambda item: item.episode_id,
        )
    )
    tenant_counts = {
        split.value: sum(tenant_split == split for tenant_split in tenant_splits.values())
        for split in DatasetSplit
    }

    canonical_examples = [
        decisions[decision_id].model_dump(mode="json")
        for decision_id in sorted(decisions)
    ]
    dataset_sha256 = _stable_digest(
        json.dumps(canonical_examples, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )
    split_payload = {
        "seed": seed,
        "assignments": [(item.episode_id, item.split.value) for item in assignments],
    }
    split_sha256 = _stable_digest(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )

    return FrozenDatasetManifest(
        schema_version="sonic.dataset-manifest/v1",
        seed=seed,
        example_count=len(examples),
        episode_count=episode_count,
        tenant_count=len(tenants),
        episode_counts={split.value: episode_counts[split] for split in DatasetSplit},
        tenant_counts=tenant_counts,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        assignments=assignments,
    )
