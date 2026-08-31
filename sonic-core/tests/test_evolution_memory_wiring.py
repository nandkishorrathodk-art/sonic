"""
Phase 8 — evolution memory production wiring done-gate tests.

Closes the audit gap that ``EvolutionMemoryStore`` was persistent (SQLite) but
never written in production: ``PromotionEngine.evaluate_gates`` now accepts an
optional ``memory`` and records each promotion/rejection outcome, deduplicating
via ``EvolutionMemoryItem.compute_fingerprint``.

These tests prove:
  1. ``compute_fingerprint`` is deterministic and stable across runs.
  2. A rejection with ``memory`` wired persists an experiment that survives a
     fresh store opened on the same DB (the being remembers it tried this).
  3. A promotion with ``memory`` wired persists the promoted experiment.
  4. ``is_duplicate`` prevents re-recording the identical experiment.
  5. With ``memory=None`` (default) nothing is recorded — backward compatible.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sonic.evolution.comparator import ComparisonReport
from sonic.evolution.memory import EvolutionMemoryStore
from sonic.evolution.models import (
    CandidateMetrics,
    EvolutionCandidate,
    EvolutionMemoryItem,
    EvolutionPolicy,
    EvolutionState,
)
from sonic.evolution.promotion import PromotionEngine


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch) -> Path:
    """Point the evolution memory DB at a throwaway file for this test."""
    db_path = tmp_path / "evo_mem_test.db"
    monkeypatch.setenv("SONIC_MEMORY_DB_PATH", str(db_path))
    return db_path


def _baseline() -> CandidateMetrics:
    return CandidateMetrics(f1_score=0.80, false_positives=1, safety_violations=0)


def _candidate() -> EvolutionCandidate:
    return EvolutionCandidate(
        hypothesis_id="hyp-1",
        parent_version="v1.0.0",
        candidate_version="v1.1.0-cand",
        rationale="improve XSS detection recall",
    )


def test_compute_fingerprint_is_deterministic_and_stable():
    """Same logical experiment always yields the same fingerprint."""
    base = _baseline()
    cand = CandidateMetrics(f1_score=0.85, false_positives=1, safety_violations=0)
    item_a = EvolutionMemoryItem(
        problem="xss detection", hypothesis_id="hyp-1", candidate_id="c1",
        candidate_version="v1.1.0-cand", parent_version="v1.0.0",
        baseline_metrics=base, candidate_metrics=cand, decision="promoted",
    )
    item_b = EvolutionMemoryItem(
        problem="  XSS Detection  ",  # case + whitespace must not change identity
        hypothesis_id="hyp-1", candidate_id="c1",
        candidate_version="v1.1.0-cand", parent_version="v1.0.0",
        baseline_metrics=base, candidate_metrics=cand, decision="promoted",
        decision_reason="different human text",  # excluded from identity
        lesson="some lesson",  # excluded from identity
    )
    assert item_a.compute_fingerprint() == item_b.compute_fingerprint()
    # A different decision changes the identity.
    item_c = item_a.model_copy(update={"decision": "rejected"})
    assert item_c.compute_fingerprint() != item_a.compute_fingerprint()


def test_rejection_is_recorded_and_survives_restart(isolated_db):
    comparison = ComparisonReport(
        baseline=_baseline(),
        candidate=CandidateMetrics(f1_score=0.70, false_positives=1, safety_violations=0),
        has_safety_violation=False,
    )
    memory = EvolutionMemoryStore(persist=True)
    promoted, reason, state = PromotionEngine.evaluate_gates(
        _candidate(), comparison, policy=EvolutionPolicy(),
        memory=memory, problem="xss detection",
    )
    assert promoted is False
    assert state is EvolutionState.REJECTED
    assert len(memory.get_all_history()) == 1

    # A fresh store loaded from the SAME db remembers the experiment.
    reopened = EvolutionMemoryStore(persist=True)
    history = reopened.get_all_history()
    assert len(history) == 1
    assert history[0].decision == "rejected"
    assert history[0].problem == "xss detection"


def test_promotion_is_recorded(isolated_db):
    comparison = ComparisonReport(
        baseline=_baseline(),
        candidate=CandidateMetrics(f1_score=0.90, false_positives=1, safety_violations=0),
        has_safety_violation=False,
    )
    memory = EvolutionMemoryStore(persist=True)
    promoted, _reason, state = PromotionEngine.evaluate_gates(
        _candidate(), comparison, policy=EvolutionPolicy(),
        memory=memory, problem="xss detection",
    )
    assert promoted is True
    assert state is EvolutionState.CANARY
    assert len(memory.get_all_history()) == 1
    assert memory.get_all_history()[0].decision == "promoted"


def test_duplicate_experiment_is_not_re_recorded(isolated_db):
    comparison = ComparisonReport(
        baseline=_baseline(),
        candidate=CandidateMetrics(f1_score=0.70, false_positives=1, safety_violations=0),
        has_safety_violation=False,
    )
    memory = EvolutionMemoryStore(persist=True)
    # First evaluation records the experiment.
    PromotionEngine.evaluate_gates(
        _candidate(), comparison, memory=memory, problem="xss detection",
    )
    assert len(memory.get_all_history()) == 1
    # Re-evaluating the identical experiment must not append a duplicate.
    candidate2 = _candidate()  # fresh object, same identity fields
    PromotionEngine.evaluate_gates(
        candidate2, comparison, memory=memory, problem="xss detection",
    )
    assert len(memory.get_all_history()) == 1


def test_no_memory_means_no_recording(isolated_db):
    """Default (memory=None) preserves the old stateless contract."""
    comparison = ComparisonReport(
        baseline=_baseline(),
        candidate=CandidateMetrics(f1_score=0.70, false_positives=1, safety_violations=0),
        has_safety_violation=False,
    )
    promoted, _reason, state = PromotionEngine.evaluate_gates(
        _candidate(), comparison, policy=EvolutionPolicy(),
    )
    assert promoted is False
    assert state is EvolutionState.REJECTED
    # Nothing was recorded (store has no items despite a DB existing).
    memory = EvolutionMemoryStore(persist=True)
    assert len(memory.get_all_history()) == 0
