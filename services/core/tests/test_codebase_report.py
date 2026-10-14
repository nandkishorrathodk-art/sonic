"""
Tests for codebase report narration truthfulness and pluralization.
"""

from __future__ import annotations

from dataclasses import dataclass
import pytest

from sonic.evolution.codebase_auditor import AuditFinding, AuditReport
from sonic.evolution.codebase_report import _count_phrase, narrate


@dataclass
class DummyCycle:
    signal: str
    stage: str = "skipped"
    tree_action: str = "rolled_back"
    proposed_diff: str = ""
    verified: bool | None = None


def test_count_phrase_pluralization():
    # Pluralization bug fix: "file breaks outright" -> "2 file breaks outright", not "outrights"
    assert _count_phrase(1, "file breaks outright") == "1 file breaks outright"
    assert _count_phrase(2, "file breaks outright") == "2 file breaks outright"
    assert _count_phrase(0, "file breaks outright") == "0 file breaks outright"

    # Words ending with sh, ch, x, z -> es
    assert _count_phrase(1, "crash") == "1 crash"
    assert _count_phrase(2, "crash") == "2 crashes"

    # Standard pluralization
    assert _count_phrase(1, "high-risk defect") == "1 high-risk defect"
    assert _count_phrase(2, "high-risk defect") == "2 high-risk defects"
    assert _count_phrase(1, "latent problem") == "1 latent problem"
    assert _count_phrase(2, "latent problem") == "2 latent problems"
    assert _count_phrase(1, "failing test") == "1 failing test"
    assert _count_phrase(2, "failing test") == "2 failing tests"
    assert _count_phrase(1, "verified change") == "1 verified change"
    assert _count_phrase(2, "verified change") == "2 verified changes"


def test_narrate_pluralization_in_report_intro():
    report = AuditReport(
        findings=[
            AuditFinding(kind="syntax_error", severity="critical", file="a.py", line=1, title="err", detail="bad"),
            AuditFinding(kind="syntax_error", severity="critical", file="b.py", line=2, title="err", detail="bad"),
        ]
    )
    text = narrate(report)
    assert "2 file breaks outright." in text
    assert "2 file breaks outrights" not in text


def test_narrate_failed_cycle_not_labeled_verified():
    report = AuditReport()
    cycle = DummyCycle(
        signal="TODO: fix memory leak",
        stage="rolled_back",
        tree_action="rolled_back",
        proposed_diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-1\n+2\n",
    )
    text = narrate(report, cycles=[cycle])
    assert "TODO: fix memory leak -> failed verification (rolled_back); tree: rolled_back." in text
    assert "verified patch" not in text
    # Must NOT claim test suite verification when verification failed
    assert "I verified the fix against the test suite" not in text
    assert "Nothing was committed; the working tree is untouched." in text


def test_narrate_rejected_cycle_labeled_rejected_patch():
    report = AuditReport()
    cycle = DummyCycle(
        signal="FIXME: unsafe eval",
        stage="rejected",
        tree_action="rolled_back",
        proposed_diff="--- a/x.py\n+++ b/x.py\n",
    )
    text = narrate(report, cycles=[cycle])
    assert "FIXME: unsafe eval -> rejected patch (rejected); tree: rolled_back." in text
    assert "verified patch" not in text
    assert "I verified the fix against the test suite" not in text
    assert "Nothing was committed; the working tree is untouched." in text


def test_narrate_verified_awaiting_approval_cycle():
    report = AuditReport()
    cycle = DummyCycle(
        signal="HACK: cleanup",
        stage="awaiting_approval",
        tree_action="rolled_back",
        proposed_diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-1\n+2\n",
    )
    text = narrate(report, cycles=[cycle])
    assert "HACK: cleanup -> verified patch (awaiting_approval); tree: rolled_back." in text
    assert "I verified the fix against the test suite, then I put the tree back exactly as it was." in text
    assert "Nothing was committed; the working tree is untouched." in text


def test_narrate_promoted_cycle():
    report = AuditReport()
    cycle = DummyCycle(
        signal="HACK: cleanup",
        stage="promoted",
        tree_action="committed",
        proposed_diff="--- a/x.py\n+++ b/x.py\n",
    )
    text = narrate(report, cycles=[cycle])
    assert "HACK: cleanup -> verified patch (promoted); tree: committed." in text
    assert "I applied and committed 1 verified change locally" in text
    assert "Nothing was committed" not in text


def test_narrate_mixed_cycles_truthfulness():
    report = AuditReport()
    c1 = DummyCycle(
        signal="signal_fail",
        stage="rolled_back",
        tree_action="rolled_back",
        proposed_diff="diff1",
    )
    c2 = DummyCycle(
        signal="signal_reject",
        stage="rejected",
        tree_action="rolled_back",
        proposed_diff="diff2",
    )
    text = narrate(report, cycles=[c1, c2])
    assert "signal_fail -> failed verification (rolled_back); tree: rolled_back." in text
    assert "signal_reject -> rejected patch (rejected); tree: rolled_back." in text
    assert "verified patch" not in text
    assert "I verified the fix against the test suite" not in text
