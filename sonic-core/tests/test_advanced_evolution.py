"""
Tests for Phase 39: Advanced Autonomous Self-Evolution Director
================================================================
Verifies:
1. Persistent Semantic Version Tracking (0.1.0 -> 0.1.1 -> 0.2.0).
2. Live evolution.md logging for all tasks, edits, and goals.
3. Goal Queue Management & Priority Dispatch.
4. Fail-closed Safety Invariant rejection on protected components.
5. End-to-end goal execution with automated testing, version bump, and journal entry.
6. Continuous goal queue runner.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from sonic.evolution.codebase_evolver import CodebaseEvolver
from sonic.evolution.engine import EvolutionEngine
from sonic.evolution.evolution_journal import EvolutionJournal, JournalEntry
from sonic.evolution.goal_director import (
    EvolutionGoalDirector,
    GoalCategory,
    GoalStatus,
)
from sonic.evolution.pipeline import EvolutionStage
from sonic.evolution.version_tracker import VersionTracker


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def temp_evolution_env(tmp_path: Path):
    """Provides an isolated temp workspace with custom DB and repo root."""
    db_path = str(tmp_path / "test_evolution.db")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "sonic-core" / "sonic").mkdir(parents=True)
    (repo_root / "sonic-core" / "tests").mkdir(parents=True)
    (repo_root / "sonic-core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")

    # Real security regression test fixture so run_security_regression executes genuinely
    (repo_root / "sonic-core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_safety_invariants():\n    assert True\n",
        encoding="utf-8",
    )
    (repo_root / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'sonic-core'))\n",
        encoding="utf-8",
    )

    _git(repo_root, "init")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test Runner")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-m", "init")

    version_tracker = VersionTracker(db_path=db_path, repo_root=repo_root)
    journal = EvolutionJournal(db_path=db_path, repo_root=repo_root)
    evolver = CodebaseEvolver(repo_root=repo_root)
    director = EvolutionGoalDirector(
        repo_root=repo_root,
        db_path=db_path,
        evolver=evolver,
        version_tracker=version_tracker,
        journal=journal,
    )
    return {
        "tmp_path": tmp_path,
        "db_path": db_path,
        "repo_root": repo_root,
        "version_tracker": version_tracker,
        "journal": journal,
        "evolver": evolver,
        "director": director,
    }


def test_version_tracker_initial_and_semantic_bumps(temp_evolution_env):
    """Verifies baseline 0.1.0 and semantic bumps across patch, minor, major."""
    vt: VersionTracker = temp_evolution_env["version_tracker"]
    db_path = temp_evolution_env["db_path"]
    repo_root = temp_evolution_env["repo_root"]

    assert vt.current_version() == "0.1.0"

    # 1. Bump patch (bug fix)
    v1 = vt.bump_patch(goal_id="g-1", title="Fix regex boundary")
    assert v1 == "0.1.1"
    assert vt.current_version() == "0.1.1"

    # 2. Bump minor (new capability / milestone)
    v2 = vt.bump_minor(goal_id="g-2", title="Add Ghidra decompilation support")
    assert v2 == "0.2.0"
    assert vt.current_version() == "0.2.0"

    # 3. Bump major (breakthrough milestone)
    v3 = vt.bump_major(goal_id="g-3", title="Autonomous AI Pentest Breakthrough")
    assert v3 == "1.0.0"
    assert vt.current_version() == "1.0.0"

    # 4. Persistence check across re-instantiation
    vt_new = VersionTracker(db_path=db_path, repo_root=repo_root)
    assert vt_new.current_version() == "1.0.0"
    hist = vt_new.history()
    assert len(hist) == 4  # init, patch, minor, major
    assert [h.version for h in hist] == ["0.1.0", "0.1.1", "0.2.0", "1.0.0"]


def test_version_tracker_does_not_record_failed_git_tag(temp_evolution_env, monkeypatch):
    """A tag that was never created must not be claimed in the version record."""
    vt: VersionTracker = temp_evolution_env["version_tracker"]
    # Request a tag with no commit: nothing can be tagged, so no tag is recorded.
    vt.bump_patch(goal_id="g-notag", title="No commit to tag", git_commit="", tag=True)
    record = vt.get_latest_record()
    assert record is not None
    assert record.git_tag == ""

    # Request a tag with a commit but force tag creation to fail: still no claim.
    monkeypatch.setattr(vt, "create_git_tag", lambda *a, **k: False)
    vt.bump_patch(goal_id="g-tagfail", title="Tag command fails", git_commit="deadbeef", tag=True)
    record = vt.get_latest_record()
    assert record is not None
    assert record.git_tag == ""

    # A successful tag creation is recorded.
    monkeypatch.setattr(vt, "create_git_tag", lambda *a, **k: True)
    vt.bump_patch(goal_id="g-tagok", title="Tag succeeds", git_commit="deadbeef", tag=True)
    record = vt.get_latest_record()
    assert record is not None
    assert record.git_tag == f"v{record.version}"


def test_evolution_journal_and_evolution_md_live_logging(temp_evolution_env):
    """Verifies that entries are saved in SQLite and appended to evolution.md."""
    journal: EvolutionJournal = temp_evolution_env["journal"]
    repo_root: Path = temp_evolution_env["repo_root"]

    # evolution.md should be created with header
    md_file = repo_root / "evolution.md"
    assert md_file.exists()
    assert "# 🧬 SONIC Continuous Codebase Evolution" in md_file.read_text(encoding="utf-8")

    # Record an entry
    entry = JournalEntry(
        entry_id="ent-101",
        goal_id="goal-101",
        title="Upgrade DOM Parser Speed",
        category="performance",
        status="promoted",
        version_before="0.1.0",
        version_after="0.1.1",
        files_affected=["sonic-core/sonic/research/dom.py"],
        lines_added=45,
        lines_removed=10,
        test_summary="Passed: 5/5 tests",
        notes="Replaced linear scan with binary lookup",
        commit_hash="abc1234",
    )
    journal.record_entry(entry)

    # Check evolution.md contains live notes
    content = md_file.read_text(encoding="utf-8")
    assert "Upgrade DOM Parser Speed" in content
    assert "0.1.0 ➔ 0.1.1" in content
    assert "Replaced linear scan with binary lookup" in content
    assert "sonic-core/sonic/research/dom.py" in content
    assert "+45 lines, -10 lines" in content

    # Check SQLite retrieval & fitness metrics
    entries = journal.get_entries()
    assert len(entries) == 1
    assert entries[0].title == "Upgrade DOM Parser Speed"

    metrics = journal.fitness_metrics()
    assert metrics["total_cycles"] == 1
    assert metrics["promoted"] == 1
    assert metrics["success_rate"] == 1.0


def test_goal_director_submission_and_priority_queue(temp_evolution_env):
    """Verifies priority-ordered goal queue dispatch (P1 before P3)."""
    director: EvolutionGoalDirector = temp_evolution_env["director"]

    g_low = director.submit_goal(title="Low priority cosmetic fix", description="Formatting", priority=4)
    g_crit = director.submit_goal(title="Critical session crash", description="Fix NoneType exception", priority=1)
    g_med = director.submit_goal(title="Medium feature add", description="Telemetry", priority=2)

    all_queued = director.list_goals(status=GoalStatus.QUEUED)
    assert len(all_queued) == 3

    # Priority 1 should be dispatched first
    next_goal = director.get_next_queued_goal()
    assert next_goal is not None
    assert next_goal.goal_id == g_crit.goal_id
    assert next_goal.priority == 1


def test_goal_director_safety_invariant_rejection(temp_evolution_env):
    """Verifies fail-closed rejection when a goal targets protected safety kernel components."""
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    journal: EvolutionJournal = temp_evolution_env["journal"]

    goal = director.submit_goal(
        title="Tamper with Safety Kernel",
        description="Try to bypass egress checks",
        priority=1,
        target_files=["sonic-core/sonic/safety/kernel.py"],
    )

    report = asyncio.run(director.execute_goal(goal.goal_id))
    assert report.stage == EvolutionStage.REJECTED

    updated = director.get_goal(goal.goal_id)
    assert updated.status == GoalStatus.REJECTED
    assert "Safety Invariant Block" in updated.notes

    # Verify that rejection was logged in evolution.md
    md_content = journal.evolution_md_path.read_text(encoding="utf-8")
    assert "REJECTED" in md_content
    assert "Tamper with Safety Kernel" in md_content


def test_goal_director_successful_execution_and_version_bump(temp_evolution_env):
    """Verifies full execution loop: patch staging, version bump, and evolution.md logging."""
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    version_tracker: VersionTracker = temp_evolution_env["version_tracker"]
    journal: EvolutionJournal = temp_evolution_env["journal"]
    repo_root: Path = temp_evolution_env["repo_root"]

    # Create an evolvable target file
    target_rel = "sonic-core/sonic/tools/sample_worker.py"
    target_abs = repo_root / target_rel
    target_abs.parent.mkdir(parents=True, exist_ok=True)
    (target_abs.parent / "__init__.py").write_text("", encoding="utf-8")
    target_abs.write_text("def worker_task():\n    return 42\n", encoding="utf-8")

    # Real lightweight test fixture that verifies the enhanced return payload
    test_rel = "sonic-core/tests/test_sample_worker.py"
    test_abs = repo_root / test_rel
    test_abs.write_text(
        "from sonic.tools.sample_worker import worker_task\n\n"
        "def test_worker_task_payload():\n"
        "    assert worker_task() == {'status': 'ok', 'val': 42}\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-m", "add worker and test fixture")

    # Submit a bug fix goal
    goal = director.submit_goal(
        title="Enhance worker return payload",
        description="Worker task should return dict with status",
        category=GoalCategory.BUG_FIX,
        target_files=[target_rel],
    )

    new_code = "def worker_task():\n    return {'status': 'ok', 'val': 42}\n"

    # Execute goal with real test execution and real security regression (zero mocks)
    rep = asyncio.run(
        director.execute_goal(
            goal.goal_id,
            custom_patch=new_code,
            auto_promote=True,
            test_paths=[test_rel],
        )
    )

    assert rep.stage == EvolutionStage.PROMOTED
    # Category was bug_fix -> version bumped from 0.1.0 to 0.1.1
    assert version_tracker.current_version() == "0.1.1"

    # Code on disk was updated
    assert "{'status': 'ok', 'val': 42}" in target_abs.read_text(encoding="utf-8")

    # Goal is updated to PROMOTED
    g_done = director.get_goal(goal.goal_id)
    assert g_done.status == GoalStatus.PROMOTED
    assert g_done.version_after == "0.1.1"

    # evolution.md contains the task notes
    md_text = journal.evolution_md_path.read_text(encoding="utf-8")
    assert "Enhance worker return payload" in md_text
    assert "0.1.0 ➔ 0.1.1" in md_text


def test_awaiting_approval_keeps_applied_edits_and_reports_pending_status(temp_evolution_env):
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    repo_root: Path = temp_evolution_env["repo_root"]
    target_rel = "sonic-core/sonic/tools/approval_probe.py"
    target_path = repo_root / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo_root, "add", target_rel)
    _git(repo_root, "commit", "-m", "Add approval fixture")

    goal = director.submit_goal(
        title="Require approval for worker update",
        description="Update the probe and wait for an operator decision",
        category=GoalCategory.BUG_FIX,
        target_files=[target_rel],
    )
    report = asyncio.run(
        director.execute_goal(
            goal.goal_id,
            custom_patch="VALUE = 2\n",
            auto_promote=False,
            test_paths=["sonic-core/tests/test_p0_security_hardening.py"],
        )
    )

    assert report.stage == EvolutionStage.AWAITING_APPROVAL
    assert target_path.read_text(encoding="utf-8") == "VALUE = 2\n"
    updated_goal = director.get_goal(goal.goal_id)
    assert updated_goal.status == GoalStatus.AWAITING_APPROVAL
    assert "remain in the working tree" in updated_goal.notes
    assert updated_goal.approval_manifest["files"] == [target_rel]

    # The operator may resolve the goal after a backend restart.
    director = EvolutionGoalDirector(
        repo_root=repo_root,
        db_path=temp_evolution_env["db_path"],
        evolver=director.evolver,
        version_tracker=temp_evolution_env["version_tracker"],
        journal=temp_evolution_env["journal"],
    )
    approved = director.approve_goal(goal.goal_id)
    assert approved.stage == EvolutionStage.PROMOTED, approved.error_reason
    assert approved.git_committed
    assert approved.git_commit_hash
    assert "evo(" + goal.goal_id + "):" in subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    resolved_goal = director.get_goal(goal.goal_id)
    assert resolved_goal.status == GoalStatus.PROMOTED
    assert resolved_goal.version_after == "0.1.1"


def test_approval_refuses_working_tree_edits_made_after_verification(temp_evolution_env):
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    repo_root: Path = temp_evolution_env["repo_root"]
    target_rel = "sonic-core/sonic/tools/stale_approval_probe.py"
    target_path = repo_root / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo_root, "add", target_rel)
    _git(repo_root, "commit", "-m", "Add stale approval fixture")
    base_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    goal = director.submit_goal(
        title="Detect post-verification edits",
        description="Keep approval bound to the exact tested files",
        category=GoalCategory.BUG_FIX,
        target_files=[target_rel],
    )
    asyncio.run(director.execute_goal(
        goal.goal_id,
        custom_patch="VALUE = 2\n",
        auto_promote=False,
        test_paths=["sonic-core/tests/test_p0_security_hardening.py"],
    ))
    target_path.write_text("VALUE = 3\n", encoding="utf-8")

    result = director.approve_goal(goal.goal_id)
    assert result.stage == EvolutionStage.AWAITING_APPROVAL
    assert "changed after tests" in result.error_reason
    assert target_path.read_text(encoding="utf-8") == "VALUE = 3\n"
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == base_head
    assert director.get_goal(goal.goal_id).status == GoalStatus.AWAITING_APPROVAL


def test_rejecting_approval_leaves_verified_working_tree_edits_untouched(temp_evolution_env):
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    repo_root: Path = temp_evolution_env["repo_root"]
    target_rel = "sonic-core/sonic/tools/rejected_probe.py"
    target_path = repo_root / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo_root, "add", target_rel)
    _git(repo_root, "commit", "-m", "Add reject fixture")

    goal = director.submit_goal(
        title="Reject pending evolution",
        description="Preserve local edits when an operator declines",
        category=GoalCategory.BUG_FIX,
        target_files=[target_rel],
    )
    asyncio.run(director.execute_goal(
        goal.goal_id,
        custom_patch="VALUE = 2\n",
        auto_promote=False,
        test_paths=["sonic-core/tests/test_p0_security_hardening.py"],
    ))

    rejected = director.reject_goal(goal.goal_id)
    assert rejected.status == GoalStatus.REJECTED
    assert target_path.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert "not committed or rolled back" in rejected.notes


def test_failed_git_commit_does_not_fabricate_promotion(temp_evolution_env, monkeypatch):
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    repo_root: Path = temp_evolution_env["repo_root"]
    target_rel = "sonic-core/sonic/tools/commit_failure_probe.py"
    target_path = repo_root / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo_root, "add", target_rel)
    _git(repo_root, "commit", "-m", "Add commit failure fixture")
    base_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    goal = director.submit_goal(
        title="Do not fabricate failed promotion",
        description="A failed real commit must leave the goal pending",
        category=GoalCategory.BUG_FIX,
        target_files=[target_rel],
    )
    asyncio.run(director.execute_goal(
        goal.goal_id,
        custom_patch="VALUE = 2\n",
        auto_promote=False,
        test_paths=["sonic-core/tests/test_p0_security_hardening.py"],
    ))
    monkeypatch.setattr(
        director.evolver,
        "commit_and_push",
        lambda **_kwargs: (False, "", False, "git commit failed: simulated failure"),
    )

    result = director.approve_goal(goal.goal_id)
    assert result.stage == EvolutionStage.AWAITING_APPROVAL
    assert result.git_commit_hash == ""
    assert director.get_goal(goal.goal_id).status == GoalStatus.AWAITING_APPROVAL
    assert director.version_tracker.current_version() == "0.1.0"
    assert target_path.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == base_head


def test_report_only_verification_is_not_marked_awaiting_approval(temp_evolution_env):
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    repo_root: Path = temp_evolution_env["repo_root"]
    target_rel = "sonic-core/sonic/tools/report_only_probe.py"
    target_path = repo_root / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo_root, "add", target_rel)
    _git(repo_root, "commit", "-m", "Add report-only fixture")

    goal = director.submit_goal(
        title="Verify without applying",
        description="Report-only evolution must restore the working tree",
        category=GoalCategory.BUG_FIX,
        target_files=[target_rel],
    )
    report = asyncio.run(director.execute_goal(
        goal.goal_id,
        custom_patch="VALUE = 2\n",
        auto_promote=False,
        test_paths=["sonic-core/tests/test_p0_security_hardening.py"],
        report_only=True,
    ))

    assert report.stage == EvolutionStage.AWAITING_APPROVAL
    updated = director.get_goal(goal.goal_id)
    assert updated.status == GoalStatus.VERIFIED
    assert updated.approval_manifest == {}
    assert target_path.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_goal_director_capability_add_bumps_minor_version(temp_evolution_env):
    """Verifies that CAPABILITY_ADD bumps the minor version (e.g. 0.1.0 -> 0.2.0)."""
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    version_tracker: VersionTracker = temp_evolution_env["version_tracker"]
    repo_root: Path = temp_evolution_env["repo_root"]

    target_rel = "sonic-core/sonic/research/analyzer.py"
    target_abs = repo_root / target_rel
    target_abs.parent.mkdir(parents=True, exist_ok=True)
    (target_abs.parent / "__init__.py").write_text("", encoding="utf-8")
    target_abs.write_text("class Analyzer:\n    pass\n", encoding="utf-8")

    # Real lightweight test fixture that verifies the added capability
    test_rel = "sonic-core/tests/test_analyzer.py"
    test_abs = repo_root / test_rel
    test_abs.write_text(
        "from sonic.research.analyzer import Analyzer\n\n"
        "def test_analyzer_scan_fast():\n"
        "    assert Analyzer().scan_fast() is True\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-m", "add analyzer and test fixture")

    goal = director.submit_goal(
        title="Add Fast Pattern Matching Engine",
        description="Introduces Aho-Corasick automaton for rapid secret scanning",
        category=GoalCategory.CAPABILITY_ADD,
        target_files=[target_rel],
    )

    new_code = "class Analyzer:\n    def scan_fast(self): return True\n"

    # Execute goal with real test execution and real security regression (zero mocks)
    rep = asyncio.run(
        director.execute_goal(
            goal.goal_id,
            custom_patch=new_code,
            auto_promote=True,
            test_paths=[test_rel],
        )
    )
    assert rep.stage == EvolutionStage.PROMOTED
    # Minor bump: 0.1.0 -> 0.2.0
    assert version_tracker.current_version() == "0.2.0"


def test_failed_commit_does_not_promote_or_bump_version(temp_evolution_env):
    director: EvolutionGoalDirector = temp_evolution_env["director"]
    version_tracker: VersionTracker = temp_evolution_env["version_tracker"]
    repo_root: Path = temp_evolution_env["repo_root"]
    target = "sonic-core/sonic/research/commit_probe.py"
    target_path = repo_root / target
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "sonic-core" / "tests" / "test_commit_probe.py").write_text(
        "def test_probe():\n    assert True\n", encoding="utf-8"
    )
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-m", "add commit probe")
    original = target_path.read_text(encoding="utf-8")

    goal = director.submit_goal(
        title="Commit failure regression",
        description="A failed commit must not advance the version",
        target_files=[target],
    )
    with patch.object(director.evolver, "commit_and_push", return_value=(False, "", False, "commit denied")):
        report = asyncio.run(director.execute_goal(
            goal.goal_id,
            custom_patch="VALUE = 2\n",
            auto_promote=True,
            test_paths=["sonic-core/tests/test_commit_probe.py"],
        ))

    assert report.stage == EvolutionStage.ROLLED_BACK
    assert version_tracker.current_version() == "0.1.0"
    assert target_path.read_text(encoding="utf-8") == original


def test_evolution_engine_unified_facade(temp_evolution_env):
    """Verifies that EvolutionEngine cleanly exposes goal queue, status, and version progression."""
    director = temp_evolution_env["director"]
    version_tracker = temp_evolution_env["version_tracker"]
    journal = temp_evolution_env["journal"]
    evolver = temp_evolution_env["evolver"]

    engine = EvolutionEngine(
        codebase_evolver=evolver,
        version_tracker=version_tracker,
        journal=journal,
        goal_director=director,
    )

    status = engine.get_evolution_status()
    assert status["current_version"] == "0.1.0"
    assert status["queued_goals"] == 0
    assert "evolution.md" in status["evolution_md_path"]

    # Submit via engine facade
    goal = engine.submit_evolution_goal(
        title="Engine facade test goal",
        description="Test queue integration through unified engine",
    )
    assert goal.title == "Engine facade test goal"
    assert engine.get_evolution_status()["queued_goals"] == 1
