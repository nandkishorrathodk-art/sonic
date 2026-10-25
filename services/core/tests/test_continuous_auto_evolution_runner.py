"""Tests for Continuous Autonomous Self-Evolution Engine Runner.

Belongs to Department: cognitive_systems, Team: identity_evolution, Subsystem: evolution.
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sonic.evolution.continuous_runner import ContinuousAutoEvolutionRunner
from sonic.evolution.self_coder_candidates import SelfCoderCandidateStore


def test_continuous_runner_initialization(tmp_path):
    """Test default initialization and parameter assignment."""
    events = []
    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        tenant_id="tenant-test",
        max_cycles=5,
        interval_seconds=2,
        auto_commit=False,
        on_event=lambda ev: events.append(ev),
    )
    assert runner.repo_root == tmp_path.resolve()
    assert runner.tenant_id == "tenant-test"
    assert runner.max_cycles == 5
    assert runner.interval_seconds == 2
    assert runner.auto_apply is False
    assert runner.auto_commit is False
    assert runner.is_running is False


def test_continuous_runner_stop_and_pause(tmp_path):
    """Test calling stop() pauses the loop and emits paused event."""
    events = []
    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=10,
        on_event=lambda ev: events.append(ev),
    )
    runner.stop()
    assert runner.is_running is False
    paused_events = [e for e in events if e.get("type") == "paused"]
    assert len(paused_events) == 1
    assert "paused by operator" in paused_events[0]["message"]


@pytest.mark.asyncio
async def test_continuous_runner_idle_when_no_signals(tmp_path):
    """Test continuous loop when no signals are found in codebase."""
    events = []
    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        on_event=lambda ev: events.append(ev),
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = []
        mock_coder_cls.return_value = mock_coder

        await runner.run()

    types = [e.get("type") for e in events]
    assert "started" in types
    assert "cycle_start" in types
    assert "cycle_idle" in types
    assert "completed" in types
    assert "stopped" in types
    assert runner.is_running is False


@pytest.mark.asyncio
async def test_continuous_runner_discovery_error(tmp_path):
    """Test handling of unexpected error during signal discovery."""
    events = []
    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        on_event=lambda ev: events.append(ev),
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.side_effect = RuntimeError("AST parse exploded")
        mock_coder_cls.return_value = mock_coder

        await runner.run()

    types = [e.get("type") for e in events]
    assert "cycle_error" in types
    assert "cycle_idle" not in types
    error_event = next(e for e in events if e.get("type") == "cycle_error")
    assert "AST parse exploded" in error_event["message"]


@pytest.mark.asyncio
async def test_continuous_runner_candidate_execution_error(tmp_path):
    """Test candidate run exception handling."""
    events = []
    mock_store = MagicMock(spec=SelfCoderCandidateStore)
    mock_store.run_candidate = AsyncMock(side_effect=RuntimeError("Worktree lock conflict"))

    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        candidate_store=mock_store,
        on_event=lambda ev: events.append(ev),
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = [
            ("Fix null ref", "sonic/core.py", "defect")
        ]
        mock_coder_cls.return_value = mock_coder

        # Replace sleep to avoid waiting
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await runner.run()

    types = [e.get("type") for e in events]
    assert "signal_found" in types
    assert "testing_worktree" in types
    assert "cycle_error" in types
    error_ev = next(e for e in events if e.get("type") == "cycle_error")
    assert "Worktree lock conflict" in error_ev["message"]


@pytest.mark.asyncio
async def test_continuous_runner_unverified_candidate_discarded(tmp_path):
    """Test that failed/unverified candidate is discarded and not promoted."""
    events = []
    mock_store = MagicMock(spec=SelfCoderCandidateStore)
    mock_store.run_candidate = AsyncMock(return_value={
        "candidates": [
            {
                "candidate_id": "cand-failed-12345",
                "status": "failed",
                "target_file": "sonic/core.py",
            }
        ],
        "report": "Tests failed in worktree",
    })
    mock_store.auto_promote = MagicMock()

    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        candidate_store=mock_store,
        on_event=lambda ev: events.append(ev),
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = [
            ("Fix null ref", "sonic/core.py", "defect")
        ]
        mock_coder_cls.return_value = mock_coder

        await runner.run()

    types = [e.get("type") for e in events]
    assert "candidate_unverified" in types
    mock_store.auto_promote.assert_not_called()


@pytest.mark.asyncio
async def test_continuous_runner_records_verified_candidate_without_promoting(tmp_path):
    """A normally configured runner records the verified candidate for review."""
    events = []
    mock_store = MagicMock(spec=SelfCoderCandidateStore)
    mock_store.run_candidate = AsyncMock(return_value={
        "candidates": [
            {
                "candidate_id": "c1a2b3c4d5e6f708192a3b4c5d6e7f80",
                "status": "verified",
                "target_file": "sonic/core.py",
                "files": ["sonic/core.py"],
                "diff": "verified patch",
                "verification": {"full_test_suite": True},
            }
        ],
        "report": "Verification succeeded in worktree",
    })

    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        candidate_store=mock_store,
        on_event=lambda ev: events.append(ev),
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = [
            ("Remove unused variable", "sonic/core.py", "optimization")
        ]
        mock_coder_cls.return_value = mock_coder

        await runner.run()

    types = [e.get("type") for e in events]
    assert "signal_found" in types
    assert "testing_worktree" in types
    assert "candidate_ready" in types
    assert "promoting" not in types
    assert "cycle_success" not in types

    mock_store.run_candidate.assert_awaited_once()
    assert mock_store.run_candidate.await_args.kwargs["auto_apply"] is False
    assert mock_store.run_candidate.await_args.kwargs["auto_commit"] is False

    ready_ev = next(e for e in events if e.get("type") == "candidate_ready")
    assert "active checkout was not changed" in ready_ev["message"]
    assert ready_ev["candidate_id"] == "c1a2b3c4d5e6f708192a3b4c5d6e7f80"
    assert ready_ev["diff"] == "verified patch"
    assert mock_store.run_candidate.await_args.kwargs["signals"] == [
        ("Remove unused variable", "sonic/core.py", "optimization")
    ]


@pytest.mark.asyncio
async def test_continuous_runner_promote_error(tmp_path):
    """Test that failed commit/version/journal promotion is reported, not success."""
    events = []
    mock_store = MagicMock(spec=SelfCoderCandidateStore)
    mock_store.run_candidate = AsyncMock(return_value={
        "candidates": [
            {
                "candidate_id": "c1a2b3c4d5e6f708192a3b4c5d6e7f80",
                "status": "applied",
                "target_file": "sonic/core.py",
                "promotion": "verified patch applied; commit failed: Git index locked",
            }
        ],
        "report": "Verification succeeded",
    })

    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        candidate_store=mock_store,
        on_event=lambda ev: events.append(ev),
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = [
            ("Remove unused variable", "sonic/core.py", "optimization")
        ]
        mock_coder_cls.return_value = mock_coder

        await runner.run()

    types = [e.get("type") for e in events]
    assert "promote_error" in types
    err_ev = next(e for e in events if e.get("type") == "promote_error")
    assert "Git index locked" in err_ev["message"]
    assert "cycle_success" not in types


@pytest.mark.asyncio
async def test_continuous_runner_respects_auto_commit_option(tmp_path):
    events = []
    mock_store = MagicMock(spec=SelfCoderCandidateStore)
    mock_store.run_candidate = AsyncMock(return_value={"candidates": []})

    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        auto_commit=False,
        candidate_store=mock_store,
        on_event=lambda ev: events.append(ev),
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = [
            ("Fix null ref", "sonic/core.py", "defect")
        ]
        mock_coder_cls.return_value = mock_coder

        await runner.run()

    mock_store.run_candidate.assert_awaited_once()
    assert mock_store.run_candidate.await_args.kwargs["auto_apply"] is False
    assert mock_store.run_candidate.await_args.kwargs["auto_commit"] is False


def test_candidate_store_auto_promote_with_commit_and_journal(tmp_path, monkeypatch):
    """Verify that SelfCoderCandidateStore.auto_promote with auto_commit=True commits and logs."""
    monkeypatch.setenv("SONIC_EVOLUTION_DB_PATH", str(tmp_path / "evolution.db"))
    repo = tmp_path / "repo"
    repo.mkdir()

    # 1. Setup git repo
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=str(repo), check=True, capture_output=True)

    # Initial file
    code_file = repo / "hello.py"
    code_file.write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(repo), check=True, capture_output=True)

    rev_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True)
    base_commit = rev_proc.stdout.strip()

    # 2. Candidate diff
    new_code = "def hello():\n    return 'hello world'\n"
    diff_text = (
        "diff --git a/hello.py b/hello.py\n"
        "--- a/hello.py\n"
        "+++ b/hello.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def hello():\n"
        "-    return 'hello'\n"
        "+    return 'hello world'\n"
    )

    store = SelfCoderCandidateStore(repo_root=repo)
    cand_id = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    record = {
        "candidate_id": cand_id,
        "tenant_id": "test-tenant",
        "created_at": "2026-10-05T00:00:00Z",
        "base_commit": base_commit,
        "base_branch": "master",
        "status": "verified",
        "description": "Improve greeting output",
        "category": "logic_improvement",
        "target_file": "hello.py",
        "files": ["hello.py"],
        "diff": diff_text,
        "verification": {
            "full_test_suite": True,
            "security_regression": True,
            "test_summary": "1 passed in 0.05s",
        },
        "report": "All tests passed",
    }
    store.save(record)

    # 3. Auto-promote with auto_commit=True
    with patch.object(store, "_verify_active_application", return_value=(True, {"full_test_suite": True})):
        promoted = store.auto_promote(cand_id, "test-tenant", auto_commit=True)

    assert promoted["status"] == "promoted"
    assert promoted["commit"] != ""
    assert promoted["version_after"]
    assert promoted["journal_recorded"] is True
    assert "verified patch committed locally" in promoted["promotion"]

    # Verify git log shows the commit
    log_proc = subprocess.run(["git", "log", "-n", "1", "--oneline"], cwd=str(repo), check=True, capture_output=True, text=True)
    assert f"evo({cand_id[:8]})" in log_proc.stdout

    # Verify file was updated
    assert code_file.read_text(encoding="utf-8") == new_code

    # Verify evolution.md exists and contains the entry
    journal_file = repo / "evolution.md"
    assert journal_file.exists()
    journal_content = journal_file.read_text(encoding="utf-8")
    assert cand_id[:8] in journal_content
    assert "Improve greeting output" in journal_content


# =====================================================================
# Signal advancement (stuck-loop regression)
# =====================================================================


def test_select_signal_skips_attempted_signals():
    """_select_signal must advance past signals already attempted.

    Regression: the runner pinned ``signals[0]`` every cycle. Once that signal
    was recorded as attempted the coder-side dedup filtered it out, so the
    runner reported ``no_candidate`` forever and never reached signals 1..N.
    """
    sigs = [("d0", "f0", "bug_fix"), ("d1", "f1", "bug_fix"), ("d2", "f2", "bug_fix")]

    assert ContinuousAutoEvolutionRunner._select_signal(sigs, set()) == sigs[0]
    assert ContinuousAutoEvolutionRunner._select_signal(sigs, {("f0", "d0")}) == sigs[1]
    assert (
        ContinuousAutoEvolutionRunner._select_signal(sigs, {("f0", "d0"), ("f1", "d1")})
        == sigs[2]
    )
    assert (
        ContinuousAutoEvolutionRunner._select_signal(
            sigs, {("f0", "d0"), ("f1", "d1"), ("f2", "d2")}
        )
        is None
    )


@pytest.mark.asyncio
async def test_continuous_runner_advances_to_next_signal(tmp_path):
    """A second cycle must attempt a DIFFERENT signal, not repeat the first."""
    events = []
    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=2,
        interval_seconds=0,
        on_event=lambda ev: events.append(ev),
    )

    sig_a = ("desc A", "file_a.py", "bug_fix")
    sig_b = ("desc B", "file_b.py", "bug_fix")

    # Cycle 1 records sig_a as attempted; cycle 2 must therefore pick sig_b.
    store = runner.candidate_store
    store.attempted_signal_keys = MagicMock(
        side_effect=[
            set(),
            {("file_a.py", "desc A")},
        ]
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls, patch.object(
        runner.candidate_store, "run_candidate", new=AsyncMock(return_value={"candidates": []})
    ):
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = [sig_a, sig_b]
        mock_coder_cls.return_value = mock_coder
        await runner.run()

    found = [e for e in events if e.get("type") == "signal_found"]
    assert len(found) == 2
    assert found[0]["target_file"] == "file_a.py"
    assert found[1]["target_file"] == "file_b.py"


@pytest.mark.asyncio
async def test_continuous_runner_idle_when_all_signals_attempted(tmp_path):
    """When every discovered signal was already attempted, emit cycle_idle."""
    events = []
    runner = ContinuousAutoEvolutionRunner(
        repo_root=tmp_path,
        max_cycles=1,
        interval_seconds=0,
        on_event=lambda ev: events.append(ev),
    )
    runner.candidate_store.attempted_signal_keys = MagicMock(
        return_value={("file_a.py", "desc A")}
    )

    with patch("sonic.evolution.continuous_runner.SelfCoder") as mock_coder_cls:
        mock_coder = MagicMock()
        mock_coder.discover_signals.return_value = [("desc A", "file_a.py", "bug_fix")]
        mock_coder_cls.return_value = mock_coder
        await runner.run()

    types = [e.get("type") for e in events]
    assert "cycle_idle" in types
    assert "signal_found" not in types


def test_candidate_store_attempted_signal_keys(tmp_path):
    """attempted_signal_keys returns only attempted (file, description) pairs."""
    store = SelfCoderCandidateStore(repo_root=tmp_path)
    store.storage_dir.mkdir(parents=True, exist_ok=True)
    (store.storage_dir / "a.json").write_text(
        '{"tenant_id": "t", "base_commit": "c1", "target_file": "f1",'
        ' "description": "d1", "status": "rejected"}',
        encoding="utf-8",
    )
    (store.storage_dir / "b.json").write_text(
        '{"tenant_id": "t", "base_commit": "c1", "target_file": "f2",'
        ' "description": "d2", "status": "no_candidate"}',
        encoding="utf-8",
    )
    (store.storage_dir / "c.json").write_text(
        '{"tenant_id": "other", "base_commit": "c1", "target_file": "f3",'
        ' "description": "d3", "status": "rejected"}',
        encoding="utf-8",
    )

    keys = store.attempted_signal_keys("t", base_commit="c1")
    assert keys == {("f1", "d1")}
