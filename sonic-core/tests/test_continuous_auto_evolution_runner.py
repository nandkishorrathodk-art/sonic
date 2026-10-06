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
async def test_continuous_runner_successful_cycle_promoted(tmp_path):
    """Test the runner requests committed promotion from the candidate workflow."""
    events = []
    mock_store = MagicMock(spec=SelfCoderCandidateStore)
    mock_store.run_candidate = AsyncMock(return_value={
        "candidates": [
            {
                "candidate_id": "c1a2b3c4d5e6f708192a3b4c5d6e7f80",
                "status": "promoted",
                "target_file": "sonic/core.py",
                "commit": "deadbeef1234567890abcdef",
                "version_after": "0.1.1",
            }
        ],
        "report": "Verification succeeded in worktree",
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
            ("Remove unused variable", "sonic/core.py", "optimization")
        ]
        mock_coder_cls.return_value = mock_coder

        await runner.run()

    types = [e.get("type") for e in events]
    assert "signal_found" in types
    assert "testing_worktree" in types
    assert "promoting" in types
    assert "cycle_success" in types

    mock_store.run_candidate.assert_awaited_once()
    assert mock_store.run_candidate.await_args.kwargs["auto_commit"] is True
    mock_store.auto_promote.assert_not_called()

    success_ev = next(e for e in events if e.get("type") == "cycle_success")
    assert "deadbeef" in success_ev["message"]
    assert success_ev["commit"] == "deadbeef1234567890abcdef"
    assert success_ev["version"] == "0.1.1"
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
