"""
Tests for Self-Coder API security, tenant isolation, and CLI enhancements.
==========================================================================
Covers:
1. self_coder API tenant-scoped reports storage and isolation.
2. self-coder candidates never push automatically and fail closed on worktree errors.
3. self_coder API non-blocking audit & signals via asyncio.to_thread.
4. CLI & Journal changelog version range filtering on promoted entries.
5. CLI --max-goals > 0 validation in build_parser() and main().
6. CLI patch subcommand path traversal prevention on --target and --summary-out.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from sonic.api.routes.self_coder import (
    SelfCoderRunRequest,
    _last_reports_by_tenant,
    self_coder_audit,
    self_coder_report,
    self_coder_run,
    self_coder_signals,
)
from sonic.auth.models import User, UserRole
from sonic.evolution.cli import (
    _positive_int,
    build_parser,
    is_safe_repo_path,
    main,
)
from sonic.evolution.cli import (
    generate_changelog as cli_changelog,
)
from sonic.evolution.evolution_journal import EvolutionJournal, JournalEntry

# =====================================================================
# 1. API Tenant Isolation & RBAC Tests
# =====================================================================

@pytest.mark.asyncio
async def test_self_coder_tenant_scoped_reports():
    """Verify that self_coder reports are isolated per-tenant."""
    _last_reports_by_tenant.clear()

    user_a = User(email="a@target.com", name="User A", tenant_id="tenant_alpha", role=UserRole.OPERATOR)
    user_b = User(email="b@target.com", name="User B", tenant_id="tenant_beta", role=UserRole.OPERATOR)
    user_none = User(email="def@target.com", name="User Def", tenant_id="", role=UserRole.OPERATOR)

    # Initially empty for all
    rep_a = await self_coder_report(user=user_a)
    assert rep_a == {"cycles": [], "changed": False, "text": ""}

    class MockStore:
        async def run_candidate(self, tenant_id, max_cycles, coder_factory):
            return {"candidates": [{"tenant_id": tenant_id, "status": "verified"}],
                    "report": f"Report for {tenant_id}"}

    with patch("sonic.api.routes.self_coder._candidate_store", return_value=MockStore()):
        run_res_a = await self_coder_run(SelfCoderRunRequest(max_cycles=1), user=user_a)

    assert run_res_a["changed"] is False
    assert run_res_a["active_branch_changed"] is False
    assert run_res_a["active_worktree_changed"] is False
    assert run_res_a["text"] == "Report for tenant_alpha"
    assert run_res_a["candidates"][0]["tenant_id"] == "tenant_alpha"
    assert "tenant_alpha" in _last_reports_by_tenant

    # User A sees Alpha report
    res_a = await self_coder_report(user=user_a)
    assert res_a["text"] == "Report for tenant_alpha"

    # User B still sees empty report
    res_b = await self_coder_report(user=user_b)
    assert res_b == {"cycles": [], "changed": False, "text": ""}

    # User with empty tenant_id maps to "default"
    with patch("sonic.api.routes.self_coder._candidate_store", return_value=MockStore()):
        await self_coder_run(SelfCoderRunRequest(max_cycles=1), user=user_none)

    assert "default" in _last_reports_by_tenant
    res_def = await self_coder_report(user=user_none)
    assert res_def["text"] == "Report for default"


@pytest.mark.asyncio
async def test_self_coder_run_never_pushes():
    """Self-coder runs only record candidates; remote pushes are never supported."""
    operator_user = User(email="op@target.com", name="Op", role=UserRole.OPERATOR)
    with pytest.raises(HTTPException) as exc_info:
        await self_coder_run(SelfCoderRunRequest(push=True), user=operator_user)
    assert exc_info.value.status_code == 400
    assert "never push" in exc_info.value.detail

    # Push remains disabled for every role, including tenant/global admins.
    auditor_user = User(email="audit@target.com", name="Auditor", role=UserRole.AUDITOR)
    with pytest.raises(HTTPException) as exc_info:
        await self_coder_run(SelfCoderRunRequest(push=True), user=auditor_user)
    assert exc_info.value.status_code == 400

    tenant_admin = User(email="admin@target.com", name="Tenant Admin", role=UserRole.TENANT_ADMIN)
    with pytest.raises(HTTPException) as exc_info:
        await self_coder_run(SelfCoderRunRequest(push=True), user=tenant_admin)
    assert exc_info.value.status_code == 400
    super_admin = User(email="super@target.com", name="Super Admin", role=UserRole.SUPER_ADMIN)
    with pytest.raises(HTTPException) as exc_info:
        await self_coder_run(SelfCoderRunRequest(push=True), user=super_admin)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_self_coder_candidate_run_fails_closed_on_worktree_error():
    user = User(email="u@target.com", name="U", role=UserRole.OPERATOR)
    with patch("sonic.api.routes.self_coder._candidate_store") as store_factory:
        store_factory.return_value.run_candidate = AsyncMock(side_effect=RuntimeError("worktree unavailable"))
        with pytest.raises(HTTPException) as exc_info:
            await self_coder_run(SelfCoderRunRequest(), user=user)
    assert exc_info.value.status_code == 500
    assert "worktree unavailable" in exc_info.value.detail


@pytest.mark.asyncio
async def test_self_coder_async_to_thread_calls():
    """Verify that auditor.audit and coder.discover_signals run in threads."""
    user = User(email="u@target.com", name="U", role=UserRole.OPERATOR)

    mock_audit_report = MagicMock()
    mock_audit_report.ok = True
    mock_audit_report.files_scanned = 10
    mock_audit_report.counts.return_value = {}
    mock_audit_report.health = {}
    mock_audit_report.tests = {}
    mock_audit_report.sorted_findings.return_value = []
    mock_audit_report.to_text.return_value = "Audit Text"

    with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
        with patch("sonic.evolution.codebase_auditor.CodebaseAuditor.audit", return_value=mock_audit_report) as mock_audit:
            res = await self_coder_audit(user=user)
            assert res["ok"] is True
            assert mock_to_thread.called
            assert any(call.args and call.args[0] == mock_audit for call in mock_to_thread.call_args_list)

    mock_coder = MagicMock()
    mock_coder.discover_signals.return_value = [("sig1", "target.py", "bug_fix")]

    with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
        with patch("sonic.api.routes.self_coder._build_coder", return_value=mock_coder):
            res = await self_coder_signals(user=user)
            assert res["count"] == 1
            assert mock_to_thread.called
            assert any(call.args and call.args[0] == mock_coder.discover_signals for call in mock_to_thread.call_args_list)


# =====================================================================
# 2. CLI & Evolution Journal Version Range Filtering Tests
# =====================================================================

def test_generate_changelog_version_range_filtering(tmp_path: Path):
    """Verify changelog filters promoted entries by version range."""
    db_file = str(tmp_path / "test_evo.db")
    journal = EvolutionJournal(db_path=db_file, repo_root=tmp_path)

    def _add_entry(title: str, version_before: str, version_after: str, status: str = "promoted"):
        safe_id = title.lower().replace(" ", "_")
        journal.record_entry(JournalEntry(
            entry_id=f"e-{safe_id}",
            goal_id=f"g-{safe_id}",
            title=title,
            category="bug_fix",
            status=status,
            version_before=version_before,
            version_after=version_after,
            files_affected=["file.py"],
            lines_added=5,
            lines_removed=2,
            test_summary="Passed",
            notes=f"Notes for {version_after}",
            commit_hash=f"hash-{version_after}",
        ))

    _add_entry("Initial Fix", "0.1.0", "0.1.1")
    _add_entry("Second Fix", "0.1.1", "0.1.2")
    _add_entry("Feature Upgrade", "0.1.2", "0.2.0")
    _add_entry("Point Release", "0.2.0", "0.2.1")
    _add_entry("Major Milestone", "0.2.1", "0.3.0")
    _add_entry("Failed Rollback", "0.2.1", "0.2.1", status="rolled_back")

    # 1. Full changelog (no filters) -> all 5 promoted entries, no rolled_back
    full_cl = journal.generate_changelog()
    assert "Version `0.1.1`" in full_cl
    assert "Version `0.1.2`" in full_cl
    assert "Version `0.2.0`" in full_cl
    assert "Version `0.2.1`" in full_cl
    assert "Version `0.3.0`" in full_cl
    assert "Failed Rollback" not in full_cl

    # 2. Filter from_version="0.1.2" and to_version="0.2.1"
    range_cl = journal.generate_changelog(from_version="0.1.2", to_version="0.2.1")
    assert "Version `0.1.1`" not in range_cl
    assert "Version `0.1.2`" in range_cl
    assert "Version `0.2.0`" in range_cl
    assert "Version `0.2.1`" in range_cl
    assert "Version `0.3.0`" not in range_cl
    assert "(0.1.2 ➔ 0.2.1)" in range_cl

    # 3. Filter from_version="0.2.0" only
    from_cl = journal.generate_changelog(from_version="0.2.0")
    assert "Version `0.1.1`" not in from_cl
    assert "Version `0.1.2`" not in from_cl
    assert "Version `0.2.0`" in from_cl
    assert "Version `0.2.1`" in from_cl
    assert "Version `0.3.0`" in from_cl

    # 4. Filter to_version="0.1.2" only
    to_cl = journal.generate_changelog(to_version="0.1.2")
    assert "Version `0.1.1`" in to_cl
    assert "Version `0.1.2`" in to_cl
    assert "Version `0.2.0`" not in to_cl

    # 5. CLI wrapper delegating to journal
    cli_out = cli_changelog(journal, from_version="0.1.2", to_version="0.2.1")
    assert cli_out == range_cl


# =====================================================================
# 3. CLI --max-goals Validation Tests
# =====================================================================

def test_cli_max_goals_positive_validation():
    """Verify --max-goals must be > 0 in build_parser() and _positive_int()."""
    assert _positive_int("5") == 5
    assert _positive_int("1") == 1

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-3")

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("not_a_number")

    parser = build_parser()
    args_valid = parser.parse_args(["run", "--max-goals", "3"])
    assert args_valid.max_goals == 3

    # Rejection during parse_args
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--max-goals", "0"])

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--max-goals", "-5"])


def test_cli_main_max_goals_runtime_validation():
    """Verify main() returns 1 if max_goals <= 0 is passed."""
    fake_args = argparse.Namespace(
        command="run",
        max_goals=0,
        push=False,
    )
    with patch("sonic.evolution.cli.build_parser") as mock_bp:
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = fake_args
        mock_bp.return_value = mock_parser

        ret = main()
        assert ret == 1


# =====================================================================
# 4. CLI Patch Path Traversal Protection Tests
# =====================================================================

def test_is_safe_repo_path(tmp_path: Path):
    """Verify is_safe_repo_path correctly detects directory traversal escapes."""
    repo_root = tmp_path / "my_repo"
    repo_root.mkdir()
    (repo_root / "subdir").mkdir()
    (repo_root / "target.py").write_text("code = 1", encoding="utf-8")

    # Safe inside repo
    assert is_safe_repo_path("target.py", repo_root) is True
    assert is_safe_repo_path("subdir/nested.py", repo_root) is True
    assert is_safe_repo_path(str(repo_root / "target.py"), repo_root) is True

    # Escaping repo
    assert is_safe_repo_path("../../etc/passwd", repo_root) is False
    assert is_safe_repo_path("subdir/../../outside.py", repo_root) is False
    assert is_safe_repo_path(str(tmp_path / "outside.txt"), repo_root) is False

    # Empty / whitespace
    assert is_safe_repo_path("", repo_root) is False
    assert is_safe_repo_path("   ", repo_root) is False
    assert is_safe_repo_path(None, repo_root) is False


def test_cli_patch_command_blocks_traversal(tmp_path: Path):
    """Verify patch subcommand refuses directory traversal in --target and --summary-out."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "valid.py").write_text("x = 1\n", encoding="utf-8")

    fake_args_bad_target = argparse.Namespace(
        command="patch",
        target="../../etc/passwd",
        desc="Malicious patch",
        diff_file=None,
        diff="dummy diff",
        auto_promote=False,
        push=False,
        tests=[],
        summary_out=None,
    )

    mock_evolver = MagicMock()
    mock_evolver.repo_root = repo

    with patch("sonic.evolution.cli.build_parser") as mock_bp:
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = fake_args_bad_target
        mock_bp.return_value = mock_parser

        with patch("sonic.evolution.cli.CodebaseEvolver", return_value=mock_evolver):
            ret = main()
            assert ret == 1

    fake_args_bad_summary = argparse.Namespace(
        command="patch",
        target="valid.py",
        desc="Safe target bad summary",
        diff_file=None,
        diff="dummy diff",
        auto_promote=False,
        push=False,
        tests=[],
        summary_out="../../escape.md",
    )

    with patch("sonic.evolution.cli.build_parser") as mock_bp:
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = fake_args_bad_summary
        mock_bp.return_value = mock_parser

        with patch("sonic.evolution.cli.CodebaseEvolver", return_value=mock_evolver):
            ret = main()
            assert ret == 1


def test_cli_status_command_lists_protected_components(tmp_path: Path, capsys):
    """`status` must import the protected-component registry from its real home.

    Regression: the registry lives in ``evolution.pipeline`` but the CLI once
    imported it from ``codebase_evolver``, raising ImportError and breaking the
    whole ``status`` command.
    """
    from sonic.evolution.pipeline import _PROTECTED_COMPONENTS

    fake_args = argparse.Namespace(command="status", repo_root=str(tmp_path))
    with patch("sonic.evolution.cli.build_parser") as mock_bp:
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = fake_args
        mock_bp.return_value = mock_parser
        ret = main()

    assert ret == 0
    out = capsys.readouterr().out
    assert "Protected Safety Components" in out
    # Every protected component must be printed (proves the import resolved).
    for component in _PROTECTED_COMPONENTS:
        assert component in out
    assert "sonic/safety" in out
