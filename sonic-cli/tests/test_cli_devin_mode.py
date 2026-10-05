"""Tests for Devin-style interactive CLI mode and zero-flag execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from sonic_cli import repl
from sonic_cli.context import LocalWorkspaceContext, get_token, resolve_local_context
from sonic_cli.main import app
from sonic_cli.repl import (
    _handle_diff,
    _handle_help,
    _handle_journal,
    _handle_status,
    _print_welcome_banner,
)

runner = CliRunner()


def test_cli_version_command():
    """Verify `sonic version` command prints clean modern panel without crashes."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "SONIC" in result.output
    assert "Autonomous AI-Human Digital Being" in result.output


def test_cli_help_flag():
    """Verify `sonic --help` lists subcommands and description."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Autonomous AI-Human Digital Being" in result.output
    assert "mission" in result.output
    assert "computer" in result.output
    assert "evolution" in result.output


def test_cli_zero_args_triggers_repl(monkeypatch):
    """Verify that bare `sonic` triggers `run_repl` callback instead of failing."""
    repl_called = []

    def fake_run_repl(**kwargs):
        repl_called.append(kwargs)

    monkeypatch.setattr("sonic_cli.repl.run_repl", fake_run_repl)

    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert repl_called == [{
        "server_url": "http://127.0.0.1:12000",
        "token": "",
    }]


def test_resolve_local_context_in_git_repo(tmp_path):
    """Verify local context resolver correctly inspects git repository."""
    ctx = resolve_local_context(Path.cwd())
    assert isinstance(ctx, LocalWorkspaceContext)
    assert ctx.is_git is True
    assert ctx.branch != ""
    assert ctx.tenant_id.startswith("tenant-")
    assert ctx.workspace_id.startswith("ws-")
    assert ctx.session_id.startswith("sess-")


def test_resolve_local_context_non_git(monkeypatch, tmp_path):
    """Verify non-git directory fallback."""
    non_git = tmp_path / "scratch"
    non_git.mkdir()
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: MagicMock(returncode=128, stdout="", stderr="fatal: not a git repo"),
    )
    ctx = resolve_local_context(non_git)
    assert ctx.is_git is False
    assert ctx.branch == "main"
    assert ctx.repo_root == non_git.resolve()


def test_get_token_precedence(tmp_path, monkeypatch):
    """Verify get_token priority: flag > env > file."""
    assert get_token("explicit-token") == "explicit-token"

    monkeypatch.setenv("SONIC_TOKEN", "env-token")
    assert get_token() == "env-token"

    monkeypatch.delenv("SONIC_TOKEN", raising=False)
    fake_home = tmp_path / "user"
    fake_sonic = fake_home / ".sonic"
    fake_sonic.mkdir(parents=True)
    (fake_sonic / "token").write_text("file-token", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    assert get_token() == "file-token"


def test_repl_help_and_status(tmp_path):
    """Verify REPL help and status commands output cleanly."""
    ctx = resolve_local_context()
    _print_welcome_banner(ctx)
    _handle_help()
    _handle_status(ctx)


def test_repl_diff_and_journal(tmp_path):
    """Verify diff and journal command handlers."""
    ctx = LocalWorkspaceContext(
        repo_root=tmp_path,
        branch="test",
        tenant_id="tenant-test",
        workspace_id="ws-test",
        session_id="sess-test",
        is_git=False,
    )
    _handle_diff(ctx)
    _handle_journal(ctx)

    (tmp_path / "evolution.md").write_text(
        "# Evolution Journal\n- Step 1: OK\n", encoding="utf-8"
    )
    _handle_journal(ctx)


def test_accepted_backend_prompt_polls_until_workstation_is_idle(monkeypatch):
    context = resolve_local_context()
    response = MagicMock(status_code=200)
    response.json.return_value = {"status": "accepted"}
    state_response = MagicMock(status_code=200)
    state_response.json.return_value = {
        "status": "IDLE",
        "worklog": [{"title": "Finished", "content": "Task completed."}],
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = response
    client.get.return_value = state_response
    monkeypatch.setattr(repl.httpx, "Client", MagicMock(return_value=client))
    monkeypatch.setattr(repl.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(repl.console, "print", MagicMock())

    repl._dispatch_prompt(
        "Update the project",
        context,
        server_url="http://localhost:12000",
        token="operator-token",
    )

    client.post.assert_called_once()
    assert client.post.call_args.kwargs["json"]["target_repo"] == str(context.repo_root)
    assert client.post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer operator-token"
    }
    client.get.assert_called_once()
    assert client.get.call_args.args[0] == "/workstation/state"
