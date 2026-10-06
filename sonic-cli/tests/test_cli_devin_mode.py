"""Tests for Devin-style Zero-Flag CLI and Auto-Evolution mode.

Belongs to Department: platform_systems, Team: api_services, Subsystem: cli.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from sonic_cli import repl
from sonic_cli.context import LocalWorkspaceContext, get_token, resolve_local_context
from sonic_cli.main import app
from sonic_cli.repl import (
    _dispatch_prompt,
    _handle_diff,
    _handle_help,
    _handle_journal,
    _handle_status,
    _print_welcome_banner,
)

runner = CliRunner()


def test_resolve_local_context():
    """Verify local repository and git context are inferred without flags."""
    ctx = resolve_local_context()
    assert ctx.repo_root.is_dir()
    assert ctx.is_git is True
    assert ctx.branch != ""
    assert ctx.tenant_id.startswith("tenant-")
    assert ctx.workspace_id.startswith("ws-")
    assert ctx.session_id.startswith("sess-")


def test_resolve_local_context_falls_back_to_source_checkout_outside_git(tmp_path, monkeypatch):
    """Launching the CLI outside a repository must not target the process cwd."""
    import sonic_cli.context as context_module

    source_checkout = Path(context_module.__file__).resolve().parents[2]
    monkeypatch.setattr(
        context_module,
        "_git_root",
        lambda path: source_checkout if Path(path).resolve() == source_checkout else None,
    )
    ctx = resolve_local_context(cwd=tmp_path)

    assert ctx.is_git is True
    assert ctx.repo_root == source_checkout
    assert ctx.branch


def test_resolve_local_context_honors_configured_repo_root(monkeypatch, tmp_path):
    """An explicit source checkout overrides the caller's unrelated working directory."""
    source_checkout = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("SONIC_REPO_ROOT", str(source_checkout))

    ctx = resolve_local_context(cwd=tmp_path)

    assert ctx.is_git is True
    assert ctx.repo_root == source_checkout


def test_get_token_resolution(monkeypatch, tmp_path):
    """Verify auth token resolution from flag, env, or file."""
    # 1. From argument
    assert get_token("direct-token") == "direct-token"

    # 2. From env var
    monkeypatch.setenv("SONIC_TOKEN", "env-secret-token")
    assert get_token("") == "env-secret-token"

    # 3. From ~/.sonic/token
    monkeypatch.delenv("SONIC_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    sonic_dir = fake_home / ".sonic"
    sonic_dir.mkdir(parents=True)
    token_file = sonic_dir / "token"
    token_file.write_text("file-secret-token\n", encoding="utf-8")

    with patch("pathlib.Path.home", return_value=fake_home):
        assert get_token("") == "file-secret-token"


def test_cli_version_command():
    """Verify version command executes and prints branding."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "SONIC" in result.output
    assert "Autonomous AI-Human Digital Being" in result.output


def test_cli_help_clean_developer_commands():
    """Verify only clean developer commands are registered and Boss/Workstation commands are removed."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Clean developer commands present
    assert "status" in result.output
    assert "diff" in result.output
    assert "test" in result.output
    assert "auto" in result.output
    assert "journal" in result.output
    assert "auth" in result.output
    assert "version" in result.output

    # Boss / Workstation commands removed from main.py
    for removed in [
        "computer", "mission", "agents", "engage", "finding",
        "evidence", "research", "security", "experiment", "graph",
        "prompt", "missions", "hypotheses", "kill",
    ]:
        assert removed not in result.output


def test_zero_arg_invokes_repl_seamlessly():
    """Verify invoking bare sonic with zero arguments triggers run_repl seamlessly."""
    with patch("sonic_cli.repl.run_repl") as mock_repl:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        mock_repl.assert_called_once()


def test_cli_diff_command():
    """Verify sonic diff runs local git diff."""
    with patch("sonic_cli.commands.diff.run_git_diff", return_value=0) as mock_diff:
        result = runner.invoke(app, ["diff"])
        assert result.exit_code == 0
        mock_diff.assert_called_once()


def test_cli_test_command():
    """Verify sonic test invokes pytest."""
    fake_ctx = LocalWorkspaceContext(
        repo_root=Path.cwd(),
        branch="main",
        tenant_id="tenant-test",
        workspace_id="ws-test",
        session_id="sess-test",
        is_git=True,
    )
    with patch("sonic_cli.commands.test.resolve_local_context", return_value=fake_ctx):
        with patch("sonic_cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["test", "-v"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "pytest" in args
            assert "-v" in args



def test_cli_status_command():
    """Verify sonic status displays local workspace status and environment check."""
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    assert "workspace" in result.output
    assert "environment" in result.output


def test_cli_journal_command(tmp_path):
    """Verify sonic journal displays evolution journal."""
    journal_file = tmp_path / "evolution.md"
    journal_file.write_text("# Evolution Journal\nTest record\n", encoding="utf-8")
    result = runner.invoke(app, ["journal", "--file", str(journal_file)])
    assert result.exit_code == 0
    assert "Evolution Journal" in result.output


def test_cli_auto_command():
    """Verify sonic auto triggers continuous auto-evolution."""
    with patch("sonic_cli.repl._run_continuous_auto_evolution") as mock_auto:
        mock_auto.return_value = None
        result = runner.invoke(app, ["auto", "--cycles", "2"])
        assert result.exit_code == 0


def test_continuous_auto_evolution_runner_init():
    """Verify ContinuousAutoEvolutionRunner initializes with correct parameters."""
    from sonic.evolution.continuous_runner import ContinuousAutoEvolutionRunner

    events = []
    runner_inst = ContinuousAutoEvolutionRunner(
        repo_root=Path.cwd(),
        max_cycles=2,
        interval_seconds=1,
        on_event=lambda ev: events.append(ev),
    )
    assert runner_inst.repo_root == Path.cwd().resolve()
    assert runner_inst.max_cycles == 2
    assert runner_inst.is_running is False


def test_auto_mode_runs_until_operator_stops_it():
    """The continuous CLI mode uses the runner's unlimited-cycle setting."""
    from sonic_cli import repl

    run_called = []

    class StubRunner:
        def __init__(self, **kwargs):
            assert kwargs["max_cycles"] == 0
            self.kwargs = kwargs

        async def run(self):
            run_called.append(True)

        def stop(self):
            return None

    context = resolve_local_context()
    with (
        patch.object(repl.console, "print"),
        patch(
            "sonic.evolution.continuous_runner.ContinuousAutoEvolutionRunner",
            StubRunner,
        ),
    ):
        asyncio.run(repl._run_continuous_auto_evolution(context))

    assert run_called == [True]


def test_repl_help_and_status():
    context = resolve_local_context()
    _print_welcome_banner(context)
    _handle_help()
    _handle_status(context)


def test_repl_diff_and_journal(tmp_path):
    context = LocalWorkspaceContext(
        repo_root=tmp_path,
        branch="test",
        tenant_id="tenant-test",
        workspace_id="ws-test",
        session_id="sess-test",
        is_git=False,
    )
    _handle_diff(context)
    _handle_journal(context)

    (tmp_path / "evolution.md").write_text(
        "# Evolution Journal\n- Step 1: OK\n", encoding="utf-8"
    )
    _handle_journal(context)


def test_dispatch_prompt_invokes_developer_agent():
    """Verify natural language prompts route directly to the local DeveloperAgent."""
    context = resolve_local_context()
    with patch("sonic_cli.developer_agent.DeveloperAgent") as mock_agent_cls:
        mock_instance = MagicMock()
        mock_agent_cls.return_value = mock_instance
        _dispatch_prompt("Refactor the login module", context)
        mock_agent_cls.assert_called_once_with(
            workspace_dir=context.repo_root,
            console=repl.console,
        )
        mock_instance.run.assert_called_once_with("Refactor the login module")


def test_dev_auth_login_saves_token_and_uses_configured_backend(monkeypatch, tmp_path):
    from sonic_cli.commands import auth

    token_path = tmp_path / ".sonic" / "token"
    monkeypatch.setattr(auth, "TOKEN_PATH", token_path)
    monkeypatch.setattr(auth.httpx, "post", MagicMock(
        return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"access_token": "test-token"}),
        )
    ))

    result = runner.invoke(app, [
        "auth", "login", "--dev", "--email", "operator@example.test",
        "--server", "http://127.0.0.1:12000",
    ])

    assert result.exit_code == 0
    assert token_path.read_text(encoding="utf-8") == "test-token"
    auth.httpx.post.assert_called_once_with(
        "http://127.0.0.1:12000/auth/login",
        json={"email": "operator@example.test"},
        timeout=15,
    )


def test_auth_whoami_uses_saved_token(monkeypatch, tmp_path):
    from sonic_cli.commands import auth

    monkeypatch.delenv("SONIC_TOKEN", raising=False)
    token_path = tmp_path / ".sonic" / "token"
    token_path.parent.mkdir()
    token_path.write_text("test-token", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(auth, "TOKEN_PATH", token_path)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"authenticated": True}
    request = MagicMock(return_value=response)
    monkeypatch.setattr(auth.httpx, "get", request)

    result = runner.invoke(app, ["auth", "whoami", "--server", "http://127.0.0.1:12000"])

    assert result.exit_code == 0
    request.assert_called_once_with(
        "http://127.0.0.1:12000/auth/me",
        headers={"Authorization": "Bearer test-token"},
        timeout=15,
    )


def test_auth_logout_removes_local_token(monkeypatch, tmp_path):
    from sonic_cli.commands import auth

    token_path = tmp_path / ".sonic" / "token"
    token_path.parent.mkdir()
    token_path.write_text("test-token", encoding="utf-8")
    monkeypatch.setattr(auth, "TOKEN_PATH", token_path)

    result = runner.invoke(app, ["auth", "logout"])

    assert result.exit_code == 0
    assert not token_path.exists()


def test_repl_greeting_handling_instant(monkeypatch):
    """Greetings (hi, hello, hii) return instant developer greeting in < 100ms without backend dispatch."""
    import time
    context = resolve_local_context()
    printed = []
    monkeypatch.setattr(repl.console, "print", lambda msg: printed.append(str(msg)))

    for greeting in ["hi", "hello", "hii", "Hi!", "Hii"]:
        printed.clear()
        t0 = time.perf_counter()
        with patch("sonic_cli.developer_agent.DeveloperAgent") as mock_agent:
            _dispatch_prompt(greeting, context)
            mock_agent.assert_not_called()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Greeting {greeting} took {elapsed_ms}ms, expected < 100ms"
        assert any("Hello!" in p and "SONIC" in p for p in printed)

