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


def test_cli_help_preserves_subcommands():
    """Verify all subcommands remain registered and available."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "status" in result.output
    assert "auth" in result.output
    assert "computer" in result.output
    assert "evolution" in result.output
    assert "prompt" in result.output


def test_zero_arg_invokes_repl(monkeypatch):
    """Verify invoking sonic with zero arguments triggers run_repl."""
    monkeypatch.setenv("SONIC_FORCE_INTERACTIVE", "1")
    with patch("sonic_cli.repl.run_repl") as mock_repl:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        mock_repl.assert_called_once()


def test_piped_stdin_dispatches_prompt():
    """Verify piped input routes to run_prompt in batch mode."""
    with patch("sonic_cli.main.run_prompt") as mock_prompt:
        result = runner.invoke(app, [], input="Fix the tests\n")
        assert result.exit_code == 0
        mock_prompt.assert_called_once_with(
            prompt="Fix the tests",
            session_id="default",
            mode="normal",
            server="http://localhost:12000",
            token="",
        )


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


def test_accepted_backend_prompt_polls_until_workstation_is_idle(monkeypatch):
    context = resolve_local_context()
    accepted = MagicMock(status_code=200)
    accepted.json.return_value = {"status": "accepted"}
    state = MagicMock(status_code=200)
    state.json.return_value = {
        "status": "IDLE",
        "worklog": [{"title": "Finished", "content": "Task completed."}],
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = accepted
    client.get.return_value = state
    monkeypatch.setattr(repl.httpx, "Client", MagicMock(return_value=client))
    monkeypatch.setattr(repl.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(repl.console, "print", MagicMock())

    _dispatch_prompt(
        "Update the project",
        context,
        server_url="http://localhost:12000",
        token="operator-token",
    )

    client.post.assert_called_once()
    assert client.post.call_args.kwargs["json"]["target_repo"] == str(context.repo_root)
    assert client.post.call_args.kwargs["headers"].get("Authorization")
    client.get.assert_called_once()
    assert client.get.call_args.args[0] == "/workstation/state"
