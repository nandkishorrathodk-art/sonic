"""Tests for Devin-style Zero-Flag CLI and Auto-Evolution mode.

Belongs to Department: platform_systems, Team: api_services, Subsystem: cli.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sonic_cli.context import resolve_local_context, get_token
from sonic_cli.main import app

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
