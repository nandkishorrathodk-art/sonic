"""
SONIC CLI — Test Command
========================
Run pytest inside the local repository workspace.
"""

from __future__ import annotations

import subprocess
import sys

import typer
from rich.console import Console

from sonic_cli.context import resolve_local_context

console = Console(legacy_windows=False)


def test_command(
    path: str | None = typer.Argument(None, help="Target test directory or file path"),
    keyword: str | None = typer.Option(None, "-k", help="Run tests matching the expression"),
    last_failed: bool = typer.Option(False, "--lf", "--last-failed", help="Run only failed tests first"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose pytest output"),
    extra_args: list[str] | None = typer.Option(None, "--arg", "-a", help="Extra arguments to pass to pytest"),
):
    """🧪 Run pytest in the local repository workspace."""
    ctx = resolve_local_context()
    repo_root = ctx.repo_root

    cmd = [sys.executable, "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    if last_failed:
        cmd.append("--lf")
    if keyword:
        cmd.extend(["-k", keyword])
    if path:
        cmd.append(path)
    if extra_args:
        cmd.extend(extra_args)

    console.print(f"[bold cyan]Running pytest in {repo_root}...[/bold cyan]")
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root))
        if proc.returncode != 0:
            raise typer.Exit(code=proc.returncode)
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Test execution interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as exc:
        console.print(f"[red]Failed to execute pytest: {exc}[/red]")
        raise typer.Exit(code=1) from exc
