"""
SONIC CLI — Diff Command
========================
Inspect local git diff in the active workspace.
"""

from __future__ import annotations

import subprocess

import typer
from rich.console import Console
from rich.syntax import Syntax

from sonic_cli.context import LocalWorkspaceContext, resolve_local_context

console = Console(legacy_windows=False)


def run_git_diff(
    ctx: LocalWorkspaceContext,
    staged: bool = False,
    stat: bool = False,
    ref: str | None = None,
) -> int:
    """Execute git diff and render formatted syntax."""
    if not ctx.is_git:
        console.print("[yellow]Not in a git repository; no git diff available.[/yellow]")
        return 1

    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    if stat:
        cmd.append("--stat")
    if ref:
        cmd.append(ref)
    elif not staged and not stat:
        # Default to HEAD so both staged and unstaged changes show up
        cmd.append("HEAD")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ctx.repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        diff_text = proc.stdout.strip()
        if not diff_text:
            console.print("[dim]Working directory clean. No uncommitted diffs.[/dim]")
            return 0

        if stat:
            console.print(diff_text)
        else:
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
            console.print(syntax)
        return proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        console.print(f"[red]Failed to compute git diff: {exc}[/red]")
        return 1


def diff_command(
    staged: bool = typer.Option(False, "--staged", "-s", help="Inspect staged changes"),
    stat: bool = typer.Option(False, "--stat", help="Show diffstat summary instead of full patch"),
    ref: str | None = typer.Argument(None, help="Optional git commit, branch, or ref to compare against"),
):
    """🔍 Inspect local git diff in current repository."""
    ctx = resolve_local_context()
    code = run_git_diff(ctx, staged=staged, stat=stat, ref=ref)
    if code != 0:
        raise typer.Exit(code=code)
