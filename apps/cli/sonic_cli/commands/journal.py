"""
SONIC CLI — Journal Command
===========================
View evolution.md journal and candidate logs.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from sonic_cli.context import LocalWorkspaceContext, resolve_local_context

console = Console(legacy_windows=False)


def display_journal(ctx: LocalWorkspaceContext, file_path: Path | None = None) -> None:
    """Render the evolution journal or candidate records."""
    journal_path = file_path or (ctx.repo_root / "evolution.md")
    if journal_path.exists():
        try:
            content = journal_path.read_text(encoding="utf-8")
            console.print(Markdown(content))
            return
        except Exception as exc:
            console.print(f"[red]Failed to read journal at {journal_path}: {exc}[/red]")
            return

    candidates_dir = ctx.repo_root / "candidates"
    if candidates_dir.exists():
        files = list(candidates_dir.glob("*.json"))
        if files:
            console.print(f"[cyan]Found {len(files)} candidate records in {candidates_dir}:[/cyan]")
            for f in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                console.print(f"  • {f.name}")
            return

    console.print("[dim]No evolution.md or candidate records found in workspace yet.[/dim]")


def journal_command(
    file: Path | None = typer.Option(None, "--file", "-f", help="Custom evolution journal file path"),
):
    """📖 View evolution.md journal and verification receipts."""
    ctx = resolve_local_context()
    display_journal(ctx, file_path=file)
