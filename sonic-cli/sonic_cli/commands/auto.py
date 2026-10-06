"""
SONIC CLI — Auto Command
========================
Trigger continuous autonomous self-evolution.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from sonic_cli.context import resolve_local_context

console = Console(legacy_windows=False)


def auto_command(
    cycles: int = typer.Option(0, "--cycles", "-c", help="Number of evolution cycles to run (0 for continuous)"),
):
    """🧬 Trigger continuous autonomous self-evolution."""
    ctx = resolve_local_context()
    if not ctx.is_git:
        console.print(
            "[red]Auto-evolution requires a Git checkout. Change to the repository "
            "directory or set SONIC_REPO_ROOT to a valid checkout.[/red]"
        )
        raise typer.Exit(code=1)

    from sonic_cli.repl import _run_continuous_auto_evolution

    try:
        asyncio.run(_run_continuous_auto_evolution(ctx, cycles=cycles))
    except KeyboardInterrupt:
        console.print("\n[yellow]Auto-evolution stopped.[/yellow]")
