"""
SONIC — CLI Main Entry Point
============================
Autonomous AI-Human Digital Being command-line interface and Devin-style interactive REPL.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.panel import Panel

try:
    from sonic import __version__
except ImportError:  # pragma: no cover - core not installed in CLI-only envs
    __version__ = "0.4.0"

from sonic_cli.commands import (
    auth,
    auto,
    diff,
    journal,
    status,
    test,
)
from sonic_cli.context import get_token, resolve_local_context

app = typer.Typer(
    name="sonic",
    help="SONIC — Autonomous AI-Human Digital Being (Developer CLI)",
    no_args_is_help=False,
    rich_markup_mode="rich",
)

console = Console(legacy_windows=False)

# ---- Register Clean Developer Commands ----
app.command(name="status")(status.status_command)
app.command(name="diff")(diff.diff_command)
app.command(name="test")(test.test_command)
app.command(name="auto")(auto.auto_command)
app.command(name="journal")(journal.journal_command)
app.add_typer(auth.app, name="auth", help="Authentication (OAuth & Session)")


# ============================================
# Root Callback (Zero-Flag Devin REPL Entry Point)
# ============================================

@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    server: str = typer.Option("http://localhost:12000", "--server", "-s", help="Backend server URL"),
    token: str = typer.Option("", "--token", "-t", help="JWT Auth token"),
):
    """⚡ SONIC — Autonomous AI-Human Digital Being.

    Run without arguments to launch the Devin-style interactive prompt session.
    """
    if ctx.invoked_subcommand is None:
        srv = server if isinstance(server, str) else "http://localhost:12000"
        tok = token if isinstance(token, str) else ""

        # Check if non-interactive piped input was supplied
        piped_text = ""
        try:
            if not sys.stdin.isatty():
                piped_text = sys.stdin.read().strip()
        except Exception:
            pass

        if piped_text:
            resolved_tok = get_token(tok)
            local_ctx = resolve_local_context()
            from sonic_cli.repl import _dispatch_prompt
            _dispatch_prompt(piped_text, local_ctx, server_url=srv, token=resolved_tok)
            return

        from sonic_cli.repl import run_repl
        run_repl(server_url=srv, token=tok)


# ============================================
# Version
# ============================================

@app.command()
def version():
    """Show SONIC version banner."""
    console.print(
        Panel(
            f"[bold cyan]⚡ SONIC[/bold cyan] v{__version__}\n"
            f"[dim]Autonomous AI-Human Digital Being[/dim]",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    app()
