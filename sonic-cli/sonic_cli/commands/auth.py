"""
SONIC-REDA CLI — Auth Commands
=================================
Google OAuth login/logout from the terminal.
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

from sonic_cli.context import get_token

app = typer.Typer(help="🔐 Authentication commands")
console = Console()

API_BASE = os.environ.get("SONIC_SERVER_URL", "http://127.0.0.1:12000")
TOKEN_PATH = Path.home() / ".sonic" / "token"


def _save_token(token: str) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token, encoding="utf-8")


@app.command()
def login(
    server: str = typer.Option(API_BASE, "--server", "-s", help="Backend server URL"),
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Use the development-only local operator login instead of Google OAuth.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        help="Email for development login; prompted when omitted.",
    ),
):
    """Use development login to save a CLI token, or open the Google OAuth page."""
    if dev:
        login_email = email or typer.prompt("Development login email")
        try:
            response = httpx.post(
                f"{server.rstrip('/')}/auth/login",
                json={"email": login_email},
                timeout=15,
            )
            response.raise_for_status()
            access_token = response.json().get("access_token")
            if not isinstance(access_token, str) or not access_token:
                raise ValueError("Login response did not contain an access token")
            _save_token(access_token)
        except (httpx.HTTPError, ValueError) as exc:
            console.print(f"[red]Development login failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]Logged in as {login_email}; token saved locally.[/green]")
        return

    login_url = f"{server.rstrip('/')}/auth/google/login"
    console.print(
        Panel(
            f"[bold cyan]Opening Google Login in your browser...[/bold cyan]\n\n"
            f"[dim]For local CLI token setup in development, use:[/dim]\n"
            f"[cyan]sonic auth login --dev[/cyan]\n\n"
            f"[dim]If browser doesn't open, visit:[/dim]\n"
            f"[link={login_url}]{login_url}[/link]",
            title="🔐 SONIC-REDA Auth",
            border_style="cyan",
        )
    )
    webbrowser.open(login_url)


@app.command()
def logout():
    """Clear the locally saved CLI token."""
    try:
        TOKEN_PATH.unlink(missing_ok=True)
    except OSError as exc:
        console.print(f"[red]Could not remove the saved token: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print("[green]Local CLI token cleared.[/green]")


@app.command()
def whoami(
    server: str = typer.Option(API_BASE, "--server", "-s"),
):
    """Show current authenticated user."""
    token = get_token()
    if not token:
        console.print("[yellow]Not logged in. Run 'sonic auth login --dev' locally.[/yellow]")
        raise typer.Exit(code=1)
    try:
        response = httpx.get(
            f"{server.rstrip('/')}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[red]Authentication check failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print_json(data=response.json())
