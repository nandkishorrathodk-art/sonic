"""
SONIC CLI — Status Command
==========================
Local workspace git status & environment check.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

from sonic_cli.context import get_token, resolve_local_context

console = Console(legacy_windows=False)
API_BASE = os.environ.get("SONIC_SERVER_URL", "http://localhost:12000")


def check_git_status(repo_root: Path) -> dict[str, object]:
    """Inspect local git branch and dirty status."""
    try:
        p = subprocess.run(
            ["git", "status", "-s"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [line.strip() for line in p.stdout.splitlines() if line.strip()]
        return {"dirty": len(lines) > 0, "dirty_count": len(lines), "summary": lines[:5]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"dirty": False, "dirty_count": 0, "error": str(exc)}


def check_git_version() -> str:
    """Get git CLI version if installed."""
    if not shutil.which("git"):
        return "not installed"
    try:
        p = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        return p.stdout.strip()
    except Exception:
        return "installed (unknown version)"


def check_backend_health(server_url: str, token: str = "") -> dict[str, object]:
    """Ping backend /health endpoint."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=3.0, headers=headers) as client:
            resp = client.get(f"{server_url.rstrip('/')}/health")
            if resp.status_code == 200:
                data = resp.json()
                return {"online": True, "status": data.get("status", "healthy"), "data": data}
            return {"online": False, "status": f"HTTP {resp.status_code}", "error": resp.text}
    except Exception as exc:
        return {"online": False, "status": "offline", "error": str(exc)}


def status_command(
    server: str = typer.Option(API_BASE, "--server", "-s", help="Backend server URL"),
    token: str = typer.Option("", "--token", "-t", help="JWT Auth token"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """📊 Local workspace git status & environment check."""
    ctx = resolve_local_context()
    resolved_token = get_token(token)

    git_info = check_git_status(ctx.repo_root) if ctx.is_git else {"dirty": False, "dirty_count": 0}
    git_ver = check_git_version()
    backend_info = check_backend_health(server, resolved_token)

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_venv = sys.prefix != sys.base_prefix
    env_data = {
        "workspace": {
            "root": str(ctx.repo_root),
            "is_git": ctx.is_git,
            "branch": ctx.branch if ctx.is_git else None,
            "dirty_files": git_info.get("dirty_count", 0),
            "tenant_id": ctx.tenant_id,
            "workspace_id": ctx.workspace_id,
        },
        "environment": {
            "python": py_version,
            "executable": sys.executable,
            "virtual_env": is_venv,
            "git": git_ver,
            "backend_url": server,
            "backend_online": backend_info.get("online", False),
            "backend_status": backend_info.get("status"),
        },
    }

    if json_output:
        console.print(json.dumps(env_data, indent=2))
        return

    # Rich formatted output
    branch_display = f"[cyan]{ctx.branch}[/cyan]" if ctx.is_git else "[dim]non-git[/dim]"
    dirty_count = git_info.get("dirty_count", 0)
    dirty_display = f"[yellow]{dirty_count} modified files[/yellow]" if dirty_count else "[green]clean[/green]"

    ws_table = Table(title="Local Workspace Status", border_style="cyan")
    ws_table.add_column("Property", style="bold")
    ws_table.add_column("Value")
    ws_table.add_row("Root", str(ctx.repo_root))
    ws_table.add_row("Branch", branch_display)
    ws_table.add_row("Git Status", dirty_display)
    ws_table.add_row("Workspace ID", f"[dim]{ctx.workspace_id}[/dim]")
    ws_table.add_row("Tenant ID", f"[dim]{ctx.tenant_id}[/dim]")

    env_table = Table(title="Environment Check", border_style="magenta")
    env_table.add_column("Component", style="bold")
    env_table.add_column("Status")
    env_table.add_row("Python", f"{py_version} ({'venv' if is_venv else 'global'})")
    env_table.add_row("Git", git_ver)

    be_status = (
        "[green]🟢 online (ready)[/green]"
        if backend_info.get("online")
        else f"[yellow]🟡 {backend_info.get('status')}[/yellow]"
    )
    env_table.add_row("Backend API", f"{server} ({be_status})")

    console.print(ws_table)
    console.print(env_table)
