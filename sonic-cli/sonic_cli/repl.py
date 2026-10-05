"""Devin-Style Zero-Flag Interactive REPL for SONIC.

Belongs to Department: platform_systems, Team: api_services, Subsystem: cli.
Provides zero-flag interactive prompt, real-time streaming, diff viewer,
and continuous Auto-Evolution mode (/auto).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from sonic_cli.context import LocalWorkspaceContext, get_token, resolve_local_context

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

console = Console(legacy_windows=False)

SLASH_COMMANDS = [
    "/auto",
    "/diff",
    "/status",
    "/journal",
    "/clear",
    "/help",
    "/exit",
    "/quit",
]


def render_banner(context: LocalWorkspaceContext, backend_online: bool) -> None:
    status_color = "green" if backend_online else "yellow"
    status_text = "Backend Online (12000)" if backend_online else "Offline / Local Standalone"

    banner_text = (
        f"[bold cyan]  ⚡\n"
        f" █▀▀▀▀█  SONIC CLI\n"
        f" █▄▄▄▄█  Autonomous AI Digital Colleague[/bold cyan]\n\n"
        f" [dim]Workspace:[/dim] [white]{context.repo_root}[/white] ([cyan]git: {context.branch}[/cyan])\n"
        f" [dim]Status:   [/dim] [{status_color}]● {status_text}[/{status_color}]\n\n"
        f"[dim]Type your goal naturally, or use slash commands:\n"
        f"  [bold cyan]/auto[/bold cyan]     Engage continuous autonomous self-evolution loop\n"
        f"  [bold cyan]/diff[/bold cyan]     View syntax-highlighted git diff\n"
        f"  [bold cyan]/journal[/bold cyan]  View recent entries from evolution.md\n"
        f"  [bold cyan]/status[/bold cyan]   Inspect system health & being state\n"
        f"  [bold cyan]/exit[/bold cyan]     Quit session[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan", padding=(0, 2)))


def check_backend(server: str, token: str) -> bool:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(base_url=server, headers=headers, timeout=2.0) as client:
            res = client.get("/health")
            return res.status_code == 200
    except Exception:
        return False


def display_git_diff(repo_root: Path) -> None:
    try:
        proc = subprocess.run(
            ["git", "diff"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        diff_text = proc.stdout.strip()
        if not diff_text:
            # Check cached diff
            proc_cached = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff_text = proc_cached.stdout.strip()

        if not diff_text:
            console.print("[green]Working tree is clean. Zero uncommitted changes.[/green]")
            return

        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="[bold]Git Diff[/bold]", border_style="yellow"))
    except Exception as exc:
        console.print(f"[red]Failed to retrieve git diff: {exc}[/red]")


def display_journal(repo_root: Path) -> None:
    journal_path = repo_root / "evolution.md"
    if not journal_path.exists():
        console.print("[dim]No evolution.md journal found in active repository.[/dim]")
        return
    try:
        content = journal_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        recent = lines[-40:] if len(lines) > 40 else lines
        console.print(Panel("\n".join(recent), title="[bold]Recent evolution.md entries[/bold]", border_style="green"))
    except Exception as exc:
        console.print(f"[red]Could not read evolution.md: {exc}[/red]")


def print_help() -> None:
    table = Table(title="Available Commands", border_style="cyan")
    table.add_column("Command / Action", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_row("/auto", "Start Continuous Autonomous Self-Evolution Engine")
    table.add_row("/diff", "Display syntax-highlighted git diff of working tree")
    table.add_row("/journal", "Read recent entries from evolution.md")
    table.add_row("/status", "Check backend health and connection")
    table.add_row("/clear", "Clear terminal screen")
    table.add_row("/exit, /quit", "Exit the interactive CLI")
    table.add_row("<Any Text>", "Natural language goal (e.g. 'fix test in test_api.py')")
    console.print(table)


def run_continuous_auto_mode(context: LocalWorkspaceContext) -> None:
    """Execute the real continuous self-evolution loop with live UI streaming."""
    console.print(Panel(
        "[bold green]⚡ AUTO-EVOLUTION CONTINUOUS ENGINE ENGAGED[/bold green]\n"
        f"[dim]Target: {context.repo_root} | Press Ctrl+C anytime to pause.[/dim]",
        border_style="green",
    ))

    try:
        from sonic.evolution.continuous_runner import ContinuousAutoEvolutionRunner
    except ImportError as e:
        console.print(f"[red]Cannot load evolution engine: {e}[/red]")
        return

    def on_event(ev: dict[str, Any]) -> None:
        ev_type = ev.get("type", "")
        msg = ev.get("message", "")
        if ev_type == "started":
            console.print(f"[bold cyan]› {msg}[/bold cyan]")
        elif ev_type == "cycle_start":
            console.print(f"\n[bold yellow]{msg}[/bold yellow]")
        elif ev_type == "signal_found":
            console.print(f"  [cyan]{msg}[/cyan]")
        elif ev_type == "testing_worktree":
            console.print(f"  [dim]{msg}[/dim]")
        elif ev_type == "promoting":
            console.print(f"  [bold yellow]{msg}[/bold yellow]")
        elif ev_type == "cycle_success":
            console.print(f"  [bold green]{msg}[/bold green]")
        elif ev_type == "cycle_idle":
            console.print(f"  [dim]{msg}[/dim]")
        elif ev_type in ("cycle_error", "promote_error"):
            console.print(f"  [bold red]⚠️  {msg}[/bold red]")
        elif ev_type == "paused":
            console.print(f"\n[yellow]⏸ {msg}[/yellow]")

    runner = ContinuousAutoEvolutionRunner(
        repo_root=context.repo_root,
        tenant_id=context.tenant_id,
        max_cycles=10,
        interval_seconds=3,
        on_event=on_event,
    )

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        runner.stop()
        console.print("[yellow]Auto-evolution paused. Returning to prompt.[/yellow]")
    except Exception as exc:
        console.print(f"[red]Auto-evolution error: {exc}[/red]")


def dispatch_natural_language_goal(
    prompt: str,
    context: LocalWorkspaceContext,
    server: str,
    token: str,
) -> None:
    """Send natural language goal to backend and poll real-time progress."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {
        "prompt": prompt,
        "session_id": context.session_id,
        "mode": "autonomous",
        "target_repo": str(context.repo_root),
    }

    try:
        with httpx.Client(base_url=server, headers=headers, timeout=15.0) as client:
            res = client.post("/workstation/prompt", json=payload)
            if res.status_code != 200:
                console.print(f"[yellow]Server returned HTTP {res.status_code}: {res.text}[/yellow]")
                return

        last_log_count = 0
        with console.status("[bold cyan]SONIC is reasoning & executing... (Press Ctrl+C to pause)[/bold cyan]") as spinner:
            while True:
                try:
                    with httpx.Client(base_url=server, headers=headers, timeout=5.0) as client:
                        state_res = client.get("/workstation/state", params={"session_id": context.session_id})
                        if state_res.status_code == 200:
                            state = state_res.json()
                            worklog = state.get("worklog", [])
                            action = state.get("current_action", "Executing...")
                            spinner.update(f"[bold cyan]SONIC: {action}[/bold cyan]")

                            if len(worklog) > last_log_count:
                                for item in worklog[last_log_count:]:
                                    title = item.get("title", "Update")
                                    content = (item.get("content") or "").strip()
                                    console.print(f"  [bold yellow]›[/bold yellow] [bold]{title}:[/bold] {content}")
                                last_log_count = len(worklog)

                            status = state.get("status", "IDLE")
                            if status not in ("RUNNING", "QUEUED"):
                                console.print(f"[bold green]✓ Completed with status: {status}[/bold green]")
                                break
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    console.print("\n[yellow]⚠️  Interrupting execution...[/yellow]")
                    try:
                        with httpx.Client(base_url=server, headers=headers, timeout=5.0) as client:
                            client.post("/workstation/session/interrupt", params={"session_id": context.session_id})
                    except Exception:
                        pass
                    break
                except Exception:
                    time.sleep(1.0)

    except Exception as exc:
        console.print(f"[red]Failed to dispatch goal: {exc}[/red]")


def run_repl(server: str = "http://localhost:12000", token: str = "") -> None:
    """Launch the zero-flag Devin-style interactive prompt session."""
    context = resolve_local_context()
    resolved_token = get_token(token)
    online = check_backend(server, resolved_token)

    render_banner(context, online)

    history_file = Path.home() / ".sonic" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PROMPT_TOOLKIT:
        completer = WordCompleter(SLASH_COMMANDS, ignore_case=True)
        history = FileHistory(str(history_file))
        session: PromptSession = PromptSession(history=history, completer=completer)

    while True:
        try:
            prompt_str = f"sonic [{context.branch}] › "
            if HAS_PROMPT_TOOLKIT:
                user_input = session.prompt(prompt_str).strip()
            else:
                user_input = console.input(f"[bold cyan]{prompt_str}[/bold cyan]").strip()

            if not user_input:
                continue

            # Command routing
            if user_input in ("/exit", "/quit", "exit", "quit"):
                console.print("[dim]Goodbye.[/dim]")
                break
            elif user_input == "/help":
                print_help()
            elif user_input == "/clear":
                console.clear()
                render_banner(context, check_backend(server, resolved_token))
            elif user_input == "/diff":
                display_git_diff(context.repo_root)
            elif user_input == "/journal":
                display_journal(context.repo_root)
            elif user_input == "/status":
                status = "Online (12000)" if check_backend(server, resolved_token) else "Offline / Standalone"
                console.print(f"[cyan]Backend:[/cyan] {server} ({status})")
                console.print(f"[cyan]Workspace:[/cyan] {context.repo_root} (git: {context.branch})")
                console.print(f"[cyan]Tenant:[/cyan] {context.tenant_id}")
            elif user_input in ("/auto", "auto"):
                run_continuous_auto_mode(context)
            else:
                dispatch_natural_language_goal(user_input, context, server, resolved_token)

        except KeyboardInterrupt:
            console.print("\n[dim]Press Ctrl+D or type /exit to quit.[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]Exiting SONIC CLI.[/dim]")
            break
