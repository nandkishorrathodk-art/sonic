"""SONIC Interactive Terminal REPL.

Belongs to Department: platform_systems, Team: api_services, Subsystem: cli.
Provides a clean, zero-flag interactive prompt for the SONIC Autonomous AI-Human
Digital Being workstation.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

from sonic_cli.context import LocalWorkspaceContext, get_token, resolve_local_context

# Reconfigure console streams to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False)


def _print_welcome_banner(ctx: LocalWorkspaceContext) -> None:
    """Print the clean ASCII logo and status header."""
    if Path.cwd().resolve() == Path.home().resolve():
        console.print(
            "[yellow]⚠️  SONIC CLI works best when run in a project directory, "
            "not your home directory.[/yellow]"
        )

    # ASCII Logo + Title
    logo_lines = [
        "  .:|:.",
        " '\"':*|:*.  SONIC CLI",
        " .:::|:::  v0.3.0",
        "  ':|:'",
    ]
    console.print("\n".join(logo_lines))
    console.print()

    # Status / Capacity summary line
    branch_display = ctx.branch if ctx.is_git else "non-git workspace"
    status_line = (
        f"Autonomous AI-Human Digital Being · "
        f"Branch: [cyan]{branch_display}[/cyan] · "
        f"Workspace: [dim]{ctx.workspace_id}[/dim]"
    )
    console.print(status_line)
    console.print()


def _handle_diff(ctx: LocalWorkspaceContext) -> None:
    """Display git diff of active repository."""
    if not ctx.is_git:
        console.print("[yellow]Not in a git repository; no git diff available.[/yellow]")
        return
    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=str(ctx.repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        diff_text = proc.stdout.strip()
        if not diff_text:
            console.print("[dim]Working directory clean. No uncommitted diffs.[/dim]")
            return
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        console.print(syntax)
    except (OSError, subprocess.SubprocessError) as exc:
        console.print(f"[red]Failed to compute git diff: {exc}[/red]")


def _handle_journal(ctx: LocalWorkspaceContext) -> None:
    """Display evolution.md journal or candidate logs."""
    journal_path = ctx.repo_root / "evolution.md"
    if journal_path.exists():
        try:
            content = journal_path.read_text(encoding="utf-8")
            console.print(Markdown(content))
            return
        except Exception as exc:
            console.print(f"[red]Failed to read evolution.md: {exc}[/red]")
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


def _handle_status(
    ctx: LocalWorkspaceContext,
    server_url: str = "http://127.0.0.1:12000",
    token: str = "",
) -> None:
    """Display workspace status, branch, and backend connectivity."""
    branch_display = ctx.branch if ctx.is_git else "non-git"
    console.print(f"[bold]Root:[/bold] {ctx.repo_root}")
    console.print(f"[bold]Branch:[/bold] [cyan]{branch_display}[/cyan]  |  [bold]Tenant:[/bold] {ctx.tenant_id}  |  [bold]Workspace:[/bold] {ctx.workspace_id}")

    # Check uncommitted git status
    if ctx.is_git:
        try:
            p = subprocess.run(
                ["git", "status", "-s"],
                cwd=str(ctx.repo_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            dirty_count = len([line for line in p.stdout.splitlines() if line.strip()])
            console.print(f"[bold]Dirty files:[/bold] {dirty_count}")
        except (OSError, subprocess.SubprocessError) as exc:
            console.print(f"[yellow]Could not inspect Git status: {exc}[/yellow]")

    # Backend ping
    backend_status = "[yellow]offline[/yellow]"
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(timeout=2.0, headers=headers) as client:
            resp = client.get(f"{server_url}/health")
            if resp.status_code == 200:
                backend_status = "[green]online (ready)[/green]"
    except httpx.HTTPError as exc:
        backend_status = f"[red]unavailable ({exc})[/red]"

    console.print(f"[bold]Backend API:[/bold] {server_url} ({backend_status})")


def _handle_help() -> None:
    """Display interactive commands help."""
    console.print("[bold]Available commands:[/bold]")
    console.print("  [cyan]/auto [N][/cyan]    - Run until stopped, or for N cycles")
    console.print("  [cyan]/diff[/cyan]        - Inspect git diff in current repository")
    console.print("  [cyan]/journal[/cyan]     - View evolution journal and verification receipts")
    console.print("  [cyan]/status[/cyan]      - Show workspace status, branch, and backend health")
    console.print("  [cyan]/clear[/cyan]       - Clear terminal screen")
    console.print("  [cyan]/help[/cyan]        - Show this help")
    console.print("  [cyan]/exit[/cyan]        - Exit SONIC CLI")


async def _run_continuous_auto_evolution(
    ctx: LocalWorkspaceContext,
    cycles: int = 0,
) -> None:
    """Run verified local evolution cycles; zero means continue until interrupted."""
    if not ctx.is_git:
        console.print(
            "[red]Auto-evolution requires a Git checkout. Change to the repository "
            "directory or set SONIC_REPO_ROOT to a valid checkout.[/red]"
        )
        return

    cycle_label = "until interrupted" if cycles == 0 else f"for {cycles} cycles"
    console.print(f"[bold cyan]Initiating Continuous Auto-Evolution Loop {cycle_label}...[/bold cyan]")
    try:
        from sonic.evolution.continuous_runner import ContinuousAutoEvolutionRunner

        runner: ContinuousAutoEvolutionRunner | None = None

        def on_event(event: dict[str, object]) -> None:
            event_type = str(event.get("type", ""))
            message = str(event.get("message", ""))
            style = (
                "green" if event_type == "cycle_success"
                else "red" if event_type in {"cycle_error", "promote_error"}
                else "cyan" if event_type in {"started", "signal_found"}
                else "dim"
            )
            console.print(f"[{style}]{message}[/{style}]")

        runner = ContinuousAutoEvolutionRunner(
            repo_root=ctx.repo_root,
            tenant_id=ctx.tenant_id,
            max_cycles=cycles,
            interval_seconds=3,
            on_event=on_event,
        )
        await runner.run()
    except KeyboardInterrupt:
        if runner is not None:
            runner.stop()
        console.print("[yellow]Auto-evolution paused; returning to the prompt.[/yellow]")
    except ImportError:
        console.print(
            "[yellow]sonic.evolution.continuous_runner not importable in current environment. "
            "Please ensure sonic-core is installed.[/yellow]"
        )
    except Exception as exc:
        console.print(f"[red]Error during auto-evolution: {exc}[/red]")


def _dispatch_prompt(
    prompt: str,
    ctx: LocalWorkspaceContext,
    server_url: str = "http://127.0.0.1:12000",
    token: str = "",
) -> None:
    """Dispatch a goal to the backend, with local handling for simple greetings."""
    lowered = prompt.strip().lower()

    if lowered in {"hi", "hello", "hey", "hi!", "hello!"}:
        console.print("Hi! How can I help you today?")
        return
    if lowered in {"who are you", "who are you?", "what is sonic", "what is sonic?"}:
        console.print(
            "I am SONIC, an Autonomous AI-Human Digital Being that can help with "
            "computer-use and software tasks."
        )
        return

    token = get_token(token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {
        "prompt": prompt,
        "session_id": ctx.session_id,
        "mode": "autonomous",
        "target_repo": str(ctx.repo_root),
    }

    try:
        with httpx.Client(base_url=server_url, timeout=60.0) as client:
            resp = client.post("/workstation/prompt", json=payload, headers=headers)
            if resp.status_code != 200:
                console.print(f"[yellow]Backend responded with status {resp.status_code}: {resp.text}[/yellow]")
                return

            data = resp.json()
            status = str(data.get("status", "")).upper()
            if status not in {"ACCEPTED", "RUNNING", "QUEUED"}:
                console.print(data.get("response") or data.get("message") or str(data))
                return

        last_log_count = 0
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            with httpx.Client(base_url=server_url, timeout=5.0) as client:
                state_resp = client.get(
                    "/workstation/state",
                    params={"session_id": ctx.session_id},
                    headers=headers,
                )
            if state_resp.status_code != 200:
                console.print(
                    f"[yellow]Could not read task progress "
                    f"(HTTP {state_resp.status_code}): {state_resp.text}[/yellow]"
                )
                return
            state = state_resp.json()
            worklog = state.get("worklog", [])
            for item in worklog[last_log_count:]:
                title = item.get("title", "Update")
                content = (item.get("content") or "").strip()
                console.print(f"[yellow]›[/yellow] [bold]{title}:[/bold] {content}")
            last_log_count = len(worklog)
            task_status = str(state.get("status", "")).upper()
            if task_status not in {"RUNNING", "QUEUED"}:
                console.print(f"[bold]Task status: {task_status or 'unknown'}[/bold]")
                return
            time.sleep(0.5)
        console.print("[yellow]Stopped waiting for task progress after 30 minutes.[/yellow]")
    except httpx.ConnectError:
        console.print(
            f"[dim]Backend server is not running on {server_url}.[/dim]\n"
            f"[dim]To start the backend server, run in another terminal:[/dim]\n"
            f"  [cyan]APP_ENV=development python -m uvicorn sonic.api.main:app --port 12000[/cyan]\n"
            f"[dim]Or run autonomous self-evolution directly with: [cyan]/auto[/cyan][/dim]"
        )
    except KeyboardInterrupt:
        try:
            with httpx.Client(base_url=server_url, timeout=5.0) as client:
                response = client.post(
                    "/workstation/session/interrupt",
                    params={"session_id": ctx.session_id},
                    headers=headers,
                )
            if response.status_code == 200:
                console.print("[yellow]Interrupt sent to the active workstation task.[/yellow]")
            else:
                console.print(
                    f"[yellow]Task observation stopped; interrupt returned "
                    f"HTTP {response.status_code}: {response.text}[/yellow]"
                )
        except httpx.HTTPError as exc:
            console.print(f"[yellow]Task observation stopped; interrupt failed: {exc}[/yellow]")
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to dispatch task: {exc}[/red]")
    except (ValueError, KeyError) as exc:
        console.print(f"[red]Failed to dispatch task: {exc}[/red]")


def run_repl(
    server_url: str = "http://127.0.0.1:12000",
    token: str = "",
) -> None:
    """Start the interactive terminal REPL."""
    import asyncio

    ctx = resolve_local_context()
    resolved_token = get_token(token)
    _print_welcome_banner(ctx)

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style
    except ImportError:
        PromptSession = None

    history_file = Path.home() / ".sonic" / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    if PromptSession is not None:
        commands_completer = WordCompleter(
            ["/auto", "/diff", "/journal", "/status", "/clear", "/help", "/exit", "/quit"],
            ignore_case=True,
        )
        style = Style.from_dict({"prompt": "bold", "bottom-toolbar": "dim"})
        session = PromptSession[str](
            message="> ",
            completer=commands_completer,
            history=FileHistory(str(history_file)),
            style=style,
        )
        placeholder_text = (
            "Ask SONIC to build features, evolve codebase, operate desktop, or run /auto"
        )

        def bottom_toolbar() -> str:
            branch = ctx.branch if ctx.is_git else "non-git workspace"
            return f"Branch: {branch} · Backend: {server_url} · /help"
    else:
        session = None
        placeholder_text = ""
        bottom_toolbar = None
        console.print(
            "[yellow]prompt_toolkit is unavailable; using the basic terminal prompt.[/yellow]"
        )

    while True:
        try:
            if session is None:
                user_input = input("> ")
            else:
                user_input = session.prompt(
                    placeholder=placeholder_text,
                    bottom_toolbar=bottom_toolbar,
                )
        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit or Ctrl+D to quit.[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break

        cmd = user_input.strip()
        if not cmd:
            continue
        if cmd in {"/exit", "/quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        if cmd == "/clear":
            console.clear()
            _print_welcome_banner(ctx)
        elif cmd == "/diff":
            _handle_diff(ctx)
        elif cmd == "/journal":
            _handle_journal(ctx)
        elif cmd == "/status":
            _handle_status(ctx, server_url=server_url, token=resolved_token)
        elif cmd == "/help":
            _handle_help()
        elif cmd == "/auto" or cmd.startswith("/auto "):
            parts = cmd.split()
            if len(parts) > 2 or (
                len(parts) == 2 and (not parts[1].isdigit() or int(parts[1]) < 1)
            ):
                console.print("[yellow]Usage: /auto [positive cycle count][/yellow]")
                continue
            cycles = int(parts[1]) if len(parts) == 2 else 0
            asyncio.run(_run_continuous_auto_evolution(ctx, cycles=cycles))
        elif cmd.startswith("/"):
            console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]")
        else:
            _dispatch_prompt(cmd, ctx, server_url=server_url, token=resolved_token)
