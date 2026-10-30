"""SONIC Interactive Terminal REPL.

Belongs to Department: platform_systems, Team: api_services, Subsystem: cli.
Provides a clean, zero-flag interactive prompt for the SONIC Autonomous AI
Software Engineer operating directly on the local workspace repository.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

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
    """Display colored syntax-highlighted git diff of active repository."""
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
            proc_unstaged = subprocess.run(
                ["git", "diff"],
                cwd=str(ctx.repo_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff_text = proc_unstaged.stdout.strip()

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


def _handle_test(ctx: LocalWorkspaceContext, test_path: str = "") -> None:
    """Run pytest directly with formatted output."""
    cmd = [sys.executable, "-m", "pytest"]
    clean_path = test_path.strip()
    if clean_path:
        cmd.extend(clean_path.split())
    else:
        # Default test discovery if path not provided
        if (ctx.repo_root / "sonic-cli" / "tests").exists():
            cmd.append("sonic-cli/tests")
        elif (ctx.repo_root / "tests").exists():
            cmd.append("tests")

    target_display = " ".join(cmd[3:]) if len(cmd) > 3 else "workspace"
    console.print(f"[bold cyan]🧪 Running pytest directly ({target_display})...[/bold cyan]")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ctx.repo_root),
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = proc.stdout.strip()
        err = proc.stderr.strip()
        if output:
            console.print(output)
        if err and proc.returncode != 0:
            console.print(f"[yellow]{err}[/yellow]")

        if proc.returncode == 0:
            console.print("[bold green]✓ Pytest passed successfully.[/bold green]")
        else:
            console.print(f"[bold red]✗ Pytest exited with code {proc.returncode}.[/bold red]")
    except subprocess.TimeoutExpired:
        console.print("[red]Pytest execution timed out after 180 seconds.[/red]")
    except Exception as exc:
        console.print(f"[red]Failed to run pytest: {exc}[/red]")


def _handle_status(
    ctx: LocalWorkspaceContext,
    server_url: str = "http://127.0.0.1:12000",
    token: str = "",
) -> None:
    """Display workspace status, branch, dirty files, repo root, and environment keys."""
    branch_display = ctx.branch if ctx.is_git else "non-git"
    console.print(f"[bold]Root:[/bold] {ctx.repo_root}")
    console.print(
        f"[bold]Branch:[/bold] [cyan]{branch_display}[/cyan]  |  "
        f"[bold]Tenant:[/bold] {ctx.tenant_id}  |  "
        f"[bold]Workspace:[/bold] {ctx.workspace_id}"
    )

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
            dirty_lines = [line for line in p.stdout.splitlines() if line.strip()]
            console.print(f"[bold]Dirty files:[/bold] {len(dirty_lines)}")
            for line in dirty_lines[:5]:
                console.print(f"  [dim]{line}[/dim]")
            if len(dirty_lines) > 5:
                console.print(f"  [dim]... and {len(dirty_lines) - 5} more[/dim]")
        except (OSError, subprocess.SubprocessError) as exc:
            console.print(f"[yellow]Could not inspect Git status: {exc}[/yellow]")

    # Check environment keys
    keys_to_check = [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "NVIDIA_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    key_statuses: list[str] = []
    for k in keys_to_check:
        val = os.environ.get(k, "")
        if val:
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
            key_statuses.append(f"[green]{k}[/green]: [dim]set ({masked})[/dim]")
        else:
            key_statuses.append(f"[dim]{k}: not set[/dim]")

    console.print(f"[bold]Environment Keys:[/bold] {', '.join(key_statuses[:2])}")
    if len(key_statuses) > 2:
        console.print(f"                  {', '.join(key_statuses[2:])}")


def _handle_help() -> None:
    """Display interactive commands help."""
    console.print("[bold]Available commands:[/bold]")
    console.print("  [cyan]/auto [N][/cyan]     - Run continuous auto-evolution in isolated worktrees (or for N cycles)")
    console.print("  [cyan]/diff[/cyan]         - Inspect git diff in current repository")
    console.print("  [cyan]/journal[/cyan]      - View evolution journal and verification receipts")
    console.print("  [cyan]/test [path][/cyan]  - Run pytest directly with formatted output")
    console.print("  [cyan]/status[/cyan]       - Show git branch, dirty files, repo root, and environment keys")
    console.print("  [cyan]/clear[/cyan]        - Clear terminal screen")
    console.print("  [cyan]/help[/cyan]         - Show this command guide")
    console.print("  [cyan]/exit[/cyan]         - Exit SONIC CLI")
    console.print()
    console.print("[dim]Enter any natural language coding prompt to inspect, edit, or fix files directly.[/dim]")


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


GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hii",
    "hiii",
    "heyy",
    "hey there",
    "hello there",
    "hola",
    "yo",
    "sup",
    "wassup",
    "what's up",
    "whats up",
    "hlo",
    "helo",
    "namaste",
    "namaskar",
    "salaam",
    "pranam",
    "gm",
    "gn",
    "good morning",
    "good evening",
    "good afternoon",
    "good night",
}

FAREWELLS = {
    "by",
    "bye",
    "byee",
    "byeee",
    "bye bye",
    "byebye",
    "goodbye",
    "good bye",
    "cya",
    "see ya",
    "see you",
    "tata",
    "alvida",
    "catch you later",
    "ttyl",
}


def _is_greeting(prompt: str) -> bool:
    """Check if the prompt is a greeting (including typos and letter repetitions)."""
    cleaned = prompt.strip().lower().rstrip("!.,?~ ")
    if cleaned in GREETINGS:
        return True
    # Matches variations like hhi, hhii, hiiii, heyyy, heeeey, etc.
    if re.match(r"^h+[iey]+$", cleaned):
        return True
    if re.match(r"^h+e+l+o+$", cleaned):
        return True
    return False


def _is_farewell(prompt: str) -> bool:
    """Check if the prompt is a farewell or exit intent."""
    cleaned = prompt.strip().lower().rstrip("!.,?~ ")
    if cleaned in FAREWELLS:
        return True
    if re.match(r"^b+y+e*$", cleaned):
        return True
    return False


def _is_identity_query(prompt: str) -> bool:
    """Check if the user is asking who/what SONIC is or why it is here."""
    cleaned = re.sub(r"[^\w\s]", "", prompt.lower()).strip()
    patterns = [
        "who are you",
        "why you are hear",
        "why are you here",
        "why you are here",
        "what are you doing here",
        "what is your purpose",
        "what is your job",
        "what can you do",
        "what do you do",
        "what is sonic",
        "who is sonic",
        "tell me about yourself",
        "introduce yourself",
    ]
    return any(p in cleaned for p in patterns)


def _is_casual_or_playful(prompt: str) -> bool:
    """Detect non-coding casual chitchat or playful inquiries."""
    lowered = prompt.lower().strip()
    triggers = [
        "can you suck",
        "tell me a joke",
        "sing a song",
        "are you alive",
        "do you love me",
        "marry me",
    ]
    return any(t in lowered for t in triggers)


def _is_short_noise(prompt: str) -> bool:
    """Detect short non-actionable keyboard smash or gibberish (e.g. 'kjn', 'asdf')."""
    cleaned = prompt.strip().lower()
    if len(cleaned) <= 4 and re.match(r"^[a-z]+$", cleaned):
        vowels = set("aeiou")
        if not any(c in vowels for c in cleaned):
            return True
        if cleaned in {"asdf", "qwe", "qwer", "zxc", "jkl"}:
            return True
    return False


def _handle_greeting() -> None:
    """Respond conversationally as a helpful software engineer without unnecessary tool calls."""
    console.print(
        "👋 Hello! I am SONIC, your autonomous software engineer.\n"
        "I can inspect code, write and edit files, run tests, fix bugs, and evolve this codebase.\n"
        "Tell me what you'd like to build or fix, or type [cyan]/help[/cyan] to see available commands."
    )


def _dispatch_prompt(
    prompt: str,
    ctx: LocalWorkspaceContext,
    server_url: str = "http://127.0.0.1:12000",
    token: str = "",
) -> None:
    """Dispatch user prompt to the local DeveloperAgent, with conversational handling for greetings."""
    if _is_greeting(prompt):
        _handle_greeting()
        return

    if _is_farewell(prompt):
        console.print(
            "👋 Goodbye! Have a great time coding. Type [cyan]/exit[/cyan] or press Ctrl+D whenever you're ready to exit."
        )
        return

    if prompt.strip().lower() in {"exit", "quit"}:
        console.print("👋 Exiting SONIC CLI. Goodbye!")
        sys.exit(0)

    if _is_identity_query(prompt):
        console.print(
            "⚡ I am SONIC, an Autonomous AI Software Engineer operating directly on your local workspace.\n"
            "I use native tool calling to inspect files, edit code, run tests, and evolve this repository.\n"
            "Tell me what file or task you'd like to work on, or type [cyan]/help[/cyan] to see available commands."
        )
        return

    if _is_casual_or_playful(prompt):
        console.print(
            "🤖 I am an AI software engineer focused strictly on reading, writing, and testing code in this workspace.\n"
            "Please provide a software development task, or type [cyan]/help[/cyan] for available commands."
        )
        return

    if _is_short_noise(prompt):
        console.print(
            "💬 I didn't recognize that command or task. If you'd like to start, tell me what file or feature to work on (e.g. [italic]'explain main.py'[/italic] or [italic]'run tests'[/italic]), or type [cyan]/help[/cyan]."
        )
        return

    try:
        from sonic_cli.developer_agent import DeveloperAgent
    except ImportError as exc:
        console.print(
            f"[red]Could not import DeveloperAgent from sonic_cli.developer_agent: {exc}[/red]\n"
            "[yellow]Please verify sonic_cli is properly installed.[/yellow]"
        )
        return

    try:
        agent = DeveloperAgent(workspace_dir=ctx.repo_root, console=console)
        agent.run(prompt)
    except KeyboardInterrupt:
        console.print("\n[yellow]Task interrupted by operator (Ctrl+C).[/yellow]")
    except Exception as exc:
        console.print(f"[red]Task execution failed: {exc}[/red]")


def run_repl(
    server_url: str = "http://127.0.0.1:12000",
    token: str = "",
) -> None:
    """Start the interactive terminal REPL."""

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
            ["/auto", "/diff", "/journal", "/test", "/status", "/clear", "/help", "/exit", "/quit"],
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
            "Ask SONIC to inspect code, fix tests, implement features, or run /auto, /test"
        )

        def bottom_toolbar() -> str:
            branch = ctx.branch if ctx.is_git else "non-git workspace"
            return f"Branch: {branch} · Local AI Developer · /help"
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
        elif cmd == "/test" or cmd.startswith("/test "):
            test_path = cmd[5:].strip() if cmd.startswith("/test ") else ""
            _handle_test(ctx, test_path=test_path)
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
