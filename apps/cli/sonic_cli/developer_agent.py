"""
SONIC Autonomous AI Software Engineer Engine (DeveloperAgent).
=============================================================
An autonomous AI coding engine operating directly on the local workspace repository,
inspired by Antigravity and Devin.

Features:
- Native OpenAI/Groq function-calling tool loop.
- Direct local tools: view_file, write_file, edit_file, run_command, git_diff, run_pytest.
- Default model: Groq qwen/qwen3.8-27b with flexible multi-provider support.
- Rich progress printing with streaming thought/reasoning visualization.
- Graceful cancellation on KeyboardInterrupt (Ctrl+C).
- Zero dependencies on external orchestrators, container daemons, or graphical workstations.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel

try:
    from dotenv import find_dotenv, load_dotenv

    # Search for .env in current directory and parent directories
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Default system prompt for the autonomous developer agent
DEFAULT_DEVELOPER_SYSTEM_PROMPT = f"""You are SONIC Developer Agent, an autonomous AI Software Engineer created by Nandkishor Rathod at Parth Lab, operating directly on the local repository workspace.
You solve coding, debugging, refactoring, and engineering tasks with rigor, precision, and verification.
Founder & Creator: Nandkishor Rathod
Organization: Parth Lab
Current OS: {sys.platform} ({'Windows' if sys.platform == 'win32' else 'Linux/POSIX'})

Operational Guidelines:
- Grounding First: When asked to modify or inspect code, inspect files (`view_file`) or search the workspace first. Never guess contents.
- Conversational Clarity: If the user is asking questions, chatting, or requesting explanations without asking to modify code, answer directly and helpfully without calling tools.
- Surgical Edits: When editing files, view the exact lines first. Ensure `target_content` has exact indentation and whitespace.
- Platform Awareness: Use `view_file` to list directories and view files. On Windows, do NOT use Linux commands like `ls`, `grep`, or `find`; use python, git, or pytest commands.
- Test and Verify: Run relevant tests or pytest (`run_pytest`, `run_command`) after making changes to verify correctness.
- Diff Verification: Review `git_diff()` before concluding your task.
- Conclude Cleanly: Once the objective is completed and verified, summarize what was achieved without making further tool calls.
"""

# Native OpenAI / Groq tool schemas
TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View contents of a file in the workspace repository, or list directory entries. Optionally specify line ranges (1-indexed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory (relative to workspace or absolute).",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional start line number (1-indexed, inclusive).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional end line number (1-indexed, inclusive).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file in the workspace with given content. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to create or overwrite.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing a unique target_content block with replacement_content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "target_content": {
                        "type": "string",
                        "description": "Exact text block in the file to replace. Must appear exactly once in the file.",
                    },
                    "replacement_content": {
                        "type": "string",
                        "description": "The replacement text block.",
                    },
                },
                "required": ["path", "target_content", "replacement_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command locally in the workspace directory and return its stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command line to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum seconds to wait before timing out (default 30).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Inspect git diff of uncommitted changes and working tree status in the local workspace repository.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pytest",
            "description": "Run the pytest test suite in the workspace or run a specific test path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "Optional specific test file or directory path (e.g. 'tests/test_cli.py').",
                    },
                },
            },
        },
    },
]


# ==============================================================================
# Output Formatting & Security Helpers
# ==============================================================================


def _truncate_output(text: str, max_lines: int = 80, max_chars: int = 4000) -> str:
    """Safely truncate excessive tool output to prevent LLM context window overflow."""
    if not text or not isinstance(text, str):
        return text
    lines = text.splitlines()
    if len(lines) > max_lines:
        keep_head = max(10, max_lines - 30)
        keep_tail = 20
        truncated_count = len(lines) - (keep_head + keep_tail)
        trimmed = (
            lines[:keep_head]
            + [f"\n... (+{truncated_count} lines truncated for context budget) ...\n"]
            + lines[-keep_tail:]
        )
        text = "\n".join(trimmed)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (+{len(text) - max_chars} characters truncated) ..."
    return text


def sanitize_secrets(text: str, extra_secrets: list[str] | None = None) -> str:
    """Mask sensitive API keys, secrets, and authorization tokens to prevent leakage in outputs/logs."""
    if not text or not isinstance(text, str):
        return text

    sanitized = text

    secret_env_vars = [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "NVIDIA_API_KEY",
        "OPENROUTER_API_KEY",
        "LOCAL_API_KEY",
        "JWT_SECRET",
        "SECRET_KEY",
        "SONIC_TOKEN",
    ]
    secrets_to_mask: set[str] = set()
    for var in secret_env_vars:
        val = os.environ.get(var, "").strip()
        if val and len(val) >= 6:
            secrets_to_mask.add(val)

    if extra_secrets:
        for s in extra_secrets:
            s_str = str(s).strip()
            if s_str and len(s_str) >= 6:
                secrets_to_mask.add(s_str)

    # Mask exact values first (longest first to handle sub-tokens cleanly)
    for sec in sorted(secrets_to_mask, key=len, reverse=True):
        sanitized = sanitized.replace(sec, "[REDACTED_API_KEY]")

    # Mask known API key patterns
    sanitized = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "gsk_[REDACTED_KEY]", sanitized)
    sanitized = re.sub(r"sk-(?:proj-)?[a-zA-Z0-9_-]{20,}", "sk-[REDACTED_KEY]", sanitized)
    sanitized = re.sub(r"nvapi-[a-zA-Z0-9_-]{20,}", "nvapi-[REDACTED_KEY]", sanitized)
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{15,}", r"\1[REDACTED_TOKEN]", sanitized, flags=re.IGNORECASE)

    return sanitized


def _resolve_workspace_path(path: str | Path, workspace_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to workspace_dir or cwd, ensuring it remains confined within workspace root."""
    path_str = str(path).strip()
    if not path_str:
        raise ValueError("Path cannot be empty.")

    base = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
    target = Path(path_str)
    resolved = target.resolve() if target.is_absolute() else (base / target).resolve()

    try:
        resolved.relative_to(base)
    except ValueError:
        raise PermissionError(
            f"Access denied: path '{path}' escapes repository workspace root '{base}'."
        ) from None
    return resolved


# ==============================================================================
# Standalone Local Tools
# ==============================================================================


def view_file(
    path: str | Path,
    start_line: int | None = None,
    end_line: int | None = None,
    workspace_dir: str | Path | None = None,
) -> str:
    """View contents of a file or list directory entries with path confinement."""
    try:
        target = _resolve_workspace_path(path, workspace_dir)
    except (PermissionError, ValueError) as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error resolving path '{path}': {exc}"

    if not target.exists():
        return f"Error: File not found at '{path}'. Check the path or list parent directory with view_file."

    if target.is_dir():
        try:
            entries = []
            for item in sorted(target.iterdir()):
                entries.append(item.name + ("/" if item.is_dir() else ""))
            preview = entries[:100]
            count_info = f" ({len(entries)} items)" if len(entries) > 100 else ""
            return f"Directory listing for '{path}'{count_info}:\n" + "\n".join(preview)
        except Exception as exc:
            return f"Error reading directory '{path}': {exc}"

    try:
        raw_bytes = target.read_bytes()
        # Binary check: null bytes in first 1024 bytes
        if b"\0" in raw_bytes[:1024]:
            return f"Error: '{path}' appears to be a binary file ({len(raw_bytes)} bytes) and cannot be viewed as text."

        content = raw_bytes.decode("utf-8", errors="replace")
        lines = content.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return f"(File '{path}' is empty, 0 lines)"

        # Check bounds: end_line < start_line first, then start_line > total_lines
        if start_line is not None and end_line is not None and int(end_line) < int(start_line):
            return f"Error: end_line ({end_line}) cannot be less than start_line ({start_line})"
        if start_line is not None and int(start_line) > total_lines:
            return f"Error: start_line {start_line} exceeds total lines ({total_lines}) in '{path}'"

        s = int(start_line) if start_line is not None else 1
        e = int(end_line) if end_line is not None else total_lines

        if s < 1:
            s = 1
        if e > total_lines:
            e = total_lines

        formatted_lines = [f"{i:4d} | {lines[i - 1]}" for i in range(s, e + 1)]
        header = f"File: {path} (lines {s}-{e} of {total_lines}):"
        return header + "\n" + "\n".join(formatted_lines)
    except Exception as exc:
        return f"Error reading file '{path}': {exc}"


def write_file(
    path: str | Path,
    content: str,
    workspace_dir: str | Path | None = None,
) -> str:
    """Write or overwrite a file in the workspace with given content, confined to workspace root."""
    try:
        target = _resolve_workspace_path(path, workspace_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        line_count = len(content.splitlines())
        byte_count = len(content.encode("utf-8"))
        return f"Successfully wrote {line_count} lines ({byte_count} bytes) to '{path}'"
    except (PermissionError, ValueError) as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error writing file '{path}': {exc}"


def edit_file(
    path: str | Path,
    target_content: str,
    replacement_content: str,
    workspace_dir: str | Path | None = None,
) -> str:
    """Edit an existing file by replacing a unique target_content block with replacement_content."""
    try:
        target = _resolve_workspace_path(path, workspace_dir)
    except (PermissionError, ValueError) as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error resolving path '{path}': {exc}"

    if not target.exists():
        return (
            f"Error: File not found at '{path}'. "
            f"Please verify the file path using view_file, or use write_file to create a new file."
        )

    if target.is_dir():
        return f"Error: '{path}' is a directory, not a file. Use view_file to inspect its contents."

    if not target_content:
        return "Error: target_content cannot be empty. Please provide the exact text block to replace."

    try:
        # Preserve the original bytes outside the edit and refuse lossy decoding.
        with target.open("r", encoding="utf-8", newline="") as stream:
            text = stream.read()

        # Match either newline convention without rewriting the rest of the file.
        target_lf = target_content.replace("\r\n", "\n")
        pattern = r"\r?\n".join(re.escape(line) for line in target_lf.split("\n"))
        matches = list(re.finditer(pattern, text))

        if len(matches) == 1:
            match = matches[0]
            newline = "\r\n" if "\r\n" in text else "\n"
            replacement = replacement_content.replace("\r\n", "\n").replace("\n", newline)
            new_text = text[:match.start()] + replacement + text[match.end():]
            with target.open("w", encoding="utf-8", newline="") as stream:
                stream.write(new_text)
            return f"Successfully edited '{path}'. Replaced unique occurrence of target content."
        elif len(matches) > 1:
            return (
                f"Error: target_content occurs {len(matches)} times in '{path}'. "
                f"Please include more surrounding context lines to provide unique surrounding context. "
                f"Use view_file('{path}') to inspect surrounding lines."
            )

        return (
            f"Error: target_content not found in '{path}'. "
            f"Please call view_file('{path}') to inspect the current file content, whitespace, and indentation, "
            f"then retry edit_file with the exact matching text block."
        )
    except Exception as exc:
        return f"Error editing file '{path}': {exc}"


def run_command(
    command: str,
    timeout: int = 30,
    workspace_dir: str | Path | None = None,
) -> str:
    """Execute a shell command locally in workspace_dir with robust timeout and process cleanup."""
    cwd = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
    timeout_sec = int(timeout) if timeout else 30
    sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=sub_env,
            )
        else:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
                env=sub_env,
            )

        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            # Cleanly terminate process tree to avoid hanging orphan processes
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    proc.kill()
            else:
                try:
                    import signal

                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()

            stdout, stderr = proc.communicate()
            parts = [f"Error: Command timed out after {timeout_sec} seconds: {command}"]
            if stdout:
                parts.append(f"STDOUT (partial):\n{stdout}")
            if stderr:
                parts.append(f"STDERR (partial):\n{stderr}")
            return "\n".join(parts)

        parts = [f"Exit code: {proc.returncode}"]
        if stdout:
            parts.append(f"STDOUT:\n{_truncate_output(stdout)}")
        if stderr:
            parts.append(f"STDERR:\n{_truncate_output(stderr)}")
        if not stdout and not stderr:
            parts.append("(No command output)")
        return "\n".join(parts)
    except Exception as exc:
        return f"Error executing command: {exc}"


def git_diff(
    workspace_dir: str | Path | None = None,
) -> str:
    """Inspect git diff and status of uncommitted changes in the repository."""
    cwd = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
    try:
        # Check if git is available and cwd is a git repo
        diff_proc = subprocess.run(
            ["git", "diff", "HEAD"],
            shell=False,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
        diff_text = diff_proc.stdout.strip()

        if diff_proc.returncode != 0:
            # Fallback to plain git diff (e.g. fresh repo with no HEAD)
            diff_proc2 = subprocess.run(
                ["git", "diff"],
                shell=False,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if diff_proc2.returncode != 0:
                return f"Git error: {diff_proc.stderr.strip() or 'Not a git repository.'}"
            diff_text = diff_proc2.stdout.strip()

        status_proc = subprocess.run(
            ["git", "status", "--short"],
            shell=False,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
        )
        status_text = status_proc.stdout.strip()

        if not diff_text and not status_text:
            return "No git diff detected (working tree clean)."

        output_parts = []
        if status_text:
            output_parts.append(f"Working Tree Status:\n{status_text}")
        if diff_text:
            output_parts.append(f"Git Diff:\n{diff_text}")
        return "\n\n".join(output_parts)
    except Exception as exc:
        return f"Error checking git diff: {exc}"


def run_pytest(
    test_path: str | None = None,
    workspace_dir: str | Path | None = None,
) -> str:
    """Run pytest in workspace_dir."""
    cwd = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
    cmd = [sys.executable, "-m", "pytest"]
    if test_path:
        cmd.append(str(test_path))
    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined_output = (proc.stdout + "\n" + proc.stderr).strip()
        parts = [f"pytest exit code: {proc.returncode}"]
        if combined_output:
            parts.append(combined_output)
        else:
            parts.append("(No pytest output)")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Error: pytest timed out after 120 seconds."
    except Exception as exc:
        return f"Error running pytest: {exc}"


# ==============================================================================
# LLM Provider Configuration
# ==============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM API connection and model parameters."""

    provider: str = "groq"
    model: str = "qwen/qwen3.8-27b"
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 60.0
    fallback_models: list[str] = field(default_factory=list)

    @classmethod
    def from_env(
        cls,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        fallback_models: list[str] | None = None,
    ) -> LLMConfig:
        """Resolve LLM configuration from environment variables with defaults."""
        prov = (
            provider
            or os.environ.get("SONIC_LLM_PROVIDER")
            or os.environ.get("DEFAULT_LLM_PROVIDER")
            or "groq"
        ).lower().strip()

        # Provider defaults
        default_base_urls = {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "deepseek": "https://api.deepseek.com",
            "nvidia": "https://integrate.api.nvidia.com/v1",
            "local": "http://localhost:11434/v1",
            "ollama": "http://localhost:11434/v1",
        }
        default_models = {
            "groq": "qwen/qwen3.8-27b",
            "openai": "gpt-4o",
            "openrouter": "qwen/qwen3.8-27b",
            "deepseek": "deepseek-chat",
            "nvidia": "nvidia/nemotron-3-super-120b-a12b",
            "local": "qwen2.5-coder:7b",
            "ollama": "qwen2.5-coder:7b",
        }
        default_fallback_models = {
            "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
            "openai": ["gpt-4o-mini"],
            "openrouter": ["qwen/qwen3.8-27b", "meta-llama/llama-3.3-70b-instruct"],
            "deepseek": [],
            "nvidia": [],
            "local": [],
            "ollama": [],
        }

        # Resolve API key
        resolved_key = api_key or ""
        if not resolved_key:
            if prov == "groq":
                resolved_key = os.environ.get("GROQ_API_KEY", "")
            elif prov == "openai":
                resolved_key = os.environ.get("OPENAI_API_KEY", "")
            elif prov == "openrouter":
                resolved_key = os.environ.get("OPENROUTER_API_KEY", "")
            elif prov == "deepseek":
                resolved_key = os.environ.get("DEEPSEEK_API_KEY", "")
            elif prov == "nvidia":
                resolved_key = os.environ.get("NVIDIA_API_KEY", "")
            elif prov in ("local", "ollama"):
                resolved_key = os.environ.get("LOCAL_API_KEY", "ollama")

        # Resolve Base URL
        resolved_base_url = (
            base_url
            or os.environ.get(f"{prov.upper()}_BASE_URL")
            or default_base_urls.get(prov, "https://api.groq.com/openai/v1")
        )

        # Resolve Model
        resolved_model = (
            model
            or os.environ.get(f"{prov.upper()}_MODEL")
            or os.environ.get("SONIC_MODEL")
            or default_models.get(prov, "qwen/qwen3.8-27b")
        )

        # Resolve Fallback Models
        if fallback_models is not None:
            resolved_fallbacks = fallback_models
        else:
            env_fallbacks = os.environ.get("SONIC_FALLBACK_MODELS", "")
            if env_fallbacks:
                resolved_fallbacks = [m.strip() for m in env_fallbacks.split(",") if m.strip()]
            else:
                resolved_fallbacks = default_fallback_models.get(prov, [])

        return cls(
            provider=prov,
            model=resolved_model,
            api_key=resolved_key,
            base_url=resolved_base_url,
            temperature=temperature if temperature is not None else 0.2,
            max_tokens=max_tokens if max_tokens is not None else 4096,
            timeout=timeout if timeout is not None else 60.0,
            fallback_models=resolved_fallbacks,
        )


# ==============================================================================
# Agent Execution Result
# ==============================================================================


@dataclass
class DeveloperAgentResult:
    """Result returned by DeveloperAgent execution."""

    success: bool
    steps: int
    final_response: str
    interrupted: bool = False
    error: str | None = None
    tool_calls_count: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# Autonomous AI Software Engineer Engine (DeveloperAgent)
# ==============================================================================


class DeveloperAgent:
    """Autonomous AI Software Engineer Engine operating directly on local workspace.

    Features:
    - Native OpenAI/Groq tool-calling agentic loop.
    - Full suite of direct local tools (view, write, edit, command, diff, pytest).
    - Streaming thought and response visualization with Rich.
    - Clean Ctrl+C KeyboardInterrupt handling.
    - Zero dependencies on external orchestrators, container engines, or graphical workstations.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        config: LLMConfig | None = None,
        max_steps: int = 25,
        console: Console | None = None,
        system_prompt: str | None = None,
        llm_client: Callable[..., Any] | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
        self.config = config or LLMConfig.from_env()
        self.max_steps = max_steps
        self.console = console or Console(legacy_windows=False)
        self.system_prompt = system_prompt or DEFAULT_DEVELOPER_SYSTEM_PROMPT
        self._llm_client = llm_client
        self._http_client: httpx.Client | None = None
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset(self) -> None:
        """Reset conversation memory to initial system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _get_client(self) -> httpx.Client:
        """Get or initialize persistent HTTP client with connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client(timeout=self.config.timeout)
        return self._http_client

    def close(self) -> None:
        """Close persistent HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            try:
                self._http_client.close()
            except Exception:
                pass

    # ---- Tool Delegation Methods ----

    def view_file(
        self,
        path: str | Path,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """View contents of a file or directory in workspace."""
        return view_file(path, start_line=start_line, end_line=end_line, workspace_dir=self.workspace_dir)

    def write_file(self, path: str | Path, content: str) -> str:
        """Write content to a file in workspace."""
        return write_file(path, content=content, workspace_dir=self.workspace_dir)

    def edit_file(
        self,
        path: str | Path,
        target_content: str,
        replacement_content: str,
    ) -> str:
        """Replace unique target content with replacement content in file."""
        return edit_file(
            path,
            target_content=target_content,
            replacement_content=replacement_content,
            workspace_dir=self.workspace_dir,
        )

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Run shell command in workspace."""
        return run_command(command, timeout=timeout, workspace_dir=self.workspace_dir)

    def git_diff(self) -> str:
        """Inspect uncommitted git changes in workspace."""
        return git_diff(workspace_dir=self.workspace_dir)

    def run_pytest(self, test_path: str | None = None) -> str:
        """Run pytest in workspace."""
        return run_pytest(test_path=test_path, workspace_dir=self.workspace_dir)

    # ---- Tool Dispatcher ----

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Safely execute a tool call by name with parsed arguments and secret sanitization."""
        try:
            if tool_name == "view_file":
                s_line = int(args["start_line"]) if args.get("start_line") is not None else None
                e_line = int(args["end_line"]) if args.get("end_line") is not None else None
                res = self.view_file(path=args.get("path", ""), start_line=s_line, end_line=e_line)
            elif tool_name == "write_file":
                res = self.write_file(path=args.get("path", ""), content=args.get("content", ""))
            elif tool_name == "edit_file":
                res = self.edit_file(
                    path=args.get("path", ""),
                    target_content=args.get("target_content", ""),
                    replacement_content=args.get("replacement_content", ""),
                )
            elif tool_name == "run_command":
                t_out = int(args.get("timeout", 30))
                res = self.run_command(command=args.get("command", ""), timeout=t_out)
            elif tool_name == "git_diff":
                res = self.git_diff()
            elif tool_name == "run_pytest":
                res = self.run_pytest(test_path=args.get("test_path"))
            else:
                res = f"Error: Unknown tool '{tool_name}'."
            return _truncate_output(sanitize_secrets(res, [self.config.api_key]))
        except Exception as exc:
            return _truncate_output(sanitize_secrets(f"Error executing tool '{tool_name}': {exc}", [self.config.api_key]))

    # ---- Rich Display Helpers ----

    def _display_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Render a tool call clearly and concisely."""
        sanitized_args = sanitize_secrets(str(args), [self.config.api_key])
        if len(sanitized_args) > 100:
            sanitized_args = sanitized_args[:95] + "..."
        self.console.print(f"  [bold yellow]→ {tool_name}[/bold yellow][dim]({sanitized_args})[/dim]")

    def _display_tool_result(self, tool_name: str, result: str) -> None:
        """Render tool execution output cleanly without bloated panels."""
        sanitized_res = sanitize_secrets(result, [self.config.api_key])
        is_error = sanitized_res.startswith("Error:") or "Exit code: 1" in sanitized_res or "pytest exit code: 1" in sanitized_res
        color = "red" if is_error else "dim green"
        lines = sanitized_res.splitlines()
        first_line = lines[0] if lines else "Done"
        if len(first_line) > 90:
            first_line = first_line[:85] + "..."
        count_info = f" ({len(lines)} lines)" if len(lines) > 1 else ""
        self.console.print(f"    [{color}]└─ {first_line}{count_info}[/{color}]")

    # ---- LLM Step & Streaming Execution ----

    def _call_single_llm(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        include_tools: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Perform one LLM turn with SSE streaming for thoughts and tool calls."""
        if not self.config.api_key:
            raise ValueError(
                f"Missing API key for provider '{self.config.provider}'. "
                f"Please set {self.config.provider.upper()}_API_KEY in your .env or system environment."
            )

        active_model = model or self.config.model
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if include_tools:
            payload["tools"] = TOOLS_SCHEMA
            payload["tool_choice"] = "auto"

        content_acc: list[str] = []
        thought_acc: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        in_thought_tag = False
        printed_thought_hdr = False
        printed_content_hdr = False

        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        client = self._get_client()
        try:
            with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_body = response.read().decode("utf-8", errors="replace")
                    clean_err = sanitize_secrets(err_body, [self.config.api_key])
                    raise RuntimeError(f"LLM API error (HTTP {response.status_code}): {clean_err}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # 1. Stream native reasoning / thoughts (DeepSeek, Qwen reasoning)
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or delta.get("thought")
                        )
                        if reasoning:
                            clean_reasoning = sanitize_secrets(reasoning, [self.config.api_key])
                            if not printed_thought_hdr:
                                self.console.print("\n[dim magenta]💭 Thinking:[/dim magenta] ", end="")
                                printed_thought_hdr = True
                            self.console.print(clean_reasoning, end="", style="dim magenta")
                            thought_acc.append(clean_reasoning)

                        # 2. Stream assistant content (and handle inline <think> tags)
                        content = delta.get("content")
                        if content:
                            if "<think>" in content:
                                in_thought_tag = True
                                content = content.replace("<think>", "")
                                if not printed_thought_hdr:
                                    self.console.print("\n[dim magenta]💭 Thinking:[/dim magenta] ", end="")
                                    printed_thought_hdr = True

                            if "</think>" in content:
                                in_thought_tag = False
                                parts = content.split("</think>")
                                if parts[0]:
                                    clean_p0 = sanitize_secrets(parts[0], [self.config.api_key])
                                    self.console.print(clean_p0, end="", style="dim magenta")
                                    thought_acc.append(clean_p0)
                                content = parts[1] if len(parts) > 1 else ""
                                self.console.print()

                            clean_content = sanitize_secrets(content, [self.config.api_key])
                            if in_thought_tag:
                                if clean_content:
                                    self.console.print(clean_content, end="", style="dim magenta")
                                    thought_acc.append(clean_content)
                            else:
                                if clean_content:
                                    if not printed_content_hdr and not tool_calls_acc:
                                        self.console.print("\n[bold cyan]🤖 Assistant:[/bold cyan] ", end="")
                                        printed_content_hdr = True
                                    self.console.print(clean_content, end="")
                                    content_acc.append(clean_content)

                        # 3. Stream tool call chunks
                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc_delta.get("function", {}).get("name", ""),
                                        "arguments": tc_delta.get("function", {}).get("arguments", ""),
                                    },
                                }
                            else:
                                if tc_delta.get("id"):
                                    tool_calls_acc[idx]["id"] += tc_delta["id"]
                                fn = tc_delta.get("function", {})
                                if fn.get("name"):
                                    tool_calls_acc[idx]["function"]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

            if printed_thought_hdr or printed_content_hdr:
                self.console.print()

            # Build assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            full_content = "".join(content_acc).strip()
            if full_content:
                assistant_msg["content"] = full_content
            else:
                assistant_msg["content"] = None

            parsed_tool_calls: list[dict[str, Any]] = []
            if tool_calls_acc:
                for idx in sorted(tool_calls_acc.keys()):
                    tc = tool_calls_acc[idx]
                    if not tc.get("id"):
                        tc["id"] = f"call_{idx}_{int(time.time() * 1000)}"
                    parsed_tool_calls.append(tc)
                assistant_msg["tool_calls"] = parsed_tool_calls

            return assistant_msg, parsed_tool_calls
        except (httpx.TransportError, httpx.RemoteProtocolError, httpx.TimeoutException):
            if self._http_client and not self._http_client.is_closed:
                try:
                    self._http_client.close()
                except Exception:
                    pass
            self._http_client = None
            raise

    def _find_alternate_provider(self) -> LLMConfig | None:
        """Find an alternative configured provider if the current provider is exhausted."""
        current_prov = self.config.provider.lower()
        candidates = [
            ("openrouter", "OPENROUTER_API_KEY", "qwen/qwen3.8-27b", "https://openrouter.ai/api/v1"),
            ("openai", "OPENAI_API_KEY", "gpt-4o-mini", "https://api.openai.com/v1"),
            ("deepseek", "DEEPSEEK_API_KEY", "deepseek-chat", "https://api.deepseek.com"),
            ("groq", "GROQ_API_KEY", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1"),
        ]
        for prov_name, env_key, default_model, base_url in candidates:
            if prov_name != current_prov:
                key = os.environ.get(env_key, "").strip()
                if key:
                    return LLMConfig(
                        provider=prov_name,
                        model=default_model,
                        api_key=key,
                        base_url=base_url,
                        fallback_models=[],
                    )
        return None

    def _step_llm(
        self,
        messages: list[dict[str, Any]],
        include_tools: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Perform one LLM turn with automatic failover to fallback models if rate-limited."""
        if self._llm_client is not None:
            if include_tools:
                return self._llm_client(messages, TOOLS_SCHEMA)
            return self._llm_client(messages, None)

        models_to_try = [self.config.model] + [
            m for m in self.config.fallback_models if m != self.config.model
        ]

        last_error: Exception | None = None
        max_tpm_retries = 2
        for i, candidate_model in enumerate(models_to_try):
            for retry_attempt in range(max_tpm_retries + 1):
                try:
                    try:
                        return self._call_single_llm(messages, model=candidate_model, include_tools=include_tools)
                    except TypeError:
                        return self._call_single_llm(messages, model=candidate_model)
                except Exception as exc:
                    err_text = str(exc)
                    is_tpd = "tpd" in err_text.lower() or "tokens per day" in err_text.lower()
                    is_rate_limit = (
                        "429" in err_text
                        or "rate_limit" in err_text.lower()
                        or is_tpd
                        or "503" in err_text
                    )
                    is_model_unavailable = (
                        "404" in err_text
                        or "does not exist" in err_text.lower()
                        or "model_not_found" in err_text.lower()
                    )
                    is_network_drop = any(
                        n in type(exc).__name__.lower()
                        for n in ["transport", "remoteprotocol", "ssl", "timeout"]
                    ) or "eof" in err_text.lower()

                    last_error = exc

                    # 1. Transient rate limit (TPM) - backoff retry same model
                    if is_rate_limit and not is_tpd and retry_attempt < max_tpm_retries:
                        backoff = 2.0 * (retry_attempt + 1)
                        self.console.print(
                            f"[dim yellow]⏳ Rate limit on '{candidate_model}' ({self.config.provider}). Retrying in {backoff:.1f}s...[/dim yellow]"
                        )
                        time.sleep(backoff)
                        continue

                    # 2. Network drop - retry same model
                    if is_network_drop and retry_attempt < max_tpm_retries:
                        backoff = 1.5 * (retry_attempt + 1)
                        self.console.print(
                            f"[dim yellow]🔌 Connection dropped. Reconnecting in {backoff:.1f}s...[/dim yellow]"
                        )
                        time.sleep(backoff)
                        continue

                    # 3. Model failover within provider (on rate limit or model unavailable)
                    if (is_rate_limit or is_model_unavailable) and i + 1 < len(models_to_try):
                        next_model = models_to_try[i + 1]
                        reason = "unavailable (HTTP 404)" if is_model_unavailable else "rate-limited (HTTP 429)"
                        self.console.print(
                            f"\n[bold yellow]⚠️ Model '{candidate_model}' {reason}. "
                            f"Failing over to fallback model '{next_model}'...[/bold yellow]"
                        )
                        self.config.model = next_model
                        break

                    # 4. Cross-provider failover if all models in current provider fail
                    if (is_rate_limit or is_tpd) and i + 1 >= len(models_to_try):
                        alt_cfg = self._find_alternate_provider()
                        if alt_cfg:
                            self.console.print(
                                f"\n[bold yellow]⚠️ All models exhausted on {self.config.provider}. "
                                f"Failing over to alternative provider '{alt_cfg.provider}' ({alt_cfg.model})...[/bold yellow]"
                            )
                            self.config = alt_cfg
                            try:
                                return self._call_single_llm(messages, model=self.config.model, include_tools=include_tools)
                            except TypeError:
                                return self._call_single_llm(messages, model=self.config.model)

                    raise last_error from None

        if last_error:
            raise last_error
        raise RuntimeError("No models available.")

    def _prune_messages(self, messages: list[dict[str, Any]], max_history: int = 12) -> list[dict[str, Any]]:
        """Keep system prompt and recent complete conversational turns to prevent context window overflow."""
        if len(messages) <= max_history + 2:
            return messages
        prefix = messages[:1]
        suffix = messages[-max_history:]
        first_user_idx = None
        for idx, msg in enumerate(suffix):
            if msg.get("role") == "user":
                first_user_idx = idx
                break
        if first_user_idx is not None:
            suffix = suffix[first_user_idx:]
        else:
            while suffix and suffix[0].get("role") == "tool":
                suffix = suffix[1:]
            # Long tool-driven turns can fill the entire window. Retain the
            # operator's objective before the remaining complete tool exchanges.
            for message in reversed(messages[:-max_history]):
                if message.get("role") == "user":
                    suffix = [message, *suffix]
                    break
        return prefix + suffix

    def chat(self, prompt: str) -> str:
        """Fast zero-shot conversational turn without tool schema overhead, streaming directly."""
        clean_prompt = sanitize_secrets(prompt, [self.config.api_key])
        self.messages.append({"role": "user", "content": clean_prompt})
        step_messages = self._prune_messages(self.messages)

        try:
            assistant_msg, _ = self._step_llm(step_messages, include_tools=False)
            self.messages.append(assistant_msg)
            ans = assistant_msg.get("content") or ""
            return sanitize_secrets(ans, [self.config.api_key])
        except Exception as exc:
            err_msg = sanitize_secrets(str(exc), [self.config.api_key])
            self.console.print(f"\n[bold red]Error: {err_msg}[/bold red]")
            return err_msg

    # ---- Main Observe-Reason-Act Execution Loop ----

    def run(self, prompt: str) -> DeveloperAgentResult:
        """Run the autonomous Observe-Reason-Act tool execution loop.

        Supports clean cancellation on KeyboardInterrupt (Ctrl+C).
        """
        clean_prompt = sanitize_secrets(prompt, [self.config.api_key])
        self.console.print(
            Panel(
                f"[bold cyan]⚡ Task:[/bold cyan] {clean_prompt}\n"
                f"[dim]Model: {self.config.model} ({self.config.provider}) · Workspace: {self.workspace_dir.name}[/dim]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

        messages = self.messages
        messages.append({"role": "user", "content": prompt})

        step = 0
        total_tool_calls = 0

        try:
            while step < self.max_steps:
                step += 1
                self.console.print(f"[bold cyan]── Step {step}/{self.max_steps}[/bold cyan]")

                # LLM Step (Stream reasoning and extract tool calls)
                step_messages = self._prune_messages(messages)
                assistant_msg, tool_calls = self._step_llm(step_messages)
                messages.append(assistant_msg)

                # Termination: No tool calls requested by model
                if not tool_calls:
                    final_ans = assistant_msg.get("content") or ""
                    clean_ans = sanitize_secrets(final_ans, [self.config.api_key])
                    self.console.print(
                        f"\n[bold green]✔ Completed in {step} step{'s' if step > 1 else ''} "
                        f"({total_tool_calls} tool call{'s' if total_tool_calls != 1 else ''}).[/bold green]"
                    )
                    return DeveloperAgentResult(
                        success=True,
                        steps=step,
                        final_response=clean_ans,
                        tool_calls_count=total_tool_calls,
                        messages=messages,
                    )

                # Tool Execution Loop
                for tc in tool_calls:
                    total_tool_calls += 1
                    fn_data = tc.get("function", {})
                    tool_name = fn_data.get("name", "")
                    raw_args = fn_data.get("arguments", "{}")

                    # Parse arguments safely
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args) if raw_args.strip() else {}
                        except json.JSONDecodeError as err:
                            args = {}
                            tool_result = f"Error: Invalid JSON arguments: {err}"
                            self._display_tool_call(tool_name, {"raw": raw_args})
                            self._display_tool_result(tool_name, tool_result)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                                "name": tool_name,
                                "content": tool_result,
                            })
                            continue
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        args = {}

                    self._display_tool_call(tool_name, args)
                    tool_result = self.execute_tool(tool_name, args)
                    self._display_tool_result(tool_name, tool_result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                        "name": tool_name,
                        "content": tool_result,
                    })

            self.console.print(f"\n[bold yellow]⚠️ Reached maximum step limit ({self.max_steps}).[/bold yellow]")
            return DeveloperAgentResult(
                success=False,
                steps=step,
                final_response="Maximum steps reached without explicit completion.",
                error=f"Step limit ({self.max_steps}) reached.",
                tool_calls_count=total_tool_calls,
                messages=messages,
            )

        except KeyboardInterrupt:
            self.console.print(
                "\n[bold yellow]⚠️ Interrupted by user (Ctrl+C). Exiting developer agent cleanly.[/bold yellow]"
            )
            return DeveloperAgentResult(
                success=False,
                steps=step,
                final_response="Task interrupted by user.",
                interrupted=True,
                tool_calls_count=total_tool_calls,
                messages=messages,
            )
        except Exception as exc:
            err_msg = sanitize_secrets(str(exc), [self.config.api_key])
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "tpd" in err_msg.lower():
                self.console.print(
                    Panel(
                        f"[bold yellow]⚠️ LLM API Daily Token / Rate Limit Reached[/bold yellow]\n\n"
                        f"The API quota for [cyan]{self.config.provider}[/cyan] ({self.config.model}) has been reached.\n"
                        f"[dim]{err_msg}[/dim]\n\n"
                        "[bold]How to continue:[/bold]\n"
                        "• Switch models in [green].env[/green]: `GROQ_MODEL=openai/gpt-oss-120b`\n"
                        "• Or configure an alternative API key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.\n"
                        "• Or wait for the token replenishment window to clear.",
                        border_style="yellow",
                        title="API Quota / Rate Limit Notice",
                    )
                )
            else:
                self.console.print(f"\n[bold red]❌ Developer agent error: {err_msg}[/bold red]")
            return DeveloperAgentResult(
                success=False,
                steps=step,
                final_response="",
                error=err_msg,
                tool_calls_count=total_tool_calls,
                messages=messages,
            )


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main() -> None:
    """Entry point for running developer agent directly from command line."""
    if len(sys.argv) < 2:
        print("Usage: python -m sonic_cli.developer_agent '<engineering task or prompt>'")
        sys.exit(1)

    task_prompt = " ".join(sys.argv[1:])
    agent = DeveloperAgent()
    agent.run(task_prompt)


if __name__ == "__main__":
    main()
