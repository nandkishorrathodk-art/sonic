"""
SONIC — Coding Agent (tool-using, multi-step autonomous engineer)
================================================================

This is the piece that makes the self-coder *actually work like an engineer*
instead of emitting one blind patch. It gives the model real tools and lets it
iterate the way a human does:

    ls / read_file / grep / run_tests / write_file / ...  ->  observe  ->  think

Two entry points:

* ``investigate(goal)`` — a read-only ReAct loop. The agent explores the repo
  (list dirs, read files, grep for symbols, run the real test suite), diagnoses
  the root cause, and returns a written diagnosis. It cannot write files.
* ``implement(goal, target_files)`` — like ``investigate``, but it may also call
  ``write_file`` to draft fixes while exploring, so it discovers which modules
  actually exist BEFORE importing them (the exact failure the blind coder made
  with ``from sonic_cli.config import clear_token``).

Every tool runs against the real workspace. Writes are staged in an in-memory
overlay (never committed) so the caller (SelfCoder) can snapshot, verify, and
roll back. Read-only tools are always safe; ``run_tests`` and ``run_command`` run
with an allowlist and no interactive shell.

This module holds no policy of its own about promotion — it produces a
diagnosis and a set of proposed file writes; the evolver's gates still decide.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sonic.llm.schemas import LLMRequest, Message, MessageRole, ToolDefinition
from sonic.logger import get_logger

logger = get_logger(__name__)

MAX_OUTPUT = 6000

# Commands the agent may run through run_command (read-only / test tooling only).
_ALLOWED_CMDS = (
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "git status",
    "git diff", "git log", "python -m pytest", "pytest", "ruff",
)

_SAFETY_MARKERS = ("sonic/safety", "sonic/sandbox/egress", "sonic/auth", "sealed")


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodingResult:
    """Outcome of an agentic coding session."""
    diagnosis: str = ""
    proposed_writes: dict[str, str] = field(default_factory=dict)
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    observed_files: list[str] = field(default_factory=list)
    finished: bool = False
    stop_reason: str = ""


_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_dir",
        "description": "List files and subdirectories of a repo-relative directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path relative to repo root."}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a repo-relative file (with line numbers), optionally a line range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "description": "First line (1-based)."},
                "end": {"type": "integer", "description": "Last line (inclusive)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": "Search the repo for a regex; returns matching file:line:text.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Subtree to search (default: repo)."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run pytest on a path and return pass/fail plus the failing assertions.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Test path (default: sonic-core/tests)."}},
        },
    },
    {
        "name": "run_command",
        "description": "Run an allowed read-only/test shell command (ls, grep, git status, pytest, ...).",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "write_file",
        "description": "Stage a full-file write (only available in implement mode).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "finish",
        "description": "Stop and return the diagnosis (and any staged writes).",
        "parameters": {
            "type": "object",
            "properties": {"diagnosis": {"type": "string", "description": "Root cause + what to change and why."}},
            "required": ["diagnosis"],
        },
    },
]


class CodingAgent:
    """A tool-using coding agent over a real repository."""

    def __init__(
        self,
        llm: Any,
        repo_root: Path | str,
        *,
        max_steps: int = 12,
        allow_write: bool = False,
        is_protected: Any | None = None,
        test_paths: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()
        self.max_steps = max_steps
        self.allow_write = allow_write
        self.is_protected = is_protected or (lambda _p: False)
        self.test_paths = test_paths
        self._overlay: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Tool definitions (filtered by mode)
    # ------------------------------------------------------------------
    def _tools(self) -> list[ToolDefinition]:
        specs = [s for s in _TOOL_SPECS if s["name"] != "write_file" or self.allow_write]
        return [ToolDefinition(name=s["name"], description=s["description"], parameters=s["parameters"]) for s in specs]

    # ------------------------------------------------------------------
    def _resolve(self, rel: str) -> Path | None:
        """Resolve a repo-relative path, refusing escapes."""
        raw = str(rel or "").replace("\\", "/").strip().lstrip("/")
        if not raw:
            return None
        p = (self.repo_root / raw).resolve()
        try:
            p.relative_to(self.repo_root)
        except ValueError:
            return None
        return p

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def _t_list_dir(self, path: str = ".") -> ToolResult:
        p = self._resolve(path or ".")
        if p is None or not p.exists():
            return ToolResult(False, f"no such directory: {path}")
        if not p.is_dir():
            return ToolResult(False, f"not a directory: {path}")
        try:
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        except Exception as e:
            return ToolResult(False, f"list failed: {e}")
        lines = []
        for e in entries[:400]:
            if e.name.startswith(".") or e.name in ("__pycache__", "node_modules", ".venv"):
                continue
            lines.append(f"{'d' if e.is_dir() else 'f'} {e.name}")
        return ToolResult(True, "\n".join(lines) or "(empty)")

    def _t_read_file(self, path: str, start: int | None = None, end: int | None = None) -> ToolResult:
        rel = str(path or "").replace("\\", "/").lstrip("/")
        if rel in self._overlay:
            text = self._overlay[rel]
        else:
            p = self._resolve(rel)
            if p is None or not p.exists() or not p.is_file():
                return ToolResult(False, f"no such file: {path}")
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return ToolResult(False, f"read failed: {e}")
        lines = text.splitlines()
        lo = max(1, int(start)) if start else 1
        hi = min(len(lines), int(end)) if end else len(lines)
        body = "\n".join(f"{i:>5}  {lines[i-1]}" for i in range(lo, hi + 1))
        return ToolResult(
            True,
            body[:MAX_OUTPUT] if body else "(empty file)",
            {"path": rel},
        )

    def _t_grep(self, pattern: str, path: str = ".") -> ToolResult:
        base = self._resolve(path or ".")
        if base is None or not base.exists():
            return ToolResult(False, f"no such path: {path}")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(False, f"bad regex: {e}")
        hits: list[str] = []
        root = base if base.is_dir() else base.parent
        for p in root.rglob("*.py"):
            if any(seg in p.parts for seg in ("__pycache__", ".venv", "node_modules")):
                continue
            rel = str(p.relative_to(self.repo_root)).replace("\\", "/")
            text = self._overlay.get(rel)
            if text is None:
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                    if len(hits) >= 200:
                        break
            if len(hits) >= 200:
                break
        return ToolResult(True, "\n".join(hits) if hits else f"no matches for {pattern!r}")

    def _t_run_tests(self, path: str = "") -> ToolResult:
        import sys

        target = path or (self.test_paths[0] if self.test_paths else "")
        if not target:
            target = "sonic-core/tests/test_evolution_strategy_and_engine.py"
        cmd = [sys.executable, "-m", "pytest", target, "-q", "--tb=short", "-p", "no:cacheprovider"]
        try:
            res = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True, text=True, timeout=600)
        except Exception as e:
            return ToolResult(False, f"pytest failed to run: {e}")
        out = (res.stdout or "") + (res.stderr or "")
        return ToolResult(res.returncode == 0, out[-MAX_OUTPUT:])

    def _t_run_command(self, command: str) -> ToolResult:
        cmd = str(command or "").strip()
        if not cmd:
            return ToolResult(False, "empty command")
        if any(c in cmd for c in (";", "&&", "|", "&", "\n", "\r")):
            return ToolResult(False, "command contains disallowed shell chaining characters")
        if any(m in cmd for m in ("rm ", ">", "curl", "wget", "sudo", "chmod", "mv ")):
            return ToolResult(False, "command contains a disallowed operator")
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return ToolResult(False, f"invalid command syntax: {e}")
        if not parts:
            return ToolResult(False, "empty command")

        def _is_allowed(cmd_parts: list[str]) -> bool:
            for prefix in _ALLOWED_CMDS:
                p_parts = prefix.split()
                if len(cmd_parts) >= len(p_parts) and cmd_parts[:len(p_parts)] == p_parts:
                    return True
            return False

        if not _is_allowed(parts):
            return ToolResult(False, f"command not allowlisted: {cmd[:80]}")
        try:
            res = subprocess.run(parts, shell=False, cwd=str(self.repo_root), capture_output=True, text=True, timeout=180)
        except Exception as e:
            return ToolResult(False, f"command failed: {e}")
        out = ((res.stdout or "") + (res.stderr or ""))[-MAX_OUTPUT:]
        return ToolResult(res.returncode == 0, out or "(no output)")

    def _t_write_file(self, path: str, content: str) -> ToolResult:
        if not self.allow_write:
            return ToolResult(False, "write_file is not available in investigate mode")
        p = self._resolve(path)
        if p is None:
            return ToolResult(False, "path escapes repository root")
        rel = str(p.relative_to(self.repo_root)).replace("\\", "/")
        if self.is_protected(rel):
            return ToolResult(False, f"refusing to write protected component: {rel}")
        if any(m in rel for m in _SAFETY_MARKERS):
            return ToolResult(False, f"refusing to write safety-sensitive path: {rel}")
        self._overlay[rel] = str(content)
        return ToolResult(True, f"staged write to {rel} ({len(str(content).splitlines())} lines)")

    # ------------------------------------------------------------------
    def _dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "list_dir":
                return self._t_list_dir(args.get("path", "."))
            if name == "read_file":
                return self._t_read_file(args.get("path", ""), args.get("start"), args.get("end"))
            if name == "grep":
                return self._t_grep(args.get("pattern", ""), args.get("path", "."))
            if name == "run_tests":
                return self._t_run_tests(args.get("path", ""))
            if name == "run_command":
                return self._t_run_command(args.get("command", ""))
            if name == "write_file":
                return self._t_write_file(args.get("path", ""), args.get("content", ""))
        except Exception as e:
            return ToolResult(False, f"tool error: {e}")
        return ToolResult(False, f"unknown tool: {name}")

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        mode = "implement (you may stage fixes with write_file)" if self.allow_write else "investigate (read-only)"
        return (
            "You are SONIC's autonomous coding engineer. You work on SONIC's OWN "
            f"repository. Mode: {mode}.\n\n"
            "Method (do NOT skip steps):\n"
            "1. Locate the relevant code with list_dir/grep BEFORE proposing changes.\n"
            "2. READ the real files. Never assume a module, class, or function exists.\n"
            "3. If the goal is a failing test, run_tests to see the real error.\n"
            "4. Make a MINIMAL, correct change. Preserve existing behaviour.\n"
            "5. Verify: re-run tests after your change.\n"
            f"6. Call finish with a concise diagnosis.\n\n"
            "Hard rules:\n"
            "- Never import or call something you have not confirmed exists (grep it).\n"
            "- Never edit sonic/safety/*, sonic/sandbox/egress, or sonic/auth/*.\n"
            "- Report honest failure; do not fabricate success.\n"
            f"You have at most {self.max_steps} tool calls."
        )

    # ------------------------------------------------------------------
    async def run(self, goal: str, *, extra_context: str = "") -> CodingResult:
        result = CodingResult()
        messages = [
            Message(role=MessageRole.SYSTEM, content=self._system_prompt()),
            Message(role=MessageRole.USER, content=f"GOAL: {goal}\n{extra_context}".strip()),
        ]
        tools = self._tools()

        for step in range(1, self.max_steps + 1):
            result.steps = step
            try:
                req = LLMRequest(messages=messages, tools=tools, task_type="coding", max_tokens=4096, temperature=0.1)
                resp = await self.llm.complete(req, task_type="coding")
            except Exception as e:
                result.stop_reason = f"llm error: {e}"
                break

            calls = list(getattr(resp, "tool_calls", []) or [])
            text = (getattr(resp, "content", "") or "").strip()

            if not calls:
                # Model answered in prose — no more tool use; treat as final thought.
                messages.append(Message(role=MessageRole.ASSISTANT, content=text or "(no content)"))
                result.diagnosis = text
                if result.tool_calls and text.strip():
                    result.finished = True
                result.stop_reason = "no_tool_call"
                break

            # Echo the assistant turn (with its tool calls) so the next request is a
            # valid OpenAI tool exchange.
            messages.append(Message(role=MessageRole.ASSISTANT, content=text, tool_calls=calls))

            finished = False
            for call in calls:
                args = call.arguments if isinstance(call.arguments, dict) else {}
                result.tool_calls.append(f"{call.name}({json.dumps(args)[:120]})")
                if call.name == "finish":
                    result.diagnosis = str(args.get("diagnosis", text or ""))
                    result.finished = True
                    finished = True
                    logger.info("coding_agent_finish", step=step)
                    out = "session finished"
                else:
                    res = self._dispatch(call.name, args)
                    out = res.output if res.ok else f"ERROR: {res.output}"
                    if call.name == "read_file" and res.ok:
                        observed_path = str(res.meta.get("path", "")).replace("\\", "/")
                        if observed_path and observed_path not in result.observed_files:
                            result.observed_files.append(observed_path)
                    logger.info("coding_agent_tool", step=step, tool=call.name, ok=res.ok)
                messages.append(Message(
                    role=MessageRole.TOOL, content=out[:MAX_OUTPUT],
                    name=call.name, tool_call_id=getattr(call, "id", "") or call.name,
                ))
            if finished:
                break
        else:
            result.stop_reason = "max_steps"

        # If the model used all its steps exploring without calling finish, ask
        # once for a written diagnosis so the caller still gets its reasoning.
        if not result.diagnosis and result.tool_calls:
            try:
                messages.append(Message(
                    role=MessageRole.USER,
                    content="You are out of tool calls. Based only on what you actually "
                            "observed, state the root cause and the minimal fix. No more tools.",
                ))
                req = LLMRequest(messages=messages, task_type="coding", max_tokens=1500, temperature=0.1)
                resp = await self.llm.complete(req, task_type="coding")
                result.diagnosis = (getattr(resp, "content", "") or "").strip()
                if result.diagnosis and result.tool_calls:
                    result.finished = True
            except Exception as e:
                logger.warning("coding_agent_summary_failed", error=str(e))

        result.proposed_writes = dict(self._overlay)
        if not result.finished and not result.stop_reason:
            result.stop_reason = "ended"
        return result
