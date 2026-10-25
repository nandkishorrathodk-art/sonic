"""
SONIC — LLM-Driven Code Patch Synthesizer (self-evolution)
==========================================================

This is the missing "code-writing brain" of SONIC's self-coder. Given a goal and
the real context of the target file(s), it asks a real LLM (via the Model
Router) to produce a **unified git diff** that achieves the goal, then
*deterministically verifies the diff is real* before anyone is allowed to apply
it:

    * the diff must be a genuine unified diff (``--- a/...`` / ``+++ b/...`` /
      ``@@`` hunks) — a bare comment/annotation is rejected;
    * it must apply cleanly to the current working tree (checked with
      ``git apply --check``);
    * it must change at least one non-comment, non-blank line (a pure
      whitespace/comment append is rejected as a no-op);
    * it must not touch any protected safety component.

Only when all four hold does ``synthesize_patch`` return the diff. Otherwise it
returns ``None`` (honest failure) and the caller records the goal as
unverified — never "success by decree".

The synthesizer never writes to disk itself. It only proposes; the
``CodebaseEvolver`` owns staging, AST validation, tests, version bump, and
commit. This keeps a single, auditable write path.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sonic.llm.prompts import code_evolver_system_prompt
from sonic.llm.schemas import LLMRequest, Message, MessageRole
from sonic.logger import get_logger

logger = get_logger(__name__)

_TAGGED_FENCE_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_UNTAGGED_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:diff|patch|git)?\s*\n(.*?)```", re.DOTALL)
_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
_BARE_HUNK_RE = re.compile(r"^@@\s*$", re.MULTILINE)
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", re.MULTILINE)


def _repair_hunk_counts(diff: str) -> str:
    """Recompute every hunk header's line counts from its own body.

    Models frequently emit a structurally valid unified diff whose ``@@``
    header counts disagree with the body (e.g. ``@@ -1,5 +1,5 @@`` above a
    single context line). ``git apply`` rejects that outright with
    ``corrupt patch``, discarding an otherwise-correct patch. The counts are
    fully derivable from the body, so we recompute them deterministically and
    keep the model's start lines untouched.
    """
    if not _HUNK_HEADER_RE.search(diff):
        return diff
    lines = diff.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HUNK_HEADER_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        body: list[str] = []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if _HUNK_HEADER_RE.match(nxt) or nxt.startswith(("diff --git ", "--- ", "+++ ")):
                break
            body.append(nxt)
            j += 1

        # ``\ No newline at end of file`` is a marker, not a counted line.
        counted = [b for b in body if not b.startswith("\\")]
        old_count = sum(1 for b in counted if b[:1] in (" ", "-"))
        new_count = sum(1 for b in counted if b[:1] in (" ", "+"))
        suffix = line[m.end():]
        out.append(f"@@ -{m.group(1)},{old_count} +{m.group(2)},{new_count} @@{suffix}")
        out.extend(body)
        i = j
    return "\n".join(out) + "\n"


def _normalize_hunks(diff: str, read_file: Any) -> tuple[str, str]:
    """Fill in hunk headers that lack line numbers (models love bare `@@`).

    git apply rejects a hunk header without ``-a,b +c,d`` outright, which causes
    perfectly good model patches to be thrown away. Here we derive the real line
    numbers from the target file (position of the first hunk line) and count the
    old/new lines, turning a bare ``@@`` into a valid header. Returns
    (normalized_diff, error) — error is "" on success.
    """
    if not _BARE_HUNK_RE.search(diff):
        return diff, ""

    lines = diff.splitlines()
    out: list[str] = []
    i = 0
    current_file: str | None = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            p = line[4:].strip()
            p = p[2:] if p.startswith("a/") else p
            if p not in ("/dev/null", "dev/null"):
                current_file = p
            out.append(line)
            i += 1
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            p = p[2:] if p.startswith("b/") else p
            if p not in ("/dev/null", "dev/null"):
                current_file = p
            out.append(line)
            i += 1
            continue
        if not _BARE_HUNK_RE.match(line):
            out.append(line)
            i += 1
            continue

        # Collect this hunk's body until the next header/hunk/file boundary.
        j = i + 1
        body: list[str] = []
        while j < len(lines):
            nxt = lines[j]
            if nxt.startswith(("@@", "diff --git ", "--- ", "+++ ")):
                break
            body.append(nxt)
            j += 1
        if not body:
            return diff, "bare hunk header with no body"

        old_count = sum(1 for b in body if b[:1] in (" ", "-", "") and not b.startswith("\\"))
        new_count = sum(1 for b in body if b[:1] in (" ", "+", "") and not b.startswith("\\"))

        # Locate the first real (context/removed) line in the target file to get
        # the old start line; fall back to line 1 if we cannot read the file.
        first = next((b[1:] for b in body if b[:1] in (" ", "-")), None)
        start = 1
        if current_file and first is not None:
            content = read_file(current_file)
            if content is None:
                return diff, f"cannot read target file for bare hunk: {current_file}"
            hay = content.splitlines()
            found = None
            for idx, hl in enumerate(hay):
                if hl.rstrip() == first.rstrip():
                    found = idx + 1
                    break
            if found is None:
                return diff, f"bare hunk context not found in {current_file}"
            start = found
        out.append(f"@@ -{start},{old_count} +{start},{new_count} @@")
        out.extend(body)
        i = j
    return "\n".join(out) + "\n", ""


@dataclass
class PatchProposal:
    """A candidate patch together with the evidence that it is real."""
    goal_id: str
    diff: str
    files: list[str]
    rationale: str = ""
    model: str = ""
    provider: str = ""
    added: int = 0
    removed: int = 0
    change_lines: int = 0

    def is_usable(self) -> bool:
        return bool(self.diff.strip()) and self.change_lines > 0 and bool(self.files)


@dataclass
class PatchSynthesisResult:
    proposal: PatchProposal | None
    attempts: int = 0
    reason: str = ""
    notes: list[str] = field(default_factory=list)


def _extract_diff(text: str) -> str:
    """Pull a unified diff out of a model response (fenced block or raw)."""
    if not text:
        return ""

    candidate = ""
    # 1. Prioritize fenced blocks tagged with ```diff or ```patch
    tagged_matches = _TAGGED_FENCE_RE.findall(text)
    for m in tagged_matches:
        if "diff --git " in m or "--- " in m:
            candidate = m
            break
    if not candidate and tagged_matches:
        candidate = tagged_matches[0]

    # 2. Prioritize untagged fences before checking other language fences
    if not candidate:
        untagged_matches = _UNTAGGED_FENCE_RE.findall(text)
        for m in untagged_matches:
            if "diff --git " in m or "--- " in m:
                candidate = m
                break

    # 3. Check any fences (e.g. ```git) that contain diff markers
    if not candidate:
        any_matches = _ANY_FENCE_RE.findall(text)
        for m in any_matches:
            if "diff --git " in m or "--- " in m:
                candidate = m
                break

    # 4. Fall back to raw text or untagged fence
    if not candidate:
        if "diff --git " in text or "--- " in text:
            candidate = text
        elif untagged := _UNTAGGED_FENCE_RE.findall(text):
            candidate = untagged[0]
        else:
            candidate = text

    idx = candidate.find("diff --git ")
    if idx == -1:
        idx = candidate.find("--- ")
    if idx != -1:
        candidate = candidate[idx:]
    return candidate.strip()


def _count_changes(diff: str) -> tuple[int, int, int]:
    """Return (added, removed, change_lines) counting only real +/- content lines."""
    added = removed = change = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
            if line[1:].strip() and not line[1:].lstrip().startswith("#"):
                change += 1
        elif line.startswith("-"):
            removed += 1
            if line[1:].strip() and not line[1:].lstrip().startswith("#"):
                change += 1
    return added, removed, change


def _clean_diff_path(p: str) -> str:
    p = p.strip()
    if "\t" in p:
        p = p.split("\t", 1)[0].strip()
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    p = p.strip()
    if p in ("/dev/null", "dev/null") or p.endswith("/dev/null"):
        return ""
    return p


def _files_touched(diff: str) -> list[str]:
    files: list[str] = []
    for m in _DIFF_FILE_RE.finditer(diff):
        for raw in (m.group(1), m.group(2)):
            p = _clean_diff_path(raw)
            if p and p not in files:
                files.append(p)
    if not files:
        # Fall back to --- a/<path> and +++ b/<path> headers.
        for line in diff.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                p = _clean_diff_path(line[4:])
                if p and p not in files:
                    files.append(p)
    return files


class PatchSynthesizer:
    """Turns a goal + real code context into a verified unified diff."""

    def __init__(
        self,
        llm: Any,
        repo_root: Path | str,
        *,
        is_protected: Any | None = None,
        max_tokens: int = 8192,
    ) -> None:
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()
        self.is_protected = is_protected or (lambda _p: False)
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    async def synthesize(
        self,
        goal: Any,
        target_files: list[str],
        *,
        max_attempts: int = 3,
        test_paths: list[str] | None = None,
        failure_digest: str = "",
    ) -> PatchSynthesisResult:
        """Ask the LLM for a patch; retry with feedback until it is real or we give up."""
        result = PatchSynthesisResult(proposal=None)
        if self.llm is None:
            result.reason = "no LLM router configured for the self-coder"
            return result
        if not target_files:
            result.reason = "no candidate target files"
            return result

        context = self._build_context(target_files, test_paths or [])
        last_reason = ""
        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt
            prompt = self._build_prompt(goal, target_files, context, last_reason, failure_digest)
            try:
                diff_text = await self._complete(prompt)
            except Exception as exc:  # honest failure, never fabricated
                last_reason = f"LLM call failed: {exc}"
                result.notes.append(last_reason)
                continue

            diff = _extract_diff(diff_text)
            diff, norm_err = _normalize_hunks(diff, self._read_target)
            if norm_err:
                last_reason = f"could not normalize hunk headers: {norm_err}"
                result.notes.append(f"attempt {attempt}: {last_reason}")
                continue
            diff = _repair_hunk_counts(diff)
            ok, reason, proposal = self._validate_and_measure(goal, diff)
            if ok and proposal is not None:
                result.proposal = proposal
                result.reason = "verified unified diff"
                return result
            last_reason = reason
            result.notes.append(f"attempt {attempt}: {reason}")

        result.reason = last_reason or "LLM produced no usable patch"
        return result

    # ------------------------------------------------------------------
    async def _complete(self, prompt: str) -> str:
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=code_evolver_system_prompt()),
                Message(role=MessageRole.USER, content=prompt),
            ],
            task_type="coding",
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        response = await self.llm.complete(request, task_type="coding")
        return getattr(response, "content", "") or ""

    def _validate_and_measure(
        self,
        goal: Any,
        diff: str,
    ) -> tuple[bool, str, PatchProposal | None]:
        if not diff.strip():
            return False, "empty response (no diff)", None
        if not _HUNK_RE.search(diff):
            return False, "response is not a unified diff (no @@ hunks)", None
        files = _files_touched(diff)
        if not files:
            return False, "diff does not name any target file", None
        norm_files: list[str] = []
        for f in files:
            nf = os.path.normpath(f).replace("\\", "/")
            if nf not in norm_files:
                norm_files.append(nf)
        files = norm_files
        for f in files:
            if self.is_protected(f):
                return False, f"diff touches protected safety component: {f}", None
        added, removed, change = _count_changes(diff)
        if change == 0:
            return False, "diff only adds comments/blank lines (no real code change)", None
        ok, reason = self._applies_cleanly(diff)
        if not ok:
            return False, f"diff does not apply cleanly: {reason}", None
        proposal = PatchProposal(
            goal_id=getattr(goal, "goal_id", ""),
            diff=diff,
            files=files,
            added=added,
            removed=removed,
            change_lines=change,
        )
        return True, "ok", proposal

    def _read_target(self, rel: str) -> str | None:
        """Read a repo file for hunk normalization (None if it does not exist)."""
        try:
            p = (self.repo_root / rel).resolve()
            if p.exists() and p.is_file():
                return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return None

    def _applies_cleanly(self, diff: str) -> tuple[bool, str]:
        import tempfile

        path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".patch",
                delete=False,
                encoding="utf-8",
                newline="\n",
            ) as tf:
                tf.write(diff if diff.endswith("\n") else diff + "\n")
                path = tf.name
            res = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", path],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            return (res.returncode == 0), (res.stderr or res.stdout or "").strip()[:300]
        except Exception as exc:
            return False, str(exc)
        finally:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _build_context(self, target_files: list[str], test_paths: list[str]) -> str:
        chunks: list[str] = []
        for rel in target_files[:3]:
            p = (self.repo_root / rel).resolve()
            if not p.exists() or not p.is_file():
                chunks.append(f"### FILE (missing): {rel}")
                continue
            body = p.read_text(encoding="utf-8", errors="replace")
            chunks.append(f"### FILE: {rel} ({len(body.splitlines())} lines)\n```python\n{body[:12000]}\n```")
        for rel in test_paths[:2]:
            p = (self.repo_root / rel).resolve()
            if p.exists() and p.is_file():
                body = p.read_text(encoding="utf-8", errors="replace")
                chunks.append(f"### TEST FILE: {rel}\n```python\n{body[:4000]}\n```")
        return "\n\n".join(chunks)

    def _build_prompt(
        self,
        goal: Any,
        target_files: list[str],
        context: str,
        previous_failure: str,
        failure_digest: str,
    ) -> str:
        retry = ""
        if previous_failure:
            retry = (
                "\nYour previous attempt was rejected for this reason:\n"
                f"  {previous_failure}\n"
                "Return ONLY a corrected unified diff.\n"
            )
        if failure_digest:
            retry += f"\nEvidence to fix (from a real failing run/observation):\n{failure_digest[:3000]}\n"
        return (
            f"GOAL ({getattr(goal, 'category', 'improvement')}): {getattr(goal, 'title', '')}\n"
            f"DETAIL: {getattr(goal, 'description', '')}\n\n"
            f"TARGET FILES (propose edits ONLY to these):\n"
            + "\n".join(f"- {f}" for f in target_files[:3])
            + f"\n\nCURRENT SOURCE CONTEXT:\n{context}\n"
            f"{retry}\n"
            "Produce a unified git diff (git apply-ready, paths relative to repo root, "
            "a/ and b/ prefixes) that makes a minimal, correct, real code change achieving "
            "the goal. Do NOT add license headers or unrelated comments. Do NOT touch any "
            "safety component. Output the diff inside a single ```diff fenced block and "
            "nothing else."
        )
