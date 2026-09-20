"""
SONIC — Codebase report narrator (human-readable, report-only)
==============================================================

Turns a machine `AuditReport` (and, optionally, what the self-coder proposed to
fix) into the plain narrative an operator actually reads:

    "Sir, while I was looking at your codebase I found 12 things worth your
     attention: 2 will crash a code path, 9 are latent defects, 1 is a leftover
     no-op stub. I verified a fix for the crash in `x.py` against the test suite
     (55 passed) — here is the diff. I did not commit anything; the tree is
     exactly as you left it."

Pure formatting: no I/O, no mutation.
"""

from __future__ import annotations

from typing import Any

from sonic.evolution.codebase_auditor import AuditReport

_KIND_PROSE = {
    "syntax_error": "a file that does not even compile",
    "bare_except": "a bare `except:` that swallows every error",
    "swallowed_exception": "a silently swallowed exception",
    "mutable_default": "a mutable default argument that leaks state",
    "dynamic_exec": "dynamic `eval`/`exec` execution",
    "shell_true": "a `shell=True` subprocess call",
    "noop_stub": "a leftover no-op self-evolution stub",
    "todo_marker": "a developer TODO/FIXME marker",
}


def _count_phrase(n: int, word: str) -> str:
    """Pluralize only when n != 1: '1 crash' vs '2 bugs' (no 'is a latent problem' wording)."""
    return f"{n} {word}" if n == 1 else f"{n} {word.rstrip()}s" if not word.endswith("s") else f"{n} {word}"


def narrate(report: AuditReport, cycles: list[Any] | None = None) -> str:
    """Render the audit (+ optional self-coder cycles) as a brief for the operator."""
    counts = report.counts()
    tests = report.tests or {}
    health = report.health or {}
    cycles = cycles or []

    lines: list[str] = ["Sir,", ""]

    # --- what the check found -------------------------------------------
    if not report.findings and not tests.get("failed"):
        lines.append(
            "while I was looking at your codebase, everything checked out — "
            "no crash-level defects, no failing tests, no leftover stubs. "
            "The tree is healthy."
        )
        lines.append("")
    else:
        crashy = [f for f in report.sorted_findings() if f.severity == "critical"]
        high = [f for f in report.sorted_findings() if f.severity == "high"]
        medium = [f for f in report.sorted_findings() if f.severity == "medium"]
        intro = (
            f"while I was looking at your codebase I found {len(report.findings)} "
            f"thing{'s' if len(report.findings) != 1 else ''} worth your attention"
        )
        parts = []
        if crashy:
            parts.append(_count_phrase(len(crashy), "file breaks outright"))
        if high:
            parts.append(_count_phrase(len(high), "high-risk defect"))
        if medium:
            parts.append(_count_phrase(len(medium), "latent problem"))
        if tests.get("failed"):
            parts.append(_count_phrase(int(tests["failed"]), "failing test"))
        if parts:
            intro += ": " + ", ".join(parts)
        lines.append(intro + ".")
        lines.append("")

        # --- specifics ---------------------------------------------------
        if tests.get("failed"):
            lines.append("First, the things that are actually failing today:")
            for t in tests.get("failed_tests", [])[:10]:
                lines.append(f"  - {t}")
            lines.append("")
        if crashy or high:
            lines.append("Ones I'd fix first:")
            for f in (crashy + high)[:10]:
                prose = _KIND_PROSE.get(f.kind, f.title)
                lines.append(f"  - `{f.file}:{f.line}` — {prose}. {f.detail}")
            lines.append("")
        if medium:
            lines.append("Then the latent ones (they don't crash today but they hide bugs):")
            for f in medium[:10]:
                prose = _KIND_PROSE.get(f.kind, f.title)
                lines.append(f"  - `{f.file}:{f.line}` — {prose}.")
            lines.append("")

    # --- health -----------------------------------------------------------
    if health:
        missing = health.get("missing_dependencies")
        if missing and missing != "none":
            lines.append(
                f"Heads up on the environment: these runtime dependencies are not "
                f"installed here — {missing}. Until they are, the backend can't boot "
                "and the test runner is unavailable."
            )
            lines.append("")

    # --- what the coder did about it --------------------------------------
    promoted = [c for c in cycles if getattr(c, "stage", "") == "promoted"]
    proposed = [c for c in cycles if getattr(c, "proposed_diff", "")]
    if cycles:
        lines.append("")
        lines.append("What I did about it:")
        for c in cycles:
            action = getattr(c, "tree_action", "rolled_back")
            stage = getattr(c, "stage", "")
            lines.append(
                f"  - {c.signal} -> verified patch ({stage}); tree: {action}."
            )
        if proposed and not promoted:
            lines.append("")
            lines.append(
                "I verified the fix against the test suite, then I put the tree back "
                "exactly as it was. I did not commit anything. If you want the change "
                "kept, say so and I'll apply it (or set allow_writes=True)."
            )
    if promoted:
        lines.append("")
        lines.append(
            f"I applied and committed {_count_phrase(len(promoted), 'verified change')} "
            "locally (no remote push)."
        )
    else:
        lines.append("")
        lines.append("Nothing was committed; the working tree is untouched.")

    return "\n".join(lines).strip()
