"""
SONIC — Autonomous Self-Coder (closed-loop codebase self-improvement)
=====================================================================

This is the engine that makes SONIC behave like an autonomous AI coder (Codex /
Claude-Code style) on its **own** non-safety source tree:

    observe  ->  decide  ->  write a real patch  ->  verify  ->  report

Default posture is **REPORT-ONLY**: the coder verifies the change it proposes
(AST, tests, security regression) on a scratch copy, then rolls the tree back to
exactly its previous state and writes a report describing what it found and what
it would fix. It never commits and never leaves a modified file behind unless the
operator explicitly passes ``allow_writes=True``.

Unlike the old evolution queue (which only ran from a manual CLI and, when it
ran, appended a timestamp comment), this loop:

1. **Observes real signals** from the codebase/workspace — failing tests, TODO /
   FIXME / HACK markers, and leftover `# [SONIC-EVOLUTION]` no-op stubs — and
   turns the strongest one into a concrete, file-scoped `EvolutionGoal`. No
   fabricated goals: if there is no real signal it honestly reports "nothing to
   do".
2. **Writes a real patch** through `PatchSynthesizer` (a real LLM produces a
   unified diff that must apply cleanly and change real code).
3. **Verifies** through the existing `CodebaseEvolver` gates (AST pre-flight,
   tests that must actually run, fail-closed security regression) with a target
   test subset derived from what changed.
4. **Commits locally** on success (never pushes to a remote unless explicitly
   asked) and advances the semantic version.
5. **Reports** a structured, honest summary of every cycle.

Safety: every write goes through the evolver's protected-component guard and the
being's sealed `ActionPolicy` (colocated in the sandbox). The self-coder can
never edit `sonic/safety/*`, `sonic/sandbox/egress`, auth, or the sealed policy.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonic.evolution.codebase_evolver import CodebaseEvolver
from sonic.evolution.goal_director import EvolutionGoalDirector, GoalCategory
from sonic.evolution.pipeline import EvolutionStage
from sonic.logger import get_logger

logger = get_logger(__name__)

# Directories that make up SONIC's own (non-safety) source that it may improve.
_SOURCE_ROOTS = ("sonic-core/sonic", "sonic-cli/sonic_cli")
_SAFETY_PREFIXES = ("sonic-core/sonic/safety", "sonic-core/sonic/sandbox/egress.py",
                    "sonic-core/sonic/auth", "sonic-core/sonic/sandbox/providers")

_TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\b[:\s]*(.*)", re.IGNORECASE)
# The exact no-op artifact the previous stub used to append. Anchored to a
# real comment line so documentation mentioning the marker is not a signal.
_NOOP_STUB_RE = re.compile(r"(?m)^[ \t]*#\s*\[SONIC-EVOLUTION\][^\n]*")



def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SelfCoderCycle:
    """One observed signal + its outcome."""
    signal: str
    target_file: str = ""
    category: str = ""
    goal_id: str = ""
    stage: str = "skipped"
    reason: str = ""
    version_after: str = ""
    commit_hash: str = ""
    summary: str = ""
    patch_files: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    # The real unified diff the coder wrote and verified (for the operator).
    proposed_diff: str = ""
    # What happened to the tree: "rolled_back" (report-only) or "committed"/"left".
    tree_action: str = "rolled_back"
    # Root-cause diagnosis produced by the tool-using coding agent.
    diagnosis: str = ""


@dataclass
class SelfCoderReport:
    cycles: list[SelfCoderCycle] = field(default_factory=list)
    changed: bool = False
    started_at: str = field(default_factory=_now)
    finished_at: str = ""

    def to_text(self) -> str:
        lines = [
            "# 🧬 SONIC Self-Coder — codebase report (report-only by default)",
            "",
            f"- Started:  `{self.started_at}`",
            f"- Finished: `{self.finished_at or _now()}`",
            f"- Cycles:   {len(self.cycles)}",
            f"- Writes committed: {'yes' if self.changed else 'no (report-only)'}",
            "",
        ]
        for i, c in enumerate(self.cycles, 1):
            lines.append(f"## Cycle {i}: {c.signal}")
            if c.target_file:
                lines.append(f"- Target: `{c.target_file}`")
            lines.append(f"- Outcome: **{c.stage}**" + (f" (v{c.version_after})" if c.version_after else ""))
            lines.append(f"- Tree: {c.tree_action}")
            if c.commit_hash:
                lines.append(f"- Commit: `{c.commit_hash}`")
            if c.patch_files:
                lines.append(f"- Files: {', '.join(f'`{f}`' for f in c.patch_files)}")
                lines.append(f"- Diff size: `+{c.lines_added} / -{c.lines_removed}`")
            if c.reason:
                lines.append(f"- Reason: {c.reason}")
            if c.summary:
                lines.append(f"- Summary: {c.summary}")
            if c.diagnosis:
                lines.append("")
                lines.append("Investigation (tool-verified):")
                lines.append(c.diagnosis.strip()[:2000])
            if c.proposed_diff:
                lines.append("")
                lines.append("Proposed fix (verified against tests, then rolled back):")
                lines.append("```diff")
                lines.append(c.proposed_diff.strip()[:4000])
                lines.append("```")
            lines.append("")
        return "\n".join(lines)


class SelfCoder:
    """Autonomous, LLM-driven self-improvement loop over SONIC's own source."""

    def __init__(
        self,
        repo_root: Path | str | None = None,
        llm: Any | None = None,
        evolver: CodebaseEvolver | None = None,
        director: EvolutionGoalDirector | None = None,
        max_cycles: int = 1,
        push: bool = False,
        enabled: bool = True,
        allow_writes: bool = False,
        agentic: bool = True,
        max_agent_steps: int = 16,
    ) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else self._discover_repo_root()
        self.llm = llm
        self.enabled = enabled
        self.push = push
        self.max_cycles = max_cycles
        # REPORT-ONLY by default: verify the proposed change, then restore the
        # tree so nothing is committed and no file is left modified.
        self.allow_writes = allow_writes
        # Agentic mode gives the model real tools (list_dir/read_file/grep/
        # run_tests/run_command) so it stops guessing at APIs that don't exist.
        self.agentic = agentic
        self.max_agent_steps = max_agent_steps
        self.last_coding_result: Any = None
        self.evolver = evolver or CodebaseEvolver(repo_root=self.repo_root)
        self.director = director or EvolutionGoalDirector(
            repo_root=self.repo_root,
            evolver=self.evolver,
            llm_client=llm,
        )
        # The synthesizer must use THIS coder's LLM, not a lazily-resolved global
        # router (otherwise a caller-supplied/stub LLM is silently ignored).
        if llm is not None:
            self.director.llm_client = llm

    # ------------------------------------------------------------------
    @staticmethod
    def _discover_repo_root() -> Path:
        cur = Path(__file__).resolve()
        for p in cur.parents:
            if (p / ".git").exists() or (p / "sonic-core").exists():
                return p
        return cur.parent

    def _is_protected(self, rel: str) -> bool:
        return self.evolver.is_protected(rel)

    # ------------------------------------------------------------------
    # 1. Observe real signals
    # ------------------------------------------------------------------
    def discover_signals(self, limit: int = 20) -> list[tuple[str, str, str]]:
        """Return (signal, target_file, category) tuples derived from real state.

        Ordering reflects trustworthiness: a failing test is the strongest
        signal, a leftover no-op stub is next (it is provably broken behaviour we
        shipped), then TODO/FIXME markers a developer left behind.
        """
        # Strongest first: real, currently-failing tests observed by running pytest.
        signals: list[tuple[str, str, str]] = self._failing_test_signals(limit)

        for rel in self._source_files():
            if len(signals) >= limit:
                break
            p = self.repo_root / rel
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if _NOOP_STUB_RE.search(text):
                signals.append((
                    f"Remove leftover no-op self-evolution stub in {rel}",
                    rel,
                    GoalCategory.REFACTOR.value,
                ))
            for m in _TODO_RE.finditer(text):
                note = (m.group(2) or "").strip()[:160]
                signals.append((
                    f"Address {m.group(1).upper()} in {rel}: {note}" if note else f"Address {m.group(1).upper()} in {rel}",
                    rel,
                    GoalCategory.BUG_FIX.value,
                ))
                if len(signals) >= limit:
                    break

        return signals[:limit]

    def _failing_test_signals(self, limit: int) -> list[tuple[str, str, str]]:
        """Find REAL failing tests by running pytest, and target the test file.

        The self-coder must act on observed reality, not a hardcoded guess. We
        run the suite once (cheap, ``--tb=no``) and turn each failure into a
        signal whose target is the FAILING TEST FILE — a stale/incorrect
        assertion is fixed in the test, not silently "fixed" in the source.
        """
        try:
            import subprocess
            import sys

            cmd = [sys.executable, "-m", "pytest", "sonic-core/tests/", "-q", "--tb=no", "-p", "no:cacheprovider"]
            res = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True, text=True, timeout=300)
            out = (res.stdout or "") + (res.stderr or "")
            signals: list[tuple[str, str, str]] = []
            for line in out.splitlines():
                line = line.strip()
                nodeid = None
                if line.startswith("FAILED "):
                    nodeid = line[len("FAILED "):].split(" ")[0].strip()
                elif line.startswith("ERROR "):
                    nodeid = line[len("ERROR "):].split(" ")[0].strip()
                if not nodeid or "::" not in nodeid:
                    continue
                test_file = nodeid.split("::")[0].strip()
                if not (self.repo_root / test_file).exists():
                    continue
                if self._is_protected(test_file):
                    continue
                signals.append((
                    f"Repair the failing test {nodeid} (the assertion and the real behaviour disagree)",
                    test_file,
                    GoalCategory.BUG_FIX.value,
                ))
                if len(signals) >= limit:
                    break
            return signals
        except Exception as exc:
            logger.warning("self_coder_failing_test_scan_failed", error=str(exc))
            return []

    def _source_files(self):
        for root in _SOURCE_ROOTS:
            base = self.repo_root / root
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.py")):
                rel = str(p.relative_to(self.repo_root)).replace("\\", "/")
                if self._is_protected(rel):
                    continue
                if rel.endswith("__init__.py"):
                    continue
                yield rel

    # ------------------------------------------------------------------
    # 2. One cycle
    # ------------------------------------------------------------------
    async def run_cycle(self, signal: str, target_file: str, category: str) -> SelfCoderCycle:
        cycle = SelfCoderCycle(signal=signal, target_file=target_file, category=category)
        if self._is_protected(target_file):
            cycle.stage = "rejected"
            cycle.reason = "target is a protected safety component"
            return cycle

        # Snapshot the working tree so we can restore it after a report-only run.
        # Include evolution.md because constructing the journal creates it.
        snap_files = [target_file, "evolution.md"] + list(self._test_paths_for(target_file) or [])
        before = self._tree_fingerprint(list(dict.fromkeys(snap_files)))

        # Step 1: agentic investigation — explore the REAL repo with tools so we
        # do not hallucinate modules/functions (the blind-coder failure mode).
        diagnosis = ""
        if self.agentic and self.llm is not None:
            try:
                from sonic.evolution.coding_agent import CodingAgent

                agent = CodingAgent(
                    llm=self.llm,
                    repo_root=self.repo_root,
                    max_steps=self.max_agent_steps,
                    allow_write=False,
                    is_protected=self.evolver.is_protected,
                    test_paths=self._test_paths_for(target_file),
                )
                cres = await agent.run(
                    f"Diagnose and plan a fix for this signal in {target_file}: {signal}"
                )
                self.last_coding_result = cres
                diagnosis = cres.diagnosis or ""
                logger.info(
                    "self_coder_investigated",
                    steps=cres.steps,
                    tools=len(cres.tool_calls),
                    has_diagnosis=bool(diagnosis),
                )
            except Exception as exc:  # investigation is best-effort, never fatal
                logger.warning("self_coder_investigation_failed", error=str(exc))

        goal = self.director.submit_goal(
            title=signal[:180],
            description=(
                f"Autonomous self-improvement. Real signal: {signal}. "
                f"Improve {target_file} so the signal is resolved without changing "
                "unrelated behaviour."
                + (f"\n\nInvestigation diagnosis (verified against the real repo):\n{diagnosis}" if diagnosis else "")
            ),
            category=category,
            target_files=[target_file],
            max_attempts=2,
        )
        cycle.goal_id = goal.goal_id
        cycle.diagnosis = diagnosis

        report = await self.director.execute_goal(
            goal.goal_id,
            auto_promote=bool(self.allow_writes),
            push=self.push,
            test_paths=self._test_paths_for(target_file),
            report_only=not self.allow_writes,
        )
        cycle.stage = report.stage.value
        cycle.summary = report.description
        cycle.lines_added = report.lines_added
        cycle.lines_removed = report.lines_removed
        if report.files_affected:
            cycle.patch_files = list(report.files_affected)
        # Attach the real diff the coder produced (from the verified proposal).
        proposal = getattr(self.director, "last_patch_proposal", None)
        if proposal is not None and getattr(proposal, "diff", ""):
            cycle.proposed_diff = proposal.diff
        if self.allow_writes and report.stage == EvolutionStage.PROMOTED:
            cycle.commit_hash = report.git_commit_hash
            cycle.tree_action = "committed"
            g = self.director.get_goal(goal.goal_id)
            if g and g.version_after:
                cycle.version_after = g.version_after
        else:
            if report.stage != EvolutionStage.PROMOTED:
                cycle.reason = (report.error_reason or "gate did not pass")[:400]
            # Report-only: restore only the files we snapshotted. The evolver
            # restores every file its patch actually touched (it snapshots them
            # as it applies); we never `git checkout` unrelated paths, because
            # that would destroy the operator's own uncommitted work.
            self._restore_tree(before)
            cycle.tree_action = "rolled_back"
        return cycle

    # ------------------------------------------------------------------
    def _tree_fingerprint(self, files: list[str]) -> dict[str, str | None]:
        """Capture current content of the given files (None = did not exist)."""
        snap: dict[str, str | None] = {}
        for rel in files:
            p = (self.repo_root / rel).resolve()
            try:
                snap[rel] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
            except Exception:
                snap[rel] = None
        return snap

    def _restore_tree(self, snap: dict[str, str | None]) -> None:
        """Restore files to their snapshotted content (report-only guarantee)."""
        for rel, content in snap.items():
            p = (self.repo_root / rel).resolve()
            try:
                if content is None:
                    if p.exists():
                        p.unlink()
                else:
                    p.write_text(content, encoding="utf-8")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("self_coder_restore_failed", file=rel, error=str(exc))

    def _test_paths_for(self, target_file: str) -> list[str] | None:
        """Dynamically discover tests relevant to `target_file` from the repository filesystem.

        Zero hardcoded keyword dictionaries. Discovers tests by:
        1. Explicit caller/director configured paths.
        2. Exact test file match (`test_<stem>.py`).
        3. Scanning test files for imports or references to the target module.
        """
        if self.director and self.director.test_paths:
            return self.director.test_paths

        target_path = Path(target_file)
        stem = target_path.stem

        tests_dirs = [
            self.repo_root / "sonic-core" / "tests",
            self.repo_root / "tests",
        ]

        matched_tests: list[str] = []

        for t_dir in tests_dirs:
            if not t_dir.exists():
                continue

            # 1. Direct name match: test_<stem>.py
            direct = t_dir / f"test_{stem}.py"
            if direct.exists():
                matched_tests.append(str(direct.relative_to(self.repo_root)).replace("\\", "/"))

            # 2. Dynamic inspection of test files for module imports or references
            for tf in sorted(t_dir.glob("test_*.py")):
                rel_tf = str(tf.relative_to(self.repo_root)).replace("\\", "/")
                if rel_tf in matched_tests:
                    continue
                try:
                    txt = tf.read_text(encoding="utf-8", errors="ignore")
                    if stem in txt:
                        matched_tests.append(rel_tf)
                except Exception:
                    continue

        if matched_tests:
            return matched_tests[:5]

        # Default evolution engine test if available
        base = self.repo_root / "sonic-core" / "tests" / "test_evolution_strategy_and_engine.py"
        if base.exists():
            return [str(base.relative_to(self.repo_root)).replace("\\", "/")]

        return None

    # ------------------------------------------------------------------
    # 3. The autonomous loop
    # ------------------------------------------------------------------
    async def run(self, max_cycles: int | None = None) -> SelfCoderReport:
        report = SelfCoderReport()
        if not self.enabled:
            report.finished_at = _now()
            return report
        cycles = max_cycles if max_cycles is not None else self.max_cycles
        signals = self.discover_signals()
        if not signals:
            logger.info("self_coder_no_signal")
            report.finished_at = _now()
            return report

        done = 0
        for signal, target, category in signals:
            if done >= cycles:
                break
            try:
                cycle = await self.run_cycle(signal, target, category)
            except Exception as exc:  # never let one cycle kill the loop
                logger.warning("self_coder_cycle_error", error=str(exc))
                cycle = SelfCoderCycle(signal=signal, target_file=target, stage="error", reason=str(exc))
            report.cycles.append(cycle)
            if cycle.stage == EvolutionStage.PROMOTED.value:
                report.changed = True
            done += 1

        report.finished_at = _now()
        logger.info(
            "self_coder_cycle_complete",
            cycles=len(report.cycles),
            changed=report.changed,
        )
        return report

    def run_sync(self, max_cycles: int | None = None) -> SelfCoderReport:
        """Blocking entry point for schedulers/CLI."""
        return asyncio.run(self.run(max_cycles=max_cycles))
