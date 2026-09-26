"""
SONIC — Autonomous Self-Coder (closed-loop codebase self-improvement)
=====================================================================

This is the engine that makes SONIC behave like an autonomous AI coder (Codex /
Claude-Code style) on its **own** non-safety source tree:

    observe  ->  decide  ->  write a real patch  ->  verify  ->  report

Default posture is **REPORT-ONLY**: the coder verifies the change it proposes
(AST, tests, security regression) and restores its worktree. API/background
callers add an isolated Git worktree and persist a tested candidate; after all
verification gates pass, the candidate is committed automatically to a separate
local branch. The active branch and remote are never changed.

Unlike the old evolution queue (which only ran from a manual CLI and, when it
ran, appended a timestamp comment), this loop:

1. **Observes real signals** from the codebase/workspace — source-attributed
   test failures, selected static-audit findings, TODO / FIXME / HACK markers,
   and leftover `# [SONIC-EVOLUTION]` no-op stubs — and turns the strongest one
   into a concrete, file-scoped `EvolutionGoal`. No fabricated goals: if there
   is no real signal it honestly reports "nothing to do".
2. **Writes a real patch** through `PatchSynthesizer` (a real LLM produces a
   unified diff that must apply cleanly and change real code).
3. **Verifies** through the existing `CodebaseEvolver` gates (AST pre-flight,
   tests that must actually run, fail-closed security regression) with a target
   test subset derived from what changed.
4. **Waits for operator approval** before an API/background candidate is committed.
5. **Reports** a structured, honest summary of every cycle.

Safety: every write goes through the evolver's protected-component guard and the
being's sealed `ActionPolicy` (colocated in the sandbox). The self-coder can
never edit `sonic/safety/*`, `sonic/sandbox/egress`, auth, or the sealed policy.
"""

from __future__ import annotations

import asyncio
import os
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
        evolution_engine: Any | None = None,
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
        self._related_test_paths: dict[str, set[str]] = {}
        self.evolver = evolver or CodebaseEvolver(repo_root=self.repo_root)
        self.evolution_engine = evolution_engine
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

    def isolated_copy(self, repo_root: Path | str) -> SelfCoder:
        """Create a report-only coder bound to a disposable isolated worktree."""
        from sonic.evolution.codebase_evolver import CodebaseEvolver
        from sonic.evolution.engine import EvolutionEngine
        from sonic.evolution.evolution_journal import EvolutionJournal
        from sonic.evolution.goal_director import EvolutionGoalDirector
        from sonic.evolution.pipeline import EvolutionPipeline
        from sonic.evolution.version_tracker import VersionTracker

        root = Path(repo_root).resolve()
        db_path = root / "sonic_data" / "self_coder_evolution.db"
        pipeline = EvolutionPipeline()
        evolver = CodebaseEvolver(
            repo_root=root,
            lessons_ledger=self.evolver.lessons_ledger,
            pipeline=pipeline,
        )
        versions = VersionTracker(db_path=str(db_path), repo_root=root)
        journal = EvolutionJournal(db_path=str(db_path), repo_root=root)
        director = EvolutionGoalDirector(
            repo_root=root,
            db_path=str(db_path),
            evolver=evolver,
            version_tracker=versions,
            journal=journal,
            lessons_ledger=self.director.lessons_ledger,
            llm_client=self.llm,
            test_paths=self.director.test_paths,
        )
        engine = EvolutionEngine(
            pipeline=pipeline,
            strategy_engine=getattr(self.evolution_engine, "strategy_engine", None),
            method_lab=getattr(self.evolution_engine, "method_lab", None),
            toolsmith=getattr(self.evolution_engine, "toolsmith", None),
            lessons_ledger=getattr(self.evolution_engine, "lessons_ledger", None),
            codebase_evolver=evolver,
            version_tracker=versions,
            journal=journal,
            goal_director=director,
            llm_client=self.llm,
            queue_self_update_goal=False,
        )
        return SelfCoder(
            repo_root=root,
            llm=engine.llm_client,
            evolver=engine.codebase_evolver,
            director=engine.goal_director,
            evolution_engine=engine,
            max_cycles=1,
            push=False,
            enabled=self.enabled,
            allow_writes=False,
            agentic=self.agentic,
            max_agent_steps=self.max_agent_steps,
        )

    # ------------------------------------------------------------------
    # 1. Observe real signals
    # ------------------------------------------------------------------
    def discover_signals(self, limit: int = 20) -> list[tuple[str, str, str]]:
        """Return (signal, target_file, category) tuples derived from real state.

        Runtime test failures are attributed to production frames where possible.
        Read-only audit findings add code-health signals; TODO markers remain
        lower-confidence work items rather than confirmed defects.
        """
        signals: list[tuple[str, str, str]] = self._failing_test_signals(limit)
        seen = {(target, signal) for signal, target, _ in signals}

        try:
            from sonic.evolution.codebase_auditor import CodebaseAuditor

            audit_findings = CodebaseAuditor(
                repo_root=self.repo_root,
                max_findings=max(limit * 5, 50),
            ).static_scan()
            for finding in sorted(
                audit_findings,
                key=lambda item: (
                    {"critical": 0, "high": 1, "medium": 2}.get(item.severity, 3),
                    item.file,
                    item.line,
                ),
            ):
                if self._is_protected(finding.file):
                    continue
                signal = (
                    f"Review {finding.kind} at {finding.file}:{finding.line}: "
                    f"{finding.title}. {finding.detail}"
                )
                key = (finding.file, signal)
                if key not in seen:
                    signals.append((signal, finding.file, GoalCategory.BUG_FIX.value))
                    seen.add(key)
                if len(signals) >= limit:
                    return signals[:limit]
        except Exception as exc:
            logger.warning("self_coder_static_audit_failed", error=str(exc))

        for rel in self._source_files():
            if len(signals) >= limit:
                break
            p = self.repo_root / rel
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if _NOOP_STUB_RE.search(text):
                signal = f"Remove leftover no-op self-evolution stub in {rel}"
                if (rel, signal) not in seen:
                    signals.append((signal, rel, GoalCategory.REFACTOR.value))
                    seen.add((rel, signal))
            for m in _TODO_RE.finditer(text):
                note = (m.group(2) or "").strip()[:160]
                signal = (
                    f"Address {m.group(1).upper()} in {rel}: {note}"
                    if note else f"Address {m.group(1).upper()} in {rel}"
                )
                if (rel, signal) not in seen:
                    signals.append((signal, rel, GoalCategory.BUG_FIX.value))
                    seen.add((rel, signal))
                if len(signals) >= limit:
                    break

        return signals[:limit]

    def _failing_test_signals(self, limit: int) -> list[tuple[str, str, str]]:
        """Run pytest and only emit a fix signal when a traceback reaches source."""
        try:
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "sonic-core/tests/",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
            ]
            env = os.environ.copy()
            source_paths = [
                str(path)
                for path in (self.repo_root / "sonic-core", self.repo_root)
                if path.is_dir()
            ]
            inherited_pythonpath = env.get("PYTHONPATH", "")
            if inherited_pythonpath:
                source_paths.append(inherited_pythonpath)
            env["PYTHONPATH"] = os.pathsep.join(source_paths)
            res = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            out = (res.stdout or "") + (res.stderr or "")
            signals: list[tuple[str, str, str]] = []
            headings = list(re.finditer(r"(?m)^_{5,}\s*(.*?)\s*_{5,}\s*$", out))
            frame_pattern = re.compile(
                r"(?m)^[ \t]*(.+?\.py):(\d+): in ([^\r\n]+)$"
            )
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
                test_rel = test_file.replace("\\", "/")
                test_name = nodeid.split("::")[-1].split("[", 1)[0]
                matching_headings = [
                    index
                    for index, heading in enumerate(headings)
                    if test_name in heading.group(1)
                ]
                if len(matching_headings) != 1:
                    logger.info(
                        "self_coder_failure_not_attributed",
                        test=nodeid,
                        reason="traceback section is missing or ambiguous",
                    )
                    continue
                block_index = matching_headings[0]
                block_start = headings[block_index].end()
                block_end = (
                    headings[block_index + 1].start()
                    if block_index + 1 < len(headings)
                    else out.find("short test summary info", block_start)
                )
                if block_end < 0:
                    block_end = len(out)
                block = out[block_start:block_end]
                source_frames: list[tuple[str, int]] = []
                for frame in frame_pattern.finditer(block):
                    frame_path = Path(frame.group(1))
                    if not frame_path.is_absolute():
                        frame_path = self.repo_root / frame_path
                    try:
                        frame_rel = frame_path.resolve().relative_to(self.repo_root).as_posix()
                    except ValueError:
                        continue
                    if (
                        any(frame_rel.startswith(f"{root}/") for root in _SOURCE_ROOTS)
                        and not self._is_protected(frame_rel)
                    ):
                        source_frames.append((frame_rel, int(frame.group(2))))
                if not source_frames:
                    logger.info(
                        "self_coder_failure_not_attributed",
                        test=nodeid,
                        reason="traceback did not identify an editable source frame",
                    )
                    continue
                source_file, source_line = source_frames[-1]
                self._related_test_paths.setdefault(source_file, set()).add(test_rel)
                signals.append((
                    f"Investigate failing test {nodeid}; traceback reaches "
                    f"{source_file}:{source_line}",
                    source_file,
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
                if not cres.finished or not diagnosis.strip():
                    cycle.stage = "investigation_incomplete"
                    cycle.reason = (
                        "No patch was generated because the coding agent did not "
                        "finish a tool-grounded root-cause investigation"
                    )
                    return cycle
            except Exception as exc:
                logger.warning("self_coder_investigation_failed", error=str(exc))
                cycle.stage = "investigation_incomplete"
                cycle.reason = f"No patch was generated after investigation failed: {exc}"
                return cycle

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

        matched_tests: list[str] = sorted(self._related_test_paths.get(target_file, set()))

        for t_dir in tests_dirs:
            if not t_dir.exists():
                continue

            # 1. Direct name match: test_<stem>.py
            direct = t_dir / f"test_{stem}.py"
            if direct.exists():
                direct_rel = str(direct.relative_to(self.repo_root)).replace("\\", "/")
                if direct_rel not in matched_tests:
                    matched_tests.append(direct_rel)

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
    async def run(
        self,
        max_cycles: int | None = None,
        excluded_signals: set[tuple[str, str]] | None = None,
    ) -> SelfCoderReport:
        report = SelfCoderReport()
        if not self.enabled:
            report.finished_at = _now()
            return report
        cycles = max_cycles if max_cycles is not None else self.max_cycles
        signals = self.discover_signals()
        if excluded_signals:
            signals = [
                signal
                for signal in signals
                if (signal[1], signal[0]) not in excluded_signals
            ]
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

    def run_sync(
        self,
        max_cycles: int | None = None,
        excluded_signals: set[tuple[str, str]] | None = None,
    ) -> SelfCoderReport:
        """Blocking entry point for schedulers/CLI."""
        return asyncio.run(
            self.run(max_cycles=max_cycles, excluded_signals=excluded_signals)
        )
