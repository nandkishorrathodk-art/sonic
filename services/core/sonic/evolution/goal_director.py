"""
SONIC v2 — Evolution Goal Director
==================================
The intelligent orchestrator for goal-driven, autonomous codebase self-evolution.

Key Capabilities:
1. Goal Queue Management: Accepts high-level operator directives ("fix bug X", "upgrade logic Y")
   and persists them in SQLite so they survive restarts.
2. Intelligent Codebase Analysis: Analyzes intent, locates affected files via AST / keyword indexing,
   and diagnoses the target code.
3. Patch Synthesis: Synthesizes code modifications (via LLM or structured rules) respecting LessonsLedger.
4. Closed-Loop Verification: Staged patch application, AST validation, test execution, atomic rollback.
5. Semantic Version Progression: Bumps version (0.1.0 -> 0.1.1 for bug fixes, 0.1.x -> 0.2.0 for milestones).
6. Live evolution.md Logging: Every edit and task updates evolution.md in real-time.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sonic.evolution.codebase_evolver import (
    CodebaseEvolver,
    EvolutionSummaryReport,
    SafetyInvariantViolation,
    VerificationTestResult,
)
from sonic.evolution.evolution_journal import EvolutionJournal, JournalEntry
from sonic.evolution.pipeline import EvolutionStage
from sonic.evolution.sqlite import ClosingSQLiteConnection
from sonic.evolution.version_tracker import VersionTracker
from sonic.logger import get_logger

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_db_path() -> str:
    from sonic.memory.sqlite_graph import _default_db_path as _g
    return os.environ.get("SONIC_EVOLUTION_DB_PATH") or os.environ.get("SONIC_MEMORY_DB_PATH") or _g()


class GoalCategory(StrEnum):
    BUG_FIX = "bug_fix"
    LOGIC_IMPROVEMENT = "logic_improvement"
    CAPABILITY_ADD = "capability_add"
    PERFORMANCE = "performance"
    REFACTOR = "refactor"


class GoalStatus(StrEnum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    TESTING = "testing"
    VERIFIED = "verified"
    PROMOTED = "promoted"
    AWAITING_APPROVAL = "awaiting_approval"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    FAILED = "failed"


def create_self_update_goal(director: EvolutionGoalDirector) -> EvolutionGoal:
    """Queue the durable maintenance goal used by the evolution engine."""
    return director.submit_goal(
        title="Self-Update: maintain autonomous assessment reliability",
        description=(
            "Review non-safety runtime reliability gaps discovered by tests or "
            "operator feedback, then propose a verified, reversible improvement."
        ),
        priority=4,
        category=GoalCategory.LOGIC_IMPROVEMENT,
        max_attempts=1,
    )


@dataclass
class EvolutionGoal:
    """A high-level improvement or bug-fix objective for SONIC to achieve."""
    goal_id: str
    title: str
    description: str
    priority: int = 2  # 1=critical, 2=high, 3=medium, 4=low
    category: GoalCategory = GoalCategory.LOGIC_IMPROVEMENT
    status: GoalStatus = GoalStatus.QUEUED
    target_files: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    completed_at: str | None = None
    version_before: str = "0.1.0"
    version_after: str | None = None
    evolution_report: str = ""
    attempts: int = 0
    max_attempts: int = 3
    notes: str = ""
    approval_manifest: dict[str, Any] = field(default_factory=dict)


class EvolutionGoalDirector:
    """
    Directs autonomous self-evolution workflows from high-level objectives
    down to code edits, tests, version bumps, and live evolution.md updates.
    """

    def __init__(
        self,
        repo_root: Path | str | None = None,
        db_path: str | None = None,
        evolver: CodebaseEvolver | None = None,
        version_tracker: VersionTracker | None = None,
        journal: EvolutionJournal | None = None,
        lessons_ledger: Any | None = None,
        llm_client: Any | None = None,
        test_paths: list[str] | None = None,
        patch_max_attempts: int = 3,
        patch_max_tokens: int = 8192,
    ) -> None:
        self.db_path = db_path or _default_db_path()
        if repo_root is None:
            cur = Path(__file__).resolve()
            root = cur.parent
            for p in cur.parents:
                if (p / ".git").exists() or (p / "services/core").exists():
                    root = p
                    break
            self.repo_root = root
        else:
            self.repo_root = Path(repo_root).resolve()

        self.version_tracker = version_tracker or VersionTracker(db_path=self.db_path, repo_root=self.repo_root)
        self.journal = journal or EvolutionJournal(db_path=self.db_path, repo_root=self.repo_root)
        self.lessons_ledger = lessons_ledger
        self.evolver = evolver or CodebaseEvolver(repo_root=self.repo_root, lessons_ledger=self.lessons_ledger)
        self.llm_client = llm_client
        # Default test target for self-evolution verification. When None the
        # evolver picks its own default; the self-coder overrides this with a
        # causal subset derived from the changed files.
        self.test_paths = test_paths
        self.patch_max_attempts = patch_max_attempts
        self.patch_max_tokens = patch_max_tokens
        # Last verified patch proposal (for reporting / the self-coder summary).
        self.last_patch_proposal: Any = None

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(
            self.db_path, timeout=10.0, factory=ClosingSQLiteConnection
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_goals (
                    goal_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority INTEGER DEFAULT 2,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_files_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    version_before TEXT NOT NULL,
                    version_after TEXT,
                    evolution_report TEXT DEFAULT '',
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    notes TEXT DEFAULT '',
                    approval_manifest_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(evolution_goals)")
            }
            if "approval_manifest_json" not in columns:
                conn.execute(
                    "ALTER TABLE evolution_goals ADD COLUMN "
                    "approval_manifest_json TEXT NOT NULL DEFAULT '{}'"
                )
            conn.commit()

    # ------------------------------------------------------------------
    # 1. Goal Queue Management
    # ------------------------------------------------------------------
    def submit_goal(
        self,
        title: str,
        description: str,
        priority: int = 2,
        category: GoalCategory | str = GoalCategory.LOGIC_IMPROVEMENT,
        target_files: list[str] | None = None,
        max_attempts: int = 3,
    ) -> EvolutionGoal:
        """Enqueues a new evolution goal into the persistent queue."""
        goal_id = f"goal-{uuid.uuid4().hex[:8]}"
        cat = GoalCategory(category) if isinstance(category, str) else category
        v_current = self.version_tracker.current_version()

        goal = EvolutionGoal(
            goal_id=goal_id,
            title=title,
            description=description,
            priority=priority,
            category=cat,
            status=GoalStatus.QUEUED,
            target_files=target_files or [],
            created_at=_now(),
            version_before=v_current,
            max_attempts=max_attempts,
        )

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO evolution_goals (
                    goal_id, title, description, priority, category, status,
                    target_files_json, created_at, completed_at, version_before,
                    version_after, evolution_report, attempts, max_attempts, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                goal.goal_id,
                goal.title,
                goal.description,
                goal.priority,
                goal.category.value,
                goal.status.value,
                json.dumps(goal.target_files),
                goal.created_at,
                goal.completed_at,
                goal.version_before,
                goal.version_after,
                goal.evolution_report,
                goal.attempts,
                goal.max_attempts,
                goal.notes,
            ))
            conn.commit()

        logger.info("evolution_goal_submitted", goal_id=goal.goal_id, title=title, priority=priority)
        return goal

    def get_goal(self, goal_id: str) -> EvolutionGoal | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evolution_goals WHERE goal_id = ?", (goal_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_goal(row)

    def list_goals(self, status: GoalStatus | str | None = None) -> list[EvolutionGoal]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                st = status.value if isinstance(status, GoalStatus) else str(status)
                cursor.execute("SELECT * FROM evolution_goals WHERE status = ? ORDER BY priority ASC, created_at ASC", (st,))
            else:
                cursor.execute("SELECT * FROM evolution_goals ORDER BY priority ASC, created_at ASC")
            return [self._row_to_goal(r) for r in cursor.fetchall()]

    def get_next_queued_goal(self) -> EvolutionGoal | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM evolution_goals
                WHERE status = 'queued'
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            """)
            row = cursor.fetchone()
            return self._row_to_goal(row) if row else None

    def _update_goal(self, goal: EvolutionGoal) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE evolution_goals SET
                    status = ?,
                    completed_at = ?,
                    version_after = ?,
                    evolution_report = ?,
                    attempts = ?,
                    notes = ?,
                    target_files_json = ?,
                    approval_manifest_json = ?
                WHERE goal_id = ?
            """, (
                goal.status.value,
                goal.completed_at,
                goal.version_after,
                goal.evolution_report,
                goal.attempts,
                goal.notes,
                json.dumps(goal.target_files),
                json.dumps(goal.approval_manifest, sort_keys=True),
                goal.goal_id,
            ))
            conn.commit()

    def _row_to_goal(self, row: sqlite3.Row) -> EvolutionGoal:
        return EvolutionGoal(
            goal_id=row["goal_id"],
            title=row["title"],
            description=row["description"],
            priority=row["priority"],
            category=GoalCategory(row["category"]),
            status=GoalStatus(row["status"]),
            target_files=json.loads(row["target_files_json"] or "[]"),
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            version_before=row["version_before"],
            version_after=row["version_after"],
            evolution_report=row["evolution_report"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            notes=row["notes"],
            approval_manifest=json.loads(row["approval_manifest_json"] or "{}"),
        )

    def _make_approval_manifest(
        self,
        report: EvolutionSummaryReport,
    ) -> dict[str, Any]:
        if (
            not report.test_result.passed
            or report.test_result.failed_tests != 0
            or not report.security_regression_passed
        ):
            raise ValueError("verified test and security-regression receipts are required")
        if not report.files_affected:
            raise ValueError("the verified report contains no files to promote")

        files: list[str] = []
        digests: dict[str, str] = {}
        git_blobs: dict[str, str] = {}
        for item in report.files_affected:
            path = Path(item)
            resolved = (path if path.is_absolute() else self.repo_root / path).resolve()
            try:
                relative = resolved.relative_to(self.repo_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"approval file escapes repository: {item}") from exc
            if not resolved.is_file():
                raise ValueError(f"approval file is missing from the working tree: {relative}")
            files.append(relative)
            digests[relative] = hashlib.sha256(resolved.read_bytes()).hexdigest()
            git_blobs[relative] = subprocess.run(
                ["git", "hash-object", f"--path={relative}", str(resolved)],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        test = report.test_result
        return {
            "base_commit": head_result.stdout.strip(),
            "files": sorted(set(files)),
            "sha256": digests,
            "git_blobs": git_blobs,
            "report": {
                "target_component": report.target_component,
                "description": report.description,
                "files_affected": list(report.files_affected),
                "lines_added": report.lines_added,
                "lines_removed": report.lines_removed,
                "ast_validated": report.ast_validated,
                "security_regression_passed": report.security_regression_passed,
                "test_result": {
                    "passed": test.passed,
                    "passed_tests": test.passed_tests,
                    "total_tests": test.total_tests,
                    "failed_tests": test.failed_tests,
                    "skipped_tests": test.skipped_tests,
                    "duration_seconds": test.duration_seconds,
                    "raw_output": test.raw_output,
                    "error_message": test.error_message,
                },
            },
        }

    def _approval_report(
        self,
        goal: EvolutionGoal,
        *,
        error: str = "",
        stage: EvolutionStage = EvolutionStage.AWAITING_APPROVAL,
    ) -> EvolutionSummaryReport:
        manifest = goal.approval_manifest
        payload = manifest.get("report", {})
        test_data = payload.get("test_result", {})
        return EvolutionSummaryReport(
            proposal_id=goal.goal_id,
            target_component=payload.get("target_component", ", ".join(goal.target_files)),
            description=payload.get("description", goal.description),
            stage=stage,
            files_affected=list(payload.get("files_affected", manifest.get("files", []))),
            lines_added=int(payload.get("lines_added", 0)),
            lines_removed=int(payload.get("lines_removed", 0)),
            ast_validated=bool(payload.get("ast_validated", False)),
            test_result=VerificationTestResult(
                passed=bool(test_data.get("passed", False)),
                passed_tests=int(test_data.get("passed_tests", 0)),
                total_tests=int(test_data.get("total_tests", 0)),
                failed_tests=int(test_data.get("failed_tests", 0)),
                skipped_tests=int(test_data.get("skipped_tests", 0)),
                duration_seconds=float(test_data.get("duration_seconds", 0.0)),
                raw_output=str(test_data.get("raw_output", "")),
                error_message=str(test_data.get("error_message", "")),
            ),
            security_regression_passed=bool(
                payload.get("security_regression_passed", False)
            ),
            error_reason=error,
        )

    def reject_goal(self, goal_id: str) -> EvolutionGoal:
        """Reject a pending promotion without silently discarding working-tree edits."""
        goal = self.get_goal(goal_id)
        if goal is None:
            raise ValueError(f"Evolution goal not found: {goal_id}")
        if goal.status != GoalStatus.AWAITING_APPROVAL:
            raise ValueError(
                f"Goal {goal_id} is {goal.status.value}, not awaiting approval"
            )
        goal.status = GoalStatus.REJECTED
        goal.completed_at = _now()
        goal.notes = (
            "Operator rejected promotion. Verified working-tree edits were left "
            "untouched; they were not committed or rolled back."
        )
        self._update_goal(goal)
        return goal

    def approve_goal(self, goal_id: str, *, push: bool = False) -> EvolutionSummaryReport:
        """Commit a still-identical, previously verified pending patch on operator approval."""
        goal = self.get_goal(goal_id)
        if goal is None:
            raise ValueError(f"Evolution goal not found: {goal_id}")
        if goal.status != GoalStatus.AWAITING_APPROVAL:
            raise ValueError(
                f"Goal {goal_id} is {goal.status.value}, not awaiting approval"
            )

        manifest = goal.approval_manifest
        files = manifest.get("files", [])
        expected_digests = manifest.get("sha256", {})
        expected_blobs = manifest.get("git_blobs", {})
        base_commit = manifest.get("base_commit", "")
        if (
            not files
            or not base_commit
            or set(files) != set(expected_digests)
            or set(files) != set(expected_blobs)
        ):
            report = self._approval_report(
                goal,
                error=(
                    "No complete verified approval manifest is available. "
                    "The goal cannot be promoted safely; rerun verification."
                ),
            )
            goal.notes = report.error_reason
            self._update_goal(goal)
            return report

        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if head != base_commit:
                raise ValueError(
                    "repository HEAD changed after verification; rerun the goal before approval"
                )

            for relative in files:
                resolved = (self.repo_root / relative).resolve()
                resolved.relative_to(self.repo_root)
                actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
                if actual != expected_digests[relative]:
                    raise ValueError(
                        f"verified working-tree file changed after tests: {relative}"
                    )

            staged = subprocess.run(
                ["git", "diff", "--cached", "--quiet", "--", *files],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )
            if staged.returncode != 0:
                raise ValueError(
                    "approval files have staged changes; unstage and rerun verification"
                )
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            report = self._approval_report(goal, error=str(exc))
            goal.notes = (
                f"Approval blocked; verified edits remain uncommitted and untouched: {exc}"
            )
            self._update_goal(goal)
            return report

        committed, _reported_hash, pushed, push_message = self.evolver.commit_and_push(
            proposal_id=goal.goal_id,
            description=goal.title,
            files=files,
            push=push,
        )

        # Confirm the actual commit, its parent, subject, and committed file bytes;
        # never advance the goal or version based only on the helper's return value.
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            parent = subprocess.run(
                ["git", "rev-parse", "HEAD^"],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subject = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            changed_files = set(subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines())
            commit_is_verified = (
                committed
                and bool(_reported_hash)
                and head != base_commit
                and head.startswith(_reported_hash)
                and parent == base_commit
                and subject.startswith(f"evo({goal.goal_id}):")
                and set(files).issubset(changed_files)
            )
            if commit_is_verified:
                for relative in files:
                    blob_hash = subprocess.run(
                        ["git", "rev-parse", f"HEAD:{relative}"],
                        cwd=str(self.repo_root),
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    if blob_hash != expected_blobs[relative]:
                        commit_is_verified = False
                        break
        except (OSError, subprocess.CalledProcessError):
            commit_is_verified = False
            head = ""

        if not commit_is_verified:
            reason = push_message if not committed else (
                "Git commit was created but could not be verified against the approval "
                "manifest; manual review is required."
            )
            reason = reason or "No verifiable goal-specific commit was created."
            report = self._approval_report(goal, error=reason)
            goal.notes = (
                f"Approval did not produce a verified promotion. No version advance was "
                f"recorded. {reason}"
            )
            self._update_goal(goal)
            return report

        commit_hash = head[:12]
        if goal.category == GoalCategory.CAPABILITY_ADD:
            version_after = self.version_tracker.bump_minor(
                goal_id=goal.goal_id,
                title=goal.title,
                git_commit=commit_hash,
                tag=False,
            )
        else:
            version_after = self.version_tracker.bump_patch(
                goal_id=goal.goal_id,
                title=goal.title,
                git_commit=commit_hash,
                tag=False,
            )
        report = self._approval_report(goal, stage=EvolutionStage.PROMOTED)
        report.git_committed = True
        report.git_commit_hash = commit_hash
        report.git_pushed = pushed
        report.git_push_message = push_message
        goal.status = GoalStatus.PROMOTED
        goal.completed_at = _now()
        goal.version_after = version_after
        goal.evolution_report = report.to_markdown()
        goal.notes = (
            f"Operator approved verified commit {commit_hash}; version advanced to "
            f"{version_after}. {push_message}"
        ).strip()
        self._update_goal(goal)
        self._record_entry(JournalEntry(
            entry_id=f"ent-{uuid.uuid4().hex[:8]}",
            goal_id=goal.goal_id,
            title=goal.title,
            category=goal.category.value,
            status="promoted",
            version_before=goal.version_before,
            version_after=version_after,
            files_affected=report.files_affected,
            lines_added=report.lines_added,
            lines_removed=report.lines_removed,
            test_summary=(
                f"Tests: {report.test_result.passed_tests}/"
                f"{report.test_result.total_tests} passed"
            ),
            notes="Operator-approved promotion after file and commit verification.",
            commit_hash=commit_hash,
        ))
        return report

    # ------------------------------------------------------------------
    # 2. Intelligent Codebase Analysis
    # ------------------------------------------------------------------
    def analyze_goal(self, goal: EvolutionGoal) -> dict[str, Any]:
        """
        Scans codebase to discover relevant target files, verifies safety invariants,
        and extracts existing structure for patch synthesis.
        """
        target_files = list(goal.target_files)

        # 1. If no target files specified, locate potential candidate files by keyword matching
        if not target_files:
            target_files = self._discover_candidate_files(goal.title + " " + goal.description)

        # 2. Safety Invariant verification
        for f in target_files:
            if self.evolver.is_protected(f):
                raise SafetyInvariantViolation(
                    f"Forbidden: '{f}' is a protected safety kernel module. AI self-modification is denied."
                )

        # 3. Read file contents and summarize functions/classes
        file_contexts: dict[str, dict[str, Any]] = {}
        for rel_path in target_files:
            try:
                full_p = self.evolver._resolve_repo_path(rel_path)
            except ValueError as exc:
                raise SafetyInvariantViolation(str(exc)) from exc
            if full_p.exists() and full_p.is_file():
                content = full_p.read_text(encoding="utf-8", errors="replace")
                symbols: list[str] = []
                if full_p.suffix == ".py":
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                symbols.append(node.name)
                    except Exception:
                        pass
                file_contexts[rel_path] = {
                    "exists": True,
                    "lines": len(content.splitlines()),
                    "symbols": symbols,
                    "content": content,
                }
            else:
                file_contexts[rel_path] = {
                    "exists": False,
                    "lines": 0,
                    "symbols": [],
                    "content": "",
                }

        return {
            "target_files": target_files,
            "file_contexts": file_contexts,
            "safe": True,
        }

    def _discover_candidate_files(self, text: str) -> list[str]:
        """Discovers relevant non-safety Python source files based on query tokens."""
        tokens = set(re.findall(r"[a-zA-Z_]{3,}", text.lower()))
        # Filter common non-informative words
        stopwords = {"the", "and", "for", "with", "this", "that", "from", "sonic", "code", "file", "make", "need", "update"}
        keywords = tokens - stopwords

        candidates: list[tuple[str, int]] = []
        search_dirs = [self.repo_root / "services/core" / "sonic", self.repo_root / "apps/cli"]

        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for p in sdir.rglob("*.py"):
                rel_str = str(p.relative_to(self.repo_root)).replace("\\", "/")
                # Skip protected safety components
                if self.evolver.is_protected(rel_str):
                    continue
                score = 0
                fname_lower = p.name.lower()
                for kw in keywords:
                    if kw in fname_lower:
                        score += 5
                try:
                    text_head = p.read_text(encoding="utf-8", errors="replace")[:1500].lower()
                    for kw in keywords:
                        if kw in text_head:
                            score += 1
                except Exception:
                    pass

                if score > 0:
                    candidates.append((rel_str, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:3]]

    # ------------------------------------------------------------------
    # 3. Patch Synthesis & Execution
    # ------------------------------------------------------------------
    async def execute_goal(
        self,
        goal_id: str,
        custom_patch: str | None = None,
        auto_promote: bool = False,
        push: bool = False,
        test_paths: list[str] | None = None,
        report_only: bool | None = None,
    ) -> EvolutionSummaryReport:
        """Run one goal end-to-end.

        By default, a non-promoting run verifies and records the exact diff,
        then restores the working tree. Passing ``auto_promote=True`` is an
        explicit request to apply and commit the verified change.
        1. Analyzes goal and codebase
        2. Applies patch in the checkout with atomic rollback guarantees
        3. Runs verification tests
        4. Optionally bumps version and records the change
        """
        if report_only is None:
            report_only = not auto_promote
        if push and not auto_promote:
            raise ValueError("push requires explicit auto_promote=True")
        if report_only and auto_promote:
            raise ValueError("report_only cannot be combined with auto_promote")

        goal = self.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Evolution goal not found: {goal_id}")

        goal.attempts += 1
        goal.status = GoalStatus.ANALYZING
        self._update_goal(goal)

        # 1. Analysis
        try:
            analysis = self.analyze_goal(goal)
            target_files = analysis["target_files"]
            goal.target_files = target_files
        except SafetyInvariantViolation as e:
            goal.status = GoalStatus.REJECTED
            goal.notes = f"Safety Invariant Block: {e}"
            goal.completed_at = _now()
            self._update_goal(goal)

            # Record rejection in journal & evolution.md
            self._record_entry(JournalEntry(
                entry_id=f"ent-{uuid.uuid4().hex[:8]}",
                goal_id=goal.goal_id,
                title=goal.title,
                category=goal.category.value,
                status="rejected",
                version_before=goal.version_before,
                version_after=goal.version_before,
                files_affected=goal.target_files,
                notes=goal.notes,
            ), report_only=report_only)
            report = EvolutionSummaryReport(
                proposal_id=goal.goal_id,
                target_component=str(goal.target_files),
                description=goal.description,
                stage=EvolutionStage.REJECTED,
                error_reason=str(e),
            )
            return report

        if not target_files:
            goal.status = GoalStatus.FAILED
            goal.notes = "No candidate target files could be determined for this goal."
            self._update_goal(goal)
            report = EvolutionSummaryReport(
                proposal_id=goal.goal_id,
                target_component="unknown",
                description=goal.description,
                stage=EvolutionStage.REJECTED,
                error_reason=goal.notes,
            )
            return report

        primary_target = target_files[0]
        goal.status = GoalStatus.SYNTHESIZING
        self._update_goal(goal)

        # 2. Patch resolution — a supplied patch wins; otherwise the LLM must
        # actually write one. If it cannot, we fail honestly (never a comment stub).
        code_diff = custom_patch
        if not code_diff:
            synthesis = await self._synthesize_patch(goal, target_files, test_paths=test_paths)
            if synthesis is None or synthesis.proposal is None:
                reason = getattr(synthesis, "reason", "no patch synthesized")
                goal.notes = f"Patch synthesis failed: {reason}"
                if goal.attempts >= goal.max_attempts:
                    goal.status = GoalStatus.FAILED
                else:
                    goal.status = GoalStatus.QUEUED
                self._update_goal(goal)
                report = EvolutionSummaryReport(
                    proposal_id=goal.goal_id,
                    target_component=", ".join(target_files),
                    description=goal.description,
                    stage=EvolutionStage.REJECTED,
                    error_reason=goal.notes,
                )
                self._record_entry(JournalEntry(
                    entry_id=f"ent-{uuid.uuid4().hex[:8]}",
                    goal_id=goal.goal_id,
                    title=goal.title,
                    category=goal.category.value,
                    status="rolled_back",
                    version_before=goal.version_before,
                    version_after=goal.version_before,
                    files_affected=goal.target_files,
                    notes=goal.notes,
                ), report_only=report_only)
                return report
            code_diff = synthesis.proposal.diff
            self.last_patch_proposal = synthesis.proposal
            synth_files = list(synthesis.proposal.files)
        else:
            synth_files = [primary_target]

        goal.status = GoalStatus.TESTING
        self._update_goal(goal)

        # 3. Evolution cycle via CodebaseEvolver.
        # A synthesized LLM patch may legitimately touch several files. When it
        # does (or when the target is the signal target rather than a single
        # primary), snapshot + apply ALL touched files so a report-only rollback
        # can restore every one of them — not just the primary target.
        diff_files = self.evolver._diff_files(code_diff)
        apply_files = list(dict.fromkeys(diff_files or (synth_files or [primary_target])))
        if not apply_files:
            apply_files = [primary_target]
        if len(apply_files) > 1:
            preflight = self.evolver.apply_patch_multi(apply_files, code_diff)
            if not preflight[0]:
                goal.notes = f"Multi-file patch failed to apply: {preflight[1][:200]}"
                goal.status = GoalStatus.QUEUED if goal.attempts < goal.max_attempts else GoalStatus.FAILED
                self._update_goal(goal)
                return EvolutionSummaryReport(
                    proposal_id=goal.goal_id,
                    target_component=", ".join(apply_files),
                    description=goal.description,
                    stage=EvolutionStage.ROLLED_BACK,
                    error_reason=goal.notes,
                )
        report: EvolutionSummaryReport = self.evolver.evolve(
            target_component=primary_target,
            description=f"[{goal.category.value}] {goal.title}: {goal.description}",
            code_diff=code_diff,
            auto_promote=auto_promote,
            push=push,
            test_paths=test_paths,
            pre_applied_files=apply_files if len(apply_files) > 1 else None,
            report_only=report_only,
        )

        # 4. Process outcome & Version Progression
        if report.stage == EvolutionStage.PROMOTED:
            # Determine version bump magnitude
            if goal.category == GoalCategory.CAPABILITY_ADD:
                v_new = self.version_tracker.bump_minor(
                    goal_id=goal.goal_id,
                    title=goal.title,
                    git_commit=report.git_commit_hash,
                    tag=push,
                )
            else:
                v_new = self.version_tracker.bump_patch(
                    goal_id=goal.goal_id,
                    title=goal.title,
                    git_commit=report.git_commit_hash,
                    tag=push,
                )

            goal.status = GoalStatus.PROMOTED
            goal.completed_at = _now()
            goal.version_after = v_new
            goal.evolution_report = report.to_markdown()
            goal.notes = f"Verified with {report.test_result.passed_tests}/{report.test_result.total_tests} tests passing."
            self._update_goal(goal)

            # Record in journal and append live notes to evolution.md
            self._record_entry(JournalEntry(
                entry_id=f"ent-{uuid.uuid4().hex[:8]}",
                goal_id=goal.goal_id,
                title=goal.title,
                category=goal.category.value,
                status="promoted",
                version_before=goal.version_before,
                version_after=v_new,
                files_affected=report.files_affected,
                lines_added=report.lines_added,
                lines_removed=report.lines_removed,
                test_summary=f"Tests: {report.test_result.passed_tests} passed in {report.test_result.duration_seconds:.2f}s",
                notes=goal.description,
                commit_hash=report.git_commit_hash,
            ), report_only=report_only)

        elif report.stage == EvolutionStage.AWAITING_APPROVAL:
            goal.evolution_report = report.to_markdown()
            if report_only:
                goal.status = GoalStatus.VERIFIED
                goal.notes = (
                    "Verified diff and test receipts were recorded in the evolution report. "
                    "The working tree was restored; no edits were applied, committed, or pushed."
                )
            else:
                goal.status = GoalStatus.AWAITING_APPROVAL
                try:
                    goal.approval_manifest = self._make_approval_manifest(report)
                    goal.notes = (
                        "Verified edits remain in the working tree and are awaiting "
                        "manual operator approval; they have not been rolled back."
                    )
                except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                    goal.approval_manifest = {}
                    goal.notes = (
                        "Verified edits remain in the working tree, but an approval "
                        f"manifest could not be captured: {exc}. Promotion is blocked "
                        "until the goal is reverified."
                    )
            self._update_goal(goal)

        else:
            # Failed or rolled back
            if goal.attempts >= goal.max_attempts:
                goal.status = GoalStatus.FAILED
            else:
                goal.status = GoalStatus.QUEUED  # keep in queue for next cycle retry

            goal.evolution_report = report.to_markdown()
            goal.notes = f"Attempt {goal.attempts}/{goal.max_attempts} failed: {report.error_reason[:300]}"
            self._update_goal(goal)

            # Record rollback in journal and append failure note to evolution.md
            self._record_entry(JournalEntry(
                entry_id=f"ent-{uuid.uuid4().hex[:8]}",
                goal_id=goal.goal_id,
                title=goal.title,
                category=goal.category.value,
                status="rolled_back",
                version_before=goal.version_before,
                version_after=goal.version_before,
                files_affected=report.files_affected,
                lines_added=report.lines_added,
                lines_removed=report.lines_removed,
                test_summary=f"FAILED (Exit Code {report.test_result.failed_tests} failures)",
                notes=goal.notes,
            ), report_only=report_only)

        return report

    def _record_entry(self, entry: JournalEntry, *, report_only: bool = False) -> None:
        """Write a journal entry unless this run is report-only (no side-effects)."""
        if report_only:
            return
        self.journal.record_entry(entry)

    async def _synthesize_patch(
        self,
        goal: EvolutionGoal,
        target_files: list[str],
        failure_digest: str = "",
        test_paths: list[str] | None = None,
    ) -> Any:
        """Ask the LLM to write a real unified diff for the goal.

        Returns a ``PatchSynthesisResult``. Unlike the old comment-append stub,
        this never fabricates a change: no LLM, no diff, or a rejected diff all
        surface as an honest failure.
        """
        from sonic.evolution.patch_synthesizer import PatchSynthesizer

        llm = self.llm_client
        if llm is None:
            llm = self._resolve_llm()
        context_test_paths = test_paths if test_paths is not None else self.test_paths
        context_test_paths = self.evolver._validate_test_paths(context_test_paths)
        synthesizer = PatchSynthesizer(
            llm=llm,
            repo_root=self.repo_root,
            is_protected=self.evolver.is_protected,
            max_tokens=self.patch_max_tokens,
        )
        return await synthesizer.synthesize(
            goal,
            target_files,
            max_attempts=self.patch_max_attempts,
            test_paths=context_test_paths,
            failure_digest=failure_digest or self._feedback_for(goal),
        )

    @staticmethod
    def _feedback_for(goal: EvolutionGoal) -> str:
        """Feed a previous attempt's real failure back to the LLM.

        On a retry (attempts > 1) the goal's notes hold the concrete failure
        (usually the pytest output), so the model can fix the actual error
        instead of re-emitting the same wrong patch. First attempts get nothing.
        """
        if getattr(goal, "attempts", 0) and goal.attempts > 1 and goal.notes:
            return str(goal.notes)[:3000]
        return ""

    def _resolve_llm(self) -> Any:
        """Resolve a Model Router lazily so the coder works even in the being loop."""
        if self.llm_client is not None:
            return self.llm_client
        try:
            from sonic.config import CONFIGS_DIR
            from sonic.llm.router import ModelRouter

            router = ModelRouter.from_config(CONFIGS_DIR / "models.yaml")
            if getattr(router, "providers", None):
                self.llm_client = router
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("evolution_llm_resolve_failed", error=str(exc))
        return self.llm_client

    # ------------------------------------------------------------------
    # 4. Continuous Runner
    # ------------------------------------------------------------------
    async def run_continuous(
        self,
        max_goals: int | None = None,
        push: bool = False,
        auto_promote: bool = False,
        report_only: bool | None = None,
    ) -> list[EvolutionSummaryReport]:
        """
        Continuously processes goals from the queue in priority order until
        the queue is exhausted or max_goals is reached.

        By default, verified diffs and their receipts are recorded without
        changing the checkout. Applying and committing requires the explicit
        ``auto_promote=True`` option; pushing additionally requires ``push=True``.
        """
        if report_only is None:
            report_only = not auto_promote
        if push and not auto_promote:
            raise ValueError("push requires explicit auto_promote=True")
        if report_only and auto_promote:
            raise ValueError("report_only cannot be combined with auto_promote")

        reports: list[EvolutionSummaryReport] = []
        processed = 0

        logger.info("evolution_continuous_runner_started", max_goals=max_goals)

        while True:
            if max_goals is not None and processed >= max_goals:
                break

            goal = self.get_next_queued_goal()
            if not goal:
                break

            logger.info("evolution_processing_goal", goal_id=goal.goal_id, title=goal.title)
            rep = await self.execute_goal(
                goal.goal_id,
                auto_promote=auto_promote,
                push=push,
                report_only=report_only,
            )
            reports.append(rep)
            processed += 1

        logger.info("evolution_continuous_runner_finished", goals_processed=processed)
        return reports
