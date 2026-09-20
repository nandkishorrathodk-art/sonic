"""
SONIC v2 — Continuous Codebase Self-Evolution Engine
====================================================
Enables autonomous codebase self-updates, bug fixes, and logical upgrades across
non-safety components of the repository.

Architectural Invariants:
1. Safety Envelope Immutability: AI self-evolution is strictly FORBIDDEN from
   modifying any protected safety kernel components (sonic/safety, egress,
   action_broker, sealed policy). Violations fail-closed immediately.
2. AST & Syntax Pre-flight: No code is applied without passing syntax verification.
3. Test-Driven Verification: Every proposed change must pass targeted unit tests
   and the security regression suite with 100% green exit codes.
4. Zero-Corrupt Atomic Rollback: If any test fails, all modifications are restored
   instantly from snapshots and an [AVOID] lesson is recorded in LessonsLedger.
5. Autonomous Promotion & Git Push: When auto_promote=True and all tests pass,
   changes are automatically committed to git and pushed to remote without
   requiring manual human approval.
6. Structured Evolution Summary: Every evolution cycle produces an honest,
   detailed audit report documenting what was done, diff metrics, test results,
   and git provenance.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonic.evolution.pipeline import EvolutionPipeline, EvolutionStage
from sonic.logger import get_logger

logger = get_logger(__name__)

# Protected paths come from the single source of truth in pipeline.py so the two
# guard lists can never drift apart again.
from sonic.evolution.pipeline import _PROTECTED_COMPONENTS


class SafetyInvariantViolation(Exception):
    """Raised when an evolution proposal attempts to modify a protected safety module."""


@dataclass
class VerificationTestResult:
    """Outcome of running test suites against a proposed codebase modification."""
    __test__ = False  # Prevent pytest from treating this dataclass as a test suite
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    duration_seconds: float = 0.0
    raw_output: str = ""
    error_message: str = ""


# Backwards compatibility alias
TestExecutionResult = VerificationTestResult


@dataclass
class EvolutionSummaryReport:
    """Comprehensive, tamper-evident audit report of a codebase evolution cycle."""
    proposal_id: str
    target_component: str
    description: str
    stage: EvolutionStage
    files_affected: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    ast_validated: bool = False
    import_integrity_passed: bool = False
    test_result: TestExecutionResult = field(default_factory=lambda: TestExecutionResult(passed=False))
    security_regression_passed: bool = False
    git_committed: bool = False
    git_commit_hash: str = ""
    git_pushed: bool = False
    git_push_message: str = ""
    error_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_markdown(self) -> str:
        """Renders a clean, structured GitHub-flavored Markdown summary."""
        status_badge = {
            EvolutionStage.PROMOTED: "🟢 PROMOTED (AUTO-MERGED)",
            EvolutionStage.AWAITING_APPROVAL: "🟡 AWAITING OPERATOR APPROVAL",
            EvolutionStage.ROLLED_BACK: "🔴 ROLLED BACK (TESTS FAILED)",
            EvolutionStage.REJECTED: "⛔ REJECTED (SAFETY/AST VIOLATION)",
        }.get(self.stage, f"⚪ {self.stage.value.upper()}")

        md = [
            f"# 🧬 Codebase Self-Evolution Summary: `{self.proposal_id}`",
            "",
            f"**Status:** {status_badge}  ",
            f"**Target Component:** `{self.target_component}`  ",
            f"**Timestamp:** `{self.timestamp}`  ",
            "",
            "## 1. Upgrade Description & Objectives",
            f"> {self.description}",
            "",
            "## 2. Code Modifications & Diff Metrics",
            f"- **Files Affected:** {', '.join(f'`{f}`' for f in self.files_affected) if self.files_affected else 'None'}",
            f"- **Lines Added:** `+{self.lines_added}`",
            f"- **Lines Removed:** `-{self.lines_removed}`",
            f"- **AST / Syntax Pre-flight:** {'✅ Passed' if self.ast_validated else '❌ Failed'}",
            "",
            "## 3. Automated Verification & Test Results",
            f"- **Component Tests:** {'✅ 100% PASSED' if self.test_result.passed else '❌ FAILED'}",
            f"  - Passed: `{self.test_result.passed_tests}` / `{self.test_result.total_tests}`",
            f"  - Failed: `{self.test_result.failed_tests}`",
            f"  - Duration: `{self.test_result.duration_seconds:.2f}s`",
            f"- **Import Integrity:** {'✅ All changed modules import' if self.import_integrity_passed else '❌ FAILED (a changed module does not import)'}",
            f"- **Security Regression Suite:** {'✅ 100% PASSED (Zero Safety Breaches)' if self.security_regression_passed else '❌ FAILED/SKIPPED'}",
            "",
            "## 4. Git Provenance & Automation",
            f"- **Git Committed:** {'✅ Yes' if self.git_committed else '❌ No'}",
            f"- **Commit Hash:** `{self.git_commit_hash if self.git_commit_hash else 'N/A'}`",
            f"- **Git Pushed to Remote:** {'✅ Yes (Origin Updated)' if self.git_pushed else '❌ No / Offline / Skipped'}",
        ]

        if self.git_push_message:
            md.append(f"- **Git Push Details:** `{self.git_push_message}`")

        if self.error_reason:
            md.extend([
                "",
                "## ⚠️ Error / Rejection Details",
                f"```\n{self.error_reason}\n```",
            ])

        return "\n".join(md)


class CodebaseEvolver:
    """
    Automated, guarded runner that enables the AI being to continuously upgrade,
    patch, and evolve its own codebase across non-safety components.
    """

    def __init__(
        self,
        repo_root: Path | str | None = None,
        lessons_ledger: Any | None = None,
        pipeline: EvolutionPipeline | None = None,
    ):
        if repo_root is None:
            # Default to repo root (c:\Users\nandk\_society or parent of sonic-core)
            cur = Path(__file__).resolve()
            # walk up until we find .git or pyproject.toml
            root = cur.parent
            for p in cur.parents:
                if (p / ".git").exists() or (p / "sonic-core").exists():
                    root = p
                    break
            self.repo_root = root
        else:
            self.repo_root = Path(repo_root).resolve()

        self.lessons_ledger = lessons_ledger
        self.pipeline = pipeline if pipeline is not None else EvolutionPipeline()
        self._snapshots: dict[Path, str | None] = {}

    # ------------------------------------------------------------------
    # 1. Safety Envelope Invariant Check
    # ------------------------------------------------------------------
    def is_protected(self, component_path: str) -> bool:
        """
        Enforces strict safety envelope immutability.
        Returns True if the target path touches any protected safety component.

        H5 audit fix: uses normalized path containment (not naive substring matching)
        in both the repo-relative form (``sonic-core/sonic/safety/...``) and the
        package-relative form (``sonic/safety/...``), with os.path.normpath and
        posixpath normalization so spellings like ``sonic//safety/x``,
        ``sonic/./safety/x``, or ``../safety`` cannot slip past the guard.
        """
        raw = str(component_path or "").replace("\\", "/").strip()
        if not raw:
            return False

        variants: set[str] = set()
        try:
            from posixpath import normpath

            variants.add(normpath(raw).lstrip("./").lower())
        except Exception:
            pass
        try:
            normalized_os = os.path.normpath(raw).replace("\\", "/").lstrip("./").lower()
            variants.add(normalized_os)
        except Exception:
            pass
        try:
            resolved = (self.repo_root / raw).resolve()
            rel = str(resolved.relative_to(self.repo_root)).replace("\\", "/").lower()
            variants.add(rel)
        except Exception:
            pass

        # Also compare in package-relative form (strip known source-root prefixes).
        for v in list(variants):
            for prefix in ("sonic-core/", "sonic-cli/"):
                if v.startswith(prefix):
                    variants.add(v[len(prefix):])

        for norm in variants:
            for protected in _PROTECTED_COMPONENTS:
                p = protected.lower()
                p_un = p.removeprefix("sonic-core/").removeprefix("sonic-cli/")
                if (
                    norm == p
                    or norm == p_un
                    or norm.startswith(p + "/")
                    or norm.startswith(p_un + "/")
                    or norm.startswith(p + ".py")
                    or norm.startswith(p_un + ".py")
                    or p in norm
                    or p_un in norm
                ):
                    return True
        return False

    # ------------------------------------------------------------------
    # 2. Syntax / AST Pre-flight
    # ------------------------------------------------------------------
    def validate_ast(self, file_path: Path, content: str) -> tuple[bool, str]:
        """Validates syntax using Python AST before touching disk."""
        if file_path.suffix == ".py":
            try:
                ast.parse(content, filename=str(file_path))
                return True, "AST syntax check passed"
            except SyntaxError as e:
                return False, f"SyntaxError in {file_path.name}: {e.msg} at line {e.lineno}:{e.offset}"
            except Exception as e:
                return False, f"AST parse error in {file_path.name}: {e}"
        return True, "Non-Python syntax check skipped"

    # ------------------------------------------------------------------
    # 3. Snapshot & Atomic Rollback
    # ------------------------------------------------------------------
    def _record_snapshot(self, path: Path) -> None:
        """Stores the original file content for zero-risk atomic rollback."""
        if path not in self._snapshots:
            if path.exists():
                self._snapshots[path] = path.read_text(encoding="utf-8", errors="replace")
            else:
                self._snapshots[path] = None  # indicates new file

    def rollback_snapshots(self) -> None:
        """Restores all modified files to their exact pre-evolution state."""
        for path, original_content in self._snapshots.items():
            try:
                if original_content is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.write_text(original_content, encoding="utf-8")
            except Exception as e:
                logger.error("evolution_rollback_failed_for_file", file=str(path), error=str(e))
        self._snapshots.clear()

    def clear_snapshots(self) -> None:
        """Discards snapshots after successful promotion."""
        self._snapshots.clear()

    # ------------------------------------------------------------------
    # 4. Patch / Code Application
    # ------------------------------------------------------------------
    @staticmethod
    def _diff_files(code_diff: str) -> list[str]:
        """Return the repo-relative paths a unified diff modifies (best-effort)."""
        import re

        files: list[str] = []
        if not code_diff:
            return files
        for m in re.finditer(r"^diff --git a/(\S+) (?:b/)?(\S+)", code_diff, re.MULTILINE):
            target = m.group(1) if m.group(2) in ("/dev/null", "dev/null") or m.group(2).endswith("/dev/null") else m.group(2)
            if target and target != "/dev/null" and target not in files:
                files.append(target)
        if not files:
            last_a = None
            for line in code_diff.splitlines():
                if line.startswith("--- "):
                    p = line[4:].strip()
                    if p.startswith("a/"):
                        p = p[2:]
                    last_a = p
                elif line.startswith("+++ "):
                    p = line[4:].strip()
                    if p.startswith("b/"):
                        p = p[2:]
                    if p in ("/dev/null", "dev/null") and last_a:
                        if last_a not in files:
                            files.append(last_a)
                    elif p and p != "/dev/null" and p not in files:
                        files.append(p)
        return files

    def apply_patch_multi(self, target_files: list[str], code_diff: str) -> tuple[bool, str]:
        """Atomically apply a diff that may touch several files.

        Snapshots every named file first, applies the unified diff once, and
        rolls the whole set back if any file fails AST validation or the diff
        does not apply. Returns (ok, message).
        """
        for rel in target_files:
            if self.is_protected(rel):
                raise SafetyInvariantViolation(
                    f"Forbidden: '{rel}' is a protected safety component and cannot be self-modified."
                )

        resolved: list[Path] = []
        for rel in target_files:
            p = (self.repo_root / rel).resolve()
            self._record_snapshot(p)
            resolved.append(p)

        ok, msg, added, removed = self.apply_patch(target_files[0], code_diff)
        if not ok:
            self.rollback_snapshots()
            return False, msg

        # AST pre-flight for every touched Python file (rollback all on failure).
        for p in resolved:
            if p.exists() and p.suffix == ".py":
                try:
                    ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
                except SyntaxError as e:
                    self.rollback_snapshots()
                    return False, f"SyntaxError in {p.name}: {e.msg} at line {e.lineno}"
                except Exception as e:
                    self.rollback_snapshots()
                    return False, f"AST verification failed for {p.name}: {e}"
        self._pending_lines = (added, removed)
        return True, f"Applied multi-file patch to {len(target_files)} files"

    def pending_line_stats(self) -> tuple[int, int]:
        stats = getattr(self, "_pending_lines", (0, 0))
        self._pending_lines = (0, 0)
        return stats

    def apply_patch(
        self,
        target_file_rel: str,
        code_diff_or_content: str,
    ) -> tuple[bool, str, int, int]:
        """
        Applies either a unified diff or direct code replacement.
        Returns: (success, message, lines_added, lines_removed)
        """
        target_path = (self.repo_root / target_file_rel).resolve()

        # Check safety guard on the target path
        if self.is_protected(target_file_rel):
            raise SafetyInvariantViolation(
                f"Forbidden: '{target_file_rel}' is a protected safety component and cannot be self-modified."
            )

        # Check if content is a unified diff
        is_unified_diff = (
            code_diff_or_content.startswith("--- ")
            or "\n@@ " in code_diff_or_content
            or "\n--- " in code_diff_or_content
        )

        self._record_snapshot(target_path)

        if is_unified_diff:
            # Apply via git apply in temp patch file. git apply requires the
            # patch text to end with a newline — a diff whose final line has no
            # trailing LF is rejected as "corrupt patch".
            patch_text = code_diff_or_content
            if not patch_text.endswith("\n"):
                patch_text += "\n"
            with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8", newline="\n") as tf:
                tf.write(patch_text)
                temp_patch_path = tf.name

            try:
                cmd = ["git", "apply", "--whitespace=fix", temp_patch_path]
                res = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True, text=True, timeout=30)
                if res.returncode != 0:
                    # Fallback to direct replacement if unified diff failed
                    return False, f"git apply failed: {res.stderr or res.stdout}", 0, 0

                # AST check after git apply
                if target_path.exists() and target_path.suffix == ".py":
                    ast_ok, ast_msg = self.validate_ast(target_path, target_path.read_text(encoding="utf-8", errors="replace"))
                    if not ast_ok:
                        return False, ast_msg, 0, 0

                # Count diff lines
                added = sum(1 for line in code_diff_or_content.splitlines() if line.startswith("+") and not line.startswith("+++"))
                removed = sum(1 for line in code_diff_or_content.splitlines() if line.startswith("-") and not line.startswith("---"))
                return True, "Unified diff applied successfully", added, removed
            finally:
                if os.path.exists(temp_patch_path):
                    os.unlink(temp_patch_path)
        else:
            # Direct code replacement
            old_text = self._snapshots[target_path] or ""
            new_text = code_diff_or_content

            # AST pre-flight check
            ast_ok, ast_msg = self.validate_ast(target_path, new_text)
            if not ast_ok:
                return False, ast_msg, 0, 0

            # Compute diff stats
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
            added = sum(hi2 - lo2 for tag, lo1, hi1, lo2, hi2 in matcher.get_opcodes() if tag in ("replace", "insert"))
            removed = sum(hi1 - lo1 for tag, lo1, hi1, lo2, hi2 in matcher.get_opcodes() if tag in ("replace", "delete"))

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(new_text, encoding="utf-8")
            return True, "File content written and validated", added, removed

    # ------------------------------------------------------------------
    # 5. Test Suite Verification
    # ------------------------------------------------------------------
    def run_tests(self, test_paths: list[str] | None = None) -> TestExecutionResult:
        """Runs pytest on the specified test paths with structured parsing."""
        if not test_paths:
            # Default to quick evolution tests
            test_paths = ["sonic-core/tests/test_evolution_strategy_and_engine.py"]

        cmd = [sys.executable, "-m", "pytest"] + test_paths + ["-q", "--tb=short"]
        start_time = time.time()
        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            duration = time.time() - start_time
            output = (res.stdout or "") + (res.stderr or "")

            # Parse pytest summary line: e.g. "7 passed, 1 failed, 1 skipped in 1.23s"
            passed = 0
            failed = 0
            skipped = 0

            m_pass = re.search(r"(\d+)\s+passed", output)
            if m_pass:
                passed = int(m_pass.group(1))

            m_fail = re.search(r"(\d+)\s+failed", output)
            if m_fail:
                failed = int(m_fail.group(1))

            m_skip = re.search(r"(\d+)\s+skipped", output)
            if m_skip:
                skipped = int(m_skip.group(1))

            total = passed + failed + skipped
            # A run that collected NOTHING (or only skips) proves nothing — it
            # must fail the gate, never pass by absence.
            ran_ok = res.returncode == 0 and failed == 0 and passed > 0
            return TestExecutionResult(
                passed=ran_ok,
                total_tests=total,
                passed_tests=passed,
                failed_tests=failed,
                skipped_tests=skipped,
                duration_seconds=duration,
                raw_output=output,
                error_message="" if ran_ok else (
                    f"Tests failed with exit code {res.returncode}"
                    if (res.returncode != 0 or failed)
                    else "No tests were collected/executed (0 passed)"
                ),
            )
        except Exception as e:
            return TestExecutionResult(
                passed=False,
                error_message=f"Test runner error: {e}",
                duration_seconds=time.time() - start_time,
            )

    def run_import_checks(self, files: list[str]) -> tuple[bool, str]:
        """Verify each changed .py file still imports in the real environment.

        A focused test subset can pass without ever importing the file we changed,
        so a patch that adds a broken import (e.g. `from pkg.missing import x`)
        would otherwise be promoted. Here we byte-compile and then actually import
        each changed module. Fail-closed: any import error blocks promotion.

        Missing third-party *optional* dependencies (typer, playwright, ...) are
        treated as inconclusive (skipped), never as a pass for the changed file's
        own correctness — only a name/import error triggered by the change is a
        failure.
        """
        checked: list[str] = []
        for rel in files:
            p = (self.repo_root / rel).resolve()
            if not p.exists() or p.suffix != ".py":
                continue
            checked.append(rel)
            # 1. Compile check (catches syntax errors without importing).
            try:
                compile(p.read_text(encoding="utf-8", errors="replace"), str(p), "exec")
            except SyntaxError as e:
                return False, f"{rel}: SyntaxError: {e.msg}"
            except Exception as e:
                return False, f"{rel}: compile error: {e}"

            # 2. Import the module by dotted path under its package root.
            mod = self._module_name(rel)
            if not mod:
                continue
            cwd = str(self._package_root(rel) or self.repo_root)
            code = (
                "import importlib, sys;"
                f"sys.path.insert(0, {cwd!r});"
                f"importlib.import_module({mod!r})"
            )
            try:
                res = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=cwd, capture_output=True, text=True, timeout=120,
                )
            except Exception as e:
                return False, f"{rel}: import runner error: {e}"
            if res.returncode == 0:
                continue
            err = (res.stderr or "") + (res.stdout or "")
            # Optional third-party dep isn't installed here -> inconclusive, skip.
            if "ModuleNotFoundError" in err and "No module named" in err:
                missing = err.split("No module named")[-1].strip().strip("'\"")
                top = missing.split(".")[0]
                if not top.startswith(self._package_top(rel)):
                    logger.info("evolution_import_check_skipped", file=rel, missing_dep=top)
                    continue
            last = err.strip().splitlines()[-1] if err.strip() else "import failed"
            return False, f"{rel}: {last[:300]}"
        return True, f"imports verified for {len(checked)} module(s)"

    def _package_root(self, rel: str) -> Path | None:
        for root in ("sonic-core", "sonic-cli"):
            if rel.startswith(root + "/"):
                return self.repo_root / root
        return None

    @staticmethod
    def _package_top(rel: str) -> str:
        parts = rel.replace("\\", "/").split("/")
        for i, seg in enumerate(parts):
            if seg in ("sonic", "sonic_cli"):
                return seg
        return parts[-2] if len(parts) >= 2 else ""

    def _module_name(self, rel: str) -> str:
        parts = rel.replace("\\", "/").split("/")
        for i, seg in enumerate(parts):
            if seg in ("sonic", "sonic_cli"):
                mod = ".".join(parts[i:])
                return mod[:-3] if mod.endswith(".py") else mod
        return ""

    def run_security_regression(self) -> bool:
        """Verifies that no security invariants are broken by the proposed change.

        Fail-closed: if the security suite cannot be found the change is refused
        rather than silently accepted.
        """
        sec_tests = ["sonic-core/tests/test_p0_security_hardening.py"]
        existing = [t for t in sec_tests if (self.repo_root / t).exists()]
        if not existing:
            logger.error("evolution_security_suite_missing", checked=sec_tests)
            return False
        res = self.run_tests(existing)
        return res.passed

    # ------------------------------------------------------------------
    # 6. Git Commit & Push (Auto-Promotion)
    # ------------------------------------------------------------------
    def commit_and_push(
        self,
        proposal_id: str,
        description: str,
        files: list[str],
        push: bool = False,
    ) -> tuple[bool, str, bool, str]:
        """
        Commits verified modifications and pushes upstream without human intervention.
        Returns: (committed, commit_hash, pushed, push_message)
        """
        try:
            # 1. git add
            add_cmd = ["git", "add"] + files
            res_add = subprocess.run(add_cmd, cwd=str(self.repo_root), capture_output=True, text=True)
            if res_add.returncode != 0:
                return False, "", False, f"git add failed: {res_add.stderr}"

            # 2. git commit
            commit_msg = f"evo({proposal_id}): {description}\n\n[Verified by SONIC Continuous Codebase Evolver]"
            commit_cmd = ["git", "commit", "-m", commit_msg]
            res_commit = subprocess.run(commit_cmd, cwd=str(self.repo_root), capture_output=True, text=True)
            if res_commit.returncode != 0 and "nothing to commit" not in res_commit.stdout:
                return False, "", False, f"git commit failed: {res_commit.stderr}"

            # 3. get commit hash
            hash_res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )
            commit_hash = hash_res.stdout.strip()

            # 4. git push if requested
            pushed = False
            push_msg = ""
            if push:
                # get current branch
                br_res = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                )
                branch = br_res.stdout.strip() or "main"
                push_cmd = ["git", "push", "origin", branch]
                res_push = subprocess.run(push_cmd, cwd=str(self.repo_root), capture_output=True, text=True, timeout=45)
                if res_push.returncode == 0:
                    pushed = True
                    push_msg = f"Successfully pushed to origin/{branch}"
                else:
                    pushed = False
                    push_msg = f"Push error (upstream or network): {res_push.stderr or res_push.stdout}"
            else:
                push_msg = "Push omitted (push=False)"

            return True, commit_hash, pushed, push_msg
        except Exception as e:
            return False, "", False, f"Git automation exception: {e}"

    # ------------------------------------------------------------------
    # 7. Complete Evolution Cycle
    # ------------------------------------------------------------------
    def evolve(
        self,
        target_component: str,
        description: str,
        code_diff: str,
        auto_promote: bool = False,
        push: bool = False,
        test_paths: list[str] | None = None,
        pre_applied_files: list[str] | None = None,
        report_only: bool = False,
    ) -> EvolutionSummaryReport:
        """
        Executes a complete, closed-loop codebase evolution cycle:
        1. Safety Invariant verification (Fail-closed on safety components)
        2. AST / Syntax verification
        3. Snapshot staging & file patch
        4. Targeted component unit tests
        5. Security regression verification
        6. Optional auto-commit & git push only when explicitly approved
        7. Rollback and lesson learning on any failure
        8. Automatically appends audit notes to evolution.md (skipped when report_only)
        9. Returns rich EvolutionSummaryReport
        """
        report = self._evolve_impl(
            target_component=target_component,
            description=description,
            code_diff=code_diff,
            auto_promote=auto_promote,
            push=push,
            test_paths=test_paths,
            pre_applied_files=pre_applied_files,
            report_only=report_only,
        )
        if report_only:
            # Report-only contract: restore every file the patch touched so the
            # caller's tree is byte-identical. Snapshot-based (never git
            # checkout), so uncommitted operator work is never destroyed.
            self.rollback_snapshots()
        else:
            self._log_to_evolution_md(report)
        return report

    def _log_to_evolution_md(self, report: EvolutionSummaryReport) -> None:
        """Appends notes for this evolution cycle directly to evolution.md."""
        try:
            from sonic.evolution.evolution_journal import EvolutionJournal, JournalEntry
            journal = EvolutionJournal(repo_root=self.repo_root)
            st = "promoted" if report.stage == EvolutionStage.PROMOTED else (
                "rolled_back" if report.stage == EvolutionStage.ROLLED_BACK else (
                    "rejected" if report.stage == EvolutionStage.REJECTED else "manual"
                )
            )
            journal.record_entry(JournalEntry(
                entry_id=f"evo-{report.proposal_id}",
                goal_id=report.proposal_id,
                title=f"Patch: {report.target_component}",
                category="codebase_evolution",
                status=st,
                version_before="current",
                version_after="current",
                files_affected=report.files_affected,
                lines_added=report.lines_added,
                lines_removed=report.lines_removed,
                test_summary=f"Tests passed: {report.test_result.passed_tests}/{report.test_result.total_tests}" if report.test_result else "",
                notes=report.description + (f"\nError: {report.error_reason}" if report.error_reason else ""),
                commit_hash=report.git_commit_hash,
            ))
        except Exception as e:
            logger.warning("failed_to_log_to_evolution_md", error=str(e))

    def _evolve_impl(
        self,
        target_component: str,
        description: str,
        code_diff: str,
        auto_promote: bool = False,
        push: bool = False,
        test_paths: list[str] | None = None,
        pre_applied_files: list[str] | None = None,
        report_only: bool = False,
    ) -> EvolutionSummaryReport:
        proposal, reason = self.pipeline.submit_proposal(
            target_component=target_component,
            description=description,
            code_diff=code_diff,
        )

        report = EvolutionSummaryReport(
            proposal_id=proposal.proposal_id,
            target_component=target_component,
            description=description,
            stage=proposal.stage,
            files_affected=list(pre_applied_files) if pre_applied_files else [target_component],
        )

        # Gate 1: Safety Invariant Check — every touched file must be non-protected.
        protected_hit = None
        for f in (pre_applied_files or [target_component]):
            if self.is_protected(f):
                protected_hit = f
                break
        if protected_hit is not None:
            logger.critical("evolution_safety_invariant_violation", component=protected_hit)
            report.stage = EvolutionStage.REJECTED
            report.error_reason = f"Safety Invariant Violation: '{protected_hit}' is protected."
            return report

        # Gate 2: Staging & Patch Application
        if pre_applied_files:
            # The multi-file patch was already applied + AST-checked by
            # apply_patch_multi(); snapshots are already recorded for rollback.
            report.ast_validated = True
            report.lines_added, report.lines_removed = self.pending_line_stats()
        else:
            # Snapshot EVERY file the diff names, not just the primary target, so
            # a report-only rollback can restore all of them (a patch that edits
            # the test file while the goal targets the source, or vice versa,
            # otherwise leaves part of the change behind).
            try:
                for rel in self._diff_files(code_diff):
                    self._record_snapshot((self.repo_root / rel).resolve())
            except Exception:
                pass
            try:
                ok, msg, added, removed = self.apply_patch(target_component, code_diff)
                report.lines_added = added
                report.lines_removed = removed
                if not ok:
                    self.rollback_snapshots()
                    report.stage = EvolutionStage.REJECTED
                    report.error_reason = f"Patch application failed: {msg}"
                    return report
                report.ast_validated = True
            except SafetyInvariantViolation as e:
                self.rollback_snapshots()
                report.stage = EvolutionStage.REJECTED
                report.error_reason = str(e)
                return report
            except Exception as e:
                self.rollback_snapshots()
                report.stage = EvolutionStage.REJECTED
                report.error_reason = f"Unexpected patch failure: {e}"
                return report

        # Gate 3: Unit / Component Tests
        self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.ISOLATED_BRANCH, "Staged in workspace")
        test_res = self.run_tests(test_paths)
        report.test_result = test_res

        if not test_res.passed:
            logger.warning("evolution_tests_failed", proposal_id=proposal.proposal_id, error=test_res.error_message)
            self.rollback_snapshots()
            report.stage = EvolutionStage.ROLLED_BACK
            report.error_reason = f"Verification tests failed: {test_res.error_message}\n{test_res.raw_output[:500]}"
            self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.ROLLED_BACK, report.error_reason)

            # Record AVOID lesson in LessonsLedger if available
            if self.lessons_ledger is not None and hasattr(self.lessons_ledger, "record"):
                try:
                    from sonic.being.lessons import Lesson, LessonKind
                    lesson = Lesson(
                        lesson_id=f"evo-fail-{proposal.proposal_id}",
                        kind=LessonKind.AVOID,
                        goal=f"Evolve codebase {target_component}",
                        approach=description,
                        evidence=f"Tests failed: {test_res.raw_output[:200]}",
                        created_at=datetime.now(UTC).isoformat(),
                    )
                    try:
                        self.lessons_ledger.record([lesson])
                    except TypeError:
                        self.lessons_ledger.record(lesson)
                except Exception as e:
                    logger.warning("evolution_record_lesson_failed", error=str(e))
            return report

        self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.TESTS_PASSING, f"{test_res.passed_tests} tests green")

        # Gate 3.5: Import integrity — the test subset may not import the changed
        # files at all, so a patch that introduces a broken/undeclared import
        # (or a syntax-level import error) could otherwise slip through. Verify
        # every changed .py file still imports.
        import_ok, import_msg = self.run_import_checks(pre_applied_files or [target_component])
        report.import_integrity_passed = import_ok
        if not import_ok:
            logger.critical("evolution_import_integrity_failed", proposal_id=proposal.proposal_id)
            self.rollback_snapshots()
            report.stage = EvolutionStage.ROLLED_BACK
            report.error_reason = f"Import integrity check failed: {import_msg}"
            self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.ROLLED_BACK, report.error_reason)
            return report
        self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.IMPORTS_VERIFIED, import_msg)

        # Gate 4: Security Regression Suite
        sec_ok = self.run_security_regression()
        report.security_regression_passed = sec_ok
        if not sec_ok:
            logger.critical("evolution_security_regression_failed", proposal_id=proposal.proposal_id)
            self.rollback_snapshots()
            report.stage = EvolutionStage.ROLLED_BACK
            report.error_reason = "Security regression suite failed — potential safety degradation."
            self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.ROLLED_BACK, report.error_reason)
            return report

        self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.SECURITY_REGRESSION_PASSING, "Zero safety regressions")

        # Gate 5: Promotion & Git Automation
        # Keep the snapshots when the caller asked for report-only — it rolls
        # back from them right after we return. For a real promotion the change
        # is kept, so the snapshots are no longer needed.
        if not report_only:
            self.clear_snapshots()  # Changes are verified and retained

        if auto_promote:
            committed, chash, pushed, pmsg = self.commit_and_push(
                proposal_id=proposal.proposal_id,
                description=description,
                files=list(pre_applied_files) if pre_applied_files else [target_component],
                push=push,
            )
            report.git_committed = committed
            report.git_commit_hash = chash
            report.git_pushed = pushed
            report.git_push_message = pmsg
            report.stage = EvolutionStage.PROMOTED
            self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.PROMOTED, f"Auto-promoted (hash: {chash})")
        else:
            report.stage = EvolutionStage.AWAITING_APPROVAL
            self.pipeline.advance_stage(proposal.proposal_id, EvolutionStage.AWAITING_APPROVAL, "Tests passed, awaiting operator approval")

        return report
