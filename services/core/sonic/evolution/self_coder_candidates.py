"""Isolated verification and guarded in-place application of self-coder candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonic.evolution.codebase_evolver import CodebaseEvolver
from sonic.evolution.evolution_journal import EvolutionJournal, JournalEntry
from sonic.evolution.pipeline import EvolutionStage
from sonic.evolution.version_tracker import VersionTracker


class CandidateWorkflowError(RuntimeError):
    """A self-coder candidate could not be safely created or promoted."""


def _run_coroutine_sync(coro: Any) -> Any:
    """Execute a coroutine synchronously, handling both non-running and running event loop contexts."""
    def _runner() -> Any:
        from sonic.memory.router import reset_memory_singleton
        try:
            return asyncio.run(coro)
        finally:
            reset_memory_singleton()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_runner).result()
    else:
        return _runner()


class SelfCoderCandidateStore:
    """Persist tested patch/report records outside the checked-out source tree."""

    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()
    _repo_locks: dict[str, threading.Lock] = {}

    def __init__(self, repo_root: Path | str, storage_dir: Path | str | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.storage_dir = Path(storage_dir).resolve() if storage_dir else (
            self.repo_root / "sonic_data" / "self_coder_candidates"
        )
        self.worktrees_dir = self.storage_dir / "worktrees"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, candidate_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", candidate_id):
            raise KeyError("Candidate not found")
        return self.storage_dir / f"{candidate_id}.json"

    def _lock(self, candidate_id: str) -> threading.Lock:
        self._path(candidate_id)
        with self._locks_guard:
            return self._locks.setdefault(candidate_id, threading.Lock())

    def _repo_lock(self) -> threading.Lock:
        key = str(self.repo_root)
        with self._locks_guard:
            return self._repo_locks.setdefault(key, threading.Lock())

    def save(self, record: dict[str, Any]) -> None:
        record["integrity_sha256"] = self._integrity_hash(record)
        path = self._path(str(record["candidate_id"]))
        tmp = path.with_suffix(".json.new")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def get(self, candidate_id: str, tenant_id: str) -> dict[str, Any] | None:
        path = self._path(candidate_id)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("tenant_id") != tenant_id:
            return None
        return record

    def list(self, tenant_id: str) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.storage_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("tenant_id") == tenant_id:
                records.append(record)
        return records

    #: Candidate statuses that mean the signal was already attempted at a commit.
    ATTEMPTED_STATUSES = frozenset({
        "verified",
        "rejected",
        "applied",
        "approved",
        "auto_promoted",
        "failed",
        "failed_revalidation",
        "stale",
        "tampered",
    })

    def attempted_signal_keys(
        self,
        tenant_id: str,
        base_commit: str | None = None,
    ) -> set[tuple[str, str]]:
        """Return ``(target_file, description)`` pairs already attempted.

        Used to advance past a signal that has already been tried so the
        continuous runner does not pin the same signal on every cycle. When
        ``base_commit`` is omitted the current ``HEAD`` is used, matching the
        per-commit dedup applied inside :meth:`run_candidate`; a new commit
        therefore makes every signal eligible again.
        """
        if base_commit is None:
            try:
                base_commit, _ = self._base_state()
            except CandidateWorkflowError:
                base_commit = None
        return {
            (str(record.get("target_file", "")), str(record.get("description", "")))
            for record in self.list(tenant_id)
            if (base_commit is None or record.get("base_commit") == base_commit)
            and record.get("target_file")
            and record.get("description")
            and record.get("status") in self.ATTEMPTED_STATUSES
        }

    def delete(self, candidate_id: str, tenant_id: str) -> bool:
        path = self._path(candidate_id)
        record = self.get(candidate_id, tenant_id)
        if record is None:
            return False
        path.unlink()
        return True

    @staticmethod
    def _integrity_hash(record: dict[str, Any]) -> str:
        immutable_fields = (
            "candidate_id", "tenant_id", "base_commit", "base_branch",
            "target_file", "description", "diff", "files", "test_paths",
            "verification", "diagnosis", "impact",
            "working_tree_changes_excluded", "report", "created_at",
        )
        payload = {key: record.get(key) for key in immutable_fields}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _git(self, args: list[str], cwd: Path | None = None, timeout: int = 60) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(cwd or self.repo_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CandidateWorkflowError(f"git {' '.join(args)} could not complete: {exc}") from exc
        if result.returncode:
            raise CandidateWorkflowError(
                f"git {' '.join(args)} failed ({result.returncode}): "
                f"{(result.stderr or result.stdout).strip()}"
            )
        return result.stdout.strip()

    def _base_state(self) -> tuple[str, str]:
        commit = self._git(["rev-parse", "--verify", "HEAD"])
        branch = self._git(["branch", "--show-current"])
        if not commit or not branch:
            raise CandidateWorkflowError("Cannot identify the active Git branch and base commit")
        return commit, branch

    def _remove_worktree(self, path: Path) -> None:
        import gc
        import os
        import stat
        import time

        for attempt in range(3):
            gc.collect()
            if not path.exists():
                break
            try:
                self._git(["worktree", "remove", "--force", str(path)])
                break
            except Exception:
                def _handle_remove_readonly(func, file_path, exc_info):
                    try:
                        os.chmod(file_path, stat.S_IWRITE)
                        func(file_path)
                    except Exception:
                        pass

                try:
                    if path.exists():
                        shutil.rmtree(str(path), onerror=_handle_remove_readonly)
                        break
                except Exception:
                    if attempt < 2:
                        time.sleep(0.3)

        try:
            self._git(["worktree", "prune"])
        except Exception:
            pass

    async def run_candidate(
        self,
        tenant_id: str,
        max_cycles: int,
        coder_factory: Callable[[Path], Any],
        *,
        auto_apply: bool = False,
        auto_commit: bool = False,
        signals: list[tuple[str, str, str]] | None = None,
    ) -> dict[str, Any]:
        if auto_commit and not auto_apply:
            raise ValueError("auto_commit requires explicit auto_apply=True")
        lock = self._repo_lock()
        await asyncio.to_thread(lock.acquire)
        try:
            return await self._run_candidate_unlocked(
                tenant_id,
                max_cycles,
                coder_factory,
                auto_apply=auto_apply,
                auto_commit=auto_commit,
                signals=signals,
            )
        finally:
            lock.release()

    async def _run_candidate_unlocked(
        self,
        tenant_id: str,
        max_cycles: int,
        coder_factory: Callable[[Path], Any],
        *,
        auto_apply: bool = False,
        auto_commit: bool = False,
        signals: list[tuple[str, str, str]] | None = None,
    ) -> dict[str, Any]:
        if not 1 <= max_cycles <= 10:
            raise ValueError("max_cycles must be between 1 and 10")
        candidate_id = uuid.uuid4().hex
        base_commit, base_branch = self._base_state()
        working_tree_changes = self._git(
            ["status", "--porcelain", "--untracked-files=all"]
        )
        working_tree_changes = [
            line[3:].strip() for line in working_tree_changes.splitlines()
        ]
        tenant_records = self.list(tenant_id)
        # Skip redundant automatic discovery only when the caller has NOT pinned
        # explicit signals. A pinned signal must always be attempted: otherwise a
        # single empty discovery result at a commit would permanently short-circuit
        # every later run (including the continuous runner's pinned retries).
        if signals is None and any(
            record.get("base_commit") == base_commit
            and record.get("status") == "no_candidate"
            for record in tenant_records
        ):
            previous = next(
                record
                for record in reversed(tenant_records)
                if record.get("base_commit") == base_commit
                and record.get("status") == "no_candidate"
            )
            return {
                "candidates": [],
                "report": previous.get("report", ""),
                "status": "no_actionable_signal",
            }
        excluded_signals = {
            (str(record.get("target_file", "")), str(record.get("description", "")))
            for record in tenant_records
            if record.get("base_commit") == base_commit
            and record.get("target_file")
            and record.get("description")
            and record.get("status")
            in {
                "verified",
                "rejected",
                "applied",
                "approved",
                "auto_promoted",
                "failed",
                "failed_revalidation",
                "stale",
                "tampered",
            }
        }
        worktree = self.worktrees_dir / candidate_id
        try:
            self._git(["worktree", "add", "--detach", str(worktree), base_commit])
        except CandidateWorkflowError as exc:
            raise CandidateWorkflowError(f"Could not create isolated candidate worktree: {exc}") from exc
        cleanup_error = ""
        persisted: list[dict[str, Any]] = []
        try:
            coder = coder_factory(worktree)
            try:
                run_kwargs: dict[str, Any] = {
                    "max_cycles": max_cycles,
                    "excluded_signals": excluded_signals,
                }
                if signals is not None:
                    run_kwargs["signals"] = signals
                report = await coder.run(**run_kwargs)
            except Exception as exc:
                raise CandidateWorkflowError(f"Self-coder execution failed: {exc}") from exc
            finally:
                close = getattr(coder, "close", None)
                if callable(close):
                    close()
            cycles = list(getattr(report, "cycles", []))
            for cycle in cycles:
                verification = {
                    "full_test_suite": False,
                    "security_regression": False,
                    "error": "",
                }
                passed = (
                    getattr(cycle, "stage", "") == EvolutionStage.AWAITING_APPROVAL.value
                    and bool(getattr(cycle, "proposed_diff", "").strip())
                    and bool(getattr(cycle, "patch_files", []))
                )
                cycle_id = candidate_id if len(cycles) == 1 else uuid.uuid4().hex
                test_paths = [
                    path
                    for path in ("services/core/tests", "apps/cli/tests")
                    if (worktree / path).is_dir()
                ]
                if passed and not test_paths:
                    passed = False
                    verification["error"] = "Candidate worktree contains no configured test suite"
                if passed:
                    verifier = CodebaseEvolver(repo_root=worktree)
                    applied, apply_message = verifier.apply_patch_multi(
                        list(cycle.patch_files), cycle.proposed_diff
                    )
                    if not applied:
                        passed = False
                        verification["error"] = f"Candidate patch could not be reapplied: {apply_message}"
                    else:
                        imports_ok, import_message = verifier.run_import_checks(
                            list(cycle.patch_files)
                        )
                        tests = verifier.run_tests(test_paths)
                        security_ok = verifier.run_security_regression()
                        verification.update({
                            "full_test_suite": tests.passed,
                            "security_regression": security_ok,
                            "test_count": tests.total_tests,
                            "passed_tests": tests.passed_tests,
                            "failed_tests": tests.failed_tests,
                            "test_output": tests.raw_output[-4000:],
                            "import_check": import_message,
                        })
                        passed = imports_ok and tests.passed and security_ok
                        if not passed:
                            verification["error"] = (
                                import_message if not imports_ok else
                                tests.error_message if not tests.passed else
                                "Security regression suite failed"
                            )
                record = {
                    "candidate_id": cycle_id,
                    "tenant_id": tenant_id,
                    "status": "verified" if passed else "failed",
                    "base_commit": base_commit,
                    "base_branch": base_branch,
                    "target_file": getattr(cycle, "target_file", ""),
                    "description": getattr(cycle, "signal", ""),
                    "diagnosis": getattr(cycle, "diagnosis", ""),
                    "impact": {
                        "assessment": (
                            "Expected to address the observed signal; production "
                            "impact is not measured by this candidate run."
                        ),
                        "basis": getattr(cycle, "diagnosis", ""),
                        "measured": {
                            "files_changed": list(getattr(cycle, "patch_files", [])),
                            "lines_added": int(getattr(cycle, "lines_added", 0)),
                            "lines_removed": int(getattr(cycle, "lines_removed", 0)),
                            "tests_passed": verification.get("passed_tests", 0),
                            "tests_failed": verification.get("failed_tests", 0),
                        },
                    },
                    "diff": getattr(cycle, "proposed_diff", "") if passed else "",
                    "files": list(getattr(cycle, "patch_files", [])) if passed else [],
                    "test_paths": test_paths,
                    "verification": verification,
                    "working_tree_changes_excluded": working_tree_changes,
                    "cycle_stage": getattr(cycle, "stage", ""),
                    "cycle_reason": getattr(cycle, "reason", ""),
                    "report": report.to_text(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "commit": "",
                    "commit_branch": "",
                    "error": "" if passed else (
                        verification["error"] or getattr(cycle, "reason", "")
                        or "Candidate did not pass all verification gates"
                    ),
                }
                self.save(record)
                persisted.append(record)
            if not persisted:
                record = {
                    "candidate_id": candidate_id,
                    "tenant_id": tenant_id,
                    "status": "no_candidate",
                    "base_commit": base_commit,
                    "base_branch": base_branch,
                    "target_file": "",
                    "description": "",
                    "diff": "",
                    "files": [],
                    "test_paths": [],
                    "verification": {},
                    "cycle_stage": "no_candidate",
                    "cycle_reason": "No actionable signal produced a verified patch",
                    "report": report.to_text(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "commit": "",
                    "commit_branch": "",
                    "error": "No actionable signal produced a verified patch",
                }
                self.save(record)
                persisted.append(record)
            report_text = report.to_text()
        finally:
            try:
                self._remove_worktree(worktree)
            except Exception as exc:
                cleanup_error = str(exc)
            if cleanup_error:
                for record in persisted:
                    record["status"] = "failed"
                    record["error"] = f"Candidate worktree cleanup failed: {cleanup_error}"
                    self.save(record)
                raise CandidateWorkflowError(
                    f"Candidate worktree cleanup failed; no candidate is available: {cleanup_error}"
                )
        applied_candidates: list[dict[str, Any]] = []
        for record in persisted:
            if record["status"] == "verified" and auto_apply:
                try:
                    applied_candidates.append(
                        await self._promote(
                            record["candidate_id"],
                            tenant_id,
                            automatic=True,
                            auto_commit=auto_commit,
                        )
                    )
                except Exception as exc:
                    latest = self.get(record["candidate_id"], tenant_id) or record
                    if latest["status"] not in {
                        "applied", "failed_application", "blocked_conflict",
                        "stale", "tampered", "rollback_failed",
                    }:
                        latest["status"] = "application_failed"
                        latest["error"] = f"Could not apply verified candidate: {exc}"
                    latest["application_error"] = str(exc)
                    self.save(latest)
                    applied_candidates.append(latest)
            else:
                applied_candidates.append(record)
        report_text = self._append_application_report(report_text, applied_candidates)
        for record in applied_candidates:
            record["report"] = report_text
            self.save(record)

        from sonic.evolution.graph_recorder import record_verified_candidate

        for record in applied_candidates:
            try:
                await record_verified_candidate(record, self.repo_root)
            except Exception as exc:
                record["graph_record_error"] = (
                    f"Candidate reached status {record.get('status')}, but verified graph "
                    f"persistence failed: {exc}"
                )
                self.save(record)
                raise CandidateWorkflowError(record["graph_record_error"]) from exc

        return {"candidates": applied_candidates, "report": report_text}

    @staticmethod
    def _append_application_report(
        report: str, records: list[dict[str, Any]]
    ) -> str:
        lines = [report.rstrip(), "", "## Candidate impact and verification receipts"]
        for record in records:
            lines.extend([
                f"- Candidate: `{record['candidate_id']}` — **{record['status']}**",
                f"  - Signal: {record.get('description') or record.get('error', 'No actionable signal')}",
                f"  - Diagnosis: {record.get('diagnosis') or 'No completed diagnosis'}",
                f"  - Expected impact: {record.get('impact', {}).get('assessment', 'Not assessed')}",
                f"  - Files: {', '.join(record.get('files', [])) or 'none'}",
                f"  - Verification: {record.get('verification', {})}",
            ])
            if record.get("status") == "verified":
                lines.append(
                    "  - Verified candidate recorded for operator review; "
                    "active working tree unchanged; no commit or push was made"
                )
            elif record.get("status") == "applied":
                if "commit failed" in str(record.get("promotion", "")).lower():
                    lines.append(
                        "  - Verified patch is applied, but the requested local commit failed; "
                        "nothing was pushed"
                    )
                else:
                    lines.append(
                        "  - Verified patch applied to the active working tree; "
                        "no commit or push was made"
                    )
            elif record.get("status") == "promoted":
                lines.append("  - Verified patch was committed locally; nothing was pushed")
            elif record.get("status") == "committed_metadata_failed":
                lines.append(
                    "  - Verified patch was committed locally, but version/journal metadata "
                    "did not fully record; nothing was pushed"
                )
            if record.get("working_tree_changes_excluded"):
                lines.append(
                    "  - Uncommitted active-checkout changes were excluded from "
                    "the candidate base: "
                    + ", ".join(record["working_tree_changes_excluded"])
                )
            if record.get("error"):
                lines.append(f"  - Error: {record['error']}")
        return "\n".join(lines)

    def approve(self, candidate_id: str, tenant_id: str) -> dict[str, Any]:
        return _run_coroutine_sync(self._promote(candidate_id, tenant_id, automatic=False))

    def auto_promote(self, candidate_id: str, tenant_id: str, *, auto_commit: bool = False) -> dict[str, Any]:
        """Apply an already verified candidate to the active working tree."""
        return _run_coroutine_sync(self._promote(candidate_id, tenant_id, automatic=True, auto_commit=auto_commit))

    async def _promote(
        self, candidate_id: str, tenant_id: str, *, automatic: bool, auto_commit: bool = False
    ) -> dict[str, Any]:
        with self._lock(candidate_id):
            record = self.get(candidate_id, tenant_id)
            if record is None:
                raise KeyError("Candidate not found")
            if record["status"] != "verified":
                raise ValueError(f"Candidate cannot be approved from status {record['status']}")
            if record.get("integrity_sha256") != self._integrity_hash(record):
                record["status"] = "tampered"
                record["error"] = "Candidate integrity check failed; exact tested patch is unprovable"
                self.save(record)
                raise ValueError(record["error"])
            current_commit, _ = self._base_state()
            if current_commit != record["base_commit"]:
                record["status"] = "stale"
                record["error"] = "Active base commit changed after candidate verification"
                self.save(record)
                raise ValueError(record["error"])

            evolver = CodebaseEvolver(repo_root=self.repo_root)
            diff = str(record.get("diff", ""))
            if diff and not diff.endswith("\n"):
                diff += "\n"
            files = list(record.get("files", []))
            try:
                diff_paths = evolver._validate_diff_paths(diff, allowed_files=files)
                if set(diff_paths) != {Path(path).as_posix() for path in files}:
                    raise ValueError("Candidate diff paths do not match its verified file set")
                if not files:
                    raise ValueError("Candidate contains no verified target files")
                if not all(record.get("verification", {}).get(key) for key in (
                    "full_test_suite", "security_regression"
                )):
                    raise ValueError("Candidate is missing passing test and security receipts")
            except Exception as exc:
                record["status"] = "failed_revalidation"
                record["error"] = f"Candidate validation failed: {exc}"
                self.save(record)
                raise ValueError(record["error"]) from exc

            snapshots: dict[Path, tuple[bytes | None, int | None]] = {}
            try:
                active_files: list[tuple[Path, bytes | None, int | None]] = []
                for relative_path in diff_paths:
                    path = (self.repo_root / relative_path).resolve()
                    base_bytes = self._git_bytes(["show", f"{current_commit}:{relative_path}"])
                    current_bytes = path.read_bytes() if path.is_file() else None
                    if self._normalize_line_endings(current_bytes) != self._normalize_line_endings(base_bytes):
                        record["status"] = "blocked_conflict"
                        record["error"] = (
                            f"Active worktree conflict on {relative_path}; "
                            "the active file differs from the verified candidate base"
                        )
                        self.save(record)
                        raise ValueError(record["error"])
                    mode = path.stat().st_mode if path.exists() else None
                    active_files.append((path, current_bytes, mode))
                snapshots = {
                    path: (contents, mode)
                    for path, contents, mode in active_files
                }

                check = subprocess.run(
                    ["git", "apply", "--check", "--recount", "--whitespace=fix"],
                    cwd=str(self.repo_root),
                    input=diff,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if check.returncode:
                    raise CandidateWorkflowError(
                        f"Verified patch no longer applies cleanly: "
                        f"{(check.stderr or check.stdout).strip()}"
                    )
                applied = subprocess.run(
                    ["git", "apply", "--recount", "--whitespace=fix"],
                    cwd=str(self.repo_root),
                    input=diff,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if applied.returncode:
                    raise CandidateWorkflowError(
                        f"Applying verified patch failed: "
                        f"{(applied.stderr or applied.stdout).strip()}"
                    )

                post_ok, post_verification = self._verify_active_application(
                    evolver, diff_paths, record.get("test_paths") or []
                )
                record["verification"]["active_worktree"] = post_verification
                if not post_ok:
                    raise CandidateWorkflowError(
                        post_verification.get("error", "Post-application verification failed")
                    )

                if auto_commit:
                    commit_msg = (
                        f"evo({record['candidate_id'][:8]}): {record.get('description', 'Auto-evolution patch')}\n\n"
                        f"[Verified by SONIC Auto-Evolution Engine]\n"
                        f"Candidate: {record['candidate_id']}\n"
                    )
                    commit_cmd = ["git", "commit", "--only", "-m", commit_msg, "--", *diff_paths]
                    commit_proc = subprocess.run(
                        commit_cmd,
                        cwd=str(self.repo_root),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if commit_proc.returncode == 0:
                        new_commit, _ = self._base_state()
                        record["commit"] = new_commit
                        try:
                            record["commit_branch"] = self._git(["branch", "--show-current"])
                        except Exception:
                            record["commit_branch"] = ""

                        try:
                            tracker = VersionTracker(repo_root=self.repo_root)
                            ver_before = tracker.current_version()
                            new_version = tracker.bump_patch(
                                goal_id=record["candidate_id"],
                                title=record.get("description", "Auto-evolution patch"),
                                git_commit=new_commit,
                                metadata={"candidate_id": record["candidate_id"], "files": diff_paths},
                            )
                            journal = EvolutionJournal(repo_root=self.repo_root)
                            journal.record_entry(
                                JournalEntry(
                                    entry_id=f"evo-{record['candidate_id'][:8]}",
                                    goal_id=record["candidate_id"],
                                    title=record.get("description", "Auto-evolution patch"),
                                    category=record.get("category", "logic_improvement"),
                                    status="promoted",
                                    version_before=ver_before,
                                    version_after=new_version,
                                    files_affected=diff_paths,
                                    test_summary=str(record.get("verification", {}).get("test_summary", "Tests passed")),
                                    notes=f"Automated evolution cycle on {', '.join(diff_paths)}",
                                    commit_hash=new_commit,
                                )
                            )
                            record["version_before"] = ver_before
                            record["version_after"] = new_version
                            record["journal_recorded"] = True
                            record["status"] = "promoted"
                            record["promotion"] = (
                                f"verified patch committed locally: {new_commit[:8]}; "
                                f"version {ver_before} -> {new_version}; journal recorded"
                            )
                        except Exception as exc:
                            record["status"] = "committed_metadata_failed"
                            record["journal_recorded"] = False
                            record["promotion_error"] = (
                                f"Commit {new_commit[:8]} succeeded, but version/journal "
                                f"recording failed: {exc}"
                            )
                            record["error"] = record["promotion_error"]
                            record["promotion"] = record["promotion_error"]
                    else:
                        record["status"] = "applied"
                        record["commit"] = ""
                        record["commit_branch"] = ""
                        record["promotion"] = "verified patch applied to active working tree; commit failed"
                else:
                    record["status"] = "applied"
                    record["error"] = ""
                    record["commit"] = ""
                    record["commit_branch"] = ""
                    record["promotion"] = (
                        "verified patch applied to active working tree; no commit or push"
                        if automatic else "operator-approved patch applied to active working tree; no commit or push"
                    )
                self.save(record)
            except Exception as exc:
                if snapshots and not self._restore_exact(snapshots):
                    record["status"] = "rollback_failed"
                    record["error"] = (
                        f"{exc}; exact pre-apply restoration failed for one or more touched files"
                    )
                elif record.get("status") != "blocked_conflict":
                    record["status"] = "failed_application"
                    record["error"] = str(exc)
                self.save(record)
                if record["status"] == "blocked_conflict":
                    raise ValueError(record["error"]) from exc
                return record

            if not automatic and record.get("status") in {"applied", "promoted", "committed_metadata_failed"}:
                from sonic.evolution.graph_recorder import record_verified_candidate

                try:
                    await record_verified_candidate(record, self.repo_root)
                except Exception as exc:
                    record["graph_record_error"] = (
                        f"Candidate reached status {record.get('status')}, but verified graph "
                        f"persistence failed: {exc}"
                    )
                    self.save(record)
                    raise CandidateWorkflowError(record["graph_record_error"]) from exc
            return record

    def _git_bytes(self, args: list[str]) -> bytes | None:
        """Read exact committed bytes, returning None for paths absent at the base."""
        result = subprocess.run(
            ["git", *args],
            cwd=str(self.repo_root),
            capture_output=True,
            timeout=60,
        )
        if result.returncode:
            return None
        return result.stdout

    @staticmethod
    def _normalize_line_endings(contents: bytes | None) -> bytes | None:
        if contents is None:
            return None
        return contents.replace(b"\r\n", b"\n")

    @staticmethod
    def _restore_exact(snapshots: dict[Path, tuple[bytes | None, int | None]]) -> bool:
        """Restore touched paths byte-for-byte, including removing candidate new files."""
        success = True
        for path, (contents, mode) in snapshots.items():
            try:
                if contents is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(contents)
                    if mode is not None:
                        path.chmod(mode)
            except OSError:
                success = False
        return success

    @staticmethod
    def _verify_active_application(
        evolver: CodebaseEvolver, files: list[str], test_paths: list[str]
    ) -> tuple[bool, dict[str, Any]]:
        imports_ok, import_message = evolver.run_import_checks(files)
        tests = evolver.run_tests(test_paths)
        security_ok = evolver.run_security_regression()
        verification = {
            "import_check": import_message,
            "full_test_suite": tests.passed,
            "security_regression": security_ok,
            "passed_tests": tests.passed_tests,
            "failed_tests": tests.failed_tests,
            "test_output": tests.raw_output[-4000:],
            "error": (
                import_message if not imports_ok else
                tests.error_message if not tests.passed else
                "Security regression suite failed" if not security_ok else ""
            ),
        }
        return imports_ok and tests.passed and security_ok, verification

    def reject(self, candidate_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
        with self._lock(candidate_id):
            record = self.get(candidate_id, tenant_id)
            if record is None:
                raise KeyError("Candidate not found")
            if record["status"] != "verified":
                raise ValueError(f"Candidate cannot be rejected from status {record['status']}")
            record["status"] = "rejected"
            record["error"] = reason.strip() or "Rejected by operator"
            self.save(record)
            return record

    def cleanup(self, candidate_id: str, tenant_id: str) -> None:
        with self._lock(candidate_id):
            record = self.get(candidate_id, tenant_id)
            if record is None:
                raise KeyError("Candidate not found")
            if record["status"] == "verified":
                raise ValueError("Reject a verified candidate before cleanup")
            if record["status"] in {
                "applied", "approved", "auto_promoted", "approval_cleanup_failed"
            }:
                raise ValueError("applied candidate records are retained for audit")
            self.delete(candidate_id, tenant_id)
