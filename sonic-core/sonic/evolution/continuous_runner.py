"""Continuous Autonomous Self-Evolution Engine Runner.

Belongs to Department: cognitive_systems, Team: identity_evolution, Subsystem: evolution.
Runs continuous cycles of signal discovery, isolated worktree verification,
and atomic commit/logging to evolution.md without requiring human hand-holding.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sonic.evolution.self_coder import SelfCoder
from sonic.evolution.self_coder_candidates import SelfCoderCandidateStore
from sonic.evolution.self_coder_config import get_self_coder_config
from sonic.logger import get_logger

logger = get_logger(__name__)


class ContinuousAutoEvolutionRunner:
    """Orchestrates the continuous non-blocking Self-Evolution loop."""

    def __init__(
        self,
        repo_root: Path | str,
        *,
        tenant_id: str = "default",
        max_cycles: int = 10,
        interval_seconds: int = 5,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        candidate_store: SelfCoderCandidateStore | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.tenant_id = tenant_id
        self.max_cycles = max_cycles
        self.interval_seconds = interval_seconds
        self.on_event = on_event or (lambda ev: None)
        self.candidate_store = candidate_store or SelfCoderCandidateStore(repo_root=self.repo_root)
        self._stop = asyncio.Event()
        self._running = False
        self._current_cycle = 0

    def stop(self) -> None:
        """Signal the continuous loop to pause or halt."""
        self._stop.set()
        self._running = False
        self._emit("paused", "Continuous Auto-Evolution paused by operator.")

    @property
    def is_running(self) -> bool:
        return self._running

    def _emit(self, event_type: str, message: str, **kwargs: Any) -> None:
        payload = {
            "type": event_type,
            "message": message,
            "cycle": self._current_cycle,
            "timestamp": datetime.now(UTC).isoformat(),
            **kwargs,
        }
        try:
            self.on_event(payload)
        except Exception as exc:
            logger.debug("continuous_runner_emit_error", error=str(exc))

    async def run(self) -> None:
        """Run the autonomous evolution loop until stopped or max_cycles reached."""
        self._stop.clear()
        self._running = True
        self._current_cycle = 0
        cfg = get_self_coder_config()

        self._emit(
            "started",
            f"⚡ Auto-Evolution continuous loop engaged for repository: {self.repo_root.name}",
            repo_root=str(self.repo_root),
        )

        while not self._stop.is_set():
            self._current_cycle += 1
            if self.max_cycles and self._current_cycle > self.max_cycles:
                self._emit("completed", f"Reached maximum configured cycles ({self.max_cycles}).")
                break

            self._emit("cycle_start", f"[Cycle {self._current_cycle}] 🔍 Mining codebase signals...")

            # 1. Inspect signals on current checkout
            try:
                coder = SelfCoder(
                    repo_root=self.repo_root,
                    allow_writes=False,
                    agentic=cfg.agentic,
                )
                signals = coder.discover_signals(limit=5)
            except Exception as exc:
                self._emit("cycle_error", f"Signal discovery failed: {exc}", error=str(exc))
                signals = []

            if not signals:
                self._emit(
                    "cycle_idle",
                    f"[Cycle {self._current_cycle}] No actionable defects found. Codebase is clean.",
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                except TimeoutError:
                    pass
                continue

            target_sig = signals[0]
            desc, target_file, category = target_sig
            self._emit(
                "signal_found",
                f"[Cycle {self._current_cycle}] 🛠 Identified issue in {target_file}: {desc}",
                target_file=target_file,
                category=category,
            )

            # 2. Execute candidate in isolated worktree
            def coder_factory(worktree_dir: Path) -> SelfCoder:
                return SelfCoder(
                    repo_root=worktree_dir,
                    allow_writes=False,
                    agentic=cfg.agentic,
                )

            self._emit(
                "testing_worktree",
                f"[Cycle {self._current_cycle}] 🧪 Testing in isolated detached Git worktree...",
            )

            try:
                candidate_result = await self.candidate_store.run_candidate(
                    self.tenant_id,
                    1,
                    coder_factory,
                )
            except Exception as exc:
                self._emit("cycle_error", f"Candidate execution error: {exc}", error=str(exc))
                await asyncio.sleep(2)
                continue

            candidates = candidate_result.get("candidates", [])
            if not candidates:
                self._emit(
                    "cycle_no_candidate",
                    f"[Cycle {self._current_cycle}] Candidate synthesis yielded no passing patch.",
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                except TimeoutError:
                    pass
                continue

            candidate = candidates[0]
            cand_id = candidate.get("candidate_id", "")
            cand_status = candidate.get("status", "")

            if cand_status != "verified":
                self._emit(
                    "candidate_unverified",
                    f"[Cycle {self._current_cycle}] Patch failed verification in worktree ({cand_status}). Discarded.",
                    candidate_id=cand_id,
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                except TimeoutError:
                    pass
                continue

            # 3. Promote, commit, and advance HEAD
            self._emit(
                "promoting",
                f"[Cycle {self._current_cycle}] ✓ Candidate {cand_id[:8]} verified! Committing to git & logging...",
                candidate_id=cand_id,
            )

            try:
                promoted = self.candidate_store.auto_promote(
                    cand_id,
                    self.tenant_id,
                    auto_commit=True,
                )
                commit_hash = promoted.get("commit", "")
                self._emit(
                    "cycle_success",
                    f"[Cycle {self._current_cycle}] 🟢 Successfully evolved! Commit: {commit_hash[:8]} | Logged to evolution.md",
                    candidate_id=cand_id,
                    commit=commit_hash,
                )
            except Exception as exc:
                self._emit(
                    "promote_error",
                    f"[Cycle {self._current_cycle}] Promotion/Commit failed: {exc}",
                    error=str(exc),
                )

            if self._stop.is_set():
                break

            # Short breath before next cycle
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except TimeoutError:
                pass

        self._running = False
        self._emit("stopped", "Continuous Auto-Evolution halted.")
