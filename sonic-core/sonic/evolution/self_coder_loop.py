"""Periodic background runner for isolated, verified in-place self-coder updates."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sonic.logger import get_logger

logger = get_logger(__name__)
_active_loop: SelfCoderBackgroundLoop | None = None


def get_self_coder_loop_status() -> dict[str, Any]:
    loop = _active_loop
    if loop is None:
        return {"running": False, "last_run_at": "", "last_error": ""}
    return {
        "running": loop._task is not None and not loop._task.done(),
        "interval_seconds": loop.interval_seconds,
        "last_run_at": loop.last_run_at,
        "last_error": loop.last_error,
        "last_candidate_statuses": loop.last_candidate_statuses,
    }


class SelfCoderBackgroundLoop:
    """Run one real self-coder cycle periodically without driving the desktop."""

    def __init__(
        self,
        candidate_store: Any,
        coder_factory: Callable[[Any], Any],
        *,
        tenant_id: str = "default",
        interval_seconds: int = 3600,
    ) -> None:
        if not 60 <= interval_seconds <= 86400:
            raise ValueError("interval_seconds must be between 60 and 86400")
        self.candidate_store = candidate_store
        self.coder_factory = coder_factory
        self.tenant_id = tenant_id
        self.interval_seconds = interval_seconds
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.last_run_at = ""
        self.last_error = ""
        self.last_candidate_statuses: list[dict[str, str]] = []

    def start(self) -> asyncio.Task:
        global _active_loop
        if self._task is not None and not self._task.done():
            return self._task
        self._stop.clear()
        _active_loop = self
        self._task = asyncio.create_task(self._run(), name="sonic-self-coder-loop")
        return self._task

    async def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        task = self._task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def run_once(self) -> dict[str, Any]:
        """Run one candidate cycle and retain its exact persisted receipts."""
        def execute():
            async def run():
                return await self.candidate_store.run_candidate(
                    self.tenant_id,
                    1,
                    self.coder_factory,
                )
            return asyncio.run(run())

        try:
            result = await asyncio.to_thread(execute)
        except Exception as exc:
            self.last_error = str(exc)
            self.last_run_at = datetime.now(UTC).isoformat()
            logger.error("self_coder_background_cycle_failed", error=self.last_error)
            raise
        self.last_error = ""
        self.last_run_at = datetime.now(UTC).isoformat()
        self.last_candidate_statuses = [
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "status": str(item.get("status", "")),
            }
            for item in result.get("candidates", [])
        ]
        logger.info(
            "self_coder_background_cycle_complete",
            tenant_id=self.tenant_id,
            candidates=len(self.last_candidate_statuses),
            applied=sum(
                item["status"] == "applied"
                for item in self.last_candidate_statuses
            ),
        )
        return result

    async def _run(self) -> None:
        logger.info(
            "self_coder_background_loop_started",
            tenant_id=self.tenant_id,
            interval_seconds=self.interval_seconds,
        )
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                logger.error(
                    "self_coder_background_loop_will_retry",
                    tenant_id=self.tenant_id,
                    error=str(exc),
                )
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.interval_seconds
                )
            except TimeoutError:
                continue
        logger.info("self_coder_background_loop_stopped", tenant_id=self.tenant_id)
