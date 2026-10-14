from __future__ import annotations

import pytest

from sonic.queue.worker import SonicWorker


class _Queue:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    async def ensure_connected(self) -> bool:
        self.connected = True
        return True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_worker_run_forever_polls_until_stopped_and_closes_queue() -> None:
    worker = SonicWorker.__new__(SonicWorker)
    worker.queue = _Queue()
    worker._running = False
    worker._queue_initialized = False
    polls = 0

    async def run_once() -> None:
        nonlocal polls
        polls += 1
        if polls == 2:
            worker.stop()
        return None

    worker.run_once = run_once
    await worker.run_forever()

    assert polls == 2
    assert worker.queue.connected is True
    assert worker.queue.closed is True
    assert worker._running is False


@pytest.mark.asyncio
async def test_worker_loop_rejects_duplicate_start() -> None:
    worker = SonicWorker.__new__(SonicWorker)
    worker._running = True

    with pytest.raises(RuntimeError, match="already running"):
        await worker.run_forever()
