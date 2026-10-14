from __future__ import annotations

import asyncio

from sonic.evolution.self_coder_loop import (
    SelfCoderBackgroundLoop,
    get_self_coder_loop_status,
)


def test_background_loop_runs_and_persists_receipt_status():
    completed = asyncio.Event()

    class CandidateStoreStub:
        async def run_candidate(self, tenant_id, max_cycles, coder_factory):
            assert tenant_id == "default"
            assert max_cycles == 1
            completed.set()
            return {
                "candidates": [
                    {"candidate_id": "candidate-1", "status": "auto_promoted"}
                ],
                "report": "Test receipt",
            }

    async def scenario():
        loop = SelfCoderBackgroundLoop(
            CandidateStoreStub(),
            coder_factory=lambda _root: None,
            interval_seconds=3600,
        )
        loop.start()
        await asyncio.wait_for(completed.wait(), timeout=2)
        await loop.stop()
        status = get_self_coder_loop_status()
        assert status["running"] is False
        assert status["last_run_at"]
        assert status["last_error"] == ""
        assert status["last_candidate_statuses"] == [
            {"candidate_id": "candidate-1", "status": "auto_promoted"}
        ]

    asyncio.run(scenario())
