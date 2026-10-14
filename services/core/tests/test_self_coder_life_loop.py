from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sonic.being.identity import get_or_create_being
from sonic.being.life_loop import BeingLifeLoop
from sonic.evolution import self_coder_candidates


def test_self_coder_respects_cadence_and_operator_activity(tmp_path, monkeypatch):
    calls = []

    class CandidateStoreStub:
        def __init__(self, repo_root):
            assert repo_root == tmp_path

        async def run_candidate(self, tenant_id, max_cycles, coder_factory):
            calls.append((tenant_id, max_cycles, coder_factory(tmp_path)))
            return {"candidates": [], "report": "No verified candidate"}

    monkeypatch.setattr(
        self_coder_candidates, "SelfCoderCandidateStore", CandidateStoreStub
    )
    being = get_or_create_being("self-coder-cadence@example.test")
    coder_copy = object()
    coder = SimpleNamespace(
        repo_root=tmp_path,
        isolated_copy=lambda root: coder_copy if root == tmp_path else None,
    )
    loop = BeingLifeLoop(
        being,
        SimpleNamespace(safety=object()),
        object(),
        "workspace",
        self_coder=coder,
        self_coder_every=2,
    )

    async def scenario():
        loop.cycles_completed = 1
        await loop._maybe_self_improve()
        assert calls == []

        loop.cycles_completed = 2
        loop.pause_for_operator()
        await loop._maybe_self_improve()
        assert calls == []

        loop.resume_after_operator()
        await loop._maybe_self_improve()

    asyncio.run(scenario())

    assert calls == [(being.tenant_id, 1, coder_copy)]
