"""Tests for evidence-grounded hypothesis behavior when no LLM is configured."""

from __future__ import annotations

import asyncio
import pytest

from sonic.agents.hypothesis import HypothesisGenerator
from sonic.memory.inmemory import InMemoryGraph
from sonic.safety.scope import ScopeChecker


def _make_generator(router=None):
    memory = InMemoryGraph()
    memory._connected = True
    scope = ScopeChecker()
    scope._loaded = True
    return HypothesisGenerator(model_router=router, graph_memory=memory,
                               scope_checker=scope)


def test_hypotheses_synth_without_llm():
    """No router → deterministic hypotheses are still produced."""
    gen = _make_generator(router=None)
    assets = [
        {"type": "endpoint", "value": "https://app.test/login"},
        {"type": "endpoint", "value": "https://app.test/api/users"},
        {"type": "url", "value": "https://app.test/static/style.css"},
    ]
    hyps = gen._synth_hypotheses_from_assets("app.test", assets, [], [])
    assert len(hyps) >= 1
    # All must have required schema fields
    for h in hyps:
        assert "title" in h and "vulnerability_class" in h
        assert isinstance(h["priority"], int) and 1 <= h["priority"] <= 10


def test_fallback_preserves_observed_asset_order_without_keyword_scoring():
    """The no-LLM fallback must not infer risk classes from URL words."""
    gen = _make_generator(router=None)
    assets = [
        {"type": "url", "value": "https://app.test/static/img.png"},
        {"type": "endpoint", "value": "https://app.test/auth/login?next=/admin"},
        {"type": "endpoint", "value": "https://app.test/api/admin/users"},
    ]
    hyps = gen._synth_hypotheses_from_assets("app.test", assets, [], [])
    assert [h["title"] for h in hyps] == [
        f"Verify behavior and access boundary on {asset['value']}" for asset in assets
    ]
    assert {h["vulnerability_class"] for h in hyps} == {"Logic"}


def test_technology_names_do_not_create_unobserved_hypotheses():
    """Technology labels alone do not generate canned exploit hypotheses."""
    gen = _make_generator(router=None)
    hyps = gen._synth_hypotheses_from_assets("app.test", [], ["WordPress", "PHP"], [])
    assert hyps == []


def test_findings_do_not_trigger_a_canned_chaining_hypothesis():
    """The fallback does not fabricate a chain solely from vulnerability labels."""
    gen = _make_generator(router=None)
    with_findings = gen._synth_hypotheses_from_assets(
        "app.test", [{"type": "url", "value": "https://app.test/x"}], [],
        [{"vulnerability_class": "IDOR"}, {"vulnerability_class": "XSS"}],
    )
    without_findings = gen._synth_hypotheses_from_assets(
        "app.test", [{"type": "url", "value": "https://app.test/x"}], [], []
    )
    assert with_findings == without_findings
    assert all(h["vulnerability_class"] != "Chaining" for h in with_findings)


def test_hypothesis_run_returns_without_llm():
    """The full run() path works with no LLM and stores hypotheses."""
    async def _run():
        gen = _make_generator(router=None)
        result = await gen.run({
            "target": "app.test", "engagement_id": "eng-h",
            "assets": [{"type": "endpoint", "value": "https://app.test/login"}],
            "technologies": ["PHP"], "findings": [],
        })
        assert result["hypotheses_generated"] >= 1
        assert result["hypotheses_stored"] >= 1
        assert isinstance(result["hypotheses"], list)
    asyncio.run(_run())


def test_hypothesis_empty_when_no_surface():
    gen = _make_generator(router=None)
    assert gen._synth_hypotheses_from_assets("", [], [], []) == []
