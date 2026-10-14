"""
SONIC-REDA — Hypothesis Generator Agent
============================================
Creative vulnerability ideation based on discovered data.
Generates novel attack hypotheses that other agents may have missed.

This agent thinks like an elite security researcher — connecting
patterns across assets, technologies, and known vulnerability classes
to propose new attack vectors, including novel methods for gaps no
existing scanner covers.
"""

from __future__ import annotations

import json
from typing import Any

from sonic.agents.base import BaseAgent
from sonic.llm.prompts import HYPOTHESIS_SYSTEM
from sonic.logger import get_logger
from sonic.memory.schemas import HypothesisNode

logger = get_logger(__name__)


class HypothesisGenerator(BaseAgent):
    """
    Creative bug ideation agent — proposes novel vulnerability hypotheses.
    Connects dots between discovered assets, known vuln patterns, and
    application-specific logic to generate unique attack ideas.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="HypothesisGenerator", **kwargs)

    def get_system_prompt(self) -> str:
        return HYPOTHESIS_SYSTEM

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate vulnerability hypotheses."""
        self.status = "running"
        target = task.get("target", "")
        engagement_id = task.get("engagement_id", "")
        assets = task.get("assets", [])
        existing_findings = task.get("findings", [])
        technologies = task.get("technologies", [])

        logger.info("hypothesis_generation_starting", target=target)

        hypotheses = await self._generate_hypotheses(
            target, assets, existing_findings, technologies
        )

        # Store in Graph Memory
        stored = 0
        for h in hypotheses:
            try:
                hyp = HypothesisNode(
                    title=h.get("title", ""),
                    description=h.get("description", ""),
                    vulnerability_class=h.get("vulnerability_class", "unknown"),
                    rationale=h.get("rationale", ""),
                    test_plan=h.get("test_plan", ""),
                    priority=h.get("priority", 5),
                    proposed_by=self.agent_id,
                    engagement_id=engagement_id,
                )
                if self.memory:
                    await self.memory.create_hypothesis(hyp)
                    stored += 1
            except Exception as e:
                logger.warning("hypothesis_store_failed", error=str(e))

        self.status = "completed"
        return {
            "agent_id": self.agent_id,
            "hypotheses_generated": len(hypotheses),
            "hypotheses_stored": stored,
            "hypotheses": hypotheses,
        }

    async def _generate_hypotheses(
        self,
        target: str,
        assets: list[dict],
        findings: list[dict],
        technologies: list[str],
    ) -> list[dict]:
        """Generate creative vulnerability hypotheses (LLM + deterministic fallback)."""
        context_parts = [f"TARGET: {target}"]

        if assets:
            context_parts.append(f"DISCOVERED ASSETS:\n{json.dumps(assets[:15], indent=2)}")
        if findings:
            context_parts.append(f"EXISTING FINDINGS:\n{json.dumps(findings[:10], indent=2)}")
        if technologies:
            context_parts.append(f"TECHNOLOGIES: {', '.join(technologies)}")

        context = "\n\n".join(context_parts)

        prompt = f"""Based on this engagement data, generate creative vulnerability hypotheses:

{context}

Think like a top security researcher. Go beyond standard checks. Consider:
1. Attack chains: Can you combine multiple low issues into something critical?
2. Logic flaws: What business rules could be bypassed?
3. Race conditions: Any concurrent operations that could be exploited?
4. Technology-specific: Known issues with the detected tech stack?
5. Edge cases: What inputs or states did developers probably not consider?
6. Self-invention: Is this a gap NO existing tool covers? If so, flag the
   test_plan as "candidate for Toolsmith/MethodLab" so the being can author a
   new tool or synthesize a novel technique rather than force a known scanner.

Return JSON array:
[
    {{
        "title": "Descriptive title",
        "description": "Detailed explanation",
        "vulnerability_class": "IDOR/XSS/RCE/Logic/Race/...",
        "rationale": "Why I think this exists based on the evidence",
        "test_plan": "Specific steps to verify this hypothesis",
        "priority": 1-10,
        "impact_if_confirmed": "What an attacker could achieve"
    }}
]

Generate 5-8 high-quality hypotheses. Avoid generic/obvious ones."""

        if self.router is not None:
            try:
                response = await self.think(prompt, task_type="hypothesis")
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                hypotheses = json.loads(content)
                if isinstance(hypotheses, list) and hypotheses:
                    return hypotheses
            except Exception as e:
                logger.warning("hypothesis_llm_failed", error=str(e))

        # Deterministic fallback: synthesize hypotheses from asset richness so the
        # pipeline keeps producing testable ideas even with no LLM configured.
        return self._synth_hypotheses_from_assets(target, assets, technologies, findings)

    def _synth_hypotheses_from_assets(
        self,
        target: str,
        assets: list[dict],
        technologies: list[str],
        findings: list[dict],
    ) -> list[dict]:
        """
        Build candidate hypotheses from observed assets and parameters without hardcoded
        technology or URL keyword-matching trees (AGENTS.md Rule 2).
        """
        if not assets and not target:
            return []

        hypotheses: list[dict] = []
        for idx, a in enumerate(assets[:8]):
            value = a.get("value") or target
            if not value:
                continue
            if not value.startswith("http") and ("." in value or ":" in value):
                value = f"https://{value}"
            atype = str(a.get("type") or "endpoint")
            params = a.get("parameters") or (a.get("metadata", {}) if isinstance(a.get("metadata"), dict) else {}).get("parameters") or []
            priority = min(10, 6 + (2 if params else 0))
            hypotheses.append({
                "title": f"Verify behavior and access boundary on {value}",
                "description": f"Observed {atype} '{value}' with parameters {params}; verify state transitions and input handling empirically.",
                "vulnerability_class": "Logic",
                "rationale": f"Directly observed {atype} in environment surface.",
                "test_plan": f"Execute controlled request against {value} and verify response receipt.",
                "priority": priority,
                "impact_if_confirmed": "Unintended state transition or boundary violation.",
            })

        return hypotheses
