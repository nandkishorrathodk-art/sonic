"""
SONIC-REDA — Curiosity / Life Loop (Phase 6)
=============================================

A self-directed cycle the agent runs when it has no operator-issued goal.
Unlike a fixed idle script, the curious goal is PROPOSED by the LLM from the
live observation plus a memory of what has already been learned, biased toward
high novelty / information gain. Each cycle:

    observe
        |
    propose curious goal  (LLM, reacting to observation + known facts + novelty)
        |
    pursue goal  (the same observe -> reason -> act loop)
        |
    measure info_gain  (NoveltyEngine: how novel was the outcome vs known?)
        |
    persist newly-learned fact  (VectorMemory — curiosity compounds across cycles)
        |
    repeat  (avoiding dead-ends, re-biasing toward the unknown)

The goal is emergent from novelty, not scripted: as the agent learns facts,
those areas become less novel, so the LLM is steered toward genuinely unknown
territory. A dead-end (repeated zero novelty) triggers a pivot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from sonic.llm.prompts import curiosity_system_prompt
from sonic.logger import get_logger
from sonic.researcher.anomaly_engine import NoveltyEngine

logger = get_logger(__name__)


@dataclass
class CuriosityCycleResult:
    """Outcome of one self-directed curiosity cycle."""
    proposed_goal: str
    rationale: str
    info_gain: float
    learned_fact: str | None
    was_novel: bool
    pivoted: bool
    cycle: int


@dataclass
class CuriosityState:
    """Mutable state carried across curiosity cycles."""
    known_facts: list[str] = field(default_factory=list)
    cycle: int = 0
    consecutive_dead_ends: int = 0
    history: list[CuriosityCycleResult] = field(default_factory=list)
    last_affect_telemetry: dict[str, float] = field(default_factory=dict)
    affect_telemetry_history: list[dict[str, float]] = field(default_factory=list)


class CuriosityLoop:
    """Self-directed, novelty-driven exploration loop.

    Requires an LLM router (the goal is proposed, not scripted) and an optional
    VectorMemory so learned facts persist (Phase 1 Mind) and novelty compounds.
    """

    # A cycle is considered a dead-end if its outcome is (near) identical to
    # what is already known — i.e. negligible information gain.
    DEAD_END_NOVELTY = 0.05
    # After this many consecutive dead-ends, the loop pivots hard (asks the LLM
    # to propose something from an entirely different angle).
    PIVOT_AFTER = 2

    def __init__(
        self,
        llm_router: Any,
        vector_memory: Any | None = None,
        max_cycles: int = 5,
        exploration_steps: int = 3,
    ):
        self.llm_router = llm_router
        self.vector_memory = vector_memory
        self.max_cycles = max_cycles
        self.exploration_steps = exploration_steps
        self.state = CuriosityState()
        self.last_affect_telemetry: dict[str, float] = {}
        if self.vector_memory is not None:
            try:
                docs: list[Any] = []
                if hasattr(self.vector_memory, "list_documents") and callable(self.vector_memory.list_documents):
                    docs = list(self.vector_memory.list_documents() or [])
                elif hasattr(self.vector_memory, "documents"):
                    raw_docs = self.vector_memory.documents
                    docs = list(raw_docs.values()) if isinstance(raw_docs, dict) else list(raw_docs or [])
                for doc in docs:
                    meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {}) or {}
                    text = doc.get("text", "") if isinstance(doc, dict) else getattr(doc, "text", "") or ""
                    source = meta.get("source") if isinstance(meta, dict) else getattr(doc, "source", None)
                    if source == "curiosity_loop" and text and text not in self.state.known_facts:
                        self.state.known_facts.append(str(text))
            except Exception as e:
                logger.warning("curiosity_init_known_facts_failed", error=str(e))

    async def propose_curious_goal(self, observation_summary: str) -> tuple[str, str]:
        """Ask the LLM to propose the most informative self-directed goal.

        The LLM sees the current observation and what is already known (with
        novelty context), so it is steered toward the unknown — not a fixed
        idle task. Returns (goal, rationale).
        """
        from sonic.llm.schemas import LLMRequest, Message, MessageRole

        known = "\n".join(f"- {f}" for f in self.state.known_facts) or "(nothing learned yet)"
        pivot_note = ""
        if self.state.consecutive_dead_ends >= self.PIVOT_AFTER:
            pivot_note = (
                "You have repeatedly learned nothing new. PROPOSE a goal from a "
                "DIFFERENT, unexplored area than your recent proposals. "
            )
        system_prompt = curiosity_system_prompt(pivot_note)
        user_prompt = (
            f"Cycle {self.state.cycle + 1}.\n"
            f"Current observation:\n{observation_summary}\n\n"
            f"Already learned (avoid re-discovering these):\n{known}\n"
        )
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_prompt),
            ],
            task_type="reasoning",
        )
        response = await self.llm_router.complete(request)
        return self._parse_proposal(response.content)

    @staticmethod
    def _parse_proposal(text: str) -> tuple[str, str]:
        goal, rationale = "", ""
        first_line = ""
        for raw_line in text.strip().splitlines():
            line = raw_line.strip().lstrip("#*- ").replace("**", "").strip()
            if not line:
                continue
            if not first_line:
                first_line = line
            if line.upper().startswith("GOAL:"):
                goal = line.split(":", 1)[1].strip() or goal
            elif line.upper().startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()
        if not goal:
            goal = first_line or "inspect active workspace and desktop state"
        return goal, rationale

    def measure_novelty(self, observation_text: str = "", outcome: str | None = None) -> float:
        """Information gain of an outcome relative to known facts (0..1)."""
        target = outcome if outcome is not None else observation_text
        return NoveltyEngine.compute_novelty(target, self.state.known_facts)

    def evaluate_laya_curiosity(self, observation_text: str) -> dict[str, float]:
        """Evaluate System-1 curiosity affect and record telemetry."""
        from sonic.llm.laya_engine import get_laya_engine

        engine = get_laya_engine()
        telemetry = engine.evaluate_curiosity_affect(observation_text, self.state.known_facts)
        self.state.last_affect_telemetry = telemetry
        self.state.affect_telemetry_history.append(telemetry)
        self.last_affect_telemetry = telemetry
        logger.info(
            "curiosity_laya_affect",
            observation=str(observation_text)[:120],
            novelty_score=telemetry.get("novelty_score", 0.0),
            worth_learning_prob=telemetry.get("worth_learning_prob", 0.0),
            telemetry=telemetry,
        )
        return telemetry

    def persist_fact(self, fact: str) -> bool:
        """Record a newly-learned fact to persistent vector memory (Phase 1).

        Returns True if it was genuinely new (not a near-duplicate of an
        existing learned fact). Deduplication keeps curiosity directed at the
        genuinely unknown.
        """
        if self.vector_memory is not None:
            try:
                dup, _ = self.vector_memory.is_duplicate(fact, threshold=0.88)
                if dup:
                    return False
                digest = hashlib.sha256(fact.encode()).hexdigest()[:8]
                doc_id = f"curiosity-{self.state.cycle}-{digest}"
                self.vector_memory.index_document(
                    doc_id, fact, metadata={"source": "curiosity_loop", "cycle": self.state.cycle}
                )
            except Exception as e:
                logger.warning("curiosity_persist_failed", error=str(e))
        # Track in-process too (used for novelty bias even without vector memory).
        if fact not in self.state.known_facts:
            self.state.known_facts.append(fact)
            return True
        return False

    async def run_cycle(self, observation_summary: str, pursue: Any) -> CuriosityCycleResult:
        """Run one curiosity cycle.

        ``pursue`` is an async callable ``(goal: str, steps: int) -> str`` that
        pursues the proposed goal via the agent's real observe->reason->act loop
        and returns the outcome summary (what was observed/learned).
        """
        self.state.cycle += 1
        goal, rationale = await self.propose_curious_goal(observation_summary)
        outcome = await pursue(goal, self.exploration_steps)
        info_gain = self.measure_novelty(outcome)
        try:
            self.evaluate_laya_curiosity(str(outcome or ""))
        except Exception as e:
            logger.debug("curiosity_laya_affect_cycle_failed", error=str(e))
        was_novel = info_gain >= self.DEAD_END_NOVELTY
        outcome_clean = str(outcome or "").strip()
        if outcome_clean.lower().startswith("safety blocked:"):
            was_novel = False

        learned: str | None = None
        if was_novel:
            learned = f"{goal}: {outcome}"
            newly = self.persist_fact(learned)
            if not newly:
                # Outcome duplicated an already-learned fact -> not genuinely novel.
                was_novel = False

        pivoted = False
        if was_novel:
            self.state.consecutive_dead_ends = 0
        else:
            self.state.consecutive_dead_ends += 1
            pivoted = self.state.consecutive_dead_ends >= self.PIVOT_AFTER

        result = CuriosityCycleResult(
            proposed_goal=goal, rationale=rationale, info_gain=round(info_gain, 4),
            learned_fact=learned if was_novel else None,
            was_novel=was_novel, pivoted=pivoted, cycle=self.state.cycle,
        )
        self.state.history.append(result)
        logger.info("curiosity_cycle", cycle=result.cycle, goal=goal,
                    info_gain=result.info_gain, novel=was_novel, pivoted=pivoted)
        return result

    async def run(self, observation_summary_fn: Any, pursue: Any) -> list[CuriosityCycleResult]:
        """Run up to max_cycles self-directed curiosity cycles.

        ``observation_summary_fn`` is an async callable ``() -> str`` returning
        the current world observation summary; ``pursue`` as in run_cycle.
        """
        for _ in range(self.max_cycles):
            obs_summary = await observation_summary_fn()
            await self.run_cycle(obs_summary, pursue)
        return list(self.state.history)
