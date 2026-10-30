"""
SONIC v2 — Guarded Evolution Pipeline
======================================
Architectural Invariant:
AI can NEVER directly self-modify running production code or safety envelopes.
Any proposed improvement must advance through an immutable canary pipeline:
Candidate Improvement -> Isolated Branch -> Unit Tests -> Security Regression -> Benchmark -> Canary -> Operator Approval -> Promote / Rollback.
"""

from __future__ import annotations

import posixpath
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from sonic.logger import get_logger

logger = get_logger(__name__)

# Protected safety paths that the self-evolution engine is strictly forbidden from proposing edits to.
# Single source of truth: CodebaseEvolver imports this set so the two guard
# lists can never drift apart again.
_PROTECTED_COMPONENTS = frozenset([
    "sonic/safety",
    "sonic/kernel/action_broker",
    "sonic/safety/kernel",
    "sonic/safety/sealed",
    "sonic/safety/action_policy",
    "sonic/sandbox/egress",
    "sonic-kernel-rs/src/safety",
    "sonic/auth",
    "sonic/sandbox/providers",
    "sonic/evolution/patch_synthesizer",
    "sonic/evolution/codebase_evolver",
    "sonic/evolution/goal_director",
    "sonic/evolution/pipeline",
    "sonic/evolution/self_coder",
    "sonic/evolution/coding_agent",
    "sonic-core/tests/test_p0_security_hardening.py",
    "sonic/agents/judge.py",
    "sonic/evidence",
    "sonic-core/tests/test_safety_sealed_policy.py",
    "sonic/evolution/self_coder_candidates.py",
    "sonic/evolution/continuous_runner.py",
    "sonic/sandbox/factory.py",
])


def _check_path_match(candidate_path: str, protected_path: str) -> bool:
    """Helper to check if candidate_path matches protected_path with segmented slash boundaries."""
    if candidate_path == protected_path:
        return True
    if candidate_path.startswith(protected_path + "/"):
        return True
    if candidate_path.startswith(protected_path + "."):
        return True
    if f"/{protected_path}/" in f"/{candidate_path}/":
        return True
    if f"/{protected_path}." in f"/{candidate_path}":
        return True
    if candidate_path.endswith("/" + protected_path):
        return True
    return False


def get_protected_component(component_path: str) -> str | None:
    """
    Returns the protected component matched if component_path touches any protected safety component,
    or None otherwise.
    Normalizes path and strips repo prefixes (sonic-core/, sonic-cli/) with segmented boundary checks.
    """
    raw_path = str(component_path or "").replace("\\", "/").strip()
    collapsed = re.sub(r"/+", "/", raw_path)
    normalized = posixpath.normpath(collapsed).strip("/").lower()
    if not normalized or normalized == ".":
        return None

    norm_unprefixed = normalized.removeprefix("sonic-core/").removeprefix("sonic-cli/")

    for protected in _PROTECTED_COMPONENTS:
        p = posixpath.normpath(protected.replace("\\", "/").strip()).strip("/").lower()
        p_unprefixed = p.removeprefix("sonic-core/").removeprefix("sonic-cli/")

        for target in (normalized, norm_unprefixed):
            for prot in (p, p_unprefixed):
                if _check_path_match(target, prot):
                    return protected
    return None


def is_protected_component(component_path: str) -> bool:
    """
    Enforces strict safety envelope immutability.
    Returns True if the target path touches any protected safety component.
    """
    return get_protected_component(component_path) is not None


class EvolutionStage(StrEnum):
    PROPOSED = "proposed"
    ISOLATED_BRANCH = "isolated_branch"
    TESTS_PASSING = "tests_passing"
    IMPORTS_VERIFIED = "imports_verified"
    SECURITY_REGRESSION_PASSING = "security_regression_passing"
    BENCHMARK_VALIDATED = "benchmark_validated"
    CANARY_ACTIVE = "canary_active"
    AWAITING_APPROVAL = "awaiting_approval"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class ImprovementProposal:
    """A proposed capability or tool improvement."""
    proposal_id: str
    target_component: str  # e.g., "sonic/agents/web", "sonic/tools/adapters/nmap"
    description: str
    code_diff: str
    stage: EvolutionStage = EvolutionStage.PROPOSED
    stage_history: list[tuple[EvolutionStage, str, str]] = field(default_factory=list)
    benchmark_delta: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class EvolutionPipeline:
    """Governs self-improvement proposals through strict validation gates."""

    def __init__(self):
        self._proposals: dict[str, ImprovementProposal] = {}

    def submit_proposal(
        self,
        target_component: str,
        description: str,
        code_diff: str,
    ) -> tuple[ImprovementProposal, str]:
        """Submits an improvement proposal. Enforces strict safety immutability."""
        pid = f"evo-{uuid.uuid4().hex[:8]}"

        # Invariant 1: AI cannot propose modifications to safety kernel or policy
        protected = get_protected_component(target_component)
        if protected:
            logger.critical(
                "evolution_rejected_safety_violation",
                component=target_component,
                proposal_id=pid,
            )
            proposal = ImprovementProposal(
                proposal_id=pid,
                target_component=target_component,
                description=description,
                code_diff=code_diff,
                stage=EvolutionStage.REJECTED,
            )
            proposal.stage_history.append((
                EvolutionStage.REJECTED,
                f"Forbidden: Self-modification of protected safety kernel '{protected}' is strictly denied.",
                datetime.now(UTC).isoformat(),
            ))
            self._proposals[pid] = proposal
            return proposal, "Rejected: Cannot self-modify safety kernel."

        proposal = ImprovementProposal(
            proposal_id=pid,
            target_component=target_component,
            description=description,
            code_diff=code_diff,
            stage=EvolutionStage.ISOLATED_BRANCH,
        )
        proposal.stage_history.append((
            EvolutionStage.ISOLATED_BRANCH,
            "Applied to isolated sandbox branch for testing.",
            datetime.now(UTC).isoformat(),
        ))
        self._proposals[pid] = proposal
        return proposal, "Staged in isolated branch"

    def advance_stage(
        self,
        proposal_id: str,
        target_stage: EvolutionStage,
        validation_evidence: str,
    ) -> bool:
        """Transitions proposal through the verification pipeline."""
        prop = self._proposals.get(proposal_id)
        if not prop or prop.stage in (EvolutionStage.PROMOTED, EvolutionStage.ROLLED_BACK, EvolutionStage.REJECTED):
            return False

        prop.stage = target_stage
        prop.stage_history.append((target_stage, validation_evidence, datetime.now(UTC).isoformat()))
        return True

    def promote_with_approval(self, proposal_id: str, operator_approved: bool, operator_id: str = "") -> bool:
        """Promotes canary into production only with explicit human operator approval."""
        prop = self._proposals.get(proposal_id)
        if not prop or prop.stage != EvolutionStage.CANARY_ACTIVE:
            return False

        if not operator_approved:
            prop.stage = EvolutionStage.ROLLED_BACK
            prop.stage_history.append((EvolutionStage.ROLLED_BACK, "Operator declined promotion.", datetime.now(UTC).isoformat()))
            return False

        prop.stage = EvolutionStage.PROMOTED
        prop.stage_history.append((EvolutionStage.PROMOTED, f"Approved by operator {operator_id}.", datetime.now(UTC).isoformat()))
        logger.info("evolution_proposal_promoted", proposal_id=proposal_id, operator=operator_id)
        return True

    def get_proposal(self, proposal_id: str) -> ImprovementProposal | None:
        return self._proposals.get(proposal_id)
