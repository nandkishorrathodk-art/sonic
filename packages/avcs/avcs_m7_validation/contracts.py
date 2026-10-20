from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

class Mode(str, Enum):
    VISUAL_ONLY = "visual_only"
    UIA_ONLY = "uia_only"
    HYBRID = "hybrid"

class ActionStatus(str, Enum):
    CONFIRMED = "confirmed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ERROR = "error"

class OutcomeCategory(str, Enum):
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    EVALUATION_INVALID = "evaluation_invalid"
    FALSE_CONFIRMATION = "false_confirmation"
    ABSTENTION = "abstention"
    BLOCKED = "blocked"
    EXECUTION_ERROR = "execution_error"

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    category: str
    application: str
    task_prompt: str
    target_property: str
    risk: str = "low"
    modes: tuple[str, ...] = ("visual_only", "uia_only", "hybrid")
    setup_notes: str = ""
    held_out: bool = True

@dataclass(frozen=True)
class AgentTask:
    """Only agent-visible task content; intentionally excludes scenario IDs and oracle metadata."""
    task_prompt: str

@dataclass(frozen=True)
class ExecutionResult:
    status: ActionStatus
    elapsed_ms: float
    action_count: int
    evidence_method: str
    detail_code: str = ""
    # Human-readable content, screenshots, and raw UI text should not be placed here.
    metadata: dict[str, Any] = field(default_factory=dict)

class EvidenceProvenance(str, Enum):
    SIMULATION_FIXTURE = "simulation_fixture"
    LIVE_APPLICATION = "live_application"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OracleResult:
    success: bool
    oracle_id: str
    assertion_id: str
    reason_code: str = ""
    valid: bool = True

@dataclass(frozen=True)
class TrialResult:
    run_id: str
    scenario_id: str
    mode: Mode
    seed: int
    status: ActionStatus
    oracle_success: bool
    oracle_valid: bool
    reported_confirmed: bool
    verified_success: bool
    false_confirmation: bool
    elapsed_ms: float
    action_count: int
    evidence_method: str
    category: OutcomeCategory
    oracle_id: str
    assertion_id: str
    detail_code: str = ""
    provenance: EvidenceProvenance = EvidenceProvenance.SIMULATION_FIXTURE
    app_identity: Optional[dict[str, Any]] = None
    action_latency_ms: Optional[float] = None
    postcondition_evidence_ref: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["status"] = self.status.value
        d["category"] = self.category.value
        d["provenance"] = self.provenance.value if isinstance(self.provenance, EvidenceProvenance) else str(self.provenance)
        return d

@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    observed: Any
    required: str
    note: str = ""


def is_reported_confirmation(status: ActionStatus) -> bool:
    return status == ActionStatus.CONFIRMED
