"""SONIC v2 — Self-Evolution & Method Lab Engine."""

from sonic.evolution.codebase_auditor import AuditFinding, AuditReport, CodebaseAuditor
from sonic.evolution.codebase_evolver import CodebaseEvolver, EvolutionSummaryReport
from sonic.evolution.codebase_report import narrate
from sonic.evolution.coding_agent import CodingAgent, CodingResult
from sonic.evolution.engine import EvolutionEngine
from sonic.evolution.evolution_journal import EvolutionJournal, JournalEntry
from sonic.evolution.goal_director import (
    EvolutionGoal,
    EvolutionGoalDirector,
    GoalCategory,
    GoalStatus,
)
from sonic.evolution.patch_synthesizer import PatchProposal, PatchSynthesizer
from sonic.evolution.pipeline import EvolutionPipeline, EvolutionStage, ImprovementProposal
from sonic.evolution.self_coder import SelfCoder, SelfCoderCycle, SelfCoderReport
from sonic.evolution.strategy import (
    DynamicStrategyEngine,
    StrategicPosture,
    StrategyAdaptationPlan,
    TargetFeedbackSignal,
)
from sonic.evolution.continuous_runner import ContinuousAutoEvolutionRunner
from sonic.evolution.version_tracker import VersionRecord, VersionTracker

__all__ = [
    "ContinuousAutoEvolutionRunner",
    "CodebaseEvolver",
    "CodebaseAuditor",
    "AuditFinding",
    "AuditReport",
    "narrate",
    "CodingAgent",
    "CodingResult",
    "EvolutionEngine",
    "EvolutionGoal",
    "EvolutionGoalDirector",
    "EvolutionJournal",
    "EvolutionPipeline",
    "EvolutionStage",
    "EvolutionSummaryReport",
    "GoalCategory",
    "GoalStatus",
    "ImprovementProposal",
    "JournalEntry",
    "PatchProposal",
    "PatchSynthesizer",
    "SelfCoder",
    "SelfCoderCycle",
    "SelfCoderReport",
    "DynamicStrategyEngine",
    "StrategicPosture",
    "StrategyAdaptationPlan",
    "TargetFeedbackSignal",
    "VersionRecord",
    "VersionTracker",
]
