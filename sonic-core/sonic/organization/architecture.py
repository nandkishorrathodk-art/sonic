"""Authoritative SONIC department, team and subsystem ownership map."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


class Department(StrEnum):
    COGNITIVE_SYSTEMS = "cognitive_systems"
    COMPUTER_INTERACTION = "computer_interaction"
    PLATFORM_SYSTEMS = "platform_systems"
    SECURITY_SAFETY = "security_safety"
    EVALUATION_RELIABILITY = "evaluation_reliability"
    EXPERIENCE_INTERFACE = "experience_interface"
    RESEARCH_EVOLUTION = "research_evolution"


class Team(StrEnum):
    REASONING_ORCHESTRATION = "reasoning_orchestration"
    MODEL_MEMORY = "model_memory"
    IDENTITY_EVOLUTION = "identity_evolution"
    DESKTOP_CONTROL = "desktop_control"
    BROWSER_WEB = "browser_web"
    PERCEPTION_MOTOR = "perception_motor"
    API_SERVICES = "api_services"
    SYSTEMS_RUNTIME = "systems_runtime"
    DATA_INFRASTRUCTURE = "data_infrastructure"
    TRUST_SAFETY = "trust_safety"
    SANDBOX_ISOLATION = "sandbox_isolation"
    IDENTITY_ACCESS = "identity_access"
    VERIFICATION_BENCHMARKS = "verification_benchmarks"
    RELIABILITY_OBSERVABILITY = "reliability_observability"
    EVIDENCE_RELEASE = "evidence_release"
    PRODUCT_UI = "product_ui"
    MISSION_OPERATOR_UX = "mission_operator_ux"
    AGENT_RESEARCH = "agent_research"
    METHODS_EXPERIMENTS = "methods_experiments"


@dataclass(frozen=True, slots=True)
class TeamDefinition:
    department: Department
    name: Team
    mission: str


@dataclass(frozen=True, slots=True)
class SubsystemOwnership:
    subsystem: str
    department: Department
    team: Team
    path_prefixes: tuple[str, ...]
    responsibility: str


DEPARTMENTS = tuple(Department)

TEAMS = tuple(
    TeamDefinition(department, team, team.value.replace("_", " ").title())
    for department, team in (
        (Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION),
        (Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY),
        (Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION),
        (Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL),
        (Department.COMPUTER_INTERACTION, Team.BROWSER_WEB),
        (Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR),
        (Department.PLATFORM_SYSTEMS, Team.API_SERVICES),
        (Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME),
        (Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE),
        (Department.SECURITY_SAFETY, Team.TRUST_SAFETY),
        (Department.SECURITY_SAFETY, Team.SANDBOX_ISOLATION),
        (Department.SECURITY_SAFETY, Team.IDENTITY_ACCESS),
        (Department.EVALUATION_RELIABILITY, Team.VERIFICATION_BENCHMARKS),
        (Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY),
        (Department.EVALUATION_RELIABILITY, Team.EVIDENCE_RELEASE),
        (Department.EXPERIENCE_INTERFACE, Team.PRODUCT_UI),
        (Department.EXPERIENCE_INTERFACE, Team.MISSION_OPERATOR_UX),
        (Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH),
        (Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS),
    )
)


def _owner(subsystem: str, department: Department, team: Team, path: str, responsibility: str):
    return SubsystemOwnership(subsystem, department, team, (path,), responsibility)


SUBSYSTEM_OWNERSHIP = (
    _owner("brain", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, "sonic-core/sonic/brain", "Cognition and reasoning."),
    _owner("agents", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, "sonic-core/sonic/agents", "Agent implementations and orchestration."),
    _owner("llm", Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY, "sonic-core/sonic/llm", "Model routing and inference."),
    _owner("memory", Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY, "sonic-core/sonic/memory", "Persistent memory and knowledge."),
    _owner("being", Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, "sonic-core/sonic/being", "Being identity and durable self."),
    _owner("evolution", Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, "sonic-core/sonic/evolution", "Controlled self-improvement."),
    _owner("swarm", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, "sonic-core/sonic/swarm.py", "Multi-agent coordination."),
    _owner("computer_use", Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, "sonic-core/sonic/computer_use", "Computer-use execution."),
    _owner("computer", Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, "sonic-core/sonic/computer", "Workstation providers."),
    _owner("browser", Department.COMPUTER_INTERACTION, Team.BROWSER_WEB, "sonic-core/sonic/browser", "Browser interaction."),
    _owner("perception", Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, "sonic-core/sonic/perception", "Visual perception."),
    _owner("world", Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, "sonic-core/sonic/world", "World state representation."),
    _owner("api", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, "sonic-core/sonic/api", "HTTP API and service composition."),
    _owner("execution", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, "sonic-core/sonic/execution", "Execution-plane control."),
    _owner("kernel", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, "sonic-core/sonic/kernel", "Native kernel boundary."),
    _owner("queue", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, "sonic-core/sonic/queue", "Background work queues."),
    _owner("tools", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, "sonic-core/sonic/tools", "Tool adapters."),
    _owner("integrations", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, "sonic-core/sonic/integrations", "External integrations."),
    _owner("db", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, "sonic-core/sonic/db", "Database persistence."),
    _owner("config", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, "sonic-core/sonic/config.py", "Runtime configuration."),
    _owner("logger", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, "sonic-core/sonic/logger.py", "Structured logging."),
    _owner("auth", Department.SECURITY_SAFETY, Team.IDENTITY_ACCESS, "sonic-core/sonic/auth", "Authentication and tenant access."),
    _owner("safety", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, "sonic-core/sonic/safety", "Fail-closed safety controls."),
    _owner("sandbox", Department.SECURITY_SAFETY, Team.SANDBOX_ISOLATION, "sonic-core/sonic/sandbox", "Sandbox containment."),
    _owner("net", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, "sonic-core/sonic/net", "Network boundary and egress."),
    _owner("evidence", Department.EVALUATION_RELIABILITY, Team.EVIDENCE_RELEASE, "sonic-core/sonic/evidence", "Evidence custody."),
    _owner("health", Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY, "sonic-core/sonic/health.py", "Operational health."),
    _owner("observability", Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY, "sonic-core/sonic/observability", "Runtime diagnostics."),
    _owner("mission_engine", Department.EVALUATION_RELIABILITY, Team.VERIFICATION_BENCHMARKS, "sonic-core/sonic/mission_engine", "Mission evaluation."),
    _owner("research", Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH, "sonic-core/sonic/research", "Research workflows."),
    _owner("researcher", Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH, "sonic-core/sonic/researcher", "Research workers."),
    _owner("meta", Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS, "sonic-core/sonic/meta", "Controlled experiments."),
    _owner("orchestration", Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS, "sonic-core/sonic/orchestration", "Cross-cutting orchestration research."),
    _owner("dashboard", Department.EXPERIENCE_INTERFACE, Team.PRODUCT_UI, "sonic-dashboard", "Operator dashboard."),
    _owner("cli", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, "sonic-cli", "CLI operator interface."),
)


def _norm(path: str) -> str:
    value = str(path).replace("\\", "/").lstrip("./")
    while "//" in value:
        value = value.replace("//", "/")
    return value


def ownership_for_path(path: str) -> SubsystemOwnership:
    normalized = _norm(path)
    matches = [
        item
        for item in SUBSYSTEM_OWNERSHIP
        if any(
            normalized == prefix
            or PurePosixPath(prefix) in PurePosixPath(normalized).parents
            for prefix in item.path_prefixes
        )
    ]
    if not matches:
        raise KeyError(f"No SONIC department owns repository path: {path}")
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous SONIC ownership for {path}")
    return matches[0]


def owner_for_path(path: str) -> tuple[Department, Team, str]:
    item = ownership_for_path(path)
    return item.department, item.team, item.subsystem


def validate_architecture() -> None:
    names = [item.subsystem for item in SUBSYSTEM_OWNERSHIP]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate subsystem ownership declaration detected")
    team_departments = {item.name: item.department for item in TEAMS}
    for item in SUBSYSTEM_OWNERSHIP:
        if team_departments.get(item.team) != item.department:
            raise ValueError(f"Team {item.team} has inconsistent department ownership")


validate_architecture()
