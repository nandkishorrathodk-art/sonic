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


DEPARTMENTS: tuple[Department, ...] = tuple(Department)

TEAMS: tuple[TeamDefinition, ...] = (
    TeamDefinition(Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, "Reasoning, planning, orchestration and agent runtime."),
    TeamDefinition(Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY, "LLM routing, model interfaces, memory and knowledge."),
    TeamDefinition(Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, "Being identity, persistent mind, curiosity and controlled evolution."),
    TeamDefinition(Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, "Desktop/workstation control and computer-use execution."),
    TeamDefinition(Department.COMPUTER_INTERACTION, Team.BROWSER_WEB, "Browser sessions and web interaction."),
    TeamDefinition(Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, "Visual grounding, world state and human-grade motor interaction."),
    TeamDefinition(Department.PLATFORM_SYSTEMS, Team.API_SERVICES, "FastAPI routes, service contracts and backend wiring."),
    TeamDefinition(Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, "Execution, kernel, queues and runtime tooling."),
    TeamDefinition(Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, "Database, integrations, configuration and logging."),
    TeamDefinition(Department.SECURITY_SAFETY, Team.TRUST_SAFETY, "Action policy, network controls, secrets and security invariants."),
    TeamDefinition(Department.SECURITY_SAFETY, Team.SANDBOX_ISOLATION, "Workspace isolation and sandbox execution containment."),
    TeamDefinition(Department.SECURITY_SAFETY, Team.IDENTITY_ACCESS, "Authentication, authorization and tenant access boundaries."),
    TeamDefinition(Department.EVALUATION_RELIABILITY, Team.VERIFICATION_BENCHMARKS, "Tests, benchmarks, replay and falsification."),
    TeamDefinition(Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY, "Health, telemetry, metrics and runtime reliability."),
    TeamDefinition(Department.EVALUATION_RELIABILITY, Team.EVIDENCE_RELEASE, "Evidence custody and release-gate evidence."),
    TeamDefinition(Department.EXPERIENCE_INTERFACE, Team.PRODUCT_UI, "Dashboard pages, components and operator-facing interfaces."),
    TeamDefinition(Department.EXPERIENCE_INTERFACE, Team.MISSION_OPERATOR_UX, "Mission, workstation and intervention UX contracts."),
    TeamDefinition(Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH, "Experimental agent architectures and research prototypes."),
    TeamDefinition(Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS, "Methods, experiments and reproducibility."),
)


SUBSYSTEM_OWNERSHIP: tuple[SubsystemOwnership, ...] = (
    SubsystemOwnership("brain", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, ("sonic-core/sonic/brain",), "Cognition and reasoning."),
    SubsystemOwnership("agents", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, ("sonic-core/sonic/agents",), "Agent implementations and orchestration specialists."),
    SubsystemOwnership("llm", Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY, ("sonic-core/sonic/llm",), "Model clients, routing and inference contracts."),
    SubsystemOwnership("memory", Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY, ("sonic-core/sonic/memory",), "Persistent graph, vector and cognitive memory."),
    SubsystemOwnership("being", Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, ("sonic-core/sonic/being",), "Persistent Being identity, mind, craft and vault."),
    SubsystemOwnership("evolution", Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, ("sonic-core/sonic/evolution",), "Curiosity, adaptation and controlled evolution."),
    SubsystemOwnership("swarm", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, ("sonic-core/sonic/swarm.py",), "Higher-order multi-agent coordination."),
    SubsystemOwnership("computer_use", Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, ("sonic-core/sonic/computer_use",), "Computer-use agents and execution contracts."),
    SubsystemOwnership("computer", Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, ("sonic-core/sonic/computer",), "Workstation providers and desktop control."),
    SubsystemOwnership("browser", Department.COMPUTER_INTERACTION, Team.BROWSER_WEB, ("sonic-core/sonic/browser",), "Browser sessions and web interaction."),
    SubsystemOwnership("perception", Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, ("sonic-core/sonic/perception",), "Visual grounding and perception."),
    SubsystemOwnership("world", Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, ("sonic-core/sonic/world",), "World/workstation state representation."),
    SubsystemOwnership("api", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("sonic-core/sonic/api",), "HTTP API and service composition."),
    SubsystemOwnership("execution", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("sonic-core/sonic/execution",), "Execution-plane contracts and process control."),
    SubsystemOwnership("kernel", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("sonic-core/sonic/kernel",), "Native/system kernel boundary."),
    SubsystemOwnership("queue", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("sonic-core/sonic/queue",), "Background work and queue consumers."),
    SubsystemOwnership("tools", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("sonic-core/sonic/tools",), "Tool registry and execution adapters."),
    SubsystemOwnership("integrations", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("sonic-core/sonic/integrations",), "External integration contracts."),
    SubsystemOwnership("db", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("sonic-core/sonic/db",), "Relational persistence and database sessions."),
    SubsystemOwnership("config", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("sonic-core/sonic/config.py",), "Runtime configuration."),
    SubsystemOwnership("logger", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("sonic-core/sonic/logger.py",), "Structured logging primitives."),
    SubsystemOwnership("auth", Department.SECURITY_SAFETY, Team.IDENTITY_ACCESS, ("sonic-core/sonic/auth",), "Authentication, authorization and tenant access."),
    SubsystemOwnership("safety", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, ("sonic-core/sonic/safety",), "Action policy, scope and immutable safety controls."),
    SubsystemOwnership("sandbox", Department.SECURITY_SAFETY, Team.SANDBOX_ISOLATION, ("sonic-core/sonic/sandbox",), "Workspace isolation and sandbox safety."),
    SubsystemOwnership("net", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, ("sonic-core/sonic/net",), "Network boundary and egress policy."),
    SubsystemOwnership("evidence", Department.EVALUATION_RELIABILITY, Team.EVIDENCE_RELEASE, ("sonic-core/sonic/evidence",), "Evidence custody and verifiable artifacts."),
    SubsystemOwnership("health", Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY, ("sonic-core/sonic/health.py",), "Health state and operational readiness."),
    SubsystemOwnership("observability", Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY, ("sonic-core/sonic/observability",), "Metrics and runtime diagnostics."),
    SubsystemOwnership("mission_engine", Department.EVALUATION_RELIABILITY, Team.VERIFICATION_BENCHMARKS, ("sonic-core/sonic/mission_engine",), "Mission evaluation and benchmarks."),
    SubsystemOwnership("research", Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH, ("sonic-core/sonic/research",), "Research workflows and experimental inquiry."),
    SubsystemOwnership("researcher", Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH, ("sonic-core/sonic/researcher",), "Research-oriented autonomous workers."),
    SubsystemOwnership("meta", Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS, ("sonic-core/sonic/meta",), "Experiments and controlled self-modification methods."),
    SubsystemOwnership("orchestration", Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS, ("sonic-core/sonic/orchestration",), "Cross-cutting orchestration research."),
    SubsystemOwnership("dashboard", Department.EXPERIENCE_INTERFACE, Team.PRODUCT_UI, ("sonic-dashboard",), "Operator dashboard and visual interfaces."),
    SubsystemOwnership("cli", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("sonic-cli",), "Command-line operator interface."),
    SubsystemOwnership("organization", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("sonic-core/sonic/organization",), "Tenant-scoped employee, project and work-item control plane."),
)


def _norm(path: str) -> str:
    value = str(path).replace("\\\\", "/").replace("\\", "/").lstrip("./")
    while "//" in value:
        value = value.replace("//", "/")
    return value


def ownership_for_path(path: str) -> SubsystemOwnership:
    normalized = _norm(path)
    matches: list[SubsystemOwnership] = []
    for item in SUBSYSTEM_OWNERSHIP:
        for prefix in item.path_prefixes:
            p = PurePosixPath(prefix)
            n = PurePosixPath(normalized)
            if normalized == prefix or p in n.parents:
                matches.append(item)
                break
    if not matches:
        raise KeyError(f"No SONIC department owns repository path: {path}")
    if len(matches) != 1:
        names = ", ".join(item.subsystem for item in matches)
        raise RuntimeError(f"Ambiguous SONIC ownership for {path}: {names}")
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
