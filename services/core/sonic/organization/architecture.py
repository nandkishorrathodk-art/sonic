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
    SubsystemOwnership("brain", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, ("services/core/sonic/brain",), "Cognition and reasoning."),
    SubsystemOwnership("core_package", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("services/core/sonic/__init__.py",), "Core package metadata and public package boundary."),
    SubsystemOwnership("agents", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, ("services/core/sonic/agents",), "Agent implementations and orchestration specialists."),
    SubsystemOwnership("llm", Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY, ("services/core/sonic/llm",), "Model clients, routing and inference contracts."),
    SubsystemOwnership("memory", Department.COGNITIVE_SYSTEMS, Team.MODEL_MEMORY, ("services/core/sonic/memory",), "Persistent graph, vector and cognitive memory."),
    SubsystemOwnership("being", Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, ("services/core/sonic/being",), "Persistent Being identity, mind, craft and vault."),
    SubsystemOwnership("evolution", Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, ("services/core/sonic/evolution",), "Curiosity, adaptation and controlled evolution."),
    SubsystemOwnership("swarm", Department.COGNITIVE_SYSTEMS, Team.REASONING_ORCHESTRATION, ("services/core/sonic/swarm.py",), "Higher-order multi-agent coordination."),
    SubsystemOwnership("computer_use", Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, ("services/core/sonic/computer_use",), "Computer-use agents and execution contracts."),
    SubsystemOwnership("avcs", Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, ("packages/avcs",), "AVCS visual perception, motor control, SDK and evaluation package."),
    SubsystemOwnership("computer", Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, ("services/core/sonic/computer",), "Workstation providers and desktop control."),
    SubsystemOwnership("browser", Department.COMPUTER_INTERACTION, Team.BROWSER_WEB, ("services/core/sonic/browser",), "Browser sessions and web interaction."),
    SubsystemOwnership("perception", Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, ("services/core/sonic/perception",), "Visual grounding and perception."),
    SubsystemOwnership("world", Department.COMPUTER_INTERACTION, Team.PERCEPTION_MOTOR, ("services/core/sonic/world",), "World/workstation state representation."),
    SubsystemOwnership("api", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("services/core/sonic/api",), "HTTP API and service composition."),
    SubsystemOwnership("execution", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("services/core/sonic/execution",), "Execution-plane contracts and process control."),
    SubsystemOwnership("kernel", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("services/core/sonic/kernel", "packages/kernel/src"), "Native/system kernel boundary."),
    SubsystemOwnership("kernel_safety", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, ("packages/kernel/src/safety",), "Native kernel safety policy and enforcement."),
    SubsystemOwnership("queue", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("services/core/sonic/queue",), "Background work and queue consumers."),
    SubsystemOwnership("tools", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("services/core/sonic/tools",), "Tool registry and execution adapters."),
    SubsystemOwnership("integrations", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("services/core/sonic/integrations",), "External integration contracts."),
    SubsystemOwnership("db", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("services/core/sonic/db",), "Relational persistence and database sessions."),
    SubsystemOwnership("config", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("services/core/sonic/config.py",), "Runtime configuration."),
    SubsystemOwnership("logger", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("services/core/sonic/logger.py",), "Structured logging primitives."),
    SubsystemOwnership("auth", Department.SECURITY_SAFETY, Team.IDENTITY_ACCESS, ("services/core/sonic/auth",), "Authentication, authorization and tenant access."),
    SubsystemOwnership("safety", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, ("services/core/sonic/safety",), "Action policy, scope and immutable safety controls."),
    SubsystemOwnership("sandbox", Department.SECURITY_SAFETY, Team.SANDBOX_ISOLATION, ("services/core/sonic/sandbox",), "Workspace isolation and sandbox safety."),
    SubsystemOwnership("net", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, ("services/core/sonic/net",), "Network boundary and egress policy."),
    SubsystemOwnership("evidence", Department.EVALUATION_RELIABILITY, Team.EVIDENCE_RELEASE, ("services/core/sonic/evidence",), "Evidence custody and verifiable artifacts."),
    SubsystemOwnership("health", Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY, ("services/core/sonic/health.py",), "Health state and operational readiness."),
    SubsystemOwnership("observability", Department.EVALUATION_RELIABILITY, Team.RELIABILITY_OBSERVABILITY, ("services/core/sonic/observability",), "Metrics and runtime diagnostics."),
    SubsystemOwnership("mission_engine", Department.EVALUATION_RELIABILITY, Team.VERIFICATION_BENCHMARKS, ("services/core/sonic/mission_engine",), "Mission evaluation and benchmarks."),
    SubsystemOwnership("research", Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH, ("services/core/sonic/research",), "Research workflows and experimental inquiry."),
    SubsystemOwnership("researcher", Department.RESEARCH_EVOLUTION, Team.AGENT_RESEARCH, ("services/core/sonic/researcher",), "Research-oriented autonomous workers."),
    SubsystemOwnership("meta", Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS, ("services/core/sonic/meta",), "Experiments and controlled self-modification methods."),
    SubsystemOwnership("orchestration", Department.RESEARCH_EVOLUTION, Team.METHODS_EXPERIMENTS, ("services/core/sonic/orchestration",), "Cross-cutting orchestration research."),
    SubsystemOwnership("dashboard", Department.EXPERIENCE_INTERFACE, Team.PRODUCT_UI, ("apps/dashboard",), "Operator dashboard and visual interfaces."),
    SubsystemOwnership("cli", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("apps/cli",), "Command-line operator interface."),
    SubsystemOwnership("apps_index", Department.EXPERIENCE_INTERFACE, Team.PRODUCT_UI, ("apps/README.md",), "Index for product applications."),
    SubsystemOwnership("services_index", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("services/README.md",), "Index for backend services."),
    SubsystemOwnership("packages_index", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("packages/README.md",), "Index for reusable runtime packages."),
    SubsystemOwnership("applications", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("apps",), "Application package manifests and shared application-level files."),
    SubsystemOwnership("services", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("services",), "Service manifests, container builds and shared service-level files."),
    SubsystemOwnership("packages", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("packages",), "Reusable package manifests and shared package-level files."),
    SubsystemOwnership("kernel_package", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("packages/kernel",), "Rust kernel package manifest, source and build metadata."),
    SubsystemOwnership("organization", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("services/core/sonic/organization",), "Tenant-scoped employee, project and work-item control plane."),
    SubsystemOwnership("core_tests", Department.EVALUATION_RELIABILITY, Team.VERIFICATION_BENCHMARKS, ("services/core/tests",), "Backend contract, regression and acceptance tests."),
    SubsystemOwnership("cli_tests", Department.EVALUATION_RELIABILITY, Team.VERIFICATION_BENCHMARKS, ("apps/cli/tests",), "CLI contract and regression tests."),
    SubsystemOwnership("workstation_image", Department.COMPUTER_INTERACTION, Team.DESKTOP_CONTROL, ("docker",), "Workstation image and desktop runtime packaging."),
    SubsystemOwnership("compose", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("docker-compose.yml", "docker-compose.prod.yml"), "Root development and production service composition."),
    SubsystemOwnership("deployment", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, ("infra", "scripts", ".github/workflows"), "Deployment and CI runtime configuration."),
    SubsystemOwnership("runtime_configs", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, ("configs",), "Shared runtime and model configuration."),
    SubsystemOwnership("engineering_docs", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("docs",), "Architecture, setup, operations and engineering documentation."),
    SubsystemOwnership("computer_user_contract", Department.EXPERIENCE_INTERFACE, Team.MISSION_OPERATOR_UX, ("docs/SONIC_COMPUTER_USER_GOAL.md",), "Operator-facing computer-use product contract and acceptance expectations."),
    SubsystemOwnership("interface_assets", Department.EXPERIENCE_INTERFACE, Team.PRODUCT_UI, ("assets",), "Repository-level product diagrams and interface reference assets."),
    SubsystemOwnership("audit_reports", Department.EVALUATION_RELIABILITY, Team.EVIDENCE_RELEASE, ("reports",), "Published audit and assessment evidence."),
    SubsystemOwnership("evolution_journal", Department.COGNITIVE_SYSTEMS, Team.IDENTITY_EVOLUTION, ("evolution.md",), "Evolution and self-improvement journal."),
    SubsystemOwnership("repository_ci_metadata", Department.PLATFORM_SYSTEMS, Team.SYSTEMS_RUNTIME, (".github",), "Repository automation and pull-request workflow metadata."),
    SubsystemOwnership("repository_security_policy", Department.SECURITY_SAFETY, Team.TRUST_SAFETY, ("SECURITY.md",), "Repository security reporting and disclosure policy."),
    SubsystemOwnership("repository_runtime_config", Department.PLATFORM_SYSTEMS, Team.DATA_INFRASTRUCTURE, (".env.example", "pyproject.toml", "uv.lock", ".gitignore", ".gitattributes"), "Root workspace metadata and environment/configuration conventions."),
    SubsystemOwnership("repository_governance", Department.PLATFORM_SYSTEMS, Team.API_SERVICES, ("AGENTS.md", "CODEOWNERS", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "README.md", "TEST_INFRA.md", "TEST_READY.md", "run_backend.ps1"), "Repository governance, contributor entry points and local developer startup."),
)


def _norm(path: str) -> str:
    value = str(path).replace("\\\\", "/").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    while "//" in value:
        value = value.replace("//", "/")
    return value


def ownership_for_path(path: str) -> SubsystemOwnership:
    normalized = _norm(path)
    matches: list[tuple[int, SubsystemOwnership]] = []
    for item in SUBSYSTEM_OWNERSHIP:
        for prefix in item.path_prefixes:
            p = PurePosixPath(prefix)
            n = PurePosixPath(normalized)
            if normalized == prefix or p in n.parents:
                matches.append((len(p.parts), item))
                break
    if not matches:
        raise KeyError(f"No SONIC department owns repository path: {path}")
    most_specific = max(depth for depth, _ in matches)
    owners = [item for depth, item in matches if depth == most_specific]
    if len(owners) != 1:
        names = ", ".join(item.subsystem for item in owners)
        raise RuntimeError(f"Ambiguous SONIC ownership for {path}: {names}")
    return owners[0]


def owner_for_path(path: str) -> tuple[Department, Team, str]:
    item = ownership_for_path(path)
    return item.department, item.team, item.subsystem


def validate_architecture() -> None:
    names = [item.subsystem for item in SUBSYSTEM_OWNERSHIP]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate subsystem ownership declaration detected")
    prefixes: dict[str, str] = {}
    team_departments = {item.name: item.department for item in TEAMS}
    for item in SUBSYSTEM_OWNERSHIP:
        if team_departments.get(item.team) != item.department:
            raise ValueError(f"Team {item.team} has inconsistent department ownership")
        for prefix in item.path_prefixes:
            existing = prefixes.get(prefix)
            if existing is not None:
                raise ValueError(f"Path prefix {prefix} is owned by both {existing} and {item.subsystem}")
            prefixes[prefix] = item.subsystem


validate_architecture()
