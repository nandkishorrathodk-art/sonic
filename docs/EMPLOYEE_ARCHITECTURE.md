# SONIC Employee & Engineering Architecture

## Purpose

This document defines how SONIC organizes people, engineering ownership, projects, review responsibility, and growth as the project scales from a small research team into a production engineering organization.

The organization is responsibility-first, not technology-first. Python, Rust, Docker, Redis, Neo4j, Next.js and other technologies belong to functional teams rather than becoming separate departments.

## Organization

```
SONIC
│
├── Founder / Chief Architect
│
├── Engineering
│   ├── Cognitive Systems
│   ├── Computer & Interaction
│   ├── Platform & Systems
│   ├── Security & Safety
│   ├── Evaluation & Reliability
│   └── Experience & Interface
│
├── Research & Evolution Lab
│
└── Operations
    ├── People / HR
    ├── Legal
    └── Finance / Operations
```

Operations may remain lightweight and external/part-time until scale requires dedicated ownership.

## Department responsibilities

### 1. Cognitive Systems

**Mission:** SONIC's cognition and autonomous decision-making.

Owns:

- agent runtime and orchestration;
- reasoning and planning;
- model routing;
- tool selection;
- persistent memory and knowledge;
- BeingMind, BeingCraft and BeingVault;
- curiosity and life-loop behaviour;
- controlled self-improvement.

Typical roles:

- AI/Agent Engineer;
- LLM Engineer;
- Memory/Knowledge Engineer;
- AI Research Engineer.

### 2. Computer & Interaction

**Mission:** SONIC's perception and computer agency.

Owns:

- desktop control;
- browser interaction;
- visual perception;
- CDP/DOM grounding;
- mouse and keyboard motor control;
- window management;
- visual-delta verification;
- multi-application interaction.

Typical roles:

- Computer-Use Engineer;
- Browser Engineer;
- Vision/Perception Engineer;
- Interaction/Motor Engineer.

### 3. Platform & Systems

**Mission:** reliable execution and infrastructure.

Owns:

- FastAPI and service APIs;
- asynchronous execution;
- Rust kernel and Python/Rust boundaries;
- Docker/runtime infrastructure;
- Redis and persistence infrastructure;
- authentication and tenancy;
- CLI;
- CI/CD;
- deployment and operational reliability.

Typical roles:

- Backend Engineer;
- Rust/Systems Engineer;
- Infrastructure Engineer;
- DevOps/SRE Engineer.

### 4. Security & Safety

**Mission:** enforce safe, isolated and auditable execution.

Owns:

- ActionPolicy;
- permission boundaries;
- sandboxing;
- workspace isolation;
- path confinement;
- egress controls;
- destructive-action controls;
- credential boundaries;
- security testing and threat modelling.

Typical roles:

- Security Engineer;
- Sandbox Engineer;
- Application Security Engineer;
- Security Researcher;
- Safety/Policy Engineer.

### 5. Evaluation & Reliability

**Mission:** prove that SONIC works rather than accepting self-reported success.

Owns:

- unit/integration/E2E testing;
- agent evaluations;
- benchmarks;
- regression suites;
- falsification;
- failure analysis;
- performance and reliability testing;
- release evidence.

Typical roles:

- QA Engineer;
- Evaluation Engineer;
- Reliability Engineer;
- AI Evaluation Researcher.

### 6. Experience & Interface

**Mission:** human-facing interaction and observability.

Owns:

- Next.js dashboard;
- mission interfaces;
- computer views;
- evidence board;
- graph/memory interfaces;
- experiments UI;
- telemetry and observability;
- human approval/intervention surfaces.

Typical roles:

- Frontend Engineer;
- Full-Stack Engineer;
- UX Engineer;
- Observability Engineer.

### 7. Research & Evolution Lab

This is initially a cross-functional lab rather than a large permanent department.

Researchers may temporarily work with Cognitive, Computer, Platform, Security or Evaluation teams to test new methods.

Research work must still produce reproducible experiments, measurable outcomes and clear promotion criteria before experimental code becomes a core dependency.

## Employee hierarchy

### Individual contributor

```
Contributor / Intern
        ↓
Junior Engineer
        ↓
Engineer
        ↓
Senior Engineer
        ↓
Staff Engineer
        ↓
Principal Engineer
        ↓
Distinguished Engineer
```

### Technical leadership

```
Engineer
   ↓
Tech Lead
   ↓
Department Lead
   ↓
Director / Head of Engineering
   ↓
VP Engineering
```

Technical and management tracks are separate. Becoming a manager is not required for technical progression.

## Initial team size

For the current development stage, a practical target is approximately 9–11 contributors plus the Founder/Chief Architect:

| Area | Initial contributors |
|---|---:|
| Cognitive Systems | 2–3 |
| Computer & Interaction | 2 |
| Platform & Systems | 2 |
| Security & Safety | 1 |
| Evaluation & Reliability | 1 |
| Experience & Interface | 1 |
| Research Lab | Cross-functional |

Department leads can remain hands-on engineers at this size.

## Scaling model

### 5–12 people

Founder + functional leads. No unnecessary management layers.

### 12–25 people

Formal team ownership begins. One engineering lead can coordinate multiple departments where appropriate.

### 25–50 people

Dedicated department heads, research leadership, security leadership and reliability leadership become practical.

### 50+ people

Departments can split into specialized teams and dedicated platform, SRE, developer-experience, research and product organizations.

## Work ownership model

Every production change should map:

```
Employee
  ↓
Department
  ↓
Team
  ↓
Subsystem ownership
  ↓
Project / Epic
  ↓
GitHub Issue
  ↓
Development branch
  ↓
Implementation
  ↓
Tests + Evidence
  ↓
Pull Request
  ↓
Review + CI
  ↓
Evaluation
  ↓
Merge
  ↓
Release
```

No employee should be assigned a vague responsibility such as "work on SONIC". Ownership should identify the subsystem and measurable outcome.

## Cross-functional projects

Large capabilities should have a Project Lead while subsystem ownership remains with departments.

Example:

```
Autonomous Browser Research
│
├── Cognitive Systems → planning/reasoning
├── Computer & Interaction → browser control/perception
├── Security & Safety → policy/isolation
└── Evaluation & Reliability → benchmarks/falsification
```

The Project Lead coordinates delivery; department owners remain accountable for the technical quality of their subsystem.

## Promotion and responsibility

Promotion should be based on demonstrated scope:

- quality of delivered work;
- ownership of a subsystem;
- reliability of execution;
- review quality;
- test/evaluation discipline;
- ability to diagnose failures;
- architectural judgement;
- ability to mentor or coordinate others where applicable.

Time alone is not a promotion criterion.

## Non-negotiable engineering principle

SONIC is an autonomous system. Therefore organizational quality must mirror system quality:

**clear ownership, independent verification, observable evidence, controlled execution, and no success-by-decree.**


## Software architecture ownership contract

The organizational model is also a code-ownership model. The canonical registry lives in `sonic-core/sonic/organization/architecture.py` and assigns every declared SONIC subsystem to exactly one department and one team.

| Department | Teams | Primary subsystems |
|---|---|---|
| Cognitive Systems | Reasoning & Orchestration; Model & Memory; Identity & Evolution | brain, agents, llm, memory, being, evolution, swarm |
| Computer & Interaction | Desktop Control; Browser & Web; Perception & Motor | computer_use, computer, browser, perception, world |
| Platform & Systems | API & Services; Systems Runtime; Data & Infrastructure | api, execution, kernel, queue, tools, integrations, db, config, logger, cli |
| Security & Safety | Trust & Safety; Sandbox & Isolation; Identity & Access | auth, safety, sandbox, net |
| Evaluation & Reliability | Verification & Benchmarks; Reliability & Observability; Evidence & Release | evidence, health, observability, mission_engine |
| Experience & Interface | Product UI; Mission & Operator UX | dashboard |
| Research & Evolution Lab | Agent Research; Methods & Experiments | research, researcher, meta, orchestration |

Ownership is resolved from repository paths with no implicit default department. An unowned path must fail validation instead of silently receiving an owner. A cross-functional project may involve multiple departments, but each repository subsystem retains one authoritative technical owner.

### Employee assignment rule

An AI employee profile must resolve to:

```
Department -> Team -> Subsystem -> Repository path scope
```

Projects and work items may span departments, but implementation ownership must point to a declared subsystem. This prevents an employee from being assigned a vague responsibility such as "work on SONIC" and makes GitHub issues, branches, reviews and release evidence traceable to an architectural owner.


## Persistent employee runtime

The employee operating model is implemented by the organization control plane. Employee profiles persist per tenant and carry department, team, subsystem ownership, role, prompt context and read-only state. Projects and work items are also tenant-scoped; every work item records the canonical subsystem owner plus optional GitHub issue/PR and development-branch references.

The organization runtime is itself a declared software subsystem, `organization`, owned by Platform & Systems / API Services. It must use `SUBSYSTEM_OWNERSHIP` from the canonical architecture module rather than maintaining a second subsystem registry.

A work item cannot be assigned to an employee who does not own the requested subsystem. A work item cannot enter `released` until the organization store has receipts for both an approved GitHub review and a successful CI/check-suite event for the referenced PR. The integration records GitHub evidence; it does not create branches, open PRs, merge code or push commits automatically.
