# SONIC departments and subsystem ownership

## What this document describes

SONIC's seven software departments are defined by the canonical registry at
[`architecture.py`](../sonic-core/sonic/organization/architecture.py). The
registry is code ownership metadata; it does not prove a department is staffed
by people or that an employee is running. Staffing levels and career ladders
below are organizational proposals, not runtime facts.

## Registered departments

| Department | Registered teams | Owned subsystems |
|---|---|---|
| Cognitive Systems | Reasoning & Orchestration; Model & Memory; Identity & Evolution | `brain`, `agents`, `llm`, `memory`, `being`, `evolution`, `swarm` |
| Computer & Interaction | Desktop Control; Browser & Web; Perception & Motor | `computer_use`, `computer`, `browser`, `perception`, `world` |
| Platform & Systems | API & Services; Systems Runtime; Data & Infrastructure | `api`, `execution`, `kernel`, `queue`, `tools`, `integrations`, `db`, `config`, `logger`, `cli` |
| Security & Safety | Trust & Safety; Sandbox & Isolation; Identity & Access | `auth`, `safety`, `sandbox`, `net` |
| Evaluation & Reliability | Verification & Benchmarks; Reliability & Observability; Evidence & Release | `evidence`, `health`, `observability`, `mission_engine` |
| Experience & Interface | Product UI; Mission & Operator UX | `dashboard` |
| Research & Evolution Lab | Agent Research; Methods & Experiments | `research`, `researcher`, `meta`, `orchestration` |

Each declared subsystem has one authoritative department and team. Unknown
paths must fail ownership resolution rather than receive an implicit owner.
The Computer & Interaction department owns the general computer-use surface,
not just web browsing; its scope includes desktop GUI, perception, grounding,
motor control and multi-window interaction.

## Responsibilities

- **Cognitive Systems:** reasoning, agent orchestration, model routing, memory,
  persistent Being systems and controlled self-improvement.
- **Computer & Interaction:** visible desktop use, workstation providers,
  browser interactions as one application category, perception, grounding,
  pointer/keyboard control and window management.
- **Platform & Systems:** APIs, runtime, tools, data infrastructure,
  integrations, configuration, logging, CLI and Rust/Python boundaries.
- **Security & Safety:** identity and access controls, policy, sandboxing,
  isolation, path confinement, egress and approval rules.
- **Evaluation & Reliability:** tests, benchmarks, failure analysis,
  observability, evidence custody and release evidence.
- **Experience & Interface:** dashboard and other operator-facing controls,
  telemetry, approvals and intervention workflows.
- **Research & Evolution Lab:** research and experimental methods; this
  cross-functional work does not override the registered ownership of core
  subsystems.

## Employee and project runtime

The organization runtime can store tenant-scoped employee profiles, projects,
work items and ownership references. A profile or role is not proof of an
active worker. Ownership context is not a filesystem or security boundary;
authorization and workspace controls remain authoritative.

Work items should resolve to:

```text
department -> team -> subsystem -> repository path -> measurable outcome
```

Projects may span departments while each declared subsystem retains one
technical owner. GitHub issue/PR references and signed review/CI receipts are
records; the organization integration does not itself push, merge or create
branches/PRs.

## Suggested staffing (not current staffing)

For planning only, a small team could start with hands-on contributors across
the seven areas and cross-functional research. Headcount, titles, leadership
tiers and scaling thresholds are planning suggestions, not facts about SONIC's
current employees or autonomous workforce.
