# SONIC Engineering Operating Model

## Purpose

SONIC is a general-purpose autonomous AI-human digital being operating a
dedicated computer workstation. It combines persistent identity, memory,
reasoning, planning, GUI and browser interaction, terminal and file execution,
Git, self-improvement, safety, evidence, and evaluation.

SONIC must not be reduced to a cybersecurity scanner, chatbot, or scripted
multi-agent wrapper. This document defines how engineering work changes the
existing system without weakening that product intent.

## Organization and ownership

The organization is responsibility-first. The canonical source of repository
ownership is
[`architecture.py`](../services/core/sonic/organization/architecture.py).
Do not create another department, team, employee, subsystem, or ownership
registry.

The current organization contains seven departments:

1. Cognitive Systems
2. Computer & Interaction
3. Platform & Systems
4. Security & Safety
5. Evaluation & Reliability
6. Experience & Interface
7. Research & Evolution Lab

Every declared subsystem has exactly one department and one team owner.
Unknown repository paths must fail ownership resolution; they must not receive
an implicit default department.

## Engineering ownership chain

```text
Department
  -> Team
  -> Subsystem
  -> Employee / AI contributor
  -> Project
  -> Work item
  -> GitHub issue
  -> Short-lived work branch
  -> Implementation
  -> Tests and evidence
  -> Pull request
  -> Review and CI
  -> Evaluation
  -> Merge
  -> Release
```

The persistent organization runtime in
[`store.py`](../services/core/sonic/organization/store.py) and the
[`organization` API](../services/core/sonic/api/routes/organization.py) stores
tenant-scoped employees, projects, work items, and ownership references.
Employee-aware subagents receive their department, team, subsystem ownership,
and capability mode through
[`teamwork.py`](../services/core/sonic/computer_use/teamwork.py).

Ownership context is not a filesystem permission boundary. Safety policy,
workspace isolation, tenant checks, and provider controls remain authoritative.

## Department responsibilities

### Cognitive Systems

Owns cognition and autonomous decision-making: brain and reasoning, agents and
orchestration, LLM routing, memory and knowledge, Being identity and durable
self, curiosity, model selection, and controlled self-improvement.

### Computer & Interaction

Owns computer agency: workstation providers, computer-use actions, browsers,
screenshots, visual grounding, accessibility/CDP interaction, mouse and
keyboard motor control, window management, and visual-delta verification.

### Platform & Systems

Owns reliable service execution: FastAPI and CLI contracts, asynchronous
runtime, queues, tools, database and configuration infrastructure, logging,
integrations, Docker, the Rust kernel boundary, and CI/CD.

### Security & Safety

Is an independent control layer. It owns authentication and access boundaries,
ActionPolicy, sandbox and isolation, path confinement, egress controls,
credential boundaries, destructive-action controls, threat modelling, and
security controls. Other departments must not bypass it.

### Evaluation & Reliability

Owns proof rather than self-reported success: tests, benchmarks, regression
suites, falsification, failure analysis, health, observability, evidence
custody, performance measurement, and release evidence.

### Experience & Interface

Owns the human-facing surface: dashboard, mission and workstation views,
evidence and graph interfaces, experiments/evolution views, logs, telemetry,
approval, and human intervention.

### Research & Evolution Lab

Is cross-functional. It owns research workflows, experimental methods, and new
agent architectures without duplicating the technical ownership of Cognitive,
Computer, Platform, Security, or Evaluation subsystems. Experimental work must
produce reproducible evidence and promotion criteria before becoming a core
dependency.

## Git and contributor model

Use the repository's configured base branch and pull-request policy for each
change. Prefer one short-lived work branch per coherent work item; department-
named branches may be used as collaboration branches when the team workflow
requires them. Do not create duplicate PRs for the same commit:

```text
short-lived work branch
  -> pull request targeting the configured base branch
  -> review + CI + evaluation
  -> merge according to repository policy
```

Examples:

```text
feature/cognitive/agent-runtime-042
feature/computer/desktop-control-017
feature/platform/api-031
feature/security/action-policy-009
feature/evaluation/regression-014
feature/experience/dashboard-022
feature/research/evolution-memory-006
```

Do not assume branch names define source ownership or target base branches.
AI coding systems are temporary contributors; they must follow the same scoped
work, review and evidence requirements as other contributors.

## Before coding

An AI contributor must:

1. Read `AGENTS.md` and this document.
2. Resolve the requested path through the canonical ownership map.
3. Identify the department, team, subsystem, and work item.
4. Inspect neighboring interfaces and existing implementations.
5. Inspect relevant tests and persistence boundaries.
6. Determine whether the requested change is inside scope.
7. State cross-department dependencies before editing them.

An AI contributor must not silently redefine what SONIC owns or rewrite an
entire department in response to a scoped bug report.

## Audit and bug-fix mode

Department audits are evidence-based and should follow this order:

1. Cognitive Systems
2. Computer & Interaction
3. Platform & Systems
4. Security & Safety
5. Evaluation & Reliability
6. Experience & Interface
7. Research & Evolution Lab

Separate confirmed bugs, possible issues, architectural concerns, and cosmetic
issues. For every confirmed bug:

1. Reproduce it where practical.
2. Identify the root cause.
3. Add a focused regression test.
4. Fix the cause, not just the symptom.
5. Run the relevant tests, lint, and applicable type checks.
6. Inspect the final diff and check for regressions.

Never claim that all bugs are fixed because one test passed. Report confirmed
issues fixed, unverified areas, and remaining risks explicitly.

## Cross-department findings

If an audit discovers an issue owned by another department, report it rather
than silently modifying it:

```text
CROSS-DEPARTMENT FINDING
Department:
Team:
Subsystem:
File:
Problem:
Evidence:
Dependency:
Suggested owner:
```

A cross-department fix requires explicit scope and must be visible in the pull
request.

## Safety and database invariants

No engineering contributor may weaken or bypass:

- ActionPolicy and sealed policy checks;
- authentication, authorization, and tenant isolation;
- sandbox and workspace confinement;
- egress and destructive-action controls;
- human approval requirements;
- evidence custody and independent verification.

Database changes must preserve tenant scoping, foreign-key relationships,
indexes, migration compatibility, and existing production fields. Do not hide
errors with broad exception handling or silence type errors with unjustified
`Any`, casts, or ignores.

## Release gates

The minimum release model is:

```text
Lint
  + focused tests
  + relevant regression suites
  + applicable type checks
  + dashboard/build checks
  + security/P0 validation
  -> production container/build checks
  -> evaluation
  -> release evidence
```

A passing unit test proves only the behavior it covers. A release-ready claim
requires the required gates to have actually passed. GitHub issue, PR, review,
check-suite, merge, and release state must be based on real receipts, not
operator or agent assertions.

## Audit report format

```text
DEPARTMENT:
TEAM:
SUBSYSTEM:

AUDIT RESULT: PASS / PARTIAL / BLOCKED

CONFIRMED BUGS:
- File:
- Line:
- Root cause:
- Impact:
- Fix:
- Regression test:

LOGIC ISSUES:
SECURITY ISSUES:
TEST GAPS:
CROSS-DEPARTMENT FINDINGS:
FILES CHANGED:
TESTS RUN:
CI / BUILD:
REMAINING RISKS:
```

The objective is verified, traceable, maintainable engineering—not the number
of commits or the number of changes produced.
