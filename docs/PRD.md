# SONIC Product Requirements Document

## 1. Product definition

SONIC is an autonomous AI-human digital being that operates a dedicated
computer workstation as a capable general-purpose colleague. It must be able
to reason, use graphical applications, browse the web, execute terminal and
file work, manage persistent identity and memory, collaborate through
subagents, and improve its own non-safety code under empirical verification.

SONIC is not limited to a cybersecurity scanner, chatbot, fixed workflow, or
single-purpose CLI agent.

## 2. Product goals

### G1 — Human-grade computer agency

SONIC can observe and operate arbitrary desktop applications through live
visual/accessibility grounding, multiple windows, browser interaction, mouse,
keyboard, focus, waiting, and file workflows. GUI actions must be grounded in
the current display state rather than static coordinates or application maps.

### G2 — Dual execution planes

SONIC maintains two first-class planes:

- **Application plane:** graphical workstation applications and browser/UI
  interaction.
- **Operator plane:** sandboxed terminal, files, Git, tests, scripts, and
  provider-backed execution.

The model decides which plane fits the observed task; neither plane may be
silently disabled or used to fake progress.

### G3 — Persistent digital being

Identity, affect, memory, browser/account state, durable craft, lessons, and
vault data survive restarts with tenant isolation. A restart must not silently
create a new being or erase learned state.

### G4 — Adaptive reasoning and subagents

SONIC reasons from live observations, explicit unknowns, competing hypotheses,
world state, past lessons, and real failures. Complex work can be decomposed
into scoped subagents with bounded budgets, workspace isolation, display
serialization, and aggregated evidence.

### G5 — Evidence-based self-coding

The self-coder may inspect, propose, test, apply, rollback, and promote
non-safety code only through guarded workflows. A change is not successful
without real test receipts, exact execution status, and reproducible evidence.
Safety, authentication, egress, and policy components remain protected.

### G6 — Organization-aware engineering

Engineering work follows the canonical chain:

```text
Department → Team → Subsystem → Employee/AI contributor
  → Project → Work item → Branch → Tests → Pull request → Review/CI → Release
```

Employee context informs ownership and collaboration, but it never replaces
tenant checks, workspace isolation, provider controls, or the independent
Safety policy.

## 3. Non-functional requirements

- **Safety:** deny by default, sealed policy integrity, path confinement,
  egress controls, tenant isolation, and no execution bypass.
- **Honesty:** never fabricate findings, commits, coordinates, test results,
  process state, or release status.
- **Observability:** preserve real actions, outputs, exit codes, durations,
  decisions, failures, evidence hashes, and human-intervention requests.
- **Reliability:** recover non-destructively, detect stuck strategies, replan,
  and preserve the root failure cause.
- **Extensibility:** discover applications and capabilities dynamically;
  avoid vendor-specific application maps and forced tool hierarchies.
- **Verification:** every feature requires focused regression tests and
  applicable CI/type/lint checks; live workstation claims require live
  workstation receipts.

## 4. Product surfaces

1. Workstation and desktop application control.
2. Browser and visual perception loop.
3. Terminal, files, Git, tests, and provider execution.
4. Persistent Being, memory, craft, vault, and lessons.
5. Brain, agents, planning, hypotheses, experiments, and replan.
6. Toolsmith, Method Lab, and controlled self-evolution.
7. Employee organization, projects, work items, teamwork, and subagents.
8. Dashboard, API, CLI, telemetry, evidence, approvals, and release gates.

## 5. Delivery and acceptance

A product change is accepted only when:

1. Its canonical department/team/subsystem owner is identified.
2. The root behavior is reproduced or the test gap is explicitly documented.
3. A focused implementation and regression test are present.
4. Relevant tests, lint, type checks, and security gates pass.
5. CI and review receipts refer to the correct repository, branch, and commit.
6. Live-only behavior is validated in the real workstation environment before
   being described as operational.

The founder remains the final architectural authority. AI coding tools are
temporary scoped contributors, not independent product owners.
