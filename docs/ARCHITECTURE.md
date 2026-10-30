# SONIC architecture

## Product boundary

SONIC is intended to be a general-purpose AI computer user: it receives an
operator's desired outcome, reasons about the available environment, selects
appropriate computer capabilities, and reports only what it can verify. This
describes the product direction; it does not claim arbitrary installed
applications or end-to-end tasks are already reliable.

The authoritative acceptance contract is
[SONIC_COMPUTER_USER_GOAL.md](SONIC_COMPUTER_USER_GOAL.md). Ownership of
repository subsystems is defined in
[`architecture.py`](../sonic-core/sonic/organization/architecture.py).

## Runtime at a glance

```text
Operator outcome
  -> API/workstation session
  -> Boss planning, delegation, evidence review and replanning
  -> ordinary ComputerUseAgent workers
       -> fresh workstation state and, when needed, screenshot pixels
       -> reason about the assigned objective
       -> ground a visible GUI action or use an allowed execution capability
       -> action policy/provider dispatch and recorded receipt
       -> observe the resulting state
  -> independent Judge checks GUI completion against fresh evidence
  -> report verified results, partial progress, blockers or uncertainty
```

Chat and observation-only requests are not workstation execution. Actionable
workstation objectives enter Boss orchestration. Boss delegates to ordinary
`ComputerUseAgent` workers; there is no dedicated browser-specialist route.
Website work, when requested, is handled as visible-desktop work by those
workers.

## Main code areas

| Area | Source | Responsibility |
|---|---|---|
| Workstation and API | `sonic-core/sonic/api/routes/workstation.py` | Sessions, workstation actions, worklog and mission entry points |
| Orchestration | `sonic-core/sonic/computer_use/boss.py` | Task analysis, bounded worker dispatch, receipts, review and replanning |
| Computer-use worker | `sonic-core/sonic/computer_use/agent.py` | Observation, reasoning, action dispatch and worker traces |
| Desktop provider | `sonic-core/sonic/computer/docker_computer.py` | Local Docker workstation access and state/screenshot observations |
| NSI state feed | `docker/` workstation entrypoint and NSI components | Compact versioned state, visible text, controls and changed-region metadata |
| Grounding | `sonic-core/sonic/computer_use/grounding.py` | Coordinate-space parsing and grounded target resolution |
| Policy | `sonic-core/sonic/safety/` | Action policy, scope, egress and sealed-policy controls |
| Independent review | `sonic-core/sonic/computer_use/judge.py` | Checks completion proposals against fresh evidence and worker receipts |
| Dashboard | `sonic-dashboard/` | Operator-facing workstation and work status |

Exact wiring varies by route and provider. A component's presence in this table
does not prove every route uses it; verify the relevant production call path
before making behavior claims.

## Computer and execution capabilities

The normal local workstation route uses `DockerComputerProvider`. Its graphical
desktop and separate execution capabilities complement one another. The
operator's requested outcome and live state should determine which capability
is appropriate. A GUI-only assignment must stay GUI-only; terminal work must
not be silently substituted for the requested visible interaction.

The Docker workstation exposes a compact NSI sidecar rather than embedding
screenshot data in every state update. `observe_state()` validates freshness;
the worker can request a screenshot when visual evidence is needed. Fresh,
unambiguous structured accessibility state may support some reasoning without
full-frame transfer. GUI actions still require grounded current evidence and
fresh post-action observation. See the
[perception contract](SONIC_COMPUTER_USER_GOAL.md#architectural-direction)
for coordinate and ambiguity rules.

This is not a guarantee of complete event subscriptions, zero-latency updates,
or reliable operation in every application. Other providers may use different
observation paths.

## Completion and evidence

- Boss owns task analysis, delegation, evidence review and replanning.
- A worker's `GOAL_COMPLETE` is a handoff proposal, not a success verdict.
- GUI completion requires a Judge decision based on fresh desktop evidence and
  worker action receipts.
- Missing, stale, ambiguous or inconclusive evidence cannot establish success.
- An unchanged frame/state is inconclusive evidence, not proof that an action
  succeeded.
- Unit tests validate code contracts; they do not demonstrate a real,
  multi-application workstation task.

## Safety and operational limits

Safety controls are implemented across Python policy, scope/egress helpers,
providers and, where wired, the native kernel bridge. Do not assume every
network path or provider has identical enforcement. Audit the actual dispatch
path before changing or documenting a boundary. The workstation container is
not, by itself, a security boundary against every host or Docker-daemon risk.
Do not expose raw VNC/noVNC ports beyond loopback without an independently
secured access design.

Production compose selects an explicitly configured Daytona workstation
provider; local Docker development and production deployment are not
interchangeable. Production readiness must be evaluated for the exact deployed
configuration.

## Historical notes

Older diagrams, phase logs and design documents describe capabilities,
providers, browser integrations or aspirations that have since changed. Treat
them as historical unless current source wiring and task-specific evidence
confirm the claim. The files under `docs/assets/` include historical UI
screenshots; they are not current live captures.
