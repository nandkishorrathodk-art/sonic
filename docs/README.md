# SONIC documentation

SONIC's product goal is a general-purpose AI computer user. These documents
describe intended behavior, current implementation, and evidence boundaries;
architecture diagrams and component tests must not be read as proof of
universal or live desktop-task capability.

## Start here

- [Product goal and GUI acceptance contract](SONIC_COMPUTER_USER_GOAL.md) —
  authoritative outcome, perception, grounding and verification requirements.
- [SONIC + AVCS system flow](SONIC_AVCS_SYSTEM_FLOW.md) — current path from
  workstation pixels through AVCS, reasoning, guarded actions and Judge review.
- [Getting started](GETTING_STARTED.md) — local development setup.
- [Architecture](ARCHITECTURE.md) — current components and their boundaries.
- [Deployment](DEPLOYMENT.md) — local and production configuration cautions.
- [Configuration reference](CONFIGURATION.md) — environment variables and settings.

## Engineering and organization

- [Engineering operating model](SONIC_ENGINEERING_OPERATING_MODEL.md) —
  ownership, contribution and review principles.
- [Employee and department architecture](EMPLOYEE_ARCHITECTURE.md) —
  registered ownership versus proposed staffing.
- [AI agent guide](AI_AGENT_GUIDE.md) — repository development and validation.
- [Rules and invariants](RULES.md) — safety, perception, execution and evidence rules.
- [Blueprint](BLUEPRINT.md) — product direction and implementation priorities.
- [License](LICENSE) — repository terms.
- [Historical architecture notes](ARCHITECTURE.md#historical-notes) — dated
  material is explicitly distinguished from current runtime guarantees.

## Visual assets

- [`assets/architecture.svg`](../assets/architecture.svg) — runtime flow showing
  AVCS System 1, SONIC System 2, action paths and Judge review. The separate
  SDK/MCP/adapters are explicitly marked as not wired into this runtime path;
  the diagram is not an end-to-end acceptance result.
- [`assets/sonic_cyber_workstation.png`](../assets/sonic_cyber_workstation.png) —
  historical interface snapshot with legacy cyber branding; not proof of task completion.
- [`assets/sonic_cognitive_memory_graph.png`](../assets/sonic_cognitive_memory_graph.png) —
  interface snapshot; not evidence that a particular live task was completed.

The copies under [`docs/assets/`](assets/) are kept in sync for documentation
and external references.
