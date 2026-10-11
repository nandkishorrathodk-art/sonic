# SONIC — General-Purpose AI Computer User

SONIC is being developed as an AI colleague that can understand an operator's
desired outcome, use the available computer and applications, and verify useful
results. Its product goal is **general-purpose computer use**, not a browser-only
assistant or a cybersecurity-first bot.

That is the goal, not a universal capability claim. A running application,
model response, screenshot, unit test, or worker's completion proposal does not
by itself prove that a real desktop task succeeded.

## What is in this repository

| Area | Location | Role |
|---|---|---|
| Backend and APIs | [`sonic-core/`](sonic-core/) | FastAPI services, orchestration, computer-use runtime, policy and evidence |
| Operator dashboard | [`sonic-dashboard/`](sonic-dashboard/) | Next.js workstation and operations interface |
| CLI | [`sonic-cli/`](sonic-cli/) | Command-line client |
| Desktop/container setup | [`docker/`](docker/) and [`docker-compose.yml`](docker-compose.yml) | Local graphical workstation and supporting services |
| Product acceptance contract | [`docs/SONIC_COMPUTER_USER_GOAL.md`](docs/SONIC_COMPUTER_USER_GOAL.md) | Authoritative computer-use goal and evidence requirements |
| Architecture and engineering docs | [`docs/`](docs/) | Setup, architecture, deployment and contributor guidance |

## Architecture and interface snapshots

![SONIC computer-use architecture overview](assets/architecture.svg)

The following PNGs are historical interface snapshots. They retain legacy
cyber-oriented dashboard branding and do not show a verified computer-use task:

| Snapshot | Context |
|---|---|
| [`SONIC workstation interface`](assets/sonic_cyber_workstation.png) | Historical workstation UI; not a live capture or task-success receipt |
| [`Cognitive memory graph`](assets/sonic_cognitive_memory_graph.png) | Interface snapshot; not proof of current runtime state |

## Current implementation, honestly stated

The codebase includes a Docker-backed graphical workstation, generic GUI
actions, local AVCS pixel-reflex processing, orchestration, action receipts
and an independent Judge handoff for GUI completion. Workstation screenshots
are processed locally and are not sent to the workstation reasoning model.
AVCS semantic target grounding and focus verification require evaluated local
model heads; unsupported actions fail closed. A separate terminal and
file-execution plane is available for work that is appropriate there. Safety
and workspace controls apply to supported execution paths.

These are implementation foundations, not proof of reliable work in every
installed application. Current acceptance still requires a fresh observation,
grounded visible actions, post-action evidence and independent Judge
confirmation. **Reliable end-to-end completion of representative tasks across
multiple non-browser applications has not yet been demonstrated.**

The latest backend regression run on the current development worktree completed
with **1,699 passed and 39 skipped**. This is a unit/integration-suite result,
not a live multi-application desktop acceptance result. Skips and live-runtime
requirements are documented in the product contract and test configuration.

## Quick start (local development)

### Prerequisites

- Python 3.12 or newer
- Node.js and npm (for the dashboard)
- Docker Engine or Docker Desktop (for the graphical workstation and optional services)
- An API key for a provider configured in [`configs/models.yaml`](configs/models.yaml) if you want model-backed reasoning

### 1. Configure local environment

```powershell
Copy-Item .env.example .env
```

Set a local `SONIC_VNC_PASSWORD` before starting the workstation. Add credentials
for the model provider you intend to use. Treat `.env` as private; never commit
credentials. The example Neo4j credentials are for local development only.
For a predictable developer run, disable the background self-coder explicitly
in the backend shell; it is enabled by default and may create candidate commits
on its own branch when an LLM is available.

### 2. Start the local containers

```powershell
docker compose up -d
```

The compose file starts supporting services and the graphical workstation; it
does **not** start the API or dashboard. Local VNC/noVNC ports are bound to
loopback. Do not expose them directly to an untrusted network.

### 3. Install and start the API

From the repository root:

```powershell
python -m pip install -e "sonic-core[dev]"
$env:APP_ENV = "development"
$env:SONIC_SELF_CODER_ENABLED = "false"
$env:CORS_ORIGINS = "http://127.0.0.1:12001,http://localhost:12001"
python -m uvicorn sonic.api.main:app --host 127.0.0.1 --port 12000
```

Health check: `http://127.0.0.1:12000/health`

### 4. Start the dashboard

In another PowerShell window:

```powershell
Set-Location sonic-dashboard
npm install
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:12000"
npm run dev
```

Open `http://127.0.0.1:12001`. For platform-specific setup, environment
variables, or production deployment, see the [documentation index](docs/README.md).

## Test commands

Run backend tests from the repository root:

```powershell
python -m pytest sonic-core/tests/
```

Some tests require live Docker or workstation infrastructure and can be skipped
when it is unavailable. Passing this suite does not substitute for the
multi-application acceptance test described in
[`docs/SONIC_COMPUTER_USER_GOAL.md`](docs/SONIC_COMPUTER_USER_GOAL.md).

## Product principles

- Start from the operator's requested outcome; do not force an application or tool sequence.
- Discover and operate installed GUI applications from current workstation state.
- Use a terminal or API when appropriate, but do not silently substitute it for a requested visible GUI task.
- Block ambiguous or stale GUI targets rather than guessing.
- Report progress, uncertainty, blockers and verified outcomes separately.
- Treat application launch and agent self-reports as proposals, not proof of task completion.

## Ownership and license

The authoritative software-department and subsystem registry is
[`architecture.py`](sonic-core/sonic/organization/architecture.py).
This repository is proprietary; see the [license](docs/LICENSE) for its terms.
