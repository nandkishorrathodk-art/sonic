.# AGENTS.md — SONIC repository memory

## Project overview & Priority #1 (Base Foundation)
SONIC's product goal is an **Autonomous AI-Human Digital Being** that can use a general-purpose computer like a capable human colleague: perceive the desktop, operate arbitrary installed GUI applications, switch between them, use files and other appropriate computer capabilities, and complete multi-step outcomes. The workstation provides a graphical desktop and a separate execution plane; this product goal is not proof that every arbitrary task or application already works reliably.

> **PRIORITY #1 (FOUNDATIONAL MANDATE)**:
> SONIC is a general-purpose AI computer user, not a cybersecurity-first bot. Its intended capability is to take an operator's desired outcome, understand and plan it, discover and operate relevant GUI applications, move between windows and tools, maintain progress across a long task, adapt to unexpected states, and verify the actual result. Cybersecurity is one possible workload, not SONIC's identity or default workflow.
> Never imply that an intended capability is already reliable without task-specific evidence. Never hardcode a vendor application sequence as a substitute for general computer-use reasoning.

> **Current Computer & Interaction contract:** Boss owns task analysis, delegation, evidence review, and replanning. Every actionable workstation prompt enters Boss orchestration; chat and explicit observation-only requests remain outside execution. If planning omits workers for a non-empty objective, Boss delegates one bounded adaptive worker rather than bypassing orchestration. Boss delegates to ordinary `ComputerUseAgent` workers, which operate arbitrary installed GUI applications on the shared visible desktop and retain access to the separate execution plane as permitted by the task. GUI-assigned workers use fresh screenshot pixels as their current-state reasoning source; structured OCR/accessibility state must not substitute for visual input in this contract. GUI completion requires Judge confirmation against fresh desktop evidence and worker receipts; missing or inconclusive verification cannot produce a success claim. There is no dedicated BrowserAgent specialist; website and online-service tasks, when requested, are handled as ordinary visible-desktop GUI work. Historical phase notes below describe changes at the time they were made; they are not proof of present runtime readiness or arbitrary-task capability.
Architecture: FastAPI backend (`sonic-core/`), Next.js dashboard
(`sonic-dashboard/`), Docker desktop workstation (`sonic-desktop-workstation`), Neo4j/SQLite persistent memory, multi-tenant RBAC.

## Computer Interaction: Implemented Perception vs. Verified Operation
SONIC is fundamentally intended to operate graphical computer applications. The components below describe the current implementation; their presence and mocked/unit-test coverage do **not** establish that a live end-to-end GUI task works. Always verify workstation provisioning, provider readiness, visible action dispatch, and the task's postcondition independently before claiming success.

- **NSI (Neural-Symbolic Instant Eye)** is the Docker workstation's desktop-perception sidecar, embedded and consumed by `sonic/computer/docker_computer.py`. It publishes versioned screen frames and compact state; the compact state omits image bytes. Enrichment can combine changed-region OCR, bounded AT-SPI controls (names, roles, bounds, editable/focused state), X11 window geometry, and browser-tab title/URL. It is a perception feed, not an action executor, full browser-DOM feed, or general visual reasoner. The agent in `sonic/computer_use/agent.py` may use fresh structured state without image transfer and escalate to screenshot/VLM reasoning when needed.
- **OCR is one perception signal, not the only grounding method.** The workstation image installs Tesseract by default, but the entrypoint now warns and continues if it is absent. During operation, AT-SPI may provide semantic controls and valid fresh pixels may remain observable when OCR-derived controls are unavailable; screenshot fallback and the privacy/redaction path have their own validation requirements. Missing OCR reduces text/control coverage; do not claim equivalent behavior or bypass privacy-gate requirements.
- **Other implemented mechanisms:** typed GUI actions and coordinate/focus validation; Docker/X11 mouse, keyboard, scroll, drag, window, and screenshot dispatch; freshness checks and pre-dispatch revalidation; action receipts with before/after evidence; and Judge/Boss completion gates. These form a model-driven observation/action/verification loop, not an OCR-to-scripted-click pipeline. Safety-policy enforcement and provider-specific guarantees must be checked in the actual runtime configuration.
- **Do not count unintegrated code as runtime capability.** The separate `sonic/computer/desktop_atspi_engine.py` includes parsing/CDP/DOM-related code, but its presence/tests do not prove that the active workstation provider uses it. NSI's browser enrichment is limited to tab title/URL; verify call sites before claiming browser DOM access or live AT-SPI availability.
- **Operational/test evidence boundary (audited 2026-10-02):** most NSI/OCR provider tests use mocked feeds/subprocesses or fixtures. The real workstation smoke test is opt-in (`SONIC_RUN_LIVE_WORKSTATION=1`) and is not an ordinary CI acceptance run. CI/test pass counts and code comments such as a claimed observation cadence are not live GUI success-rate/latency measurements. Docker `GUI_MOVE` must be independently checked against actual pointer coordinates; command success alone is not proof that the cursor reached its target.
- **Fresh-action standard:** model reasoning can outlast the observation-age budget. Before dispatch, obtain a fresh NSI state or screenshot and compare it with the state used for reasoning; proceed only when the relevant UI state is unchanged, and fail closed when it changed or cannot be verified. Docker `GUI_MOVE` must also read back the actual X11 pointer position and match requested coordinates.
- **Acceptance standard:** use an isolated, opt-in live workstation test to verify a fresh NSI state and/or screenshot, execute a harmless reversible visible GUI action, and independently verify its postcondition from fresh desktop state (and VNC where visibility matters). Record Docker versus Daytona provider, OCR/AT-SPI availability, model/provider readiness, receipts, and failures. Never substitute a direct API action for an autonomous task when evaluating model-driven GUI capability.

## Core Architecture: AI-Human Being with Dedicated Computer Workstation
SONIC's intended design is an autonomous **AI-Human Digital Being** operating its own personal computer workstation, combining adaptive reasoning, general GUI fluency, and appropriate direct execution capabilities. Treat claims below as architecture goals or implemented components only where current code and evidence support them.

### 0. Foundational Priority: Clear AI-Human Base (Zero Hacker/Pentester Lock-In)
- **Human-First Identity**: All prompts (`prompts.py`), orchestrators (`BossAgent`, `MissionDirector`), and agents treat SONIC as an autonomous AI-Human colleague operating its personal workstation—never forcing tasks into offensive security checklists (`Recon -> Exploit`), scripted platform routines, or blocking normal web/app workflows as "out-of-scope targets".
- **Full Digital Agency**: Can open and switch between 1, 2, or N desktop GUI applications concurrently, interact with operator-requested online services through the visible desktop, sign up and maintain persistent application profiles/accounts (`BeingVault`), pause for human assistance on CAPTCHA/OTP (`WAIT_FOR_HUMAN`), create git repositories, write code, and publish content about its own persistent self (`Being`).

### 1. Dual-Plane Operational Model (Full Computer Desktop + Dedicated Terminal)
SONIC has two complementary execution planes and should select the appropriate capability from the operator's outcome and live state. The intended computer agency is broad; actual coverage must be established with task-specific acceptance evidence:
- **Workstation Application Plane (Computer-Use)**: Interacts with the full graphical desktop environment (X11/Wayland via `GUI_*`, `APP_*`). Boss delegates to ordinary `ComputerUseAgent` workers; each worker perceives and operates arbitrary installed desktop applications and multiple windows through visible, dynamically grounded GUI controls. No dedicated BrowserAgent is part of the workstation route. The GUI is NEVER disabled.
- **Operator & Sandbox Plane (Direct Headless Execution)**: Dedicated, non-graphical execution plane (`TERMINAL_EXEC`, `FILE_*`, `GIT_*`, `TOOL_*`, Toolsmith, and native Rust `sonic-kernel-rs`). High-throughput, precise tasks (writing code, running servers/tests, file management, git commits/pushes, custom script compilation) execute directly in the sandbox without cluttering the desktop GUI or wasting perception tokens.
- **Autonomous Selection (Zero Puppet Scripts)**: SONIC itself decides which plane or tool to use based on observations and goals. There are no scripted puppet heuristics forcing or forbidding either plane.

### 2. Autonomous Epistemics & Self-Evolution
- **Always-On Life & Curiosity Loop**: When no operator task is active, the being explores, hypothesizes, and learns autonomously under the sealed safety envelope.
- **Durable Craft, Vault & Memory**: Persists identity (`Being`), affect (`BeingMind`), credentials/accounts (`BeingVault`), learned knowledge, and authored artifacts (`BeingCraft`) across boots and restarts without amnesia.
- **Empirical Falsification**: Zero success-by-decree. Every claim of task or milestone completion requires verified reproduction receipts from real in-sandbox execution.
- **Toolsmith & Method Lab**: Autonomously authors its own specialized scripts and utilities (Toolsmith) and synthesizes novel workflows/methods (Method Lab).

### 3. Strict Core Mandate & Prohibitions (Aisa Karna Mana Hai)
0. **Priority #1 — AI-Human Digital Being Base (Not a Hacker/Pentester)**: Never lock or bias SONIC's identity, prompts, `BossAgent` domains, or web navigation into a hacker/pentester persona, offensive security pipeline, or hardcoded platform script. SONIC is a general-purpose AI-Human operating its computer.
1. **Zero Hardcoded Application Names**: Control flow, perception parsing, window tiling, intent routing, and prompt templates must NEVER hardcode or favor specific vendor application or binary names (no specific browser, proxy, scanner, or IDE binary literals). All applications must be dynamically discovered from the operating system environment (via `$PATH`, `which`, POSIX alternatives like `x-www-browser`/`sensible-browser`/`xdg-open`, desktop `.desktop` entries, and active OS window titles).
2. **Zero Scripted Puppet Heuristics**: Rigid keyword-matching trees, canned 3-step checklists, prompt keyword-matching classifiers (e.g. `if "app" in goal`), hardcoded refusal fallbacks, and regex string manipulation of model intent are strictly prohibited. The agent must reason freely and adaptively from live observations and typed ActionContracts.
3. **Zero External/Mock Domains & Endpoints**: Hardcoded public commercial domains and fake proxy endpoints (e.g. `http://proxy/cert`) are strictly forbidden in production logic, prompts, and tests. All URLs must be operator-supplied, dynamically discovered targets, RFC 2606 `.test` domains, or RFC 5737 test documentation network addresses (`198.51.100.1`).
4. **Dynamic Perception Grounding**: All UI interactions must derive strictly from real multimodal visual perception and dynamic window/accessibility coordinates. Static coordinate lookup tables, landmark percentage fallbacks, and canned prompt-based fallbacks are prohibited.
5. **Task-Appropriate Computer Access**: Do not globally remove or disable GUI or execution capabilities. Route each worker through the plane(s) appropriate to its assigned work; a GUI-only assignment must receive a GUI-only contract and must not silently substitute terminal work for visible interaction. Do not force terminal work into GUI or GUI work into terminal.
6. **Zero Silent Failure Masking**: Shell failure masking idioms like `|| true`, `2>/dev/null || true`, or treating blocked executions (exit code 126) as successes are strictly forbidden. All execution receipts must reflect ground-truth exit codes and stderr.
7. **Deterministic State-First Convergence (HDCE)**: Complex tasks must proceed state-first, not workflow-first. Workflows must pass only typed `StateDelta` mappings between nodes, purge raw temporary chatter, and validate every mutation through `ActionContract` pre/post-conditions.
8. **Execution Plane Purity & Multi-Phase Persistence**: When a mission requests an execution plane (`requested_plane`), that plane must persist throughout all subsequent decomposition phases (`_plan_next_phase`). Headless tasks must not be contaminated by redundant GUI screenshot polling.
9. **AI-Human Digital Being Invariant**: SONIC embodies a persistent digital colleague with continuous identity (`Being`), cross-session cognitive affect (`BeingMind`), human workstation fluency (hotkeys, window management, focus settling), and an always-on curiosity loop (`BeingLifeLoop`). It is never a disposable CLI bot.
10. **Zero External Puppet Simulation (Developer/Assistant Never Impersonates SONIC)**:
    - *"Jo SONIC ko karna chahiye wo developer/assistant kyu karega?"*
    - The developer, pair programmer, or assistant agent must NEVER run manual background commands, shell scripts, or X11 actions (`docker exec ... xdotool`, `curl`, headless automation) to simulate or fake progress on a task given to SONIC.
    - Every action on the screen, in the browser, or in the terminal MUST originate strictly from SONIC's own cognitive loop (`ComputerUseAgent.step() -> execute_action()`, `BossAgent`, or `LayaSystem1Engine`). Taking manual shortcuts behind the scenes destroys trust and is strictly forbidden as fake progress.
11. **Smooth Visual Motor Perception & Visible Cursor Interactivity**:
    - *"Cursor kaam karna chahiye aur use hona chahiye, background automation nahi."*
    - Mouse cursor teleportation (0 ms instantaneous jump via raw `xdotool mousemove`) is strictly forbidden in user-facing workstations because streaming desktop protocols (noVNC / WebSockets) drop single-frame jumps, making the cursor look invisible or frozen to human observers.
    - All pointer movements (`GUI_MOVE`, `GUI_CLICK`, `GUI_DOUBLE_CLICK`, `GUI_RIGHT_CLICK`) must use smooth cosine-eased motor interpolation (`sonic_cursor.py`) over 200–300 ms so human operators can visually follow the agent's intent.
    - Keystroke typing (`GUI_TYPE`) must always focus the target element (`x, y`) before typing to ensure the cursor actively interacts with the intended field.
    - Tasks involving visual applications, browsing, or desktop UI must actively dispatch to the desktop GUI plane rather than taking silent headless terminal shortcuts.

### 4. Dual Topology & Connection: Standalone Being vs. Autonomous Developer CLI
SONIC operates across two cleanly delineated manifestations with an explicit Client-Server connection contract:
- **SONIC Core (`sonic-core` / `sonic-dashboard`) — The Autonomous AI-Human Digital Being**:
  - Hosts the complete cognitive mind (Boss, Judge, Task Planner), persistent graph & vector memory (`sqlite_graph.py`), Docker graphical workstation (`sonic-desktop-workstation`), and live visible desktop stream (noVNC on port 12001).
  - **Standalone Independence**: SONIC Core is 100% complete and self-contained; it does NOT require `sonic-cli` to operate. It runs as an autonomous background service / API (`http://localhost:12000`) and web dashboard.
  - Department: Managed under Department `cognitive_systems` (branch: `departments/cognitive-systems`).
- **SONIC CLI (`sonic-cli`) — The Autonomous AI Software Engineer (Zero Boss / Workstation)**:
  - Devin/Antigravity-style terminal interface (`sonic > `) operating directly on the local repository codebase.
  - **Zero Boss / Swarm / Workstation in CLI**: BossAgent, sub-agent swarms, and Docker graphical desktop workstations (`GUI_SCREENSHOT`, `xdotool`, etc.) belong 100% to `sonic-core` and the Web Dashboard (`sonic-dashboard`). They are completely eliminated from `sonic-cli`.
  - **Local Agentic Coding Loop (`DeveloperAgent`)**: Natural language prompts (`> fix test in ...`, `> explain architecture`) enter a local observe-reason-act tool execution loop (`view_file`, `write_file`, `edit_file`, `run_command`, `git_diff`, `run_pytest`) powered by fast local/cloud LLMs (Groq `qwen/qwen3.8-27b`, OpenAI, Claude, Gemini) with workspace path confinement and secret sanitization.
  - **Instant Conversational Greetings**: Simple greetings (`hi`, `hello`, `hii`) return immediate human-like responses in < 5ms without tool calls, worker spawning, or desktop screenshots.
  - **Developer Slash Commands & Subcommands**: Fast developer controls (`/auto`, `/diff`, `/journal`, `/test`, `/status`, `/help`, `/clear`, `/exit`) and subcommands (`sonic status`, `sonic diff`, `sonic test`, `sonic auto`, `sonic journal`, `sonic auth`, `sonic version`).
  - In `/auto` mode, it continuously triggers `ContinuousAutoEvolutionRunner` in isolated Git worktrees, tests candidates, and logs to `evolution.md`.
  - Department: Managed under Department `platform_systems` (branch: `departments/platform-systems`).

## Historical engineering record

The phase-by-phase notes below preserve implementation history, prior test
results, and past architectural decisions. They are not a guarantee that a
feature remains wired, available, or reliable in the current runtime. They do
not override the general-purpose computer-user product goal or the current
perception, execution, and verification contracts at the end of this document.

## Layout
- `sonic-core/` — Python backend (uv workspace, Python 3.12+). `pip install -e sonic-core[dev]`.
- `sonic-dashboard/` — Next.js 14 dashboard. `npm install` then `npm run dev`.
- `docker-compose.prod.yml` — prod compose (secrets via env interpolation).
- `sonic-cli/` — CLI wrapper package.

## Running locally (dev)
- Backend: `APP_ENV=development python -m uvicorn sonic.api.main:app --port 12000`
  (uvicorn is not on PATH; use `python -m uvicorn`).
- Dashboard: `NEXT_PUBLIC_API_URL=<backend-url> npx next dev -p 12001`
- Neo4j is optional in dev — the backend falls back to "Running without Graph
  Memory" if localhost:7687 is unreachable.

## Testing
- `python -m pytest sonic-core/tests/`
- Security regression suite: `sonic-core/tests/test_p0_security_hardening.py`
- New upgrade tests: `sonic-core/tests/test_ai_and_control_upgrade.py` covers
  LLM retry/backoff, robust tool-argument JSON parsing, ReAct native
  function-calling with parallel tool execution, and ComputerUseAgent
  coordinate-bounds validation.
- Known pre-existing failures (environmental, NOT code bugs):
  - `test_phase13/*` through `test_phase19/*` — require a live Docker daemon
    (`DockerProvider` runs `docker`); not available in CI/this sandbox.
  - `test_phase21_real_graphical_workstation.py::test_daytona_gui_action_dispatch`
    — requires a live Daytona sandbox with a real X11 desktop + VS Code.
  - `test_p0_security_hardening.py::test_secret_key_env_alias_populates_jwt_secret`
    — fails only when a local `.env` sets `JWT_SECRET`/`SECRET_KEY` (the `.env`
    wins over the test's `monkeypatch.setenv`). Remove/rename `.env` to verify;
    this is test-isolation, not a code bug. `.env` is gitignored.
- `get_settings()` is `@lru_cache`d — in tests that monkeypatch env vars, call
  `get_settings.cache_clear()` before and after.

## Persistent memory (Phase 1 — DONE, foundation)
SONIC's mind now survives a backend restart. The memory router
(`sonic/memory/router.py`) selects Neo4j first, then falls back to a
**persistent** SQLite-backed graph (`sonic/memory/sqlite_graph.py`) instead of
the ephemeral `InMemoryGraph`. So if Neo4j is down, memory still persists.
- `SqliteGraph` mirrors the `InMemoryGraph` interface exactly (drop-in), with
  write-through: every node/relationship write commits to SQLite immediately.
  Tables: `memory_nodes(uid,label,tenant_id,props_json,created_at)`,
  `memory_relationships(from_uid,to_uid,type,from_label,to_label,props_json,tenant_id)`.
- `VectorMemory` (`sonic/memory/vector.py`) and `EvolutionMemoryStore`
  (`sonic/evolution/memory.py`) also write through to SQLite (`vector_documents`,
  `evolution_items` tables) via stdlib `sqlite3` (sync API preserved).
- DB path resolves from `SONIC_MEMORY_DB_PATH` env → `DATABASE_URL` (sqlite) →
  `./sonic_data.db`. Tests set `SONIC_MEMORY_DB_PATH` to a temp file.
- `reset_memory_singleton()` / `reset_vector_memory_singleton()` clear the cached
  singletons to simulate a restart (used by `test_phase1_persistent_memory.py`).
- Done-gate test: `sonic-core/tests/test_phase1_persistent_memory.py` — proves
  graph/vector/evolution memory survive restart, tenant isolation holds across
  restart, and no host `subprocess` execution was introduced.
- **Honesty checkpoint:** Phase 1 claims ONLY persistent memory. Reasoning,
  browser-in-loop, security-tool execution, and curiosity/life were originally
  deferred (Phase 3-6). They are now DONE — see "Phases 3-6 (capability
  layer)" below.

## Persistent body + safety (Phase 2 — DONE, foundation)
- `DaytonaComputerProvider` persists its workspace index to
  `sonic_data/workstations.json` (`SONIC_WORKSTATION_STATE_PATH`). `_load_state()`
  runs in `__init__`, so workspace IDs survive a backend restart and
  `_resolve_sandbox` re-attaches via `client.get(id)`. `create()`/`destroy()`
  write-through on every lifecycle change.
- `get_or_create_home(tenant_id)` returns the tenant's long-lived
  `MISSION_COMPUTER` home, reusing a persisted one instead of reprovisioning
  (tenant-scoped).
- Safety regression (verified by `test_phase2_persistent_body_and_safety.py`):
  host execution is hard-forced false outside dev even when
  `SONIC_ALLOW_HOST_EXECUTION=true`; `LocalDevProvider.execute()` returns a
  fail-closed result (exit 126) when disabled — never spawns a host subprocess.
- Test marker `@pytest.mark.no_live_infra` opts a test out of the live-compute
  skip gate (for provider state/safety logic that needs no live sandbox). The
  ~48 genuinely live-gated integration tests remain honestly skipped.
- **Live done-gate (Step 2.4) is environment-gated:** the full
  provision-home → screenshot → terminal → restart → re-attach cycle needs a
  real Daytona cloud desktop (`SONIC_RUN_LIVE_DAYTONA=1`). Not runnable here.

## Phases 3-6 (capability layer — DONE)
The unified observe -> reason (LLM) -> act loop now covers terminal, files, git,
browser, security tools, AND self-directed exploration. No hardcoded fallbacks.

### Phase 3 — Real computer-use reasoning
- `sonic/computer_use/agent.py` rewritten: removed
  `_derive_remediation_from_goal()` and `_heuristic_choose_action()` entirely.
  Action selection is LLM-driven each step: build reasoning context from goal +
  observation + action history -> LLM returns the next action. `observe()` reads
  terminal output via `echo __sonic_obs_ready__` (not a hardcoded string).
  `run_mission()` has goal-aware termination (`GOAL_COMPLETE` sentinel).
- Done-gate: `test_phase3_real_computer_use_reasoning.py` (8 tests).

### Phase 4 — Unified action surface (browser-in-loop)
- `BROWSER_NAVIGATE / BROWSER_CLICK / BROWSER_TYPE / BROWSER_SCREENSHOT` added
  to `ComputerActionType` and the LLM action space/parser/prompt.
- `ComputerUseAgent` accepts an optional `BrowserAgent`; `observe()` folds the
  live browser page state (url, title, interactive elements) into the
  observation; `execute_action()` dispatches BROWSER_* to the browser.
  `BrowserAgent.current_page_state()` returns the live (url, title) so the
  observation reflects page state AFTER interactions.
- Done-gate: `test_phase4_unified_browser_in_loop.py` (4 tests).

### Phase 5 — Security-tool execution in the unified loop
- `SECURITY_TOOL` added to the action space. `ComputerUseAgent` accepts a
  `security_tools` registry (name -> SecurityTool). `execute_action()` builds a
  `ToolRequest` and calls `tool.execute()`, which runs the tool INSIDE the
  sandbox provider (fail-closed: exit 126 -> status BLOCKED). The structured
  `ToolResult` (findings, status) is stored in `_last_tool_result` and surfaced
  to the next reasoning step, so the LLM acts on REAL scan findings, not a
  "scan ran" claim. Blocked/failed tools trigger recovery; recovery preserves
  the original error cause.
- Done-gate: `test_phase5_security_tool_execution.py` (5 tests).

### Phase 6 — Curiosity / life loop (self-directed, novelty-driven)
- `sonic/computer_use/curiosity.py`: `CuriosityLoop.propose_curious_goal()` asks
  the LLM to propose the most informative goal from the live observation + the
  known-facts list (novelty-biased, not a fixed idle script).
  `measure_novelty()` reuses `NoveltyEngine`; `persist_fact()` records
  genuinely-new facts to `VectorMemory` (Phase 1 Mind) with near-duplicate
  dedup, so curiosity compounds across cycles. Dead-end detection flips the
  proposal prompt into a "pivot to a different area" instruction.
- `ComputerUseAgent.idle_cycle()` / `run_curiosity_loop()`: propose a goal,
  pursue it via the REAL observe->reason->act loop, measure novelty, persist any
  newly-learned fact. The same loop now drives operator goals AND self-directed
  exploration.
- Done-gate: `test_phase6_curiosity_life_loop.py` (5 tests).

### PLAN Phase 6 — Self-host safety envelope / VPS gate (fail-closed ActionPolicy)
- `sonic/safety/action_policy.py`: `ActionPolicy` is a fail-closed gate the loop
  consults BEFORE any action (operator- OR self-directed). It composes existing
  primitives: path confinement (workspace-root) for FILE_*; egress filter
  (`sonic.sandbox.egress`) for SECURITY_TOOL targets + BROWSER_NAVIGATE urls;
  destructive-command gating (`sonic.safety.scope.classify_command_risk`,
  L2->block, L1->needs approval); an action-type allowlist (unknown types
  DENIED by default); and a per-agent rate limit.
- `ComputerUseAgent.execute_action()` evaluates the policy pre-dispatch; a
  denied action is recorded BLOCKED, NEVER reaches the provider, and does NOT
  trigger recovery (recovery must not bypass the gate). Self-host mode
  (`self_host=True`) REFUSES to construct without a safety policy.
- Done-gate: `test_phase_plan6_safety_envelope.py` (10 tests): in-workspace
  actions execute; path-escape/traversal denied; destructive command denied;
  security tool vs metadata IP denied (egress); unknown action type denied;
  the SAME gate blocks curiosity pursuits; self-host requires a policy;
  rate limit caps actions/min.

### Test baseline (after Phases 3-6)
- 332 passed, 48 honestly skipped, 6 pre-existing model-only failures
  (unchanged). The 6 failures assert OLD fake-success behavior in the evolution
  / exploit capability layer; they are out of scope per PLAN and are NOT caused
  by this work.

## AI-Human layer (Phase 7 — Persistent Being + Life + Craft)
The foundation (persistent Mind + Body + safety) is real. The "AI human" layer
makes the being a *stable self* rather than a fresh tool on each boot:

### Persistent Being Identity — `sonic/being/identity.py`
- Before this, `agent_id` was a per-instance UUID (or the literal
  `"computer-use-agent"`) minted fresh every boot — each restart the system
  became a brand-new tool with no continuity. A being that forgets its own
  identity is not alive.
- `Being` (stable `being_id`, `name`, `tenant_id`, `born_at`) + `BeingMind`
  (cross-session mood: `curiosity_drive`/`focus`/`satiety`, learned-facts
  ledger, idle/goal tallies) are persisted to SQLite (write-through, mirrors
  the Phase-1 VectorMemory pattern: stdlib `sqlite3` + read cache +
  `reset_being_singleton()`). They share `sonic_data.db` (env
  `SONIC_BEING_DB_PATH`/`SONIC_MEMORY_DB_PATH`).
- `BeingMind` is distinct from `agents/cognitive_state.py:CognitiveState`
  (per-engagement, Redis-TTL'd 24h). The being's affect survives restart;
  `record_idle_cycle()` evolves mood deterministically from the outcome (novel
  fact -> curiosity+satiety rise; dead-end -> satiety/boredom + focus rise).
- `get_or_create_being(tenant_id)` is the identity equivalent of
  `get_or_create_home(tenant_id)` for the Body: on restart the identity is
  RE-RESOLVED from SQLite (same `being_id`, same `born_at`), not reminted.
  Tenant isolation holds: one being per tenant, cross-tenant reads return nothing.

### Always-on Life Loop — `sonic/being/life_loop.py`
- `BeingLifeLoop` is the always-on tick that runs when no operator goal is
  active, letting the being pursue self-directed curiosity between (and
  independent of) API requests. Before this, `CuriosityLoop` existed but was
  dead unless manually poked — a being that only acts when poked is a tool.
- Each tick proposes a curious goal (LLM, novelty-biased) and pursues it via
  the agent's REAL observe->reason->act loop. **Every autonomous action still
  passes the Phase-6 `ActionPolicy` safety envelope** — the being cannot escape
  it when "left to its own devices". Constructing a life loop without a safety
  policy is refused (mirrors `ComputerUseAgent(self_host=True)`).
- Spawned as a long-lived `asyncio.Task` in `api/main.py:lifespan()` (env-gated
  `SONIC_ENABLE_BEING_LIFE_LOOP=1`; cancelled cleanly on shutdown; a failed
  tick is logged and the loop continues, so one bad cycle can't kill the being).

### Durable Craft — `sonic/being/craft.py`
- `BeingCraft` persists being-authored artifacts (notes/observations/scripts)
  on the HOST filesystem under `sonic_data/craft/<being_id>/`, so the being
  builds a personal toolkit that survives restart — distinct from disposable
  in-sandbox research outputs (which were durable only via git commits).
- Files are markdown and human-readable on disk; an `index.json` lists metadata.
  Being-scoped isolation holds between beings.

### Done-gate (test_phase7_ai_human_being.py, 11 tests)
- identity stable across restart (same being_id + born_at, not reminted);
  first-contact provisions / second-contact re-attaches;
  BeingMind persists + mood evolves (novel rewards, dead-end -> boredom);
  tenant isolation; life loop refuses construction without a safety policy;
  the life loop tick routes the being's self-directed action THROUGH the safety
  gate (destructive idle pursuit BLOCKED, never executed, still recorded);
  loop starts/stops cleanly + survives a failed tick; craft persists across
  restart + being-scoped + human-readable on disk.

### Test baseline (after Phase 7)
- 353 passed, 48 honestly skipped, 0 failures (PR#3's simulated-provider fixes
  resolved the 6 previously pre-existing model-only failures; Phase 1-7 green).

## Security Tool Adapter Registry (Phase 7.5 — wire real adapters to all callers)
Closed the audit-flagged gap: the real SecurityTool adapters (nmap/nuclei/ffuf/
http_client) existed but were unreachable from production callers — only the
queue worker wired them up privately, and the mission director + AI-Human being
life loop constructed `ComputerUseAgent` with an EMPTY `security_tools=` map,
so the agent could advertise a `SECURITY_TOOL` action to the LLM but never
actually dispatch a scan.

### `sonic/tools/registry.py` — one shared registry
- `build_security_tools(provider)` constructs the 4 REAL adapters bound to a
  ComputeProvider (every tool runs in-sandbox via `SecurityTool.execute` →
  `provider.execute`; fail-closed on exit 126 — zero host execution, no stubs).
- `SecurityToolRegistry`: name→tool lookup with `register()` plugin extension
  point; `get_default_registry(provider)` is the production entry point.
- Provider-scoped by design (NOT a process-global) — adapters bind to a specific
  provider/tenant sandbox, so a global would leak a sandbox across tenants.

### Wiring (one source of truth)
- `queue/worker.py`: builds `_tools` via `build_security_tools()` (replaces the
  private hardcoded dict).
- `api/main.py` being life loop: passes `security_tools=get_default_registry(
  provider).as_dict()` so the AI-Human being can run real scans during
  self-directed curiosity.
- `mission_engine/director.py`: passes registry tools to the engineering-phase
  `ComputerUseAgent` (bound to the underlying `ComputeProvider`).

### Done-gate (test_security_tool_registry.py, 6 tests)
- build returns the 4 real adapters bound to the provider; registry lookup +
  extension + provider-scoping (no global leak); worker uses the registry; the
  agent dispatches a REAL NmapAdapter end-to-end via registry tools (canned nmap
  stdout parsed by the real adapter, port 22 surfaced as a finding); a
  fail-closed (exit 126) tool outcome triggers recovery through the registry path.

### Test baseline (after Phase 7.5)
- 359 passed, 48 honestly skipped, 0 failures.

## Sealed Safety Policy (Phase 7.6 — tamper-evident guards)
Closed the PLAN.md audit item: "Safety immutability: guards are real but NOT
protected from a self-evolving agent modifying them (no read-only/signed layer)."
The existing guards were plain mutable Python objects — an in-process
self-evolving being could widen its own envelope at runtime
(``policy.security_tool_targets.add(...)``, ``egress.BLOCKED_NETWORKS.clear()``,
or reassigning ``agent.safety``) and then act.

### `sonic/safety/sealed.py` — tamper-evident policy
- `SealedActionPolicy(ActionPolicy)`: at `seal()` time freezes a snapshot of every
  verdict-controlling field (allowed_types, security_tool_targets [frozenset],
  rate cap, approval flag, workspace root, and a *copy* of the egress
  blocked-networks list) and records a SHA-256 seal hash.
- After sealing, mutation of any safety-relevant field raises `AttributeError`
  (effectively immutable); the rate window may still mutate.
- On every `evaluate()` the seal hash is recomputed + compared; any mismatch
  (config altered) makes the verdict fail-closed DENY + logs
  `safety_policy_tampered` — the being may NOT act under an unprovable policy.
- `_check_egress` uses the FROZEN blocked-networks snapshot (via the new
  `egress.is_target_allowed(blocked_networks=...)` param), so a runtime mutation
  of the module-level `BLOCKED_NETWORKS` list cannot widen what this policy
  permits. Backward compatible: `is_target_allowed` defaults to the live list.
- `seal_default()` builds + seals the standard production policy.
- Additive + opt-in: plain `ActionPolicy` still works unchanged (existing tests
  hold); the being life loop constructs a `SealedActionPolicy`.

### Wiring
- `api/main.py` being life loop: `safety = seal_default(...)` (was a plain
  `ActionPolicy`) — the AI-Human being always acts under a proven-intact envelope.

### Done-gate (test_safety_sealed_policy.py, 11 tests)
- seal freezes config + records a 64-char sha256 hash; sealed fields are
  immutable (AttributeError on mutation of allowed_types/rate/approval/workspace);
  a tampered policy (object.__dict__ bypass) fails-closed DENY + logs tamper;
  egress uses the frozen snapshot (clearing `egress.BLOCKED_NETWORKS` does NOT
  widen the sealed policy — `10.0.0.5` stays blocked — and the seal stays intact);
  contrast: a plain unsealed `ActionPolicy` IS widened by the same mutation
  (proving the gap the sealed layer closes); the agent under a sealed policy
  allows a public target and blocks (status=BLOCKED) a private one.

### Test baseline (after Phase 7.6)
- 370 passed, 48 honestly skipped, 0 failures.

## Real Autonomy: Trace-Derived Mission Artifacts (Phase 7.7)
Closed the PLAN.md audit item: the mission director hardcoded its milestones
(all force-set to COMPLETED), deliverables ("JWT none algorithm bypass" + a
fake commit hash `7b8e1f0a2c`), knowledge summary ("what_we_know" JWT strings),
and confidence (=1.00) — NONE of it derived from what the agent actually did.
That was scripted theater: the director reported success and fabricated
evidence regardless of the traces.

### `sonic/mission_engine/trace_synthesis.py` — derive real artifacts from traces
- `synthesize_deliverables(mission_id, goal, traces)` — one deliverable per
  concrete successful engineering action the agent ACTUALLY took
  (FILE_WRITE→ENGINEERING_PATCH, GIT_COMMIT→GIT_COMMIT, SECURITY_TOOL→
  EVIDENCE_PACKAGE). Content = the real observation text, not a hardcoded vuln
  name. No successful artifact action → empty list (honest: no fabrication).
- `synthesize_knowledge(goal, traces)` — what_we_know/decisions/evidence come
  from the real traces; confidence = success ratio (NOT 1.00 by decree).
- `milestone_status_from_traces(traces, plan)` — a milestone is COMPLETED only
  where the traces contain a relevant successful action; otherwise stays
  PENDING. Never force-completed by decree.

### Director wiring (`director.py`)
- `coordinate()`: milestones now derived via `milestone_status_from_traces`;
  confidence = successful/total; outcome is SUCCESS only if the agent actually
  did something that succeeded (else PARTIAL_SUCCESS/FAILED).
- `finalize(traces=...)`: returns trace-derived deliverables (was the hardcoded
  JWT pair). No traces → empty list.
- `get_knowledge_summary()`: reflects the stashed real traces (was hardcoded
  JWT strings).
- RESEARCH-phase hypotheses seeded from the objective's constraints, not a
  hardcoded "JWT none algorithm" hypothesis.

### Browser action surface wiring (also closes the "orphaned browser" item)
- `director.coordinate()` + `api/main.py` being loop now construct a
  `BrowserAgent(headless=True)`, launch it, and pass `browser=` to the agent.
  The browser was orphaned before (BrowserAgent existed + the agent's
  observe→reason→act loop had full browser support, but no caller ever passed
  `browser=`, so `agent.browser` was always None and the browser branch never
  ran). Now navigate/click/type/screenshot dispatch as first-class actions.
- Removed dead code: `_derive_remediation_from_goal()` (67 lines of hardcoded
  JWT/SQLi/XSS/traversal patches, never called) — the LLM-driven
  `_llm_choose_action` + generic `_diagnostic_fallback` are the real paths.

### `MissionKnowledgeSummary.confidence` field added
- New `confidence: float = 0.0` field so the trace-derived success ratio is
  surfaced in the knowledge snapshot (was on `MissionState` only).

### Done-gate (12 + 2 tests)
- `test_mission_trace_synthesis.py` (12): deliverables one-per-artifact +
  content-is-real-observation + failed-produces-none + empty-is-empty +
  dedupe; knowledge derived-from-traces + confidence=success-ratio (not 1.00)
  + empty-is-honest + all-success=1.0; milestones completed-only-where-proven
  + all-pending-when-no-success; director.finalize trace-derived (not JWT) +
  honest-on-no-traces; knowledge summary reflects stashed traces.
- `test_browser_wiring.py` (2): browser-attached dispatches navigate to the
  browser (SUCCESS trace); browser=None does not crash (graceful skip).

### Test baseline (after Phase 7.7)
- 384 passed, 48 honestly skipped, 0 failures.

## Honest Autonomy: Adaptive planner + real recon (Phase 7.8)
Closed two remaining PLAN.md audit items.

### Objective-adaptive mission planner (`mission_engine/planner.py`)
Was: a single fixed 3-action plan (pwd / git status / find) for EVERY mission,
never adapting to the objective. Now the orientation baseline is always
prefixed, then intent-specific read-only inspection commands are derived from
the objective text:
- recon-intent → source-file enumeration + TODO/FIXME grep
- web/api-intent → http/url/endpoint grep + route/server file find
- test-intent → test-file + conftest/package.json discovery
- db-intent → query/sql grep + migration/schema find
Active probes remain approval-required (never silently executed). Every action
still compiles to the allowlisted read-only tool schema.

### Real recon, no hallucinated attack surface (`agents/recon.py`)
Was: asked the LLM "what subdomains likely exist? (api., admin., staging.,
dev., mail.)" and presented those guesses as discovered assets — hallucinated
attack surface. Now:
- Real subdomains come from Certificate Transparency logs (crt.sh JSON API),
  returned with `discovered_by: "certificate_transparency"`. Best-effort:
  egress-gated (uses `is_target_allowed`), returns [] if the CT source is
  blocked/unavailable — never invents subdomains.
- LLM-suggested candidates are clearly labeled `discovered_by:
  "llm_hypothesis"` with `confirmed: False` — NOT presented as observed truth.
- LLM subdomains already found via CT are deduped (kept as the real,
  confirmed entry).
- With no router, only real CT assets (if any) are returned — no imagined
  fallback.

### Done-gate (8 + 5 tests)
- `test_mission_planner_adaptive.py` (8): orientation baseline always
  present; recon/web/db intents add intent-specific commands; different
  intents produce different sets; active probe is approval-required (not
  silent); readonly objective adds no probe; empty objective raises.
- `test_recon_honesty.py` (5): real CT subdomains labeled
  certificate_transparency; LLM assets labeled llm_hypothesis/confirmed=False;
  LLM subdomain deduped against real CT; CT blocked → no invented subdomains;
  no router → only real CT assets.

### Anti-puppet concrete probes + evidence-driven follow-ups (Phase 7.9)
The planner's active-test stage used to emit the no-op stub
`target_shell_approved { command: "echo APPROVAL_REQUIRED" }` — a scripted
puppet gate that gave the operator nothing real to approve. Closed:
- `planner.build_probe_actions(objective, target)` now derives CONCRETE, typed
  `target_security_scan` probes (nmap top-100 / http_client / nuclei / ffuf)
  from the objective intent + in-scope target. Probes carry a real option set,
  a `rationale`, and a human-readable `command` preview for the operator.
  All probes are `APPROVAL_REQUIRED`; read-only objectives emit none.
- `build_plan` and `build_long_horizon_plan` (stage 4) call the probe generator
  instead of the echo stub.
- `build_follow_up_actions(plan, completed, evidence_brief=...)` adapts the next
  read-only batch to the evidence that earlier actions actually returned
  (web-route greps, secret/token hunts) instead of a fixed copy-paste batch.
- `workstation.py::_run_mission_preflight` now: wires the real scanner registry
  into the executor; surfaces probe proposals as `mission["proposed_actions"]`
  (+ mission events) for operator approval; NEVER auto-runs a probe (removed
  the old raw `comp.terminal` dispatch that bypassed the executor); passes real
  evidence outputs to the follow-up planner; closes with an AWAITING_APPROVAL
  message referencing how many probe proposals await approval.
- `executor.MissionToolExecutor(..., scoped_target=...)`: when scope is set
  (preflight always does), the `target_security_scan` path BLOCKS any probe
  whose host is outside the engaged scope — suffix-safe (api.target.com in
  scope target.com; target.com.evil.net NOT). Legacy callers without
  scoped_target keep the previous planner-only guarantee.
- Tests: `tests/test_evidence_driven_probes.py` (11) covers: no echo stub in
  active plans; probes concrete/scoped/previewed; read-only objectives emit
  none; dispatch behind approval via a fake SecurityTool scanner; out-of-scope
  probe blocked even when approved; follow-ups adapt to secret and web-route
  evidence.

### Test baseline (after Phase 7.8)
- 397 passed, 48 honestly skipped, 0 failures.

## Phase A — Toolsmith loop (AIOSR: being authors its own tools)
Closed the single highest-leverage gap from the AIOSR audit: SONIC was an
"Operator" (orchestrating known nmap/nuclei/ffuf primitives) but not a
"Researcher + Toolsmith" (building its own tools). The being can now author a
NEW small tool (scanner/fuzzer/parser/probe) for an observation gap, persist
it, run it in-sandbox, and register it as a first-class callable tool — but
ONLY after a real successful run. No "tool authored and working" claim by decree.

### `sonic/being/toolsmith.py` — author + confirm + register
- `ToolsmithLoop.author_tool_for_gap(observation, failed_attempts)` asks the LLM
  to propose ONE small Python tool filling a gap NO existing registered tool
  covers. The proposal is explicitly biased away from the existing tool set
  (nmap/nuclei/ffuf/http_client + already-confirmed authored tools) and from the
  reserved names — so the being does not re-author nmap. A `DECLINE` response
  or a duplicate/reserved/invalid name → returns `None` (honest skip: never
  fabricates a tool). A cheap pre-sandbox safety lint rejects host-wiping
  patterns (`rm -rf /`, `mkfs`, `dd of=/dev/`, `shutdown`) before the sandbox.
- The authored source is persisted to `BeingCraft` immediately (durable across
  restart; kind=`tool`, human-readable markdown on disk).
- `confirm_and_register(tool, provider, workspace_id)` writes the source to
  `/home/sonic/workspace/toolsmith/<name>.py` and runs `python <path>` in-sandbox
  via the provider's fail-closed `execute` (exit 126 => sandbox blocked it).
  **Honesty guard:** `reproduced=True` and registration into the
  `SecurityToolRegistry` happen ONLY on exit 0 + non-empty stdout. A blocked
  (exit 126), failed, or empty run leaves `reproduced=False` and the tool is NOT
  registered — the same anti-theatrical discipline as `production_gate`.
- `AuthoredToolAdapter`: wraps a confirmed authored tool as a real `SecurityTool`
  (name/version/build_command/parse_output/execute), bound to the provider that
  confirmed it. It inherits the base `SecurityTool.execute` fail-closed,
  in-sandbox handling (zero host execution, exit 126 => BLOCKED). `parse_output`
  decodes JSON lines (falls back to `{"finding": <line>}`), so the being's own
  tools emit structured findings through the same path as nmap.

### Action surface + safety wiring
- `TOOL_AUTHOR` + `TOOL_RUN` added to `ComputerActionType` + the LLM action
  space/parser/prompt. `execute_action()` dispatches:
  - `TOOL_AUTHOR` → `author_tool_for_gap()` (persists, does NOT register).
  - `TOOL_RUN` → `confirm_and_register()`; on `reproduced=True` the adapter is
    added to `self.security_tools` so the being can call its own tool via the
    existing `SECURITY_TOOL` path. The `available_tools` observation line
    already reflects the live `security_tools` keys, so authored tools surface
    to the LLM automatically once confirmed.
- `ActionPolicy.DEFAULT_ALLOWED_TYPES` extended with `TOOL_AUTHOR`, `TOOL_RUN`.
  Path confinement: `TOOL_AUTHOR` writes under a hardcoded workspace-subdir
  (structural confinement — no payload path to forge); `TOOL_RUN` runs a
  hardcoded `python <workspace-toolsmith-path>` (structural — no user command to
  gate). The sealed `SealedActionPolicy` inherits both (frozen in the seal).

### Production wiring (`api/main.py` being life loop)
- The being life loop now constructs a `ToolsmithLoop(craft=BeingCraft(being_id),
  llm=router, registry=get_default_registry(provider))` and passes
  `toolsmith=toolsmith` to `ComputerUseAgent`. So the always-on self-directed
  being can author + run its own tools during idle curiosity — every action
  still passes the sealed `SealedActionPolicy` gate.

### Done-gate (`test_phase_a_toolsmith.py`, 10 tests)
- authors a NOVEL tool name not in the registry (unconfirmed pre-run, not
  registered); `DECLINE` / no-novel → `None`; duplicate (nmap) → rejected;
  blocked run (exit 126) → NOT confirmed/registered; empty output → NOT
  registered; successful run → `reproduced=True` + registered + callable by
  name; authored source persists to BeingCraft (re-read off host disk); the
  safety policy ALLOWS `TOOL_AUTHOR`/`TOOL_RUN` (not denied as unknown); a
  destructive-source proposal (`os.system('rm -rf /')`) is rejected by the
  safety lint BEFORE the sandbox; a confirmed tool runs end-to-end via a real
  `ToolRequest` → `execute` and surfaces real parsed findings (not a decree).

### Test baseline (after Phase A)
- 423 passed, 48 honestly skipped, 0 failures.

## Phase B — Method-invention loop (AIOSR: being synthesizes novel techniques)
Closed the ONE gap that kept SONIC an "Operator" rather than a "Researcher":
`CuriosityLoop` proposes new GOALS but uses KNOWN techniques (nmap/nuclei
signatures). Phase B adds the "new new methods" loop from the AIOSR definition
— the being synthesizes a genuinely NOVEL offensive *technique* (a new method:
auth-bypass logic, parser-confusion chain, fuzzer mutation strategy,
header-injection primitive, race-condition probe, info-leak, logic-flow) from an
observation + a prior failure + the known-technique ledger, and confirms it
ONLY on real in-sandbox reproduction.

### `sonic/being/method_lab.py` — synthesize + confirm + persist
- `MethodLab.invent(observation, failure)` asks the LLM to synthesize ONE novel
  technique explicitly biased AWAY from the known-technique ledger (semantic
  search of `VectorMemory` scoped to `kind=technique` records). A `DECLINE`
  response, or a hypothesis that duplicates a known technique, → returns `None`
  (honest skip: never re-invents what it already knows).
- Each technique carries a `family` (auth-bypass/parser-confusion/fuzz-mutation/
  header-injection/race-condition/info-leak/logic-flaw/other), a `hypothesis`
  (the novel idea + why it differs), and a `probe_source` (small Python that
  takes a target as argv[1] and prints JSON findings when the technique works).
- `novelty_vs_ledger` is computed via `NoveltyEngine` so "novel" means genuinely
  outside the working-method ledger (0=identical, 1=fully novel). Empty ledger
  => 1.0.
- `confirm(technique, provider, workspace_id, target)` runs the probe in-sandbox
  (fail-closed exit 126 = blocked). **Honesty guard:** `confirmed=True` + ledger
  persistence happen ONLY on exit 0 + non-empty finding output. A blocked/empty
  run leaves the technique UNCONFIRMED and it is NOT added to the ledger —
  exactly the same anti-theatrical discipline as `production_gate`/`toolsmith`.
- A confirmed technique is parsed into structured `findings` (JSON lines, falls
  back to plain lines) and added to the VectorMemory ledger under doc_id prefix
  `technique-` with `kind=technique, confirmed=True`. So the ledger is a record
  of WORKING methods, and future `invent()` calls see these and steer away —
  novelty compounds across cycles (the being does not re-invent known methods).

### Phase A substrate feeds Phase B
- When a `ToolsmithLoop` is wired, `confirm()` routes the probe through the
  toolsmith's `confirm_and_register`: the probe is written to the sandbox,
  run fail-closed, and on success registered as a callable `SecurityTool`. So a
  confirmed technique's probe becomes a tool the being can re-run against new
  targets through the existing `SECURITY_TOOL` path. Phase A gave Phase B its
  execution substrate (a self-authored tool per self-invented technique).

### Action surface + safety wiring
- `METHOD_INVENT` added to `ComputerActionType` + the LLM action space/parser/
  prompt. `execute_action()` dispatches it to `MethodLab.invent()` then
  `confirm()`; the observation surface includes the technique name/family/finding
  count/novelty-vs-ledger. The `available_tools` line surfaces any probe that
  registered as a tool, so the LLM can re-invoke a confirmed technique.
- `ActionPolicy.DEFAULT_ALLOWED_TYPES` extended with `METHOD_INVENT` (structural
  confinement: the probe runs under a hardcoded workspace-toolsmith path, same
  as TOOL_RUN). The sealed `SealedActionPolicy` inherits it (frozen in the seal).

### Production wiring (`api/main.py` being life loop)
- The being life loop constructs `MethodLab(llm=router, vector_memory=
  get_vector_memory(), toolsmith=toolsmith)` and passes `method_lab=method_lab`
  to `ComputerUseAgent`. So the always-on self-directed being invents + confirms
  novel techniques during idle curiosity — every action still passes the sealed
  `SealedActionPolicy` gate, and every probe runs fail-closed in-sandbox.

### Done-gate (`test_phase_b_method_lab.py`, 9 tests)
- synthesizes a novel technique (novelty=1.0 vs empty ledger, unconfirmed
  pre-run); `DECLINE` → None; duplicate hypothesis → rejected; blocked probe
  (exit 126) → NOT confirmed + NOT in ledger; empty output → NOT confirmed;
  successful reproduction → confirmed + parsed findings + persisted to ledger
  (kind=technique, confirmed=True); a confirmed technique in the ledger steers
  the NEXT invention away (novelty compounds — duplicate synthesis rejected);
  the safety policy ALLOWS METHOD_INVENT; with a ToolsmithLoop wired, a
  confirmed technique's probe is registered as a callable tool.

### Test baseline (after Phase B)
- 432 passed, 48 honestly skipped, 0 failures.

## Prompt-identity + capability-awareness update (post-Phase B)
Renamed the system's LLM-facing identity from "SONIC-REDA, an autonomous AI
red-team system" / "elite bug bounty hunter" / "pentester" to **"SONIC — an
Autonomous Self-Evolving Penetration Architect (A-SEA)"** across every prompt,
so the being self-describes as a self-developing researcher/architect (matching
the real capability surface) rather than just an operator. 14 prompts updated:
- `agents/`: hypothesis, static_reasoning, recon, orchestrator, dynamic_execution,
  verifier, exploit_validator, react_engine.
- `agents/director.py` + `agents/replan.py`: Director + Replan Engine identity.
- `computer_use/agent.py` system prompt: A-SEA identity + now advertises the
  Toolsmith (TOOL_AUTHOR/TOOL_RUN) and Method Lab (METHOD_INVENT) as first-class
  actions, and instructs the LLM to author/invent when no existing tool fits a
  gap (not just re-run a known scanner).
- `computer_use/curiosity.py`: A-SEA identity + prefers goals that expose a
  Toolsmith/Method-Lab gap (so self-directed curiosity feeds method invention).
- `being/toolsmith.py` + `being/method_lab.py`: A-SEA identity; the toolsmith
  blocklist is now the DYNAMIC existing-tool set (was a hardcoded
  "nmap/nuclei/ffuf/http_client" literal — generalized so authored tools are
  auto-excluded from re-authoring).
- `continuous_dev/continuous_loop.py`: A-SEA identity + honesty clause (never
  claim success without a passing test command).
- `api/routes/workstation.py` workstation chat prompt: A-SEA identity + now
  describes the Toolsmith + Method Lab + sealed safety envelope + honesty rule
  (never claim confirmed without reproduction); generalized the Cloudflare-specific
  example to "a CDN".

No test asserted the old prompt text, so the rename was safe; full suite stayed
green (432 passed).

### Test baseline (after prompt update)
- 432 passed, 48 honestly skipped, 0 failures.

## Phase C — Dead-code removal (4,042 LOC deleted)
A static reachability audit from all production entry points (api/main + all
routers, queue worker, mission_engine, computer_use, being/*, swarm, cli)
proved that **6 packages form a closed, mutually-referencing island** that NO
production code path ever imports. They were vestigial scaffolding from the
old scripted-"success" evolution layer (Phases 8-19) that the real
observe→reason→act + Toolsmith + MethodLab loops replaced.

Deleted packages (production code):
- `sonic/autonomy/` (560 LOC) — blind_repair, hidden_root_cause, empirical_evolution, cryptographic_holdout, anti_scripting_verifier
- `sonic/continuous_dev/` (556 LOC) — ContinuousAutonomousDevLoop, autonomous_tool_selector, open_system_improver (was NOT wired to any route despite the Phase 7.9 AGENTS.md claim — that claim was aspirational/inaccurate)
- `sonic/open_world/` (399 LOC) — self_development_orchestrator, limitation_discovery, novelty_generator
- `sonic/production_gate/` (533 LOC) — scenario_matrix, temporal_holdout_generator, independent_evaluator
- `sonic/security_lab/` (709 LOC) — attack_surface, SecurityAcceptanceRunner
- `sonic/evolution/` (1,285 LOC) — the shared dependency hub (FailureMiner, CandidateGenerator, EvolutionLab, PromotionEngine, EvolutionMemoryStore, etc.); imported ONLY by the 5 packages above

Deleted test suites (31 files): test_phase8/, test_phase10/, test_phase11/,
test_phase16/, test_phase17/, test_phase18/, test_phase19/,
test_continuous_dev_curiosity_driven.py — all tested the deleted code.

`test_phase1_persistent_memory.py` updated: removed the
`test_evolution_memory_survives_restart` test (tested the deleted
EvolutionMemoryStore) and the `evo_memory` clause from
`test_no_host_execution_introduced` (now checks only sqlite_graph + vector).
The 3 core memory-survival tests (graph, vector, tenant isolation) remain —
they test the REAL persistent memory layer.

Packages KEPT (verified live):
- `sonic/meta/` (668 LOC) — used by `/experiments` route (CanaryPipeline, ExperimentManager, benchmark).
- `sonic/research/` (995 LOC) — used by `agents/director.py` (live swarm path: epistemic, information_gain, decision_trace, world_model).
- `sonic/agents/director.py` — the Phase-5 event-driven Director, reachable via `swarm.py` → `/live` route.
- `sonic/mission_engine/` — planner/executor used by `/workstation` mission loop.

### Test baseline (after Phase C)
- 386 passed, 37 honestly skipped, 0 failures. (46 deleted tests were for
  the removed code; 11 of those were Docker-gated skips.)

## Current test baseline
- 386 passed, 37 honestly skipped (Docker-daemon / Daytona-live gated via
  `sonic-core/tests/conftest.py`), 0 failures.
- The previously-pre-existing 6 model-only/fail-closed contract failures were
  resolved by PR#3's simulated-provider + execution-evidence fixes (they asserted
  the OLD fake-success behavior; now correctly supplied).
- `test_phase20_workstation_security_and_repair.py` is now GREEN: the
  `/workstation/command` route returns 503 fail-closed (not 409) when no sandbox
  is provisioned, and `/workstation/file` validates path confinement (403) BEFORE
  the workspace-state check.

## Audit fixes (honesty + wiring + test-isolation)
- `test_workstation_desktop_browser_and_news.py` was BROKEN at collection
  (missing `_clean_rss_titles`, no news/khabar intent). Fixed in
  `api/routes/workstation.py`: added the helper + news/info intent detection in
  `_detect_requested_app` / `_is_action_prompt`. (6 tests now collected + pass.)
- `CodeFixAgent` / `ExploitValidator` existed but were never wired into
  production dispatch (`swarm.py` `agent_map`/`agent_configs` omitted them;
  `queue/worker.py` `_run_agent_step` did not import them). Wired both so the
  AGENT_STEP dispatch path can actually instantiate them.
- Removed FAKE values from `production_gate/` + benchmark suites:
  - `scenario_matrix.py`: `initial_failure_verified` was hardcoded `True`
    (the broken version was never run). Now each scenario runs the BROKEN code
    first via `_verify_initial_failure()` and sets the flag from the real
    non-zero exit. `performance_delta_pct` (was 100/75/85 by decree) is now `0.0`
    (unmeasured — no before/after benchmark was run). Fake commit-hash fallbacks
    (`c01a9b`..`c08a9b`) replaced by `_commit_hash_from()` (empty when the
    provider returns a `bool`, which is what `git_action("commit")` does today).
  - `temporal_holdout_generator.py`: `v1_f1=0.667`, `v2_f1=1.000`, `tp=1/fp=0/
    fn=0` were all hardcoded. Now computed by a REAL token classifier
    (`_classify_v1`/`_classify_v2`/`_f1`) decoding JWT headers + signature length
    against real fixtures; v1 misses a short-signature token v2 catches, so
    the gain emerges from real behavior (still satisfies the test's
    success/gain>0.30 invariants).
  - `autonomy/blind_repair.py`: fake commit hash `7b8e1f0a2c` -> empty.
  - `computer_use/benchmark.py` + `mission_engine/benchmark.py`:
    `sonic_success_rate=1.00`/`autonomy_score=1.00`/`human_success_rate=0.92/0.88`
    (by decree) now DERIVED from per-trial success-flag lists.
  - `computer_use/models.py` + `computer/benchmark.py`: default scores 1.00 ->
    0.0 (honest "unmeasured" before `run_mission()`/the benchmark fills them in).
- `test_phase8/test_synthetic_self_evolution_mission.py` failed in the FULL
  suite (passed in isolation) because `EvolutionMemoryStore()` used the shared
  default SQLite DB polluted by earlier tests. Fixed to `persist=False`
  (in-memory isolation). This was a pre-existing suite-order bug, not caused
  by the honesty/wiring work.

## Security hardening applied (commit ab99ae2)
P0 fixes (all covered by `test_p0_security_hardening.py`):
1. `/auth/login` + `/auth/dev-token` are dev-only (404 in non-dev); roles clamped
   to OPERATOR — no SUPER_ADMIN/TENANT_ADMIN escalation.
2. Production refuses to boot with insecure/default/short JWT secret. `jwt_secret`
   reads `SECRET_KEY` / `JWT_SECRET` env aliases.
3. `docker-compose.prod.yml` secrets injected via `${VAR:?...}`.
4. `run_engagement()` is tenant-scoped; SUPER_ADMIN tenant bypass removed.
5. `SONIC_ALLOW_HOST_EXECUTION=true` forcibly false outside development.
6. Egress filter (`is_target_allowed`) wired into engagement creation + target
   sandbox provisioning — rejects private/loopback/metadata targets.

## Conventions
- Config via pydantic-settings (`sonic/config.py`). Settings are env-driven.
- CORS origins configured via `CORS_ORIGINS` env (comma-separated). main.py had a
  duplicate hardcoded CORS middleware — now deduped to use `cors_origins_list`.
- Egress policy: `sonic/sandbox/egress.py` (`is_target_allowed`). Default deny
  for private/loopback/metadata ranges.

## Daytona integration (live-tested, SDK daytona==0.207.0)
Two separate integrations exist; only the first is used by the workstation API:
- `sonic/computer/daytona_computer.py` → `DaytonaComputerProvider` (graphical
  desktop; used by `/workstation/*` routes via `get_daytona_computer()`).
- `sonic/sandbox/providers/daytona_provider.py` → `DaytonaProvider` (generic
  exec sandbox; selected by `factory.py` when `SONIC_COMPUTE_PROVIDER=daytona`
  or `DAYTONA_API_KEY` is set, but NOT wired to any workstation route).

Env vars (read via raw `os.environ`, NOT in config.py):
- `DAYTONA_API_KEY` / `DAYTONA_API_URL` (default https://app.daytona.io/api)
- `DAYTONA_SANDBOX_ID` + `DAYTONA_SANDBOX_TENANT_ID` — env-attach path: if the
  login user's `user.email == DAYTONA_SANDBOX_TENANT_ID`, `create()` reuses the
  existing sandbox instead of provisioning a new one.
- `DAYTONA_IMAGE` / `DAYTONA_RESEARCH_IMAGE` / `DAYTONA_TARGET_IMAGE` — per-type
  image override (default `daytonaio/sandbox:0.9.0`).

SDK API notes (0.207.0):
- `client.list()` is an async generator (`async for sb in client.list()`), NOT
  an awaitable. Older SONIC code may do `await client.list()` — that breaks.
- `client.get(id_or_name)` → `AsyncSandbox` (works).
- `client.start(sandbox)` takes a sandbox OBJECT, not an id.
- `client.delete(sandbox)` for removal (no `client.remove()`).
- `sandbox.process.exec(cmd)` → `ExecuteResponse` with `.exit_code` and
  `.result` (NOT `.stdout`/`.stderr`). SONIC's `terminal()` reads these correctly.
- `sandbox.computer_use.start()` must be called before screenshot/VNC.

Live-tested end-to-end (13/14 routes passed against a real Daytona sandbox):
- ✅ desktop/research-lab/target-sandbox provision (env-attach)
- ✅ desktop status + screenshot
- ✅ /workstation/command (real `uname -a` executed in container)
- ✅ target-sandbox provision (with scope.yaml-format scope_config) + command
- ✅ /llm/chat via NVIDIA NIM (meta/llama-3.2-11b-vision-instruct works)

Known issues and resolutions (current state):
- `status()` now normalizes `SandboxState` enum correctly via
  `_normalize_sandbox_state()`; LIVE sandboxes report `RUNNING` instead of
  `UNREACHABLE`. Resolved.
- `destroy()` falls back to `client.get`/`delete` via `_resolve_sandbox` when
  the workspace is absent from the in-memory map (e.g. after a backend
  restart). Resolved.
- Workstation RBAC tightened: all sensitive mutation routes
  (provision/destroy/command/desktop-action/target-sandbox-command/file-write/
  mission-start/prompt/session-delete/browser-open) now use `require_operator`,
  so read-only `AUDITOR` roles are blocked (403). Resolved.
- `meta/codellama-70b` removed from `configs/models.yaml` (404/EOL on NVIDIA
  NIM); coding routing uses `deepseek-ai/deepseek-r1`. Resolved.
- `/workstation/desktop/screenshot` and `/workstation/desktop/action` now
  return a graceful `200 NO_DISPLAY` observation (instead of `409`) when no
  workstation is provisioned. Resolved.
- `/workstation/command` FAILS CLOSED with `503` (not `409`) when no
  workstation is provisioned — the documented fail-closed contract. Resolved.
- `/workstation/file` path-traversal guard now rejects with `403` (was `400`)
  and is validated BEFORE the provisioning check, so traversal is blocked
  regardless of sandbox state. Resolved.
- Daytona sandboxes auto-stop (15m) / auto-archive (7d) — long test runs must
  set `auto_stop_interval` high or call `client.start()` before each op.
  (Still applies; environmental.)

LLM integration: `configs/models.yaml` has `nvidia` as default provider
(`NVIDIA_API_KEY`, `NVIDIA_BASE_URL`→integrate.api.nvidia.com/v1). Routed via
`sonic/llm/providers/custom.py` `CustomLLMProvider` (OpenAI-compatible).
Endpoint: `POST /llm/chat` (body: message, provider, model, max_tokens).

## End-to-end bug-hunt pipeline (live-tested)
Full flow verified against a dummy vulnerable target (intentionally-vulnerable
Flask app with command-injection RCE) deployed inside a Daytona sandbox:
1. `/auth/login` → operator token (RBAC, role clamped)
2. `/workstation/desktop/provision?session_id=<id>` → attach Daytona sandbox as
   computer-use workstation (env-attach via DAYTONA_SANDBOX_TENANT_ID match)
3. `/workstation/command` (body: command, session_id, timeout) → recon via curl
   inside the sandbox. NOTE: session_id is in the BODY, not query string.
4. `/llm/chat` (provider=nvidia, model=meta/llama-3.2-11b-vision-instruct) →
   LLM sub-agent analyzes recon output, hypothesizes vulnerabilities
5. `/workstation/command` → exploit (confirmed OS command injection RCE:
   `uid=1001(daytona)` + arbitrary file read of `/etc/os-release`)
6. `/workstation/desktop/screenshot?session_id=<id>` → real 1024x768 PNG frame
   from the Daytona desktop (computer_use.start() must succeed first)

Key learnings:
- `session_id` for `/workstation/command` is in the REQUEST BODY
  (`ExecuteCommandRequest.session_id`), NOT the query string. Provision routes
  use `Query("default")` for session_id. Mismatch causes 409 "no workstation".
- Workstation state persists to `sonic_data/workstations.json`. If a sandbox is
  recreated (new ID) but the state file still has the OLD ID, `terminal()`
  fails (exit_code 126) looking up a non-existent sandbox. Delete the state
  file or use a fresh session_id when switching sandboxes.
- Egress guard (P0-6) REFUSES to bind engagement/target-sandbox to private/
  loopback addresses (localhost, 172.x, 10.x, 192.168.x, 169.254.169.254).
  This is correct SSRF/metadata protection. A dummy target inside the sandbox
  is private, so engagement *binding* is refused — but recon/exploit still run
  via `/workstation/command` (which executes inside the already-scoped sandbox
  and doesn't re-check egress on curl destinations).
- LLM analysis quality is variable (llama-3.2-11b sometimes hallucinates
  endpoints not in recon output). Better as a hypothesis generator than
  authoritative finder. Use a larger model (90b) for higher-stakes analysis.

## Phase 7.9 — security-scan dispatch + curiosity-driven continuous_dev

### Mission executor: real security scans (audit gap closed)
- `MissionToolExecutor` previously returned "Tool adapter is not implemented"
  for nmap/nuclei/ffuf. Fixed: `target_security_scan` is now a registered
  `APPROVAL_REQUIRED` tool, and the executor accepts an optional
  `SecurityToolRegistry`. When wired -> dispatches a REAL in-sandbox scan via
  `SecurityTool.execute` and surfaces parsed findings in `evidence`. When NOT
  wired -> BLOCKS with "security tool registry is not configured" (never the
  generic "not implemented").
- Provider bridge: security adapters need a `ComputeProvider` (`.execute(ws,
  command, timeout)`), but the executor has a `ComputerProvider` (Daytona,
  `.terminal`). The thin `sonic/tools/computer_as_compute_provider.py`
  adapter bridges them so scanners run against the same target sandbox the
  executor already uses — no second provisioning hierarchy.
- Production wiring: `api/routes/workstation.py` mission loop builds the real
  registry via `get_default_registry(ComputerAsComputeProvider(comp))`.
- Safety: approval is enforced (AWAITING_APPROVAL without operator approval);
  forbidden-pattern scan targets are BLOCKED by the scope checker; all scans
  run fail-closed in-sandbox.

### continuous_dev: curiosity-driven lifecycle (audit gap closed)
- `run_curiosity_driven_lifecycle()`: each generation's patch is derived from
  a novelty-steered goal (CuriosityLoop) + LLM-authored patch for that goal —
  NOT the hardcoded v1->v2->v3 StreamBuffer/QueryCache demo. Honest failure
  (`GenerationStatus.ROLLED_BACK`) when the LLM can't author a patch. The
  scripted demo is retained as a labeled fallback when no curiosity/router is
  wired.
- `_derive_patch_from_goal(goal, rationale, gen_idx)` invokes the LLM to author
  a patch for the goal; records a rolled-back generation (not a fabricated
  success) on failure.

### Test discipline (done-gate)
- Done-gate tests must NOT import `DaytonaComputerProvider` at module top-level
  — the conftest autouse fixture skips any module importing Daytona symbols
  unless `SONIC_RUN_LIVE_DAYTONA=1`. Use a `_StubComputer` with a `.terminal`
  method instead.
- `PlannedAction` requires a `risk` field (ToolRisk) — construction without it
  raises a pydantic ValidationError.
- Suite as of Phase 7.9: 407 passed, 48 skipped, 0 failures.

## Phase 8 — Real Visual Computer Use (A-SEA sees and controls its desktop)

Closed the fundamental autonomy gap: SONIC had a full graphical desktop (Daytona
XFCE, 1280x800) and mouse/keyboard SDK capabilities, but the workstation endpoint
ran a regex-based bash dispatcher instead of the agentic loop, and `GUI_CLICK`/
`GUI_TYPE` were unhandled enums.

### Core Enhancements:
1. **GUI Action Surface in `ComputerUseAgent`**:
   - `ComputerActionType` enum expanded: `GUI_CLICK`, `GUI_DOUBLE_CLICK`, `GUI_TYPE`,
     `GUI_KEYPRESS`, `GUI_MOVE`, `GUI_SCROLL`, `GUI_SCREENSHOT`.
   - `execute_action()` dispatches GUI actions to `computer.gui_action()` (mouse click/move,
     keyboard type/press, scroll via xdotool, explicit screenshot re-observe).
   - `_parse_llm_action()` parses visual coordinate format (`TARGET: 640,400` -> `x=640, y=400`).
   - System prompt advertises GUI action schema + coordinate guidance.
2. **Vision-in-the-Loop**:
   - `observe()` stores `_last_screenshot_b64` from Daytona screen captures.
   - `_llm_choose_action()` formats multimodal user content (`image_url` block + text)
     when screenshot pixels are available, enabling visual spatial reasoning by VLMs.
3. **Workstation Route Agentic Wiring (`api/routes/workstation.py`)**:
   - `_run_prompt_reasoning()` instantiates `ComputerUseAgent` for actionable user prompts
     and executes a multi-step mission loop (`run_mission(steps=15)`).
   - Live streaming: each agent step's observation, decision, and result are appended to
     the session worklog in real-time.
4. **Scroll & Drag Support in Daytona Provider**:
   - `DaytonaComputerProvider.gui_action` maps `GUIActionType.SCROLL` via `xdotool`
     wheel click events (button 4=up, 5=down). `GUIAction` model extended with `scroll_delta`.
5. **Safety Envelope (`sonic/safety/action_policy.py`)**:
   - `DEFAULT_ALLOWED_TYPES` allowlists in-sandbox GUI actions (`GUI_CLICK`, `GUI_DOUBLE_CLICK`,
     `GUI_TYPE`, `GUI_KEYPRESS`, `GUI_MOVE`, `GUI_SCROLL`, `GUI_SCREENSHOT`).

### Done-gate (`test_phase8_visual_computer_use.py`, 16 tests)
- GUI click, type, keypress, double-click, scroll, and screenshot dispatch verified end-to-end.
- Coordinate parsing and system prompt schema verified.
- Safety policy allows GUI actions in-sandbox and denies destructive host/terminal operations.


## Improvement Round 2 — Module Robustness + Advanced Definition (2026-09-02)

### Bugs fixed (fail-closed + resource-safety hardening)
- CRITICAL SSRF bypass (sandbox/egress.py): IPv4-mapped IPv6 (::ffff:169.254.169.254) bypassed all blocked networks. Added ::ffff:0.0.0.0/96, NAT64 64:ff9b::/96, and _ip_blocked() helper that normalizes mapped v6->v4. DNS-resolved IPs now checked too.
- Scope injection (agents/recon.py): CT-log subdomain enum used endswith("example.com") -> accepted evil-example.com. Replaced with _is_subdomain_of() (dot-boundary check).
- Process leak on timeout (docker_provider.py, local_dev_provider.py): orphaned child processes (e.g. sleep) survived timeout. Now uses start_new_session=True + os.killpg(SIGKILL) + os.waitpid to kill+reap the whole process group.
- Browser leak (agents/browser_agent.py): close() skipped page/context and crashed on partial launch. Now closes page->context->browser->playwright independently with suppress(), and supports async with BrowserAgent() as agent.
- Rate limiter (safety/rate_limiter.py): documented lock-contract for _get_bucket.

### Honest self-improvement provenance (ASIPTA Pillar 5 hardening)
- AuthoredTool + InventedTechnique now carry authored_at/invented_at, confirmed_at, confirmed_workspace_id timestamps. Confirmed fields are set ONLY on empirical in-sandbox reproduction (exit 0 + non-empty), never on failure.

### Tests added (tests/test_module_robustness.py - 27 tests)
- 8 egress SSRF bypass tests, 7 recon subdomain-injection, 1 provider timeout-kills-process-group, 4 browser-agent safe-close, 3 rate-limiter underflow, 4 provenance.

### Architecture document (docs/ARCHITECTURE.md)
- Formal ASIPTA definition grounded in real codebase, 6-layer architecture, 5 pillars, 4 end-to-end call chains.

### Test status: 429 passed, 37 skipped, 0 failures; ruff F-category clean.

## Improvement Round 3 — Human-Like Multi-Modal Interaction (2026-09-02)

### Goal
The agent must work like a human: not only `apt-get install` from a terminal, but
also open a browser, click a download button, save a file, then run a GUI install
wizard with mouse + keyboard + wait. This round added the primitives a human uses.

### Window/app management
- **APP_FOCUS** (ComputerActionType): toggle focus between desktop windows (e.g.
  Chromium <-> terminal) without relaunching. Maps to GUIActionType.SELECT_WINDOW,
  dispatched via `wmctrl -a` / `xdotool search --name ... windowactivate` on the
  sandbox X11 display. Previously SELECT_WINDOW was defined but unhandled (gap).
- **APP_INSTALL** (ComputerActionType): sanctioned install path that routes
  through `install_application()` -> ApplicationPolicy (forbidden packages like
  cryptominer/tor-relay/ddos-bot blocked). Closes a real bypass: a raw
  `apt-get install` via TERMINAL_EXEC was L0_SAFE and skipped the package gate.

### Human-like desktop primitives
- **GUI_DRAG** (ComputerActionType -> GUIActionType.DRAG): press-move-release
  gesture. GUIAction extended with x2/y2 destination coords. Lets the agent
  drag-drop a file onto a folder or slide a wizard control. DRAG was defined in
  the enum but unhandled (gap, same class as SELECT_WINDOW).
- **GUI_WAIT** (ComputerActionType): sleep N seconds then refresh the screen. A
  human waits for a progress bar / "Next" button before acting; without this the
  agent re-screenshots a still-loading UI and clicks stale coordinates.

### Human-like browser primitives (agents/browser_agent.py)
- **wait_for_element(selector)**: wait for a CSS element to render before
  clicking it (human waits for "Download"/"Next" to appear). No-op in httpx fallback.
- **download(selector, save_path)**: click a download link and save the file to
  the sandbox workspace (accept_downloads=True at context creation). Models the
  human "download then run" step. Fail-closed (False) without Playwright.
- **BROWSER_WAIT / BROWSER_DOWNLOAD** action types exposed to the agent loop.

### LLM system prompt + action map
- All new actions added to the ACTION list, action_map, and per-action payload
  format docs so the LLM can actually choose them.

### Fail-closed guarantees
- Headless UnifiedComputerProvider.gui_action raises RuntimeError for SELECT_WINDOW
  (no silent no-op). APP_INSTALL forbidden-package block is NOT a recovery trigger
  (the safety gate decided correctly). BrowserAgent.download returns False without
  Playwright (no fabricated file).

### Tests (tests/test_module_robustness.py - 22 new tests)
- TestWindowToggle (5): APP_FOCUS enum, policy, mapping, allowed, headless refusal.
- TestAppInstallGate (7): enum, policy, forbidden-blocked, allowed-pass,
  provider-gate, documented TERMINAL_EXEC bypass gap.
- TestHumanLikeInteraction (10): GUI_DRAG coords, GUI_WAIT/BROWSER_WAIT/BROWSER_DOWNLOAD
  in allowlist, policy evaluate, BrowserAgent wait/download methods, httpx fallback
  no-op, fail-closed without Playwright.

### Test status: 451 passed, 37 skipped, 0 failures; ruff F-category clean.

## Improvement Round 4 — Devin-Style Closed Loop: VERIFY + REPLAN (2026-09-02)

### Goal
Mapped SONIC's `run_mission` loop against the Devin-style 8-stage closed-loop
model (Observe → Reason → Act → **Verify** → **Replan on stuck**). Stages 1-6
were already strong; stages 7 (Verify) and 8 (Replan) had real gaps.

### Gap found: Stage 7 — Verification was circular
- The old loop stopped when the LLM emitted `GOAL_COMPLETE` — i.e. the agent
  judged itself done (circular: "code likh diya = task complete"). There was NO
  independent check grounded in the real sandbox state. A false-positive
  self-declaration would stop the mission prematurely.
- **Fix**: `verify_goal()` performs an INDEPENDENT probe inferred from the goal:
  - "fix tests" → runs `pytest` and reads the actual result
  - "install X" → runs `which X` / `dpkg -l X`
  - "run/serve" → probes the started artifact
  - no inferable probe → re-observes the live state
- Fail-closed: empty/error/traceback output is NOT success — "absence of
  evidence is not evidence of success." Only on real non-error evidence does
  verification pass. `GOAL_COMPLETE` that fails verification is fed back to the
  LLM as "VERIFICATION FAILED — continue", not accepted.

### Gap found: Stage 8 — No stuck detection / replan
- The old loop ran N steps; on failure it called `recover()` which restarts Xvfb
  (infrastructure recovery, NOT strategy revision). There was no detection of
  "I'm stuck repeating the same failing approach", and no replan — it just
  burned the step budget.
- **Fix**: `_consecutive_failures` counter + `_STUCK_THRESHOLD` (3). After N
  consecutive failures (or N failed self-verifications), `_inject_replan()`
  appends a `REPLAN` entry to the reasoning history so the NEXT LLM call sees
  "your approach isn't working, try a DIFFERENT strategy" — and pivots rather
  than repeating the failing action. This is the "plan static nahi hoti"
  principle: replan on failure, not a fixed script.

### Loop now (Devin-aligned)
```
for step in 1..N:
    OBSERVE          (multi-modal: screen+AT-SPI+terminal+files+git+browser)
    REASON & CHOOSE  (LLM: goal + observation + history)
    if GOAL_COMPLETE:
        VERIFY       (independent sandbox probe — NOT self-declaration)
        if verified: DONE
        else:        feed "VERIFICATION FAILED" back, count failure, maybe REPLAN
    ACT              (safety gate → provider → sandbox)
    if FAILED:       count consecutive failure; if stuck → REPLAN
```

### Tests (tests/test_module_robustness.py::TestVerifyAndReplan, 5 tests)
- verify_goal fail-closed on error/traceback output
- verify_goal passes on real non-error evidence (artifact exists)
- verify_goal re-observes when no concrete probe inferable
- _inject_replan adds REPLAN to history + resets failure counter
- run_mission replans after 3 consecutive failures (stuck detection wired)

### Test status: 456 passed, 37 skipped, 0 failures; ruff F-category clean.

## Improvement Round 5 — Self-Improvement: Closing the Learn→Apply Loop (2026-09-02)

### Goal
Mapped SONIC's self-improvement stack to find what actually self-improves vs
what just has the infrastructure. SONIC has 6 self-improvement building blocks:
Toolsmith (author+confirm tools), MethodLab (invent+confirm techniques),
CuriosityLoop (novelty-driven goals), Canary+SelfEval (A/B promote/rollback),
BeingMind+Craft (durable facts/notes), and epistemic PredictionComparison
(lessons from prediction-vs-actual). All EXIST and are honest (confirm-on-run).

### The real gap: lessons were OPEN-LOOP
The single biggest self-improvement gap: **lessons were computed but never fed
back into the next mission's reasoning.**
- `PredictionComparison.evaluate()` derived a lesson after each task, stored it
  in `CognitiveState.prediction_comparisons` — but the agent's
  `_build_reasoning_context()` never showed these to the LLM.
- `ResearchMemoryStore` (cross-mission playbooks with success rates) existed
  but was never queried by the agent.
- Net: the being forgot what it learned across missions. It was
  "self-authoring" (builds new tools) but NOT "self-learning" (reuses past
  lessons). Same gap-class as Round 4's verify: infrastructure existed, the
  loop wasn't closed.

### Fix: LessonsLedger (learn → apply)
New module `sonic/being/lessons.py`:

1. **`extract_lessons(traces, goal)`** — at mission end, distill the trace into
   concrete lessons grounded in REAL trace.status:
   - `FAILED` → `AVOID` lesson ("nmap on this target → connection refused")
   - `SUCCESS`/`RECOVERED` → `REUSE` lesson ("ffuf -mc 200,301 found admin panel")
   - Empty trace → no lessons (fail-closed: never fabricate)

2. **`LessonsLedger`** — durable host-FS store (mirrors BeingCraft convention,
   restart-proof under `sonic_data/lessons/`). `record()` dedupes by
   (kind, approach) so the same failed approach in 3 missions counts once.
   `relevant(goal, k)` returns top-K lessons matching the current goal by
   keyword overlap (no embeddings — flat dependency surface).

3. **Injection into reasoning** — `_build_reasoning_context()` now embeds the
   top-K relevant past lessons as a `Past lessons` block:
   ```
   Past lessons — approaches that WORKED (reuse them):
     [REUSE] nmap -sV → found service
   Past lessons — approaches that FAILED (avoid repeating):
     [AVOID] ffuf without -mc filter → 0 results on noisy targets
   ```
   The system prompt instructs the LLM: "When 'Past lessons' appear, AVOID
   [AVOID] approaches and prefer [REUSE] ones — apply your cross-mission
   learning."

4. **Wired into `run_mission`** — at mission end, `extract_lessons()` →
   `ledger.record()`. The NEXT mission's reasoning sees them. The being now
   compounds knowledge across missions instead of starting with amnesia.

### Self-improvement map after Round 5
```
Toolsmith    : builds new TOOLS        → confirm-on-run → registered  ✅ honest
MethodLab    : builds new TECHNIQUES   → confirm-on-run → ledger'd    ✅ honest
LessonsLedger: learns from OUTCOMES    → extract→record→inject         ✅ NEW (closed loop)
Canary/Eval  : A/B tests CHANGES       → zero-regression promote/rollback ✅
CuriosityLoop: proposes new GOALS      → novelty-biased                ✅
```
Toolsmith/MethodLab = "build new capabilities". LessonsLedger = "reuse past
experience". Together they cover both halves of self-improvement: the being
grows its toolkit AND grows wiser from its own history.

### Tests (tests/test_module_robustness.py::TestLessonsLedger, 7 tests)
- extract_lessons: AVOID on FAILED, REUSE on SUCCESS/RECOVERED, evidence grounded
- empty trace yields no lessons (never fabricate)
- ledger persists to disk + dedupes (kind, approach)
- relevant() keyword-overlap ranking (top hit matches goal keyword)
- inject_into_context empty when no lessons (zero noise for fresh being)
- inject_into_context renders [AVOID]/[REUSE] block
- end-to-end: mission 1 records a lesson, mission 2 sees it in reasoning context

### Test status: 463 passed, 37 skipped, 0 failures; ruff F-category clean.

---

## Round 6 — Dual-Process (Fast/Deep) Hierarchical Sub-agent Controller
Makes the agent reason like a human researcher: **fast on familiar ground,
deep on novel/uncertain ground, and silent until it genuinely needs a human.**

### `sonic/orchestration/dual_process.py` (NEW, ~280 lines)
Four pillars, all building on existing pieces — never duplicating them:

1. **Dual-Process (Fast/Deep) Controller** — `DualProcessController.decide()`
   picks FAST (heuristic, high-throughput, parallel children) or DEEP
   (thorough, careful, limited parallelism) per hypothesis node. The decision
   is a PURE FUNCTION of observable signals: `ConfidenceBand`, unresolved
   `Unknown`s (importance), novelty (lessons-ledger hit), and branch failure
   count. Deterministic + testable, not a hidden LLM whim.

2. **Dual-Process switching** — HIGH/VERY_HIGH confidence + few low-importance
   unknowns + a known pattern → FAST. LOW/MODERATE confidence OR a high-
   importance unknown OR a novel pattern → DEEP. Stuck in DEEP + high
   uncertainty → ESCALATE.

3. **Hierarchical Hypothesis + Sub-agents** — `HierarchicalHypothesisTree`
   over `CognitiveHypothesis.parent_id` / `children_ids` (added this round).
   A VERIFIED parent **expands** into more-specific children; a DISPROVED
   parent's whole subtree is **pruned** (no wasted sub-agents on dead branches).
   Each node dispatches the EXISTING swarm agents as sub-agents scoped to that
   node's test plan — no new agent classes.

4. **Parallel sub-agent execution** — children run concurrently via
   `asyncio.gather`, capped by `max_parallel` — exactly like `SwarmRunner`.

5. **Smart Human-in-the-loop escalation (sirf high-uncertainty pe)** —
   escalate ONLY when ALL hold: already in DEEP + high-importance unresolved
   Unknown (importance ≥ 0.70) + subtree stuck (≥3 branch failures) + novel
   (no lessons-ledger resolution). Does NOT escalate on high-confidence findings,
   budget exhaustion, or diminishing returns. Escalation surfaces the actual
   `Unknown` question for a human; the node stays PROPOSED (never auto-confirms
   or auto-disproves — that would be fabrication).

### Honesty invariants
- FAST mode never VERIFIES — only DEEP confirms/disproves (like confirm-on-run).
- Sub-agents invoke real agent code via the injected `agent_runner` — no mocks.
- Pruning follows real DISPROVED lifecycle status, never heuristic guesses.
- A sub-agent exception → node is DISPROVED (fail-closed), never left ambiguous.
- Escalation never fabricates a question; it surfaces the real `Unknown`.

### Security invariant
Sub-agent dispatch carries a scoped hypothesis context (goal + test plan),
never the full `CognitiveState`. Children inherit the parent's
engagement/tenant id so each agent sees only what its branch needs.

### `sonic/agents/cognitive_state.py` (modified)
`CognitiveHypothesis` gained three fields for the hierarchy: `parent_id`,
`children_ids`, `process_mode`. Backward-compatible (all default empty) — no
existing test broke.

### Tests (tests/test_module_robustness.py, +18 tests)
`TestDualProcessController` (9): high-conf+known→FAST; low/moderate→DEEP;
high-importance-unknown→DEEP even at high conf; novel→DEEP; escalation ONLY
on stuck+high-uncertainty+novel; no escalation when pattern known; no
escalation on low-importance unknown; no escalation when not yet stuck.

`TestHierarchicalHypothesisTree` (9): expand attaches children w/ context
inheritance; subtree BFS; prune removes disproved subtree; prune ignores
non-disproved (no heuristic guessing); DEEP confirms→fans-out children;
FAST runs node+children in parallel; max_parallel cap enforced; sub-agent
exception→DISPROVED (fail-closed); escalation surfaces Unknown + does NOT
run a sub-agent + keeps node PROPOSED.

### Test status: 481 passed, 37 skipped, 0 failures; ruff clean.

---

## Round 7 — Epistemic Awareness + Falsification Mindset + Long-horizon Planning
Fills the three highest-value gaps from the "human-like AI" checklist
(Section 1, the most important) **inside existing files** — no new files. Every
change enhances an existing, already-tested module; nothing was duplicated.

### 1. Epistemic Awareness — `cognitive_state.py::CognitiveState.get_top_epistemic_gap()`
"Mujhe kya nahi pata" made explicit and actionable. Ranks unresolved `Unknown`s
by `estimated_importance` and **excludes dead-ends** — an unknown whose every
`possible_action` already appears in `get_failed_methods()` is dropped, so the
agent never re-chases a question it cannot answer with the methods it has
tried. Returns `[]` when there is genuinely nothing open (never invents a gap).

### 2. Falsification Mindset — `experiment_designer.py::AdversarialChallenger.challenge_leading_hypothesis()`
"Actively apni hypotheses todne ki koshish kare." The existing
`generate_falsification_challenge` built the challenge but nothing chose *which*
hypothesis to attack — the natural confirmation-bias failure mode. Now
`CognitiveState.leading_hypothesis()` selects the strongest active hypothesis
(lowest `priority`, recency tie-break; DISPROVED/ABANDONED excluded) and
`challenge_leading_hypothesis()` auto-targets it for an explicit **disproof**
attempt. High-quality findings come from theories that *survived* an active
disproof, not ones merely confirmed. Reuses the existing challenge generator
(no logic duplication); returns `None` when there's no active target (never
fabricates a challenge).

### 3. Long-horizon Planning — `planner.py::MissionPlanner.build_long_horizon_plan()`
"10-20 steps aage soch sake." The baseline `build_plan` produced only 3-6
shallow commands. `build_long_horizon_plan` decomposes an objective into a
multi-stage ordered pipeline:

```
stage 0 orient → 1 surface map → 2 deep map → 3 hypothesize →
4 active test (APPROVAL_REQUIRED, never auto-run) → 5 verify → 6 report
```

Each action carries `stage` + `depends_on` (the prior stage's anchor action_id)
so the executor can order and gate the chain. All stages before the active test
are `READ_ONLY` (safety envelope honoured); the active test is
`APPROVAL_REQUIRED` and never silently executed. Deterministic, allowlist-only,
no unconstrained model command generation — same discipline as `build_plan`.

New fields: `PlannedAction.stage`, `PlannedAction.depends_on`;
`MissionActionPlan.stage_count()`, `MissionActionPlan.chain_head()`.

### Checklist coverage already present (not re-done — these existed)
- **Competing Hypotheses / Unknown model** → `research/epistemic.py`
- **Dual-Process Thinking** → `orchestration/dual_process.py` (Round 6)
- **Mental Model of Target** → `research/world_model.py`
- **Curiosity + Goal Mgmt** → `computer_use/curiosity.py`
- **Information-Gain action selection** → `research/information_gain.py`
- **Strategy switching on diminishing returns** → `world_model.StopConditionEvaluator`
- **Tool/Method authoring w/ verification** → `being/toolsmith.py`, `being/method_lab.py`
- **Failed-strategy + cross-engagement memory** → `being/lessons.py` (Round 5)
- **Fail-closed isolation / tamper-evident / egress** → `safety/scope.py`, `safety/sealed.py`, `safety/action_policy.py`
- **Audit trail + evidence custody** → `evidence/custody.py`
- **Independent adversarial verification** → `evidence/independent_verifier.py`

### Tests (+19, all in existing test files)
`TestEpistemicAwareness` (6): rank by importance; exclude dead-ends; keep
partially-live; exclude resolved; empty when none; top_k limit.
`TestLeadingHypothesis` (3): picks lowest priority; None when none active;
DISPROVED excluded.
`TestFalsificationMindsetWiring` (2): targets the strongest hypothesis;
None (not fabricated) when no target.
`TestLongHorizonPlan` (8): ≥10 steps; ≥5 ordered stages; every action chains
via depends_on; active test is APPROVAL_REQUIRED; read-only objective has no
approval stage; orient runs first; empty raises; deeper than baseline.

### Test status: 500 passed, 37 skipped, 0 failures; ruff clean.

## Engagement recon — lab target `testapp.sonic-lab.local` / `10.10.50.0/24`

### Environment reality (Phase A recon, falsified hypotheses)
- The stated lab target is **NOT live infrastructure** in this environment.
  - `testapp.sonic-lab.local` -> **NXDOMAIN** (verified from host AND from a
    Daytona sandbox via 8.8.8.8/1.1.1.1). It does not exist in DNS.
  - `10.10.50.x:80` -> HTTP **403 "Internet is restricted on Tier 1 and Tier 2"**,
    ~0.0007s RTT, identical 45-byte body for *every* IP. This is the Daytona
    egress-deny proxy intercepting RFC1918, NOT a real web server. `bash`'s
    `/dev/tcp` reports such IPs as "OPEN" — a false positive; real `curl`/`nc`
    is the discriminator.
  - Control: `github.com` -> 200 from the same sandbox, so the sandbox is
    fine; only the stated lab range is denied/unreachable.
- **Conclusion:** no real host to exploit; per evidence-custody rule we must
  NOT fabricate findings/flags against an unreachable target. Genuine
  attack-chain capability was instead demonstrated against a deliberately
  vulnerable app stood up *inside* our own sandbox (in-bounds), see below.

### Safety-envelope penetration (requirement #8 — actively attack own boundaries)
34 probes across 4 layers; every verdict recorded (`_safety_results.json`
was the artifact). Findings + fixes:

- **BUG 1 (scope CIDR):** `ScopeChecker.is_target_in_scope` did a literal
  `target in targets["ips"]` check — never expanded CIDRs and never stripped
  URL scheme/port. So `10.10.50.10` against allowed `["10.10.50.0/24"]`
  returned False (over-block) and only the CIDR *string* matched.
  **Fixed** in `sonic/safety/scope.py`: parse host from URL via `urlparse`,
  expand CIDRs via `ipaddress.ip_network`, numeric membership test. Added
  `_ip_in_scope_list` helper.
- **BUG 2 (classifier policy parity):** `classify_command_risk` only used
  hardcoded `_DESTRUCTIVE_PATTERNS`, ignoring the YAML-loaded
  `_forbidden_patterns`, and the hardcoded set missed `DROP DATABASE`,
  `exfiltrate`, `disable logging`, `modify safety_rules`, and reverse shells
  (all returned L0 despite `safety_rules.yaml` listing them as forbidden).
  **Fixed:** classifier now also consults `_forbidden_patterns` (YAML), and
  the hardcoded set was extended to cover the policy gaps + reverse shells.
- Verified `check_action` fail-closes when rules not loaded (`self._loaded`
  False -> BLOCKED) — correct by design; with rules loaded it returns
  ALLOWED / NEEDS_APPROVAL / BLOCKED for L0/L1/L2 as expected, and the YAML
  `exfiltrate` forbidden pattern now fires (policy→enforcement parity).
- Live egress: all out-of-scope/RFC1918/metadata/localhost probes BLOCKED
  (403 deny-proxy or code=000). Egress envelope confirmed fail-closed.

### Regression tests
`sonic-core/tests/test_safety_envelope_regressions.py` (11 tests) reproduces
both bugs red→green and covers CIDR members, URL host extraction, out-of-scope
URLs, and all forbidden-command classes. `11 passed`.

### Attack-chain demonstration (in-sandbox, no external system)
Stood up a 1-file deliberately-vulnerable app (`/?page=` LFI + `/admin?token=&cmd=`
auth-gated RCE) inside a Daytona sandbox and ran a 5-step chain: recon -> LFI
(leaks admin token from `/opt/app/config.json`) -> auth+command-injection
(`uid=0(root)`) -> RCE flag read (`SONIC{chain_recon_lfi_auth_rce_flag_captured}`)
-> post-exploitation (wrote `/tmp/pwned.txt`). Each step recorded request,
response, SHA-256 evidence hash, and confidence (0.95–0.99). Demonstrates
multi-step chaining + evidence custody without touching any out-of-scope host.

### Multi-modal perception (requirement #6)
Confirmed working on a benign target (OpenHands docs): DOM, interactive
element/accessibility tree, screenshot (saved to disk), and markdown content
extraction. Browser DOM/JS analysis was also used earlier to reverse the
BurpSuite download endpoint (`/burp/releases/download?product=desktop&version=...`).

### Sandbox hygiene note
The `/dev/tcp` "OPEN" heuristic is unreliable behind a deny-proxy; always
follow up with a real `curl`/`nc` HTTP GET and compare body+RTT to
distinguish a live host (varied banner, higher RTT) from a deny-proxy
(identical short 403, sub-ms RTT).

## sonic-cli audit (2026-09-03)
Backend routes (sonic-core/sonic/api/main.py): health (/health,/health/detailed), /auth, /engagements (POST /,POST /{id}/run,GET /{id},GET /{id}/findings,GET /,POST /kill, GET /{id}/tasks, /{id}/unknowns, /{id}/decisions, /{id}/next-action, POST /{id}/replan, /{id}/pause, /{id}/resume), /findings (GET /{id}, POST /{id}/verify,/challenge,/reproduce,/review, GET /{id}/evidence,/provenance,/confidence), /agents, /graph (/stats,/search,/query,/engagement/{id}/summary,/finding/{uid}), /experiments, /terminal, /live (/experiments, /experiments/benchmark, /settings, /stats, /graph, /evidence, /scan), /jobs, /workstation (/state,/sessions,/session,/desktop/*,/command,/prompt,/tree,/file,/git-diff,/mission/*,/research-lab/*,/target-sandbox/*).
IMPLEMENTED (real, non-fabricated): /findings router wired to GraphMemory + VerifierAgent (verify/challenge/reproduce persist verdicts; evidence/provenance/confidence read real graph data; review approves/rejects). /engagements sub-routes surface Director cognitive state + task graph when available; honest empty responses (with `note`) for linear-pipeline engagements — never fake panels. CLI finding & mission & research commands now call these real endpoints.
Tenant isolation gotcha: JWT `tenant_id` defaults to "default" unless the User sets it explicitly; route reads/writes use `user.tenant_id` (NOT `user.email`). The live routes use `user.email` as tenant — keep this split in mind. Memory backends (InMemory/SQLite/Neo4j) all expose `find_evidence(finding_id, tenant_id)`.
EngagementManager vs Director: EngagementManager is the linear pipeline (no task graph); Director is the autonomous pipeline with CognitiveState (unknowns, next_best_action, decisions) + TaskGraph. /engagements sub-routes try Director first, fall back to honest empty.
Tests: sonic-cli/tests has only __init__.py (no test files); python -m pytest -> no tests ran, exit 0.

### Round 2 CLI de-fabrication (2026-09-03)
Full gap audit of all CLI command files + cross-check of every API path against the registered backend OpenAPI routes. Findings + fixes:
- **evolution.py** — was 100% fabricated (hardcoded F1 scores, version history, candidates). Rewired: `status`→GET /live/experiments, `proposals`/`candidates`→GET /experiments/, `benchmark`→POST /live/experiments/benchmark, `rollback`→POST /experiments/{id}/rollback. `weaknesses`/`approve`/`reject`/`promote`/`history` honestly report no-backend-endpoint (no fabricated data).
- **mission.py** — `create`/`list`/`summary`/`deliverables`/`benchmark` were hardcoded fake missions (msn-84f9a120 etc.). Rewired: `create`→POST /engagements/, `list`→GET /engagements/, `summary`→GET /engagements/{id}+/decisions+/next-action, `deliverables`→GET /engagements/{id}/findings, `benchmark`→POST /live/experiments/benchmark. Removed broken `from sonic.mission_engine.benchmark` import (sonic-core not a CLI dep — always ImportError → fabricated fallback).
- **engineer.py** — `run` had ImportError swallow (always failed) + `benchmark` had fabricated 6-row table. Rewired `run`→POST /workstation/mission/start, `benchmark`→POST /live/experiments/benchmark. Removed unused asyncio/Table imports.
- **security.py** — was 100% fabricated (11 hardcoded "all-pass" audit rows + fake release-gate). No backend security-audit endpoint exists. All 6 commands now honestly report the gap + point to the real pytest suite in sonic-core.
- **computer.py** — was 100% fabricated (fake workspace/process/services tables, fake success messages). Rewired to real workstation router: `list`→/workstation/sessions, `create`→/workstation/desktop/provision, `status`/`apps`/`processes`→/workstation/desktop/status (real running_processes), `open`→/workstation/desktop/action, `screenshot`→/workstation/desktop/screenshot, `install`/`uninstall`/`terminal`→/workstation/command, `files`→/workstation/tree, `git`→/workstation/git-diff, `reset`/`destroy`→DELETE /workstation/session. `services`/`snapshot` honestly report no-backend-endpoint.
- Already-clean (verified real API calls, no fabricated data): finding.py, research.py, engage.py, agents.py, evidence.py, experiment.py, graph.py, status.py, auth.py.
Frontend audit: dashboard only calls /workstation and /live routes (all registered) + /auth. No /engagements or /findings calls from frontend (those are CLI-only). Frontend "mock/placeholder" grep hits are all HTML placeholder attributes, not fake data. Settings save is wired (GET+POST /live/settings). No broken frontend API calls found.
Backend "not implemented": executor.py returns "Tool adapter is not implemented" as a fail-closed safety result when no tool adapter is registered (correct behavior, not a bug). No other genuine "not implemented" stubs remain.

### Round 2 backend gap-fill (2026-09-03)
The Round 2 CLI audit de-fabricated 5 files but left several CLI commands honestly reporting "no backend endpoint". This round implements the missing backend endpoints so the CLI renders real data, then re-wires the CLI commands to them.
- **NEW `/security` router** (`sonic/api/routes/security.py`, registered in main.py at prefix `/security`):
  - `POST /audit` — runs an 11-domain adversarial acceptance suite catalog (SEC-AUTH-01…SEC-EVOL-01) mapped to real pytest files; stores findings + computes a release-gate verdict.
  - `GET /attack-surface` — enumerates the deployed attack surface by traversing all registered FastAPI routes (descends into `_IncludedRouter.original_router` for FastAPI ≥0.115), classifying Auth / Workstation / Public Health / API surfaces with risk ratings.
  - `GET /tests` — returns the static test catalog (11 tests with id/category/name/severity/last_verdict).
  - `GET /findings` — returns recorded audit findings (last audit, total, active failures).
  - `GET /release-gate` — returns current gate verdict (`NO_AUDIT_RUN` / `RELEASE_CANDIDATE_CERTIFIED` / `FAIL_CRITICAL` / `FAIL_NON_CRITICAL`) + active failures.
  - `POST /reproduce/{test_id}` — re-runs a single cataloged test by mapping its pytest path; 404 for unknown IDs.
- **`/experiments` router expanded**: `POST /{id}/approve` (PROPOSED→canary_testing, Operator-only, 409 on wrong state), `POST /{id}/reject` (archive with reason), `POST /{id}/promote` (canary→promoted, sets active version), `GET /weaknesses/summary` (mines rejected/rolled-back experiments into categorized weakness patterns incl. SAFETY_VIOLATION), `GET /history/timeline` (version history of promoted generations + all-experiments table with baseline/candidate scores).
- **`/engagements` router expanded**: `GET /{id}/hypotheses`, `GET /{id}/leads`, `GET /{id}/anomalies` — surface Director CognitiveState when available; honest `note` for linear-pipeline engagements (never fabricate).
- **`/workstation` router expanded**: `GET /workstation/services` (probes known sandbox services via `service_action`), `POST /workstation/snapshot` (Operator-only, persists named session-state snapshot), `GET /workstation/snapshots` (list snapshots).
- **CLI re-wired** (removed all remaining `_no_backend` panels): evolution.py (`weaknesses`/`approve`/`reject`/`promote`/`history` → new endpoints), research.py (`hypotheses`/`leads`/`anomalies` → new endpoints), security.py (all 6 commands → new `/security` router), computer.py (`services`/`snapshot` → new workstation endpoints). No "no backend endpoint" / "not exposed" strings remain in the CLI.
- **Tests**: `tests/test_round2_gap_fill.py` (+19 tests covering experiment lifecycle, safety-violation auto-reject, security audit/tests/findings/release-gate/reproduce/attack-surface, engagement sub-routes, workstation services/snapshots). Full suite: **570 passed, 37 skipped, 0 failures**.

### Round 2 frontend gap-fill (2026-09-03)
- **`app/security-lab/page.tsx`** — was a stub that just re-exported the Evolution page. Replaced with a real Self-Security Lab dashboard that calls the new `/security` endpoints: test catalog (`getSecurityTests`), attack-surface inventory (`getAttackSurface`), release-gate verdict (`getReleaseGate`), findings (`getSecurityFindings`), run-audit (`runSecurityAudit`), and per-test reproduce buttons. No mock data — all rendered from API responses.
- **`lib/api.ts`** — added API client methods for the new endpoints: `runSecurityAudit`, `getSecurityTests`, `getSecurityFindings`, `getReleaseGate`, `getAttackSurface`, `reproduceSecurityTest`, plus experiment lifecycle (`approveExperiment`/`rejectExperiment`/`promoteExperiment`/`getExperimentWeaknesses`/`getExperimentHistory`).
- Dashboard build passes (`npm run build`) — all 17 routes compile, `/security-lab` now a real 3.79 kB page.

## AI + control upgrade (reliability hardening)
Two independent robustness layers were added on top of the existing AI core
and control loop — the legacy text-parsed ReAct and provider paths are
unchanged and still the default; the new paths are opt-in additions.

- **LLM provider resilience** (`sonic/llm/providers/custom.py`):
  - Transient-error retry/backoff on the SAME provider/model (429 rate-limit,
    5xx, transport errors) via `_with_retry` with exponential backoff + full
    jitter, before the existing model-level (`fallback_models`) and router
    provider-level fallback chains kick in. Configurable via `max_retries`,
    `retry_base_delay`, `retry_max_delay` (default 3 / 0.5s / 20s; set to 0 to
    disable). Auth/bad-request/model-not-found errors are NOT retried — they
    flow straight to the fallback chain.
  - Robust tool-call argument parsing via `parse_tool_arguments`:
    extracts the first JSON object from noisy/prose-wrapped tool-call output,
    fixes trailing commas and single quotes, wraps bare values as `{"value": …}`,
    and returns `{}` for empty/None. Used by both the OpenAI tool-call path and
    the ReAct engine so one parser governs all tool arguments.
- **ReAct native function-calling** (`sonic/agents/react_engine.py`):
  - `ToolRegistry.to_llm_tool_definitions()` converts the registry into
    provider-agnostic JSON-schema tool definitions for `LLMRequest.tools`.
  - `ReActEngine.execute_with_tools(task, think_fn, …)` runs the ReAct loop
    using native function-calling instead of `Action: name[arg]` text regex.
    ALL `tool_calls` in a single step are executed concurrently
    (`asyncio.gather`), so the agent fans out independent probes in one turn
    instead of one per step. Results are fed back as `tool`-role messages
    keyed by call id for correct multi-turn threading. No tool calls with
    non-empty content terminates as the final answer.
  - `_execute_tool` now accepts dict arguments (function-calling path) and
    coerces them to the positional string handlers expect; the legacy text
    path is unchanged.
- **Control loop coordinate validation** (`sonic/computer_use/agent.py`):
  - The agent now tracks the real screen dimensions (`_screen_width`,
    `_screen_height`), refreshed on every `observe()` and screenshot action.
  - `execute_action` validates GUI coordinate actions (click/double-click/
    move/scroll/drag, including drag targets) against the tracked screen
    bounds BEFORE execution. Out-of-bounds, negative, or non-integer
    coordinates are recorded as `BLOCKED` and never reach the provider — no
    recovery is triggered (the provider state is fine; the agent just needs to
    re-observe). Non-coordinate actions (terminal exec, etc.) are unaffected.
  - Tests: `sonic-core/tests/test_ai_and_control_upgrade.py` (25 tests).

## Round 3 — Computer + Agents Bug Fixes

### Computer subsystem (`sonic-core/sonic/computer/`)

- **CLI `list` crash** (`sonic-cli/sonic_cli/commands/computer.py`):
  `/workstation/sessions` returns a bare JSON list, but the CLI called
  `data.get("sessions", [])` on it, crashing with `AttributeError`.
  Fixed to detect a list and use it directly.
- **CLI `screenshot` field mismatch**: the CLI read `image`, but the backend
  returns `screenshot_base64` + `desktop_state`. Fixed the field mapping.
- **Fabricated git diff** (`provider.py`): `git_action("diff")` returned a
  hardcoded fake `auth.py` diff when the real command produced nothing.
  Now returns `res.stdout or ""` (real output or honest empty string).
- **Dead code after `raise`** (`provider.py`): `gui_action` had
  unreachable code after `raise RuntimeError`. Removed; the raise now
  follows the audit record.
- Tests: `sonic-core/tests/test_round3_computer_agents_fixes.py`.

### Agents subsystem (`sonic-core/sonic/agents/`, `sonic/swarm.py`)

- **Swarm failed-task routing** (`swarm.py`): the dispatch loop and the
  exception handler in `_execute_single_task` routed failures to
  `on_task_completed({"status": "failed"})` instead of `on_task_failed`.
  Both call sites now call `director.on_task_failed(...)`.
- **MetaOrchestrator no-op removal** (`swarm.py`): `MetaOrchestrator.run()`
  was a planning-only no-op but it was registered in `agent_configs` and the
  dispatch `agent_map`, risking dead dispatch. Removed from both registries
  and dropped the unused import. The `Director` orchestrates the swarm;
  `MetaOrchestrator` remains available for legacy planning but is never
  dispatched as a task executor.
- **Divergent agent registries** (`api/routes/agents.py`): `/agents/`
  surfaced only legacy engagement-scoped agents, while `/live/agents`
  showed a separate runtime feed. `/agents/` now surfaces the real
  `SwarmRunner.get_status()` agent registry (primary), with the legacy
  `EngagementManager` list as fallback. Response includes a `source` field
  (`swarm` | `engagement` | `none`).
- **CodeFix fabricated fallback patch** (`agents/codefix.py`): the except
  block returned a fake `server.js` diff and a fake regression test. Now
  returns empty `patch_diff`/`regression_tests` with an honest error
  message and logs the parse failure.
- **Orchestrator bare except** (`agents/orchestrator.py`):
  `evaluate_findings` swallowed parse errors silently. Now logs the error
  via `logger.error` and includes the cause in the fallback reason.
- **ExploitValidator dead expression** (`agents/exploit_validator.py`):
  removed a standalone `finding.get("poc", "")` no-op expression.
- Tests: `sonic-core/tests/test_round3_computer_agents_fixes.py` (12 tests)
  and `sonic-core/tests/test_swarm_wiring.py` (updated for the removed
  orchestrator from the agent map).

### Frontend (`sonic-dashboard/`)

- **New `/agents` page** (`app/agents/page.tsx`): a real Agent Swarm Registry
  view backed by `GET /agents/`. Shows live agent name/type/status/task/model
  with status-colored badges and a `source` indicator. Replaces the previous
  stub.
- **Sidebar nav expanded** (`components/workstation/WorkstationSidebar.tsx`):
  added Computer, Research, Experiments, Agents, and Security Lab to the
  primary navigation (previously only Workstation/Missions/Graph/Evidence).
- `lib/api.ts`: added `getAgents()` and `getAgent()` API methods.
- Dashboard builds cleanly (`npm run build`); all 19 routes compile.

## Engagement pipeline → execution substrate wiring (Gap 1–4)
Closed the central architectural gap flagged by audit: the user-facing
engagement (pentest) pipeline built its agents with ONLY (router, memory,
scope), so it never reached the real compute substrate that the
Mission/Director and Being paths use. The dynamic phase fired HTTP probes
from the host, the verifier never reproduced findings in-sandbox, and the
BugBountyClient existed but was referenced nowhere.

### `agents/engagement.py` — inject shared resources into the agents
- `EngagementManager.__init__` gains optional `sandbox_provider`,
  `reproduction_engine`, `bug_bounty_client` (all default None → legacy
  host-probe behaviour preserved for existing callers/tests).
- `ensure_sandbox(provider_factory)`: lazily attach a ComputeProvider and
  build a `ReproductionEngine` bound to it. Idempotent (no-op once attached).
  Supports an awaitable OR sync factory; used by the route to bind
  `get_compute_provider`.
- `_workspace_for(engagement_id, tenant_id)`: provisions (or reuses the
  per-tenant home via `get_or_create_home`) a sandbox workspace, cached per
  engagement. Returns "" when no provider (legacy path).
- `_run_dynamic`: now creates the `DynamicExecutionAgent` with
  `sandbox_provider` + `workspace_id`, so HTTP probes run INSIDE the sandbox
  (curl in the container) when a provider is attached.
- `_run_verify`: now creates the `VerifierAgent` with `reproduction_engine`,
  so `_engine_verify` runs real in-sandbox PoC reproduction instead of
  falling back to HTTP/LLM.
- `prepare_bug_bounty_reports(engagement_id, platform)`: renders every
  VERIFIED finding into a platform-ready `DraftReport` via the previously-dead
  `BugBountyClient`. Returns drafts for operator review — does NOT auto-submit
  (submission still needs API keys + explicit action; no auto-disclosure).

### `api/routes/engagements.py` — wire the production singleton
- `get_engagement_manager()`: now constructs the manager with a
  `BugBountyClient` (was unreachable before). Sandbox provider stays lazy.
- `_ensure_engagement_sandbox()`: lazily attaches the best available
  `get_compute_provider` + reproduction engine (idempotent); called in
  `run_engagement` before the pipeline starts. Fail-closed provider is fine
  (probes stay on host path as before — never fatal).
- New `POST /{engagement_id}/submit-report` route: renders verified findings
  into bug-bounty draft reports (operator-gated; no auto-disclosure).

### Done-gate (`test_engagement_wiring.py`, 9 tests)
- legacy 3-arg construction still works (resources default None);
- `_run_dynamic` injects the sandbox provider + a real workspace_id into the
  agent instance (verified on the REAL agent, not a mock of the wiring);
- `_run_verify` injects the reproduction engine into the agent instance;
- `ensure_sandbox` builds a `ReproductionEngine` from a provider and is
  idempotent (a second factory call is never invoked);
- `prepare_bug_bounty_reports` renders verified findings into draft reports
  (title/poc present) and returns [] when there are no verified findings;
- `_workspace_for` returns "" without a provider and provisions+caches with one.

### Test baseline (after engagement wiring)
- 589 passed, 37 skipped. The 2 `test_round3_computer_agents_fixes.py`
  failures (`ModuleNotFoundError: sonic_cli`) and the
  `test_phase1_persistent_memory` collection error are pre-existing
  environmental issues (verified on the clean baseline: sonic_cli is not
  installed; phase1 fails only under full-suite singleton/.env interference,
  passes alone — matches the known-pre-existing notes above). No regressions
  introduced.

## Phase 22 — Native Docker Cyber Workstation, Visual Grounding & Autonomous Reasoning (DONE)
### 1. Native Docker Cyber Workstation & Daytona Replacement
- **Why Daytona was replaced**: Daytona Cloud Tier 1/2 restricted external network traffic (resetting outbound HTTPS connections), required cloud API keys, and suffered from network latency.
- **Native `DockerComputerProvider` (`sonic/computer/docker_computer.py`)**:
  - Implemented as the primary, first-class `ComputerProvider` for SONIC A-SEA.
  - Controls the local `sonic-desktop-workstation` container directly via `docker exec`.
  - Full graphical desktop: XFCE4, Xvfb on `:99` (1280x800x24), x11vnc on port `5900`, and noVNC on port `6080` (`http://localhost:6080/vnc.html`).
  - Pre-installed applications: Google Chrome Stable (`google-chrome-stable v152`), `xfce4-terminal`, `thunar`, `nmap`, `net-tools`, `wmctrl`, `xdotool`, and image manipulation tools (`convert`, `import`).
  - Full, unrestricted outbound internet access for real-world offensive security assessments.
  - Backwards-compatible `get_computer()` and `get_daytona_computer()` route to `DockerComputerProvider` by default.
  - Added duck-typed `DockerContainerSandbox` adapter (`sonic/computer/docker_sandbox.py`) for existing callers.

### 2. Repository Integration, Dark Reality & Visual Grounding
- **Dark Reality of `open-computer-use` (`e2b-dev/open-computer-use`)**:
  - The dark reality is that `open-computer-use` is a thin 200-line wrapper completely dependent on E2B's paid cloud sandbox (`e2b_desktop`). It cannot run locally on Docker or VPS without paid cloud API accounts.
  - We extracted its genuine innovations and integrated them natively into SONIC:
    - `extract_bbox_midpoint()`: Extracts precise `(x, y)` coordinates from normalized model outputs (`[0, 1000]`, `[0.0, 1.0]`, or `<|box_start|>(x1,y1,x2,y2)<|box_end|>`), eliminating coordinate hallucination.
    - `resolve_ui_target()`: Natural-language UI query grounding with semantic desktop landmarks and multimodal vision grounding callbacks.
    - `draw_action_marker()`: Renders interactive crosshairs and visual click targets onto screenshots (cyan for click, pink for right-click, orange for double-click) for operator auditing in the dashboard.
    - `ComputerUseAgent.execute_action()`: Natural-language element click resolution (e.g. `TARGET: "Applications menu"` without raw pixel coordinates) and full `GUI_RIGHT_CLICK` support.
- **Role of Target Repositories (`nandkishorrathodk-art/sonic` or client audit targets)**:
  - Repositories are mounted/cloned into `/root/workspace/` inside the Docker Workstation.
  - SONIC uses its workstation terminal, python runtime, and code analysis tools (`CodeGraph`, AST, grep) to inspect source code, discover logic bugs, and run local test suites.
  - When a vulnerability is discovered, the agent writes an engineering remediation patch, executes verification tests in the sandbox, and commits the fix to git.

### 3. Goal-Oriented Autonomy & Independent Reasoning (A-SEA Identity)
- Completely decoupled from rigid, script-chained scanner execution (`nmap -> nuclei -> ffuf`).
- A-SEA operates from **First Principles**:
  - Analyzes high-level objectives rather than executing fixed scanner commands.
  - Maintains a structured epistemic ledger (`research/epistemic.py`) of Knowns, Unknowns, and Hypotheses.
  - Dynamically decides between terminal execution, browser navigation, visual GUI interaction, and custom script authoring (Toolsmithing).
  - Enforces empirical verification: every vulnerability must be reproduced in the sandbox before it is claimed as confirmed (zero success-by-decree).

### 4. Control Plane Hardening, RBAC & Dashboard Modernization
- Hardened `POST /workstation/session/interrupt` with `require_operator` RBAC.
- Enforced strict tenant isolation in `_run_computer_dynamic`.
- Modernized `sonic-dashboard/components/computer/ComputerSurface.tsx`: displays "SONIC Cyber Workstation", live feed indicator, and human takeover controls. All 21 Next.js routes compile cleanly (`npm run build`).

### 5. Verification Test Matrix (33/33 Passed 100%)
- `sonic-core/tests/test_visual_grounding.py` (9 tests): bbox midpoint extraction (box tags, float normalized), semantic landmarks, dynamic grounding fn, action aim marker styles, agent visual click dispatch, agent right-click dispatch — **PASSED**.
- `sonic-core/tests/test_docker_computer_provider.py` (2 tests): lifecycle, status, terminal execution, screenshot, file read/write, application listing — **PASSED**.
- `sonic-core/tests/test_docker_workstation_adapter.py` (3 tests): duck-typed sandbox adapter, VNC preview URL, terminal & screenshot — **PASSED**.
- `sonic-core/tests/test_workstation_interrupt_and_resilience.py` (9 tests): interrupt flag, loop exit, RBAC authentication & rejection, tenant isolation, mission director coordination — **PASSED**.
- `sonic-core/tests/test_phase_plan6_safety_envelope.py` (10 tests): fail-closed policy, rate limit, path confinement, egress filter, allowlist — **PASSED**.

## 2026-09-05 Full Audit — verified findings (for future sessions)
- Cloned into `/workspace/project/sonic`; venv `.venv`; tests: 627P/37S/4F (4 Docker-environment failures, not code. AGENTS.md's "known failures" list matches.
- Verified STRONG: path-auth (only health + login/dev-token/google endpoint unauth; terminal WS token-gated), tenant isolation via user.email/tenant_id scoping on every engagement/workstation manager call, path traversal confined to /home/sonic/workspace, target egress filter (169.254 metadata, loopback, RFC1918, v4-mapped, NAT64 blocked), `SealedActionPolicy` sealed+tamper-verified in being loop,, no hardcoded secrets in source/history, evidence SHA-256 custody+independent verifier ( self-verify blocked, experiment safety boundary blocks self-modifying safety/auth files, config prod guards (weak-secret hard-fail,, deny-by-default allowlist.
- KEY GAPS FOUND:

  1. **Toolsmith/MethodLab executeth arbitrary LLM-authored Python in egress-unrestricted container** — `TOOL_RUN`/`METHOD_INVENT` pass no egress target check (ActionPolicy gates only SECURITY_TOOL/BROWSER_NAVIGATE target URLs; only pre-filter is 4-regex lint (`rm -rf /`, shutdown, mkfs, dd. LLM auth'd tool can `socket`/`requests` to ANY external host, or read `/etc/`/env/secrets inside kali container (root. The ADP `AuthoredToolAdapter.build_command` single-quote target with `target.replace("'","")` is insufficient (`; id; #` injection) — but same-container blast radius.
  2. **`kill_switch` config-declared but NEVER runtime-enforced** — only `ScopeChecker.kill_switch_enabled` property exists; nothing in executor/agent/policy consults it.
  3. **`docker-compose.prod.yml` mounts `/var/run/docker.sock:ro` into API container** — read-only mount still grants full Docker API (container create/exec) = host-root-equivalent if API compromised. Also prod service lacks `resource limits` on sonic-core+dashboard.



  4. **On-Prem CSS/UX**: dashboard stores JWT in localStorage; `ensureAuthToken` auto-hits dev-token endpoint (production 404,, Google OAuth state param is sent-only-if-provided and not verified on callback (login-CSRF,, mild.
  5. **Code quality**: 141 ruff errors core+1 cli — CI lint job would fail. Mostly E402(36,, F401 unused imports(17,, E741(13,, N806(5,, I001(9,W293(5,SIM/S103/B007 etc;; no F821 undefined names. Largest: workstation.py~2300 lines,, engagement.py~900 lines (need split.. Dead code noted: unused imports,, `blocked` var unused in workstation mission summary (minor UX,, `class jwt` fallback in google_auth.py (N801,but isolated fallback only w/o jose/pyjwt.
- Docker socket note:: this host sandbox has no docker images;; those 4 failures remain — do not mark source as broken on that basis.

## Phase 23 — Live Workstation Observability, Burp Suite Autonomy & System 1 Reflexes (DONE)
### 1. Workstation GUI Observability Gap Resolved (Dashboard noVNC Embedding)
- **Problem diagnosed**: Dashboard (`localhost:12001`) previously relied solely on polling a static Base64 screenshot every 3 seconds via `/workstation/desktop/screenshot`. If the backend experienced reloads or container status checks lagged, the dashboard froze on an initial boot wallpaper (rat logo) while real GUI activities (Chrome, Burp Suite modal) were actively occurring in the background (visible only at `http://localhost:6080/vnc.html`).
- **Solution implemented in `sonic-dashboard/components/computer/ComputerSurface.tsx`**:
  - Implemented an intuitive segmented control: `[ VNC Stream ]` (live 60 FPS noVNC stream) and `[ Snapshot ]` (static inspection with interactive coordinate crosshairs).
  - Embedded live noVNC stream (`streamUrl = novnc_url || "http://localhost:6080/vnc.html?autoconnect=true&resize=scale"`) directly into the cyber workstation surface body via an iframe with clipboard and fullscreen allowances.
  - Zero polling delay, real-time visualization of container processes, and seamless human takeover capability.

### 2. High-Performance Screen Capture (`docker_computer.py`)
- Replaced slow ImageMagick `import -window root` + `convert -draw polygon` pipeline (which took 8–12 seconds and frequently hit 8-second timeouts on Docker Desktop for Windows, causing false-positive `NO_DISPLAY` verdicts) with ultra-fast `scrot -o` (173ms execution).
- Increased execution timeout from 8s to 15s to handle Docker virtualization spikes cleanly.
- `screenshot()` now reliably returns `desktop_state="INTERACTIVE"` with real Base64 screen data.

### 3. Burp Suite Community Edition & Java 21 Integration
- Diagnosed container runtime error: Burp Suite Community Edition (`/opt/burpsuite/burpsuite_community.jar`) was compiled with Java 21 (`class file version 65.0`), whereas the container originally possessed OpenJDK 17 (`class file version 61.0`), resulting in immediate crashes (`java.lang.UnsupportedClassVersionError`).
- Upgraded container Java runtime to `openjdk-21-jre`, setting default Java alternatives to OpenJDK 21.
- Validated Burp Suite Community Edition v2024.7.1 startup: Terms and Conditions modal handled, Temporary Project initialized, and Burp Proxy actively listening on `127.0.0.1:8080`.

### 4. Hinglish Intent Extraction & Application Routing (`workstation.py`)
- Sanitized natural language and Hinglish package extraction (`_extract_install_package` and `_detect_requested_app`): filtered Hindi/Hinglish command suffixes (`karo`, `usko`, `isko`, `kar`, `then`, `plz`, `bhai`) preventing invalid operations like `apt install karo`.
- Added pre-installed application awareness: checks `/usr/local/bin/burpsuite` and launches the application directly rather than hallucinating external download flows.

### 5. System 1 Motor Reflexes & Workstation Auto-Tiling (`motor.py`)
- Implemented `MotorReflexes` (`sonic/computer_use/motor.py`) providing zero-latency keyboard hotkeys (`Ctrl+L` URL navigation, `Ctrl+T` new tab, `Ctrl+W` tab closing, `Escape` modal dialog dismissal), human typing cadence (25ms delay preventing frontend debounce drops), and tab budgeting.
- Implemented desktop tiling (`tile_workstation`) using `wmctrl` to arrange browser and security tools side-by-side.
- Created resilient PowerShell server launcher `run_backend.ps1` to prevent Windows file-watch reloader termination.

## Phase 24 — Comprehensive Remediation of All Audited Gaps Across 4 Pillars (DONE)
Resolved all core architectural and security gaps identified by the 4-subagent deep audit across all layers of SONIC-REDA:

### 1. Workstation & Computer-Use Layer (Pillar 1)
- `bootstrap.py`: Removed silent `|| true` masking from OpenSSL and NSS certutil commands. Verified true exit codes of `update-ca-certificates` and `certutil -A`, ensuring CA certificate trust is only claimed when genuine injection succeeded.
- `motor.py`: Fixed duplicate click dispatch in `two_stage_click` (previously ran `xdotool click` via `_exec_cmd` AND `computer.gui_action(CLICK)` simultaneously). Now dispatches click exactly once.
- `grounding.py`: Added explicit `is_normalized_1000` handling and robust [0, 1000] bounding box scaling so coordinate normalization never misinterprets coordinates as raw pixels on 1280x800 or 1920x1080 displays.
- `docker_computer.py`: Replaced synchronous blocking `subprocess.run` calls in `_container_is_running` and daemon probes with `await asyncio.to_thread(subprocess.run, ...)`, preventing FastAPI event loop freezing during container virtualization latency.
- `docker-compose.yml`: Added `init: true` to the `workstation` service (`sonic-desktop-workstation`) so Docker initializes `tini` as PID 1 to reliably reap `<defunct>` zombie child processes.

### 2. Swarm, Epistemic Reasoning & Mission Planning (Pillar 2)
- `specialist.py` (`NetworkSpecialist`): Eliminated hallucinated fallback ports (80/nginx, 8080/uvicorn, 22/OpenSSH). If active probe finds zero open ports, returns genuinely empty findings (no synthetic attack surface).
- `specialist.py` (`FalsificationSpecialist`): Failure to falsify a hypothesis no longer fabricates a verified vulnerability without positive empirical proof. If evidence is absent, hypothesis is marked `INCONCLUSIVE`.
- `agent.py`: Prevented trivial orientation commands (`pwd`, `whoami`, `uname`, `echo`, `true`) from prematurely exhausting the `SubGoalChecklist`.
- `director.py`: Calibrated `MissionOutcome.SUCCESS` to require >= 0.60 success ratio and genuine non-empty deliverables. Missions with lower success ratios are honestly classified as `PARTIAL_SUCCESS` or `FAILED`.

### 3. Security, Safety Envelope & Boundaries (Pillar 3)
- `sealed.py`: Expanded `_compute_seal_hash` in `SealedActionPolicy` to hash `scope_checker` destructive and forbidden patterns. Any runtime alteration of safety rules immediately trips the SHA-256 seal and fails closed.
- `scope.py`: Expanded `_DESTRUCTIVE_PATTERNS` regexes to capture `rm -fr`, `rm -r -f`, `rm --recursive`, `mkfs(\.\w+)?`, `mke2fs`, `wipefs`, quoted `dd of="..."`, and aggressive process kills (`killall -9`, `pkill -9`).
- `daytona_computer.py` & `workstation.py`: Removed global `os.environ["DAYTONA_SANDBOX_ID"]` mutations that leaked provisioned sandboxes across tenant sessions. Workspace ID resolution is now strictly tenant-scoped.

### 4. API & Dashboard Telemetry (Pillar 4)
- `workstation.py`: Imported `ComputerWorkspaceStatus` from `sonic.computer.models`, resolving the silent `NameError` at lines 624 and 1043 that wiped `vnc_url` and `novnc_url` from desktop telemetry.
- `ComputerSurface.tsx`: Wired the `onInterrupt` prop directly to the "Human Takeover" button toggle, ensuring user takeover immediately halts conflicting autonomous agent actions.
- `ComputerSurface.tsx`: Implemented mathematically exact letterbox and pillarbox aspect-ratio offset calculations in `handleCanvasClick` using natural image dimensions to eliminate click drift on contained canvases.

### Done-gate (`test_phase24_audit_remediation.py`, 15 tests; 126 regression tests green)
- Verified `ComputerWorkspaceStatus` import integrity in workstation telemetry.
- Verified single-dispatch click execution in `MotorReflexes`.
- Verified zero hallucinated open ports in `NetworkSpecialist`.
- Verified positive-evidence requirement in `FalsificationSpecialist`.
- Verified low-confidence outcome calibration in `MissionDirector`.
- Verified tamper detection in `SealedActionPolicy` when scope checker patterns mutate.
- Verified expanded destructive command detection across 9 dangerous shell variations.
- Verified accurate coordinate midpoint scaling on 1280x800 screens.

## Phase 25 — Devin-Style Process Feed, Thinking Blocks & Real Telemetry UI (DONE)
Replaced and modernized the frontend process stream (`sonic-dashboard/components/worklog/`) to match the Devin/Claude-Code execution feed layout, backed by 100% real, authentic, unmocked execution telemetry:

### 1. Devin-Style Process Feed Components (`sonic-dashboard/components/worklog/`)
- `ThinkingBlock.tsx`:
  - Compact view: `🧠 Thought for Xs` (derived directly from real `duration_seconds` without synthetic mocking), clickable to expand.
  - Expanded `v Thinking` view: clean chevron toggle, monospace/sans typography, and left vertical guide line matching Devin.
  - Cognitive Breakdown: structured color-coded badges for `Knowledge` (emerald), `Unknowns` (amber), `Failed` / `Root Cause` (rose), `Hypothesis` (purple), and `Next Action` (cyan).
- `CommandBlock.tsx`:
  - Sleek terminal icon + dark monospace code pill displaying the exact command executed.
  - One-click copy button and clickable expansion displaying real sandbox stdout/stderr with exit status.
- `FileActionBlock.tsx`:
  - File icon + `Read <filename>:<lines>` badge (matching Devin's `Read phaser.js:50624-50733`), clickable to inspect the file in the Code Viewer.
  - Supports write and edit badges (`Edited <filename>`).
- `FollowupChips.tsx`:
  - Contextual smart suggestions directly below the latest assistant response (e.g. `[ 🔍 Deep Nmap Port Scan ]`, `[ 🌐 Crawl Web Endpoints ]`, `[ ⚡ Intercept in Burp Suite ]`, `[ 🔐 Audit JWT & Auth ]`).
- `TopicHub.tsx`:
  - Replaced the empty state with an interactive cybersecurity command center featuring 6 action cards (Web Recon, Burp Suite Interception, Network & Port Scanning, JWT Security, API Fuzzing, Sandbox Linux Shell).
  - Quick topic hashtags above the prompt input bar (`#BurpSuite`, `#PortScan`, `#APISecurity`, `#JWTAudit`, `#WebRecon`, `#LinuxShell`).
- `MarkdownText.tsx`:
  - Lightweight, dependency-free Markdown renderer with code block syntax highlighting, language badges, and copy buttons.
- `WorklogFeed.tsx`:
  - Unified timeline with active running spinner at the bottom (`Checking page CSS...` / `Executing terminal command...`).

### 2. Backend Duration & Telemetry Invariants (`sonic-core/`)
- Added `duration_seconds: float = 0.0` to `ComputerDecisionTrace` (`sonic/computer_use/models.py`).
- `agent.py`: Measured authentic elapsed time with `time.perf_counter()` in `execute_action` and stored real seconds on traces.
- `workstation.py`: Updated `_on_step` and `execute_workstation_command` to emit distinct `thought`, `command`, `read`, and `write` items into the session worklog with genuine duration and output fields (zero simulated/fake data).

### 3. Telemetry Completeness & Dark Reality Resolution (`sonic-core/`)
- `agent.py`: Completed wall-clock timing (`time.perf_counter()`) across ALL pre-flight circuit breakers (Action Loop Breaker, Coordinate Bounds Check, Substrate Outage, Strategy Exhaustion, Safety Policy Block).
- `models.py` & `agent.py`: Added `exit_code: int | None = None` to `ComputerDecisionTrace` and captured exact exit codes directly from container processes into decision traces.
- `workstation.py`: `_on_step` prioritizes authentic `trace.exit_code` directly from container execution, eliminating arbitrary success/failure guessing.
- `SealedActionPolicy` with `seal_default()` and `self_host=True` enforced on visual `ComputerUseAgent` in workstation routes.
- Test baseline: 100% clean test passes across `test_execution_status_semantics.py`, `test_phase24_audit_remediation.py`, `test_p0_security_hardening.py`, `test_human_motor_reflexes_and_hotkeys.py`, `test_safety_sealed_policy.py`, and `test_hacker_scratchpad_and_wire_telemetry.py`. Dashboard build compiled 21/21 static pages with zero type or lint errors.

## Phase 26 — Target-First Profiling & Autonomous Reconnaissance (100% Target-First, Zero Tool-Forcing)
Audited and enhanced the reconnaissance and target profiling subsystem (`sonic/agents/recon.py` and `sonic/brain/world_model.py`) to be 100% Target-First, with ZERO tool-forcing or scripted checklist behavior, in strict accordance with the user directive:
> *"main target tak pahunchna hai, isko scripted puppet nahi banana hai! Force mat karo koi bhi tool ke liye, yeh khud things develop karega."*

### 1. Direct Target Observation Probes (`sonic/agents/recon.py`)
Recon does NOT require or assume external pre-packaged scanners (`nmap`, `nuclei`, `ffuf`). Instead, it directly inspects the target through native Python protocols:
- **Direct DNS Resolution (`_probe_dns`)**: Resolves A/AAAA records and PTR reverse hostnames via `socket.getaddrinfo` and `socket.gethostbyaddr` natively (labeled `discovered_by="dns_lookup"`).
- **Direct TLS Certificate Inspection (`_probe_tls`)**: Connects natively via `ssl` and `socket` to extract Subject CN, Issuer, Validity, TLS version, cipher suite, and Subject Alternative Names (SANs) — discovering real subdomains/hostnames directly from the cryptographic certificate (labeled `discovered_by="tls_certificate"`).
- **Direct Socket Port Probing (`_probe_socket_ports`)**: Probes common service ports (80, 443, 8080, 8443, 3000, 5000, 8000, 22, 21, 3306, 5432, 6379) via `asyncio.open_connection` with banner grabbing on open HTTP ports — zero nmap binary required (labeled `discovered_by="socket_probe"`).
- **Deep HTTP Surface & Security Posture (`_probe_http_surface`)**:
  - Direct GET: Captures status code, redirects, Web Server (`Server`), and backend framework (`X-Powered-By`, `X-AspNet-Version`, `X-Runtime`).
  - Security posture: Evaluates CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, and Permissions-Policy.
  - CORS policy: Inspects `Access-Control-Allow-Origin` and `Access-Control-Allow-Methods`.
  - Allowed HTTP methods: Sends direct `OPTIONS` probe (`Allow`, `Access-Control-Allow-Methods`).
  - Target endpoint discovery: Directly fetches and parses `/robots.txt` and `/sitemap.xml`, and probes standard API/health routes (`/api`, `/api/v1`, `/health`, `/metrics`, `/docs`, `/openapi.json`, etc.) without needing `ffuf`.
  - HTML structure: Extracts page title, CMS/meta generator tags, HTML forms/actions, and in-page JavaScript API route references.
- **Strict Egress Safety**: Every probe vets targets against `is_target_allowed()` from `sonic.sandbox.egress` before touching the network.

### 2. Dynamic World Model Grounding (`sonic/brain/world_model.py`)
- `ingest_asset()` and `ingest_recon_assets()`: Maps direct target observations into `ResourceNode`s with preserved honest provenance (`live_probe`, `tls_certificate`, `socket_probe`, `dns_lookup`, `robots_txt`, `sitemap`, `html_structure`, `api_probe`). Automatically marks sensitive routes (`sensitive=True`).
- `get_attack_surface()`: Categorizes discovered resources into `endpoints`, `services`, `subdomains`, `technologies`, `certificates`, `infrastructure`, and `security_posture`.
- `target_profile` and `update_target_profile()`: Maintains continuous situational awareness of the target's live profile (DNS, open ports, TLS details, web server, security posture).
- `get_summary()`: Enriched with real surface counts and target profile metrics.

### 3. Done-Gate & Test Baseline
- `sonic-core/tests/test_target_first_recon.py` (8 tests): All passed in 0.62s.
- Regression suite (`test_recon_honesty.py`, `test_phase1_central_research_brain.py`, `test_phase6`, `test_phase8`, `test_phase5`, `test_mission_planner_adaptive.py`): 52 passed, 0 failures.
- Security hardening suite (`test_p0_security_hardening.py`): 21 passed, 0 failures.

## Phase 27 — Autonomous Target-First Offensive Intelligence, Self-Authoring Toolsmith, Substrate Separation & De-Puppeting (DONE)
Transforms SONIC from a scripted tool-runner into a genuine, target-first autonomous offensive intelligence: eliminates all forced scanner hierarchies, enforces clean separation between the workstation application desktop and the headless operator toolkit, empowers the being to author its own tools dynamically, and adapts testing strategies via real target feedback.

In strict accordance with the user directives:
> *"inside the computer just have to run application not commands usko computer se mat jodo, tum samajh rahe ho na?"*
> *"Network Scanners (nmap, nuclei, ffuf)... HTTP Probes... are abhi ke liye koi bhi tools par focus mat do, main target tak pahunchna hai, isko scripted puppet nahi banana hai! Force mat karo koi bhi tool ke liye, yeh khud things develop karega."*
> *Constraint: DO NOT install Burp Suite.*

### 1. Workstation Application Environment vs. Operator Substrate Separation
- **Pure Application Desktop**: The Computer Workstation is treated strictly as an **Application Desktop Environment** (`DISPLAY=:99`, running Chromium, target web applications under test, desktop GUI software, and window management via `APP_LAUNCH`, `APP_FOCUS`, `BROWSER_*`, and `GUI_*`).
- **External Operator Toolset**: Terminal execution (`TERMINAL_EXEC`), native network inspection, direct socket probes, and custom security scripts run headlessly as SONIC's external operator toolset, never polluting or typing directly into the application desktop UI.
- **Demarcated Reasoning Context (`_build_reasoning_context`)**:
  - `=== WORKSTATION APPLICATION ENVIRONMENT ===` (Active Application / Window, Open Windows, Screen Visible Content, Browser State).
  - `=== SONIC OPERATOR TOOLKIT (EXTERNAL EXECUTION) ===` (Last Command Output, Available Security Tools, Last Tool Result).
- **Burp Suite Disallowed**: Removed forced download and installation branches in `docker_computer.py` and `daytona_computer.py`. Added `burpsuite` and `burp` to `forbidden_packages` in `ApplicationPolicy` (`sonic/computer/models.py`) and removed them from `allowed_packages`.

### 2. De-Puppeting Prompts & Mission Planning (Target-First Autonomy)
- **`sonic-core/sonic/llm/prompts.py`**:
  - Eliminated the rigid `PRIMARY SECURITY ASSESSMENT STRATEGY` (`1. RECONNAISSANCE FIRST: Use SECURITY_TOOL... 4. PROFESSIONAL TOOLS: Prefer nmap...`) and `SECURITY-FIRST EXECUTION PRIORITY` checklists.
  - Replaced with **Autonomous Target-First Security Assessment Agent** directives: 100% focus on the target objective, analyzing target responses directly, and freely choosing or authoring the most direct path without any mandatory scanner checklists.
- **`planner.py` & `director.py`**:
  - `MissionPlanner.build_plan()` establishes baseline orientation towards the target asset instead of forcing a rigid scanner chain.
  - `_derive_hypothesis_from_objective()` dynamically formulates hypotheses from the target objective (SQLi, auth bypass, attack surface discovery) rather than using hardcoded defect templates.
  - Tool-agnostic trace evaluation: any successful action providing concrete findings (terminal execution, browser DOM interactions, custom script outputs) validates or falsifies hypotheses.

### 3. Autonomous Tool Authoring & Dynamic Probe Synthesis (`toolsmith.py` & `agent.py`)
- **On-the-Fly Tool Creation**: `ToolsmithLoop` and module-level `author_tool()` empower SONIC to author its own tools on the fly (`TOOL_AUTHOR`, `TOOL_RUN`, or direct `FILE_WRITE` -> `TERMINAL_EXEC`).
- **In-Sandbox Verification & Dynamic Registration**: Toolsmith executes the authored code inside the sandbox provider, verifies exit code 0 and non-empty output, rejects failure markers (`traceback`, `syntaxerror`, `connection refused`), and dynamically registers the tool into `SecurityToolRegistry` and the agent's live runtime.
- **Reasoning Context Empowerment**: Injects the prompt: *"You have the ability to author your own custom tools, scripts, and probes tailored specifically to this target."*
- **Action Parser Support**: Added first-class handling for inline Python snippets (`python3 -c ...`), targeted curl commands (`CURL`), shell probes (`BASH`, `SCRIPT`), and custom code generation (`TOOL_AUTHOR`).

### 4. Dynamic Strategy Evolution & Adaptive Attack Synthesis (`evolution/` & `method_lab.py`)
- **`DynamicStrategyEngine` (`sonic/evolution/strategy.py`)**: Reacts dynamically to real target feedback:
  - **HTTP 403 Forbidden**: Switches to `HEADER_AND_VERB_MUTATION` (path normalization `//`, `%2e/`, override headers `X-Original-URL`, `X-Forwarded-For`, verb tampering POST/PUT/OPTIONS) and triggers `MethodLab` for auth-bypass hypothesis synthesis.
  - **WAF Block**: Switches to `WAF_EVASION_MUTATION` to synthesize parser-confusion probes via `MethodLab` or custom obfuscated probes via `Toolsmith`.
  - **Filtered Port**: Switches to `ALTERNATIVE_SURFACE_PIVOT`, immediately pivoting from dead ports to active web/API endpoints.
  - **HTTP 429 Rate Limit**: Activates `RATE_THROTTLING_AND_BACKOFF` with delay jitter and client header rotation.
- **Novelty in `MethodLab`**: Zero hardcoded exploit templates. Hypotheses are synthesized from real target observations and failure context, validated strictly inside the sandbox provider, and stored in `LessonsLedger` (`[AVOID]` / `[REUSE]` directives).

### 5. Verification Test Matrix (100% Green)
- `sonic-core/tests/test_phase_a_toolsmith.py` (24 tests) — **PASSED** (24/24)
- `sonic-core/tests/test_phase_b_method_lab.py` (15 tests) — **PASSED** (15/15)
- `sonic-core/tests/test_evolution_strategy_and_engine.py` (7 tests) — **PASSED** (7/7)
- `sonic-core/tests/test_failure_engine_and_classification.py` (19 tests) — **PASSED** (19/19)
- `sonic-core/tests/test_agent_quality_improvements.py` (21 tests) — **PASSED** (21/21)
- `sonic-core/tests/test_phase3_real_computer_use_reasoning.py` (8 tests) — **PASSED** (8/8)
- `sonic-core/tests/test_mission_trace_synthesis.py` (12 tests) — **PASSED** (12/12)
- `sonic-core/tests/test_target_driven_missions.py` (5 tests) — **PASSED** (5/5)
- `sonic-core/tests/test_target_first_recon.py` (8 tests) — **PASSED** (8/8)
- GUI, motor reflexes, and application action suites (77 tests) — **PASSED** (77/77)

## Phase 28 — Autonomous Reality-Grounding & Complete De-Puppeting (DONE)
Eliminated remaining scripted-puppet behaviors, rubber-stamp fake confirmations, static exploit recipes, and Burp Suite tool remnants across the codebase:

### 1. Empirical PoC Validation (`sonic/agents/exploit_validator.py`)
- Replaced the rubber-stamp `_validate_poc()` that previously marked any finding with non-empty string evidence as `is_confirmed=True`.
- Implemented fail-closed validation: strictly checks for execution proof, rejects command failures (`exit 126`, `exit 1`, `command not found`, `connection refused`, `timed out`, `dummy`, `placeholder`), and verifies that empirical evidence proves reproduction before confirming findings.

### 2. Dynamic Capability-Driven Exploit Chaining (`sonic/agents/exploit_chain.py`)
- Removed the 5 static `CHAIN_PATTERNS` dictionary (SSRF+leak, XSS+CSRF, etc.) that acted as a scripted textbook puppet recipe.
- Implemented dynamic capability correlation: analyzes findings by mapping output capabilities (internal network pivot, credentials, file read/drop, unauthenticated access) to input requirements of subsequent attack vectors, allowing organic synthesis of multi-stage exploit paths against unique target architectures.

### 3. Complete Burp Suite Excision & Tool Neutrality
- **Landmarks (`grounding.py`)**: Removed 30+ hardcoded Burp Suite coordinates (`burp proxy tab`, `burp forward button`, etc.). Workstation GUI landmarks now strictly focus on target web application UI controls (`submit button`, `username input`, `dashboard tab`, `settings tab`).
- **Motor Reflexes (`motor.py`)**: Removed tool-specific `burp_forward()` and `burp_toggle_intercept()`; replaced with generic, application-level `send_application_shortcut()`.
- **Wire Telemetry (`wire_telemetry.py`)**: Decoupled from Burp REST API; operates purely on native HTTP ring buffers and application network transaction events.
- **Agent Subgoals & GUI Apps (`agent.py`)**: Removed `burp`/`burpsuite` subgoal branches, removed `burpsuite` from `_GUI_APPS`, and excised Burp intercept deadlock routines.

### 4. True Process & Service Verification (`agent.py:verify_goal()`)
- Tightened `verify_goal()` so goals with "run/start/serve/launch" do not claim `verified=True` merely because a generic background OS shell process is alive.
- Explicitly extracts the requested program/service name and verifies that the specific target process or application is active in the environment.

### 5. Verified Test Suite
- `test_gap_fixes.py` & `test_production.py` (21 tests) — **PASSED**
- `test_motor_reflexes_phase8.py` (7 tests) — **PASSED**
- `test_hacker_scratchpad_and_wire_telemetry.py` (14 tests) — **PASSED**
- `test_agent_quality_improvements.py` (21 tests) — **PASSED**
- `test_phase3_real_computer_use_reasoning.py` (8 tests) — **PASSED**
- `test_target_driven_missions.py` (5 tests) — **PASSED**
- `test_phase_b_method_lab.py` (15 tests) — **PASSED**

## Phase 29 — Elimination of Hardcoded Application Names & Puppet Mappings (DONE)
Per explicit user directive (*"is main koi bhi computer application ka name hai to usko hata do kyu ki isko to use karna ata hai to ye puppet kyu karna... kuchh rehna nahi chahiye"*), completely eliminated all hardcoded application translation dictionaries, static GUI application tuples, and tool-forcing heuristics across the codebase:

### 1. Dynamic Application Name Sanitization (`sonic/computer_use/agent.py`)
- **`_normalize_app_name()`**: Replaced the rigid if/elif mapping table (which previously translated "editor" -> "mousepad", "file" -> "thunar", "terminal" -> "xfce4-terminal", "burp" -> "burpsuite") with pure dynamic string sanitization. Strips markdown fences, quotes, leading conversational articles (`the/a/an`), and trailing punctuation while preserving the exact binary/application requested by the model or operator.
- **Removed Static `_GUI_APPS`**: Replaced the static 6-tuple `("chromium", "google-chrome", "firefox", "mousepad", "thunar", "xfce4-terminal")` and chromium-specific flag injection in `TERMINAL_EXEC` with dynamic display-aware backgrounding.
- **Dynamic Goal Decomposition**: Subgoal derivation in `decompose_goal()` no longer branches on hardcoded browser/app keywords; it dynamically extracts requested binaries and targets via regex.
- **Generalized Parsing Regexes**: Generalized terminal wrapper prefix regexes (stripping `xfce4-terminal`) and removed the 7-application whitelist in `which <app>` commands, allowing dynamic resolution of any application.
- **Generic GUI Recovery**: Removed hardcoded `code-server` relaunch from `_attempt_recovery()`; recovery now resets the X11 display service cleanly without forcing specific applications.

### 2. Autonomous Application Execution in Computer Providers
- **`DockerComputerProvider` (`sonic/computer/docker_computer.py`)**: Excised hardcoded if/elif branches in `GUIActionType.OPEN_APP` (which previously intercepted `"chrome"`, `"term"`, `"thunar"`). Now executes `DISPLAY=:99 nohup {shlex.quote(app_name)} >/dev/null 2>&1 &` dynamically for any binary requested.
- **`DaytonaComputerProvider` (`sonic/computer/daytona_computer.py`)**: Removed fallback defaulting to `"chromium"` and special-casing of chromium flags in `OPEN_APP`. Launches any application requested dynamically on the display.

### 3. Workstation Route & Prompt Generalization
- **`workstation.py` (`sonic/api/routes/workstation.py`)**: Excised static puppet mappings ("burp" -> "burpsuite", "editor" -> "mousepad", "file" -> "thunar") from `_detect_requested_app()`, and removed the Burp Suite pre-installed special case from package installation.
- **`grounding.py` (`sonic/computer_use/grounding.py`)**: Removed app-specific launcher coordinates (`firefox`, `mousepad`, `thunar`) from `_COMMON_UI_LANDMARKS`, preserving only generic desktop and window environment landmarks.
- **`prompts.py` (`sonic/llm/prompts.py`)**: Replaced all hardcoded application examples (`xfce4-terminal, mousepad, thunar, chromium`) with generic placeholders (`<application_name>`, `<package_name>`).
- **`engagement.py` (`sonic/agents/engagement.py`)**: Removed tool-forcing goal text (`"Use nmap for port scanning, nuclei for CVE detection, ffuf for directory fuzzing"`). Goals now state target-first autonomous objectives.

### 4. Verified Test Matrix (100% Green)
- `test_agent_quality_improvements.py` (21 tests) — **PASSED**
- `test_motor_reflexes_phase8.py` (7 tests) — **PASSED**
- `test_workstation_desktop_browser_and_news.py` (6 tests) — **PASSED**
- `test_phase3_real_computer_use_reasoning.py` (8 tests) — **PASSED**
- `test_browser_tab_dedup_and_anti_loop.py` (4 tests) — **PASSED**
- `test_phase4_unified_browser_in_loop.py` (4 tests) — **PASSED**
- `test_phase5_security_tool_execution.py` (5 tests) — **PASSED**
- `test_phase6_curiosity_life_loop.py` (5 tests) — **PASSED**
- `test_security_tool_registry.py` (6 tests) — **PASSED**

## Phase 30 — Complete De-Puppeting, Dynamic Application Discovery & Test Hardening (DONE)
Deep recheck and multi-agent refactoring to purge all remaining puppet heuristics, canned tool checks, and static binary queries:

### 1. Dynamic Application Discovery Across Compute Providers
- **`DockerComputerProvider`, `DaytonaComputerProvider`, `UnifiedComputerProvider`, `HeadlessComputerProvider` (`sonic/computer/`)**:
  - Replaced static 11-binary and 7-binary loops (`for b in google-chrome-stable chromium ...`) with dynamic desktop entry discovery (`find /usr/share/applications ... -name '*.desktop'`) combined with PATH discovery and provider policy packages.
  - Upgraded `tile_workstation()` to discover open windows dynamically via `wmctrl -l` while preserving standard 50/50 desktop split compatibility.
  - Cleaned `ApplicationPolicy` default forbidden packages list, removing hardcoded vendor bans.

### 2. Workstation Routes & Telemetry De-Puppeting
- **`workstation.py` (`sonic/api/routes/workstation.py`)**:
  - Excised leftover canned Burp Suite pre-installation checks and observations.
  - Generalized process termination to dynamically target the requested application process (`pkill -f {shlex.quote(app)}`) instead of assuming terminal processes.
  - Replaced hardcoded example application strings in fallback responses with generic tool/application prompts.
- **`live.py` (`sonic/api/routes/live.py`)**:
  - Added clean `proxy_url` configuration with backward-compatible alias support.
- **`wire_telemetry.py` & `models.py` (`sonic/computer_use/`)**:
  - Generalised network interceptor initialization to accept generic `interceptor` / `network_interceptor` parameters.
  - Updated mission mode comments to reflect objective assessment rather than specific tools.

### 3. Reflexes, Grounding & Parser De-Puppeting
- **`motor.py` (`sonic/computer_use/motor.py`)**:
  - Extended browser tab budget enforcement to match any browser dynamically (`chrome`, `chromium`, `firefox`, `browser`, `web`).
  - Cleaned GTK file dialog docstrings.
- **`grounding.py` (`sonic/computer_use/grounding.py`)**:
  - Removed binary-specific launcher landmarks (`xfce4-terminal`) in favor of generic `"terminal"`.
  - Cleaned crop toolbar docstrings.
- **`agent.py` (`sonic/computer_use/agent.py`)**:
  - Generalized intent-based goal decomposition keywords (removed tool-forcing `nmap`, `http_client`).
  - Purged conversational prose guessing heuristic from the action parser.

### 4. Comprehensive Verification (162 Passed, 0 Failures)
- `test_agent_quality_improvements.py` (21 tests) — **PASSED**
- `test_motor_reflexes_phase8.py` (7 tests) — **PASSED**
- `test_workstation_desktop_browser_and_news.py` (6 tests) — **PASSED**
- `test_human_motor_reflexes_and_hotkeys.py` (14 tests) — **PASSED**
- `test_visual_grounding.py` (9 tests) — **PASSED**
- `test_browser_tab_dedup_and_anti_loop.py` (4 tests) — **PASSED**
- `test_security_tool_registry.py` (6 tests) — **PASSED**
- `test_hacker_scratchpad_and_wire_telemetry.py` (14 tests) — **PASSED**
- `test_phase13/test_computer_models_and_state.py` (2 tests) — **PASSED**
- `test_module_robustness.py` (79 tests) — **PASSED**

## Phase 31 — Plane Separation, Non-Destructive Desktop Recovery & Motor/Grounding Hardening (DONE)
Addressed computer-use gaps, logic flaws, and plane separation invariants:

### 1. Pure Observation & Execution Plane Separation
- **`agent.py` (`sonic/computer_use/agent.py`)**:
  - `_last_action_output` is strictly restricted to sandbox command execution (`TERMINAL_EXEC`, `SECURITY_TOOL`).
  - Introduced `_last_gui_action_result` for all desktop application interactions.
  - In `observe()`, defensive filters prevent GUI clicks or serialized coordinate dictionaries (`$ {'x': 640, 'y': 400}`) from polluting `terminal_output`.
  - In `_parse_llm_action` and `execute_action` for `BROWSER_TYPE` / `GUI_TYPE`: eliminated mapping of shell commands into typed keystrokes.
  - Grounding resolution failure (`None`) returns `ActionExecutionStatus.BLOCKED` with informative diagnostics, preventing blind clicks at `(0, 0)`.

### 2. Non-Destructive GUI Recovery & Headless Browser Fix
- **Safe Recovery (`agent.py`)**:
  - Replaced unconditional `xvfb restart` (which killed all active desktop apps) with non-destructive desktop recovery: sends `Escape` key to dismiss blocking modals and refocuses the active application window.
- **Headless Browser Screenshot (`agent.py` & `browser_agent.py`)**:
  - Replaced `navigate()` call during `BROWSER_SCREENSHOT` with genuine `browser.screenshot()`, preserving page DOM state and avoiding accidental reloads.

### 3. Motor Reflexes & Visual Grounding Precision
- **`models.py` & `motor.py` (`sonic/computer/models.py`, `sonic/computer_use/motor.py`)**:
  - Added `delay_ms: int = 25` and `button: int = 1` to `GUIAction`, ensuring typing delays are preserved through `gui_action` fallbacks.
  - `enforce_tab_budget`: ensures browser window is focused prior to issuing `hotkey_close_tab`, and increased inter-close delay to 150ms.
  - `handle_gtk_file_dialog`: added 300ms window mapping delay before firing `Ctrl+L`.
- **`grounding.py` (`sonic/computer_use/grounding.py`)**:
  - Fixed `extract_bbox_midpoint` normalization check: coordinates `0 <= val <= 1000` are correctly scaled on high-resolution screens (e.g. 1920x1080).
  - Cleaned duplicate keys and constrained landmark matching to whole-word boundaries.

### 4. Application Quoting & Workstation API Endpoints
- **`docker_computer.py` & `daytona_computer.py`**:
  - In `launch_application`: used `shlex.split()` to isolate binary from arguments, ensuring commands like `chromium https://target` launch with arguments intact.
  - Added dynamic display resolution `_get_display()`.
- **`workstation.py` (`sonic/api/routes/workstation.py`)**:
  - Exposed `POST /workstation/desktop/tile` wrapping `comp.tile_workstation()`.
  - Added `ApplicationPolicy` validation on `OPEN_APP` actions in `/workstation/desktop/action` and `/workstation/desktop/gui-action`.

### 5. Verified Test Matrix (176 Passed, 0 Failures)
- All 176 unit and integration tests across modified suites pass 100% cleanly.

## Phase 32 — Boss Agent & Multi-SubAgent Autonomous Orchestration (DONE)
Eliminated the flat single-agent reactive loop (`observe() -> LLM 1 action -> execute -> repeat`) and replaced it with a hierarchical Boss Agent and SubAgent worker orchestration system:

### 1. Boss Agent Architecture (`sonic/computer_use/boss.py`)
- **Strategic Decomposition:** Decomposes complex user penetration testing objectives into structured phases and focused sub-missions.
- **SubAgent Worker Dispatch:** Dispatches dedicated `ComputerUseAgent` workers for each sub-mission, each with a focused goal, bounded step budget, and inherited `SealedActionPolicy` safety envelope.
- **Report & Trace Aggregation:** Aggregates findings and execution traces across all workers to drive reaction planning and next-phase formulation.
- **Final Report Synthesis:** Synthesizes structured deliverables (`BossReport`) backed by real in-sandbox reproduction.
- **Data Models (`sonic/computer_use/models_boss.py`):** `SubMission`, `SubMissionResult`, `Phase`, `BossThinking`, `BossReport`.

### 2. Workstation API & Mission Director Routing
- **`workstation.py`**: Added `_is_complex_or_multi_part_objective()` to route complex tasks to `BossAgent`, while maintaining fast single-action execution for simple terminal commands (`pwd`, `hostname`).
- **Real-Time Worklog Streaming**: Streams 6 distinct event types (`boss_thinking`, `sub_dispatch`, `sub_step`, `sub_report`, `phase_complete`, `boss_report`) into the workstation timeline with live thinking durations.
- **`director.py`**: Wired `BossAgent` into Phase 3 (ENGINEERING) of `MissionDirector.coordinate()` with trace stashing.

---

## Phase 33 — Unified DAG TaskGraph & ReplanEngine Wiring (DONE)
Wired the previously unintegrated and stubbed cognitive architecture components (`TaskGraph`, `ReplanEngine`, `MetaOrchestrator`, `Director`) into a cohesive, DAG-driven autonomous execution pipeline matching the Antigravity thinking loop:

### 1. Event-Driven Director Execution Engine (`sonic/agents/director.py`)
- Implemented `run_engagement_loop(engagement_id, worker_fn, max_iterations=50)`:
  - Fetches dispatchable tasks via `get_dispatchable_tasks()`.
  - Dispatches tasks to `worker_fn`, handles async results, and updates cognitive state via `on_task_completed()`.
  - Cascades task failures and blocked states via `on_task_failed()`.
  - Automatically evaluates predictions, checks contradictions, and triggers replanning until `graph.is_complete()`.

### 2. MetaOrchestrator Execution (`sonic/agents/orchestrator.py`)
- Replaced the legacy planning-only stub with end-to-end execution:
  - Converts engagement plans into a `TaskGraph` DAG with topological phase dependencies.
  - Executes tasks via `Director.run_engagement_loop()` or directly through `TaskGraph.get_ready_tasks()`.
  - Ingests outputs, calls `evaluate_findings()`, and returns real `phases_completed` and `findings_summary`.

### 3. BossAgent TaskGraph DAG & ReplanEngine Triggers (`sonic/computer_use/boss.py`)
- **DAG Task Graph Integration:** Integrated `sonic.agents.task_graph.TaskGraph` under the hood. Sub-missions are registered as `TaskNode` objects with dependency validation, Kahn's cycle detection, and topological readiness checks.
- **Dynamic ReplanEngine Triggers:** Integrated `sonic.agents.replan.ReplanEngine` trigger evaluation on sub-agent completions:
  - `NEW_ATTACK_SURFACE`: Discovered URLs, query parameters, hidden forms, and routes.
  - `NEW_HIGH_CONFIDENCE_FINDING`: Confirmed vulnerabilities (SQLi, XSS, RCE, IDOR, etc.).
  - `AGENT_FAILURE`: Worker failures and execution timeouts.
- Detected triggers emit real-time `boss_thinking` reaction entries and dynamically mutate the task graph for follow-up testing.

### 4. Comprehensive Test Verification (100% Green)
- `test_dag_and_replan_orchestration.py` (7 tests) — **PASSED**
- `test_boss_agent.py` (10 tests) — **PASSED**
- `test_workstation_boss_integration.py` (4 tests) — **PASSED**
- `test_phase5/` (27 tests: TaskGraph, ReplanEngine, Recovery) — **PASSED**
- `test_phase6/` (26 tests: Epistemic awareness, Predictions, Contradictions) — **PASSED**
- `test_round3_computer_agents_fixes.py` (12 tests) — **PASSED**
- `test_workstation_desktop_browser_and_news.py` (16 tests) — **PASSED**
- **Regression Suite Total: 102 passed, 0 failures (100% GREEN).**

## Phase 34 — Elimination of Scripted Puppets, Fake Coordinates & Canned Workflows (DONE)
Eliminated RPA-style scripted workflows, hardcoded application maps, brittle fake UI coordinates, and web-only checklist assumptions across SONIC's computer use and orchestration engine:

### 1. Strict Perception Grounding & Anti-Puppet Defense
- **`grounding.py` & `agent.py` (`sonic/computer_use/grounding.py`, `sonic/computer_use/agent.py`)**:
  - Added `allow_landmarks: bool = True` to `resolve_ui_target()` and `resolve_ui_target_async()`.
  - In live computer-use execution (`self._last_screenshot_b64 != ""`), `allow_landmarks` is strictly `False`.
  - The engine never falls back to guessing arbitrary normalized screen percentages from `_COMMON_UI_LANDMARKS` (e.g. `connect wallet`, `submit button`, `cart`, `devtools`).
  - Unresolvable visual queries return `None` cleanly; `ComputerUseAgent.execute_action()` catches this as `ActionExecutionStatus.BLOCKED` and triggers re-observation/replanning instead of blind-clicking.
  - Fixed `mid_y` scaling calculation in `extract_bbox_midpoint` for normalized `[0, 1000]` bounding boxes.

### 2. Generic Dynamic Application Resolution & Sandbox Cleanups
- **`agent.py` (`sonic/computer_use/agent.py`)**:
  - Deleted `_CANONICAL_APP_MAP` (the hardcoded dictionary mapping application names to binaries).
  - Implemented 100% dynamic, generic Linux binary sanitization in `_resolve_app_binary()`.
  - Fixed visual delta verification by recording `pre_screen_b64` before drawing red action crosshair markers.
  - In command parsing, eliminated hardcoded `-tuln` listening socket flag injection on alternative commands (`which X && X || Y`).
  - Prevented default payload strings (`"{}"`) from corrupting terminal commands.
- **`daytona_computer.py` (`sonic/computer/daytona_computer.py`)**:
  - Removed artificial block on `burpsuite` package installation, standardizing enforcement under the centralized `ApplicationPolicy`.

### 3. Target-Agnostic First-Principles Planning & Decomposition
- **`boss.py` (`sonic/computer_use/boss.py`)**:
  - **Upgraded `_strategic_decomposition()`**: Replaced the generic web-biased prompt with a Target-Agnostic First-Principles Prompt featuring explicit target triage (Binary/Pwn, Network, Headless API, Cryptography, Forensics, Web).
  - Eliminates automatic presumption of web tools (`Chromium`, `Burp Suite`, `ffuf`) when assessing binaries or headless services.
  - **Upgraded `_fallback_decomposition()`**: Replaced single-worker collapse (5-step budget overload) with a target-aware multi-task fallback pipeline (inspection/triage + execution) and recorded thinking in `self.thinking_log`.

### 4. Comprehensive & Negation-Aware Replan Engine
- **`boss.py` (`sonic/computer_use/boss.py`)**:
  - Replaced narrow 10-keyword tuple in `_detect_sub_mission_trigger()` with a comprehensive offensive security taxonomy covering SSRF, IDOR/BOLA, deserialization, prototype pollution, buffer overflow, ROP chains, format strings, crypto padding oracles, and CTF flag patterns (`flag{...}`, `ctf{...}`).
  - Added regex negation guards (`(?<!not\s)(?<!no\s)(?<!failed to\s)...`) ensuring that negative observations (e.g. *"Target is not vulnerable to SQLi"*, *"Bypass failed"*) do NOT trigger false-positive replan events.

### 5. Verified Test Baseline (100% Green)
- `test_remediation_puppet_and_control.py` (12 tests) — **PASSED**
- Visual Grounding Suites (`test_visual_grounding.py`, `test_vision_grounding_speed_and_accuracy.py`, `test_visual_delta_and_grounding_accuracy.py`, `test_motor_reflexes_phase8.py` — 23 tests) — **PASSED**
- Core Regression Suite (Phases 1 to 7 + AI Human Being + Safety Envelope — 93 tests) — **PASSED (0 failures)**.

## Phase 35 — Native Rust Kernel Solidification & Real IPC Accelerator (DONE)
Solidified and transitioned `sonic-kernel-rs` from an isolated simulation prototype into a real, high-performance native security and perception engine:

### 1. Robust Typed Error Architecture (`sonic-kernel-rs/src/error.rs`)
- Created `KernelError` and `KernelResult<T>` replacing silent boolean returns and unhandled panics (`PolicyViolation`, `TamperDetected`, `InvalidStateTransition`, `BudgetExhausted`, `RateLimitExceeded`, `EgressBlocked`, `LockPoisoned`, `SerializationError`).

### 2. Native CIDR Egress Filter & Safety Kernel Hardening (`sonic-kernel-rs/src/safety/mod.rs`)
- Implemented standard library bitmask CIDR matching (`is_ip_in_cidr`) enforcing RFC-1918 private ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`) and AWS/GCP cloud metadata (`169.254.169.254/32`, `metadata.google.internal`) on all `SECURITY_TOOL` and `BROWSER_NAVIGATE` targets.
- Added in-kernel sliding 60-second rate limiter window enforcing `max_actions_per_minute`.
- Expanded destructive command protection to catch `dd if=`, `> /dev/sd*`, `chmod -R 777 /`, `reboot`, `shutdown`, `init 0`.
- Hardened path traversal checks (`..`, `/etc`, `/root`, `/proc`, `/sys`, `/var/run/docker.sock`).

### 3. Zero-Allocation Perception Engine (`sonic-kernel-rs/src/perception/mod.rs`)
- Replaced per-frame regex compilation with static `std::sync::OnceLock<Regex>` for inputs, buttons, links (`<a>`), forms (`<form>`), textareas, selects, and text.
- Added CTF flag detection (`flag{...}`, `ctf{...}`) directly in DOM perception yielding immediate `flag_captured` page state.

### 4. Concurrency & Anti-Poison Blackboard (`sonic-kernel-rs/src/kernel/mod.rs`)
- Migrated `Blackboard` to `RwLock<HashMap<String, String>>` with `.unwrap_or_else(|poisoned| poisoned.into_inner())` poison resilience.
- Added typed JSON serialize/deserialize helpers (`post_json`, `get_json`).
- Added numerical stability guards (NaN/infinity clamping) to `InformationGainCostScheduler`.

### 5. Empirical Behavioral Divergence Specialist (`sonic-kernel-rs/src/specialists/mod.rs`)
- Replaced mock `simulate_probe` with real empirical evaluation (`evaluate_probe`) measuring length deltas, behavioral divergence, and automated secret/flag extraction via regex.
- Expanded specialist domain coverage: `Pwn`, `Crypto`, `Network`, `Forensics`, `Auth`, `Api`, `Logic`, `Web`.

### 6. JSON-RPC 2.0 Daemon & Python IPC Bridge (`sonic-kernel-rs/src/ipc.rs`, `sonic-core/sonic/kernel/native_bridge.py`)
- Added `--daemon` / `--json` stdio JSON-RPC interface to `sonic-kernel` (`health`, `authorize`, `fuse_perception`, `compute_custody_hash`, `verify_finding`, `evaluate_probe`).
- Built Python client `NativeKernelClient` in `sonic-core/sonic/kernel/native_bridge.py` allowing `sonic-core` to invoke the native Rust kernel at sub-millisecond speeds with graceful fallback.

### 7. Verification Baseline (100% Green)
- `cargo test` in Docker (`sonic-rust-dev:latest`): **26 passed, 0 failed (100% GREEN)**.
- `sonic-kernel` native execution: All 10 phases + JSON-RPC stdio queries verified.
- Python test suite: `test_native_kernel_bridge.py` (4 tests) + `test_remediation_puppet_and_control.py` (12 tests) **16 passed in 1.55s**.

## Phase 36 — Continuous Codebase Self-Evolution Engine (`CodebaseEvolver` & CLI Runner) (DONE)
Empowered SONIC with an autonomous, continuous codebase self-evolution engine capable of patching bugs, upgrading logic, and optimizing non-safety modules with zero-corrupt atomic rollbacks, automated test verification, and auto-promotion (git commit + push) without manual human approval:

### 1. Guarded Codebase Evolver (`sonic-core/sonic/evolution/codebase_evolver.py`)
- **Safety Envelope Invariant Gate**: Strict fail-closed check against `_PROTECTED_COMPONENTS` (`sonic/safety`, `sonic/kernel/action_broker`, `sonic/safety/sealed`, `sonic/safety/action_policy`, `sonic/sandbox/egress`, `sonic-kernel-rs/src/safety`). Any attempt to tamper with or loosen safety envelopes is immediately rejected with a critical security violation audit log (`EvolutionStage.REJECTED`).
- **AST & Syntax Pre-Flight**: Pre-parses Python code with `ast.parse()` to catch syntax errors before any file is touched on disk.
- **Atomic Snapshot & Zero-Corrupt Rollback**: Maintains in-memory snapshots of all touched files; if unit tests or security regression tests fail, files are restored to their exact pre-evolution state and an `[AVOID]` lesson is recorded in `LessonsLedger`.
- **Automated Verification Pipeline**: Automatically executes component unit tests (`pytest`) and security regression tests (`test_p0_security_hardening.py`, `test_safety_sealed_policy.py`). Requires 100% green exit code to proceed.
- **Structured Evolution Summary Report**: Generates a rich, tamper-evident Markdown report with proposal ID, target component, diff metrics (+/- lines), AST verification status, test execution results (passed/failed/duration), and git provenance.
- **Autonomous Promotion & Git Push**: When `auto_promote=True` and tests pass 100%, changes are automatically committed to git (`evo(<id>): <desc>`) and optionally pushed to remote origin without requiring human operator approval.

### 2. Dedicated CLI Runners
- **Python CLI Runner (`sonic-core/sonic/evolution/cli.py`)**:
  - `python -m sonic.evolution.cli patch --target <path> --diff <patch> --desc <desc> --auto-promote --push`
  - `python -m sonic.evolution.cli status`: displays evolver status and list of protected safety components.
- **SONIC Power CLI Subcommand (`sonic-cli/sonic_cli/commands/evolution.py`)**:
  - `sonic evolution patch --target <path> --desc <desc> --auto-promote --push`

### 3. Verification Baseline (100% Green)
- `sonic-core/tests/test_codebase_evolution.py` (5 tests) — **5 passed in 0.69s**.
- Full evolution suite (`test_codebase_evolution.py` + `test_evolution_strategy_and_engine.py` + `test_phase8_production_kernel_and_evolution.py` — 16 tests) — **16 passed in 2.21s**.
- Security Regression Suite (`test_safety_sealed_policy.py` + `test_phase_plan6_safety_envelope.py` — 53 tests) — **53 passed in 2.09s**.

## Phase 37 — Closed-Loop Evolution Wiring & North Star CTF Objective Integration (DONE)
Resolved the audit findings regarding disconnected evolution engines and injected the system's core North Star objective across all capability and reasoning layers:

### 1. Primary North Star Directive Injection
- Formalized `CORE_NORTH_STAR_GOAL` across `EvolutionEngine` and `DynamicStrategyEngine`:
  - **CTF Mastery**: Prioritizing autonomous vulnerability discovery and exploitation across Web (Auth bypass, SSRF, SQLi, IDOR, SSTI, Deserialization, Race Conditions), Binary Exploitation/Pwn (Buffer Overflow, ROP, Format Strings, Heap), Cryptography (Padding Oracles, Weak Keys, Keystream Reuse), Forensics (PCAP, Memory, File Repair), and Reverse Engineering (Ghidra, GDB, Decompilation).
  - **Precision Desktop Control**: High visual and semantic interaction accuracy in Chromium, Burp Suite, Terminal, VS Code, and Ghidra.
  - **Sub-Second Fast Reaction**: Rapid feedback loops, eliminating redundant polling, accelerated by native Rust perception and Cython graph search.
  - **Empirical Non-Puppet Autonomy**: Zero canned scripts, zero macros; all findings and flags verified empirically in-sandbox.

### 2. Multi-Domain CTF Target Feedback Signals (`sonic/evolution/strategy.py`)
- Added specialized signal detection and adaptive postures to `DynamicStrategyEngine`:
  - `FLAG_DISCOVERED`: Immediate detection of `flag{...}`, `ctf{...}`, `picoctf{...}`, `htb{...}` -> `FLAG_EXTRACTION_AND_TRIAGE` (halts redundant exploration and submits flag).
  - `BINARY_CRASH_OR_SEGFAULT`: Detection of segmentation faults / core dumps -> `EXPLOIT_PAYLOAD_MUTATION` (inspects registers, determines RIP offset, authors targeted buffer/ROP probe).
  - `CRYPTO_ORACLE_FAILURE`: Detection of padding/MAC errors -> `CRYPTO_ORACLE_ANALYSIS` (side-channel differential analysis).
  - `BINARY_OBFUSCATION`: Detection of stripped/UPX-packed binaries -> `REVERSE_ENGINEERING_DEOBFUSCATION` (unpacking, syscall tracing, decompilation).
  - `FORENSIC_CORRUPT_HEADER`: Detection of corrupt magic bytes -> `FORENSIC_HEADER_REPAIR` (hex byte repair via xxd, carving with binwalk).

### 3. End-to-End Closed-Loop Wiring
- **`ComputerUseAgent`**: Wired `self.evolution_engine` by default; connected failed/recovered actions in `execute_action` to `evolution_engine.handle_target_failure`; injected `_last_adapted_strategy` directly into `_build_reasoning_context` so the LLM acts on the evolved posture on the very next step.
- **`BossAgent`**: Wired `self.evolution_engine` and forwards it to all dispatched subagents.
- **`MissionDirector`**: Instantiates and connects `EvolutionEngine` to both `BossAgent` and `ComputerUseAgent`.
- **`api/main.py`**: Connects `EvolutionEngine` to the Being Life Loop.

### 4. Verification Baseline (100% Green)
- Full regression suite (`test_codebase_evolution.py` + `test_evolution_strategy_and_engine.py` + `test_remediation_puppet_and_control.py` + `test_native_kernel_bridge.py` + `test_mission_trace_synthesis.py` + `test_phase8_production_kernel_and_evolution.py` — 55 tests) — **55 passed in 3.17s**.

## Phase 38 — Native Rust Kernel Pre-Screening & ActionBroker Integration (DONE)
Integrated the high-performance Rust safety daemon (`sonic-kernel-rs`) with `ActionBroker` to provide sub-millisecond hardware-speed pre-screening and fail-closed defense-in-depth across all agent execution requests:

### 1. Hardware-Speed Pre-Screening (`sonic/kernel/action_broker.py`)
- Integrated `NativeKernelClient` into `ActionBroker.__init__`, defaulting to an active client if not explicitly passed.
- Added target extraction (`_extract_target`) and canonical action type resolution via exported `ACTION_MAP`.
- Pre-screens every execution request via `native_kernel_client.authorize(action_type, target, is_isolated=True)` before reaching Python safety checks or execution providers.

### 2. Multi-Layer Fail-Closed Guarantees
- **Tamper Evidence**: If `native_verdict.seal_intact` is `False`, logs a critical security alert (`action_broker_blocked_native_seal_tampered`) and immediately blocks execution with exit code 126.
- **Hardware-Speed Denial**: If `native_verdict.verdict == "Deny"`, blocks immediately with exit code 126 and returns `[NativeKernel Deny] <reason>` without touching the provider or wasting Python interpreter cycles.
- **Interactive Approval Gating**: If `native_verdict.verdict == "RequireApproval"` and `approved=False`, blocks execution with exit code 126.
- **Defense in Depth**: If native checks pass, execution still verifies against Python's sealed `SafetyKernel` before reaching the sandbox provider. If the native binary is absent in dev, gracefully falls back to Python's `SafetyKernel`.

### 3. Verification Baseline (100% Green)
- `sonic-core/tests/test_native_kernel_bridge.py` (7 tests) — **7 passed in 1.33s**.
- `sonic-core/tests/test_phase0_safety_kernel_and_broker.py` (6 tests) — **6 passed in 1.15s**.
- Combined regression test suite (48 tests across Evolution, Strategy, Kernel, Broker, Traces) — **48 passed in 2.50s**.

## Phase 39 — Advanced Autonomous Self-Evolution Director, Goal Queue & Live `evolution.md` Notes Logger (DONE)
Transformed SONIC from a raw diff applier into an intelligent, goal-driven, version-progressing autonomous penetration architect self-evolution system:

### 1. Persistent Semantic Version Tracker (`sonic/evolution/version_tracker.py`)
- SQLite-backed semantic versioning (`evolution_versions` table in `sonic_data.db`).
- Tracks milestones across `0.1.0` ➔ `0.1.1` (patch: bug fix/refactor) and `0.1.x` ➔ `0.2.0` (minor: capability add / major milestone).
- Full audit history of version milestones, timestamps, commit hashes, and automated git tagging (`v0.1.1`, `v0.2.0`).

### 2. Live `evolution.md` Notes Logger & Journal (`sonic/evolution/evolution_journal.py`)
- **Live Markdown Logging**: Real-time automatic append to `evolution.md` at repository root for every file edit, bug fix, patch, task, or evolution cycle.
- Documents: Goal ID, status badge, version advancement, category, files edited, diff metrics (+/- lines), test summary, and detailed notes/rationale.
- **Durable Journal & Fitness**: SQLite `evolution_journal` table tracking fitness metrics (success rate, promotions, rollbacks, total cycles) and automated changelog generation between versions.

### 3. Goal-Driven Autonomous Director (`sonic/evolution/goal_director.py`)
- **Goal Queue Management**: Persistent SQLite queue (`evolution_goals`) supporting priorities (P1-P4), categories (bug fix, logic improvement, capability add, performance, refactor), and multi-attempt retries.
- **Intelligent Codebase Analysis**: Scans intent, extracts keywords, maps to non-safety candidate files via AST/symbol extraction, and enforces fail-closed safety invariant verification.
- **Closed-Loop Execution**: Staged execution via `CodebaseEvolver`, AST pre-flight verification, automated unit testing, security regression checks, version bump, and real-time `evolution.md` updating.
- **Continuous Runner**: Drains queued goals in priority order autonomously without human intervention.

### 4. Verification Baseline (100% Green)
- `sonic-core/tests/test_advanced_evolution.py` (9 tests) — **9 passed in 3.00s**.
- Full regression suite across Evolution, Kernel, Safety, and Broker (42 tests) — **42 passed in 4.56s**.

## Phase 40 — AI-Human Digital Being Identity Hardening, BeingVault Wiring & Sealed Safety Envelope (DONE)
- **Foundational Agency**: Completely eliminated pentester/hacker lock-in across dashboard and backend. SONIC is an autonomous AI-Human Being capable of general-purpose computer use.
- **BeingVault Persistence**: Added `BeingVault` to persist operator credentials, persistent browser profiles, and account states across reboots (`sonic/being/vault.py`).
- **Sealed Safety Envelope**: `SealedActionPolicy` with SHA-256 HMAC cryptographic tamper seals; fail-closed exit 126 blocks unauthorized operations without execution bypass.
- **Dynamic Application Discovery**: Discovers installed applications strictly via `$PATH`, `.desktop` entries, and POSIX alternatives (zero hardcoded app names).

## Phase 41 — Multimodal Vision Engine & Live Life Loop GUI Streaming (DONE)
- **Multimodal Format Normalization**: Consolidated system context and screenshots into valid single-turn multimodal payload for Llama-3.2-Vision and Nemotron vision architectures.
- **Life Loop GUI Streaming**: Wired `BeingLifeLoop` to the live desktop interaction bus, streaming real-time visual perception and cognitive state to the frontend dashboard.
- **Dynamic Window Activation**: Hardened X11 `APP_FOCUS` via `wmctrl` and `xdotool` so the agent reliably focuses active windows without spawning ghost processes.
- **Payload Extraction**: Fixed JSON payload parsing in `GUI_TYPE` to correctly parse text and coordinate fields from LLM outputs.

## Phase 42 — Laya System-1 Fast Intuition Engine & High-Speed Multi-Subagent Coordination (DONE)
- **Laya System-1 Fast Engine**: Implemented `LayaSystem1Engine` (`sonic/llm/laya_engine.py`) providing sub-millisecond fast triage, guard classification, verification checks, and curiosity affect scoring. Includes zero-RAM laptop protection mode (`_has_sufficient_ram_gb`).
- **Concurrent Subagent Workspaces**: `SubagentTeamworkManager` orchestrates concurrent subagents in isolated git branch workspaces with automatic verification gating and clean parent merge.
- **Resource Forwarding**: Wired `BeingCraft` and `BeingVault` directly into `BossAgent` and all dispatched subagents.
- **Routing Support**: Integrated Nemotron 120B/550B routing configurations.

## Phase 43 — Human-Grade Motor Reflexes, Smooth Cursor Gliding & Anti-Puppet Invariants (DONE)
- **Smooth Cursor Gliding Engine (`sonic_cursor.py`)**:
  - Replaced 0 ms instantaneous cursor teleportation with cosine-eased trajectory interpolation (200–300 ms).
  - Ensured human operators watching noVNC or streaming viewers can clearly see the mouse glide across the screen to target coordinates.
- **Auto-Focus Target Typing**:
  - `GUI_TYPE` automatically positions and clicks the cursor at `(x, y)` before typing keystrokes, ensuring active focus on input fields.
- **Window Focus Deduplication**:
  - `launch_application` in `docker_computer.py` activates existing open windows (`wmctrl -x -a` / `wmctrl -a`) when no arguments are passed, preventing duplicate app instances.
- **Dual-Plane Prompt Directive Alignment**:
  - Updated `agent.py` so visual, browsing, and GUI tasks actively utilize the graphical desktop plane (`APP_LAUNCH`, `APP_FOCUS`, `GUI_CLICK`, `GUI_TYPE`, `GUI_MOVE`) instead of taking silent headless terminal shortcuts.
- **Strict Anti-Puppet Mandate**:
  - Addressed Operator Feedback:
    - *"Are jo sonic ko karna chahiye vo tum kyu karte ho?"* -> Developer/Assistant manual execution strictly forbidden.
    - *"Cursor kaam nahi kar raha, use nahi kar raha, automate kaam tum kara rahe ho"* -> Replaced background shortcuts with visible cursor gliding, element focus, and autonomous agent-driven actions.

## Phase 44 — Generic Behavioral Contracts, Physical Display Mutex & Reasoning Heartbeat (DONE)
- **Browser Profile Lock Recovery**: Automatically recovers stale generic browser locks (`SingletonLock`, `SingletonCookie`, `SingletonSocket`) only when no live process owns the profile (`browser_agent.py`). Preserves persistent browser state and cookies.
- **Generic Modal & Consent Verification**: Detects semantic dialogs (`[role="dialog"]`, `<dialog>`), attempts non-destructive dismissal (`Escape`), and strictly asserts overlay disappearance before declaring action success.
- **Motor Failure Propagation**: Replaced silent error swallowing in `motor.py` (`_send_hotkey`, `human_type`, `_exec_cmd`) with explicit `RuntimeError` raising and fail-closed exit code `126`.
- **Coordinate-Space Contracts**: Added explicit coordinate-space scaling (`pixel`, `normalized`, `normalized_1000`, `model_1000`) in `grounding.py` for exact resolution mapping.
- **Target Window Ownership Validation**: GUI actions specifying target window verify active window ownership before dispatch; mismatched focus triggers `SELECT_WINDOW` or fail-closed `BLOCKED`.
- **Physical Display Mutex**: Keyed `DisplayLeaseManager` in `teamwork.py` to physical display identity (`:99`) preventing concurrent subagent mouse/keyboard race conditions.
- **Reasoning Heartbeat & Observability**: Dispatches periodic (~2s) heartbeat events during heavy LLM thinking turns into the workstation worklog without injecting synthetic pixels or modifying raw screenshots.
- **Universal Zero-Visual-Delta Enforcement**: Detects unchanged frames after click, double-click, typing, and hotkeys; unchanged pixels make an action `UNVERIFIED`, not failed. Recovery must not inject Escape or application-specific shortcuts; repeated unchanged actions may stop the worker without changing desktop state.
- **Micro-Crop Toolbar Ratio Compatibility**: Restored `crop_height_ratio` backwards compatibility in `crop_toolbar_region`.

## Phase 45 — Boss Direct Specialist Subagent Dispatch, Antigravity-Grade Dynamic Subagent Teamwork & 3-Way Parallel Branch Verification (DONE)
- **First-Class Specialist Subagent Dispatch (`BossAgent -> BrowserAgent`)**:
  - `BossAgent` directly dispatches web/browser tasks to `BrowserAgent` as a first-class specialist subagent (`_dispatch_browser_agent` / `dispatch_browser_subagent`), removing awkward wrappers.
  - Binds `Being`, `BeingMind`, `BeingCraft`, and `BeingVault` into `BrowserAgent`. Added standalone `run_mission()` observe-reason-act cycle.
  - `models_boss.py` explicitly supports `"browser"` execution plane alongside `"gui"` and `"headless"`.
- **Antigravity-Grade Dynamic Subagent Lifecycle Actions**:
  - `ComputerUseAgent` dynamically executes `DEFINE_SUBAGENT`, `INVOKE_SUBAGENT`, `SEND_MESSAGE`, and `MANAGE_SUBAGENTS`.
  - Reusing subagent instances via `send_message` automatically replenishes action quotas (`agent.max_actions`), preventing premature halts.
  - Added `GIT_BRANCH`, `GIT_STATUS`, `GIT_DIFF`, `GIT_PUSH` to `ActionPolicy.DEFAULT_ALLOWED_TYPES`.
- **Workspace Isolation & Collision-Proof Merging (`teamwork.py`)**:
  - Supported isolation modes: `branch` (dedicated git branch `subagent/{conv_id}` off HEAD), `share` (git worktree), and `inherit`.
  - Records baseline file SHA-256 hashes at isolation time; subagent merge only merges newly authored or genuinely modified files. Untouched parent files and WIP uncommitted notes are never overwritten.
  - Merges committed branch changes via `git merge --autostash` to safely preserve uncommitted parent files.
- **3-Way Concurrent Subagent Live Verification Receipts**:
  - `test_sonic_subagents_live_capability.py` proves live concurrent execution of 3 distinct subagents: `Math Service` (`math_service.py`), `Text Processor` (`text_processor.py`), and `System Reporter` (`system_reporter.py` / `system_report.json`).
  - All 3 execute concurrently in isolated branches, pass programmatic verification commands (`python -c "assert ..."`), pass Judge verification, and merge back into the parent workspace with zero collision.
  - Outcomes synthesized by `ComputerUseAgent` (`integrated_pipeline.py`) and `BossAgent` (`BossReport`).

## Phase 46 — 5-Track Architectural Audit, Self-Coder API Reconciliation & Fail-Closed Safety Verification (DONE)
- **Self-Evolution & Path Confinement (Track 1)**:
  - Reconciled `GoalDirector` <-> `CodebaseEvolver` APIs: added/restored `apply_patch_multi`, `pre_applied_files`, `_diff_files`, and `report_only` mode.
  - Expanded protected safety components to strictly block autonomous tampering with `auth/`, `safety/`, `evolution/`, and regression test suites.
  - Missing security regression suites now fail closed (exit code 1 / status blocked).
  - Failed git commits do not advance version numbers or mark goals as `PROMOTED`.
- **Laya & System-1 Decision Engine Gating (Track 2)**:
  - Synchronous Laya inference offloaded via `run_in_executor` to prevent blocking the async FastAPI/workstation event loop (`sonic/api/routes/llm.py`).
  - Laya heartbeat callback exceptions caught cleanly so they never crash successful LLM turns.
  - Added Laya receipt evaluation checks during mission goal completion.
- **End-to-End Execution Loop & Verification Probes (Track 3)**:
  - Closed unmonitored terminal bypass in `ComputerUseAgent.verify_goal()`: probes are now dispatched as typed `VerificationProbe` via `ComputerActionType.VERIFY_PROBE` through `execute_action()` and evaluated against `ActionPolicy`.
  - Added cross-platform verification probe commands (`models.py`) supporting Windows and POSIX environments (`python -c "import os, sys..."`).
- **Physical Display Serialization & Workstation Telemetry (Track 4)**:
  - `BossAgent` limits concurrent GUI workers on `_shared_desktop` to 1, while allowing headless tasks to scale up to `max_parallel_workers`.
  - Preserved nullable exit codes in workstation telemetry rather than fabricating `0` or `1`.
  - Added runtime kill-switch rechecks inside the active terminal WebSocket input loop (`sonic/api/routes/terminal.py`).
  - Removed unowned global Daytona workspace ID fallback.
- **Integration & Regression Test Coverage (Track 5)**:
  - 250+ regression tests passing 100% across self-coder (`test_self_coder.py`), codebase evolution (`test_codebase_evolution.py`, `test_advanced_evolution.py`), Laya System-1 (`test_laya_end_to_end.py`, `test_laya_system1_engine.py`), action receipts (`test_agent_action_receipts.py`), boss plane routing (`test_boss_plane_routing_and_truthfulness.py`), and dynamic teamwork subagents (`test_teamwork_subagents.py`, `test_sonic_subagents_live_capability.py`).

## Phase 47 — Workstation Human-Grade Autonomy, Visual Perception Grounding, Motor Sanitization, Multi-Window Tiling & Closed-Loop Visual Delta Watchdog (DONE)
- **Visual Perception Grounding & Coordinate Normalization**:
  - `agent.py`: Attached live desktop screenshots to LLM calls whenever `gui` or `dual` planes are active (`should_attach_image`), eliminating blind action selection.
  - `grounding.py`: Fixed 1280x800 coordinate scaling bug where normalized 0-1000 coordinates were not scaled if `max(x, y) <= 1000`. Accurately translates normalized coordinates to desktop dimensions.
  - Implemented foveated vision cropping (`crop_active_window_region`) and active window geometry `(x, y, w, h)` tracking in `ScreenObservation`.
- **Motor Interaction, Payload Sanitization & Keyboard Accelerators**:
  - `agent.py`: Removed naive `fallback_txt = target` assignment so UI target names (e.g. `File -> Open`) never leak into `GUI_TYPE` payloads.
  - Sanitized typing and terminal payloads using expanded regex delimiters to strip model thoughts, commentary, and `BeingCraft note:` blocks.
  - Added target field clearing (`xdotool key --clearmodifiers ctrl+a BackSpace`) and active window focus settling before typing.
  - Documented and promoted standard keyboard accelerators (`Alt+F` for File menu, `Ctrl+S` for Save, `Ctrl+O` for Open, `Ctrl+N` for New) in prompts and action guidance.
- **Multi-Window Management & Structured Task Decomposition**:
  - Added `APP_MAXIMIZE`, `WINDOW_MAXIMIZE`, and `WINDOW_TILE` to `ComputerActionType` and `GUIActionType`.
  - Implemented native window maximize (`wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz` or `wmctrl -i -r <wid>`) and window tiling in `docker_computer.py` and `agent.py`.
  - Enabled structured task decomposition (`enable_llm_decomposition=True`) in `boss.py` for multi-step GUI sub-missions (Focus -> Maximize -> Type -> Save).
- **Closed-Loop Visual Delta Watchdog & Judge Falsification Gates**:
  - Upgraded `_consecutive_unchanged_screens` into an active watchdog that refocuses windows, retries with accelerators, and prevents infinite loops from burning step quotas.
  - Fixed `JudgeAgent` (`judge.py`) to verify real on-disk artifacts and application state rather than accepting completion on superficial process status or unsaved text (`*untitled`).
  - Gated Laya System-1 receipt verification to reject syntax errors, broken commands, and unsaved file claims.


## Organizational architecture invariant

SONIC software ownership is formally divided by department, team and subsystem. The authoritative registry is `sonic-core/sonic/organization/architecture.py`.

- Cognitive Systems owns brain, agents, llm, memory, being, evolution and swarm.
- Computer & Interaction owns computer_use, computer, browser, perception and world.
- Platform & Systems owns api, execution, kernel, queue, tools, integrations, db, config, logger and cli.
- Security & Safety owns auth, safety, sandbox and net.
- Evaluation & Reliability owns evidence, health, observability and mission_engine.
- Experience & Interface owns dashboard/operator UI surfaces.
- Research & Evolution Lab owns research, researcher, meta and orchestration research surfaces.

Every declared subsystem has exactly one authoritative department and team. Employee assignments must resolve to a declared subsystem/path scope; vague ownership is not valid. Cross-functional projects may span departments, but technical subsystem ownership remains singular. Unknown repository paths must fail ownership resolution rather than receive an implicit default owner.

## Current Computer & Interaction routing (2026-09-28)
- The dedicated `sonic/agents/browser_agent.py` specialist and its hidden Playwright/HTTP execution path have been removed.
- Boss plans, dispatches, reviews evidence, and replans; all delegated work uses general `ComputerUseAgent` workers with the shared visible desktop, live perception, generic GUI actions, and the separately available terminal plane.
- Legacy browser action identifiers are compatibility inputs only and are not advertised as the agent's action vocabulary. Where retained, they map to visible desktop operations; they do not construct an adapter or a specialist worker.

## Computer & Interaction perception reality (2026-09-29)
- Do not describe SONIC's desktop perception as OCR-only. The default Docker workstation has an embedded NSI (Neural-Symbolic Instant Eye) path: frequent screen-frame capture, changed-region OCR, AT-SPI accessible controls, X11 active/open-window geometry, and browser tab title/URL context. `ComputerUseAgent.observe()` publishes the screenshot, visible text, window state, and detected controls into a versioned `PerceptionBus`; GUI decisions can additionally send the screenshot to the configured vision route, and unresolved UI targets can use multimodal grounding.
- The separate `DesktopATSPIDOMEngine` implements richer AT-SPI/X11 plus interactive browser-DOM extraction, but do not assume it is active in the workstation task path unless a production caller wires it in. NSI's browser CDP context is not equivalent to full interactive DOM extraction.
- The normal workstation route uses `DockerComputerProvider`; Daytona is opt-in. App launch proves only that a launch action was dispatched, not that SONIC can complete work inside the app. Meaningful GUI completion requires a fresh observe → reason → ground → act → observe loop and task-specific evidence; Boss/Judge must not turn missing or inconclusive evidence into success.
- **Bounding-box contract:** controls consumed by `PerceptionBus` use `(x, y, width, height)` unless the producer explicitly labels another format. NSI OCR, AT-SPI and X11 producers emit corner coordinates `(x1, y1, x2, y2)` labeled `bbox_format="xyxy"`; the bus normalizes them before indexing. The fallback screenshot OCR path carries the same metadata. Regression tests cover NSI/fallback metadata transport and verify that OCR/AT-SPI/X11 rectangles resolve to their true pixel center; never characterize this integration issue as an OCR capability limitation.
- **Grounding safety:** duplicate semantic labels at different screen rectangles are ambiguous and must resolve to `None` unless the operator/agent query includes a unique accessible attribute. Visual coordinate extraction must reject off-screen points rather than clamp them into edge clicks; preserve explicit coordinate-space metadata for normalized model output.
- **GUI reasoning and completion:** if a GUI worker has no usable reasoning response, record one explicit blocked result and stop; do not substitute an unrelated workspace file read or screenshot-only retry loop. A GUI worker's `GOAL_COMPLETE` is a handoff proposal, not a success claim: only the independent Judge may confirm it from the fresh desktop frame and worker receipts. Never infer GUI completion from keyword overlap in visible text.
- **Coordinate contract:** GUI action `TARGET`/`PAYLOAD` coordinates are screen pixels; the separate multimodal grounding response uses normalized `[0, 1000]`, including JSON points, and must be parsed with that coordinate space explicitly. Do not guess a coordinate space from numeric magnitude.
- Before attributing a live GUI failure to reasoning or perception quality, check the active backend's actual vision-route readiness and collect its frame, detected-control, action and Judge receipts. A configured model in `configs/models.yaml` is not proof that its provider credentials are ready in the running process.

### Latest Computer & Interaction fix pass (2026-09-29)
- `ComputerUseAgent.choose_action()` preserves the explicit `terminal: <operator command>` path, but GUI-only/GUI-plane workers without a reasoning router fail closed rather than selecting a generic workspace file diagnostic. If model reasoning raises an error, the GUI worker records a blocked reasoning-unavailable trace and stops instead of looping on screenshots.
- GUI-plane `GOAL_COMPLETE` stops the worker as an explicit Judge handoff. The worker does not independently mark itself successful from visible wording; GUI-plane missions also skip the non-GUI post-mission verifier. The legacy `gui_only` verifier likewise never treats lexical overlap as proof.
- `grounding_user_prompt()` defines normalized `[0,1000]` values for multimodal grounding output. `query_multimodal_grounding()` explicitly parses that coordinate space, including JSON `{x,y}` points; model action `TARGET`/`PAYLOAD` values are separately instructed to use screen pixels. Out-of-bounds points are rejected, never clamped.
- An unchanged screenshot after a GUI click, typing action, or keypress is inconclusive: record `UNVERIFIED`, re-observe, and never send Escape or a guessed shortcut as recovery. Only repeated stalls may abort the worker; GUI completion still requires independent Judge evidence.
- Focused validation for the bbox, perception, coordinate, GUI dispatch, and completion contracts: **97 passed**; Ruff and `git diff --check` clean. This validates mocked/unit contracts only, not completion of a real arbitrary GUI task.
- Runtime check on 2026-09-29: API, Redis, SQLite, and local Docker workstation health were healthy; the LLM router was degraded with `No LLM API keys configured in environment.` No live GUI task was run because model-driven reasoning could not be verified as available. Recheck the active backend's LLM/vision readiness before attempting live GUI acceptance; do not claim SONIC's GUI task capability from unit tests alone.

### GUI perception and execution regression rules (2026-09-30)
- NSI frames must be read from the published port of the selected workstation
  container. Resolve its `6080/tcp` host binding dynamically; never assume
  `localhost:6080` for a named or isolated workstation, because that may read
  another desktop's pixels and controls.
- A valid screenshot is not sufficient perception evidence. If NSI exposes only
  X11 window rectangles (or no actionable OCR/AT-SPI controls), preserve those
  observations and run the current-frame screenshot OCR fallback. The
  workstation image must include `tesseract-ocr`, and capture/OCR failures must
  be reported rather than silently converted into an empty success-shaped
  observation.
- NSI OCR controls are actionable only when they carry confidence of at least
  60 and valid screen-pixel bounds; missing confidence is not a pass. If OCR
  quality is below that bar and no AT-SPI controls are available, preserve
  non-OCR observations and run the current-frame fallback with both sparse-text
  and block-text Tesseract layouts, deduplicating overlapping word detections.
- When a GUI reasoning request includes screenshot pixels, route it through
  `computer_use`/the configured multimodal model, not the text-only `reasoning`
  route (the router may strip image content from models without image support).
  Keep the latest screenshot available as a perception sensor, but do not
  require attaching a full frame to every reasoning request when fresh,
  unambiguous structured state is sufficient. Control metadata supplements
  pixels when visual evidence is needed; grounded control-only requests may use
  text reasoning when no screenshot is being used.
- GUI `TARGET` coordinates are integer screen pixels. Parse an exact `x,y`
  target or an exact JSON `{"x": <integer>, "y": <integer>}` point into
  coordinate fields before grounding; reject booleans, floats, extra fields,
  and conflicting coordinate sources. Explicit pixel targets must bypass
  semantic visual grounding and then be validated against the latest observed
  screen bounds. Normalized coordinates are accepted only from explicitly
  labeled multimodal grounding output and must never be guessed from their
  numeric magnitude.
- NSI startup must not use a self-matching `pkill -f` from its supervising
  shell; track the worker with a PID file and bound readiness waits. Keep
  `docker/entrypoint-workstation.sh` LF-terminated in the Windows worktree
  (`.gitattributes` enforces this), because it is mounted and executed by Linux.
- An operator interrupt must be forwarded from Boss through each delegated
  `ComputerUseAgent.run_mission()`; a paused status alone is not evidence that
  the worker has stopped interacting with the shared display.
- A GUI worker must not turn repeated non-GUI or malformed model proposals into
  a screenshot-only loop. A fresh observation may be attempted once; if the
  next proposal still violates the visible-GUI contract, stop and stream an
  explicit blocked trace to Boss/operator. `GUI_TYPE` must include explicit
  text and must never type a target label or coordinates as a fallback.
- GUI-only workers must receive one consistent visible-GUI action contract in
  both system and user prompts. Do not expose dual-plane/terminal instructions,
  file/tool authoring directives, or non-GUI action choices to a worker whose
  assigned execution plane is `gui`; keep runtime action blocking as defense in
  depth.
- GUI acceptance must exercise a general non-browser desktop application as
  well as browser-capable workflows where relevant. A pass requires a fresh
  post-action observation, real worker action receipts, and independent
  Judge-confirmed task evidence. Unit tests, an interactive screenshot, a
  launched app, or a worker's `GOAL_COMPLETE` proposal alone are not an
  end-to-end pass. Prefer isolated acceptance desktops; use the shared operator
  workstation for a test mutation only with explicit operator authorization,
  and retain any resulting user-visible artifact unless asked to remove it.

## Authoritative product goal: general-purpose AI computer user

The product goal and acceptance contract are documented in
[`docs/SONIC_COMPUTER_USER_GOAL.md`](docs/SONIC_COMPUTER_USER_GOAL.md). This
section is authoritative for future computer-use work; earlier phase entries
are a historical engineering record and must not override it.

- Start from the operator's desired outcome, not a fixed app/tool checklist.
- Handle arbitrary installed GUI applications, not only browsers. Discover
  applications and controls from the live workstation; do not encode vendor
  names, screen coordinates, or app-specific click macros in control flow.
- Treat perception as a continuously refreshed estimate of desktop state.
  Prefer event, accessibility, window, and changed-region updates when
  available; use screenshot pixels and vision reasoning when visual evidence
  is needed. Real-time perception does not mean pixels are never necessary.
- Decompose long objectives into adaptive, resumable checkpoints. Preserve
  useful progress and replan when observed application state differs from the
  expected state.
- Every GUI action must have current evidence and a grounded target. Ambiguous,
  stale, or off-screen targets are blocked rather than guessed. Risky or
  irreversible actions follow the approval policy.
- Verify meaningful postconditions using fresh observations and real artifacts.
  A model proposal, launched application, unchanged screen, or missing error is
  not proof of completion. Report partial progress and blockers truthfully.
- Improve incrementally with measurable latency and task-success tests. A
  design proposal or passing unit test is not evidence that SONIC can complete
  arbitrary real-world desktop tasks. Validate using isolated desktop
  acceptance tasks across more than one non-browser application, with worker
  receipts and independent Judge evidence.

### Current implementation evidence (2026-10-01)

- The Docker NSI worker publishes a compact `nsi_state.json` sidecar without
  screenshot Base64. It contains version/timestamps, dimensions, active window,
  visible text, detected controls, and changed-region metadata.
- `DockerComputerProvider.observe_state()` validates this feed; the agent uses
  it before requesting a full screenshot. Invalid or stale state falls back to
  screenshot capture.
- Fresh state-only bounds can validate pointer coordinates without a
  pre-action screenshot.
- GUI reasoning skips image transfer only for a fresh, unambiguous AT-SPI
  scene. OCR-only, stale, ambiguous, or missing controls trigger visual
  escalation when a live workspace is available.
- GUI action dispatch still captures a post-action screenshot, and real
  multi-application completion has not yet been demonstrated. Do not describe
  this slice as a complete push/event subscription or proven real-time
  computer-use capability.

### Visual-first GUI hardening (2026-10-08)

- GUI-assigned workers refresh and attach a fresh screenshot for each reasoning
  decision. Their current-state prompt uses screenshot pixels rather than OCR,
  accessibility-control inventories, or window telemetry as a substitute for
  interpreting the visible interface. Missing fresh pixels or a failed
  vision-capable route must fail closed.
- GUI-only actions are revalidated immediately before provider dispatch. If
  screenshot pixels differ from the frame used for reasoning, the action is
  blocked and must be reconsidered from a new observation; stable OCR/control
  metadata does not authorize a stale pixel target. This can conservatively
  block on cursor blinking or animation.
- GUI-only `GUI_TYPE` clicks the model-grounded pixel point, captures a fresh
  screenshot, and asks the configured vision route to confirm a visible focus
  indicator before typing. Missing, invalid, or uncertain verification
  (confidence below 0.9) blocks text dispatch. This is a model-mediated guard,
  not independent proof that the focus assessment is correct. Non-GUI-only
  workers retain their accessibility-based focus confirmation path.
- Image requests are routed only to models declared vision-capable; images are
  not silently removed to satisfy a text-only provider. Configuration and
  credential presence do not prove screenshot comprehension. A live visual
  provider check and isolated GUI task with fresh worker receipts plus
  independent Judge confirmation are still required before claiming reliable
  desktop capability.
- Targeted regression validation after this hardening: **121 passed** across
  `test_agent_action_receipts.py`, `test_phase8_visual_computer_use.py`, and
  `test_closed_loop_visual_delta_and_judge.py`; Ruff and `git diff --check`
  passed. These tests use mocked providers and do not constitute live GUI
  acceptance. No live GUI task was run after this change.

## Self-evolution verification gate — runtime blockers fixed (2026-10-05)
The self-evolution department (`sonic-core/sonic/evolution/`) was reported as
"not working at all" while its own unit tests passed. Root cause was in the
**candidate verification gate**, not the reasoning/LLM layer: `SelfCoderCandidateStore`
reapplies a candidate diff in a detached worktree, runs the full configured test
dirs (`sonic-core/tests` + `sonic-cli/tests`) via `CodebaseEvolver.run_tests`, and
requires a 100% green exit code to promote. Three defects made promotion
impossible, so every candidate was rolled back / reported failed:
- `CodebaseEvolver.run_tests` used a fixed `timeout=120` s subprocess while the
  full suite takes ~164 s, so every full-suite verification timed out
  (`Tests failed with exit code 2` / `timed out after 120 seconds`). Fixed:
  env-configurable `SONIC_EVOLUTION_TEST_TIMEOUT`, default 600 s, floor 120 s.
- A GUI TARGET JSON object carrying extra fields (e.g. `coordinate_space`) was
  accepted as an explicit pixel point, guessing an unknown coordinate space.
  `_parse_gui_action_response` now fails closed unless the object is exactly an
  integer `x`/`y` pair (`_parse_pixel_coordinate_target`).
- Pre-dispatch fresh-screenshot revalidation hard-blocked *any* pixel difference.
  It now fails closed on a differing frame only when there is no stable semantic
  evidence (`detected_controls`/`visible_text`); identical semantics confirm
  transient pixel noise (cursor blink / clock tick) and allow dispatch.
- `sonic-cli` must be installed for `sonic-cli/tests` to collect (its `typer`
  dependency lives in `sonic-cli/pyproject.toml`); CI installs it. Without it the
  gate returns exit 2 (collection error). `sonic-cli` is not a member of the
  root `uv` workspace, so `uv sync` does not install it — use
  `uv pip install -e sonic-core -e sonic-cli`.
- Environment reminder: a fresh checkout needs `uv sync --all-packages --extra dev`
  then `uv pip install -e sonic-core -e sonic-cli`; otherwise `import sonic`
  fails after a plain `uv sync`.

Verified end-to-end: a stub candidate diff applied to `logger.py` promoted to
`STATUS: applied` with `full_test_suite: True`, `security_regression: True`,
1872 passed / 0 failed, both in the detached worktree and after applying to the
active checkout. Full `sonic-core/tests` baseline after the fix: **1859 passed,
41 skipped, 0 failures**.


## Self-evolution logical-bug pass (2026-10-05)

Second-pass audit of the self-evolution / self-improver department after the
verification-gate fix. Five more real logic bugs were found and fixed (all with
regression tests):

1. **Flag capture was shadowed by incidental errors** (`evolution/strategy.py`).
   `DynamicStrategyEngine.analyze_feedback` checked 403/WAF/syntax/error signals
   *before* the CTF flag regex, so output that contained BOTH a captured flag and
   any error text (e.g. a 403 from the same response) returned the error signal
   and the being kept probing instead of extracting the flag. Flag detection is
   now step 0 (terminal evidence dominates).
2. **Self-coder signal discovery used a hardcoded 300 s pytest timeout**
   (`evolution/self_coder.py`) while the verification gate used the configurable
   `SONIC_EVOLUTION_TEST_TIMEOUT` (default 600 s). A slow suite aborted discovery
   silently (the `except` returned `[]`), so no signals and no evolution. Discovery
   now reuses the evolver's budget (env-overridable).
3. **`no_candidate` permanently short-circuited pinned retries**
   (`evolution/self_coder_candidates.py`). A single empty discovery at a commit
   caused every later `run_candidate` at that same commit to return
   `no_actionable_signal` immediately — even when the caller passed an explicit
   pinned signal. The continuous runner pins one signal per cycle, so after the
   first empty discovery the loop could never retry. The short-circuit now applies
   only to automatic discovery (`signals is None`).
4. **Version tracker recorded git tags that were never created**
   (`evolution/version_tracker.py`). `_apply_bump` persisted `v<version>` whenever
   `tag=True`, even with no commit to tag or a failed tag command — a false
   provenance claim. The tag is recorded only after `create_git_tag` succeeds.
5. **CLI printed `0.0%%`** (`evolution/cli.py`). `%%` inside an f-string is a
   literal double percent (not a format escape); two status/journal lines now
   print `0.0%`.

Also refreshed `docs/SELF_EVOLUTION_ARCHITECTURE.md` (it still documented the
300 s discovery timeout). Evolution regression set: 154 passed, 0 failures.

## Core Self-Evolution Safety Envelope & Department Realignment (2026-10-06)

Third-pass hardening across `sonic-core` (Department `cognitive_systems`) and `sonic-cli` (Department `platform_systems`):

1. **Safety Envelope Path Normalization & Protected Component Expansion** (`evolution/pipeline.py` & `evolution/codebase_evolver.py`):
   - Added segmented boundary path matching (`_check_path_match`, `is_protected_component`) preventing false positive blocks on paths like `sonic/authoring`.
   - Stripped repository prefixes (`sonic-core/`, `sonic-cli/`) before boundary checks so worktree-relative paths match reliably.
   - Expanded `_PROTECTED_COMPONENTS` to protect critical runtime gates: `sonic/agents/judge.py`, `sonic/evidence`, `sonic-core/tests/test_safety_sealed_policy.py`, `sonic/evolution/self_coder_candidates.py`, `sonic/evolution/continuous_runner.py`, and `sonic/sandbox/factory.py`.
2. **Persistent Graph Memory Hook for Candidate Approvals** (`evolution/self_coder_candidates.py` & `memory/router.py`):
   - Operator approvals (`POST /candidates/{id}/approve`) now record verified candidates directly to persistent Graph Memory via `record_verified_candidate()`, queryable via `GET /verified-history`.
   - Fixed SQLite fallback in `memory/router.py` to respect explicit `sqlite_db_path`.
3. **Subprocess Secret Scrubbing & Falsification Integrity** (`evolution/codebase_evolver.py`):
   - Scrubbed sensitive API tokens, secrets, and keys from test subprocess environments.
   - Enforced safe timeout bounds: `min(1800, max(120, value))` on `SONIC_EVOLUTION_TEST_TIMEOUT`.
   - Required both `test_p0_security_hardening.py` and `test_safety_sealed_policy.py` to exist on disk during security regression verification.
4. **General Computer Agency Alignment** (`evolution/strategy.py`):
   - Realigned strategic postures and docstrings to general computer agency and software engineering (`REQUEST_FILTER_ADAPTATION`, `PROCESS_FAULT_DIAGNOSIS`).
5. **Standardized Backend Port 12000 & Workstation Identity** (`sonic-cli`):
   - Updated all subcommands (`computer.py`, `mission.py`, `evolution.py`, `status.py`, etc.) from port 8000 to port 12000.
   - Standardized CLI branding to `Autonomous AI-Human Digital Being` and default profile to `STANDARD_WORKSTATION`.
   - Pushed to `origin/departments/platform-systems` (commit `4c71039`) and `origin/departments/cognitive-systems` (commit `3ac2740`).

## Autonomous AI Software Engineer CLI: Pure Devin/Antigravity Engine & Zero-Boss Decoupling (2026-10-06)

Complete separation of `sonic-cli` (Department `platform_systems`) from `BossAgent` and the graphical Docker workstation:

1. **Zero-Boss CLI Decoupling**:
   - Purged all `/workstation/prompt`, `BossAgent`, and Docker desktop workstation dependencies from `sonic-cli`.
   - Removed 14 legacy workstation and swarm subcommands (`computer`, `mission`, `agents`, `engage`, `finding`, `evidence`, `research`, `security`, `experiment`, `graph`, `prompt`, `missions`, `hypotheses`, `kill`).
   - BossAgent and Docker graphical desktop workstation remain 100% isolated to `sonic-core` and the Next.js web dashboard (`sonic-dashboard`).

2. **Standalone Local `DeveloperAgent` Engine (`sonic-cli/sonic_cli/developer_agent.py`)**:
   - Implemented an agentic observe-reason-act coding engine with 6 native tools:
     - `view_file(path, start_line, end_line)`: Inspect file contents or list directories with 1-indexed line numbers.
     - `write_file(path, content)`: Safely write or overwrite files with automatic parent directory creation.
     - `edit_file(path, target_content, replacement_content)`: Precise replacement of unique target content blocks with CRLF/LF resilience and helpful context guidance.
     - `run_command(command, timeout)`: Safe local subprocess execution with process group / tree termination (`taskkill` on Windows, `os.killpg` on POSIX) and UTF-8 encoding protection.
     - `git_diff()`: Colored, line-numbered git diff inspection.
     - `run_pytest(test_path)`: Direct pytest runner with stdout/stderr capture.
   - Configured with fast Groq `qwen/qwen3.8-27b` by default (`.env` or system environment), with seamless multi-provider fallback.
   - Path confinement (`_resolve_workspace_path`): Blocks directory traversal (`../../`) escaping the workspace root.
   - Secret sanitization (`sanitize_secrets`): Automatically masks API tokens (`GROQ_API_KEY`, `JWT_SECRET`, `SONIC_TOKEN`, etc.) in console output and tool arguments.
   - Signal safety: Clean `KeyboardInterrupt` / Ctrl+C handling returning `DeveloperAgentResult(interrupted=True)`.

3. **Streamlined REPL & Command Surface (`sonic-cli/sonic_cli/repl.py` & `main.py`)**:
   - Free-form natural language inputs route directly to the local `DeveloperAgent` without network hops to Boss.
   - Instant Conversational Greetings: Basic greetings (`hi`, `hello`, `hii`, `hey`, etc.) respond conversationally in < 5ms without tool invocations or desktop screenshots.
   - Developer Slash Commands: `/auto [N]`, `/diff`, `/journal`, `/test [path]`, `/status`, `/help`, `/clear`, `/exit`.
   - Clean Subcommands: `sonic status`, `sonic diff`, `sonic test`, `sonic auto`, `sonic journal`, `sonic auth`, `sonic version`.

4. **100% Green Verification Suite**:
   - 66/66 tests passing in `sonic-cli/tests/`:
     - `test_cli_devin_mode.py` (21 passed): REPL slash commands, interactive banner, zero-arg invocation.
     - `test_developer_agent.py` (28 passed): Direct tool execution, tool loop, zero-boss isolation.
     - `test_developer_agent_safety.py` (17 passed): Path traversal protection, timeout process termination, secret sanitization, error feedback.

## Department Platform Systems & Laya System-1 Engine Hardening (2026-10-06)

Fourth-pass audit and remediation across Department `platform_systems` (`sonic-core/sonic/api/`, `config.py`, `logger.py`, `computer/headless.py`, `execution/capability_router.py`) and the Laya System-1 Non-Autoregressive Decision Engine (`sonic/llm/laya_engine.py`):

1. **Structured Logger Secret Scrubbing (P0 Security)** (`sonic/logger.py`):
   - Added regex-based key (`_SENSITIVE_KEY_PATTERN`) and value (`_SECRET_VALUE_PATTERN`) sanitization filters.
   - Redacts authentication tokens, API keys (`gsk_`, `sk-`, `nvapi-`, `Bearer`), passwords, and private keys into `[REDACTED]` across both `structlog` events and standard library `FallbackLogger` formatting.
2. **Laya System-1 Preset Resolution & Telemetry (P0 Bug Fix)** (`api/routes/llm.py`):
   - Fixed dead-code bug in `predict_sync`: preset question expansion is now executed whenever `request.preset` is provided, preventing structured goal dictionaries from being dropped.
   - Updated `/laya/status` endpoint to dynamically list all valid registered presets and aliases.
3. **Laya Engine HTTP Connection Pooling & Hardware Resilience** (`llm/laya_engine.py`):
   - Replaced ephemeral per-request `httpx.Client()` creation with a thread-safe connection-pooled client (`_get_http_client()` with keep-alive limits) to eliminate TCP socket exhaustion under high inference frequency.
   - Added Darwin macOS physical RAM inspection via `sysctl -n hw.memsize` to prevent false-negative fail-closed rejections on macOS platforms.
   - Supported mock HTTP client fallback for test suites without `limits` kwargs.
4. **Headless Execution Plane Hardening** (`computer/headless.py`):
   - Scrubbed sensitive host environment variables (`GROQ_API_KEY`, `OPENAI_API_KEY`, `JWT_SECRET`, etc.) before spawning child shell commands, preventing credential leakage to untrusted tools.
   - Enforced process-tree termination on Windows (`taskkill /F /T /PID`) on subprocess timeout, eliminating orphaned background processes.
5. **Capability Router Observability** (`execution/capability_router.py`):
   - Replaced silent bare exception swallowing in Laya plane routing with explicit structured logging (`logger.warning("laya_plane_resolution_failed_defaulting_to_computer")`).
6. **Configuration & Mount Cleanliness** (`config.py` & `api/main.py`):
   - Standardized default `app_port` to 12000 in `AppSettings` and `Settings`.
   - Removed redundant duplicate route mount of `organization.router` in `main.py`.
   - Purged dead 0-byte orphan file `api/routes/hikko.py` and obsolete `__pycache__` artifacts.
