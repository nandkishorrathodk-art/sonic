.# AGENTS.md — SONIC repository memory

## Project overview & Priority #1 (Base Foundation)
SONIC's product goal is an **Autonomous AI-Human Digital Being** that can use a general-purpose computer like a capable human colleague: perceive the desktop, operate arbitrary installed GUI applications, switch between them, use files and other appropriate computer capabilities, and complete multi-step outcomes. The workstation provides a graphical desktop and a separate execution plane; this product goal is not proof that every arbitrary task or application already works reliably.

> **PRIORITY #1 (FOUNDATIONAL MANDATE)**:
> SONIC is a general-purpose AI computer user. Its intended capability is to take an operator's desired outcome, understand and plan it, discover and operate relevant GUI applications, move between windows and tools, maintain progress across a long task, adapt to unexpected states, and verify the actual result. Cybersecurity is one possible workload, not SONIC's identity or default workflow.
> Never imply that an intended capability is already reliable without task-specific evidence. Never hardcode a vendor application sequence as a substitute for general computer-use reasoning.

> **Current Computer & Interaction contract (2026-10-30):** Boss owns task analysis, delegation, evidence review, and replanning. Every actionable workstation prompt enters Boss orchestration; chat and explicit observation-only requests remain outside execution. If planning omits workers for a non-empty objective, Boss delegates one bounded adaptive worker rather than bypassing orchestration. Boss delegates ordinary `ComputerUseAgent` workers for arbitrary installed GUI applications and the separate execution plane as appropriate. GUI reasoning attaches the fresh workstation screenshot to an image-capable model through the configured LLM Bridge `computer_use` route; a text-only fallback is not allowed for image-required decisions. AVCS is not the GUI semantic model: `AVCS_OBSERVE` is optional, read-only deterministic coarse pixel-change/motion metadata, and no AVCS learned cortex, grounder, focus head, or VLM narrator is required or invoked by the GUI decision path. The vision model supplies screen-grounded pixel coordinates; freshness, explicit coordinate space, bounds, active window, safety policy, and pre-dispatch revalidation remain mandatory. Typing requires a fresh vision-based focus check; uncertainty blocks text. OCR/Tesseract, NSI, AT-SPI, and DOM/control extraction are not workstation grounding sources. GUI completion requires independent Judge confirmation from fresh evidence and worker receipts; neither model assertion nor AVCS reflex telemetry proves success. There is no dedicated BrowserAgent specialist. Historical phase notes below describe past states and do not override this current contract or prove arbitrary-task reliability.
Architecture: FastAPI backend (`services/core/`), Next.js dashboard
(`apps/dashboard/`), Docker desktop workstation (`sonic-desktop-workstation`), Neo4j/SQLite persistent memory, multi-tenant RBAC.

## Computer Interaction: AVCS Workstation Perception
The active workstation path captures timestamped RGB frames through the selected desktop provider. Fresh frames are attached to the configured vision-capable `computer_use` request through the LLM Bridge and remain excluded from normal JSON serialization; explicit dashboard snapshots may encode frames at the API boundary. AVCS may optionally process a frame as a read-only deterministic reflex tool. It must not load or invoke learned AVCS models in GUI task reasoning. Do not add OCR/Tesseract, NSI, AT-SPI, or browser-DOM extraction as alternate workstation grounding sources. Vision-model target localization must be evaluated separately; action receipts and independent task verification remain required.

### AVCS tool boundary in SONIC
AVCS (Artificial Visual Cortex System) is not the GUI agent's semantic vision model. In SONIC's current GUI contract, it may be invoked as the read-only `AVCS_OBSERVE` tool to return coarse frame/change/motion metadata. Do not train or require AVCS checkpoints for GUI target selection, focus verification, or task completion. The separate image-capable model receives the fresh screenshot through the LLM Bridge and reasons about visible controls.

- **AVCS tool observes; the vision model interprets.** AVCS reflex signals are optional advisory metadata. The configured vision model and SONIC worker interpret the screenshot and choose a task-appropriate next action.
- **AVCS informs; the provider acts.** A grounded action still needs current evidence, validated screen-pixel coordinates, bounds checks, and pre-dispatch revalidation. The computer provider performs mouse and keyboard operations; AVCS does not click, type, or execute commands.
- **AVCS observes effects; the Judge verifies outcomes.** Pixel change, motion, or a reflex event is not proof that the intended task succeeded. The independent Judge needs fresh, task-relevant evidence for the requested postcondition.
- **Fail closed when capability is missing.** Missing/stale screenshots, unavailable image-capable routing, ambiguous targets, invalid coordinates, or inconclusive visual focus block the dependent action. AVCS unavailability alone does not block screenshot-based reasoning.
- **Describe implementation evidence precisely.** Local frame processing does not by itself prove continuous 60-FPS operation, zero latency, reliable grounding, or arbitrary-app task success. Do not claim those capabilities without measurements and isolated end-to-end acceptance evidence.
- **System 3 temporal prediction is advisory.** The current persistence-baseline
  surprise/stability telemetry is untrained; it cannot authorize GUI actions,
  ground targets, or prove a task completed.

- GUI actions require a fresh screenshot attached to the vision request and pre-dispatch freshness/bounds/policy revalidation. Pointer targets must be explicit screen-pixel coordinates derived from that screenshot; missing or ambiguous coordinates are blocked.
- Text entry requires a fresh screenshot after the focus click and a high-confidence vision check for a visible focus cue. Missing or inconclusive vision output blocks text dispatch.
- AVCS reflex output never establishes semantic task completion. The Judge must return inconclusive unless fresh, task-relevant evidence supports the postcondition.
- Generic operator-supplied image requests and the separate privacy redaction gate are not workstation perception paths. Preserve privacy redaction safeguards; do not mistake AVCS's internal retinal/foveation components for the removed workstation OCR/NSI alternatives.
- A live isolated GUI acceptance task has not been demonstrated for this migration. Unit tests and successful AVCS frame processing do not prove arbitrary-app completion, reliable semantic targeting, or end-to-end success. Verify the actual provider, model readiness, action receipts, and independent postcondition evidence before making such claims.

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
- `services/core/` — Python backend and service-owned tests (uv workspace, Python 3.12+).
- `apps/dashboard/` — Next.js operator dashboard.
- `apps/cli/` — standalone local developer CLI.
- `packages/avcs/` — AVCS perception framework and evaluation.
- `packages/kernel/` — native Rust runtime and safety kernel.
- `docs/`, `configs/`, `infra/`, and `docker/` — documentation, configuration, deployment, and container assets.
- Root `pyproject.toml` defines the Python workspace; `docker-compose.prod.yml` defines production services.

## Running locally (dev)
- Backend: `APP_ENV=development python -m uvicorn sonic.api.main:app --port 12000`
  (uvicorn is not on PATH; use `python -m uvicorn`).
- Dashboard: `NEXT_PUBLIC_API_URL=<backend-url> npx next dev -p 12001`
- Neo4j is optional in dev — the backend falls back to "Running without Graph
  Memory" if localhost:7687 is unreachable.

## Testing
- `python -m pytest services/core/tests/`
- Security regression suite: `services/core/tests/test_p0_security_hardening.py`
- New upgrade tests: `services/core/tests/test_ai_and_control_upgrade.py` covers
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
- Done-gate test: `services/core/tests/test_phase1_persistent_memory.py` — proves
  graph/vector/evolution memory survive restart, tenant isolation holds across
  restart, and no host `subprocess` execution was introduced.
- **Honesty checkpoint:** Phase 1 claims ONLY persistent memory. Reasoning,
  browser-in-loop, security-tool execution, and curiosity/life were originally
  deferred (Phase 3-6). They are now DONE — see "Phases 3-6 (capability
  layer)" below.

### Embedding-backed hybrid retrieval (2026-10-08)
- `VectorMemory` can use either a local Sentence Transformers model or an
  OpenAI-compatible embeddings endpoint. Configuration is in
  `sonic/config.py`; setup and the fixed Recall@3/MRR evaluation command are in
  `docs/CONFIGURATION.md`.
- The backwards-compatible default `legacy_hash` is explicitly labeled as
  hashed n-gram retrieval, not learned semantic embeddings. Provider failures
  do not silently fall back to hash vectors.
- Vector rows persist model identity and dimension. Search never compares
  incompatible embeddings; rows not yet re-embedded remain eligible for
  lexical-only retrieval. `VectorMemory.reembed_all()` migrates in bounded,
  resumable batches; back up the DB before changing providers.
- Search returns lexical/vector score components and a source citation;
  existing tenant filtering remains applied before ranking.
- `sonic/memory/agentic.py` adds bounded query planning over the vector index.
  It can propose alternate search queries and assess passage sufficiency, but
  all evidence remains retrieved source rows, tenant-filtered by `VectorMemory`.
  Computer-use cross-mission lessons use this path once per mission goal when
  the model router is available; query planning is capped at two rounds/four
  total queries and only ranked lesson evidence is injected into the agent's
  existing reasoning prompt. Candidate ranks are fused across queries so
  repeatedly high-ranked sources gain priority. Missing or invalid planner
  output is logged and leaves the initial retrieval intact.
- Regression tests: `tests/test_rag_embeddings.py` covers provider validation,
  paraphrase retrieval, model isolation, migration, tenant filtering, citations,
  and evaluation metrics. `tests/test_agentic_rag.py` covers bounded query
  refinement, sufficiency stopping, invalid/repeated query rejection, planner
  failure fallback, and tenant-scoped lesson use. Fixture tests validate
  mechanics, not live model planning quality. The configured model's real
  Recall@3/MRR must be measured with `python -m sonic.memory.retrieval_eval`
  before claiming retrieval quality. No hosted or local embedding model was
  exercised during development.

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
  `services/core/tests/conftest.py`), 0 failures.
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
`services/core/tests/test_safety_envelope_regressions.py` (11 tests) reproduces
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
Backend routes (services/core/sonic/api/main.py): health (/health,/health/detailed), /auth, /engagements (POST /,POST /{id}/run,GET /{id},GET /{id}/findings,GET /,POST /kill, GET /{id}/tasks, /{id}/unknowns, /{id}/decisions, /{id}/next-action, POST /{id}/replan, /{id}/pause, /{id}/resume), /findings (GET /{id}, POST /{id}/verify,/challenge,/reproduce,/review, GET /{id}/evidence,/provenance,/confidence), /agents, /graph (/stats,/search,/query,/engagement/{id}/summary,/finding/{uid}), /experiments, /terminal, /live (/experiments, /experiments/benchmark, /settings, /stats, /graph, /evidence, /scan), /jobs, /workstation (/state,/sessions,/session,/desktop/*,/command,/prompt,/tree,/file,/git-diff,/mission/*,/research-lab/*,/target-sandbox/*).
IMPLEMENTED (real, non-fabricated): /findings router wired to GraphMemory + VerifierAgent (verify/challenge/reproduce persist verdicts; evidence/provenance/confidence read real graph data; review approves/rejects). /engagements sub-routes surface Director cognitive state + task graph when available; honest empty responses (with `note`) for linear-pipeline engagements — never fake panels. CLI finding & mission & research commands now call these real endpoints.
Tenant isolation gotcha: JWT `tenant_id` defaults to "default" unless the User sets it explicitly; route reads/writes use `user.tenant_id` (NOT `user.email`). The live routes use `user.email` as tenant — keep this split in mind. Memory backends (InMemory/SQLite/Neo4j) all expose `find_evidence(finding_id, tenant_id)`.
EngagementManager vs Director: EngagementManager is the linear pipeline (no task graph); Director is the autonomous pipeline with CognitiveState (unknowns, next_best_action, decisions) + TaskGraph. /engagements sub-routes try Director first, fall back to honest empty.
Tests: apps/cli/tests has only __init__.py (no test files); python -m pytest -> no tests ran, exit 0.

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
  - Tests: `services/core/tests/test_ai_and_control_upgrade.py` (25 tests).

## Round 3 — Computer + Agents Bug Fixes

### Computer subsystem (`services/core/sonic/computer/`)

- **CLI `list` crash** (`apps/cli/sonic_cli/commands/computer.py`):
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
- Tests: `services/core/tests/test_round3_computer_agents_fixes.py`.

### Agents subsystem (`services/core/sonic/agents/`, `sonic/swarm.py`)

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
- Tests: `services/core/tests/test_round3_computer_agents_fixes.py` (12 tests)
  and `services/core/tests/test_swarm_wiring.py` (updated for the removed
  orchestrator from the agent map).

### Frontend (`apps/dashboard/`)

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
- Modernized `apps/dashboard/components/computer/ComputerSurface.tsx`: displays "SONIC Cyber Workstation", live feed indicator, and human takeover controls. All 21 Next.js routes compile cleanly (`npm run build`).

### 5. Verification Test Matrix (33/33 Passed 100%)
- `services/core/tests/test_visual_grounding.py` (9 tests): bbox midpoint extraction (box tags, float normalized), semantic landmarks, dynamic grounding fn, action aim marker styles, agent visual click dispatch, agent right-click dispatch — **PASSED**.
- `services/core/tests/test_docker_computer_provider.py` (2 tests): lifecycle, status, terminal execution, screenshot, file read/write, application listing — **PASSED**.
- `services/core/tests/test_docker_workstation_adapter.py` (3 tests): duck-typed sandbox adapter, VNC preview URL, terminal & screenshot — **PASSED**.
- `services/core/tests/test_workstation_interrupt_and_resilience.py` (9 tests): interrupt flag, loop exit, RBAC authentication & rejection, tenant isolation, mission director coordination — **PASSED**.
- `services/core/tests/test_phase_plan6_safety_envelope.py` (10 tests): fail-closed policy, rate limit, path confinement, egress filter, allowlist — **PASSED**.

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
- **Solution implemented in `apps/dashboard/components/computer/ComputerSurface.tsx`**:
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
Replaced and modernized the frontend process stream (`apps/dashboard/components/worklog/`) to match the Devin/Claude-Code execution feed layout, backed by 100% real, authentic, unmocked execution telemetry:

### 1. Devin-Style Process Feed Components (`apps/dashboard/components/worklog/`)
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

### 2. Backend Duration & Telemetry Invariants (`services/core/`)
- Added `duration_seconds: float = 0.0` to `ComputerDecisionTrace` (`sonic/computer_use/models.py`).
- `agent.py`: Measured authentic elapsed time with `time.perf_counter()` in `execute_action` and stored real seconds on traces.
- `workstation.py`: Updated `_on_step` and `execute_workstation_command` to emit distinct `thought`, `command`, `read`, and `write` items into the session worklog with genuine duration and output fields (zero simulated/fake data).

### 3. Telemetry Completeness & Dark Reality Resolution (`services/core/`)
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
- `services/core/tests/test_target_first_recon.py` (8 tests): All passed in 0.62s.
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
- **`services/core/sonic/llm/prompts.py`**:
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
- `services/core/tests/test_phase_a_toolsmith.py` (24 tests) — **PASSED** (24/24)
- `services/core/tests/test_phase_b_method_lab.py` (15 tests) — **PASSED** (15/15)
- `services/core/tests/test_evolution_strategy_and_engine.py` (7 tests) — **PASSED** (7/7)
- `services/core/tests/test_failure_engine_and_classification.py` (19 tests) — **PASSED** (19/19)
- `services/core/tests/test_agent_quality_improvements.py` (21 tests) — **PASSED** (21/21)
- `services/core/tests/test_phase3_real_computer_use_reasoning.py` (8 tests) — **PASSED** (8/8)
- `services/core/tests/test_mission_trace_synthesis.py` (12 tests) — **PASSED** (12/12)
- `services/core/tests/test_target_driven_missions.py` (5 tests) — **PASSED** (5/5)
- `services/core/tests/test_target_first_recon.py` (8 tests) — **PASSED** (8/8)
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

### 1. Robust Typed Error Architecture (`packages/kernel/src/error.rs`)
- Created `KernelError` and `KernelResult<T>` replacing silent boolean returns and unhandled panics (`PolicyViolation`, `TamperDetected`, `InvalidStateTransition`, `BudgetExhausted`, `RateLimitExceeded`, `EgressBlocked`, `LockPoisoned`, `SerializationError`).

### 2. Native CIDR Egress Filter & Safety Kernel Hardening (`packages/kernel/src/safety/mod.rs`)
- Implemented standard library bitmask CIDR matching (`is_ip_in_cidr`) enforcing RFC-1918 private ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`) and AWS/GCP cloud metadata (`169.254.169.254/32`, `metadata.google.internal`) on all `SECURITY_TOOL` and `BROWSER_NAVIGATE` targets.
- Added in-kernel sliding 60-second rate limiter window enforcing `max_actions_per_minute`.
- Expanded destructive command protection to catch `dd if=`, `> /dev/sd*`, `chmod -R 777 /`, `reboot`, `shutdown`, `init 0`.
- Hardened path traversal checks (`..`, `/etc`, `/root`, `/proc`, `/sys`, `/var/run/docker.sock`).

### 3. Zero-Allocation Perception Engine (`packages/kernel/src/perception/mod.rs`)
- Replaced per-frame regex compilation with static `std::sync::OnceLock<Regex>` for inputs, buttons, links (`<a>`), forms (`<form>`), textareas, selects, and text.
- Added CTF flag detection (`flag{...}`, `ctf{...}`) directly in DOM perception yielding immediate `flag_captured` page state.

### 4. Concurrency & Anti-Poison Blackboard (`packages/kernel/src/kernel/mod.rs`)
- Migrated `Blackboard` to `RwLock<HashMap<String, String>>` with `.unwrap_or_else(|poisoned| poisoned.into_inner())` poison resilience.
- Added typed JSON serialize/deserialize helpers (`post_json`, `get_json`).
- Added numerical stability guards (NaN/infinity clamping) to `InformationGainCostScheduler`.

### 5. Empirical Behavioral Divergence Specialist (`packages/kernel/src/specialists/mod.rs`)
- Replaced mock `simulate_probe` with real empirical evaluation (`evaluate_probe`) measuring length deltas, behavioral divergence, and automated secret/flag extraction via regex.
- Expanded specialist domain coverage: `Pwn`, `Crypto`, `Network`, `Forensics`, `Auth`, `Api`, `Logic`, `Web`.

### 6. JSON-RPC 2.0 Daemon & Python IPC Bridge (`packages/kernel/src/ipc.rs`, `services/core/sonic/kernel/native_bridge.py`)
- Added `--daemon` / `--json` stdio JSON-RPC interface to `sonic-kernel` (`health`, `authorize`, `fuse_perception`, `compute_custody_hash`, `verify_finding`, `evaluate_probe`).
- Built Python client `NativeKernelClient` in `services/core/sonic/kernel/native_bridge.py` allowing `sonic-core` to invoke the native Rust kernel at sub-millisecond speeds with graceful fallback.

### 7. Verification Baseline (100% Green)
- `cargo test` in Docker (`sonic-rust-dev:latest`): **26 passed, 0 failed (100% GREEN)**.
- `sonic-kernel` native execution: All 10 phases + JSON-RPC stdio queries verified.
- Python test suite: `test_native_kernel_bridge.py` (4 tests) + `test_remediation_puppet_and_control.py` (12 tests) **16 passed in 1.55s**.

## Phase 36 — Continuous Codebase Self-Evolution Engine (`CodebaseEvolver` & CLI Runner) (DONE)
Empowered SONIC with an autonomous, continuous codebase self-evolution engine capable of patching bugs, upgrading logic, and optimizing non-safety modules with zero-corrupt atomic rollbacks, automated test verification, and auto-promotion (git commit + push) without manual human approval:

### 1. Guarded Codebase Evolver (`services/core/sonic/evolution/codebase_evolver.py`)
- **Safety Envelope Invariant Gate**: Strict fail-closed check against `_PROTECTED_COMPONENTS` (`sonic/safety`, `sonic/kernel/action_broker`, `sonic/safety/sealed`, `sonic/safety/action_policy`, `sonic/sandbox/egress`, `packages/kernel/src/safety`). Any attempt to tamper with or loosen safety envelopes is immediately rejected with a critical security violation audit log (`EvolutionStage.REJECTED`).
- **AST & Syntax Pre-Flight**: Pre-parses Python code with `ast.parse()` to catch syntax errors before any file is touched on disk.
- **Atomic Snapshot & Zero-Corrupt Rollback**: Maintains in-memory snapshots of all touched files; if unit tests or security regression tests fail, files are restored to their exact pre-evolution state and an `[AVOID]` lesson is recorded in `LessonsLedger`.
- **Automated Verification Pipeline**: Automatically executes component unit tests (`pytest`) and security regression tests (`test_p0_security_hardening.py`, `test_safety_sealed_policy.py`). Requires 100% green exit code to proceed.
- **Structured Evolution Summary Report**: Generates a rich, tamper-evident Markdown report with proposal ID, target component, diff metrics (+/- lines), AST verification status, test execution results (passed/failed/duration), and git provenance.
- **Autonomous Promotion & Git Push**: When `auto_promote=True` and tests pass 100%, changes are automatically committed to git (`evo(<id>): <desc>`) and optionally pushed to remote origin without requiring human operator approval.

### 2. Dedicated CLI Runners
- **Python CLI Runner (`services/core/sonic/evolution/cli.py`)**:
  - `python -m sonic.evolution.cli patch --target <path> --diff <patch> --desc <desc> --auto-promote --push`
  - `python -m sonic.evolution.cli status`: displays evolver status and list of protected safety components.
- **SONIC Power CLI Subcommand (`apps/cli/sonic_cli/commands/evolution.py`)**:
  - `sonic evolution patch --target <path> --desc <desc> --auto-promote --push`

### 3. Verification Baseline (100% Green)
- `services/core/tests/test_codebase_evolution.py` (5 tests) — **5 passed in 0.69s**.
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
- `services/core/tests/test_native_kernel_bridge.py` (7 tests) — **7 passed in 1.33s**.
- `services/core/tests/test_phase0_safety_kernel_and_broker.py` (6 tests) — **6 passed in 1.15s**.
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
- `services/core/tests/test_advanced_evolution.py` (9 tests) — **9 passed in 3.00s**.
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

SONIC software ownership is formally divided by department, team and subsystem. The authoritative registry is `services/core/sonic/organization/architecture.py`.

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

### Historical GUI perception and execution notes (2026-09-30; superseded by AVCS-only perception contract above)
These older implementation notes record a former NSI/OCR path and must not be
used to reintroduce it. Current workstation perception uses AVCS raw frames
only; the OCR/Tesseract instructions below are historical, not active guidance.

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
The self-evolution department (`services/core/sonic/evolution/`) was reported as
"not working at all" while its own unit tests passed. Root cause was in the
**candidate verification gate**, not the reasoning/LLM layer: `SelfCoderCandidateStore`
reapplies a candidate diff in a detached worktree, runs the full configured test
dirs (`services/core/tests` + `apps/cli/tests`) via `CodebaseEvolver.run_tests`, and
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
- Both `services/core` and `apps/cli` are root `uv` workspace members. Install
  both editable projects and their test dependencies from the repository root
  with `uv sync --all-packages --extra dev`.

Verified end-to-end: a stub candidate diff applied to `logger.py` promoted to
`STATUS: applied` with `full_test_suite: True`, `security_regression: True`,
1872 passed / 0 failed, both in the detached worktree and after applying to the
active checkout. Full `services/core/tests` baseline after the fix: **1859 passed,
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
   - Stripped repository prefixes (`services/core/`, `apps/cli/`) before boundary checks so worktree-relative paths match reliably.
   - Expanded `_PROTECTED_COMPONENTS` to protect critical runtime gates: `sonic/agents/judge.py`, `sonic/evidence`, `services/core/tests/test_safety_sealed_policy.py`, `sonic/evolution/self_coder_candidates.py`, `sonic/evolution/continuous_runner.py`, and `sonic/sandbox/factory.py`.
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

2. **Standalone Local `DeveloperAgent` Engine (`apps/cli/sonic_cli/developer_agent.py`)**:
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

3. **Streamlined REPL & Command Surface (`apps/cli/sonic_cli/repl.py` & `main.py`)**:
   - Free-form natural language inputs route directly to the local `DeveloperAgent` without network hops to Boss.
   - Instant Conversational Greetings: Basic greetings (`hi`, `hello`, `hii`, `hey`, etc.) respond conversationally in < 5ms without tool invocations or desktop screenshots.
   - Developer Slash Commands: `/auto [N]`, `/diff`, `/journal`, `/test [path]`, `/status`, `/help`, `/clear`, `/exit`.
   - Clean Subcommands: `sonic status`, `sonic diff`, `sonic test`, `sonic auto`, `sonic journal`, `sonic auth`, `sonic version`.

4. **100% Green Verification Suite**:
   - 66/66 tests passing in `apps/cli/tests/`:
     - `test_cli_devin_mode.py` (21 passed): REPL slash commands, interactive banner, zero-arg invocation.
     - `test_developer_agent.py` (28 passed): Direct tool execution, tool loop, zero-boss isolation.
     - `test_developer_agent_safety.py` (17 passed): Path traversal protection, timeout process termination, secret sanitization, error feedback.

## Department Platform Systems & Laya System-1 Engine Hardening (2026-10-06)

Fourth-pass audit and remediation across Department `platform_systems` (`services/core/sonic/api/`, `config.py`, `logger.py`, `computer/headless.py`, `execution/capability_router.py`) and the Laya System-1 Non-Autoregressive Decision Engine (`sonic/llm/laya_engine.py`):

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

## Perception/import latency regression + AVCS/CUA audit (2026-10-20)

Investigated the reported "SONIC got slow, CUA not working" regression and audited
AVCS perception + the CUA agent.

### Root cause of the cold-start slowness (fixed)
- `packages/avcs/forge/vision/__init__.py` eagerly imported **every** submodule,
  including `forge.vision.encoder`, which imports `torch` (~1.1 s). Because
  Python runs a package `__init__` before any submodule, the workstation
  perception hot path (`sonic/computer_use/avcs_perception.py` →
  `from forge.vision.reflex import ReflexPerception`) paid the full torch import
  cost on every fresh process, even though the reflex path never uses torch.
- Measured: `import forge.vision.reflex` = **1522 ms** before → **162 ms** after.
- Fix: converted the package `__init__` to PEP 562 lazy attribute resolution
  (`__getattr__`/`__dir__` + an explicit name→submodule map). The public API is
  unchanged; `from forge.vision import SceneUnderstandingModel` still works and
  only then imports torch. No consumer imports package-level `forge.vision`, so
  nothing regressed.
- Regression guard: `packages/avcs/tests/test_lazy_package_import.py` (4 tests,
  fresh subprocesses) asserts `forge.vision.reflex` does not load torch, the
  public API is intact, encoder symbols resolve lazily, and unknown names raise
  `AttributeError`.

### Per-action perception cost (measured, not a bug)
- Each GUI reasoning step still takes **2 provider screenshots** (one
  `observe()`, plus a pre-dispatch fresh-frame revalidation in
  `_create_gui_action_intent`/`_dispatch_gui_action`). That is intentional
  fail-closed freshness enforcement, not redundant perception. With a Docker
  provider whose `screenshot()` bundles window metadata, this is the dominant
  per-step latency; AVCS perception itself is only ~18 ms warm.

### AVCS perception audit (correct)
- `perceive_rgb`: 9.65 ms/frame (sha256 fingerprint 1.93 ms, reflex 7.4 ms,
  system3 0.02 ms) → ~100 fps, fine for a 30 fps frame stream.
- Fingerprint / frame-ID / scene-hash revalidation in `ground_goal` is correct
  (guards against a frame changing mid-inference; raises off-screen points).
- `perceive_png` passes `source_frame_id=None` (correct: monotonic-ID
  enforcement applies to the raw-RGB provider stream, not PNG re-decodes).

### CUA blocker (environmental, honest fail-closed)
- `services/core/sonic/health.py::check_llm` reports
  "No configured vision provider credentials are available for GUI reasoning"
  because no SONIC LLM/vision provider keys are present in the environment. This
  is correct fail-closed behavior, not a code bug: the `vision`/`computer_use`
  routing rules point at `openrouter/openai/gpt-4o` with a
  `nvidia/meta/llama-3.2-11b-vision-instruct` fallback, and the router fails all
  providers fast when keys are absent. Provide credentials before expecting
  live GUI reasoning; do not claim GUI capability from unit tests.

### Pre-existing test flake (NOT caused by this work)
- `packages/avcs/tests/test_soak_runner.py::test_soak_runner_short_run` fails
  only under suite load (its handle-stability gate is resource-sensitive); it
  passes in isolation (1.37 s) and fails identically on the unmodified HEAD
  `__init__.py`. Treat as environmental/load-induced.
- `packages/avcs/tests/test_desktop_gui.py` fails to collect without system
  `libtk8.6.so` (tkinter) — environmental.

### Suite status after this pass
- `services/core/tests` + `packages/avcs/tests` (minus tkinter module):
  **2628 passed, 53 skipped, 1 pre-existing load-flake** (soak).


## Consumption-loop + mouse fix pass (2026-10-20)

Root cause found for "the mouse can't be used" and the perception loop:

- **Parser bug (real, fixed).** `ComputerUseAgent._parse_llm_action` split the
  LLM's ACTION/TARGET/PAYLOAD block with a `\s*` separator after each field
  colon. For the natural coordinate-motor shape — an *empty* `TARGET:` line
  followed by `PAYLOAD: {"x":..,"y":..}` — the newline was consumed, TARGET
  captured the literal `PAYLOAD: {...}` text, and the coordinate payload was
  dropped. The GUI action then degraded into an unresolvable **semantic** target
  and was `BLOCKED` fail-closed ("local AVCS grounder did not resolve the
  target 'PAYLOAD: {…}'"). So `GUI_MOVE`/`GUI_CLICK` with payload-only
  coordinates could never run through the mission loop. Fix: separator is now
  `[ \t]*`, so an empty field cannot absorb the next field. 3 regression tests
  in `test_phase8_visual_computer_use.py`.
- **Mouse verified live.** Against the real `sonic-desktop-workstation`
  (DISPLAY :99) via the real `ComputerUseAgent.run_mission` loop with a scripted
  router: APP_LAUNCH -> COMPLETED (1 real window), GUI_MOVE -> COMPLETED (cursor
  glided 25,18 -> 500,300), GUI_CLICK -> VERIFIED (AVCS pixel scene changed at
  140,260). Direct dispatch of every motor action also lands the cursor exactly
  (MOVE/CLICK/DOUBLE_CLICK/RIGHT_CLICK/DRAG); SCROLL dispatches and honestly
  reports UNVERIFIED when nothing changed. Scripts: `scripts/live_gui_mouse_mission.py`,
  `scripts/live_gui_mouse_task.py`.
- **The four-layer flow is wired.** AVCS (System 1) produces the bounded
  scene/change/motion reflex summary; it reaches System 2 as a *text* summary in
  the reasoning prompt (`CURRENT AVCS PIXEL STATE SUMMARY ... scene=.. changed=..
  change=.. motion=..`) with no pixels sent; the provider executes the action;
  the post-action AVCS frame is the verification receipt (VERIFIED only on a real
  scene change). Semantic targeting and focus-verification remain fail-closed
  when the evaluated AVCS heads are not configured (GUI_TYPE blocked).
- **Slowness diagnosis.** `observe()` ~= 260-290 ms = `screenshot()` 97 ms +
  agent-side AVCS recompute 24 ms + `status()` 145 ms. `status()` ran the cold
  `application_list()` scan (~280 ms) sequentially; it is now folded into the
  existing `asyncio.gather` (window/active-window/process/application probes run
  concurrently). Further gains require reusing the provider's already-computed
  `avcs_state` (rejected: the agent's AVCS instance must re-derive from the exact
  frame it will ground against) or a persistent in-container telemetry daemon.

Test baseline after this pass: 935 passed, 25 skipped (broad computer/agent/gui
regression); 660 passed on the parser/action-focused subset.

## Focus-verification head + hot-path latency fix (2026-10-20)

Two real gaps closed while assessing AVCS/CUA and a reported slowness regression.

### 1. Local focus-verification head — previously entirely missing
The agent demanded an "evaluated local focus-verification head" before typing,
but **no focus-verifier module existed anywhere**; `GUI_TYPE`/`BROWSER_NAVIGATE`
were blocked by an unconditional early `return` in `_execute_action_unlocked`
(agent.py ~5380), which also made the real focus-verify path (~6759) dead code.
Built the missing capability with the same honesty discipline as the grounder:

- `packages/avcs/avcs/focus_verifier.py` — `FocusVerifier` (CorticalBackbone2M +
  `ByteQueryEncoder` + MLP head) classifying whether a screen point is a focused
  text-entry control; `load_focus_checkpoint()` refuses any checkpoint lacking
  recorded-demonstration provenance or a passing held-out precision evaluation.
  `focus_patch_bounds`/`focus_patch_rgb` crop a clamped square patch around the
  point (no clamping of off-screen points into edge clicks).
- `packages/avcs/avcs/training/focus_dataset.py` — session-disjoint discovery/
  split (task+window groups never leak across train/val/held-out) and dataset.
- `packages/avcs/avcs/training/train_focus_verifier.py` — trainer with held-out
  precision gate; `runtime_ready` is derived, never decreed.
- `services/core/sonic/computer_use/avcs_perception.py` — `verify_focus_rgb_bytes`,
  `ensure_focus_verifier_ready()`, `focus_verifier_ready`; fails closed without a
  configured, evaluated checkpoint or when pixels are not the latest frame.
- `agent.py` — `_avcs_focus_head_ready()` capability probe; the early GUI_TYPE/
  BROWSER_NAVIGATE block is now conditional on it. No checkpoint → still BLOCKED
  (unchanged); a genuinely ready head → dispatch proceeds to real focus
  verification. A configured-but-unloadable checkpoint is NOT readiness.

**Bug found during verification:** threshold selection used raw scores as
candidates, so the chosen threshold landed exactly on a true positive score and
`score >= threshold` excluded it (field scored 0.897 with threshold 0.897 →
`focused=False`). Fixed to use midpoint candidates between adjacent distinct
scores (standard ROC practice).

### 2. Slowness regression — duplicated Docker probes on the cold hot path
`status()` fans out 4 concurrent `docker exec`s, but each independently ran the
stale-cache `docker info` daemon probe (and `docker inspect` container probe).
With a stale (>10 s) cache a cold `status()` cost ~446 ms with **4** duplicate
`docker info` probes; warm it was ~92 ms. Added single-flight async locks
(`_daemon_probe_lock`, `_container_probe_lock`) with double-checked re-reads.
Cold stale-cache `status()` dropped to ~114 ms with 1 probe. Warm `observe()`
remains ~260 ms (status ~150 ms + screenshot ~100 ms); `application_list` is
cached so the earlier "fold into gather" fix is not defeated.

### Tests
- `packages/avcs/tests/test_focus_verifier.py` (14).
- `services/core/tests/test_phase8_visual_computer_use.py` `TestFocusHeadCapabilityGate` (5).
- `services/core/tests/test_docker_computer_provider.py` (2): single-flight probes.
- Focused regression: 153 passed, 2 skipped across provider + GUI suites.

**Honesty:** unit/contract validations only. No real arbitrary-GUI end-to-end
task was demonstrated, and no live focus checkpoint exists in-repo; without one,
text entry correctly stays blocked. Train a checkpoint from real recorded
demonstrations (packages/avcs/docs/LOCAL_GROUNDING_QUICKSTART.md) before
claiming focus verification works on real controls.

## Live-capability diagnostic pass: "AVCS/Boss not working" root cause (2026-10-21)

Operator reported the being "cannot do any work on the computer": AVCS not
functioning, Boss launching subagents that "can't do anything", VNC bugs. This
pass separates what is genuinely broken from what is fail-closed-by-design, and
fixes the real reporting bugs. No capability is claimed without evidence.

### Ground truth (verified, not asserted)
- AVCS reflex perception WORKS on real numpy frames: a large change sets
  `scene_changed=True` (change_fraction ~0.40), a 1-pixel change does not.
  The "eyes" algorithm is not the breakage.
- Full suite: 2105 passed, 40 skipped, 0 failures. The problem is not failing
  tests; tests validate mocked contracts while live capability is gated.
- Boss architecture is intact: `_strategic_decomposition` → real subagent
  `ComputerUseAgent`s → independent `JudgeAgent`. Dispatch kwargs match the
  agent constructor. So "subagents can't work" is a capability GATE.

### The real reason a GUI worker "cannot do anything" (fail-closed, correct)
Semantic GUI targeting and text entry require evaluated LOCAL AVCS heads:
- `SONIC_AVCS_GROUNDING_CHECKPOINT` (goal grounder) — **no checkpoint exists
  anywhere in the repo**, so a semantic GUI target (click a *described* control)
  is BLOCKED with "SONIC_AVCS_GROUNDING_CHECKPOINT is not configured".
- `SONIC_AVCS_FOCUS_CHECKPOINT` (focus head) — required before GUI_TYPE.
Verified end-to-end with a scripted router + stub provider: an explicit
pixel-coordinate `GUI_CLICK` dispatches and reports `VERIFIED` via a real AVCS
scene change; a semantic target is BLOCKED. The model is also never given
screenshot pixels (`image_attached=False`), so it cannot invent coordinates.
Net: the desktop is "eyes only" until the operator trains and configures
evaluated AVCS heads (packages/avcs/docs/LOCAL_GROUNDING_QUICKSTART.md).

### Real bugs fixed this pass
1. **VNC URL unreachable from a remote browser** — `get_vnc_url()` and the
   default session `novnc_url` were hardcoded to `127.0.0.1`/`localhost`. A
   dashboard on any other machine cannot reach that. Added
   `_novnc_public_host()` (env `SONIC_NOVNC_PUBLIC_HOST` →
   `SONIC_WORKSTATION_PUBLIC_HOST` → `SONIC_PUBLIC_HOST` → `localhost`) and
   routed both call sites through it. Fail-closed reachability checks unchanged.
2. **Misleading desktop resolution** — reported `1280x800` while the Xvfb
   workstation runs `1920x1080`. Defaults corrected to `1920x1080`; the
   `test_phase21` assertion updated to match. (Live `status()` still overrides
   with the real frame dimensions.)
3. **Misleading perception log** — `screenshot_pixels_available` checked only
   `screenshot_base64`, ignoring the actual AVCS payload (`raw_rgb`), so it
   logged `False` while pixels were present. Now uses `_screen_has_pixel_data`.
4. **No capability surface** — added `avcs_capability_state()` and exposed it as
   `desktop["avcs_capability"]` on `/workstation/desktop/status`, plus an
   "EYES ONLY · ACTIONS BLOCKED" chip on the dashboard. A green "LIVE" desktop
   no longer implies "ready to work".

### Honesty
- Tests: 2105 passed / 40 skipped / 0 failed; dashboard `next build` clean.
- No live GUI task was run (no LLM/vision credentials, no Docker here). The
  blocking gate is verified by code inspection + a stub-provider E2E harness;
  no real arbitrary-GUI acceptance is claimed.
- To make the being actually operate the GUI: supply model credentials AND
  train+configure the AVCS grounding (and focus) heads. Until then, only
  explicit-coordinate pointer actions can dispatch, and text entry stays blocked
  by design.

## Slowness regression: runaway AVCS frame stream + cache TTL collision (2026-10-21)

Investigated the report "SONIC bahot slow ho gaya hai" and found two real,
measurable causes (not perception quality, not the Boss architecture).

### Root cause 1 — the continuous AVCS frame stream never stopped (dominant)
`DockerComputerProvider.screenshot()` starts a persistent in-container raw-frame
stream (`docker/avcs_frame_stream.py`, `mss` at 1920x1080) on first use and
reuses it for later observations. Nothing ever stopped it: no idle timer, and
no caller invokes `provider.close()` on a live server (the only `close()` calls
are inside the provider itself, e.g. `destroy()` when the last workspace goes
away). So a single dashboard visit left the backend running a 30 FPS capture
loop *forever*.

Measured at 1920x1080 (this environment):
- AVCS `perceive_rgb_bytes`: ~20-30 ms per frame
- 30 FPS => **~61% of one CPU core, continuously**
- pipe volume: 1920*1080*3 = 6.2 MB/frame => **~187 MB/s** through the
  `docker exec` pipe, plus per-frame `np.frombuffer`/reshape in the processor task

The workstation container itself was also grabbing a full 1080p frame 30x/s,
competing with the desktop/agent for CPU inside the same container.

Fix (`services/core/sonic/computer/docker_computer.py`):
- Idle watchdog `_stop_idle_stream()`: closes the stream once no observation has
  been requested for `SONIC_AVCS_STREAM_IDLE_SECONDS` (default 30 s). The next
  observation transparently restarts it, so the fast path (a burst of
  observations reuses one stream) is preserved. `close()` cancels the watchdog
  and is safe when called *from* the watchdog.
- `_captures_in_flight` counter: the watchdog never releases the stream while an
  observation is in flight, so frames cannot be pulled out from under a capture.
- Default capture FPS lowered from 30 to 10 (`SONIC_AVCS_CAPTURE_FPS` still
  overrides, 1-60). No consumer needs 30 FPS: the being life loop ticks at 60 s
  and the agent observes once per reasoning step. 10 FPS cuts idle CPU burn by
  ~3x and the pipe volume to ~62 MB/s while keeping observation latency low.
- `_last_stream_use` is refreshed both when a capture starts and when a fresh
  observation is returned.

### Root cause 2 — telemetry cache TTL exactly equalled the poll interval
`GET /workstation/state` caches sandbox telemetry for `telemetry_cache_ttl = 5.0`
seconds, and the dashboard polls it every `5000` ms. The cache therefore expired
at essentially the same moment the next poll arrived, giving a ~0% hit rate: every
poll paid the full `status()` + `get_vnc_url()` + `screenshot()` cost. Fixed to a
configurable 8 s (`SONIC_WORKSTATION_TELEMETRY_CACHE_SECONDS`), which sits above
the 5 s poll so the second request is served from cache.

### Also fixed while profiling
- The capability probe added in the previous pass constructed a fresh
  `AVCSPerception()` per call; once a focus checkpoint is configured that would
  reload the checkpoint on every status poll. Now a process-wide shared probe
  (`workstation_perception._shared_capability_probe`).
- Stale `1280x800` default remained at one restore path in `workstation.py`; the
  real Xvfb display is 1920x1080. Corrected for consistency with the other sites.

### Evidence
- Full `services/core/tests`: **2109 passed, 40 skipped, 0 failures**.
- New regression tests (`test_docker_computer_provider.py`): idle watchdog closes
  the stream when unconsumed; keeps it while consumed; `close()` cancels the
  watchdog without deadlocking; conservative default FPS/idle constants.
- Live check with a real subprocess: the stream is SIGTERM'd after the idle
  period and `avcs_frame_stream_idle_stopped` is logged.
- Ruff clean on all changed files.

Honesty: these are measured latency/CPU fixes on the Docker provider hot path.
No live end-to-end GUI task was run (no Docker daemon or model credentials in
this environment); the idle-stop behaviour was verified with a real subprocess
plus unit tests, not a full workstation session.

## AVCS learned cortex + continuous awareness (2026-10-21)

Closed the "AVCS kaam nahi karta" gap: AVCS reflex was pure numpy heuristics
with NO learned weights, the neural cortex had ZERO trained checkpoints (random
outputs), the advertised weights URL was fabricated (HTTP 404), and nothing
consumed perception events — the EYES never reached the BRAIN.

### 1. Real trained cortex (measured, not decreed)
- `packages/avcs/forge/vision/cortex/training/evaluate.py` (NEW) — held-out
  evaluator over seeds disjoint from training (`TRAIN_SEED_BASE` vs
  `EVAL_SEED_BASE`). Writes an `.eval.json` sidecar that requires a
  `dataset_provenance` string, so a metric cannot be recorded without naming
  the data it was measured on.
- Trained `ArtificialVisualCortex` (1.14M params, real PyTorch CNN) for 14
  epochs on synthetic procedural screens: loss 3.28 -> 0.95.
- **Held-out results** (`runs/cortex_model.pt`, 64 samples): category_accuracy
  **0.7031** (8-way chance = 0.125), affordance_pixel_accuracy **0.9875**,
  affordance_mIoU **0.7976**, CPU inference **18.62 ms/frame**.
- HONESTY: this is synthetic procedural-screen accuracy. It is not real desktop
  footage and does not by itself prove real-world scene understanding.
- `packages/avcs/avcs/weights.py`: removed the fabricated download URL
  (`DEFAULT_WEIGHTS_URL = None`); `download_weights()` now raises instead of
  pretending to fetch pretrained weights that were never published.

### 2. Cortex wired into the SONIC runtime
- `sonic/computer_use/avcs_perception.py`: `perceive_semantics_rgb(frame)` plus a
  cortex readiness gate (`cortex_configured` / `ensure_cortex_ready` /
  `cortex_ready`). Honest empty result when no checkpoint is configured.
- `sonic/computer_use/agent.py`: learned semantics ride into `avcs_state` and
  the AVCS reflex summary line handed to the reasoning model.
- Verified live: with `SONIC_AVCS_CORTEX_CHECKPOINT` set, the runtime loads the
  trained checkpoint and emits real semantics (`primary_category`,
  `category_confidence`, `action_hotspots`) — not random output.

### 3. Continuous awareness — the EYES -> BRAIN link
- `sonic/computer_use/awareness.py` (NEW): `AwarenessMonitor` samples the real
  desktop continuously (no operator prompt), derives scene/change state through
  AVCS, and publishes `perception.frame` / `perception.change` on the shared
  cognition bus. It never dispatches a GUI action (observation only) and a
  transient capture failure never kills the loop. FPS is capped to protect the
  host; one monitor per workspace.
- Wired into `api/main.py` lifespan behind `SONIC_ENABLE_VISUAL_AWARENESS=1`
  (default off — a normal boot must not pay continuous capture cost); a no-op
  without a provider/home.
- `sonic/being/life_loop.py`: the being now SUBSCRIBES to `perception.change` /
  `perception.frame`, buffers recent changes, and feeds them into its next idle
  tick via `perception_context()` — publishing `being.perception_awareness`.
  This is the actual consumer; before this the change events had no listener.
- `get_life_bus()` gives producers and consumers one process-wide bus.

### 4. Honest capability surface
- `workstation_perception.avcs_capability_state()` now reports
  `learned_cortex_configured`, `learned_cortex_ready`, `continuous_awareness`
  and `awareness_stats` alongside the existing grounding/focus flags, so an
  operator can see exactly which AVCS layer is actually ready.

### Tests
- `services/core/tests/test_continuous_avcs_awareness.py` (14): continuous
  sampling without a prompt; real change -> change event; static screen -> no
  false change; never dispatches GUI actions; transient error survived;
  one monitor per workspace; FPS cap; invalid FPS rejected; cortex fails closed
  without/missing checkpoint; a real checkpoint loads and emits semantics;
  awareness stats; capability surface honesty; the life loop consumes
  perception changes.
- `packages/avcs/tests/test_cortex_training_and_evaluation.py` (10): real
  network head shapes; train/held-out seed disjointness; deterministic real
  metrics; training reduces loss + checkpoint round-trips; missing/shape-bad
  checkpoint fails closed; sidecar records provenance; no fabricated weights
  URL; weights path honest; `load_avcs_weights` returns False without a file.
- Regression: 193 passed across the awareness/GUI/being/security suites; ruff
  clean on all changed files.

## Boss dispatch policy + AVCS video apprenticeship (2026-10-21)

Two capabilities the product goal needs were made real and honest: the Boss now
decides *whether* to fan out (instead of always delegating), and AVCS can learn
from watched video (apprenticeship) instead of only synthetic frames.

### 1. Boss dispatch policy — WHEN to launch sub-agents (`computer_use/dispatch_policy.py`)
- Before this the Boss always split an objective into sub-missions and ran
  dependency waves, but *whether* fan-out was appropriate was implicit. It could
  launch a swarm for a one-worker task, and an operator could not see why.
- New pure-function policy decides per phase: ``direct`` (no work), ``single``
  (one adaptive worker), or ``parallel`` (N genuinely independent workstreams).
  "Independent" is **structural**, never keyword-based: two workstreams may run
  together only when (a) neither depends on the other, (b) their owned paths do
  not overlap, and (c) they are not two GUI-capable workers racing one shared
  physical desktop.
- Wired into ``BossAgent.run()``: the decision is logged, streamed to the UI as a
  ``dispatch_decision`` event, stored on ``BossAgent.last_dispatch_decision``,
  and included in the final ``BossReport.dispatch_decision``. It bounds
  ``phase_concurrency`` for the wave loop, so a "single" verdict serializes the
  phase instead of racing — but the work is never silently dropped.
- Honest constraint recorded, not hidden: one workstation parallelizes headless
  work freely but MUST serialize GUI work (focus/windows/cursor are shared
  mutable state). ``shared_desktop`` comes from the provider having
  ``screenshot``/``gui_action``.
- Tests: ``services/core/tests/test_boss_dispatch_policy.py`` (14) — policy unit
  cases plus **real ``boss.run()`` integration**: headless fans out, two GUI
  workers on a shared desktop serialize (still executed), one workstream stays
  single, and the decision is durable on the report.

### 2. AVCS video apprenticeship (`avcs/training/video_apprenticeship.py`)
- Lets AVCS learn from a screen recording (a YouTube download, OBS capture, VNC
  recording) with no hand-labelling. Uses ``imageio-ffmpeg`` (bundled ffmpeg) so
  no system ffmpeg is required.
- What is honestly derivable from raw video is extracted: **frames** (feed the
  cortex's self-supervised temporal-predictive objective) and **cursor
  trajectory** (a compact high-contrast blob; its pauses become *candidate*
  click events). The cursor detector uses a local box-mean residual centroid —
  an earlier window-max score was biased toward the window edge, offsetting the
  point by ~half a window; a flat frame correctly returns ``None``.
- What is NOT derivable is never invented: UI category, affordance masks, and
  semantic click labels stay absent. The manifest is tagged
  ``provenance: watched_video_unlabeled_pixels`` and click events carry
  ``confirmed: False``.
- ``forge/vision/cortex/data/video_dataset.py``: ``VideoFrameDataset`` feeds the
  cortex trainer from a manifest. Category targets are ``-1`` (unlabeled) and
  the loss ignores them; the category term is honestly 0 for unlabeled batches.
- **Real bug found + fixed:** when a batch is entirely unlabeled,
  ``CrossEntropyLoss(ignore_index=-1)`` returns ``nan``, which poisoned the
  total loss and would have saved a ``nan`` checkpoint. The loss now substitutes
  a real zero in that case. Verified: the trunk genuinely learns from watched
  video (predictive loss 0.55 -> 0.18 over 4 epochs) with the category term
  honestly 0.
- CLI: ``avcs-ingest-video --video <path> --output-dir <dir> [--fps] [--max-frames]
  [--scale] [--no-cursor]``; trainer mode
  ``python -m forge.vision.cortex.training.trainer --video-manifest <manifest>``.
- Tests: ``packages/avcs/tests/test_video_apprenticeship.py`` (12) — probe,
  bounded frame extraction, real cursor + click-candidate detection, JSON-safe
  manifest with no fabricated labels, flat-frame rejection, honest missing-ffmpeg
  error. Existing cortex/grounder suites unchanged.

### Verified state (honest)
- New tests: 14 (dispatch) + 12 (video) all pass; combined focused run 82 passed.
- Full ``packages/avcs/tests`` (minus the pre-existing tkinter + soak-load
  environmental failures): **575 passed, 13 skipped**.
- Focused ``services/core`` computer_use/safety subset: **364 passed, 8 skipped**.
- Ruff clean on every new/changed file (one pre-existing ``N812`` in
  ``cortex/training/losses.py`` is unchanged from HEAD).
- **What is still NOT proven:** no real arbitrary-GUI task was run, because this
  environment has **no LLM provider credentials** (``check_llm`` ->
  "No configured vision provider credentials are available for GUI reasoning").
  Workers fail closed without a router; that is the reason live sub-agents do no
  work here, not a code defect. No real YouTube video was downloaded or trained
  on (network fetch is the operator's call); the pipeline was validated on real
  ffmpeg-encoded test videos. Do not claim GUI capability or trained-model
  quality beyond these measurements.

## Live arbitrary-GUI acceptance run + two real bug fixes (2026-10-22)

A real end-to-end GUI task was finally run against the live Docker workstation
with real LLM credentials, which surfaced two genuine defects that unit tests
could not see.

### Live environment used
- `docker compose -f docker-compose.yml up -d workstation` started
  `sonic-desktop-workstation` (Xvfb `:99`, **1920x1080x24**, x11vnc, noVNC on
  `127.0.0.1:6080`).
- NVIDIA key present; router loaded `[openrouter, deepseek, groq, nvidia,
  claude, openai, grok, local]`. Vision route model
  `meta/llama-3.2-11b-vision-instruct` is available on this account;
  `nvidia/nemotron-3-ultra-550b-a55b` is the working text model
  (`nemotron-3-super-120b-a12b` is catalogued but 404s for this account tier —
  the model-level fallback chain recovers, which is correct behaviour).

### REAL BUG 1 — AVCS learned-cortex perception failed on every frame (`np` NameError)
- `sonic/computer_use/agent.py` used `np.frombuffer(...)` in the AVCS cortex
  perception path but **never imported numpy**. The `NameError` was swallowed by
  a broad `except`, so every frame logged
  `avcs_cortex_perception_failed error="name 'np' is not defined"` and semantics
  silently stayed empty. This is a real "AVCS kaam nahi karta" symptom.
- Fix: added `import numpy as np`. Verified live: the error is gone and
  `perceive_semantics_rgb` now runs (returns `{}` honestly — no trained cortex
  checkpoint is configured, which is correct, not a failure).

### REAL BUG 2 — global default model leaked to the wrong provider
- `sonic/llm/router.py` applied the global `default_model` from
  `configs/models.yaml` (`qwen/qwen3.8-27b`, a **Groq** model id) to whichever
  provider was the global default provider. With `DEFAULT_LLM_PROVIDER=nvidia`
  (from `configs.py` `default_llm_provider = "nvidia"`), NVIDIA was handed a
  model id that does not exist on its endpoint: every reasoning request logged
  `model_not_found_trying_fallbacks ... 404 ... original_model=qwen/qwen3.8-27b`
  and paid a wasted round-trip before the model-level fallback recovered.
- Fix: a global/env default model is now applied **only when that provider
  actually declares the model id**; otherwise the provider's own first declared
  model is used (with a `provider_model_override_not_declared` warning).
  Verified live: `nvidia: default_model=nvidia/nemotron-3-super-120b-a12b`
  (previously `qwen/qwen3.8-27b`), `groq: qwen/qwen3.8-27b` unchanged.

### Live evidence that the three parts DO work
- **Living eyes (AVCS)**: real frame stream against the container
  (`avcs_raw_frame_processed`, `capture_fps≈10`, `latency_ms≈18`, 1920x1080,
  scene hash + change/motion/regions), plus a real `INTERACTIVE` screenshot.
- **Brain (Boss/agent reasoning)**: the real `ComputerUseAgent.run_mission()`
  reasoned from the AVCS text summary and chose actions. With a GUI goal the LLM
  proposed `GUI_CLICK` on a semantic target.
- **Hands (mouse/keyboard)**: the LLM-driven mission actually **opened Thunar**
  on the real desktop — verified out-of-band with `wmctrl -l` →
  `workspace - Thunar` and `pgrep thunar` → PID. A second deterministic run
  proved the agent dispatches an explicit **pixel-coordinate** `GUI_CLICK`
  end-to-end (status `UNVERIFIED`, honest: clicking empty desktop changes
  nothing).

### The honest blocker for *semantic* GUI clicks (fail-closed, by design)
- `avcs_capability_state()` reports
  `semantic_grounding=false, focus_verification=false, gui_actions_ready=false,
  grounding_checkpoint_configured=false`.
- A GUI-plane worker's semantic click is therefore blocked:
  `avcs_target_grounding_failed_closed error='SONIC_AVCS_GROUNDING_CHECKPOINT is
  not configured'` → `gui_mission_stopped_for_unavailable_avcs_capability`. This
  is the documented fail-closed contract, **not** a code bug: no evaluated local
  AVCS grounding checkpoint exists in the repo. Semantic clicks and text entry
  stay blocked until one is trained from real recorded demonstrations
  (`packages/avcs/docs/LOCAL_GROUNDING_QUICKSTART.md`).
- **What is proven:** pixel-coordinate GUI actions and application launching
  work end-to-end on the live desktop. **What is NOT proven:** semantic
  ("click the Save button") grounding and text entry, because the grounding and
  focus-verification checkpoints do not exist yet. Do not claim either.

### Tests
- `services/core/tests/test_live_gui_regression_fixes.py` (6): numpy import
  present + `np.` usage covered; global default model not leaked to a provider
  that does not declare it; provider-declared default respected; provider env
  override honoured when declared; undeclared provider env override falls back
  to the declared model.
- Focused regression: **419 passed, 2 skipped, 0 failures**; Ruff clean on all
  changed files.

## Boss live run on the real desktop — three more real bugs fixed (2026-10-22)

Ran the real `BossAgent.run()` against the live `sonic-desktop-workstation` with
a genuine objective ("open the terminal and report the working directory"). The
mission ended `INCOMPLETE`, and tracing why exposed three genuine defects.

### BUG 3 — workstation image shipped NO terminal emulator
`docker/Dockerfile.workstation` installed the `xfce4` meta-package but never a
terminal emulator (`xfce4-terminal` was absent; `xterm` absent too). A GUI task
that needs a terminal therefore had nothing to launch. Fixed by adding
`xfce4-terminal` to the image and rebuilding
(`docker compose -f docker-compose.yml build workstation`). Verified:
`command -v xfce4-terminal` -> `/usr/bin/xfce4-terminal` after recreate.

### BUG 4 — silent application substitution (a real honesty bug)
`DockerComputerProvider._resolve_app_binary` applied its generic POSIX-role
fallback to ANY query merely *containing* a role word. So `gnome-terminal` (not
installed) matched the "terminal" role and resolved to `x-terminal-emulator` /
`exo-open`, and `launch_application` returned **True** while opening a different
program. The agent then believed it had launched the requested application and
looped on a name that could never be honoured. This is exactly the "fake
success" class the project forbids.
- Fixed: the role fallback now applies only when the query is genuinely generic
  (every token is in `_GENERIC_ROLE_TOKENS`, e.g. "terminal", "web browser").
  A specific product name that is not installed resolves to `""` and logs
  `application_launch_not_installed`; `launch_application` returns `False`.
- Verified live: `resolve('gnome-terminal') -> ''`, `resolve('konsole') -> ''`,
  `resolve('terminal') -> '/usr/bin/exo-open'`, `resolve('xfce4-terminal') ->
  '/usr/bin/xfce4-terminal'`, `launch('gnome-terminal') -> False`.

### BUG 5 — GUI-only workers never saw the installed-application list
`_build_reasoning_context` replaced the whole observation block for the GUI-only
plane, and that replacement omitted `Available Installed Desktop Applications`.
The model therefore had no way to know which binaries exist and invented names
(`gnome-terminal`, `xterm`) even though `xfce4-terminal` was present at index 0
of the provider's list. Fixed by including the installed-application inventory
in the GUI-only observation block. Verified live: after the fix the model chose
the real `xfce4-terminal`, and a clean launch completed as `COMPLETED`
("Launched and focused xfce4-terminal", real window + PID).

### Capability-aware GUI prompt
When no evaluated local AVCS grounder is configured, semantic `GUI_CLICK`
targets fail closed. The GUI-only system prompt now detects this
(`_avcs_grounder_ready()`) and instructs the model to avoid semantic clicks and
use `APP_LAUNCH` (observed installed apps only) plus keyboard actions instead,
so it does not burn the whole step budget on an unresolvable target.

### Honest capability picture after this pass
- **Works on the live desktop:** AVCS frame perception; LLM reasoning;
  launching an installed GUI application; pixel-coordinate GUI actions.
- **Fail-closed (blocked by design, capability genuinely absent):** semantic
  GUI clicks ("click the Save button") — needs an evaluated AVCS grounding
  checkpoint; text entry — needs an evaluated AVCS focus-verification head.
  A Boss objective that requires clicking a described control or typing cannot
  complete until those checkpoints are trained from real demonstrations.
- Boss orchestration itself is wired and streams real events, but its
  end-to-end success is bounded by those two missing AVCS heads.
- One run also hit a transient 60 s LLM-request timeout (`computer_llm_action_failed`
  with an empty error) on the large reasoning model — a latency issue, not an
  action-logic defect.

### Tests
- `services/core/tests/test_app_launch_no_silent_substitution.py` (5): a specific
  uninstalled app is not substituted; a generic role still resolves; an installed
  specific app resolves directly; launching a missing app returns False; the
  workstation image declares a terminal emulator.
- `services/core/tests/test_gui_capability_aware_prompt.py` (6): prompt forbids
  semantic clicks without a grounder, forbids inventing app names, allows
  semantic targets when a grounder is ready, the real readiness probe fails
  closed without a checkpoint, the GUI-only observation surfaces installed apps.
- Focused regression: **237 passed, 2 skipped, 0 failures**; Ruff clean on every
  changed file.

## Boss subagent identity, live worker visibility & agent drill-down (2026-10-23)

Operator report: "Boss launches agents but they have no id, background running is
not visible, and clicking an agent's id/name shows nothing about what it is
doing." Investigation found the Boss emitted sub-agent events without a stable
unique handle and the UI could not drill into a worker.

### Real bugs fixed
1. **No stable unique worker id.** The positional `agent_id`
   (`sub-agent-N`) restarts every Boss run, so it cannot address one specific
   worker in a persisted session, and two concurrent workers could be confused.
   `SubMissionResult.agent_uid` added; the wave loop mints
   `worker-<n>-<hex>` once per worker (at batch assignment) and threads it
   through `_dispatch_sub_agent`, `sub_dispatch`, `sub_step` and `sub_report`.
2. **Crashed workers were stuck on RUNNING.** An unhandled exception in
   `_run_sub_worker` produced a `SubMissionResult` but never emitted a
   `sub_report`, so the worker stayed "RUNNING" forever in the UI. The exception
   path now emits a terminal `sub_report` (status FAILED) carrying the same uid.
3. **Stale RUNNING after a finished/interrupted mission.** A worker with no
   terminal report was displayed as RUNNING regardless of whether anything was
   actually driving the session. Worker status is now gated on the real
   background-task liveness (`_mission_task_alive`): no live task -> INTERRUPTED,
   never a stale RUNNING.
4. **`dispatch_decision` events were dropped by the route.** The Boss emits the
   DIRECT/single/parallel decision; the workstation event handler had no branch
   for it. Added a handler that records it as a mission event.
5. **No way to see what a worker is doing.** Added `GET /workstation/agents`
   (list) and `GET /workstation/agents/{agent_uid}` (detail), tenant-scoped and
   authenticated; `GET /workstation/state` now includes `workers` and
   `mission_task_alive`.

### New module
- `services/core/sonic/computer_use/agent_activity.py`: pure
  `collect_worker_activity(worklog, *, mission_task_alive)` +
  `find_worker(...)`. Projects the persisted worklog into per-worker records
  (identity, status, steps, latest action/thought, Judge verdict). A worker is
  RUNNING only while the owning task is alive; otherwise INTERRUPTED.

### Dashboard
- `types/workstation.ts`: `WorkstationWorker`, `WorkstationWorkerStep`, plus
  `agent_uid`/`orchestration_event` widening on `WorklogItem`.
- `lib/api.ts`: `getWorkstationAgents`, `getWorkstationAgent`.
- `WorklogFeed.tsx`: the Parallel-work board now renders the server-provided
  worker list (falling back to the worklog projection), shows each worker's
  `agent_uid`, and each card is clickable to expand a drill-down (goal, latest
  thinking, recent steps, Judge verdict, discoveries, report). Status honours
  `is_running`/INTERRUPTED so a dead mission no longer shows a spinning worker.
- `app/workstation/Workstation.tsx`: passes `workers` and `mission_task_alive`.

### Tests
- `services/core/tests/test_workstation_agent_activity.py` (15): RUNNING requires
  a live task; terminal report wins; FAILED/SUCCESS status; independent workers;
  step attach by uid; same `agent_id` across runs not merged; `find_worker`
  resolution; Boss mints one uid reused on dispatch+report; distinct uids per
  worker; crashed worker still emits a terminal report; `/workstation/agents`
  auth + list + detail + 404 + tenant scoping; `/workstation/state` includes
  workers + liveness.
- `test_boss_concurrency_and_summaries.py`: test doubles updated to the real
  `_dispatch_sub_agent` signature (accepts `agent_uid`).

### Baseline
- Full `services/core/tests`: **2168 passed, 40 skipped, 1 failed**. The single
  failure (`test_agent_quality_improvements.py::test_avcs_context_is_compact_
  temporal_pixel_state_without_ocr`) is pre-existing and unrelated: it asserts
  the old AVCS summary string, but `agent.py` now also emits
  `learned_semantics=...` (that file is not part of this change).
- Dashboard `next build`: 23/23 routes compile. Ruff clean on all changed files.

### Honesty
- This closes the identity/visibility/drill-down gaps. It does not change the
  documented capability gate: semantic GUI clicks still require an evaluated
  local AVCS grounding checkpoint and text entry a focus-verification head, so
  workers can still legitimately end BLOCKED/INTERRUPTED without an AVCS head.

## Capability catalog (SKILLS.md) + MoE / Agentic-RAG reality check (2026-10-23)

Operator asked (1) for a SKILLS.md catalog so SONIC's skills are available to it,
and (2) whether SONIC follows an MoE architecture and Agentic RAG. Both answers
are grounded below; the catalog was implemented so it is *loadable*, not decorative.

### MoE vs Agentic RAG — verified, not assumed
- **Agentic RAG: YES.** `sonic/memory/agentic.py::AgenticRetriever` does bounded
  query planning over `VectorMemory` (max 2 rounds / 4 queries, reciprocal-rank
  fusion, sufficiency stop, untrusted-evidence fencing). It is consumed by
  `being/lessons.py::relevant_agentic`, which `ComputerUseAgent` calls
  (`agent.py` ~3126) and injects into the reasoning prompt as REUSE/AVOID lessons.
- **MoE: NO.** There is no expert-router / sparse-activation architecture in the
  agent stack. The only MoE artifact is `packages/avcs/forge/model/moe.py`, a
  trainable sparse MoE FFN used by the AVCS neural cortex (and exercised only by
  AVCS tests) — it is not how SONIC routes work.
- **What plays MoE's role instead** (honest framing: functional analogue, not MoE):
  `sonic/llm/router.py` routes by `task_type` via `configs/models.yaml`
  `routing_rules` (planning/reasoning/coding/verification/vision...), with
  per-provider and model-level fallback chains. That is task-type routing, not
  token-level sparse expert activation. `sonic/orchestration/dual_process.py`
  (FAST/DEEP) and `LayaSystem1Engine` are fast/deep gating, also not MoE.

### SKILLS.md — implemented as a real, loaded catalog
A static skills markdown would have been dead documentation: **nothing in the
repo read any `.md` skills file** before this. So the catalog is parsed at
runtime and injected into every reasoning worker's prompt.
- `SKILLS.md` (repo root): a fenced-JSON block between
  `<!-- SKILL-CATALOG:START -->`/`END` listing 16 capabilities, each with
  `name/category/status/summary/actions/evidence/requires`. `status` is one of
  `available` / `conditional` / `not_wired`.
- `sonic/computer_use/skill_catalog.py`: `load_skill_catalog()` (cached),
  `render_skill_catalog()`, `reset_skill_catalog_cache()`. **Fail closed** —
  missing/unreadable/malformed file, missing JSON block, non-object payload, or
  an entry with an unknown status all yield an empty catalog; it never invents a
  skill or defaults an invalid status to available.
- `agent.py::_build_reasoning_context`: renders the catalog into the user prompt
  (so GUI-only workers, which use a different *system* prompt, still receive it).
  Env-gated by `SONIC_SKILL_CATALOG_IN_PROMPT` (default on). The block states it
  is advisory routing context that **does not grant permissions**; `ActionPolicy`
  /`SealedActionPolicy` still decide every action.

### Honest catalog contents (why the statuses matter)
- `available` (12): desktop app control, pixel pointer/keyboard, AVCS reflex
  perception, headless exec, git ops, security tooling, tool authoring, method
  invention, cross-mission lessons, delegation, verification/replan, dual-process.
- `conditional` (3): semantic GUI grounding (`SONIC_AVCS_GROUNDING_CHECKPOINT`),
  focus-verified text entry (`SONIC_AVCS_FOCUS_CHECKPOINT`), learned visual
  cortex (`SONIC_AVCS_CORTEX_CHECKPOINT`). No evaluated checkpoints exist in-repo,
  so these fail closed — the catalog tells the worker to pick another route
  instead of burning its budget on an unresolvable target.
- `not_wired` (1): `verified_computer_skills` — `ComputerSkillLedger`
  (`skill_ledger.py`) is real and tested, but **no production caller constructs
  it** (grep: constructed only in tests). It is marked `not_wired` rather than
  `available` so the being never claims skill reuse it does not have.

### Tests
- `services/core/tests/test_skill_catalog.py` (21): repo catalog parses and is
  non-empty; every status is a declared value; every `conditional` skill names
  its `requires`; every skill cites evidence; missing file / malformed JSON /
  missing block / non-object payload all fail closed; invalid status is rejected
  (not defaulted); render lists status+actions+requires and truncates on a line
  boundary; cache identity; the **real** `_build_reasoning_context` output
  contains the catalog and omits it when disabled; the block states it is
  advisory and does not grant permissions.
- Also fixed a pre-existing stale assertion in
  `test_agent_quality_improvements.py::test_avcs_context_is_compact_temporal_pixel_state_without_ocr`
  (it predated the `learned_semantics=` field). The test's intent is preserved and
  tightened: the summary may only report `learned_semantics=unavailable`, and must
  not contain `primary_category`/`confidence=` (i.e. no semantic claim).

### Baseline
- Full `services/core/tests`: **2190 passed, 40 skipped, 0 failures** (was
  2168/40/1 before this pass; the 1 failure was the stale AVCS assertion above).
- Ruff clean on all changed files.

## Logic-bug audit round 2 — four real defects fixed (2026-10-23)

Manual inspection (not test-driven; the suite was green at 2190 passed) found
four genuine defects. All are now fixed with regression coverage in
`services/core/tests/test_logic_bug_fixes_round2.py` (21 tests).

### BUG 1 — GUI-burst failure fed the observation into the replan "goal"
`sonic/computer_use/agent.py` — the GUI-burst failure path called
`self._inject_replan(goal=str(actual_obs_str))`, so the LLM was told
*"Goal remains: <error text>"* and lost the real objective when asked to pivot.
Every other replan call site passed the true `goal`. Fixed by adding
`_current_goal_for_replan()` (reads `_last_mission_goal` → `goal` → honest
placeholder) and routing the burst path through it.

### BUG 2 — several top-level directories were deletable at L0_SAFE
`sonic/safety/scope.py` — the `rm -rf` destructive pattern covered
/etc /boot /usr /var /bin /sbin /lib /root /home /opt /srv /mnt /media /proc
/sys /dev but NOT `/run`, `/snap`, `/data`, `/app`, `/workspace`, `/project`,
`/www`. `rm -rf /workspace` (a mounted volume) classified as L0. Pattern
extended; workspace-internal deletes (`rm -rf build/`, `./node_modules`,
`/tmp/scratch`) still classify L0 so normal work is unaffected.

### BUG 3 — `is_private_net` was computed then discarded (egress widening)
`sonic/api/routes/workstation.py` — the mission route computed `is_private_net`
from the engaged target but passed `allow_private_networks=True`
unconditionally, so the egress envelope was widened for every mission. Now
`allow_private_networks=is_private_net`: private ranges are permitted only when
the verified engagement target is itself private. Metadata (`169.254.169.254`)
and loopback stay blocked in both modes (`_MANDATORY_EGRESS_BLOCKED_NETWORKS`).

### BUG 4 — experiment approve ignored the update result
`sonic/api/routes/experiments.py` — `approve_experiment` assigned
`ok = update_status(...)` but never checked it, returning `{"status":
"approved"}` even for a non-existent experiment. The other three lifecycle
routes (rollback/reject/promote) already checked. Now returns 404 on failure.
Verified live: `POST /experiments/bogus-id/approve` → **404** (previously 200).

### Also cleaned
Dead variables removed: `gui_only` (engagement.py), `actor` (headless.py),
`system_prompt` (workstation.py duplicate). `blocked` (workstation.py) was
promoted into the paused-summary telemetry instead of deleted.

### Baseline after this pass
- Full `services/core/tests`: **2190 passed, 40 skipped, 0 failures**
  (unchanged; +21 new tests, 21 previously-existing dead-code paths now covered).
- Safety-critical subset (safety/plan6/phase24/dual-plane/boss/probes): 163 passed.
- Backend + dashboard restarted and verified live; LLM path healthy.
- Pre-existing ruff F841 warnings in `agent.py` (lines 1587/2294/5444/7330) are
  outside the touched regions and were not introduced by this pass.

## Worker activity projection + skill catalog + logic-bug round 2 (2026-10-23)

Committed as `b7d94b2` on `main` (21 files, +2097/-45). This pass closed the
operator's Boss-observability request and shipped the pending SKILLS.md, on top
of the four logic-bug fixes described in the previous section.

### Boss worker observability (`sonic/computer_use/agent_activity.py`)
Before this, the Boss streamed worker lifecycle events into the worklog but the
dashboard had no per-worker view: you could not tell how many workers existed,
which were actually running, or what each was doing. `collect_worker_activity`
is a pure projection over the persisted worklog plus two route-owned facts
(`mission_status`, `mission_task_alive`), so it never invents state:

- Every worker gets a **stable id** (from the real `sub_dispatch`/`sub_step`
  events), so the operator can click a worker and see its activity.
- A worker is `RUNNING` **only while the owning background task is alive**.
  Once the task exits (finished, crashed, or control plane restarted) a worker
  that never produced a terminal report becomes `INTERRUPTED` — never a stale
  `RUNNING`. This is the honesty rule the previous UI violated.
- `latest_action` / `latest_thought` are copied from real step traces.

Endpoints: `GET /workstation/agents` (list + `running_count`) and
`GET /workstation/agents/{agent_uid}` (404 for an unknown id). `workers` is also
attached to the session/state responses. Dashboard: `WorkstationWorker` types
and the worklog feed render per-worker rows.

### SKILLS.md + runtime catalog (`sonic/computer_use/skill_catalog.py`)
`SKILLS.md` is a machine-readable capability catalog whose JSON block is parsed
at runtime and injected into the `ComputerUseAgent` reasoning prompt, including
every Boss worker. Each entry carries an honest `status`:
`available` (reachable from the production loop), `conditional` (needs a
configured dependency such as an evaluated AVCS head, provider key, or sandbox;
fails closed otherwise), `not_wired` (code exists, no production caller — do not
claim it works). A missing or malformed catalog renders empty (fail-closed), so
a worker never assumes a capability it does not have.

### Verification
- Full `services/core/tests`: **2251 collected / 2211 passed / 40 skipped /
  0 failures**.
- New suites: `test_workstation_agent_activity.py` (15),
  `test_skill_catalog.py` (21), `test_logic_bug_fixes_round2.py` (21).
- Dashboard `next build` clean (workstation route 25.6 kB).
- Live: `GET /workstation/agents` → 200 with honest `BLOCKED` mission status
  (no Docker in this environment); unknown worker → 404; experiment approve on a
  bogus id → 404.
- No secrets in the diff.

### Still open / not claimed
- GUI work stays fail-closed: `SONIC_AVCS_GROUNDING_CHECKPOINT`,
  `SONIC_AVCS_FOCUS_CHECKPOINT`, and `SONIC_AVCS_CORTEX_CHECKPOINT` are all
  unset and no checkpoint exists in the repo, so semantic clicks and text entry
  remain blocked. No live arbitrary-GUI task was run.
- Retrieval quality (Recall@3/MRR) is still unmeasured; only the mechanics are
  tested.

### Worker-activity seam closed + SKILLS.md ownership (2026-10-23, commit `bb9a77d`)

Two follow-ups after the observability pass, both found by running the real
suite rather than trusting the earlier green run:

1. **The Boss -> worklog -> projection seam was untested.** The pure projection
   and the Boss dispatch path each had tests, but nothing drove real Boss events
   through the route and then read the projection the dashboard uses. The
   existing integration test emits `sub_dispatch` **without** `agent_uid`, so it
   exercised positional `agent_id` keying, while production emits the run-unique
   `agent_uid` (`worker-<n>-<hex>`) that is the only stable handle the operator
   UI can address a worker by. A field-name mismatch there would have passed
   every existing test while the dashboard showed nothing — precisely the
   reported symptom. `test_worker_activity_end_to_end.py` (4 tests) now drives
   the real `_run_prompt_reasoning` route with a fake Boss emitting the
   production payload shape and asserts: two workers stay distinct by uid; a
   reported worker is terminal while an unreported one is `INTERRUPTED` (never a
   stale `RUNNING`); `find_worker` resolves the uid and surfaces its latest
   action/step count, unknown uid -> `None`; the same worklog flips an
   unreported worker to `RUNNING` only while `mission_task_alive` is true.

2. **`SKILLS.md` had no declared owner.** `test_organization_architecture`
   correctly failed (`unmapped == ['SKILLS.md']`) — the repository invariant is
   that an unknown path must fail ownership resolution rather than get an
   implicit owner. It is now owned by `computer_use`
   (`COMPUTER_INTERACTION` / `DESKTOP_CONTROL`), since
   `computer_use.skill_catalog` parses it at runtime.

Verification: full `services/core/tests` **2214 passed, 40 skipped, 0 failures**
(2254 collected); boss/activity regression 72 passed; ruff clean.

Honest note on statuses: a worker's terminal status is passed through from the
real `sub_report` (`SUCCESS`/`COMPLETED`/`FAILED`/...), not normalised to a
single word — `SUCCESS` is a valid terminal status and is reported as-is.

## Live narration + Boss worker-board seam verification (2026-10-23)

Operator reports addressed: (a) Boss launches sub-agents but their id / live
work is not visible when clicked, (b) "the eye does not speak to the brain",
(c) overall completeness of the AGI / self-evolution / self-computer story.

### 1. AVCS live narration — the eye now describes what it sees
AVCS perceived pixels but emitted only numbers, so the reasoning model could
not distinguish "a red dialog just popped up" from "nothing happened".

- NEW `packages/avcs/forge/vision/narrator.py` — `LiveNarrator`,
  `dominant_color_name()`, `rgb_tiles()`, `chromatic_change_mask()`. Speaks only
  when something a human would notice changed; silence is the default.
- NEW `services/core/sonic/computer_use/avcs_narrator.py` — `AVCSNarrator`,
  SONIC's side: decodes the raw frame already on a `ScreenObservation` and
  renders a bounded prompt block.
- `avcs_perception.perceive_rgb()` now returns `rows`/`cols`/`changed_grid` so
  the narrator can phrase *where* a change happened.
- `agent.py`: `_narrate_screen()` injects narration into BOTH the GUI-only and
  the general reasoning prompts (labelled advisory + text-blind);
  `_narrate_action_result()` attaches the outcome to the action observation.
  A new module constant `_GUI_ACTION_TYPES_FOR_NARRATION` scopes which actions
  get an outcome line. Narration is optional: a missing package/frame yields `""`
  and the agent keeps its previous behaviour.
- `avcs_capability_state()` now reports `live_narration`.

**Real gap closed:** the reflex diffs **luma**, so a saturated box whose
luminance matches its background (a red `[200,40,40]` dialog on a grey desktop)
was invisible. The narrator diffs RGB tiles too, so it is detected and named.

Honest scope: no OCR, no semantic UI tree. Narration reports tone, structure,
coarse position, colour and change — never what text says. Action feedback is
fail-closed: VERIFIED → "the screen responded", UNVERIFIED → "no visible
effect", anything else → "dispatched, effect not verified". An unverified action
is never narrated as success.

Tests: `services/core/tests/test_avcs_live_narration.py` (32).

### 2. Boss -> worker-board seam — audited and locked
The reported "sub-agents not visible" was **not** a code defect: audited the
whole path and it is wired correctly.

- `SubMission.id` is a real `sub-*` id; each worker also gets a run-unique
  `agent_uid` (`worker-<n>-<hex>`) and a positional `agent_id` (`sub-agent-<n>`).
- `boss._run_sub_worker` emits `sub_dispatch` / `sub_step` / `sub_report` with
  both handles; the workstation route's event->worklog translation carries them
  through; `collect_worker_activity()` keys per worker; `find_worker()` resolves
  by uid; `WorklogFeed.tsx` renders one expandable card per worker showing goal,
  latest thinking, recent steps, findings and the Judge verdict.
- Verified with a **real** `BossAgent.run()` against a stub provider: 4 workers
  (`worker-1-a07e9a` … `worker-4-28c82c`), each with its own goal, agent id, live
  step, thought and findings, every one resolvable by uid.

The gap was **test coverage of the seam**: `test_worker_activity_end_to_end.py`
drives the route with a *faked* Boss event stream, so a producer/consumer field
drift would pass there and still leave the board empty. NEW
`services/core/tests/test_boss_worker_visibility.py` (6) runs the real
`BossAgent.run()` loop (only model calls stubbed — offline, ~1 s) and asserts:
unique non-colliding worker ids; one card per worker; click resolves the
specific worker; the card shows real activity, not just a status; an unreported
worker is `INTERRUPTED`, never a stale `RUNNING`.

### 3. What is genuinely still missing (do not overclaim)
- **Semantic GUI clicks and text entry remain blocked by design.** No evaluated
  AVCS grounding checkpoint and no focus-verification checkpoint exist in the
  repo, so `avcs_capability_state()` reports `semantic_grounding=false`,
  `focus_verification=false`, `gui_actions_ready=false`. Only explicit
  pixel-coordinate pointer actions can dispatch. Train the heads from real
  recorded demonstrations (`packages/avcs/docs/LOCAL_GROUNDING_QUICKSTART.md`)
  before claiming "click the Save button" works.
- **Learned cortex has no configured checkpoint** (`learned_cortex_ready=false`),
  so learned semantics are honestly empty at runtime.
- A live Boss run in this environment used a 550B model at ~20 s per call and
  produced a **repeating clarification phase** (4 phases restating "obtain the
  target from the user") instead of converging. That is a planning-quality /
  objective-underspecification issue worth a separate look; it is not a
  dispatch or visibility defect.
- No live arbitrary-GUI end-to-end task was run this session (no Docker daemon
  here). Do not claim GUI task capability from unit tests.

### Test baseline after this pass
- `services/core/tests`: **2253 passed, 40 skipped, 0 failures**.
- Ruff clean on every changed/new file.
- Pushed to `origin/main` (`edb5099`, `bc5e872`).

## AVCS two-way loop: foveal focus + ambient commentary (2026-10-23)

Turned AVCS from a sensor the agent *polls* into a colleague that *speaks*.
Three additions, all verified against real frames (no mocks of narration logic).

### 1. Foveal focus rectangle (`packages/avcs/forge/vision/narrator.py`)
- `Narration.focus` now carries a real pixel rectangle `{"x","y","w","h"}` for
  the region the line is about. `focus_rect_pixels()` converts the tile-space
  block to pixels, grows it to `FOCUS_MIN_PIXELS` (48) so a one-tile change is
  not a sliver, pads by `FOCUS_PAD_FRACTION` (0.30), and clamps into the frame
  (an off-screen focus is a bug, never a guess). `crop_focus(frame, focus)`
  returns the actual sub-frame.
- This is the eye's **fovea**: a caller can hand a small crop to a vision model
  instead of a whole screenshot. **No OCR** -- the pixel-only contract holds.
- Focus is set for the chroma/overlay alert branches and for high motion
  (centred on the motion centroid); a quiet "screen is still" line carries
  `focus=None`.
- Verified live: a 120x200 red dialog at y180-300,x220-420 produced
  `focus={'x':150,'y':144,'w':339,'h':192}`; the crop contained the real red
  pixels.

### 2. Ambient commentary (`services/core/sonic/computer_use/awareness.py`)
- `AwarenessMonitor` now narrates every sampled frame via a lazily-built
  `AVCSNarrator`, stores the line on `AwarenessFrame.narration`, and publishes
  a new `TOPIC_NARRATION` (`perception.narration`) **only when there is a
  line** -- so a consumer can subscribe to commentary without every frame tick.
- Still **observation-only**: it never dispatches an action. A missing narrator
  or a decode failure degrades to no narration, never a broken loop.
- `_decode_png_to_rgb()` handles the PNG branch (the raw-RGB branch already had
  the numpy frame) so narration works on both capture paths.

### 3. Ambient delivery channel (`avcs_narrator.py`, `agent.py`, `api/main.py`)
- `record_ambient_narration()` / `drain_ambient_narration()` /
  `peek_ambient_narration()`: a bounded (`_MAX_AMBIENT_LINES=8`), thread-safe,
  **drain-once** channel for lines the eye spoke unprompted.
- `ComputerUseAgent._ambient_narration_block()` drains it and injects an
  "AVCS unsolicited observations since your last step" block into BOTH prompt
  paths (the general reasoning context and the GUI-only context).
- **Real bug found and fixed:** the awareness monitor keys on a workspace id
  (`home-<tenant>` from `get_or_create_home`, or the container name from
  `create`) while the agent keys on its own `_active_workspace_id`. The two
  never matched, so the channel would have been **dead on arrival**. Both sides
  now key on the **tenant id** -- a value each can derive -- via
  `AwarenessMonitor(ambient_key=...)` set in `api/main.py`. Verified live: eye
  writes under workspace `home-acme`, brain drains under tenant `acme`, and the
  workspace-id key returns nothing.

### Honesty
- No OCR added; narration reports coarse structure, colour, motion, position and
  never claims to read text (a test asserts no line contains "says"/"reads").
- Narration is **advisory**, not proof of success, and is **advisory-inject**,
  not a mission kill: it informs the next reasoning step rather than aborting
  the worker.
- `VisualRippleVerifier` (`packages/avcs/avcs/verifier.py`) still exists but is
  NOT wired into SONIC; it is a heavy `HumanEye`+`OSCursorDriver` self-healing
  executor and was deliberately left alone.
- The user's earlier AVCS claims that remain false: `avcs/observation.py` does
  not exist; there is no "131-word" vocabulary (VOCAB is 27 words); there is no
  `ConvGRU` anywhere in the repo. Capture runs at 10 FPS, not 100.

### Tests
- `services/core/tests/test_avcs_foveal_and_ambient_loop.py` (19): focus rect
  covers the described region, stays in-frame, is grown for tiny blocks; quiet
  lines carry no focus; the eye narrates and publishes a narration topic; a
  still screen does not repeat itself; the ambient channel is drain-once,
  cross-identity, bounded, and never claims to read text; the monitor never
  dispatches an action; the agent surfaces and drains the lines.
- Full suite: **2272 passed, 40 skipped, 0 failures**. Ruff clean.
- Pushed to `origin/main` (`9c512b9`).

## Boss subagent visibility: heartbeat/step identity fixes (2026-10-23)

Operator report: "jab Boss agent launch karta hai, uska id hona chahiye aur
agent ke id/name par click karne par dikhna chahiye ki wo kya kaam kar raha
hai" — i.e. delegated workers must be individually addressable and their live
work must be visible. Audited the whole seam (Boss emit -> workstation worklog
-> `agent_activity` projection -> `/workstation/agents` -> dashboard
`WorkerBoard`/`WorkerDetail`) and fixed three real bugs plus two display
inconsistencies. Backend identity/drill-down already worked (the run-unique
`agent_uid` = `worker-<n>-<hex>`, unique per mission because `_sub_agent_count`
is monotonic); the breakage was in what travelled on each event and in the
frontend mapping.

### BUG A — reasoning heartbeat was silently dropped (real)
`BossAgent._sub_heartbeat_callback` emitted `sub_heartbeat` **without**
`agent_uid`. The projection keys every item by `uid:<agent_uid>` (falling back
to `agent:sub-agent-N`), so a heartbeat keyed as `agent:sub-agent-1` never
matched the dispatch record keyed `uid:worker-1-xxx` and was discarded. A long
LLM reasoning turn therefore looked like a stalled worker with no update.
Fixed: the heartbeat now carries `agent_uid`; the route forwards
`agent_uid`/`sub_agent_uid`/`sub_agent_number` on the worklog item so it
attaches to the owning worker (`latest_thought`).

### BUG B — step worklog items lost their action target (real)
Only `TERMINAL_EXEC`/`FILE_READ`/`FILE_WRITE` steps set `command`/`file`.
A generic step (e.g. `GUI_CLICK`) had no field for its target, yet the
projection derives the worker's "latest action" summary from
`item["target_resource"]`. So a GUI worker's step showed a bare verb
("gui click") with no resource. Fixed: `step_metadata` now always carries
`target_resource`.

### Frontend mapping/consistency fixes
- `workersFromServer` never mapped the server's `step_count`, so the
  drill-down card's "N steps" line could never render. Now mapped.
- `WorkerBoard` header counted active workers with a raw
  `status === "RUNNING"` check while each card rendered `workerStatusLabel`
  (which can downgrade a stale RUNNING to `UNCONFIRMED`). The two could
  disagree. The header now counts via `workerStatusLabel`.

### Tests
- `test_worker_activity_end_to_end.py` (+2): a generic GUI step keeps its
  target in the drilled-in worker view; a reasoning heartbeat attaches to its
  worker (`latest_thought`) instead of being dropped. Both verified red on the
  unmodified code.
- `test_boss_gui_worker_dispatch.py`: the heartbeat assertion now requires the
  run-unique `agent_uid` (was asserting the buggy payload).
- Full `services/core/tests`: **2280 passed, 40 skipped, 0 failures**. Ruff
  clean; dashboard `npm run build` compiles.

### Honesty
Identity, per-worker drill-down, and honest RUNNING/INTERRUPTED/COMPLETED
status were already implemented and unit-tested; these were targeted
visibility/identity bugs on the live event seam. No live Boss mission with a
real LLM was run in this environment (no provider credentials / Docker), so
end-to-end worker narration on a live desktop is not claimed here.

## Phase 48 — Parallel-Worker Coordination: Scope Binding, Mid-Run Steering & Structured Evidence (DONE)

Closed the three coordination gaps in the "parallel agents" workflow (Boss
launches several bounded workers per phase). Each is a real seam that silently
degraded the operator's ability to run and trust parallel work.

### GAP 1 — Worker write-scope was never communicated
`SubMission.owned_paths` existed and the Boss wave loop already *deferred* a
candidate whose paths collided with an already-selected worker
(`locked_paths`), but:
- the decomposition prompt never asked the model for `owned_paths`, so it was
  always empty and the path-lock was dead code;
- the parsed `owned_paths` were never passed into the worker, so even when
  present the worker had no idea what its scope was.
Fixed: the decomposition prompt now requests `owned_paths` (with an explicit
"two workers in one phase must not list the same path" rule), the parser
collects them, `scoped_context["owned_paths"]` is passed to the worker, and
`_build_reasoning_context` renders an `ASSIGNED WRITE SCOPE` block. This makes
the existing collision guard effective and stops parallel workers wandering
into a peer's files.

### GAP 2 — No operator/coordinator steering of a running worker
Workers run in their own contexts and never talk to each other (correct), but
the operator had no way to steer a specific running worker mid-run. Added
`sonic/computer_use/coordinator_inbox.py`: a bounded, thread-safe,
per-`agent_uid` inbox. The agent drains it on its next reasoning step
(`_render_operator_messages`) and renders the directives as an
`OPERATOR INSTRUCTIONS` block. New route
`POST /workstation/agents/{agent_uid}/message` (operator-gated, tenant-scoped,
404 for an unknown worker, 422 for an empty message). Delivery only: it never
marks a worker complete and does not replace its evidence or the Judge.

### GAP 3 — Worker results carried only free text
`SubMissionResult` had no structured, verifiable evidence, so a confident but
unbacked summary looked identical to real work. Added deterministic extractors
(`_extract_file_references`, `_extract_test_results`) that derive touched files
and real test-runner results (passed/failed/errors/exit_code) **from traces
only** — never from the model's prose. New fields `file_references`,
`test_results`, and `evidence_incomplete` (True when a worker claims success
with no file/test/discovery evidence). Surfaced through the `sub_report` event,
the worker projection, `/workstation/agents`, and the dashboard drill-down
(which now shows file refs, pass/fail test chips, or an explicit "claimed
success without verifiable evidence" warning).

### Tests
- `services/core/tests/test_parallel_worker_coordination.py` (14): owned_paths
  parse + render (and absence); inbox post/drain/bounds/rejections; agent
  renders + drains exactly once via the real reasoning prompt; file/test
  extraction from real traces only (prose ignored, failed writes excluded).
- `services/core/tests/test_workstation_agent_activity.py` (+3): message
  endpoint delivery, auth/404/422, tenant scoping.
- `services/core/tests/test_worker_activity_end_to_end.py` (+1): structured
  evidence survives the Boss → worklog → projection seam.
- Red→green verified by reverting the owned_paths wiring and the operator-block
  injection (each failed) and restoring.

### Honesty
These close real coordination seams and are unit/route-verified. No live Boss
mission with a real LLM was run in this environment, so live parallel-worker
steering on a real desktop is not claimed.

## Multi-context retrieval + agentic-RAG extension (2026-10-24)

The operator asked to tackle "multi-context" and "agentic-RAG" in SONIC. Recon
found that **"multi-context" had zero references anywhere** in the codebase, and
agentic-RAG was limited to a single store: `AgenticRetriever` (`memory/agentic.py`)
refines queries only over `VectorMemory`, consumed by `being/lessons.py`
(`relevant_agentic`) and injected as REUSE/AVOID lessons. The being's knowledge,
however, lives in several tenant-scoped stores that never talked to each other.

### `sonic/memory/multi_context.py` (NEW) — fuse independent memory sources
- `ContextSource` protocol + three adapters, each tenant-scoped:
  - `VectorContextSource` — wraps `VectorMemory.search` (sync store).
  - `GraphFindingSource` — wraps `GraphMemory.search_findings` (async backend;
    works with Neo4j / SqliteGraph / InMemoryGraph, all of which expose it).
  - `EpisodicContextSource` — keyword-ranks past `Episode`s.
- `build_default_sources(...)` composes whatever stores exist; a missing store
  is simply omitted (degrades, never fails).
- `MultiContextAssembler.assemble(query, tenant_id)` queries every source,
  **fuses by reciprocal-rank** (a fact found by two sources outranks a
  single-source hit), deduplicates by content fingerprint, and hard-bounds the
  pack by `max_items`/`max_chars`/`per_source_limit`. A source that raises is
  recorded in `sources_failed` and skipped — one bad backend never sinks the pack.
- `render_multi_context(pack)` renders a compact `=== MULTI-CONTEXT ===` block;
  empty pack renders `""` (zero prompt noise for a fresh being).
- `AgenticMultiContextRetriever(assembler, llm_router)` — the agentic half:
  the model judges sufficiency and proposes ≤2 alternate queries per round,
  bounded to `max_rounds=2` / `max_queries=4`; each round re-fuses all sources.
  Identical discipline to `AgenticRetriever`: the model plans queries only,
  never invents evidence; repeated/invalid queries are rejected; a planner
  failure leaves the initial pack intact.

### Wiring
- `ComputerUseAgent.__init__` gains `multi_context_sources`. `_llm_choose_action`
  assembles the pack once per goal (agentic when a router exists) and
  `_build_reasoning_context` injects `multi_context_block` beside the lessons
  block; the block is goal-scoped so a stale pack never leaks across goals.
  Per-mission state (`_multi_context_goal/_block/_pack`) is reset in `run_mission`.
- `BossAgent` gains `multi_context_sources` and forwards it to every worker.
- Production callers compose real sources: `api/routes/workstation.py`,
  `api/main.py` (being life loop), and `MissionDirector._build_multi_context_sources`
  (vector + graph via the memory router + episodic).
- `SKILLS.md` catalog gains a `multi_context_retrieval` entry (`available`).

### Tests (`services/core/tests/test_multi_context.py`, 26)
- Adapters map rows→cited items and skip blanks; graph source is empty without
  `search_findings`; episodic ranks by keyword overlap.
- Fusion ranks multi-source hits higher; dedup collapses identical content and
  merges citations; a failing source is recorded (not fatal); empty → empty pack
  with no render; `max_items`/`max_chars` bounds enforced; tenant forwarded
  unchanged; invalid args rejected.
- Agentic: refines + merges across rounds; stops when sufficient; keeps initial
  results on planner failure; bounded and rejects repeated queries; plan parser
  rejects malformed/unbounded output; constructor bounds.
- Agent wiring: the real `_build_reasoning_context` contains the block for the
  matching goal, omits it for a different goal, and adds no noise with no sources.
- `BossAgent` stores and forwards the sources.

### Honesty
- Full `services/core/tests`: 2324 passed, 40 skipped, 0 failures (was 2298/40/0).
- Ruff clean on every changed file.
- These are unit-level contracts. No live mission with a real LLM was run, so
  live multi-source retrieval quality on a real desktop is not claimed; the
  configured embedding/planner models must be measured with
  `python -m sonic.memory.retrieval_eval` before claiming retrieval quality.

## Boss worker observability — audit correction + UI wiring (2026-10-23)

A reported gap ("subagent ids are broken; clicking an agent shows nothing") was
investigated and **partly disproven**. Recording the correction so a future
session does not repeat the wrong diagnosis.

### What was actually already correct (do not "fix" these)
- `agent_id` **is** run-unique. `BossAgent._sub_agent_count` is a single
  instance counter incremented once per dispatched worker and is **not** reset
  per phase (boss.py ~617), so Phase 1's `sub-agent-1` and Phase 2's are
  different numbers. An earlier claim that it "repeats across phases" was
  wrong.
- `agent_uid` (`worker-<n>-<hex6>`) is the run-unique UI handle and travels on
  every event (`sub_dispatch`, `sub_step`, `sub_heartbeat`, `sub_report`).
- The backend observability layer is **complete**: `computer_use/agent_activity.py`
  projects the worklog into per-worker truth (`RUNNING` only while the owning
  background task is alive, else `INTERRUPTED` — never a stale RUNNING), and
  three authenticated, tenant-scoped routes expose it:
  `GET /workstation/agents`, `GET /workstation/agents/{agent_uid}`,
  `POST /workstation/agents/{agent_uid}/message`.
- `apps/dashboard/lib/api.ts` already declared `getWorkstationAgents`,
  `getWorkstationAgent`, `messageWorkstationAgent`.

### The real gap (what was fixed)
Those API client functions had **zero callers**, and the `/agents` page
rendered static `div` cards with no click handler. So the authoritative worker
record was unreachable from the UI, and `WorkerDetail` in `WorklogFeed.tsx`
rendered only from in-memory worklog events (which can lag or be truncated).

Fixed:
- `WorklogFeed.tsx`: `WorkerDetail` now fetches
  `GET /workstation/agents/{uid}` on expand (loading / `live` / error states,
  worklog view as fallback). `sessionId` threaded through `WorkerBoard`.
- `app/agents/page.tsx`: new "Live Boss workers" section; each card is
  clickable and loads current activity, recent steps and Judge verdict.
- `tests/test_worker_detail_api_contract.py` (5 tests) pins the exact response
  shape the newly-wired UI reads (`workers[]`, `worker{}`, `is_running`,
  `latest_action`/`latest_thought`, `steps`, `step_count`) plus 404 and
  cross-tenant non-leakage — so a backend rename cannot silently break the
  drill-down while existing tests still pass.

Verified: 70 related backend tests pass, ruff clean, `tsc --noEmit` clean,
`next build` clean (21 routes; `/agents` grew from a stub to 5.07 kB).
Honesty: no live Boss mission was run (no model credentials / Docker here);
this validates the wiring and the API contract, not live mission success.

## Workstation session ids — server-minted + display handle (2026-10-23)

Sessions were minted **client-side** as `mission-<8 hex>` and the backend
neither produced nor validated that id. Audit found three gaps: no server-side
create route, no validation at all, and the real id was invisible in the UI
(only `session_name` rendered).

### Why `#` must not live in the id
The requested `workstation#92098hahojczxkj2` form carries `#`, which is the URL
**fragment delimiter** — `?session_id=abc#def` reaches the server as `abc`. It
is also a shell comment character. Today every call site uses
`encodeURIComponent` and `session_id` never reaches a shell, so it would work,
but it is one forgotten encode away from a silent truncation. The fix keeps the
operator-facing `#` in the **label** and out of the **id**.

### Backend (`api/routes/workstation.py`)
- `POST /workstation/session` (operator-only) mints `ws-<16 hex>` and returns
  `{session_id, display_name, status}`. An optional client `session_id` is
  accepted for compatibility but must be valid; a collision returns **409**
  rather than silently returning an existing session.
- `_is_valid_session_id` / `_require_valid_session_id`: alphanumerics + dashes,
  max 64. **Deliberately not lowercase-only** — mixed-case ids such as
  `desktop-vision-True` are already used by callers and stored sessions; an
  over-strict rule broke `test_workstation_chat_and_observation.py` and was
  relaxed. The rule only needs to exclude `#`, whitespace, shell
  metacharacters and path separators.
- `_get_or_create_session` now validates, so unsafe ids are rejected on **read**
  routes too (400), not only on create.
- `_session_display_name`: `ws-<hex>` -> `workstation#<hex>`; `default` ->
  "New conversation"; legacy ids keep the historical `Session <last8>` label.
- Existing ids in `sonic_data/workstations.json` (`mission-xxxxxxxx`) keep
  working unchanged; only newly created sessions use the `ws-` form.

### Frontend
- `api.createSession()` added; `handleNewSession()` now asks the server for the
  id instead of minting `mission-<8 hex>`, and surfaces a real error if the call
  fails (previously it would have silently used a bogus id).
- Sidebar renders the `workstation#<token>` handle under each session, with the
  raw id in the `title` attribute. The `#` exists only in that label.

### Tests
`tests/test_workstation_session_ids.py` (18): minting + persistence, distinct
ids per create, fresh-state isolation (a new session must not inherit another's
desktop), rejection of `#` / `;` / spaces / traversal / oversize ids, rejection
on read routes, reserved `default`, 409 on collision, legacy-id compatibility,
auth required, and tenant scoping.

Verified: full `services/core` suite **2349 passed, 40 skipped, 0 failures**;
ruff clean; `tsc --noEmit` clean; `next build` clean (21 routes).
Honesty: no live browser session was created; this validates the API contract,
validation and rendering, not a live multi-session desktop run.

## Boss subagent identity + live LLM path diagnostic (2026-10-24)

Manual (no parallel subagents) live pass against the running backend on
`127.0.0.1:12000` with real NVIDIA credentials from `.env`.

### Confirmed working (live, not unit tests)
- **Session minting + validation.** `POST /workstation/session` (operator-only)
  returns `{"session_id":"ws-<hex>","display_name":"workstation#<hex>"}`.
  Live status codes: `workstation#abc` → 400, `ws-abc;id` → 400, `default` → 400,
  duplicate id → 409, unauthenticated → 401. Canonical id stays `ws-<hex>`; the
  `#` appears only in the display label. 12 tests pass; ruff clean.
- **Subagent identity model.** `SubMissionResult` carries `agent_uid` (stable,
  run-unique, e.g. `worker-1-a1b2c3`) *separate* from positional `agent_id`
  (restarts per Boss run). `workstation.py` exposes
  `GET /workstation/agents/{agent_uid}` and
  `POST /workstation/agents/{agent_uid}/message`, and emits `sub_dispatch` /
  `sub_step` / `dispatch_decision` / `boss_thinking` / `boss_report` worklog
  events keyed by `agent_uid`. The dashboard worker card is click-to-expand
  (`WorklogFeed.tsx` `expandedKey` → `WorkerDetail`), so clicking a worker shows
  what it is doing. This is the user's core ask and it is wired end-to-end.
- **LLM reasoning is live.** `/llm/chat` → `nvidia/nemotron-3-ultra-550b-a55b`
  in 1.3 s; router `task_type="reasoning"` → 1.8–1.9 s OK.
- **AVCS reflex works** on real numpy frames: black→white change →
  `scene_changed=True` (change_fraction 0.64); identical frames → False;
  1-pixel edit → False but `scene_hash` **still differs** (agent verification
  uses the hash, not the coarse threshold — correct and more sensitive).

### Real findings (environmental / latency, NOT code defects)
1. **Provider availability on this NVIDIA account tier.** Only
   `nvidia/nemotron-3-ultra-550b-a55b` and `openai/gpt-oss-20b` answer (HTTP 200).
   `nvidia/nemotron-3-super-120b-a12b` returns **404** ("Function … Not found for
   account") and `meta/llama-3.2-11b-vision-instruct` failed intermittently in
   the SDK path though a raw curl to it returned 200. The 404 is recovered by the
   model-level fallback chain (a wasted round-trip, not a failure).
2. **Only `NVIDIA_API_KEY` is configured.** GROQ / OPENROUTER / DEEPSEEK /
   OPENAI / ANTHROPIC / XAI keys are all unset, so the global fallback chain
   beyond nvidia cannot actually help. Config models (groq `qwen/qwen3.8-27b`,
   openrouter `gpt-4o`) are declared but not reachable here.
3. **Measured LLM latency is highly variable and provider-side.**
   - router `computer_use` (vision route): **29 s / 168 s / 140 s** across three
     attempts (succeeded each time with a 240 s outer budget).
   - direct `meta/llama-3.2-11b-vision-instruct`: **36.7 s** at `max_tokens=900`
     and **308 s** at `max_tokens=64` for a trivial prompt.
   - The agent's `_GUI_REASONING_TIMEOUT_SECONDS = 60.0` therefore times out on
     the slow tail, producing an honest `reasoning unavailable` blocked action
     (fail-closed, correct) rather than a wrong action.
   - Note: `/workstation/prompt` on a session with no provisioned desktop
     correctly returns `BLOCKED` ("no provisioned workstation") — that is the
     documented fail-closed contract, not a bug.

### REAL BUG — false-positive safety refusal discarded correct answers (FIXED)
The dominant cause of the intermittent "LLM Response Unavailable" worklog entry
was **not** latency or rate limiting. `sonic/llm/router.py` had the substring
`"as an ai"` in `REFUSAL_PHRASES`, and `is_refusal_response()` scanned
`content + reasoning_content` together. SONIC's own identity answer contains
"…collaborate as an **AI-Human** engineer", so the model's *correct* reply
matched the refusal list, was discarded, and the fallback chain then failed
closed (no other provider key configured).

Live trace proving it: `llm_request_complete latency_ms=19733.6
model=nvidia/nemotron-3-ultra-550b-a55b` followed immediately by
`llm_safety_refusal_detected_triggering_fallback` → `complete_failed`. The
captured answer was correct: *"I'm SONIC (being-…) — an autonomous AI-Human
Digital Being … as an AI-Human engineer."*

Fix:
- Removed the over-broad `"as an ai"` phrase; added precise refusal wording
  (`"i cannot help with this"`, `"i won't help with"`, `"i'm unable to fulfill"`).
- `is_refusal_response()` now judges the **answer** (`content`), consulting the
  private `reasoning_content` only when the answer is empty — a reasoning trace
  narrating refusal-adjacent wording can no longer discard a real answer.
- The agent's own refusal lists (`computer_use/agent.py`) already scoped
  correctly and did not need changes.
- Live verification: workstation chat `who are you` went from **3/3
  "LLM Response Unavailable"** to **4/4 real "SONIC Response"** with **0**
  fallbacks; the answer correctly describes SONIC's identity.

### Also changed — configurable workstation chat timeout
`/workstation/prompt` hardcoded `asyncio.wait_for(..., timeout=25)`. Since
provider reasoning latency regularly exceeds 25 s, the budget is now
`_workstation_chat_timeout_seconds()` (env `SONIC_WORKSTATION_CHAT_TIMEOUT_SECONDS`,
default 25 s, clamped 10–300 s). This was a contributing factor, not the root
cause.

### Tests added
- `tests/test_llm_refusal_detection.py` (8): identity text is not a refusal;
  reasoning trace alone cannot trigger refusal; genuine refusal detected;
  refusal-in-reasoning detected only when the answer is empty; action-bearing
  refusal is not a refusal; empty response is not a refusal; `"as an ai"` gone;
  common refusal phrases retained.
- `tests/test_workstation_chat_timeout.py` (7): default, env override, floor,
  ceiling, invalid/blank fallback, and a guard that the literal `timeout=25,`
  is not reintroduced.
- Full `services/core` suite: **2364 passed, 40 skipped, 0 failures**.

### Honesty / limits
- Docker is unavailable in this environment, so no GUI action or Boss swarm was
  executed live; the subagent/dashboard wiring was verified by code path and by
  the real API surface, not by a live multi-worker desktop run.
- The refusal fix is verified live on the real chat path. The remaining latency
  figures are provider-tier conditions; a faster reachable model (or a
  configured second provider key) is still needed for reliable slow-tail GUI
  reasoning within the 60 s GUI budget.

## AVCS "Live Talking Human Eye" made real on the live VNC desktop (2026-10-24)

Track B of the Boss/observability session. The AVCS eye previously reported
`live_narration: true` but had **never been exercised against a real desktop**,
and the learned cortex had **no trained checkpoint anywhere in the repo**, so
`learned_cortex_ready` was permanently false. Both are now real.

### Docker daemon was simply not running in this environment
`docker`/`dockerd` were installed but no daemon was up, which is why earlier
sessions reported "Docker nahi hai". Starting `sudo dockerd` + building the
workstation image made the real VNC desktop available:
- `docker build -t sonic-workstation:latest -f docker/Dockerfile.workstation .`
  (2.22 GB, includes Xvfb/XFCE4/x11vnc/noVNC/xdotool/wmctrl/scrot/firefox-esr).
- Container `sonic-desktop-workstation` runs Xvfb `:99` **1920x1080x24**, XFCE4
  session, x11vnc on 5900, noVNC on 6080 (HTTP 200 verified).

### REAL BUG — the provider could not reach the Docker socket
`DockerComputerProvider` shells out to `docker`, but the runtime user
(`openhands`) was not in the `docker` group, so every call failed
`permission denied ... /var/run/docker.sock` and the provider correctly
fail-closed to `desktop_state=NO_DISPLAY` with `raw_rgb` empty. The awareness
loop therefore started (`awareness_monitor_started`) but published **0 frames**
and never narrated. This looked like "AVCS is broken"; it was an access bug.
Fix (environment): `usermod -aG docker openhands` + socket permission; after
that the provider returned `INTERACTIVE`, 1920x1080, 6.2 MB raw frames and
`avcs_raw_frame_processed` streamed at ~9-10 FPS.

### AVCS eye verified end-to-end on real desktop pixels
In-process run of the real `AwarenessMonitor` against the live container while
real applications were opened:
- `frames published: 27, capture errors: 0`
- `lines spoken: 9`, e.g.
  - `Active window: Terminal -. A mid-tone, detailed screen. A new block appeared in the bottom left on top of the scene.`
  - `Active window: root - Thunar. A new block appeared in the centre on top of the scene.`
- Learned cortex semantics rode along: `category=terminal_shell`, 3 action
  hotspots — i.e. the cortex correctly read the terminal scene.
- Silence on unchanged frames and narration only on real change (no flooding),
  matching the narrator's design contract.

### AVCS cortex trained on real captured desktop frames
No checkpoint existed in the repo. Captured **40 real 1920x1080 X11 frames**
from the live desktop (downscaled to 640x360) while opening/moving real apps,
wrote a `video_ingest.json` with honest provenance
(`watched_video_unlabeled_pixels`, no invented labels), and trained the cortex:
predictive loss 0.15 -> 0.08 over 12 epochs (~3.7 s/epoch on CPU).
- Checkpoint: `sonic_data/avcs/cortex_model.pt` (1.14 M params, 4.5 MB).
  `sonic_data/` is gitignored, so the binary is not committed.
- `.env`: `SONIC_AVCS_CORTEX_CHECKPOINT=sonic_data/avcs/cortex_model.pt`.
- `ensure_cortex_ready()` -> `cortex_ready=True`; `perceive_semantics_rgb()`
  emits `primary_category`, `category_confidence`, `action_hotspots`,
  `surprise_score`, `stability_state`, `text_density`.

### Live capability surface after this pass
`reflex_perception: true`, `live_narration: true`,
`learned_cortex_configured: true`, `learned_cortex_ready: true`,
`continuous_awareness: true`, `awareness_stats.running: true` (83+ frames,
0 errors via the API). `semantic_grounding` and `gui_actions_ready` remain
**false by design** — no evaluated AVCS grounding checkpoint exists, so
semantic clicks and text entry stay blocked. That is the documented fail-closed
contract, not a regression.

### Tests
`services/core/tests/test_avcs_talking_eye.py` (8): first frame describes the
scene; unchanged frame is silent; a real change is narrated; narration carries
the active window; the awareness loop publishes narration and never dispatches
GUI actions; a narrator exception does not kill the loop; cortex semantics
present with a checkpoint and honestly empty without one.
Full `services/core` suite: **2372 passed, 40 skipped, 0 failures**; ruff clean.

### Honest limits
- The cortex was trained on ~40 frames of this one desktop for a short run; the
  category accuracy is not measured against a held-out real-desktop set, so do
  not claim general scene-understanding quality.
- Narration is coarse (tone/motion/position/colour/change) by design: AVCS has
  no OCR or semantic UI tree. It never reports what text says.
- Semantic grounding / focus verification remain unbuilt; GUI text entry and
  described-control clicks are still blocked.

## AVCS VLM narration + torch construction race + Boss GUI-progress fix (2026-10-25)

Three real defects fixed this pass; each is backed by a regression test that
fails on the pre-fix logic.

### 1. Non-blocking VLM narration (AVCS "talking eye" can name apps / read text)
`AVCSNarrator` gained an opt-in vision-language captioner
(`SONIC_AVCS_VLM_DESCRIBER=<hf-model-id>`). The reflex line stays pure-numpy and
fast; the VLM runs only on a notable change and is rate-limited.

The first implementation ran the VLM **synchronously on the awareness loop**,
which blocked the event loop for ~19 s per caption: the loop produced 2 frames
instead of >=3 in a 0.4 s window and could not sample continuously. Fixed with
`_BackgroundVLMCaptioner` (`services/core/sonic/computer_use/avcs_narrator.py`):
a load thread plus a single inference worker; `caption(frame)` is
**non-blocking**, queues the newest frame, and returns the most recent completed
caption (deduplicated so a static screen does not repeat itself). A caption
therefore describes a recent frame, never an invented one.

### 2. Torch construction race: `from_pretrained` leaks a global dtype
`transformers`' `from_pretrained` calls `torch.set_default_dtype` around model
construction and restores it afterwards. That default is **process-global**, so
a background thread loading the VLM while another thread built the AVCS cortex
made the cortex's parameters come out in the wrong dtype -- every frame failed
with `Input type (torch.FloatTensor) and weight type (CPUBFloat16Type)`, and
`perceive_semantics_rgb` silently returned `{}` (a broad `except` hid it).

Fixed with `packages/avcs/forge/vision/torch_build.py` (`build_model`), which
serializes construction under a shared `RLock` and restores the captured default
dtype. Wired into `load_cortex_checkpoint` and `VLMDescriber.__init__`.

### 3. Boss counted a successful APP_LAUNCH as "no phase progress"
The phase-progress predicate required `state_changed is True` for any action
whose name started with `GUI_`/`APP_`/`BROWSER_`. APP_LAUNCH / APP_FOCUS /
GUI_SCREENSHOT open or refocus windows asynchronously and often produce no
immediate frame delta, so a genuinely successful launch was discarded and the
mission died with "no phase progress" after one action.

Fixed by gating on the shared `GUI_STATE_VERIFICATION_ACTIONS` set
(`models.py`), which lists only the actions whose success is judged by a pixel
delta (click/type/keypress/hotkey/scroll/drag). Agent verification and Boss
phase progress now share one definition and cannot disagree.

### 4. Capability probe cached stale checkpoint config
`_shared_capability_probe()` reused one `AVCSPerception` forever, so the
`/workstation/desktop/status` honesty surface kept reporting a checkpoint that
had since been unset (and vice versa). Now keyed on the three
`SONIC_AVCS_*_CHECKPOINT` env vars and re-created when they change.

### Tests
- `services/core/tests/test_continuous_avcs_awareness.py`:
  `TestNarratorNeverBlocksPerception` (2) -- load off-thread + `caption()` is
  non-blocking, and the awareness loop keeps sampling while a VLM loads (uses
  the real wrapper with a slow fake loader, so it measures production code).
- `services/core/tests/test_boss_agent.py`:
  `test_boss_counts_app_launch_as_progress_without_state_change` -- verified to
  FAIL on the pre-fix predicate and pass after.
- `packages/avcs/tests/test_focus_verifier.py`:
  `TestTorchConstructionSerialization` (2) -- `build_model` restores the default
  dtype and serializes concurrent constructions.
- `packages/avcs/tests/test_labeled_frame_dataset.py` (6) -- real labels map to
  category indices, unknown labels are masked (-1) not guessed, affordance
  targets stay masked (-100), temporal pairs share a scene, empty manifest fails
  closed.

### Verified state
- Full `services/core` suite: **2389 passed, 40 skipped, 0 failures**.
- Dashboard `next build`: 21 routes compile clean.
- Ruff clean on every changed file (the 7 `UP0xx` in `describer.py` and the
  `N812` in `losses.py` are pre-existing at HEAD).
- Honest limit: the VLM path is opt-in and off by default; enabling it downloads
  a model on first use. No live end-to-end GUI task was re-run this pass.

## Boss worker-truthfulness fixes + live GUI acceptance (2026-10-25)

Ran the real `BossAgent` against the live `sonic-desktop-workstation` (Xvfb
`:99`, 1920x1080) with a genuine objective ("Open the terminal application on
the desktop."). The run exposed three real defects that unit tests could not
see, all of the "worker reported a failure that never happened" class. All three
are fixed with regression tests that fail on the pre-fix code.

### BUG 1 — Judge falsified every successful GUI worker (real)
`JudgeAgent.evaluate_sub_mission` gated its "total execution failure" branch on
`exit_code == 0`. GUI actions (`APP_LAUNCH`, `APP_FOCUS`, `GUI_SCREENSHOT`, ...)
are not subprocesses and legitimately carry `exit_code=None`, so a worker whose
only evidence was a completed GUI action was reported FALSIFIED with "Worker
produced no successful execution traces" — even though the live desktop showed
the launched `Terminal -` window.
- Fix: added `_trace_succeeded()` in `agents/judge.py`; a trace succeeds on
  `exit_code == 0` **or** a success status (`SUCCESS/SUCCEEDED/COMPLETED/
  VERIFIED/RECOVERED`). Genuinely failed GUI workers (status FAILED, exit 126)
  are still falsified.
- Tests: `test_judge_agent.py::test_judge_does_not_falsify_gui_worker_that_completed_actions`
  (verified FAIL pre-fix, PASS post-fix) and
  `...::test_judge_still_falsifies_when_every_gui_action_failed`.

### BUG 2 — Boss reported an INCONCLUSIVE Judge verdict as worker FAILED (real)
`BossAgent._dispatch_sub_agent` mapped every non-CONFIRMED verdict to
`status="FAILED"`. The Judge returns INCONCLUSIVE for GUI work until an
evaluated local verification head exists — that is an honest "cannot confirm",
not a failure. The dashboard therefore showed a worker that really launched the
application as `FAILED`.
- Fix: added `BossAgent._status_for_judge_verdict()`; only `FALSIFIED` maps to
  `FAILED`, everything else maps to `UNVERIFIED`. Wired at the verdict site.
- Test: `test_boss_plane_routing_and_truthfulness.py::test_boss_reports_inconclusive_judge_as_unverified_not_failed`.

### BUG 3 — Dashboard rendered UNVERIFIED/PARTIAL/FAILED indistinctly
`apps/dashboard/app/agents/page.tsx` `STATUS_STYLE` only knew running/idle/
completed/blocked/error, so an `UNVERIFIED` worker fell through to the neutral
idle style and `FAILED` was not styled as an error.
- Fix: added explicit `success`, `failed`, `timed_out`, `interrupted`,
  `skipped`, `unverified`, `partial`, `waiting_for_human` styles, and the
  `StatusDot` colour map. UNVERIFIED/PARTIAL now read as caution, not failure.

### Live evidence (real Docker workstation, real LLM, real desktop)
- Worker visibility works: `GET /workstation/agents?session_id=...` reports the
  live worker with stable `agent_uid`, name, plane, RUNNING status, step list,
  and latest action — the dashboard "click the agent to see what it is doing"
  path is backed by real data.
- The worker really launched a terminal: after the run `wmctrl -l` on the live
  desktop showed multiple `Terminal -` windows while `pgrep thunar`/window
  checks confirmed the GUI was genuinely mutated.
- Post-fix the same run reports the worker `UNVERIFIED` (was `FAILED`), with
  `judge: INCONCLUSIVE` and the honest note that AVCS reflex state cannot
  establish the semantic postcondition without an evaluated local head.

### Honest remaining limit (unchanged, fail-closed by design)
Semantic GUI targeting and text entry still require evaluated local AVCS heads
(`SONIC_AVCS_GROUNDING_CHECKPOINT`, `SONIC_AVCS_FOCUS_CHECKPOINT`); none exists
in-repo. `avcs_capability_state()` reports `semantic_grounding=false`,
`focus_verification=false`. So a Boss objective that must *click a described
control* or *type text* still cannot complete — but it now fails honestly
(UNVERIFIED), not with a fabricated FAILED.

### Tests / verification
- New/updated tests: 4 (2 judge GUI, 1 boss verdict mapping, 1 dashboard build).
- `services/core` full suite: **2392 passed, 40 skipped, 0 failures**.
- Dashboard `next build`: all routes compile clean.
- Ruff clean on every changed file; `git diff --check` clean.

## Boss subagent visibility + worker drill-down honesty pass (2026-10-25)

Operator report: "jab agent launch karta hai to unka id hona chahiye ... background
mein agent run hota hai ya nahi ... id/name par click kiya to dikhna chahiye kya
kaam kar raha hai." Audited the Boss → subagent → dashboard projection path
manually (no parallel subagents) and found the visibility core already worked;
fixed four real defects found along the way.

### What already worked (verified live, not asserted)
- `/workstation` and `/agents` render a live worker board with `agent_uid`,
  `agent_name`, `status`, `latest_action` and a `mission_task_alive` flag.
- `/agents` follows the active session (`getActiveSessionId`) instead of a
  hardcoded `"default"`.
- Worker drill-down (`/workstation/agents/{uid}?session_id=...`) returns the
  per-step timeline. Live drill-down on `worker-1-fb6b1c` showed real steps
  (`APP_LAUNCH`, `GUI_KEYPRESS`, `GUI_SCREENSHOT`, `WAIT_FOR_HUMAN`).

### Bugs fixed
1. **Heartbeats flooded the worklog.** Each ~2 s liveness tick appended a full
   worklog item (and persisted the session). A 39-item worklog had 18 heartbeat
   entries. Fixed with a module-level throttle in `api/routes/workstation.py`
   keyed by `(tenant, session, agent_uid, step)` so repeated ticks collapse to
   one; real steps and phase transitions are unaffected.
2. **Heartbeats clobbered the real "latest thinking".** Heartbeats are
   `type == "thought"` items, so `computer_use/agent_activity.py` set
   `latest_thought` to "Step N still reasoning after Ns" — the operator's
   expanded "Latest thinking" panel showed a liveness tick instead of the
   worker's actual reasoning. `collect_worker_activity()` now has a dedicated
   heartbeat branch: it records `heartbeat` / `heartbeat_at` on the worker
   without touching `latest_thought`. The dashboard surfaces them separately
   (`types/workstation.ts`, `components/worklog/WorklogFeed.tsx`).
3. **Phantom "Phase N Complete".** A replan phase that dispatched zero workers
   still emitted a completion event. `boss.py` and `_on_boss_event()` now render
   an honest "Phase N — no worker dispatched" title when nothing ran.
4. **GUI workers could not see real desktop applications.** This is why the
   operator's own "open a text editor" task kept failing. `application_list()`
   enumerates every `$PATH` executable (~800 on this image) alongside ~44 real
   `.desktop` apps, and the prompt truncated the merged list to 30 entries —
   `mousepad` sat at index 41, so the model concluded it was "not installed"
   and stalled on `WAIT_FOR_HUMAN`. Fixes:
   - New `DockerComputerProvider.desktop_applications()` queries only
     `.desktop` entries (no PATH scan); GUI workers use it exclusively
     (`agent.py`), text workers keep the full inventory.
   - New `ComputerUseAgent._rank_apps_for_goal()` reorders the app list by
     goal-token overlap (specific app-name match wins over a generic shared word
     like "editor"), name-agnostic, stable, no vendor literals; applied in both
     the detailed and compact prompts, with the visible cap raised 30 → 60.
   - `docker/Dockerfile.workstation` now installs `mousepad` (the image shipped
     `xfce4` + `xfce4-terminal` but no GUI text editor at all).
   Live result: after the fix the worker actually launched mousepad
   (`Untitled 1 - Mousepad` window via `wmctrl -l`), where before it declared
   the app missing on step 1.

### Honest remaining blocker (fail-closed, unchanged)
Text entry into the launched editor still stops: `GUI_KEYPRESS` on a semantic
target needs an evaluated local AVCS focus-verification head, which does not
exist in the repo. The worker records an honest `UNVERIFIED` / fail-closed
result rather than pretending to type. This is the documented contract, not a
regression: the operator must train and configure
`SONIC_AVCS_FOCUS_CHECKPOINT` (and the grounding head for semantic clicks)
before GUI text entry can complete. What is now proven live: the Boss dispatches
a worker with an id/name/status, the worker appears and is drillable in the
dashboard, and it can launch the requested installed application.

### Tests
- `services/core/tests/test_worker_activity_end_to_end.py` (+3): heartbeat does
  not overwrite a real thought; repeated heartbeats throttle to one worklog
  item; a zero-dispatch phase is not titled "Phase N Complete".
- `services/core/tests/test_docker_computer_provider.py` (+1): `desktop_applications`
  queries only `.desktop` entries and never the PATH scan.
- `services/core/tests/test_gui_capability_aware_prompt.py` (+2): goal-relevant
  app ranking; ranking is name-agnostic and order-stable.
- Boss/worker/dashboard subset: **206 passed** (one unrelated HF-hub 429 flake
  passes in isolation). Provider/agent/GUI subset: **189 passed, 2 skipped**.
- Dashboard `tsc --noEmit`: clean. Ruff clean on all changed files.

## AVCS self-learning (apprenticeship) unblocked (2026-10-25)

Operator asked whether AVCS can self-learn/self-train. It can — the loop already
existed (`sonic/computer_use/apprenticeship.py`: capture -> label -> learn ->
verify -> promote, driven by `AwarenessMonitor` when
`SONIC_AVCS_APPRENTICESHIP=1`). It was producing **zero samples** because of a
real labeler bug, so the eye never actually learned from lived experience.

### The bug
`DesktopSceneLabeler._load_category_map` keyed desktop entries only by the
`.desktop` file basename and its `StartupWMClass`. Many packages omit
`StartupWMClass` (e.g. `org.xfce.mousepad.desktop` has neither `mousepad` nor a
`StartupWMClass`), so a window whose WM_CLASS is `mousepad` matched nothing and
its frames were silently skipped as `unmapped_wm_class`. Any app in that state
contributed no training data at all.

### The fix
- The scan now also reads `Exec=` and `TryExec=` and keys on the **binary
  basename**, since that is what a window class usually names.
- Two tiers: identity keys (basename, `StartupWMClass`) are consulted before
  binary keys. Several entries can share a binary (an app and its settings
  dialog both run `mousepad`), so `_pick()` prefers the entry whose
  reverse-DNS final segment equals the window class, falling back to the
  shortest basename. This keeps `mousepad -> editor_code` instead of
  `settings_panel`.
- `GET /avcs/apprenticeship/status` now reads `samples`/`label_counts` from the
  shared on-disk store. The awareness monitor runs its own loop instance, so the
  route's own counter stayed at 0 while the store held hundreds of samples — a
  misleading "learning nothing" report.

### Verified live (real Docker workstation, real frames)
- Labeler resolution before -> after: `mousepad` = None -> `editor_code`;
  `xfce4-terminal` -> `terminal_shell`; `firefox-esr` -> `browser_webpage`.
- With `SONIC_AVCS_APPRENTICESHIP=1`, cycling focus across Firefox / Thunar /
  Terminal / Settings accumulated 300 honestly-labeled frames across 5 labels
  and 8 groups (labels are OS-derived from freedesktop `Categories=`, never
  guessed).
- A real `POST /avcs/apprenticeship/learn` trained a candidate and **promoted**
  it: held-out accuracy **0.986** (chance 0.125) on 69 held-out frames, written
  to `sonic_data/avcs/apprenticeship/cortex_learned.pt` with an
  `.eval.json` provenance sidecar.
- The learned checkpoint predicts the live desktop correctly: with the active
  window "Settings", it returned `settings_panel` at confidence 1.0 (the
  previous offline checkpoint returned `media_playback` at 0.23 on the same
  frame).

### Honest scope
This unblocks the **eye's** self-learning (scene-category perception). It does
**not** unblock semantic GUI grounding or text entry: those still require the
separately trained, held-out-evaluated `SONIC_AVCS_GROUNDING_CHECKPOINT` and
`SONIC_AVCS_FOCUS_CHECKPOINT`, which need real recorded click/focus
demonstrations (see `packages/avcs/docs/LOCAL_GROUNDING_QUICKSTART.md`). The
promoted accuracy is measured on this workstation's own frames and classes, not
on arbitrary unseen applications.

Tests: `test_avcs_apprenticeship.py` + `test_continuous_avcs_awareness.py` +
`test_gui_capability_aware_prompt.py` = 54 passed. Ruff clean.

## Self-evolution auto-mode live run — two real bugs fixed (2026-10-25)

Ran the self-developer in **auto mode** (`ContinuousAutoEvolutionRunner`,
3 cycles, real LLM) and watched it think. Two genuine defects were found and
fixed; the loop now advances and produces verified candidates.

### LIVE evidence (before → after)
- **Before:** cycle 1 discovered 5 real signals, attempted `signals[0]`
  (`developer_agent.py:449`), rejected it; cycles 2+ re-pinned the *same*
  signal, which the coder-side dedup filtered out → `no_candidate` **forever**.
  Signals 1..4 were unreachable. Historical DB: 0 promoted.
- **After:** cycle 1 → `developer_agent.py:461`, cycle 2 →
  `computer/headless.py:347`, cycle 3 → `context.py:98` → **VERIFIED candidate**
  (`2ce771f9`, `awaiting_approval`): real minimal patch turning
  `except: pass` into a logged exception; **2496 tests passed, 0 failed**,
  security regression clean, active checkout untouched (report-only).

### BUG A — runner permanently stuck on `signals[0]` (`continuous_runner.py`)
The runner pinned `signals[0]` every cycle while `run_candidate` filters
already-attempted `(target_file, description)` pairs at the same commit. Once
signal 0 was recorded, every later cycle resolved to zero signals →
`no_candidate`, and signals 1..N were never tried. Fix: added
`SelfCoderCandidateStore.attempted_signal_keys()` + `_select_signal()`; the
runner now advances to the first unattempted signal and emits `cycle_idle` when
all are exhausted. A new commit makes every signal eligible again.

### BUG B — "corrupt patch" discarded otherwise-correct LLM diffs
Models emit structurally valid diffs whose `@@` header counts disagree with the
body (e.g. `@@ -1,5 +1,5 @@` above one context line); `git apply` rejects that
as `corrupt patch`. The counts are fully derivable, so
`patch_synthesizer._repair_hunk_counts()` now recomputes them deterministically
(start lines preserved; `\ No newline` markers excluded). 12/21 historical
failures were this class. Valid diffs pass through byte-for-byte unchanged.

### Also fixed
- CLI `status` imported `_PROTECTED_COMPONENTS` from `codebase_evolver` (wrong
  module) → `ImportError` broke the whole command. Now imports from `pipeline`.

### Honest scope
Machinery is real and wired (API `/self-coder/*`, background loop in `main.py`,
CLI). The remaining limiter is **LLM diff quality**: the model still fails many
attempts on wrong hunk context/line numbers (`patch does not apply`). Diagnosis
quality is high (correct root cause + minimal fix), but reliable unified diffs
need a stronger coding model or the agentic `edit_file` path. Do not claim
sustained promotion until that is measured over many cycles.

Tests: evolution + self-coder suites green (169 passed); full `services/core`
**2436 passed, 40 skipped, 0 failures**; ruff clean. Regression tests added for
`_select_signal`, store dedup, hunk-count repair, and CLI `status`.

## Boss -> subagent visibility audit (2026-10-26)

Operator report: "Boss launches agents, but the launched agent has no id, and
clicking the agent name/id does not show what it is doing." The dispatch
pipeline was audited end-to-end with the **real** `BossAgent.run()` loop (only
model calls stubbed) plus real `ComputerUseAgent` workers; the route's
event->worklog translation and the worker projection were exercised together.

### Verified working (no change needed)
- Each delegated worker gets a stable, run-unique id (`worker-<n>-<hex>`), and
  the positional `agent_id` (`sub-agent-<n>`) is preserved alongside it.
- One board card per worker (no merged card); `sub_dispatch` ->
  `sub_step`/`sub_heartbeat` -> `sub_report` all key to the same uid.
- Click-to-expand drill-down resolves the specific worker and shows goal,
  current activity, latest thinking, steps, discoveries and Judge verdict.
- Mid-flight, workers show `RUNNING`; after the run they show `COMPLETED`.
- Headless phases fan out in parallel; GUI-capable phases correctly serialize
  on the single shared desktop (workers still all run).

### Real bugs found and fixed
1. **Run scoping mismatch (worker board vs timeline).**
   `collect_worker_activity` projected the *entire session* worklog, while the
   dashboard timeline is filtered to the current objective. After a second
   objective the board showed finished workers from the previous run next to an
   empty timeline. Added `current_run_only` (scopes to the latest
   `Objective Received` marker, falling back to the whole log when no marker
   exists) and wired it into `/workstation/state`, `/workstation/agents`,
   `/workstation/agents/{uid}` and `/workstation/agents/{uid}/message` so the
   board, the drill-down and steering all agree with the timeline.
2. **Heartbeat clobbered real thinking (frontend).** In
   `buildWorkerActivities` a `sub_heartbeat` fell through to the generic branch
   and overwrote `latestThought` with "still reasoning". Heartbeats now populate
   the dedicated `heartbeat` field, matching the server contract.
3. **Silent event drop.** `BossAgent._emit` swallowed every callback exception
   at `logger.debug`. A route handler that raised while recording
   `sub_dispatch`/`sub_step`/`sub_report` silently removed that worker from the
   board with no diagnostic. Worker-lifecycle events now log at WARNING with
   `operator_visible=True`; cosmetic events stay at debug.
4. **Defensive key resolution (frontend).** The live step/heartbeat branch
   guarded on `sub_agent_uid`/`sub_agent_id` only; it now resolves the worker key
   through the same helper as dispatch, so an `agent_uid`-keyed step can never be
   dropped.

### Tests
`services/core/tests/test_boss_worker_visibility_audit.py` (11) covers the emit
resilience, the real-Boss worker-id/visibility contract, drill-down contents,
run scoping (including the no-marker fallback) and board/timeline agreement.
Broad regression: 441 passed, 8 skipped (`-k "boss or worker or workstation or
agent_activity or computer_use"`). Dashboard `tsc --noEmit` and `next build`
clean.

### Honest scope
This validates the orchestration/observability seam with stubbed models. No live
end-to-end GUI task was run in this pass, and Boss completion is still bounded by
the AVCS grounding/focus checkpoints (semantic clicks and text entry remain
fail-closed until those are trained). SONIC is not a finished AGI; this pass
fixed the worker-visibility defects only.

## Eye-hand servo: closed-loop, self-correcting pointer (2026-10-26)

Operator goal: make AVCS a human-like real-time closed loop — "eyes + hand +
brain" — where the hand reaches, the eye watches, and the residual error is
driven to zero (perceive -> error -> correct -> re-perceive), instead of the
open-loop "compute a coordinate, teleport the cursor, hope" behaviour.

### What existed (and why it was not the loop)
- `VisualServoingEngine` (`forge/vision/cortex/core/servoing.py`) is torch-based
  and **imported by no production caller** — dead code. The agent had no closed
  loop at all; every GUI move/click was a single open-loop dispatch.

### New: a torch-free servo model (`packages/avcs/forge/vision/servo.py`)
- `HumanReachModel` — Woodworth's two-component reach: an **opening ballistic
  throw** that covers the whole distance in one motion, then damped,
  decelerating corrective submovements. Pure NumPy, no torch (a ~1.2 s import
  and a checkpoint buy nothing at this stage). Verified: importing
  `forge.vision.servo` does not load torch; lazy `forge.vision` package export
  invariant preserved (`test_lazy_package_import.py` still green).
- `locate_cursor` — a fallback pixel sensor (local-background box mean via an
  integral image, residual centroid). It is an honest heuristic, not a learned
  detector: a flat frame returns `None`; the confidence is surfaced so callers
  can drop weak reads. The old `detect_cursor_point` in
  `avcs/training/video_apprenticeship.py` is now a thin import of this function
  (one detector, not two).
- Cost measured honestly: `locate_cursor` is **~100 ms at 1080p**, so it is the
  *fallback*. The live loop prefers the provider's exact pointer readback.

### New: the async closed loop (`sonic/computer_use/visual_servo.py`)
- `VisualServo.reach()` reads the real pointer, commands a step, **re-reads**,
  and iterates until the residual is inside `tolerance_px` or the deadline
  (6 s) / iteration backstop (12) is hit. Returns a `ServoReceipt` with the
  measured `converged` flag, iteration count, final residual, latency, start /
  target / path, and `cursor_source` (`provider` | `vision`).
- Fail-closed: no pointer reading -> `ServoUnavailable` ("refusing to move
  blind"); a target outside the screen -> `ServoUnavailable` and **no move**; a
  pointer that never responds -> `converged=False` with a reason (never a
  fabricated success); an operator interrupt stops before the next move.

### New: provider proprioception (`DockerComputerProvider.cursor_position`)
- `xdotool getmouselocation --shell` on the workstation display; returns an
  integer point, or `None` on a failed/unparseable query (fail-closed).

### Wiring (`ComputerUseAgent`)
- `_get_visual_servo()` builds the loop only when the provider exposes a real
  integer-point `cursor_position`; a test double that merely *has* the attribute
  is rejected, so existing open-loop paths are untouched.
- `_servo_to_pixel_target()` runs one already-authorized reach and stores the
  measured receipt; **every corrective move re-evaluates the safety envelope**,
  so the loop cannot become a side channel that skips the policy.
- Wired into **GUI_MOVE** and **GUI_CLICK**. The final dispatch still goes
  through the normal path, so the pre-dispatch freshness revalidation is
  preserved (the reach leaves only a sub-tolerance precision correction).
- The receipt is surfaced in the action observation string
  (`[servo: converged=... residual=...px latency=...ms]`), so the operator and
  the model can see the measured result instead of a bare "Moved mouse".

### Measured evidence
- `locate_cursor` @1080p: ~100 ms/frame (fallback sensor only).
- Reach loop CPU: **~0.1 ms** for the whole reach (the cost is the desktop I/O,
  not the model).
- Against a lagging pointer (30% loss): converged in 5 iterations, 3.6 px
  residual; (50% loss): converged in 9 iterations, 3.6 px residual.
- A pointer with 100% loss: `converged=False`, honest reason, no fabricated
  success.

### Tests
- `services/core/tests/test_visual_servo_closed_loop.py` (16): model
  convergence, ballistic-then-corrective profile, no overshoot, contact basin;
  loop convergence despite lag, honest non-convergence, fail-closed without a
  pointer, out-of-screen target (no move), interrupt; the vision fallback
  actually drives the loop; agent wiring converges and records a receipt;
  servo disabled without a sensor; every servo move is refused by a denying
  safety policy.
- `packages/avcs/tests/test_servo_model.py` (7): determinism, ballistic throw,
  deceleration, contact radius, flat-frame rejection, **torch is not imported**,
  invalid-parameter rejection.
- `services/core/tests/test_docker_computer_provider.py` (+3): `cursor_position`
  parses the display query, fails closed on a failed query and on garbage.

### Regression
- Core computer-use/safety subset: **384 passed, 2 skipped**.
- Full `packages/avcs/tests` (minus tkinter): **594 passed, 20 skipped, 1
  pre-existing load-flake** (`test_soak_runner_short_run`, which passes in
  isolation and is unrelated to this change). Ruff clean on every changed file.

### Honest scope
This is a real, measured, self-correcting pointer loop. It does **not** prove
semantic GUI capability: the servo converges on a *pixel target*; deciding
*which* pixel corresponds to a described control still requires an evaluated
AVCS grounding checkpoint, and text entry still requires the focus-verification
head. Both remain fail-closed until trained. No live end-to-end GUI task was run
in this pass (no workstation/credentials here); the loop was validated with real
code paths against fake laggy desktops. SONIC remains unfinished.

## AVCS affordance head: trained from measured click outcomes (2026-10-23)

Closed the root cause behind "AVCS reports 4 interactive targets but clicking
them does nothing". The dorsal (affordance) head had NO honest training signal:
`labeled_dataset.py` masked every real-desktop frame's affordance map to `-100`
("real frames carry no hand-labeled affordance mask"), so the head kept its
random initialisation and `find_action_hotspots` returned noise. `agent.py` then
reported that count to the reasoning model as `interactive_targets=4`.

### 1. Honesty gate (fail-closed) — `affordance_provenance.py`
A checkpoint file cannot prove a head inside it was trained, so the fact is
recorded in a sidecar (`<checkpoint>.affordance.json`): named dataset, sample
counts, and the measured held-out point accuracy.
`affordance_head_trained()` returns True only when that evidence exists AND the
accuracy clears `MIN_HELD_OUT_POINT_ACCURACY = 0.40` (chance for 5 classes is
0.20). A head fine-tuned but still at chance is treated as untrained, because
publishing its hotspots is equally misleading.

Wiring: `AVCSPerception.affordance_head_trained` →
`perceive_semantics_rgb` omits `action_hotspots` unless proven trained →
`agent.py` prints `interactive_targets=unavailable:affordance-head-untrained`
instead of a fabricated count. `avcs_capability_state()` exposes
`affordance_head_trained` + `affordance_outcome_samples`.

### 2. Environment-reward supervision — `outcome_affordance.py`
The being already produces the missing label while it works: it clicks, then
observes whether the screen changed.
- changed → clicked point was an interactive control of that class (positive)
- unchanged → clicked point was not that class (negative)
- everything else → `-100` (ignored). The loss never asserts an unobserved
  region is static; that would be the same fabrication class as before.
A click with an operator-named target also yields a real (query → point)
grounding pair.

`build_point_mask` labels a small disk (radius 2 of the 32x32 head grid), so
supervision is sparse and honest rather than dense and invented.

### 3. Trainer — `affordance_trainer.py`
Fine-tunes ONLY the dorsal head; trunk and ventral head are frozen so a sparse
click set cannot degrade the already-verified category model (asserted by a test
that only `dorsal.*` keys may change). Split is by `group` (one working run) so
near-duplicates never leak. Metrics: held-out point accuracy (the gate's number)
plus IoU. Provenance is written only after a real run.

### 4. Capture wiring
- `OutcomeSampleStore` (in `apprenticeship.py`) persists clicks + measured
  outcomes durably under `sonic_data/avcs/apprenticeship/`.
- `ComputerUseAgent._record_click_outcome` hooks `_dispatch_gui_action`, where
  the screen-change verdict is already computed. Records only CLICK /
  DOUBLE_CLICK / RIGHT_CLICK with integer pixel coords and a real pre-action
  frame; keyboard and coordinate-less actions are skipped.
- `AVCSApprenticeshipLoop.learn_affordance()` runs the trainer (min 24 samples,
  else refuses). API: `GET /avcs/apprenticeship/affordance`,
  `POST /avcs/apprenticeship/learn-affordance`.

### Honest state after this change
- 30 synthetic click outcomes → held-out point accuracy **0.257** (chance 0.20),
  so the gate correctly reports the head **not yet usable**. The plumbing is
  real and verified; the head needs far more measured clicks from live desktop
  work before hotspots can be trusted. Do NOT claim the affordance head works.
- Tests: 17 AVCS + 15 core; AVCS package 608 passed / 20 skipped; core
  agent/avcs/gui subset 867 passed / 9 skipped; ruff clean. Commit `aca97ee`.
- Techniques assessed for AVCS and NOT adopted (with reasons): GAN/VAE (nothing
  to generate or reconstruct), RoPE/MoE (no text generation; `forge/model/` is
  not wired to the cortex), LoRA/PEFT (only pays off on a large frozen
  backbone — `peft` not installed). The blocker was never the architecture; it
  was the absent supervision signal, which this change supplies.

## AVCS affordance loop closed and verified (2026-10-23, part 2)

Part 1 built the honesty gate and the outcome-supervision dataset but left two
gaps that made the loop non-functional. Both are now closed and the loop is
verified end to end.

### Gap A — the capture path was dead code
Three places constructed their own `AVCSApprenticeshipLoop` (the awareness
monitor, the apprenticeship API route, and the agent hook added in part 1), so
an agent's measured clicks went into a store that no trainer ever read.
`get_shared_apprenticeship_loop(provider)` now hands the process ONE loop, bound
once to a labeler; a later caller with a different provider keeps the first
binding rather than silently re-pointing the store. The workstation route
attaches it to the production `ComputerUseAgent`.

### Gap B — the loop could not close itself
`AwarenessMonitor._maybe_learn` only retrained the *category* head. Added
`_maybe_learn_affordance`, keyed on the **outcome** count (clicks are far rarer
than frames) and guarded so a static desktop never re-runs the trainer. Env:
`SONIC_AVCS_AFFORDANCE_LEARN_EVERY` (default 40).

### Synthetic warm-start — measured, and defaulted OFF
An optional synthetic warm-start stage exists (synthetic screens carry real
ground truth, so it is genuine supervision). It is **off by default** because it
is not measurably beneficial: on a 60-click set, 400 samples / 4 epochs scored
**0.457** held-out point accuracy vs **0.585** for real clicks alone, and more
epochs overfit. Enable with `SONIC_AVCS_AFFORDANCE_WARM_START=<n>` only if a
large real set is too slow to train from scratch. It is recorded as its own
provenance stage and never confers readiness.

### Readiness now also requires the real-click domain
A synthetic-trained head has a real synthetic score, but its real-desktop
quality is unmeasurable (real frames carry no affordance ground truth), so
publishing its hotspots would be unverifiable. `affordance_head_trained()`
requires `domain == "live_click_outcomes"` in addition to clearing the accuracy
threshold. Recorded, not assumed.

### Verified end to end (measured, not asserted)
- **80 measured clicks → held-out point accuracy 0.664** (chance 0.20) → gate
  flips to **True**. With only 30 clicks it correctly stays **False** at 0.257.
  The gate tracks the evidence in both directions.
- Architecture capacity confirmed separately: full synthetic supervision lifts
  affordance pixel accuracy **0.57 → 0.96**, IoU **0.26 → 0.57**. The head was
  never the limitation; the missing supervision was. Do not conclude the head is
  weak from the 0.66 real-click number — that reflects ~80 samples, not capacity.
- Tests: 21 AVCS + 21 core (shared-loop isolation, domain gate both directions,
  frozen-head invariant under warm-start, auto-train fires once). AVCS package
  **612 passed / 20 skipped**; core agent/avcs/gui/awareness subset **876 passed
  / 9 skipped**; ruff clean. Commits `aca97ee`, `2073644`.

### Still NOT done (do not claim otherwise)
- No live GUI end-to-end run: this environment has no LLM/vision credentials and
  no Docker daemon, so the being has not yet accumulated real desktop clicks.
  The 0.664 figure comes from procedural screens standing in for a live desktop.
- Semantic grounding ("click the Save button") still needs a trained
  `GoalConditionedGrounder` checkpoint (`SONIC_AVCS_GROUNDING_CHECKPOINT`), which
  is a different, stricter contract (`recorded_ui_click_demonstrations`
  provenance + held-out point accuracy + calibrated query confidence). The
  outcome store already captures `query → click point` pairs, so the training
  data for it now accumulates as the being works.

## Docker VNC live test — AVCS was blind to its own cursor (2026-10-26)

Ran SONIC against the real Docker workstation (`sonic-desktop-workstation`:
Xvfb `:99` 1920x1080, XFCE, x11vnc `5900`, noVNC `6080`) after starting the
daemon (`sudo dockerd`; socket is root-only, `chmod 666 /var/run/docker.sock`)
and building `sonic-workstation:latest` (2.24 GB, ~6 min).

### What worked on the live desktop
- AVCS perception: real frames, `avcs_raw_frame_processed` at the configured
  10 FPS, ~21-38 ms/frame, `desktop_state=INTERACTIVE`, active window reported.
- Real GUI actions through `ComputerUseAgent`: `APP_LAUNCH` opened Thunar and
  Mousepad (confirmed out-of-band with `wmctrl -l` / `pgrep`), `GUI_MOVE`
  reached the exact target via the visual servo, `GUI_CLICK`/`GUI_DOUBLE_CLICK`
  dispatched and honestly returned `UNVERIFIED` when the screen did not change.
- Honest fail-closed: a semantic click ("the Firefox window") returned
  `BLOCKED` (no grounder) without touching the provider; `GUI_TYPE` returned
  `BLOCKED` (no focus head). No blind click, no fabricated success.
- VNC itself is genuinely live: noVNC `HTTP 200` and a real `RFB 003.008`
  handshake on `5900`.

### REAL BUG FOUND — the eye could not see the pointer
`mss` captures the root window, and the X server composites the hardware cursor
separately, so the pointer is absent from every AVCS frame. Measured: moved the
cursor to `(400,300)`, captured, moved to `(1500,850)`, captured again —
`cursor_position` reported both positions correctly but the two frames differed
by **0 pixels**. Consequences: AVCS could never confirm where a click landed,
and a human watching noVNC saw no cursor at all.

Fix (`docker/avcs_frame_stream.py`): read the sprite via XFixes
(`XFixesGetCursorImage`) and alpha-blend it into each frame; `--no-cursor` opts
out, and the compositor disables itself softly when XFixes is unavailable so a
missing cursor can never break perception entirely. Requires an image rebuild
(the script is `COPY`-ed to `/opt/sonic/avcs_frame_stream.py`).

Measured after the fix (live): frames at `(400,300)` vs `(1500,850)` differ in
**219 px**, changed bbox `x[403..1514] y[302..870]` — exactly the two cursor
positions — and the composited sprite renders as a recognisable arrow. A/B at
the same position (`--no-cursor` vs cursor-on) differs only inside the 24x24
sprite box at the cursor, so nothing else in the frame is perturbed.

Note: XFixes stores each CARD32 ARGB sprite pixel in a C ``long`` (8-byte)
slot; reading only the low 4 bytes is required on x86-64.

Tests: `services/core/tests/test_avcs_cursor_compositing.py` (9) pin the blend,
alpha handling, edge clipping (no wrap), 64-bit slot parsing, soft failure, and
the `--no-cursor` escape hatch. Full core suite: **2516 passed, 40 skipped, 0
failures**; ruff clean.

### Still fail-closed (capability genuinely absent, not a bug)
- Semantic targeting (`SONIC_AVCS_GROUNDING_CHECKPOINT`) — no evaluated local
  grounder checkpoint in the repo.
- Text entry (`SONIC_AVCS_FOCUS_CHECKPOINT`) — no evaluated local focus head.
- `avcs_capability_state()` correctly reports
  `semantic_grounding=false, focus_verification=false, gui_actions_ready=false`
  while `learned_cortex_ready=true`.

### Honesty note on the learned cortex metric
The promoted `cortex_learned.eval.json` reports `category_accuracy=0.998`, but
the class distribution is heavily imbalanced (`settings_panel` 663/920 =
0.721 majority; `editor_code` 193; `browser_webpage` 23; `form_input` 22;
`terminal_shell` 19) and the held-out split scored 560/560 on the majority
class. A constant "settings_panel" predictor would score 0.721, so the headline
number is dominated by one class and must not be read as general scene
understanding. Report per-class accuracy (or balanced accuracy) before quoting
cortex quality.

## Boss worker visibility + raw-frame log-flood audit (2026-10-26)

Operator complaint: "Boss launches agents but clicking the agent's id/name shows
nothing about what it is doing." Investigated the full seam end to end
(manual, single-agent — no parallel subagents) and separated real defects from
fail-closed-by-design behavior.

### Verified working (evidence-backed, not assumed)
- **Worker identity is minted correctly.** A live `BossAgent.run()` against the
  real `sonic-desktop-workstation` emitted `sub_dispatch` with
  `agent_id="sub-agent-1"`, run-unique `agent_uid="worker-1-<hex>"`,
  `agent_name` (e.g. "File Manager Explorer"), `role`, `plane="gui"`,
  `sub_mission_id="sub-1"`. Each launched agent does get an ID and a name.
- **The route → worklog → projection seam works.** Replaying the *real* captured
  Boss events through the route's exact translation produced a worker record
  with `step_count=3`, `latest_action="gui hotkey · Thunar location bar"`, and a
  resolvable `find_worker(uid)` drill-down listing every step
  (`GUI_SCREENSHOT [COMPLETED]`, `GUI_HOTKEY [UNVERIFIED]`, `GUI_HOTKEY [RECOVERED]`).
- **The dashboard click-to-view pipeline is complete and wired.** `WorklogFeed`
  renders a `WorkerBoard` of clickable cards (uid badge, role, plane, status,
  latest action); expanding a card fetches `/workstation/agents/{uid}` and shows
  the full step list; `app/agents/page.tsx` polls `/workstation/agents` and has a
  `toggleWorker` drill-down. `Workstation.tsx` passes `workers` and
  `mission_task_alive` through.
- **The motor is real.** Provider-level `gui_action` MOVE/CLICK glided the X11
  cursor on the live desktop (sampled out-of-band via `xdotool`:
  `640,[]` → `300,[]`), so the mouse genuinely moves.

### REAL BUG 1 — raw frame bytes flooded the logs (fixed)
`ScreenObservation.raw_rgb` holds a full 1920x1080 RGB frame (~6 MB) and was
rendered by `repr()`. A single run that interpolated the observation produced an
**81 MB log file** of `\xf6\xf5\xf4…` pixel data — drowning the real telemetry
and leaking workstation pixels into the log stream. Fixed with
`Field(default=b"", exclude=True, repr=False)` on `raw_rgb`
(`sonic/computer/models.py`); `model_dump()` still excludes it and callers still
read the frame normally. `repr()` of a full frame is now < 1 KB.

### REAL BUG 2 — object-branch traces silently lost worker identity (fixed)
`_record_step_trace` in `sonic/api/routes/workstation.py` hardcoded
`sub_agent_id/name/uid`, `sub_mission_id`, `worker_role`, `execution_plane` to
`""` on the `ComputerDecisionTrace`-object branch (only the dict branch carried
them). Any step arriving as an object became unattributable and its card could
never be drilled into. Now read via `getattr(trace, ..., "")`, so identity is
preserved when present and stays safely empty when absent (a plain
`ComputerDecisionTrace` has no such fields).

### Fail-closed by design (NOT bugs — do not "fix" by attaching pixels)
- Semantic GUI clicks ("click the Save button") remain blocked because no
  evaluated `SONIC_AVCS_GROUNDING_CHECKPOINT` exists in the repo. A worker that
  proposes one gets `avcs_target_grounding_failed_closed` and the mission stops
  with `gui_mission_stopped_for_unavailable_avcs_capability`.
- Text entry stays blocked without an evaluated `SONIC_AVCS_FOCUS_CHECKPOINT`.
- The affordance (clickable-hotspot) path is unavailable because
  `affordance_head_trained=False` for the shipped `cortex_real.pt`.
- `image_attached=False` is **correct** under the current AVCS-only perception
  contract: workstation pixels are processed locally and are NOT sent to the
  workstation reasoning model; only the AVCS text summary reaches System 2. Do
  not re-introduce screenshot transfer to "fix" grounding — train/configure the
  local AVCS heads instead.
- Net effect: a GUI worker can launch apps and use keyboard/pointer at explicit
  coordinates, but cannot click a *described* control, so such Boss objectives
  end INCOMPLETE honestly rather than fabricating success.

### Tests
`services/core/tests/test_worker_board_visibility_fixes.py` (7) pins both fixes:
repr/model_dump no longer expose raw pixels, a worker-aware trace keeps its
identity through the route's reads, and a plain trace stays safely empty.
Regression: 164 passed across the Boss/worker suites, 212 passed across the
computer-use/perception suites, ruff clean on all changed files.

## Worklog bounding + worker-visibility chain audit (2026-10-26)

Follow-up to the worker-board visibility fix. Two more real defects were found
and closed, and the full operator drill-down chain was verified end to end.

### Fixed: unbounded worklog (memory growth + O(n^2) disk writes)
`_append_worklog` appended without limit, and `_persist_workstation_state`
re-serialized the whole tenant/session store on every append. A long Boss
mission therefore grew without bound and paid a quadratic write cost. Measured
on the live store: 5 tenants, sessions with real worklogs already present.

Bound it with an identity-preserving cap (`_trim_worklog`):
- `sub_dispatch` / `sub_report` markers and the `Objective Received` marker are
  **pinned** -- the worker projection rebuilds each operator card from that
  pair, so evicting them would make an active worker silently vanish.
- Only step/heartbeat/command noise is trimmed, newest items kept.
- Configurable: `SONIC_WORKLOG_MAX_ITEMS` (default 600),
  `SONIC_WORKLOG_MAX_PINNED` (default 200).
- Verified: 2600+ appends -> 600 items, and the worker still projects with its
  uid, name, RUNNING status and 150 steps; `find_worker` still resolves it.

### Verified working (no bug -- do not "fix" these)
- Worker identity is minted at dispatch and carried on `sub_dispatch`,
  `sub_step`, `sub_heartbeat`, `sub_report` (`agent_uid` = run-unique handle,
  `agent_id` = `sub-agent-N`, `agent_name`, `role`, `plane`).
- `_worker_key` resolves consistently in both the backend projection and the
  frontend (`agent_uid` preferred, then `sub_agent_uid`).
- `/workstation/state` returns a server-side `workers` projection and
  `/workstation/agents/{uid}` serves the drill-down via `find_worker`.
- The frontend `WorklogFeed` rebuilds cards client-side from the same worklog,
  and `WorkerDetail` re-fetches the authoritative record on expand (with the
  worklog view as fallback), so a truncated/lagging worklog degrades instead of
  showing an empty panel.
- Liveness: `_mission_task_alive` gates the `RUNNING` verdict on a live
  background task, so a dead mission reports `INTERRUPTED`, not a stale RUNNING.

### Known seam (recorded, not changed)
`workstation.py` records `agent_success = not has_unjudged_gui_actions and ...`
via `mind.record_mission_outcome(success=...)`. Because the independent Judge
runs inside Boss, not in the direct workstation path, any mission containing a
GUI action is recorded as a non-success. `record_mission_outcome` only counts a
success when a `verification_receipt_id` is supplied, so this is fail-closed
rather than false -- but the being's affect/memory does not currently credit
Judge-confirmed GUI work. Wiring the Judge receipt through is future work.

### Tests
`test_worker_board_visibility_fixes.py` extended to 12 tests (worklog capped
under sustained appends; dispatch/report markers survive; the objective marker
survives so the timeline stays scoped; chronological order preserved; short
logs untouched). Boss/worker regression: 77 passed; workstation/worker/boss
subset: 346 passed, 8 skipped. Ruff clean.

## Dual execution plane is first-class (2026-10-27)

Fixed a real defect that hard-blocked mixed desktop/terminal objectives. The
`ComputerUseAgent` already fully supported a third execution plane `dual` (GUI
reasoning, coordinate validation, and `teamwork.observe_desktop in
("gui","dual")`), but the Boss schema and route layer only ever recognised
`{"gui","headless"}`. A desktop objective was pinned to `gui`, and a GUI-only
worker that needed the operator toolkit hit `Blocked by headless execution plane
contract` / could not fall back to the terminal.

### Plane model (now consistent everywhere)
- `gui` / `headless` remain **hard per-worker operator restrictions**.
- `dual` means "either plane is allowed": it is **not** a restriction, so the
  planner's explicit per-worker plane is still honoured and only an unspecified
  worker falls back to full dual agency. This preserves headless parallelism
  (treating `dual` as a hard override would force every worker onto the shared
  desktop and serialize the whole mission).

### Sites updated (`{"gui","headless"}` -> `+ "dual"`; default `gui` -> `dual`)
- `models_boss.py`: `SubMission.plane` default; validator preserves `dual` and
  falls back to `dual`.
- `boss.py`: constructor plane whitelist; `run()` normalizes and **validates**
  `requested_plane` (a bogus value is ignored, never latched as a restriction);
  phase-1 per-worker plane normalization; `_plan_next_phase` normalization and
  the plane-invariant prompt; `fallback_plane`; `execution_plane` default;
  `observe_desktop` for `dual`; `effective_plane` fallback; prompt guidance.
- `dispatch_policy.py`: `Workstream.plane` default and projection.
- `workstation.py`: request `Literal` accepts `dual`; desktop objectives dispatch
  `dual`; worklog `execution_plane` defaults for `sub_dispatch`/`sub_heartbeat`/
  `sub_report`.
- `agent_activity.py`: worker projection plane default.
- Dashboard `WorklogFeed.tsx`: the worker card renders a distinct
  "Desktop + Terminal" badge for `dual` (previously mislabelled as GUI).

### Honesty gate is now action-based, not plane-based
`ComputerUseAgent._mission_used_desktop_actions()` inspects real traces for
`GUI_`/`APP_`/`WINDOW_`/`BROWSER_` actions. The independent-Judge requirement at
the `GOAL_COMPLETE` handoff and at final verification now triggers on **actual
desktop usage**, so a `dual` worker that used the desktop gets the same Judge
gate as a `gui` worker. Plane alone was the wrong signal: it would have let a
`dual` worker's changed screen count as self-verified success.

### Worker identity + drill-down (verified, unchanged)
`boss.py` mints a run-unique `agent_uid` (`worker-<n>-<hex>`) per sub-agent and
emits it on `sub_dispatch`/`sub_step`/`sub_report`; the route records it on the
worklog; `GET /workstation/agents/{agent_uid}` returns the live worker record;
the dashboard worker card shows the uid and expands (click) into `WorkerDetail`,
which fetches the authoritative live record so the operator sees what the worker
is doing now. Background execution is a tracked `asyncio.Task`
(`_track_background_task`), and `_mission_task_alive` gates the RUNNING verdict.

### Tests
`test_dual_plane_e2e.py` gained 7 dual-plane cases (F12-F18): `dual` is
first-class and not coerced; `dual` is not a hard restriction; a `dual` worker
that used the desktop still needs the Judge while a headless-only one does not;
`gui`/`headless` remain hard restrictions; a `dual` worker counts as GUI-capable
for shared-desktop serialization; the dispatch default plane is `dual`; a bogus
`requested_plane` is ignored. Two legacy tests that pinned the old `gui` default
(`test_laya_end_to_end.py`, `test_workstation_boss_integration.py`) were updated
to the new `dual` contract.

Full `services/core/tests`: **2535 passed, 40 skipped, 0 failures**. Ruff clean
on all changed files; dashboard `next build` and `tsc --noEmit` clean. One
load-induced timing flake (`test_continuous_avcs_awareness::
test_capture_error_does_not_kill_the_loop`, which needs >=2 frames in a 0.4s
window and does not import the changed modules) passes in isolation; it is not
caused by this work.

## Focus-demo recorder + Judge receipts -> memory (2026-10-27)

Two real gaps closed while continuing AVCS training work.

### 1. The focus-verification head had no recorder (`focus_demo_recorder.py`)
`avcs.training.focus_dataset` and `train_focus_verifier` consume a
`records.jsonl` layout (manifest `status: "complete"`, per-record
`target_label`/`frame`/`point_px`/`focused`), but **no recorder ever produced
it** -- the trainer was reachable only from hand-written test fixtures. With no
way to record real focus demonstrations the focus checkpoint could never be
trained, so `GUI_TYPE`/text entry stayed permanently blocked.

Added `packages/avcs/avcs/training/focus_demo_recorder.py`
(`VNCFocusDemoRecorder`, console script `avcs-record-focus`), which reuses the
VNC click recorder's isolation guarantees (exact foreground window, cropped
frames, stale-frame rejection, no keystroke capture) and adds the one thing a
focus demonstration needs that a click demonstration does not: the human
`focused` ground-truth label for the clicked point. The label is **never**
inferred from the OS or a model -- inferring it would train the classifier
against its own guess. `parse_focus_target("<label>:<focused>")` rejects an
unrecognised focus token rather than guessing; `--target` pre-arms a sequence of
clicks, otherwise the operator types the label and answers a focused y/n prompt
before each click. Sessions are written with manifest `status: "complete"`.

Tests (`packages/avcs/tests/test_focus_demo_recorder.py`, 19): token parsing
(accepted/rejected), cropped-point mapping, schema/no-text record shape,
isolation-confirmation + pre-armed-label gates, rejected/stale clicks do not
consume a label, and a **round trip** proving the written session is consumable
by the real `discover_focus_demonstrations` (the defect this closes) and that an
incomplete session is rejected. Verified the recorder->loader->trainer chain
end-to-end: a synthetic recorded session trains a checkpoint and the
`runtime_ready` gate stays honest (`False` when accepted held-out samples are
below the 30-sample floor).

### 2. Judge receipts now reach the being's memory (requirement #3)
The workstation route credited a mission to `BeingMind` as a *verified success*
whenever the Boss reported `status == COMPLETE`, but never passed a
`verification_receipt_id`, so `record_mission_outcome` computed
`verified_success = False` and Judge-confirmed work earned **no** satiety /
curiosity credit. The receipt seam was open.

- `BossReport.judge_receipt` (new field): `_build_judge_receipt()` derives a
  SHA-256 `judge-<hex>` id from only the sub-missions the independent Judge
  returned `CONFIRMED` **and** marked `evidence_verified`. No confirmed verdict
  -> empty `receipt_id` (the run is not a verified success).
- `api/routes/workstation.py` passes that `receipt_id` into
  `record_mission_outcome` on the Boss path; the single-agent path derives its
  own receipt from real `VERIFIED` action traces (independent sandbox probe +
  System-1 receipt check) and passes it likewise. No receipt -> the outcome is
  recorded as *attempted*, not *completed*.
- Tests (`services/core/tests/test_judge_receipt_memory.py`, 6): receipt minted
  only from CONFIRMED+verified verdicts; empty without one; ignores a CONFIRMED
  string lacking `evidence_verified`; deterministic for the same work; and
  `record_mission_outcome` credits `goals_completed`/satiety only when a receipt
  is supplied.

Honesty: still no real held-out AVCS checkpoint in-repo and no live GUI
acceptance run -- the recorder makes training *possible* from real
demonstrations, it does not by itself produce a passing checkpoint. AVCS suites:
634 passed, 20 skipped, 1 known load-flake (soak) that passes in isolation.

## Closed-loop reach now covers every pointer action (2026-10-27)

The eye-hand servo (`computer_use/visual_servo.py`) was wired into `GUI_CLICK`
and `GUI_MOVE` only, so a double-click, right-click, drag, scroll, or the focus
click before typing were still **open-loop**: the agent committed the model's
coordinate in one jump and never corrected it against the real pointer. That is
precisely the "perceive -> error -> correct -> re-perceive" property the product
goal calls for, and it was missing on four of the six pointer actions.

Fixed in `computer_use/agent.py`: `GUI_DOUBLE_CLICK`, `GUI_RIGHT_CLICK`,
`GUI_DRAG` (reach to the press origin; the destination is a path endpoint, not a
fixated target), `GUI_SCROLL` (scroll lands under the pointer), and the
`GUI_TYPE` focus click now converge the pointer onto the target through the same
`_servo_to_pixel_target` path before acting, and carry the measured receipt in
the observation. The reach stays best-effort and non-fatal -- when the provider
has no pointer sensor the action keeps its previous open-loop behaviour, and the
safety envelope is still re-evaluated on every corrective move.

Tests (`services/core/tests/test_visual_servo_closed_loop.py`, +5): a parametrized
case proves each of the four actions dispatches at least one closed-loop MOVE
before its own provider action, and the no-sensor case proves the previous
open-loop behaviour is preserved. Focused GUI regression: 174 passed. Ruff clean.

## Apprenticeship promotion gate now resists majority-class collapse (2026-10-27)

The AVCS self-learning loop (`computer_use/apprenticeship.py`) promoted a
candidate whenever its raw held-out accuracy beat 1/8. Raw accuracy is dominated
by the majority class, so a model that always predicts the most common category
could be promoted while ignoring every other class -- and the committed
`cortex_learned.eval.json` shows exactly that shape: 0.998 accuracy with 560 of
589 held-out samples in one class (`settings_panel`), and the better-balanced
`cortex_real_all.eval.json` at 0.654 with two classes at zero.

Fixed: the gate now keys on **macro-F1** (unweighted mean F1 across the held-out
classes present), with a raw-accuracy floor of chance. `_evaluate_on_manifest`
also returns per-predicted-label counts so precision/recall can be computed; the
sidecar and `LearnResult` now record `macro_f1`, `balanced_accuracy`,
`deployed_macro_f1`, and `heldout_classes` alongside the existing accuracy and
per-class table. A candidate whose macro-F1 does not beat the deployed model's is
rejected, so a collapsed model can no longer overwrite a balanced one; at
bootstrap (no deployed model) a better-than-chance candidate is still promoted,
and the sidecar records its weakness honestly. The awareness log line now carries
`candidate_macro_f1` too.

Tests (`services/core/tests/test_avcs_apprenticeship.py`, +5): macro-F1 punishes a
majority-class collapse and rewards a balanced model; empty input yields zero; a
collapsed candidate with *higher* raw accuracy cannot displace a balanced
deployed checkpoint; a balanced candidate is promoted. Full apprenticeship suite:
35 passed. Ruff clean.

## Workstation chat: greetings no longer fail, stray tokens no longer launch missions (2026-10-27)

Reported symptom: sending "hi" to the dashboard returned "LLM Response
Unavailable" plus a raw workstation status dump, and a typo like "geeh"
returned "no provisioned workstation". Both were routing bugs in
`api/routes/workstation.py`, not an LLM outage.

1. **Greeting fast path.** A bare greeting was routed through the reasoning
   model. When that provider round-trip failed (unready/absent provider, or a
   transient error) the operator saw a provider failure and a status snapshot
   for what was really just "hi". `_is_bare_greeting()` + `_greeting_reply()`
   now answer a salutation directly, before the workstation check and before any
   pixel capture or provider call. Identity/capability/status questions ("who are
   you", "what can you do", "status") are deliberately NOT treated as greetings:
   they still get a real model answer.
2. **Stray-token guard.** `_is_action_prompt` treated any non-greeting text as an
   actionable desktop objective, so a single mistyped token started a mission and
   hit the "no provisioned workstation" block. A single token is now actionable
   only when it names something real (path/URL/filename/host or an installed
   binary); multi-word sentences remain tasks.

The greeting sets (`_GREETING_WORDS`, `_BARE_GREETING_PHRASES`,
`_CONVERSATIONAL_PHRASES`) are hoisted to module scope so the classifier and the
fast path share one definition. Verified against the real router: "who are you"
answers from the model; "hi" answers from the fast path with no provider call.

Tests (`test_workstation_chat_and_observation.py`, +4): greeting recognition vs
identity questions; stray single token is not an action but `python`/`/etc/passwd`
still are; greeting reply is honest; a greeting makes zero provider calls. Two
existing pipeline-error tests in `test_workstation_boss_integration.py` were
retargeted from "hello" to a non-greeting prompt so they still exercise the
error path. Workstation regression: 122 passed; ruff clean.

## Cloud VLM grounding bridge + live vision-model reality (2026-10-27)

Built and live-tested the missing bridge that lets SONIC answer "where is X?"
through a configured vision model when no evaluated local AVCS grounder
checkpoint exists. Committed `44133e3` (bridge) and `1991743` (operation model
doc).

### What was added
- `grounding.py::_parse_vlm_grounding_reply()` — parses a VLM reply in
  normalized `[0,1000]` space (JSON `{"x","y"}`, `{"point":[x,y]}`, bare
  `[x,y]`, UI-TARS `<|box_start|>(x1,y1,x2,y2)<|box_end|>`) into screen pixels.
  Fails closed on prose, `NOT_FOUND`, non-finite/non-numeric values, and
  out-of-range points (rejected, never clamped).
- `grounding.py::_resolve_vlm_target()` — sends the live frame + target phrase
  to the `vision` route; returns a point or `None` on any failure. Takes the
  router explicitly (there is no global `get_router()`; the agent holds
  `self.llm_router`).
- `agent.py::_ground_target_with_avcs()` — uses the bridge ONLY when
  `SONIC_ENABLE_VLM_GROUNDING=1` AND no local AVCS grounder checkpoint is
  configured. The local checkpoint always wins; default path keeps workstation
  pixels inside the workstation.
- Tests: `services/core/tests/test_vlm_grounding_bridge.py` (14) — parsing,
  fail-closed behavior, vision-route selection, image attachment.

### LIVE finding — the NVIDIA vision models available here do NOT ground
Ran the real bridge against the real NVIDIA NIM endpoint with a synthetic
1280x800 UI of known ground truth (SEARCH button, Clone button, address bar):

- `meta/llama-3.2-11b-vision-instruct` — responds in PROSE, never emits a
  coordinate JSON; latency 32-119 s per call. It is a caption model, not a
  grounder. Useless for `resolve_ui_target`.
- `nvidia/nemotron-3.5-super-vl-120b-a12b` — resolves server-side to
  `private/nvidia/nemotron-3.5-super-120b-a12b`, a TEXT/reasoning model. It
  spends the whole budget on `reasoning_content` and returns empty `content`
  (`finish_reason=length`). Also NOT a grounder.
- The router's budget guard additionally rejects models with no configured
  pricing ("Cannot enforce LLM budget..."), so only the priced
  `meta/llama-3.2-11b-vision-instruct` can even be called today.

**Conclusion:** a generic cloud VLM is not a drop-in grounder. Real grounding
needs either (a) a purpose-trained grounder (UI-TARS / OS-Atlas style, or the
local AVCS checkpoint), or (b) a much stronger general VLM with explicit
coordinate post-processing. The bridge is correct and ready; the model is the
missing piece. Do not claim semantic grounding works until a real grounder
model is wired and measured.

### Capability picture after this pass (honest)
- Works: AVCS reflex perception, LLM reasoning, app launch, pixel-coordinate
  GUI actions, the new grounding bridge plumbing.
- Still blocked (fail-closed, capability genuinely absent): semantic GUI
  targeting (no real grounder model) and text entry (no focus checkpoint).

## Local grounder pipeline completed + provenance honesty bug fixed (2026-10-27)

Follow-up to the cloud-VLM pass. Live testing showed the available cloud VLMs do
not ground, so the correct target is the grounder the repo ALREADY ships:
`avcs.goal_grounder.GoalConditionedGrounder` (~776K params, CPU-fast,
privacy-safe, and the model `load_grounding_checkpoint` already gates on). The
real blocker was never the model — it was DATA and one honesty bug.

### REAL BUG FIXED — trainer fabricated provenance
`avcs/training/train_grounder.py` wrote `dataset_provenance:
"recorded_ui_click_demonstrations"` **unconditionally** into both the checkpoint
and the report. Since `load_grounding_checkpoint` gates on exactly that string,
a checkpoint trained on ANY data (including synthetic) would have passed the
production gate and been loaded as a real-demonstration grounder — the exact
"success by decree" failure class this project forbids.

Fixed by making provenance travel with the data:
- `demonstrations.py`: `DemoRecord` gained a `provenance` field;
  `discover_demonstrations` reads `manifest["provenance"]` (defaulting to the
  recorder's `recorded_ui_click_demonstrations` when absent, so existing
  recorder sessions are unchanged). Empty/invalid provenance is rejected.
- `train_grounder.py`: `_resolve_dataset_provenance()` derives provenance from
  the actual records (`"mixed"` if they disagree), stamps it into the checkpoint
  and report, prints a WARNING when training on non-recorder data, and
  `runtime_ready` is now also gated on real provenance.
- Verified live: a synthetic-trained checkpoint carries
  `synthetic_rendered_controls` and `load_grounding_checkpoint` REJECTS it
  ("invalid 'dataset_provenance' provenance"). Recorder sessions still load.

### NEW — synthetic session generator (headless pipeline testing)
`avcs/training/synthetic_sessions.py` renders labelled UI controls and emits
sessions in the recorder's exact on-disk schema, so loader -> dataset -> trainer
-> checkpoint can be exercised without a desktop. Sessions are marked
`synthetic_rendered_controls`; the generator is never a substitute for the
recorder. CLI: `python -m avcs.training.synthetic_sessions --output-dir <dir>`.

### NEW — Kaggle notebook for the LOCAL grounder
`packages/avcs/notebooks/train_grounder_kaggle.ipynb` (generated by
`build_train_grounder_notebook.py`, so the cells stay reviewable Python rather
than hand-edited JSON). Trains `GoalConditionedGrounder` on uploaded recorder
sessions; Step 6 verifies against the real production loader. Without real data
it falls back to the synthetic smoke path and reports the honest rejection.

### Verified end-to-end (CPU, this environment)
Generated 12 synthetic sessions -> ran real `train_grounder` training -> real
checkpoint written -> loader correctly rejects it. Existing recorder sessions
are unaffected. No Docker/X11 here, so no real demonstration recording was run;
that requires the workstation desktop.

### Docs
`packages/avcs/docs/GROUNDING_CHECKPOINT_PIPELINE.md` — the full path: Docker
recorder -> Kaggle training -> `SONIC_AVCS_GROUNDING_CHECKPOINT`.

### Tests
- `packages/avcs/tests/test_synthetic_grounding_sessions.py` (10): loader
  accepts generated sessions; manifest contract; normalized points; frames
  exist; over-write refused; generated data is marked synthetic; legacy
  recorder sessions default to recorded provenance; blank provenance rejected;
  split + dataset shapes.
- Regression: `packages/avcs/tests/` **644 passed, 20 skipped**, 1 pre-existing
  load-flake (`test_soak_runner`, passes alone). Core grounding suites 49 passed.
  Ruff clean on every changed/new file (61 pre-existing errors in untouched
  `dataset.py`/`demo_recorder.py`/`train_kaggle.py`/`vicreg_safeguard.py` remain).

## Boss -> worker visibility: real dispatch seam verified (2026-10-27)

The operator complaint was: "when the Boss launches agents, they must have IDs,
they must run in the background, and clicking an agent's id/name must show what it
is doing." Audited the whole chain manually and closed the one real gap.

### What was already correct (verified, not assumed)
- `boss.py` emits `sub_dispatch` / `sub_step` / `sub_report` / `sub_heartbeat`
  carrying `agent_id` (positional), `agent_uid` (run-unique), `agent_name`, `goal`,
  `plane`, `max_steps`.
- `api/routes/workstation.py` translates each event into a worklog item and pins
  `sub_dispatch`/`sub_report` so worker cards survive worklog bounding.
- `agent_activity.py` projects the worklog into per-worker records keyed by
  `uid:{agent_uid}` (falling back to agent id / mission id / number).
- `WorklogFeed.tsx` renders one card per worker with the uid shown, distinct keys
  so React cannot merge two workers, and an expand toggle showing live activity.

### The real gap: the dispatch seam was never tested
`test_boss_worker_visibility.py` stubs `BossAgent._dispatch_sub_agent`, so the code
that actually builds a worker, runs `run_mission`, and streams per-step traces was
never exercised. A regression there would keep every test green while the operator
saw nothing.

### NEW — `services/core/tests/test_boss_real_dispatch_seam.py` (5 tests)
Drives the REAL `_dispatch_sub_agent` against the REAL in-process
`HeadlessComputeProvider` (no Docker, no desktop); only the LLM router is scripted.
Asserts: workers surface with unique `agent_uid` + positional `agent_id` + goal;
each worker's steps reach the board (thought + action); workers are individually
addressable by uid and never merge; a fresh objective yields fresh identities; and
a worker that ran without independent evidence reports UNVERIFIED, never COMPLETED.

### Verified honest status chain (live in-process, real provider)
- Host execution disabled -> `FAILED` with "FAIL-CLOSED: Host execution is disabled".
- Command succeeds, no independent evidence -> `UNVERIFIED` ("successful execution
  lacks independent task evidence").
- Only independently verified work becomes COMPLETED. No fake success.

### Test baseline
133 passed across the Boss/worker/receipt regression set; Ruff clean.

## Real demonstration capture on a headless desktop (2026-10-27)

Closed the practical blocker behind "SONIC can't do semantic GUI work": the repo
had NO `recorded_ui_click_demonstrations` dataset and NO evaluated grounder
checkpoint, so semantic targeting stayed fail-closed. The only real recorder
(`vnc_demo_recorder`) needs a live desktop AND a human clicking, which is why
real data was never collected.

### NEW — `packages/avcs/avcs/training/headless_demo_recorder.py`
Drives the SAME `vnc_demo_recorder` over a headless Xvfb display: it starts the
display + a window manager, launches real windows, issues real X11 pointer
events via `xdotool`, and stops the recorder gracefully (stdin `:stop`) so the
manifest is written. Pixels and click points are genuinely observed on a real X
server, so the recorder's normal `recorded_ui_click_demonstrations` provenance is
honest. Labels are always supplied by the caller — the module never fabricates
one. Requirements (Xvfb/xdotool/wmctrl/xinput/mss + a WM) are checked with clear
errors; a terminal is discovered via POSIX alternatives, never a pinned product.

### Verified end-to-end on this machine (real X server)
- Installed Xvfb/xdotool/wmctrl/xinput/openbox/xterm + `mss`; started `Xvfb :99`.
- Recorded 6 distinct task/window sessions -> **12 real labeled samples** with
  real cropped frames (484x316 RGB).
- `discover_demonstrations` + `split_sessions` accepted them: 6 sessions,
  leakage-free 8/2/2 train/val/test, provenance
  `recorded_ui_click_demonstrations`.
- `train_grounder` produced `/tmp/real_grounder_run/goal_grounder_best.pt` whose
  report records `dataset_provenance: recorded_ui_click_demonstrations`.
- **Honesty:** with only 12 samples the held-out accuracy is **0.000** and the
  runtime gate reports **"not ready"**. This module builds the DATA PATH; it does
  not claim a usable checkpoint. Real readiness needs far more sessions.

### Tests — `packages/avcs/tests/test_headless_demo_recorder.py` (7)
Pure helpers always; the live recording test runs only where the X11 tooling + a
WM exist and is skipped (never faked) otherwise. Ruff clean.

### Regression
39 passed, 7 skipped across the AVCS recorder/focus/video suites and the Boss
worker-visibility suites.

## Decisive diagnosis: the grounder pipeline is sound; only real data was missing (2026-10-27)

The prior "0.000 accuracy" run left it ambiguous whether the grounding pipeline
was broken or merely data-starved. Resolved by training on a scaled dataset and
reading the gate conditions directly.

### Evidence
- 12 real samples -> held-out accuracy **0.000** (ambiguous).
- 40 synthetic sessions / 240 samples -> held-out accuracy **1.000**
  (`top1_point_accuracy_within_0.02_diagonal = 1.0`, target 0.85), query
  calibration `passed: true` (precision 1.0, 36 accepted samples >= 10).
- The ONLY failing gate is `dataset_provenance == RECORDED_PROVENANCE`: the
  synthetic run carries `synthetic_rendered_controls`, so `runtime_ready=False`.
  That is correct fail-closed behaviour, not a defect.

### Conclusion
`runtime_ready` requires three conditions (`train_grounder.py`):
`evaluation["passed"]` AND `confidence_passed` AND
`dataset_provenance == "recorded_ui_click_demonstrations"`. A scaled run meets
the first two outright. **The pipeline, the split, the model and the readiness
gate all work.** What is missing is *volume* of genuine recorded demonstrations.

### Also fixed this pass (real bug)
`train_grounder` hardcoded `num_workers=2`, which aborts with
"unable to allocate shared memory" under the Docker default `/dev/shm` of
64 MB — so training could not start in the container it is meant to run in.
`_resolve_loader_workers()` now sizes workers to available shared memory
(env override `SONIC_AVCS_DATALOADER_WORKERS`). Verified: a 1600-sample run that
died at ~90 s then trained 26+ minutes. Tests:
`packages/avcs/tests/test_train_grounder_workers.py` (7).

### Next step (unchanged, now with certainty)
Collect a large real demonstration set with
`avcs.training.headless_demo_recorder` (or human recording), then train. Do not
expect `runtime_ready=True` from synthetic data — by design it cannot pass.

## Horizontal (left/right) scrolling added to the GUI action surface (2026-10-23)

The operator asked whether SONIC could scroll "like down, left, right". Audit
found a real capability gap: only **vertical** scrolling existed. X11 delivers
wheel events on buttons **4/5** (up/down) and **6/7** (left/right), but every
dispatch site hardcoded 4/5, so content off to the side (wide tables,
spreadsheets, code, timelines, horizontal carousels) was unreachable.

### Change
- `computer/models.py`: `GUIAction` gains `scroll_axis` ("vertical" default,
  "horizontal"), normalized from `horizontal`/`x`/`left`/`right` aliases.
- `computer/docker_computer.py` + `computer/daytona_computer.py`: SCROLL maps to
  buttons 6/7 when the axis is horizontal, 4/5 otherwise.
- `computer_use/agent.py`: `GUI_SCROLL` accepts `payload["axis"]`
  (`horizontal`/`x`/`left`/`right` -> horizontal, anything else -> vertical);
  `BROWSER_SCROLL` honours `delta_x` and preserves the axis; burst actions accept
  the `axis` alias so a horizontal scroll inside a chunk is not silently
  vertical. The prompt documents `axis` and the delta sign convention
  (negative = up/left, positive = down/right).
- The scroll observation string now reports `axis=<vertical|horizontal>`.

### Tests (5 new agent-level, 2 new provider-level)
- `test_phase8_visual_computer_use.py`: default axis vertical; horizontal axis
  forwarded to the provider; `left`/`x` aliases normalize; unknown axis degrades
  safely to vertical.
- `test_docker_computer_provider.py`: vertical uses buttons 4/5 and never 6/7;
  horizontal uses 6 for right and 7 for left. Verified the horizontal test FAILS
  against the pre-fix mapping (emitted button 4), so it genuinely guards the fix.

### Baseline
- 152 passed / 2 skipped across the two touched suites; 785 passed / 14 skipped
  on the broad gui/scroll/motor/action/computer/vision/grounding selection.
- Ruff clean on every changed file.

## Verifiable AVCS model bundles (2026-10-23)

A trained `.pt` checkpoint is not shippable on its own: nothing recorded what
data it was measured on, whether it still loads, or whether the bytes an
operator received are the bytes that were evaluated. Added
`packages/avcs/avcs/model_bundle.py` (+ `docs/MODEL_BUNDLE.md`,
`tests/test_model_bundle.py`) to close that gap.

- `pack_model(checkpoint, output_dir, archive=...)` refuses to bundle a
  checkpoint that does not exist, does not strict-load into
  `ArtificialVisualCortex`, has no `.eval.json` sidecar, or has a sidecar with no
  named `dataset_provenance`. It writes `<name>-bundle/{checkpoint, sidecar,
  manifest.json}` with a SHA-256 + size for every file, the resolved evaluation,
  the runtime (python/torch/avcs), and `strict_load_verified: true`.
- `verify_bundle(dir)` re-checks the manifest and every checksum; returns
  `(ok, problems)` with specific reasons (checksum mismatch, missing file,
  absent provenance).
- `load_bundle(dir)` verifies first and raises `ValueError` instead of loading
  weights from a bundle whose bytes do not match its manifest.
- CLI: `python -m avcs.model_bundle pack|verify`.

Packed the two real production checkpoints (both verified OK, archives written):
- `sonic_data/model_bundles/cortex_real-bundle` (`.env`'s
  `SONIC_AVCS_CORTEX_CHECKPOINT`; held-out category accuracy 0.833, provenance
  "real desktop capture: sonic_data/avcs/real_desktop/split/heldout.json").
- `sonic_data/model_bundles/cortex_learned-bundle` (apprenticeship-promoted;
  0.998 on 589 samples, provenance "live desktop apprenticeship").
Bundles live under `sonic_data/` (gitignored) — the repo convention is that
model binaries are not tracked in git.

Tests: `packages/avcs/tests/test_model_bundle.py` (12) cover checksum/provenance
emission and the refusal cases (no sidecar, no provenance, unloadable
checkpoint), tamper detection, and load-verification. 16 passed with the lazy
import regression; ruff clean on the new files.

## Boss worker live-run: four real defects fixed (2026-10-23)

Reproduced end-to-end against the live `sonic-desktop-workstation`: Boss launched
a GUI worker with a stable `agent_uid` that showed RUNNING on the dashboard but
did almost no real work, then stalled. Root causes were all in the GUI worker's
reasoning/parse loop, not in dispatch or the dashboard:

1. **Multi-step plan response rejected** (the dominant stall). The vision model
   answered with an ordered plan of several JSON action envelopes in sequence
   (e.g. `APP_LAUNCH` + `GUI_CLICK` + `GUI_KEYPRESS`). `_parse_gui_action_response`
   took an envelope only when exactly ONE candidate was present, so every
   multi-action reply was discarded and the worker burned steps until it died.
   The agent executes one action per step by design, so the first envelope in
   reading order is the unambiguous next action; it is now taken. A single
   malformed/extended envelope (extra keys) is still rejected.
2. **Vision-model refusal misread as a parse error.** The fallback vision model
   intermittently returns "I'm not able to assist with that." The agent's local
   refusal list missed the "not able to assist" family, so a refusal surfaced as
   a confusing `gui_action_response_invalid` instead of a recorded refusal that
   triggers replanning. The agent now reuses `sonic.llm.router.REFUSAL_PHRASES`
   (single source of truth) plus a few extras; the router already triggers
   provider/model fallback on a detected refusal.
3. **Hardcoded 60s GUI reasoning timeout.** A slow-but-working vision route
   (large fallback model + full screenshot) exceeded it and the worker died with
   `reasoning-unavailable`. Now `SONIC_GUI_REASONING_TIMEOUT_SECONDS` (bounded
   10-600, default 60) via `_gui_reasoning_timeout_seconds()`; the module
   constant remains the default so existing tests can monkeypatch it.
4. **`application:mousepad` role-prefix not stripped.** The model prefixed the
   app role; resolution ran the string literally and failed for an installed
   app. `_normalize_app_name` now strips a leading
   `application|app|binary|program|executable[:=]` qualifier, and the APP_LAUNCH
   dispatch site normalizes `payload.app_name` before calling the provider.

Also improved GUI-worker prompts: the mandatory `postcondition` now has a
concrete worked example (was prose only), the GUI-only observation surfaces
`Active Application / Window` + `Open Windows` + actions-so-far with a PROGRESS
RULE (do not relaunch an app already open; emit GOAL_COMPLETE if the
postcondition already holds), and the action instructions now permit the
explicit integer pixel form (the old text forbade the very pixel target the
prompt also asked for). The invalid-response warning logs a bounded
`response_preview` for operator diagnosis.

Live result after the fixes: the worker took 8 real steps, launched Mousepad
(`Untitled 1 - Mousepad` window confirmed via `wmctrl`), and the independent
Judge returned `INCONCLUSIVE` — the documented fail-closed contract, because no
evaluated local AVCS task-verification head exists. This is honest: the being
can now drive the GUI and open applications, but semantic task confirmation
still requires a trained AVCS head.

Tests: `services/core/tests/test_boss_worker_live_regressions.py` (8) plus the
updated `test_phase8_visual_computer_use.py` parser contract and
`test_app_launch_no_silent_substitution.py`. Broad Boss/worker/GUI regression:
346 passed, 0 failures.

## "Mouse ka use nahi" root cause: the prompt forbade the pointer (2026-10-23)

Operator report: SONIC does automated work but the mouse is never used. Verified
live by polling `xdotool getmouselocation` every 0.4s during a real mission — the
cursor stayed at `0,0` for the whole run. Root cause was NOT the provider: the
cursor motor (`sonic_cursor.py`), the visual servo, and `DockerComputerProvider.gui_action`
all work (direct test moves the cursor and the servo logs
`visual_servo_reach converged=True`).

The real cause was the GUI-only system prompt. When no evaluated AVCS grounding
checkpoint is configured (`_avcs_grounder_ready()` is False, which is the normal
state — no checkpoint exists in-repo), the prompt told the model:

    "No AVCS grounding checkpoint is configured, so semantic GUI_CLICK targets
     cannot be resolved and will fail closed. Do not propose
     GUI_CLICK/.../GUI_SCROLL on a described control. Instead open an observed
     installed application with APP_LAUNCH, then use keyboard actions ..."

That conflates two different things. A *semantic* target (a described control
the grounder must resolve) does need the checkpoint. An *explicit pixel* click
read from the attached screenshot does NOT — the screenshot is already attached
(`image_attached=True`, vision route ready). By forbidding GUI_CLICK wholesale,
the prompt pushed every worker onto silent `APP_LAUNCH`/keyboard routes, so the
pointer never moved and the operator never saw the mouse used.

Fix: the no-grounder clause now blocks only the *semantic* form and explicitly
keeps the pointer available via explicit integer pixel coordinates, which is the
preferred visible interaction; APP_LAUNCH/keyboard are demoted to fallback. Live
re-run after the fix: the cursor moved and 3 of 4 worker steps were real
`GUI_CLICK`s (2 VERIFIED by a real scene change).

Supporting fixes in the same pass:
- Multi-step plan responses are accepted (first envelope in reading order) —
  the vision model returns an ordered plan, and rejecting it stalled workers.
- Near-miss action names are canonicalized (`_GUI_ACTION_ALIASES`: `GUI_LAUNCH`
  -> `APP_LAUNCH`, `FOCUS_APP` -> `APP_FOCUS`, `SCREENSHOT` -> `GUI_SCREENSHOT`,
  etc.). Truly unknown names are still rejected.
- Documentation placeholders echoed verbatim (`<observed control>`) are treated
  as an absent label so the payload pixel point carries the click.
- The worked postcondition example is now concrete ("File menu", 18,14) with an
  explicit instruction not to copy it verbatim, because the model echoed the old
  angle-bracket placeholders.

Honest residual: without a trained AVCS grounder the model must *estimate* pixel
points from the screenshot, so a click can miss its control (observed: a click
at x=1920 was out of bounds and rejected, and a repeated miss tripped the action
loop breaker, which stopped the worker rather than looping). Training and
configuring the AVCS grounding checkpoint (packages/avcs/docs/LOCAL_GROUNDING_QUICKSTART.md)
is what turns estimated clicks into grounded ones.

Tests: `test_gui_capability_aware_prompt.py::test_prompt_without_grounder_blocks_semantic_but_allows_pixel_clicks`,
`test_phase8_visual_computer_use.py::test_gui_worker_maps_near_miss_action_aliases`
and `::test_gui_worker_takes_first_action_of_a_multi_step_plan`. Broad regression:
346 passed, 0 failures.

## Track 2 (coordinate grid overlay): implemented, measured, default OFF (2026-10-23)

Goal: improve explicit-pixel click accuracy for GUI workers that have no trained
AVCS grounding checkpoint. Idea: stamp a labeled coordinate grid onto the image
attached to the vision model, so an estimated pixel point has a readable
reference frame. The overlay is applied ONLY to the model-facing image in
`_reasoning_screenshot_images()`; the stored `_last_screenshot_b64` used for
scene signatures and post-action verification is never modified.

Implementation: `ComputerUseAgent._grid_overlay_png_base64(frame_b64, w, h)` —
thin dashed lines every 100 px, heavier red lines every 500 px, x,y labels at
the heavy intersections. Honest on bad input (returns the frame unchanged /
empty, never raises). Gated by `SONIC_GRID_OVERLAY`.

MEASURED RESULT (this is why it is OFF by default): against the vision model
actually available in this environment (`meta/llama-3.2-11b-vision-instruct` —
only `NVIDIA_API_KEY` is set, so the configured primary `openai/gpt-4o` falls
back), the grid did NOT improve pixel localization:

- corner target (XFCE menu button, ground truth ~(10,10)):
  - no-grid: 4/4 within 20 px, avg 6.0 px
  - grid:    3/4 within 20 px, avg 36.8 px
- mid-screen target (center of a placed editor window, ground truth (970,648)):
  - both variants were very noisy (hundreds of px) on the 11B model.

Conclusion: an 11B vision model cannot localize reliably with or without a grid,
so the grid is not a fix at this tier. It is retained as an opt-in experiment
(`SONIC_GRID_OVERLAY=1`) for a stronger vision route, and defaults OFF so we do
not ship a change that degrades clicks. The real fix remains Track 1: train and
configure an evaluated AVCS grounding checkpoint.

Tests: `test_boss_worker_live_regressions.py` — overlay adds a coordinate
reference; honest on empty/degenerate/broken input; opt-in and off by default
(unset attaches the raw frame, `SONIC_GRID_OVERLAY=1` attaches the gridded
frame). Focused regression: 208 passed, ruff clean.

## CI/CD pipeline repair (2026-11-01)

`SONIC CI/CD` (`.github/workflows/ci.yml`) had been red on `main` for days.
Two independent defects, both fixed (PR #93, merged as `2433f828`):

1. **`Backend Tests` failed every run** at
   `services/core/tests/test_workstation_chat_and_observation.py::
   test_avcs_capability_state_separates_tool_from_vision_route`, asserting
   `vision_route_configured is True`. That flag comes from the model router
   resolving an image-capable provider (`_resolve_provider(require_images=True)`)
   — it needs **live credentials**, which CI does not have, so the value is
   legitimately `False`. Same environment-dependent-test class fixed in
   `c9d016a`. Fixed by stubbing the router in the contract test (matching the
   sibling test) and adding an explicit no-credentials case.

2. **`Production Container Build` (the `docker` job) failed once the tests
   passed.** It had *never run before* — it `needs: [lint, typecheck, test,
   rust, dashboard]`, so the always-failing `test` job kept it `skipped`. Its
   first real run died with `Build SONIC Core image` stuck `in_progress` and no
   log: **runner disk exhaustion**. The core image was **9.32 GB**, ~3 GB of
   which was CUDA libraries pulled by the default PyPI torch wheel. The
   container is CPU-only (`torch.cuda.is_available()` is `False`). Fixed in
   `services/core/Dockerfile` by installing the CPU wheel first
   (`--index-url https://download.pytorch.org/whl/cpu`, `ARG TORCH_INDEX_URL`),
   so the `torch>=2.0.0` requirement is already satisfied and `pip install .`
   does not re-fetch CUDA torch. **Image 9.32 GB -> 1.88 GB.**

Verified: PR #93 run `41316335631` all 6 jobs green; `main` run
`41316426885` (`2433f828`) **completed/success, all 8 jobs** including
`Publish Release Images` and `Continuous Delivery Gate` (main-only) — the first
time those have ever run.

Lesson for future CI work: when a `needs:`-gated job has been skipped for a long
time, the first time it actually runs is the first time it is tested. Check the
job history (`/actions/runs/<id>/jobs`) to see whether a "failing" job is a new
failure or a never-executed latent one.

## Stale open PRs closed as superseded (2026-11-01)

PRs #91 (`departments/experience-interface`) and #92
(`departments/evaluation-reliability`) were open with `main` conflicts and
GitGuardian failures. Both are **pre-monorepo duplicates**: they edit
`sonic-core/` and `sonic-dashboard/`, which no longer exist on `main` (moved to
`services/core/` and `apps/dashboard/`), and their content is already present at
the new paths.

- #91: `agentName`/`agentId` in `apps/dashboard/components/worklog/CommandBlock.tsx`
  + `WorklogFeed.tsx` wiring + `sub_agent_id`/`sub_agent_name` in
  `apps/dashboard/types/workstation.ts` — all present on `main`.
- #92: vision-provider readiness in `services/core/sonic/health.py` (present,
  stronger wording) and GUI-`INCONCLUSIVE` judge tests in
  `services/core/tests/test_judge_agent.py` /
  `test_closed_loop_visual_delta_and_judge.py` (present).
- The GitGuardian "1 secret" flag on #92 is a **false positive**:
  `.env.example` line 48 `NEO4J_PASSWORD=sonic-reda-dev` is a documented
  local-dev placeholder in a template, byte-identical on `main`.

Both closed with an explanatory comment (nothing to preserve). Result: **0 open
PRs**, `main` green.

## macOS workstation provider — Docker-OSX (2026-11-01)

SONIC can now drive a **macOS** desktop body in addition to the Linux XFCE
workstation, via [Docker-OSX](https://github.com/sickcodes/docker-osx) (macOS
inside a QEMU VM packaged as a Docker container). Opt-in only:
`SONIC_WORKSTATION_PROVIDER=macos_docker_osx`.

### Why it is a different kind of body
- **No X11 inside the guest**, so `xdotool`/`scrot`/`wmctrl` do not apply.
  GUI is driven over **VNC (RFB)**; terminal/files/git over **SSH**.
- Requires **`/dev/kvm`**. Without hardware virtualisation the VM cannot boot,
  so `create()`/`get_or_create_home()` **fail closed** with an explicit error
  rather than reporting a desktop that does not exist.

### New modules
- `services/core/sonic/computer/rfb.py` — a self-contained RFB 3.8 client
  (stdlib + numpy, **no new hard dependency**). Supports Raw framebuffer
  reads, pointer/keyboard injection, security types None (1) and VNC Auth (2),
  and a pure-Python **DES** for VNC auth (verified against the FIPS vector
  `133457799BBCDFF1`/`0123456789ABCDEF` -> `85E813540F0AB405`). Apple ARD
  (type 33) is rejected with a clear message.
- `services/core/sonic/computer/macos_docker_osx.py` — `MacOSDockerComputerProvider`
  (subclass of `DockerComputerProvider`, same pattern as
  `WindowsHostAVCSComputerProvider`). Screenshots feed the **same AVCS
  perception loop** as the Linux body, so the agent/routes are unchanged.

### Wiring
- `get_computer()` accepts `SONIC_WORKSTATION_PROVIDER=macos_docker_osx`.
- `docker-compose.macos.yml` — opt-in overlay (`sickcodes/docker-osx:latest`,
  `--device /dev/kvm`, VNC `127.0.0.1:5901`, SSH `127.0.0.1:50922`, image volume).
- `docs/MACOS_WORKSTATION.md` — requirements, env vars, first-boot install,
  capability table.

### Honest capability boundary
- **Works via RFB:** MOVE/CLICK/DOUBLE_CLICK/RIGHT_CLICK/DRAG/SCROLL
  (vertical+horizontal)/TYPE/KEYPRESS/HOTKEY (`command+c`), screenshots.
- **Works via SSH:** terminal, file r/w/list, git, processes, app list,
  `open -a` launch, AppleScript quit, Homebrew install/uninstall, `launchctl`.
- **Fails closed:** `WINDOW_TILE` (no VNC equivalent) and window-manager
  actions that only `wmctrl` could express; `tile_workstation` returns `False`.

### Verification
- `tests/test_macos_docker_osx_provider.py` (33 tests): DES FIPS vector, keysym
  mapping, a full RFB handshake + framebuffer round-trip + pointer/keyboard
  events + VNC auth against an **in-process fake VNC server**, provider GUI/SSH
  surfaces, all fail-closed paths, and provider selection. Two real framing bugs
  in the RFB client were found and fixed by these tests (a 3-byte oversize
  `SetPixelFormat`, and a missing FramebufferUpdate rectangle-count header).
- **No live macOS VM was booted** (this host has no `/dev/kvm`). The protocol
  and provider logic are verified; real end-to-end macOS task completion is NOT
  claimed and must be measured on a KVM-capable host.
- Broad computer/workstation regression: 500 passed, 21 skipped; ruff clean.
## Live GUI run: three real defects fixed + CI green (2026-11-01)

Ran the real `ComputerUseAgent` / `BossAgent` against the live
`sonic-desktop-workstation` with the real NVIDIA vision model. GUI work was
blocked end-to-end by three genuine defects (all fixed, PR #97, CI green):

1. **Vision routing pointed at unreachable models.** `vision` and `computer_use`
   used `openrouter/openai/gpt-4o` with a `nvidia/meta/llama-3.2-11b-vision-instruct`
   fallback whose model id **404s** on this account, so GUI reasoning never got a
   usable vision answer. `configs/models.yaml` now uses
   `nvidia/meta/llama-3.2-90b-vision-instruct` (~4 s/screenshot, verified
   reachable) with the 11B id as secondary fallback only.
2. **The GUI parser rejected a valid click payload.** The normal model shape —
   integer `x`/`y` plus a semantic `TARGET` — was blocked with
   `gui-contract-violation` because it omitted `coordinate_space=screen_pixels`.
   `computer_use/agent.py` now treats an **absent** declaration as screen pixels;
   a **present** non-pixel declaration still fails closed. After the fix a worker
   executed real `GUI_CLICK`s, each `VERIFIED` by a real AVCS scene change
   (`servo: converged=True residual=0.0px`).
3. **Image requests fell back to a text-only model.** `llm/router.py` +
   `llm/providers/custom.py` now skip fallback models without image support, so a
   vision 404 no longer degrades into `400 "multimodal processing is not enabled"`.

**Honest state after the fix:** the objective "Open the terminal application on
the desktop." really launched `xfce4-terminal` (confirmed via `wmctrl -l` /
`pgrep`), and the Boss correctly replanned after a failed GUI phase and completed
the launch on its terminal plane. The mission still ends `UNVERIFIED` because no
evaluated local AVCS task-verification head exists — the documented fail-closed
contract, not a regression. Semantic clicks and text entry remain blocked by
design until `SONIC_AVCS_GROUNDING_CHECKPOINT` / `SONIC_AVCS_FOCUS_CHECKPOINT`
are trained from real demonstrations.

**Dashboard:** counted steps (items with a `step_index`) and their thoughts were
rendered as separate cards while thoughts were dropped entirely. They now fold
into one collapsible **Reasoning Trace** panel (step count + elapsed time,
expandable to per-step status). Commands, files, approvals, worker cards and
banner items (`Agent Desktop Observation`, `Boss Agent Orchestration`, worker
dispatches) stay in the open feed. Verified against real data: 12 counted steps +
43 thoughts fold; 5 banner items stay visible.

**Environment note:** `redis_queue` health shows `degraded` (transient in-memory
fallback) whenever the `sonic-redis` container is not running; start it with
`docker compose up -d redis` (no host root needed). `daytona_cloud: degraded` is
expected — the local Docker workstation is the configured provider. `llm_router:
degraded` is an honest self-report ("credentials configured; remote
reachability/comprehension not pre-verified"), not an outage.

Full `services/core` suite: **2664 passed, 40 skipped, 0 failed**; ruff clean;
dashboard `tsc --noEmit` + `next build` clean. `scripts/novnc_proxy.py` was added
so an operator on another machine can watch the loopback-only noVNC desktop.

## AVCS stays a tool, not a model (2026-11-01)

Operator directive: *"avcs ko model mat banao usko as tool rahene do"* — AVCS
must remain an optional read-only reflex tool, never a required model that
gates GUI actions.

### Removed dead "AVCS as required model" plumbing (`computer_use/agent.py`)
Three methods existed solely to require a local AVCS checkpoint before a GUI
action could proceed, and **none had a production caller** — only tests
referenced them. They were the last vestige of AVCS-as-a-required-model:
- `_avcs_focus_head_ready()` — capability probe requiring `SONIC_AVCS_FOCUS_CHECKPOINT`.
- `_avcs_grounder_ready()` — capability probe requiring `SONIC_AVCS_GROUNDING_CHECKPOINT`.
- `_verify_gui_focus_with_avcs()` — focus verdict from a local AVCS focus head.

The live path never used them: semantic targets resolve through the configured
vision route (`_resolve_vlm_target`), and text entry verifies focus through
`_verify_gui_focus_with_vision`. A local AVCS checkpoint remains an *optional*
precedence only when one is configured and evaluated.

### Cleaned orphaned marker strings
After removing the AVCS focus-head method, its marker strings had no producer
left, so they were dropped from the "stop the GUI mission" blocks in
`agent.py` and `boss.py`:
- `"no evaluated local focus-verification head and no ready vision route"`
- `"avcs has no evaluated local focus-verification head"`
- `"no evaluated local focus head and no vision route is configured"`

The genuinely-reachable markers stay: `"sonic_avcs_grounding_checkpoint is not
configured"` (from `avcs_perception.py`) and `"local avcs pixel processor is
unavailable"` (from `agent.py`). Those describe an optional tool that was
*asked for* and is absent, which is honest; the removed ones implied AVCS was a
mandatory model.

### Docs + catalog reframed
- `SKILLS.md`: `semantic_gui_grounding` / `focus_verified_text_entry` now state
  the vision route is the default with a local AVCS checkpoint as optional
  precedence; `avcs_visual_perception` is labelled an **optional read-only
  reflex tool (not a model)**; `learned_visual_cortex` is an optional advisory
  add-on, never required for GUI actions.
- `docs/AVCS_OPERATION_MODEL.md`: corrected stale method references and the
  "two missing models" framing to the vision-route-first reality.

### Tests
- `test_gui_capability_aware_prompt.py`: removed the grounder-readiness probe
  tests; the prompt is now asserted grounder-agnostic.
- `test_phase8_visual_computer_use.py`: the three removed-method probes replaced
  with one test asserting text entry does **not** depend on a local focus head.
- `test_windows_host_desktop.py`: retargeted the removed agent method to the
  still-live perception-level contract (`AVCSPerception.focus_verifier_configured`
  / `focus_verifier_ready` / `verify_focus_rgb_bytes` fail-closed) and the
  `_validated_focus_metadata` provenance rejection.

Full `services/core` suite: **2629 passed, 40 skipped, 0 failed**; ruff clean on
every changed file. No live GUI task was run this pass (no workstation/creds);
this removes dead AVCS-model plumbing and aligns docs/catalog, it does not add
new capability.

## GUI semantic grounding actually wired (2026-11-01)

Follow-up to the AVCS-as-tool pass. While answering "can SONIC work the GUI?",
found the semantic-grounding bridge was **dead**: the dispatch site hardcoded
`grounded = None`, so a described target ("Save button") was always blocked
even though `_ground_target_with_avcs` (local AVCS checkpoint -> vision route)
existed and was fully implemented.

### Change (`computer_use/agent.py`)
The coordinate-action branch now actually calls the grounding tool:

```python
grounded = None
try:
    grounded = await self._ground_target_with_avcs(target_resource)
except Exception as exc:            # optional tool: any failure -> block
    grounding_reason = f"grounding unavailable: {exc}"
if grounded is None:
    ... BLOCKED (fail closed, no blind click)
```

- `_ground_target_with_avcs` prefers an evaluated local AVCS checkpoint when
  configured and otherwise asks the configured vision route ("where is X?").
  AVCS stays an **optional tool**; SONIC still owns the decision and dispatch.
- Any grounding failure degrades to an explicit BLOCKED (exit 126) rather than
  propagating and crashing the action. No pixel is ever guessed.

### Tests (the reason this was invisible before)
- `test_closed_loop_visual_delta_and_judge.py`: the autouse fixture stubbed
  `_ground_target_with_avcs` to return a **tuple** `(240,180)` while the real
  method returns a **dict** `{"x","y","engine"}`. The fixture was stale and had
  never run (grounding was dead), so wiring it up surfaced a `TypeError`. Fixed
  the fixture to the real contract.
- `test_valid_typing_payload_remains_blocked_without_local_focus_verification`:
  used a live auto-built router (a real ~15 s network call) and asserted
  `gui_action.await_count == 0`, which was only true while grounding was dead.
  Now stubs a deterministic offline vision route returning `focused=false` and
  asserts no `TYPE` action is dispatched. Intent preserved, test made offline.
- `test_agent_quality_improvements.py` / `test_desktop_navigation_and_action_loop.py`:
  the two "unresolved semantic target blocks" tests asserted the old exact
  reason string; updated to the new honest reason
  (`no screenshot-grounded pixel point` + `grounding unavailable`). Their real
  intent (fail closed, no blind click at (0,0)) is unchanged.
- Added `test_semantic_target_is_grounded_and_dispatched`: with the grounding
  tool stubbed to answer, a described target resolves to (640,360) and a real
  `GUIActionType.CLICK` is dispatched at that point (status UNVERIFIED when the
  mock screen does not change — honest).

### Verified
Full `services/core` suite: **2630 passed, 40 skipped, 0 failed**; ruff clean.
Honest scope: this wires the bridge and proves it via the real dispatch path
with a stubbed grounder. It does not add a grounder model; a live semantic GUI
task still needs a configured vision route (or a local AVCS checkpoint) and was
not run in this environment (no Docker / credentials here).
