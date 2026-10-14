# SONIC: General-Purpose AI Computer User

## Product goal

SONIC should work like a capable human computer user. The operator describes an
outcome; SONIC figures out how to achieve it using the computer and applications
available to it. It must not be defined by a single application category,
cybersecurity workflow, vendor, or fixed sequence of actions.

This is the product direction, not a claim that every task is already
implemented or reliable. Capability claims require evidence from real
workstation tasks.

## Expected behavior

### Understand outcomes

- Convert the operator's request into a clear result and observable completion
  conditions.
- Ask for clarification only when ambiguity materially changes the work, risk,
  or expected output.
- Decompose large tasks into adaptive milestones and preserve progress so work
  can continue after a pause or interruption.

### Use the whole computer

- Discover available applications and choose among them based on the task.
- Operate arbitrary installed GUI applications through the visible desktop,
  including opening, focusing, switching, and working across multiple windows.
- Use files, keyboard, pointer, and other available computer capabilities as
  appropriate. A terminal or API may complement computer use, but must not
  silently replace a requested GUI workflow.
- Treat browsers as one application class among many, not the definition of
  computer use.

### Perceive continuously and efficiently

- Maintain a current estimate of desktop pixels and recent visual changes.
- Process workstation frames locally through AVCS. Do not send workstation
  screenshots to any model, including workstation reasoning or the Judge, or
  use OCR, AT-SPI, DOM, or accessibility controls as alternate workstation
  grounding sources.
- Use AVCS reflex state as compact text context for reasoning, and use only a
  configured, evaluated AVCS head for semantic pointer grounding. Freshness,
  coordinate bounds, and action safety must still be checked before dispatch.
- Do not claim that local reflex features establish visible text, focus, semantic
  controls, or task completion. Missing AVCS capability must fail closed.
- For non-GUI tasks, avoid unnecessary image transfer; use image reasoning when
  visual appearance, layout, icons, custom-drawn controls, or ambiguous state
  require it.
- Treat stale, incomplete, or mismatched observations as unusable for grounding.

### Act, adapt, and recover

- Ground each GUI action in current evidence, a unique target, valid screen
  bounds, and the intended active window.
- Associate meaningful actions with expected effects; compare those expectations
  with fresh observations after dispatch.
- If the interface differs from expectation, stop the affected action sequence,
  re-observe, and replan instead of repeating blind clicks.
- Block actions when the target is ambiguous or unresolved. Request human
  approval for actions that are sensitive, destructive, or irreversible under
  the safety policy.
- Resume from verified checkpoints after interruption without claiming that
  unverified work succeeded.

### Prove completion

- Define observable postconditions for milestones and final outputs.
- Verify important results in the application or from the real saved artifact;
  do not accept a model's completion statement as proof.
- Report completed, incomplete, blocked, and unverified work separately.
- Preserve user data and artifacts unless the operator explicitly asks for
  cleanup.

## Architectural direction

The intended computer-use loop is a stateful, event-driven control system:

```text
operator outcome
  -> adaptive task plan and checkpoints
  -> live desktop state estimate
  -> grounded application action
  -> incremental state-change observation
  -> postcondition check
  -> continue, replan, request approval, or report
```

Workstation perception follows a pixel-only AVCS contract:

1. GUI decisions require a fresh workstation frame processed locally by AVCS.
   Reasoning may receive a compact AVCS summary; raw pixels remain local unless
   an explicitly authorized dashboard snapshot is requested.
2. Do not use OCR, AT-SPI, browser DOM, or accessibility controls as alternate
   workstation perception or target-grounding sources. Non-visual window
   metadata may support telemetry and safety checks but does not identify UI
   meaning.
3. Use AVCS pixel-derived change signals to decide when to observe again.
   Reflex state and coarse regions are not semantic controls or pixel targets.
4. Use independent task-level verification for completion claims; AVCS scene
   change alone is not proof of the requested outcome.

## Delivery and acceptance roadmap

### 1. Trustworthy action contract

Every GUI action must bind to fresh perception, a unique target or explicit
pixel point, screen dimensions, active-window context, and a recorded receipt.
Unresolved targets stay blocked; they must never become success-shaped recovery
results.

### 2. Incremental desktop state

Publish versioned AVCS frame state and pixel-derived changed-region metadata.
Keep window metadata separate from visual grounding. Measure
observation-to-action latency and report capture or stream errors.

### 3. Event-driven action verification

Check expected effects using the cheapest trustworthy state signal first.
Capture or route a full image only when the state feed cannot answer the
verification question.

### 4. Long-task checkpoints

Persist task intent, milestones, verified artifacts, pending decisions, and the
latest known application state. Resume only from evidence that remains valid.

### 5. Multi-application acceptance

Test a meaningful, multi-step task involving at least two non-browser GUI
applications. Require:

- actual visible GUI actions and worker receipts;
- fresh observations before grounding and after important actions;
- verified milestone and final-output postconditions;
- independent Judge confirmation;
- truthful reporting of failures and uncertainties;
- measured end-to-end duration and avoidable full-frame/model calls.

A passing unit test, successful application launch, screenshot, or browser-only
task is not sufficient to claim general-purpose computer-use capability.

## Non-goals and guardrails

- Do not implement a fixed macro for each application.
- Do not make SONIC a browser-only agent, terminal-only agent, or
cybersecurity-first product.
- Do not remove visual sensing; avoid unnecessary full-frame transfer, not
pixels themselves.
- Do not trade safety, user approval, or truthful completion reporting for
lower latency.
- Do not claim universal application coverage. Expand capability from measured
acceptance evidence.

## Current evidence boundary

The current Docker and Daytona workstation screenshot paths validate captured
frames and process them locally through AVCS. In the active workstation
reasoning path, the LLM receives a compact AVCS state summary rather than the
workstation screenshot. This statement is scoped to those workstation paths;
it is not a claim that OCR, NSI, accessibility, or image-processing code has
been removed everywhere else in the repository. Older entries below describe
earlier implementations and are retained as historical records.

AVCS reflex output describes pixel-level scene and change signals. It does not
by itself identify UI meaning, establish a grounded semantic target, verify
keyboard focus, or prove that an operator's requested task succeeded. A
configured optional grounding checkpoint is subject to training-provenance
and held-out-evaluation metadata gates; no live acceptance evidence here
establishes that such a checkpoint is available or reliable in the deployed
workstation. Missing or inconclusive perception/grounding evidence must block
the affected action. GUI text entry remains unavailable without an evaluated
local focus-verification capability.

Unit tests and frame-processing checks do not establish arbitrary-app task
success. No live AVCS workstation acceptance task or OSWorld-2 comparative
benchmark is recorded here, so no superiority or performance comparison is
claimed.

### First implementation slice (2026-10-01)

- The Docker NSI worker continuously publishes a compact state sidecar with
  frame/control timestamps, dimensions, active window, visible text, controls,
  and changed-region metadata. It excludes screenshot Base64.
- `DockerComputerProvider.observe_state()` consumes that structured feed;
  `ComputerUseAgent.observe()` prefers it when valid and fresh, otherwise it
  retains the existing screenshot capture path.
- Fresh state-only screen bounds can validate pointer coordinates without
  forcing a pre-action screenshot.
- GUI reasoning omits image transfer only when a fresh, unambiguous AT-SPI
  scene is available. Missing, OCR-only, stale, or ambiguous control state
  causes an on-demand visual capture when a live workspace is available.
- Unresolved visual targets remain blocked rather than falling back to guessed
  landmarks.
- At the time of this 2026-10-01 implementation slice, `GUI_TYPE` required a
  uniquely resolved AT-SPI editable control and confirmed focus through
  accessibility state. This describes the non-GUI-only compatibility path;
  the later visual-first GUI-only contract below supersedes it for workers
  assigned to visible GUI work.
- GUI action selection uses a single JSON-schema response contract. It does
  not simultaneously advertise native function calling to models whose
  configured capabilities do not declare tool use.

This is a state-feed and conditional image-transfer slice, not a complete
real-time event subscription. A live multi-application acceptance task has not
yet been run, so this change does not establish arbitrary-app task completion
or a measured end-to-end latency improvement.

### Perception telemetry and prompt alignment (2026-10-01)

- Each agent observation records its source (`nsi_state`, `screenshot`,
  `screenshot_unavailable`, or another explicit plane state), frame/control
  ages, perception version, screenshot-pixel availability, and whether the
  current AT-SPI scene passed the freshness/ambiguity gate.
- Each reasoning request records whether image pixels were attached, why visual
  reasoning was used (including stale/ambiguous/missing accessibility state),
  and the worker's cumulative visual-escalation count. The same evidence summary
  is attached to GUI action traces and workstation worklog entries. The action
  latency is recorded on the final trace and in the structured service event
  after the next observation arrives; it is not retroactively inserted into an
  already-streamed worklog item.
- GUI action traces also record **action-start to next-observation acquisition**
  latency. This is a measured loop interval, not a claim about X11 dispatch
  latency alone. Structured events `computer_perception_observation`,
  `computer_reasoning_perception`, and `computer_action_to_state_observed`
  expose these measurements in service logs.
- At the time of this 2026-10-01 implementation slice, GUI-only prompts allowed
  fresh accessibility state when no screenshot was attached. The later
  visual-first GUI-only contract below supersedes that behavior.
- These are instrumentation and contract-alignment changes. They do not remove
  screenshots after GUI actions or prove any fixed accessibility coverage or
  latency target; those require live acceptance across representative
  non-browser applications.

### State-first GUI action receipts (2026-10-01)

- Added `ComputerProvider.gui_action_state()` as a backwards-compatible hook.
  Providers without a structured feed retain their existing screenshot result.
- `DockerComputerProvider` now offers the agent a state-first GUI dispatch:
  after an action it accepts only a fresh NSI state whose perception version is
  newer than the pre-action state. If the feed is missing, stale, or has not
  advanced, it falls back to the existing screenshot capture.
- The agent fingerprints active-window, visible-text, and accessibility
  control state before/after the action (excluding timestamps/version). A
  changed scene is recorded as accessibility evidence, not task completion.
  Unchanged or incomparable state is `UNVERIFIED`; repeated stalls still use
  the existing watchdog. This does not bypass the Judge completion gate.
- State receipts refresh the agent's perception bus and do not leave an older
  screenshot as current visual evidence. Image-dependent reasoning can request
  a fresh frame through the existing escalation path.
- This is an opt-in optimization on the Docker provider only. Other providers
  retain screenshot behavior unless they implement the structured hook.

### Bounded GUI action intents (2026-10-07)

- The `ComputerUseAgent.run_mission()` path wraps each grounded pointer/keyboard
  action in a one-action intent carrying its action, target, payload, expected
  outcome, source perception version, active window, display dimensions, and
  evidence-derived expiration.
- The intent is checked after acquiring the shared-display lease, so queued or
  substituted actions cannot reuse an expired contract. Immediately before
  dispatch, the existing NSI/screenshot revalidation must still establish a
  fresh observation in the same window and display geometry; the existing
  safety policy, grounding, coordinate, and Judge gates remain authoritative.
- An expired or mismatched intent is held before provider dispatch and produces
  a blocked trace. The expected outcome remains a recorded prediction checked
  against subsequent receipts/observations; it is not treated as a machine-
  verified postcondition unless an explicit verifier establishes it.
- The existing motor-burst path is now advertised as an optional chunk for
  GUI reasoning, bounded to four typed actions. Each action must state an
  observable postcondition, a final postcondition is required, and the executor
  checks per-action state change, coordinates, and expected evidence while
  retaining one serialized display lease. Any mismatch stops the remaining
  actions. This can reduce reasoning round-trips for predictable short
  sequences; it does not create a separate learned/local fast policy.
- Single-action GUI clicks, typing, keypresses, scrolling, and drags in the
  mission loop now also require a structured `payload.postcondition` describing
  observable text, window, control, or screen-change evidence. Missing or
  malformed conditions are blocked before dispatch. After dispatch, the agent
  checks the condition against the provider's fresh screen/state receipt:
  matching conditions mark the action `VERIFIED`; unavailable or mismatching
  evidence marks it `UNVERIFIED` and feeds the reason into the next reasoning
  step. This is action-level verification only; it does not replace the
  independent Judge's final-goal check. Pointer movement, waiting, and
  observation-only actions are excluded because they need different measurable
  postconditions.
- Regression coverage for matching/mismatching postconditions and missing
  conditions passed in `test_agent_action_receipts.py`. These are mocked
  provider tests, not a live desktop acceptance result.
- This is a dispatch-safety and evidence-binding foundation, **not** a fast
  action policy: action selection still waits for the configured reasoning
  model, and deep reasoning is not yet running concurrently with a local
  vision-motor controller. No latency improvement or live GUI success claim is
  made by these contracts or their unit tests.

Focused verification run:

```text
python -m pytest services/core/tests/test_docker_computer_provider.py services/core/tests/test_agent_quality_improvements.py services/core/tests/test_vision_perception_and_coordinate_normalization.py services/core/tests/test_closed_loop_visual_delta_and_judge.py services/core/tests/test_visual_delta_and_grounding_accuracy.py services/core/tests/test_phase8_visual_computer_use.py services/core/tests/test_boss_gui_completion_gate.py services/core/tests/test_workstation_boss_integration.py -q -k "not docker_computer_provider_lifecycle and not docker_computer_provider_files_and_apps"
179 passed, 2 deselected
```

The same command without the exclusions produced 178 passing tests and two
failures in live-container checks (VNC URL availability and workspace file
write) even though the test helper detected a running workstation container.
These are not counted as passing. Neither run is live, multi-application
acceptance evidence.

### Validation snapshot (2026-10-02)

- Backend suite on the development worktree after the post-merge regression
  fixes: **1,699 passed, 39 skipped**. This is automated regression evidence,
  not live desktop acceptance.
- A successful end-to-end task across multiple non-browser GUI applications,
  with fresh observations, actual worker action receipts and independent Judge
  confirmation, has **not** yet been demonstrated.
- The dashboard PNGs in `assets/` are historical UI snapshots and do not show a
  verified task. The updated architecture SVG is a component/flow overview,
  not execution evidence.

### Runtime integrity follow-up (2026-10-07)

- Display leases no longer transfer ownership just because a TTL elapsed.
  Another worker waits for release (and can stop waiting on interrupt), while
  release verifies that the current task still owns the lease.
- Interrupt checks now run between mission phases, before GUI dispatch, and
  between motor-burst actions. An in-flight provider call is cancelled
  best-effort; an action already delivered to the desktop cannot be undone.
- GUI intents bind to both the agent observation version and the NSI state
  version, plus active window, observed focus/target identity when available,
  screen dimensions, and evidence age. Missing or stale state-only evidence
  blocks dispatch rather than reusing cached metadata or guessing.
- The Docker NSI cache is updated only after state validation accepts the feed.
  Its state version advances on semantic desktop changes rather than every
  capture, so unchanged state can be revalidated without weakening freshness.
- A successful generic command receipt is not completion evidence by itself:
  headless confirmation requires a successful execution plus independent
  verification evidence, and file outcomes are read back when the provider
  supports it. Boss completion also requires successful Judge-verified worker
  outcomes whose sub-mission goals cover the operator objective; unrelated
  verified work cannot complete the mission.

Focused verification from `services/core/`:

```text
python -m pytest tests\test_action_contracts.py tests\test_teamwork_subagents_hardening.py tests\test_motor_execute_burst.py tests\test_workstation_interrupt_and_resilience.py tests\test_docker_computer_provider.py tests\test_agent_quality_improvements.py tests\test_judge_agent.py tests\test_boss_gui_completion_gate.py tests\test_phase8_visual_computer_use.py tests\test_agent_action_receipts.py tests\test_agent_motor_burst.py tests\test_closed_loop_visual_delta_and_judge.py tests\test_security_audit_remediation.py tests\test_e2e_autonomous_loop.py tests\test_boss_concurrency_and_summaries.py tests\test_boss_agent.py -q
345 passed, 2 skipped, 1 warning (AsyncMock coroutine warning in an existing e2e regression test)
```

This is mocked/unit regression evidence only. No live Docker/NSI or
multi-application desktop acceptance run was performed; general-purpose GUI
task reliability remains unproven.

### Coordinate grounding edge cases (2026-10-07)

- Explicit pixel coordinate metadata now takes precedence over normalized-value
  heuristics, so small pixel coordinates such as `(1, 1)` are not scaled across
  the display. Normalized `[0, 1000]` responses remain explicitly scaled.
- Direct coordinate queries now reject points outside the observed screen bounds
  in both synchronous and asynchronous grounding paths; they are never clamped
  to an edge coordinate.

Focused grounding validation from `services/core/`:

```text
python -m pytest tests\test_visual_delta_and_grounding_accuracy.py tests\test_visual_grounding.py tests\test_vision_grounding_speed_and_accuracy.py tests\test_semantic_target_grounding.py tests\test_phase8_visual_computer_use.py -q
124 passed
```

### Boss fallback routing and interrupt boundary (2026-10-07)

- If strategic decomposition fails or returns no workers, Boss now asks Laya to
  select the execution plane using the original operator objective. An explicit
  plane constraint still wins; uncertain/failed inference is logged and keeps
  the conservative GUI default.
- The fallback worker now receives the operator's objective unchanged instead
  of a generated wrapper sentence, preventing routing and completion checks
  from reasoning over an altered goal.
- If interruption is already signaled when Boss starts, it returns an
  incomplete, no-actions report before planning or dispatch.

Focused verification from `services/core/`:

```text
python -m pytest tests\test_boss_plane_routing_and_truthfulness.py tests\test_boss_agent.py tests\test_dag_and_replan_orchestration.py tests\test_workstation_boss_integration.py tests\test_workstation_interrupt_and_resilience.py -q
62 passed
```

### Desktop provider outcome checks (2026-10-07)

- A scroll action with a zero delta is now rejected by the shared `GUIAction`
  contract. Previously Docker interpreted zero as three upward wheel events,
  while Daytona dispatched no effective scroll.
- Docker window focus now reads back the active window after requesting focus
  and reports failure if the active title or window ID does not match.
- Docker window tiling now reads the display dimensions instead of assuming
  1280x800, and fails if window unmaximization or placement commands fail.
- Tiling tests now provide discovered windows and a display size; an empty
  window listing is not accepted as a successful tiling outcome.

Focused verification from `services/core/`:

```text
python -m pytest tests\test_docker_computer_provider.py tests\test_human_motor_reflexes_and_hotkeys.py tests\test_multi_window_management_and_structured_workflow.py -q
76 passed, 2 skipped
python -m ruff check sonic\computer\models.py sonic\computer\docker_computer.py tests\test_docker_computer_provider.py
All checks passed
```

These are mocked provider regressions, not a live desktop acceptance run. A
broader Ruff invocation over the two existing motor/tiling test files still
reports unused imports that predate these changes; they were left untouched.
No claim is made here that a launched app or tiling command proves task
completion.

### Consequential action approval (2026-10-07)

- The autonomous agent now pauses for a tenant/session-scoped operator decision
  before Git commit/push, application installation, policy-classified
  intrusive terminal commands, or GUI controls whose grounded label indicates
  sending, publishing, purchasing, or destructive changes. Raw action payloads
  are not copied into approval worklog items; only the action type, target
  preview, and approval reason are shown.
- The workstation approval endpoint resolves only a currently pending,
  random-ID request belonging to the authenticated operator's tenant and
  session. Rejection, timeout, missing approval callback, or callback failure
  blocks dispatch. The dashboard displays explicit approve/reject controls.
- Native `Deny` always blocks in the direct agent-to-provider path.
  `RequireApproval` now requires the trusted workstation operator callback;
  the model cannot self-approve by adding a payload field. A GUI intent
  remains bound to its original target and state; after approval, final
  dispatch still revalidates fresh desktop evidence.
- This semantic control-label gate is a targeted safeguard, not a general
  classifier for all possible consequences in every application. A control
  without a grounded semantic label cannot be classified by its meaning.

Focused validation from `services/core/`:

```text
python -m pytest tests\test_agent_action_receipts.py tests\test_workstation_action_approval.py tests\test_native_kernel_bridge.py -q
31 passed
python -m ruff check sonic\computer_use\agent.py sonic\api\routes\workstation.py tests\test_agent_action_receipts.py tests\test_workstation_action_approval.py
All checks passed
```

Dashboard production build:

```text
npm run build
Build completed successfully; existing React hook and image optimization warnings remain.
```

All approval-route/provider tests are mocked; no live workstation action was
approved or dispatched as part of validation.

### Operator outcome feedback (2026-10-07)

- Delegated work is shown as `COMPLETED` only when the worker reports success,
  the independent Judge verdict is `CONFIRMED`, and evidence is marked verified.
  Positive worker claims without all three signals remain `UNVERIFIED`.
- Boss report outcomes are visibly differentiated: `COMPLETE` is distinct from
  `PARTIAL` and `INCOMPLETE`; partial/incomplete reports include the actual
  summary. A blocked workstation now has an explicit alert with its current
  blocker, and paused / waiting-for-human states retain their separate
  continuation guidance.
- This is presentation of existing backend receipts and verdicts; it does not
  create verification evidence or change the backend's completion decision.

Dashboard validation:

```text
npm run build
Build completed successfully; existing React hook and image optimization warnings remain.
npm run lint
Passed with existing React hook and image optimization warnings.
```

### NSI and live-workstation readiness check (2026-10-07)

Focused checks from `services/core/`:

```text
python -m pytest tests\test_observation_freshness.py tests\test_perception_bus.py tests\test_docker_computer_provider.py tests\test_action_contracts.py -q -rs
68 passed, 2 skipped
Skipped: two shared-workstation provider tests require SONIC_RUN_LIVE_WORKSTATION=1.
```

The Docker inventory showed `sonic-desktop-workstation` running with VNC/noVNC
ports published. This is only container availability evidence: no live NSI
state-feed freshness, configured vision-provider readiness, or GUI task
postcondition was checked. The opt-in shared-workstation tests were not run, so
live computer-use readiness and task completion remain unverified.

### CI regression follow-up and isolated acceptance blocker (2026-10-08)

- Reproduced the repository CI environment locally with
  `APP_ENV=development` and `SONIC_ALLOW_HOST_EXECUTION=false`. The full backend
  suite completed with **1,934 passed, 41 skipped, 0 failed**. Two remaining
  test assertions had expected an older generic stale-state message; they now
  assert the concrete perception-version mismatch, and both pass.
- The P0 security and sealed-action safety suites passed: **60 passed**.
  The separate CLI suite passed **68 tests**.
- Ruff passed for `services/core/sonic` and `apps/cli/sonic_cli`; the documented
  six-file mypy check reported no issues; `git diff --check` passed.
- Local validation used Python **3.14.7**, while CI specifies Python **3.12**.
  These results therefore do not replace a CI run on its configured runtime.
- Docker Desktop was started for the isolated acceptance check. The local API
  and Docker engine became healthy, and `sonic-acceptance-20261008` was
  restarted on localhost-only VNC/noVNC ports. A real screenshot was captured
  at 1280x800 with a 109 ms age, but the workstation status reported
  `perception_version=0`; the NSI state sidecar was absent, so this was not
  accepted as coherent, versioned NSI evidence.
- Model readiness remains blocked. The configured NVIDIA vision model returned
  HTTP 403 (authorization failed). The configured Groq Qwen route returned a
  response to an attached screenshot but said it could not inspect the screen;
  this is not vision-readiness evidence. The local health endpoint's
  `llm_router=healthy` only means providers were configured, not that a usable
  visual model was authenticated or able to interpret pixels.
- At this initial readiness check, no GUI task or GUI action was dispatched.
  The later retry and its outcome are recorded below.

### Isolated GUI retry and runtime fixes (2026-10-08)

- The API was temporarily bound to `sonic-acceptance-20261008` and reported its
  localhost-only noVNC port (16091) and fresh desktop evidence before the retry.
  Afterward, the API was restored to its normal `sonic-desktop-workstation`
  mapping; no prompt was sent to the shared desktop.
- SONIC's GUI-only worker did not complete the folder task. Its GUI click
  receipts were blocked before dispatch because the bounded intent expired or
  the fresh screenshot no longer matched the state used for reasoning. The
  worker was interrupted when its reasoning returned a non-GUI fallback.
- The actual vision route failed closed: Groq returned HTTP 429, the OpenRouter
  vision provider was not configured, and NVIDIA returned HTTP 403 for the
  configured vision model. A healthy router status did not establish usable
  visual reasoning. No worker receipt or independent Judge confirmation proves
  the requested folder was created; live computer-use success remains
  unverified.
- Runtime fixes now renew GUI intents only after fresh evidence proves the same
  window, geometry, focus, target, and UI state; they continue to block on any
  changed or ambiguous state. Next-phase planning now supplies stable task IDs
  and resolves an exact prior-task description to its unique ID, while unknown
  or ambiguous dependencies remain fail-closed. GUI payload parsing accepts
  wrapped JSON through the shared parser but still rejects non-object payloads.
- Focused regression suites: **120 passed**. Ruff and `git diff --check` passed.
  These are code-contract results, not evidence that the live GUI task succeeded.

Full backend command (from the repository root):

```powershell
$env:APP_ENV='development'
$env:SONIC_ALLOW_HOST_EXECUTION='false'
python -m pytest services/core/tests/ -ra --tb=short
1934 passed, 41 skipped
```

The live acceptance gate remains open until the isolated workstation publishes
fresh, versioned perception, a working vision route interprets its pixels, and
SONIC produces GUI action receipts plus independent Judge evidence for a
harmless task.

### Historical visual-first GUI reasoning and fail-closed vision routing (2026-10-08; superseded below)

- GUI-assigned workers now refresh and attach a fresh screenshot for every
  reasoning decision, even when NSI/accessibility controls are fresh. Their
  current-state prompt omits OCR text, control inventories, and window
  telemetry; screenshot pixels are the source for target interpretation.
- GUI-only workers stop before model reasoning when no fresh screenshot is
  available. `GUI_TYPE` in this contract requires screen-pixel coordinates,
  clicks the grounded point, refreshes pixels, and asks the configured vision
  route to confirm a visible focus indicator at that point. It sends no text if
  the route is unavailable, the response is invalid/uncertain, or focus is not
  visually confirmed at confidence >= 0.9. This is a fail-closed model check,
  not independent proof that the model's focus judgment is correct.
- Before dispatch, a GUI-only action is blocked if the current screenshot pixels
  differ from the pixels used to ground the action, even when OCR/control
  metadata appears unchanged. This avoids using structured metadata to excuse
  a stale visual target; animations and blinking indicators can therefore cause
  safe false blocks and require a new reasoning turn.
- Requests containing images now route only to models configured as
  vision-capable. The router does not strip image bytes to make a text-only
  provider appear successful. Groq's text model was removed from the vision
  routes; the configured routes prefer OpenRouter GPT-4o and fall back to
  NVIDIA Llama Vision.
- These changes make visual input the GUI worker's reasoning source, but do not
  establish that the deployed vision provider is currently authenticated or can
  interpret the screenshot. Live GUI acceptance remains blocked until provider
  readiness is demonstrated and an isolated task passes independent Judge
  verification. Pixels-only perception still requires pixel coordinates for
  motor actions; it is not yet evidence of human-level reliability.

### AVCS-only workstation perception (current contract)

- **Role in SONIC:** AVCS is the local visual-perception layer—the computer's
  digital eye. It turns workstation pixels into pixel-derived scene and change
  signals for SONIC's reasoning loop; it does not plan tasks, interpret
  application intent by itself, or execute mouse, keyboard, or terminal
  actions. Boss/agents reason and plan, the computer provider performs actions,
  and the independent Judge evaluates task-relevant postconditions.
- **System 3 temporal prediction:** the active AVCS reflex path now exposes a
  persistence-baseline prediction error and a hysteretic stability label from
  successive pixel-derived luminance grids. This is untrained advisory
  telemetry only: it does not ground targets, authorize actions, or establish
  task success. It is not evidence of a learned world model or measured
  real-time performance.
- The Docker workstation now captures timestamped RGB frames in memory through
  a persistent X11 stream worker (`mss`), then processes frames locally through
  `AVCSPerception`. The stream uses a bounded latest-frame queue, discarding
  stale intermediate frames instead of building a backlog. `ScreenObservation`
  keeps raw RGB pixels out of normal JSON serialization. The agent's GUI
  dispatch and freshness checks accept these raw frames and use AVCS scene
  fingerprints for action-level change checks; the raw pixels are not sent to
  the workstation reasoning model.
- GUI action intents accept either an encoded screenshot or a dimension- and
  frame-ID-validated raw RGB frame whose AVCS source-frame ID matches. This
  preserves the Docker raw-frame path without converting private pixels to
  Base64 just to satisfy the intent contract; fresh AVCS identity and dispatch
  revalidation remain required.
- The explicit dashboard snapshot endpoints may encode a raw frame as PNG /
  Base64 for display. This compatibility encoding happens at that API boundary,
  not in the agent's perception transport. The Docker worker is configured for
  30 FPS by default (environment cap: 60); this is a requested capture rate,
  not a measured sustained rate. No live rebuilt-container measurement has yet
  established capture/processing FPS, latency, or resource cost.
- This is only the raw-frame transport and AVCS reflex integration slice. It
  does not yet provide semantic affordance discovery, trained
  target grounding, eye-hand servoing, stability-gated action dispatch, or
  safe autonomous retries. AVCS scene change is an action-level signal, not
  proof that a click had the intended effect or that a task succeeded. Target
  grounding and focus verification still require evaluated local heads, and
  independent Judge evidence remains necessary for task completion.
- The scope of this contract is the active workstation perception path. It
  is strictly AVCS pixel perception: do not add OCR/Tesseract, NSI,
  accessibility, or browser-DOM extraction as alternate workstation
  perception or grounding sources.
- GUI intents bind to fresh frame/state evidence and screen bounds. A
  model-proposed coordinate alone is not evidence of target identity:
  semantic target grounding is available only through the optional local AVCS
  grounder when its checkpoint passes the configured provenance and held-out
  evaluation checks. If the required grounding evidence is missing, invalid,
  stale, or inconclusive, the action must fail closed. Explicit pixel points
  still require current-frame bounds and dispatch revalidation.
- AVCS reflex output does not establish semantic target grounding, application
  identity, focus, or task completion. `GUI_TYPE` remains unavailable because
  there is no evaluated local focus-verification head; do not dispatch text
  based on a reflex summary or a model's focus assertion.
- Local processing is a privacy and visual-awareness property, not evidence
  of a continuously sustained frame rate or latency target. Claim real-time
  streaming performance only after measuring the deployed capture and AVCS
  pipeline under representative workstation load.
- The Judge processes fresh screenshot pixels locally through AVCS and must not
  attach desktop pixels to any model request. Scene change, motion, reflex
  events, or a worker's completion proposal alone cannot prove the requested
  semantic postcondition. Until a validated local task-verification head exists,
  GUI completion remains inconclusive without other independent,
  task-relevant evidence.
- A 1920x1080 default display is a resolution setting, not evidence of precise
  semantic grounding or successful general-purpose task completion. Evaluate
  diverse, harmless desktop tasks with fresh action receipts and independent
  Judge evidence before reporting OSWorld-like success rates, latency, or cost.
- No live GUI acceptance run or OSWorld-2 comparison is documented for this
  migration. AVCS processing, unit tests, and checkpoint metadata gates do not
  establish arbitrary GUI task success or comparative superiority.
- The privacy redaction gate and AVCS's own retinal/foveation implementation
  are distinct from workstation OCR perception and are not removed by this
  migration.
