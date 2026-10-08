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

- Maintain a current estimate of desktop state: active window, focus, visible
  controls, relevant text, application state, and recent changes.
- Prefer incremental signals such as accessibility events, window changes,
  control state, and changed screen regions where the workstation exposes them.
- Keep pixel perception available for visual appearance, layout, icons,
  custom-drawn controls, and other cases structured signals cannot resolve.
- Do not send a full screenshot to a remote model on every action by default.
  Escalate to image-based reasoning when evidence is ambiguous, novel, or
  contradictory.
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

Perception should be a hybrid sensor system rather than an OCR-only pipeline or
a screenshot-per-action pipeline:

1. Reuse fresh accessibility, window, focus, and control-state information.
2. Consume desktop change signals and inspect only relevant changed regions
   when available.
3. Use screenshot pixels and a vision model when semantic visual evidence is
   actually needed.
4. Use independent task-level verification for completion claims.

No single model, OCR engine, accessibility API, or world model can fully
represent every application. SONIC should combine sensors and expose
uncertainty instead of pretending one source is complete.

## Delivery and acceptance roadmap

### 1. Trustworthy action contract

Every GUI action must bind to fresh perception, a unique target or explicit
pixel point, screen dimensions, active-window context, and a recorded receipt.
Unresolved targets stay blocked; they must never become success-shaped recovery
results.

### 2. Incremental desktop state

Publish versioned state updates from the live workstation, including window and
focus changes, accessibility/control changes, and changed-region metadata.
Measure observation-to-action latency and report capture or event-stream errors.

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

Existing SONIC components provide foundations for this direction, including a
visible workstation, generic GUI actions, perception snapshots, accessibility
and OCR inputs, task orchestration, action receipts, and independent Judge
checks. Their presence does not establish reliable arbitrary-app completion.
Each production path must be checked for actual wiring, and real multi-app
acceptance remains the evidence required for stronger claims.

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
- `GUI_TYPE` requires a current, uniquely resolved AT-SPI editable control;
  OCR-only labels and coordinates without an editable-control match are not
  sufficient. Any supplied point must fall inside that control. SONIC clicks
  the control, obtains a fresh observation that confirms the same control has
  keyboard focus, and only then types. OCR controls retain their own confidence
  and freshness metadata; missing/low confidence or stale controls are not
  indexed as actionable targets.
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
- GUI-only system and user prompts now agree on the evidence source: use the
  attached screenshot when present, otherwise use only the fresh grounded
  accessibility/control inventory. If neither is available, request one visual
  observation and do not guess.
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
- This is a dispatch-safety and evidence-binding foundation, **not** a fast
  action policy: action selection still waits for the configured reasoning
  model, and deep reasoning is not yet running concurrently with a local
  vision-motor controller. No latency improvement or live GUI success claim is
  made by these contracts or their unit tests.

Focused verification run:

```text
python -m pytest sonic-core/tests/test_docker_computer_provider.py sonic-core/tests/test_agent_quality_improvements.py sonic-core/tests/test_vision_perception_and_coordinate_normalization.py sonic-core/tests/test_closed_loop_visual_delta_and_judge.py sonic-core/tests/test_visual_delta_and_grounding_accuracy.py sonic-core/tests/test_phase8_visual_computer_use.py sonic-core/tests/test_boss_gui_completion_gate.py sonic-core/tests/test_workstation_boss_integration.py -q -k "not docker_computer_provider_lifecycle and not docker_computer_provider_files_and_apps"
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

Focused verification from `sonic-core/`:

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

Focused grounding validation from `sonic-core/`:

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

Focused verification from `sonic-core/`:

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

Focused verification from `sonic-core/`:

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

Focused validation from `sonic-core/`:

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

Focused checks from `sonic-core/`:

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
