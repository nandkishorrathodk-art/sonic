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
