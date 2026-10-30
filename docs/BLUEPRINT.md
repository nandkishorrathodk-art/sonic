# SONIC product blueprint

## North star

Build SONIC as a general-purpose AI colleague that can use a computer to
complete operator-requested outcomes. The product is not defined by a single
vendor, browser, security workflow, or fixed application sequence.

This is an engineering direction, not a claim of general reliability. The
acceptance criteria in [SONIC_COMPUTER_USER_GOAL.md](SONIC_COMPUTER_USER_GOAL.md)
are authoritative.

## Product loop

1. Understand the requested result and what would count as evidence.
2. Observe the live computer and discover relevant applications and state.
3. Plan bounded, resumable steps; ask for approval when the safety policy requires it.
4. Use the assigned execution plane and ground each visible action in fresh evidence.
5. Re-observe, verify effects, and replan when reality differs from expectation.
6. Have an independent Judge assess GUI completion.
7. Report verified results, partial progress, blockers and uncertainty honestly.

The operator-requested outcome—not a canned checklist—drives the choice of
applications and capabilities. A terminal or API may complement a GUI task,
but must not silently replace an explicitly requested GUI workflow.

## Current building blocks

- FastAPI backend and workstation routes.
- Boss orchestration for task analysis, worker delegation, evidence review and replanning.
- Ordinary `ComputerUseAgent` workers operating the visible desktop.
- Docker-backed local workstation with generic GUI actions and a compact,
  freshness-checked NSI state feed; screenshots remain available when needed.
- Separate execution capabilities for suitable file, terminal and Git work.
- Action receipts, safety controls and independent Judge review.
- Next.js operator dashboard and CLI.

The existence of a building block is not evidence that it is wired into every
production route or reliable for arbitrary tasks. Check the current call path
and test the relevant behavior.

## Product boundaries

- Browsers are one application type among many; there is no dedicated
  browser-specialist worker in the current workstation route.
- Computer-use acceptance must include non-browser desktop applications.
- GUI workers receive a visible-GUI contract and must not fall back to
  unrelated shell actions.
- Ambiguous, stale or off-screen targets are blocked; no guessed click is a
  successful action.
- A screenshot, process launch, model declaration, passing unit test or
  unchanged desktop is not proof of task completion.
- Human approval remains necessary for actions classified as sensitive,
  destructive or irreversible.

## Delivery priorities

### 1. Trustworthy perception and action

Maintain a current state estimate with explicit freshness and coordinate
metadata. Use structured accessibility/window evidence when sufficient, and
escalate to screenshot/vision when required. Record real action outcomes and
stop on repeated invalid visible-GUI proposals.

### 2. General application coverage

Measure useful task completion in multiple non-browser GUI applications.
Discover installed applications dynamically; do not introduce vendor-specific
click macros or hardcoded app routes as a substitute for general reasoning.

### 3. Adaptive long-running work

Persist verified checkpoints and pending decisions. Resume only from state
that remains valid; preserve user-created files and other artifacts.

### 4. Independent verification

Define observable task postconditions. Extend Judge evidence for real artifacts
and application state, and treat missing evidence as incomplete rather than
successful.

### 5. Measured performance and reliability

Track end-to-end task outcomes, failure modes, latency, observation freshness,
avoidable screenshot/model calls and recovery quality. A design proposal or
microbenchmark must not be presented as real task success.

## Readiness statement

The repository has a substantial implementation and regression-test
foundation. The current recorded backend suite result is 1,699 passed and 39
skipped on the development worktree. This is not a live desktop acceptance
result. The open product proof is successful, independently judged,
multi-step work across representative non-browser GUI applications using fresh
observations and real worker receipts.
