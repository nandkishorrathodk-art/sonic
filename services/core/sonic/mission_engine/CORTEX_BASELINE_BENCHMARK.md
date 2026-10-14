# SONIC Cortex baseline benchmark

## Purpose and boundary

This benchmark evaluates SONIC first and foremost as a general-purpose
AI-human being operating its own graphical computer. Scenarios should resemble
ordinary computer use requested by an operator: observing the desktop,
switching or opening an appropriate application, interacting with visible
controls, authoring or editing an artifact, and verifying the result. They
must not assume a cybersecurity, pentesting, or scanner workflow. Security
work is in scope only when an operator explicitly requests it.

The benchmark checks whether perception, reasoning, action, recovery,
verification, memory, and delegation work together. It does not choose a
central subsystem: each journey crosses multiple layers of the AI-human
workstation.

This is an Evaluation & Reliability artifact, not a feature or data collector.
The current benchmark runs deterministic tests with fixtures and simulated
providers. It does not operate a live desktop, persist execution examples,
inspect production artifacts, call external targets, train models, alter model
routing, replay actions, or promote candidates. The separate
[`AI_ML_EVALUATION_PROTOCOL.md`](./AI_ML_EVALUATION_PROTOCOL.md) governs any
future dataset-based evaluation.

## Cross-layer scenarios

| ID | Journey under test | Existing test coverage | What the result proves |
|---|---|---|---|
| CB-01 | Observe the desktop and combine visible screen, windows, and available controls | `test_phase6_perception_fusion.py`, `test_perception_bus.py` | Component behavior for observed controls/state, not completeness or freshness of a live frame |
| CB-02 | Find a named visible UI target, interact with it, and refuse a blind action if it cannot be grounded | `test_visual_grounding.py`, `test_visual_delta_and_grounding_accuracy.py`, `test_perception_bus.py` | Coordinate/target handling and stale-target defenses in fixtures, not real-world grounding accuracy |
| CB-03 | Perform an ordinary GUI action, compare expected and actual state, and recover from a stalled screen | `test_computer_experience.py`, `test_closed_loop_visual_delta_and_judge.py`, `test_agent_action_receipts.py` | Recorded state deltas, watchdog behavior, and receipt semantics, not causal action-value learning |
| CB-04 | Choose GUI or terminal based on an operator's general computer task and preserve both execution capabilities | `test_laya_end_to_end.py`, `test_laya_system1_engine.py`, `test_boss_plane_routing_and_truthfulness.py` | Current Laya fallback/routing behavior on test inputs, not historical-outcome learning |
| CB-05 | Verify an operator-requested result independently (for example, an opened application or saved artifact) | `test_closed_loop_visual_delta_and_judge.py`, `test_laya_end_to_end.py`, `test_agent_action_receipts.py` | Test-level verification gates, not a live mission completion rate |
| CB-06 | Retain/replay-check general computer-use skills and coordinate delegated work | `test_computer_skill_ledger.py`, `test_teamwork_subagents.py`, `test_boss_parallel_workers.py`, `test_boss_plane_routing_and_truthfulness.py` | Ledger and delegation behavior in isolation/fixtures; not production skill reuse or restart-resumable delegation |
| CB-07 | Calculate a descriptive trajectory baseline without inventing missing data | `test_ai_ml_trajectory_evaluation.py` | Empty data remains unavailable; episode outcome and verification accounting behave as specified |

Run the component benchmark from `sonic-core`:

```powershell
python -m pytest -q `
  tests/test_phase6_perception_fusion.py `
  tests/test_perception_bus.py `
  tests/test_visual_grounding.py `
  tests/test_visual_delta_and_grounding_accuracy.py `
  tests/test_computer_experience.py `
  tests/test_closed_loop_visual_delta_and_judge.py `
  tests/test_agent_action_receipts.py `
  tests/test_laya_end_to_end.py `
  tests/test_laya_system1_engine.py `
  tests/test_boss_plane_routing_and_truthfulness.py `
  tests/test_computer_skill_ledger.py `
  tests/test_teamwork_subagents.py `
  tests/test_boss_parallel_workers.py `
  tests/test_ai_ml_trajectory_evaluation.py
```

## Scorecard and reporting rules

Report test coverage and live operational metrics separately. The component
suite reports test pass/fail counts only. It is not a proxy for real SONIC
success, latency, or accuracy.

For any future controlled live-workstation benchmark, publish each journey's
attempt count, successful count, failures, and unavailable measurements. Define
success before running: the requested state must be reached and independently
verified from a fresh observation or artifact; an agent's completion claim or
provider dispatch status alone is insufficient.

| Metric | Required evidence | Current status |
|---|---|---|
| Observation freshness | Source capture timestamp, capture age, and age refreshed at action selection; unavailable when the source timestamp is missing or invalid | Measurement fields are now propagated for Docker screenshots; live-workstation freshness distribution remains unmeasured. Source-to-host age assumes synchronized wall clocks and must be validated on the live workstation |
| Grounding accuracy | Intended target plus independently observed reached target, on the same episode | Unavailable: no adjudicated live target labels |
| Action success | Typed action receipt and independently checked postcondition | Test semantics covered; live rate unavailable |
| Recovery success | Linked failure, recovery attempt, subsequent verified postcondition | Test behavior covered; episode-level live rate unavailable |
| Verified task completion | Unique task episode, predefined success criteria, independent verifier/evidence | Unavailable: no reviewed real-episode cohort |
| Steps per task / corrections | Complete linked episode trajectory | Unavailable for real tasks |
| Observation and decision latency | Monotonic timestamps around capture, fusion, decision, dispatch, and verification | Unavailable end-to-end; do not infer from daemon sleep interval or unit-test duration |
| Skill reuse | Skill selected, precondition match, replay, and independently verified postcondition | Unavailable in production; ledger code is not wired into the inspected workstation agent constructor |
| Delegation recovery | Durable task identity/checkpoint and verified continuation after process restart | Unavailable; concurrent worker tests do not prove restart resumption |

Never combine fixture test pass rates with live-task success rates. Keep metrics
unavailable until the required evidence is available. Do not treat action
completion, visual change, Laya confidence, or a worker's self-report as
independent goal success.

## Current measured result

On 2026-09-28, the complete listed regression command collected and passed
**122 tests, 0 failures**. These are deterministic regression
checks, not independent scenario episodes; the timing is test-run duration,
not SONIC observation or mission latency.

No live workstation benchmark was run. Accordingly, production observation
freshness distribution, grounding accuracy, action success, recovery success,
verified goal completion, and end-to-end latency remain **unmeasured**, not zero.
The capture-age fields only make freshness measurable when the provider supplies
a valid source timestamp and the provider/host wall clocks are aligned; they do
not certify OCR/control completeness. The next
measurement stage, if authorized, should use ordinary reversible computer-use
tasks chosen by the operator, with predefined goals, fresh observations,
independent verification, and privacy-approved aggregate evidence. It must
not start collection/training or alter the production agent/routing. Do not
introduce a cybersecurity task unless the operator requests one.

## Known integration gaps exposed by this baseline

1. The desktop observer may retain the last screenshot if a new capture fails;
   consumers do not receive an explicit age/freshness proof with every image.
2. Screenshot/OCR/AT-SPI/CDP and semantic target resolution exist, but no
   benchmark-grade intended-target versus reached-target label is produced.
3. `WorldTwin` can record and roll forward observed transitions, but the audit
   found no production call sites feeding/consulting it from the GUI action
   loop; it is not a demonstrated pre-action visual predictor.
4. Experience records are post-action discrepancy signals, not a learned
   action-value or failure-prediction model.
5. `ComputerSkillLedger` has persistence/replay APIs, but its constructor was
   not found on the production workstation-agent construction path. Its
   existence alone does not establish skill reuse.
6. Boss/teamwork tests demonstrate concurrent delegation behavior, not durable
   resume after backend restart.
7. Laya's plane selection is callable in routing/orchestration, but its
   fallback is not shown to learn from outcomes; the receipt-verification
   helper has no production call site in the audited code.

These gaps are evaluation findings, not authorization to add collection,
training, online learning, new routing, or autonomous promotion.
