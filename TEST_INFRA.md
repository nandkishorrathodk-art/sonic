# TEST_INFRA.md — Autonomous Loop Verification Track Infrastructure

**Author**: E2E Test Writer (`teamwork_preview_test_writer_e2e`)  
**Track**: Autonomous Loop Verification Track (`ORIGINAL_REQUEST.md § R4`)  
**Target Suite**: `sonic-core/tests/test_e2e_autonomous_loop.py`  
**Status**: COMPLETE (31 / 31 Tests Passing, 100% Pass Rate)

---

## 1. Executive Overview & Scope

The Autonomous Loop Verification Track establishes an opaque-box, end-to-end integration and verification test suite covering the entire lifecycle of SONIC's autonomous execution loop:

$$\text{Objective} \longrightarrow \text{BossAgent Decomposition} \longrightarrow \text{Mission Delegation} \longrightarrow \text{ComputerUseAgent Execution} \longrightarrow \text{Perception/State Update} \longrightarrow \text{GUI/Motor Dispatch} \longrightarrow \text{Visual Verification} \longrightarrow \text{JudgeAgent Receipt} \longrightarrow \text{Cognitive State Update} \longrightarrow \text{Task Completion}$$

The test suite exercises all **9 Operational Verification Points** mandated by `ORIGINAL_REQUEST.md § R4` across four progressive test tiers, guaranteeing zero silent failure masking, fail-closed safety enforcement, anti-self-attestation verification, and strict dual-plane fidelity.

---

## 2. Nine Operational Verification Points Coverage

| Point | Verification Requirement | Failure Mode Prevented | Implementation Mechanism Tested |
|---|---|---|---|
| **P1** | **GUI actions do not hang indefinitely** | Deadlock / hung worker freezing orchestrator | Display lease timeouts (`DisplayLeaseManager`), screen watchdog loop abort on consecutive unchanged screens (`ScreenWatchdog`), worker action step timeouts. |
| **P2** | **Invalid/stale coordinates rejected before dispatch** | Blind clicks on vanished buttons or out-of-bounds clicks | Coordinate range bounding checks against screen dimensions ($0 \le x < W$, $0 \le y < H$), screen observation staleness detection against dynamic screen frame hashes. |
| **P3** | **Focus validation occurs before actions requiring focus** | Keystrokes sent to wrong application or dropped | Active window check (`get_active_window()`), pre-type focus click dispatch (`GUI_MOVE` + `GUI_CLICK` before `GUI_TYPE`), and text validation before dispatch. |
| **P4** | **Computer actions produce observable execution results** | Ghost actions with no verifiable effect | Structured `ComputerDecisionTrace` generation with before/after state captures, non-empty terminal stdout/stderr receipts, screenshot hashes, and exit codes. |
| **P5** | **Perception failures propagate correctly** | Agent operating blind on black/corrupt screen | Zero-byte screenshot validation failure, corrupt PNG header detection, negative or infinite perception capture age rejection. |
| **P6** | **Verification failures are not silently swallowed** | Worker self-reporting false success ("Done!") | Independent `JudgeAgent` anti-self-attestation: Judge inspects screen deltas and workspace artifacts; missing, inconclusive, or falsified verdicts downgrade worker to `FAILED`. |
| **P7** | **Worker failures reach the supervising agent** | Boss unaware that sub-mission crashed | Worker exceptions and timeouts wrapped in `SubMissionResult(status="FAILED", success=False)`, recorded in `TaskGraph` as `FAILED`, triggering Boss replan or escalation. |
| **P8** | **Cognitive state reflects actual execution state** | Memory / findings out of sync with reality | Boss `thinking_log`, `all_traces`, and phase summaries derived strictly from verified execution traces, rejecting empty/hallucinated worker claims. |
| **P9** | **Security and safety invariants remain enforced** | Worker breaking containment or mutating policy | `ActionPolicy` workspace boundary enforcement (blocking `../../` path traversal), blocked shell command classification, and `SealedActionPolicy` immutability under runtime tamper attempts. |

---

## 3. Four-Tier Test Architecture

The suite in `sonic-core/tests/test_e2e_autonomous_loop.py` is partitioned into 4 distinct classes matching the tiered testing methodology:

```
sonic-core/tests/test_e2e_autonomous_loop.py
├── TestTier1FeatureCoverage              (13 tests)  -> Point-by-point primary behavior coverage
├── TestTier2BoundaryAndCornerCases       (10 tests)  -> Boundary conditions, malformed inputs, edge cases
├── TestTier3CrossFeatureCombinations     (5 tests)   -> Multi-component interactions (Boss + Worker + Judge)
└── TestTier4RealWorldWorkloadScenarios   (3 tests)   -> End-to-end multi-step dual-plane autonomous missions
```

### Tier 1: Feature Coverage (Primary Happy/Unhappy Paths)
Tests each component of the autonomous loop in isolation:
- `test_tier1_point1_lease_timeout_prevents_indefinite_hang`: Lease acquisition timeout raises `TimeoutError` cleanly when display lease is held.
- `test_tier1_point1_watchdog_aborts_on_consecutive_unchanged_screens`: Screen watchdog detects identical screen hashes across steps and aborts runaway loops.
- `test_tier1_point2_out_of_bounds_coordinates_blocked_before_dispatch`: Negative coordinates and coordinates beyond screen resolution are rejected with `ActionExecutionStatus.FAILED`.
- `test_tier1_point2_stale_screen_observation_rejected_before_dispatch`: Action dispatched against an older screen hash is rejected as stale before reaching motor actuators.
- `test_tier1_point3_focus_validation_enforced_before_window_action`: Verifies active window focus validation and automatic focus-before-click mechanics.
- `test_tier1_point3_typing_focus_and_text_validation`: Verifies `GUI_TYPE` validates non-empty text and focuses target coordinates prior to typing.
- `test_tier1_point4_terminal_action_produces_observable_receipt`: `TERMINAL_EXEC` produces observable execution receipt with exit code and stdout.
- `test_tier1_point4_gui_action_produces_screen_observation_receipt`: `GUI_CLICK` produces observable trace with before/after screen hash updates.
- `test_tier1_point5_perception_failure_propagates_without_blind_dispatch`: Computer perception failure raises runtime exception, preventing blind execution.
- `test_tier1_point6_verification_failures_not_swallowed_by_judge`: Judge evaluates worker claims against visual evidence and downgrades falsified claims.
- `test_tier1_point7_worker_failure_propagates_to_boss_and_task_graph`: Worker failure marks sub-mission `FAILED` and updates TaskGraph state.
- `test_tier1_point8_cognitive_state_reflects_actual_execution_traces`: Sub-agent execution traces accurately summarize completed actions into cognitive memory.
- `test_tier1_point9_safety_action_policy_blocks_path_traversal`: `ActionPolicy` blocks `FILE_READ` attempting directory traversal outside workspace root.

### Tier 2: Boundary & Corner Cases
Exercises extreme limits, degenerate inputs, and edge scenarios:
- `test_tier2_point2_negative_and_zero_coordinates_handling`: Negative coordinates rejected; (0,0) valid origin accepted.
- `test_tier2_point2_exact_screen_boundary_coordinates_rejected`: Exact screen resolution boundaries ($x = 1920$, $y = 1080$) properly rejected (0-indexed boundary).
- `test_tier2_point2_malformed_and_none_coordinates_rejected`: Non-integer strings, None, and non-numeric coordinate dictionaries rejected.
- `test_tier2_point1_fractional_lease_timeout_fails_cleanly`: Microsecond lease timeouts (`timeout_seconds=0.001`) fail without resource leaks or lock corruption.
- `test_tier2_point5_empty_perception_screenshot_fails_validation`: Empty screenshot bytes (`b""`) rejected by screen validation.
- `test_tier2_point5_infinite_and_negative_capture_age_rejected`: Stale capture validation rejects timestamps with infinite or negative age.
- `test_tier2_point6_anti_self_attestation_blocks_unsupported_worker_claims`: Unverified worker claim falsified by Judge when expected visual pattern is absent.
- `test_tier2_point6_unsaved_buffer_detected_and_falsified_by_judge`: Judge detects dirty unsaved editor buffers in visual perception and falsifies claim of file saved.
- `test_tier2_point8_empty_objective_creates_no_workers_and_is_incomplete`: Blank/empty objective creates 0 sub-missions and reports `INCOMPLETE`.
- `test_tier2_point9_sealed_safety_policy_immutable_under_mutation_attempts`: Attempting to alter rules on `SealedActionPolicy` raises `RuntimeError` or `TypeError`.

### Tier 3: Cross-Feature Combinations
Tests multi-component interactions across Boss, Worker, and Judge:
- `test_tier3_autonomous_loop_delegation_execution_judge_confirmation`: Full loop cycle: Boss plans sub-mission $\to$ Worker executes dual-plane actions $\to$ Judge confirms visual evidence $\to$ Boss marks `COMPLETED`.
- `test_tier3_autonomous_loop_fails_closed_when_judge_falsifies_worker`: Boss plans sub-mission $\to$ Worker claims success $\to$ Judge falsifies claim $\to$ Boss marks sub-mission `FAILED` and fails closed.
- `test_tier3_delegated_worker_stale_coordinates_blocks_before_dispatch`: Boss delegates GUI task $\to$ Worker screen invalidates between reasoning and dispatch $\to$ Pre-dispatch check fails closed.
- `test_tier3_worker_timeout_during_gui_delegation_handled_by_boss`: Worker times out while holding display lease $\to$ Lease is released cleanly $\to$ Boss handles timeout in phase summary without hanging.
- `test_tier3_parallel_workers_serialize_display_leases`: Two concurrent workers requesting display lease are serialized cleanly without deadlocking.

### Tier 4: Real-World Workload Scenarios
Tests end-to-end multi-step missions:
- `test_tier4_complete_multi_step_dual_plane_autonomous_workflow`: Complete 3-step mission:
  1. Terminal plane: scaffold test project directory and source file.
  2. Workstation plane: launch application and interact via GUI clicks.
  3. Terminal plane: run test suite and verify green results.
- `test_tier4_autonomous_loop_failure_recovery_and_replan`: Adaptive multi-phase recovery:
  1. Phase 1: Discovery sub-mission completes with verified evidence.
  2. Boss completion check recognizes objective not yet fully complete.
  3. Boss plans Phase 2: executes service verification sub-mission with verified confirmation $\to$ final report `COMPLETE`.
- `test_tier4_safety_envelope_contained_autonomous_loop`: Full mission executing within sealed safety envelope where unauthorized egress/traversal actions are blocked while legitimate commands execute successfully.

---

## 4. Test Infrastructure Components & Mock Harness

### 4.1 MockE2EComputer
A hermetic, deterministic test double simulating both workstation graphical desktop and headless execution planes without requiring a live Docker daemon or Daytona cloud sandbox:

```python
class MockE2EComputer:
    """Hermetic test computer supporting dual-plane operations."""
    - Screen Resolution: 1920 x 1080
    - Multi-plane support: Headless terminal + Visible workstation GUI
    - Dynamic frame generation: Returns valid PNG screenshot bytes and auto-increments
      screen frame hashes upon simulated motor interactions
    - Motor actuation recording: Tracks dispatch history for GUI_MOVE, GUI_CLICK,
      GUI_TYPE, GUI_DRAG, APP_LAUNCH
    - Active window focus: Tracks focused window ID and title; enforces window activation
    - Terminal execution engine: In-memory virtual file tree and shell execution
    - Display lease integration: Compatible with DisplayLeaseManager
```

### 4.2 Cross-Test Async Event Loop Isolation
In Python 3.14+, `asyncio.Lock` instances bound to one event loop raise `RuntimeError: Lock is bound to a different event loop` when accessed from another test function. The test infrastructure provides an autouse fixture resetting `DisplayLeaseManager._locks` and `DisplayLeaseManager._lock_metadata` before and after every test:

```python
@pytest.fixture(autouse=True)
def clean_display_lease_manager():
    DisplayLeaseManager.reset()
    yield
    DisplayLeaseManager.reset()
```

### 4.3 Deterministic LLM Mocking
To test orchestration logic without non-deterministic external LLM APIs, `_make_llm_mock(responses)` provides a sequential scripted response router returning valid JSON responses matching Pydantic schemas (`Phase`, `BossThinking`, `SubMissionResult`, `CompletionCheck`).

---

## 5. Execution & Verification Guide

### Command
```powershell
python -m pytest sonic-core/tests/test_e2e_autonomous_loop.py -v
```

### Environment Requirements
- **No live Docker daemon required**: Uses `@pytest.mark.no_live_infra` marker.
- **Python version**: Python 3.12+ (tested and verified on Python 3.14.7 on Windows 11).
- **Dependencies**: `pytest`, `pytest-asyncio`, `pydantic`, `structlog`.
- **Execution Time**: ~14 seconds for all 31 tests.
