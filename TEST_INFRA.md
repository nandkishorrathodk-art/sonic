# Test Infrastructure Specification: Dual-Plane Computer Operator Hardening (R1 - R4)

**Author**: `teamwork_preview_test_writer` (E2E Test Suite Author)  
**Date**: 2026-09-23  
**Working Directory**: `c:\Users\nandk\_society\.agents\teamwork\test_writer_e2e`  
**Test Suite Path**: `sonic-core/tests/test_dual_plane_e2e.py`  
**Authoritative Architectural Source**: `c:\Users\nandk\_society\AGENTS.md`  
**Reference Document**: `c:\Users\nandk\_society\.agents\teamwork\ORIGINAL_REQUEST.md`  
**Project Document**: `c:\Users\nandk\_society\.agents\teamwork\orchestrator\PROJECT.md`  
**Methodology**: 4-Tier Testing Hierarchy with Zero Mock Gates & Zero Puppet Heuristics  

---

## 1. Testing Philosophy & Architectural Mandates

The test infrastructure for SONIC's Dual-Plane Computer Operator execution engine enforces the architectural principles defined in `AGENTS.md`:

1. **Dual-Plane Concurrency**: SONIC operates concurrently across two clean execution planes:
   - **Workstation Application Plane (`ExecutionSubstrate.COMPUTER`)**: Manages the full graphical desktop environment (X11 display `:99`) for GUI applications (`GUI_*`, `APP_*`, interactive `BROWSER_*`). The GUI is NEVER disabled.
   - **Operator & Sandbox Plane (`ExecutionSubstrate.HEADLESS`)**: Dedicated non-graphical execution plane for high-throughput commands, security scans, custom probes, and git operations (`TERMINAL_EXEC`, `FILE_*`, `GIT_*`, `SECURITY_TOOL`) routing directly through `CapabilityRouter` to headless substrates without cluttering the desktop GUI.
2. **Zero Mock Gates**: No facade tests that return unconditional `True` or simulate success without exercising real logic. All tests must validate actual state transitions, exit codes, receipts, and contract checks.
3. **Zero Hardcoded Application Names (Rule 1)**: Tests must never assume or hardcode specific application or browser binary names (no "chrome", "chromium", "firefox", "burp", "vscode"). All selectors, roles, and discovery structures are generic POSIX / X11 / DOM structures.
4. **Zero Scripted Puppet Heuristics (Rule 2)**: Rigid keyword-matching trees, canned 3-step checklists, prompt keyword-matching classifiers (e.g. `if "app" in goal`), hardcoded refusal fallbacks, and regex string manipulation of model intent are strictly prohibited. The agent must reason freely and adaptively from live observations and typed `ActionContract`s.
5. **Zero External/Mock Domains (Rule 3)**: Tests must NEVER use external domains (`example.com`, `google.com`, `httpbin.org`, `target.com`). All URLs must use local loopback (`127.0.0.1`) or RFC 5737 test documentation network addresses (`198.51.100.1`, `203.0.113.1`, `192.0.2.1`).
6. **Zero Siloing or Disabling of Planes (Rule 5)**: Turning off GUI (`observe_desktop=False`) or turning off terminal (`gui_only=True`) is strictly forbidden. Both planes must be active concurrently at all times. Headless tasks must skip redundant screenshot polling without disabling the GUI capability.
7. **Zero Silent Failure Masking (Rule 6)**: Shell failure masking idioms like `|| true`, `2>/dev/null || true`, or treating blocked executions (exit code 126) as successes are strictly forbidden. All execution receipts must reflect ground-truth exit codes and stderr.
8. **Execution Plane Purity & Multi-Phase Persistence (Rule 8)**: When a mission requests an execution plane (`requested_plane`), that plane must persist throughout all subsequent decomposition phases (`_plan_next_phase`). Headless tasks must not be contaminated by redundant GUI screenshot polling.

---

## 2. 4-Tier Test Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               Tier 4: Real-World Workload Scenarios                    │
│  - Scenario A: "Audit web server source code" remains pure headless   │
│  - Scenario B: Multi-phase penetration assessment retains GUI plane   │
│  - Zero desktop contamination in terminal tasks                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│               Tier 3: Cross-Feature Combinations                       │
│  - BossAgent multi-phase decomposition -> SubAgent plane constraints   │
│  - Workstation route with ActionContract -> CapabilityRouter routing   │
│  - ActionPolicy pre-execution block intercepts before provider compute │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│               Tier 2: Boundary, Corner & Adversarial Cases             │
│  - Empty/unspecified/invalid plane strings fall back safely           │
│  - Strict exit code discrimination: 0 (SUCCESS), 126 (BLOCKED), 1/2/128│
│  - Branch name sanity: illegal flags, path traversal, shell injection  │
│  - Git commit message adversarial payloads: metachars, newlines, sizes │
│  - Workspace path confinement and escape prevention                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│               Tier 1: Feature Coverage (R1 - R4)                       │
│  - F1: SubMission explicit plane preservation & inference fix          │
│  - F2: BossAgent multi-phase plane propagation in reaction planning    │
│  - F3: Explicit plane constraints in sub-agent dispatch                │
│  - F4: Symmetric plane boundary enforcement (fail-closed exit 126)     │
│  - F5: CapabilityRouter wiring into ComputerUseAgent                   │
│  - F6: Headless GUI observation decoupling without siloing             │
│  - F7: Objective-driven plane resolution (zero keyword traps)          │
│  - F8: seal_default runtime parameters compatibility                   │
│  - F9: Provider git error unmasking (no || true, exit 126 = BLOCKED)   │
│  - F10: Honest Git ActionReceipt generation (BLOCKED / FAILED)         │
│  - F11: Git commit safety policy validation in ActionPolicy            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tier 1: Feature Coverage Matrix (R1 - R4)

| Feature | Requirement | Test Function | Description & Invariants |
|---|---|---|---|
| **F1** | R1 | `test_tier1_submission_explicit_plane_preservation` | Asserts `SubMission(plane="gui")` preserves explicit plane and does not get overridden by goal inference. Asserts generic browser detection without hardcoded browser names ("chrome", "firefox"). |
| **F2** | R1 | `test_tier1_boss_multi_phase_preserves_gui_plane` | Asserts `BossAgent._plan_next_phase()` propagates `requested_plane="gui"` across reaction planning, ensuring subsequent phase sub-missions retain `plane="gui"`. |
| **F3 / F4** | R1 | `test_tier1_symmetric_boundary_headless_rejects_gui_actions` | Asserts `ComputerUseAgent(execution_plane="headless")` rejects inadvertent GUI actions (`GUI_CLICK`, `APP_LAUNCH`, `BROWSER_CLICK`) fail-closed with `status=BLOCKED` and `exit_code=126`. |
| **F4** | R1 | `test_tier1_gui_agent_serializes_on_display_99` | Asserts GUI plane operations execute targeting display `:99` with single-worker concurrency limit on shared workstations. |
| **F5** | R1, R2 | `test_tier1_capability_router_substrate_routing` | Asserts `CapabilityRouter.route_action` classifies `GUI_*`, `APP_*`, and interactive `BROWSER_*` as `COMPUTER`, and `TERMINAL_EXEC`, `FILE_*`, `GIT_*`, `SECURITY_TOOL` as `HEADLESS`. |
| **F5** | R2 | `test_tier1_agent_execute_action_routes_via_capability_router` | Asserts `ComputerUseAgent.execute_action()` delegates substrate routing to `CapabilityRouter.route_action()` rather than ad-hoc checks. |
| **F6** | R2 | `test_tier1_headless_observation_decoupled_from_gui_screenshot` | Asserts `ComputerUseAgent.observe()` bypasses full-frame screenshot captures during headless tasks (`desktop_state="HEADLESS_DECOUPLED"`), while keeping `observe_desktop=True` (Rule 5 invariant). |
| **F7** | R3 | `test_tier1_workstation_objective_driven_plane_resolution` | Asserts `_requires_desktop_observation()` does not falsely classify "audit web server source code" or "inspect application directory" as GUI based on substring matches. |
| **F8** | R3 | `test_tier1_seal_default_accepts_runtime_parameters` | Asserts `seal_default()` accepts `require_approval_for_intrusive` and `max_actions_per_minute` kwargs without raising `TypeError`. |
| **F9** | R4 | `test_tier1_provider_git_action_unmasked_exit_126_and_nonzero` | Asserts `UnifiedComputerProvider.git_action()` removes `\|\| true`, returns `ok=False` for exit 126, records `result="BLOCKED"`, and records `result="FAILED"` for non-zero exit codes. |
| **F10** | R4 | `test_tier1_agent_git_action_receipt_honest_status` | Asserts `ComputerUseAgent.execute_action()` generates an `ActionReceipt` with `status=BLOCKED` (exit 126) or `status=FAILED` (non-zero) instead of false `COMPLETED`. |
| **F11** | R4 | `test_tier1_action_policy_git_commit_validation` | Asserts `ActionPolicy.evaluate("GIT_COMMIT", ...)` validates commit message safety, path workspace confinement, and branch ref sanity. |

---

## 4. Tier 2: Boundary & Corner Cases Matrix

| Boundary Category | Test Function | Inputs / Vectors | Expected Behavior |
|---|---|---|---|
| **Unspecified Planes** | `test_tier2_empty_or_unspecified_plane_defaults_gracefully` | `plane=None`, `plane=""`, `plane="invalid"` | Defaults safely to `"headless"` or infers from goal without unhandled exceptions. |
| **Exit Code Discrimination** | `test_tier2_git_exit_codes_distinction` | Exit codes 0, 126, 1, 2, 128 | Strict mapping: 0 -> `ok=True`, 126 -> `ok=False (BLOCKED)`, 1/2/128 -> `ok=False (FAILED)`. Zero exit code coercion. |
| **Branch Name Sanity** | `test_tier2_git_branch_name_sanitization_and_boundaries` | Legal: `main`, `feat/login`, `fix_42`<br>Illegal: `-rf`, `../heads/main`, `feat; rm`, `feat$(id)`, `feat\n` | Legal branches allowed; flag injections, path traversal, control chars, and shell metacharacters denied with descriptive verdict. |
| **Commit Injection Payloads** | `test_tier2_git_commit_injection_payloads` | `'; rm -rf /; '`, `$(cat /etc/passwd)`, `` `whoami` ``, `msg\nmalicious`, `\0`, `""`, `len > 500` | All malicious, empty, or oversized commit messages denied by `ActionPolicy`. |
| **Workspace Confinement** | `test_tier2_git_commit_path_traversal_and_confinement` | `/home/sonic/workspace/app`, `../../etc`, `/etc/shadow`, `../../outside` | In-workspace path allowed; path traversal and external paths denied fail-closed. |

---

## 5. Tier 3: Cross-Feature Combinations Matrix

| Interaction | Test Function | Components Involved | Invariant Verified |
|---|---|---|---|
| **Multi-Phase SubAgent Constraints** | `test_tier3_multiphase_gui_mission_propagates_subagent_constraints` | `BossAgent` ↔ `ComputerUseAgent` | Boss decomposes GUI mission (`requested_plane="gui"`). Sub-agent receives `execution_plane="gui"`. Next phase preserves plane and dispatches with desktop concurrency limit (1 worker on display `:99`). |
| **Contract-Driven Capability Routing** | `test_tier3_workstation_contract_triggering_capability_router` | `workstation.py` ↔ `CapabilityRouter` ↔ `ActionContract` | Workstation route accepts typed `ActionContract`. `TERMINAL_EXEC` routes to `HEADLESS` (`_requires_desktop_observation -> False`); `GUI_CLICK` routes to `COMPUTER` (`_requires_desktop_observation -> True`). |
| **Pre-Execution Policy Interception** | `test_tier3_git_commit_safety_blocks_before_provider_execution` | `ComputerUseAgent` ↔ `ActionPolicy` ↔ `UnifiedComputerProvider` | `ActionPolicy` intercepts malicious `GIT_COMMIT` *before* provider execution. Provider `compute.execute` is NEVER called. Trace records `status=BLOCKED`, `exit_code=126`. |

---

## 6. Tier 4: Real-World Workload Scenarios Matrix

| Workload Scenario | Test Function | Sequence & Verification Flow |
|---|---|---|
| **Scenario A: Headless Server Code Audit** | `test_tier4_workload_audit_web_server_remains_pure_headless` | 1. Prompt: "Audit web server source code in /home/sonic/workspace/app and test endpoints"<br>2. Workstation classifier returns `False` (zero keyword traps for "web" or "app")<br>3. SubMission inferred as `"headless"`<br>4. ComputerUseAgent `observe()` skips full-frame screenshot polling<br>5. Dispatched actions route to `HEADLESS` substrate<br>6. GUI desktop plane remains unpolluted and available. |
| **Scenario B: Multi-Phase Penetration Assessment** | `test_tier4_workload_multiphase_penetration_assessment_retains_gui` | 1. Mission requested with `requested_plane="gui"`<br>2. Phase 1: Interactive reconnaissance on Web UI -> SubMission `plane="gui"`<br>3. Phase 1 captures screen and interacts with controls on display `:99`<br>4. Phase 1 findings feed into reaction planning<br>5. Phase 2 reaction planning preserves `requested_plane="gui"`<br>6. Phase 2 sub-mission executes on GUI plane on display `:99`<br>7. Full assessment report reflects end-to-end GUI evidence without plane degradation. |

---

## 7. Test Execution & Verification Protocol

### Test Execution Command
```bash
python -m pytest sonic-core/tests/test_dual_plane_e2e.py -v
```

### Combined Regression & Hardening Verification Command
```bash
python -m pytest sonic-core/tests/test_boss_agent.py sonic-core/tests/test_headless_and_capability_router.py sonic-core/tests/test_action_contracts.py sonic-core/tests/test_dual_plane_e2e.py -v
```

### Environment Opt-Out Marker
All tests in `sonic-core/tests/test_dual_plane_e2e.py` are decorated with `pytestmark = [pytest.mark.no_live_infra]`, ensuring they exercise the state, safety, routing, and contract logic cleanly without requiring a live Docker daemon or cloud Daytona sandbox.

