# AVCS: Production Engineering Specification & Roadmap

## 1. Executive Direction & Operating Philosophy

The objective of AVCS is to move from a fast visual-reflex prototype to a reliable, testable, production-grade computer-use runtime.

The core insight is: **Reliability is the first product milestone; speed is the second.**
A visual-reflex loop executing in 8 ms is ineffective if it acts on stale targets, triggers false confirmations, or misclicks non-idempotent enterprise actions. 

AVCS operates as a **hybrid, closed-loop computer interaction runtime**:
1. **Local Vision & Reflex Layer**: Processes high-frequency visual updates, optical flow, and transient UI shifts at 60–120 FPS on CPU.
2. **Hybrid Targeting Layer**: Fuses native Windows UI Automation (structured control patterns, element trees) with pure-pixel visual grounding (for custom canvas, WebGL, games, and unexposed controls).
3. **Action Safety Controller & Broker**: Validates target freshness, active window focus, UIPI integrity boundaries, action idempotency, and postcondition expectations before dispatching input.
4. **Outcome Verifier**: Evaluates semantic postconditions using local photometric deltas as supporting evidence, classifying every action into discrete terminal states: `CONFIRMED`, `FAILED`, `AMBIGUOUS`, or `BLOCKED`.
5. **Provider-Agnostic Cognitive Planner**: Coordinates goals through a provider-neutral interface (Claude, GPT-4o, Gemini, local models), without bypassing the action safety controller.

---

## 2. Evidence Status & Truth Baseline

To maintain absolute scientific and engineering integrity, we separate what has been demonstrated in code from what remains an unverified hypothesis:

| Assertion | Evidence Status | Requirement to Formally Establish |
|---|---|---|
| **Local CPU vision pipeline p50 latency** | Measured locally on test machine | Multi-run automated benchmark exporting raw timing arrays, CPU telemetry, and hardware details |
| **FPS cadence vs latency consistency** | Demonstrated in synthetic benchmark | Full timing breakdown separating serial frame time from overlapped processing |
| **Test suite pass rate** | Verified locally (351+ passing tests) | Clean-environment CI run across multiple Windows builds and architectures |
| **Handle leak elimination** | Validated via explicit `CloseDesktop` calls | 8-hour continuous soak test with Windows Performance Counter tracking |
| **Action outcome verification** | Verified in synthetic tests | 3,000-trial test suite measuring false-positive rate under adversarial and flat UI conditions |
| **98% token reduction** | Modeled mathematically (<50 token diff vs 1,600 token screenshot) | Head-to-head empirical task execution against identical LLM planner with full token logs |
| **Zero-egress privacy** | Architectural design | Explicit data policy enforcement (`local_only`, `redacted_cloud`, `cloud_vision`) with network auditing |

---

## 3. Seven-Milestone Production Roadmap

### Milestone 0: Establish a Reproducible Baseline
- Build instrumented benchmark suite with monotonic timers for every stage (Capture, Preprocessing, Backbone, Affordance, Scene Diff, IPC, Verifier).
- Export machine-readable JSON reports recording p50, p95, p99, max, std, CPU utilization, and hardware configuration.
- Validate serial vs pipelined frame cadence.

### Milestone 1: Stabilize Sensing and Coordinate Geometry
- Formalize `ScreenCaptureAdapter` interface supporting GDI DIBSection, DXGI Desktop Duplication, Windows.Graphics.Capture, and Synthetic Replay.
- Implement explicit multi-monitor topology support, negative virtual-desktop coordinates, and Per-Monitor v2 DPI awareness.
- Add typed capture error states and buffer exhaustion safeguards.

### Milestone 2: Build the Action and Verification State Machine
- Replace open-loop input injection with explicit action lifecycle management: `CONFIRMED`, `FAILED`, `AMBIGUOUS`, `BLOCKED`.
- Classify actions into idempotent, conditionally repeatable, non-idempotent, and high-impact.
- Block blind offset retries on non-idempotent operations.
- Require semantic postcondition verification alongside optical flow and photometric deltas.

### Milestone 3: Create the Public Benchmark Suite
- **Suite A**: Deterministic Windows fixtures (menus, forms, dialogs, drag-and-drop, mixed DPI).
- **Suite B**: External long-horizon benchmarks (OSWorld 2.0 evaluation).
- **Suite C**: No-DOM canvas tasks (Figma, Miro, custom-drawn legacy apps, games).

### Milestone 4: Reliability and Security Hardening
- Adversarial UI, unexpected popups, window focus stealing, and privilege boundary (UIPI) compliance.
- 3,000-trial outcome verification test targeting $\le 0.1\%$ false-success rate.
- 8-hour continuous soak test ensuring zero upward trend in handle count or RSS.

### Milestone 5: Publish a Stable SDK
- Standardized Python package (`pyproject.toml`) installable via `pip install avcs`.
- Provider-neutral client API with versioned observation and action schemas.
- Explicit data privacy modes: `local_only`, `redacted_cloud`, `cloud_vision`.

### Milestone 6: Controlled Production Pilot
- Execute 300 held-out tasks on named Windows applications.
- Require $\ge 95\%$ task completion threshold for general release.
- Audit local logging, secret redaction, and outbound network traffic.

