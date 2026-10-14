# AVCS Suite A — deterministic Windows GUI fixtures

**Status:** starter benchmark kit, version `0.1.0`. This is not yet wired to the private AVCS repository or a verified AVCS SDK adapter.

Suite A gives AVCS a deterministic Windows GUI target set, an evaluator-only state oracle, a benchmark adapter contract, and a JSON report. The UI is a real Tk window so it can be operated by the Windows desktop runtime; the expected state and target bounding boxes are retrieved separately by the runner.

## What is included

- 16 deterministic cases covering buttons, similar labels, checkbox disambiguation, dropdown options, form entry, confirmation/cancel dialogs, duplicate labels in context, disabled controls, overlays, delayed state changes, reflow, and false-success checks.
- A Windows GUI fixture app (`avcs_suite_a.fixture_app`).
- An authenticated loopback-only oracle/reset API.
- A project-specific adapter interface (`AgentAdapter`) that calls the real AVCS SDK.
- Independent end-state scoring, IoU-based target-box scoring, task-level false-confirmation scoring, timing summaries and a JSON trace.
- Unit tests that run cross-platform without starting a GUI.

## Important benchmark boundary

The fixture oracle is independent of the AVCS agent: the adapter receives only the visible task and timeout, never case IDs, target IDs, expected state, ground-truth rectangles or the oracle token. The benchmark is designed to catch accidental false confirmations and mis-targets from ordinary adapters; it is not a hardened security sandbox against a malicious local process running as the same Windows user. For a hostile-agent evaluation, put the evaluator/oracle in a separate account or VM and use a controlled observation/action bridge.

Target precision is computed from the first action trace entry containing `target_bbox`, using IoU >= 0.5 against the expected target's pre-action screen rectangle. This score is only valid if the AVCS adapter reports real target-selection telemetry; fabricated or missing telemetry must not be represented as a measured target precision. Cases where a target is covered or dynamically changes geometry need per-action ground-truth snapshots before being used for a formal target-precision headline.

`false_confirmation_rate` in this starter kit is **task-level**: reported `confirmed` results where the oracle says the task's expected state was not reached, divided by all task-level confirmations. It is not the action-level false-success metric for 3,000 individual actions. That requires an action trace with independently evaluable postconditions for each action.

## Install / test

Python 3.10+ is recommended. The unit tests use only the standard library and pytest.

```powershell
python -m pip install pytest
python -m pytest -q
```

The fixture window and real end-to-end runs require an interactive Windows session (not a Windows service or headless CI worker). Run them from the same desktop session that AVCS will control.

## Wire the actual AVCS SDK

Create a module in the AVCS repository, e.g. `benchmarks/avcs_suite_adapter.py`, and implement:

```python
class MyAVCSAdapter:
    def __init__(self):
        # Construct the actual AVCS SDK client and connect to its user-session broker.
        ...

    def execute(self, task: str, timeout_s: float) -> dict:
        # Use the real AVCS system to perform the task against the visible window.
        # Do not import CASES, query /oracle, or use expected labels/bounds.
        # Return status plus the SDK's real trace data.
        return {
            "status": "confirmed",  # confirmed | failed | ambiguous | blocked
            "duration_ms": 123.4,
            "actions": [
                {
                    "status": "CONFIRMED",
                    "target_bbox": {"x": 400, "y": 300, "width": 140, "height": 40},
                }
            ],
        }


def create_adapter():
    return MyAVCSAdapter()
```

Run:

```powershell
$env:AVCS_SUITE_ADAPTER = "benchmarks.avcs_suite_adapter:create_adapter"
python -m avcs_suite_a.runner --output benchmarks/results/suite_a.json
```

The adapter must be implemented inside the actual AVCS repo and imported via `PYTHONPATH`. Its `status` is the agent's own report; the oracle, not that status, determines whether the expected UI state was reached. The example result above is illustrative only and must not be returned by a real adapter without actual execution and evidence.

Run a subset:

```powershell
python -m avcs_suite_a.runner --case button_exact --case similar_button_labels --output suite_a_smoke.json
```

## Current limitations / integration work required

1. The package is not modified into the AVCS repo because the repo referenced by `file:///C:/...` is not mounted in this session. Merge the package or provide repository access and map the adapter to the real SDK methods.
2. The `overlay_close` case is a visual-overlay fixture; Windows UI Automation may still expose underlying controls, so the broker must verify occlusion and active target visibility before acting.
3. The target-box scorer initially evaluates the first reported target against the pre-action target. Dynamic multi-step flows need timestamped per-action oracle snapshots to claim comprehensive target precision.
4. Target labels are not supplied to the adapter, but this local fixture app is not a security sandbox. Use process/account isolation for adversarial-agent benchmarks.
5. DPI, mixed-monitor geometry, negative virtual coordinates, lock/unlock, and handle leaks remain separate Windows integration/soak tests. The fixture cases do not by themselves establish those guarantees.
6. `no_false_success` has a visible refresh request that never sets `preview_refreshed`. The task expects the adapter to return `ambiguous` after the click, not `confirmed`. The fixture oracle records both the request and the still-incomplete preview state.

## Exit criteria for formal Suite A reporting

- The AVCS adapter uses the real capture, grounding, broker and verification components.
- A fixed commit and environment are recorded.
- At least five repeated runs are collected and raw JSON traces retained.
- Every fixture has a resettable start state and an evaluator-owned postcondition.
- Target precision and task false-confirmation metrics clearly show their denominators and telemetry coverage.
- No headline benchmark mixes synthetic fixture results with OSWorld or real-customer workflow results.
