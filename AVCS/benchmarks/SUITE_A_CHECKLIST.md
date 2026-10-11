# Suite A integration checklist for the AVCS repository

- [ ] Copy `avcs_suite_a/` into the AVCS repo (or place it beneath `benchmarks/` and adjust import paths).
- [ ] Add the fixture app, catalog, evaluator and tests to CI; run the unit tests on all supported Python versions.
- [ ] Implement `benchmarks/avcs_suite_adapter.py` against the actual `ActionBroker`, capture adapter, target grounding and postcondition verifier in AVCS.
- [ ] Return only actual SDK telemetry in adapter `actions`; do not invent bounding boxes or statuses.
- [ ] Add per-action trace events with action ID, target box, dispatch result, expected postcondition, verification status and timestamps.
- [ ] Refactor target evaluation to snapshot oracle boxes at every action boundary for multi-step/dynamic cases before publishing target precision.
- [ ] Ensure the adapter cannot read the evaluator token, `/oracle`, expected UI state, target labels, or expected target rectangles.
- [ ] Run a Windows interactive-session smoke test for each of the 16 cases.
- [ ] Capture OS build, Python version, DPI scale, monitor geometry, CPU model, AVCS commit, model artifact and capture backend in output metadata.
- [ ] Run at least five full suite repetitions and retain raw JSON traces.
- [ ] Confirm `suite_case_pass_rate` includes the safety/negative controls while `task_completion_rate` uses only positive task cases.
- [ ] Keep long-running resource/handle soak tests and mixed-monitor geometry tests as separate release gates.
