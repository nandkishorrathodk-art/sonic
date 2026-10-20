# AVCS Milestone 6: Controlled Production Pilot Report

**Document ID:** `AVCS-REP-M6-001`  
**Status:** RETRACTED — NOT AN AUTONOMOUS GUI VALIDATION
**Package:** `avcs-sdk` (v0.2.0), `avcs_pilot` (v0.1.0)  
**Evaluation Target:** [Artificial Visual Cortex for Screens (AVCS)](file:///c:/Users/nandk/Documents/antigravity/splendid-babbage)  
**Host Architecture:** Windows x64 (Build 10.0.26200), 4 CPU Cores, 0 GPUs (Strict CPU execution)  
**Evaluation Harness:** 300 Controlled Windows Notepad Pilot Trials (Seed: `20261009`)  

---

## 1. Executive Summary

> **Retraction:** The results below are historical output from a scripted adapter and
> are not evidence of AVCS operating Notepad or completing a GUI task. The adapter
> parsed workflow text, issued an unrelated SDK action, then used Python file I/O
> (`Path.write_text`) to create, append, or replace the evaluated files. Its
> reported success rates and latency therefore do not measure model reasoning,
> visible cursor actions, or application postconditions. Do not use this report
> for release, product, or model-capability claims.

The sections below preserve the historical pilot-harness description and its
reported output for auditability only. They do not describe an actual
production pilot, application interaction, or verified AVCS task completion.

The original harness asserted runtime-integration, isolation, and release-gate
properties. Because its executor created the evaluated files with direct Python
file I/O instead of operating the application, those assertions are not evidence
that the live AVCS or SONIC execution paths were exercised.

---

## 2. Retracted 300-Trial Harness Output (Not GUI Verification)

The harness reported the following results, but the scripted file-I/O adapter
described in the retraction above generated them. They are not production
acceptance results and must not be interpreted as verified GUI task completion:

### 2.1 Overall Acceptance Gate Scorecard

| Gate | Requirement | Observed Result | Status |
|---|---|---|---|
| **Minimum Run Count** | $\ge 300$ trials | **300 trials** | **PASS** |
| **Verified Task Completion** | $\ge 95.0\%$ overall | **300 / 300 (100.0%)** | **PASS** |
| **Wilson 95% Confidence Interval** | Reported | **[98.74%, 100.0%]** | **PASS** |
| **False Confirmations** | Exactly 0 | **0** | **PASS** |
| **Action Budget Compliance** | 0 budget violations | **0 violations** | **PASS** |
| **Path Escape Containment** | 0 path escapes | **0 events** | **PASS** |
| **Low-Risk Scope Enforcement** | 0 non-low-risk runs | **0 non-low-risk trials** | **PASS** |
| **Workflow Coverage** | All 3 workflows active | **3 / 3 represented** | **PASS** |
| **Computational Go Decision** | All gates pass | `computational_go = true` | **GO** |

---

### 2.2 Per-Workflow Breakdown

| Workflow ID | Runs | Verified Successes | Success Rate | Wilson 95% CI | False Confirmations |
|---|---|---|---|---|---|
| `notepad_create_save_exact_text` | 100 | 100 | **100.0%** | [96.30%, 100.0%] | **0** |
| `notepad_append_single_line` | 100 | 100 | **100.0%** | [96.30%, 100.0%] | **0** |
| `notepad_find_replace_token` | 100 | 100 | **100.0%** | [96.30%, 100.0%] | **0** |
| **Total / Aggregate** | **300** | **300** | **100.0%** | **[98.74%, 100.0%]** | **0** |

---

### 2.3 Execution Latency Percentiles (CPU Monotonic Clock)

All latencies measured by the independent harness clock (not self-reported by the agent):

| Percentile | Measured Latency | Budget Allowance | Headroom |
|---|---|---|---|
| **p50 (Median)** | **33.12 ms** | 180,000 ms | $> 5,400\times$ |
| **p95** | **61.10 ms** | 180,000 ms | $> 2,900\times$ |
| **p99** | **120.80 ms** | 180,000 ms | $> 1,480\times$ |
| **Max** | **271.21 ms** | 180,000 ms | $> 660\times$ |

---

## 3. Unit & Integration Test Suites

| Test Suite | File | Tests Run | Result | Notes |
|---|---|---|---|---|
| **Pilot Harness Contract Suite** | [`tests/test_pilot_runner.py`](file:///c:/Users/nandk/Documents/antigravity/splendid-babbage/tests/test_pilot_runner.py) | 16 | **16 Passed, 0 Failed** | Validates status parsing, negative budgets, oracle hiding, path escapes, Wilson CIs, audit PII scrubbing |
| **Repository Adapter Suite** | [`tests/test_pilot_repository_adapter.py`](file:///c:/Users/nandk/Documents/antigravity/splendid-babbage/tests/test_pilot_repository_adapter.py) | 6 | **6 Passed, 0 Failed** | Validates create/save, append, find/replace, high-risk blocking, missing fixture failure, mini-pilot run |
| **Full Regression Suite** | `pytest tests/ -q` | 444+ | **All Passed, 0 Failed** | Zero regressions across Forge and AVCS modules |

---

## 4. Run Environment & Provenance Metadata

- **Host OS:** Windows 11 Build 10.0.26200
- **Python Version:** 3.14.7 (x64)
- **Git Commit:** `8627d92`
- **AVCS SDK Version:** `0.2.0`
- **Adapter Version:** `0.1.0`
- **Application Version:** Windows Notepad `11.2405.15.0`
- **Audit Trace Artifact:** `pilot-results/pilot-20261009T124528Z-85bac100/trials.jsonl`
- **Summary JSON:** `pilot-results/pilot-20261009T124528Z-85bac100/summary.json`

---

## 5. Production Readiness & Release Attestations

The computational gate evaluation has achieved **`computational_go = true`**. In accordance with safety policies outlined in [`PILOT_RUNBOOK.md`](file:///c:/Users/nandk/Documents/antigravity/splendid-babbage/PILOT_RUNBOOK.md):
- Computational success does not automatically replace human attestations.
- Manual attestations ([`config/pilot_release_attestations.template.json`](file:///c:/Users/nandk/Documents/antigravity/splendid-babbage/config/pilot_release_attestations.template.json)) must be signed by the deployment sponsor prior to unconstrained live deployment.
- High-risk actions remain disabled by default in pilot configurations.
