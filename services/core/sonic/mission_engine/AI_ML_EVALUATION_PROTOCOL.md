# SONIC AI/ML baseline and offline evaluation protocol

## Scope and authority

This protocol defines Evaluation & Reliability's baseline for the Cognitive
Systems `sonic.training-example/v1` contract. It is offline and observational:
it does not collect or persist examples, call a model, execute candidate actions,
modify routing, train weights, or promote candidates.

The typed input contract is in
[`../llm/trajectory.py`](../llm/trajectory.py); the current pure baseline
aggregator is `trajectory_evaluation.evaluate_training_examples`.

## Baseline metrics

Report each metric with its numerator/count and denominator. A missing
denominator or measurement is **unavailable**, never a synthetic zero or a
successful score.

| Metric | Definition | Unit |
|---|---|---|
| Action outcome distribution | Counts of receipt outcomes, reported separately | Decision |
| Action success rate | `SUCCEEDED` action receipts / all decisions | Decision |
| Verified mission success rate | Unique episodes with `SUCCESS` and independent passed verification / unique episodes | Episode |
| Independent verification coverage | Unique episodes with an independent verification attempt and evidence / unique episodes | Episode |
| Correction rate | Decisions with at least one linked correction / all decisions | Decision |
| Policy block rate | `BLOCK` or `REQUIRE_APPROVAL` verdicts / all decisions | Decision |
| Prediction error | Mean of available, bounded receipt prediction errors | Measured decision |
| Model latency and cost | Mean of available measured latency/cost values, with measurement count | Measured decision |

Mission-level rates are episode-weighted, not action-weighted. A dataset with
conflicting final mission outcomes for one episode is rejected. Report
`RECOVERED`, `FAILED`, `BLOCKED`, and `TIMED_OUT` separately; do not relabel
them as successes. These are descriptive baselines, not causal evidence that a
model or policy improved.

Every reported rate includes its numerator, denominator, and two-sided **95%
Wilson score interval**. Numeric means include their measurement count;
uncertainty intervals for means are not yet implemented, so they cannot be used
alone as promotion evidence.

## Frozen dataset eligibility gate

The first approved collection/evaluation cohort has these minimums:

- At least **300 unique episodes** with validated `TrainingExample/v1` records.
- Tenant-grouped split floors: **150 train / 75 validation / 75 holdout**
  unique episodes.
- At least **3 distinct tenant groups**. Each HMAC-pseudonymized tenant is
  assigned wholly to exactly one split; all decisions in an episode stay
  together. If tenant group sizes make any split floor impossible, reject the
  freeze and collect more diverse data rather than split a tenant.
- Episode final outcome is consistent across all decision rows.
- Every artifact reference is covered by the example's redaction-review
  receipt. The trusted privacy gate verifies source digests, inspects and
  redacts referenced content, and signs the receipt. Dataset freeze verifies
  the signature through an injected verifier; it does not receive the signing
  key.
- The HMAC-SHA256 signature covers the UTF-8 canonical JSON serialization of
  the complete review receipt excluding `signature_sha256`, with sorted keys,
  compact separators, and `ensure_ascii=true`. Signing implementations must
  use this exact encoding.
- A successful episode is labeled **only** when the final outcome is
  `SUCCESS` and there is an independent verifier, `PASSED` status, and evidence
  reference. `SUCCEEDED` action receipt is an action-level outcome, not a
  successful mission label. Historical selected action is not ground truth;
  it is at most a weak preference label.

`privacy_gate.TrustedPrivacyRedactionGate` resolves each artifact by opaque ID,
checks its source SHA-256, scans/redacts supported UTF-8 text, and requires
complete OCR inspection for PNG/JPEG screenshots. It returns replacement
artifact bytes in memory, updates the example to refer only to those redacted
bytes, and signs a receipt covering each redacted artifact's output digest and
corresponding verified source digest. Unsupported,
malformed, oversized, digest-mismatched, or incompletely inspected artifacts
are rejected. No artifacts or dataset are persisted by this gate.

The HMAC-SHA256 signing key is obtained through an external key-provider
callback used only by the trusted gate/authenticator. It is not included in
the contract, attestation, returned dataset, or model/training inputs.
`dataset_manifest.freeze_dataset` takes a verifier interface, enforces the
sample/grouping/consistency requirements, and returns a deterministic
in-memory manifest with dataset and split SHA-256 digests. It does not store
the examples or manifest. The verifier checks signature integrity and
artifact-reference coverage; byte inspection/redaction happens only in the
trusted gate before the attestation is issued.

### Promotion/non-regression criterion (future candidates only)

No candidate may be promoted from the descriptive baseline alone. For a future
candidate with paired, independently verified holdout outcomes:

1. Require at least 75 distinct holdout episodes and the frozen split digest.
2. Any confirmed safety-policy violation, cross-tenant leakage, invalid
   redaction attestation, or false success label is an immediate **REJECT**.
3. Compare candidate and baseline on the same holdout episodes. The primary
   metric is verified mission success; compute the 95% confidence interval for
   the paired per-episode success-rate difference by a preregistered,
   episode-level paired bootstrap (10,000 resamples, frozen seed).
4. A non-regression claim requires the difference interval's lower bound to be
   at least zero. A performance-improvement claim requires it to be strictly
   greater than zero. Also require no regression in safety, independent
   verification coverage, and correction rate; publish all denominators and
   intervals.
5. If the available labels are only prior model/operator action choices, report
   decision agreement as weak preference agreement—not task success and not
   promotion evidence.

The paired-bootstrap promotion calculation is a future Evaluation & Reliability
implementation requirement; this change defines the rule but does not execute
candidate comparisons or promote models.

## Metrics not yet supported by this contract

Do not publish estimates for the following until separately labeled
observations and denominators exist:

- model routing accuracy or fallback rate (requires an expected route and
  explicit fallback provenance);
- memory precision/recall, MRR, or nDCG (requires query-to-memory relevance
  judgments);
- visual grounding accuracy or misclick rate (requires intended and actually
  reached UI targets);
- recovery success and correction-round reduction (requires episode-level
  recovery linkage and final verification);
- action confidence calibration (requires candidate probabilities);
- task cost/latency comparisons by model unless request metadata is complete
  and calls are independently attributable.

An absent metric is a data requirement, not a reason to infer a proxy.

## Dataset and split rules (for a future collector)

1. Accept only validated v1 examples; reject unknown fields and contradictory
   decision/receipt links.
2. Require a separate ingestion gate to inspect/redact referenced artifacts.
   `redaction_status` is an attestation field, not proof that the artifact is
   safe. Never place raw credentials, tokens, cookies, unredacted screenshots,
   prompts, or terminal output in the example store.
3. Verify artifact digests and keep an auditable source-to-redacted-artifact
   mapping outside the model-facing dataset.
4. Use keyed tenant pseudonyms. Keep the HMAC key outside the dataset, do not
   publish pseudonyms, and prevent the same episode or tenant from crossing
   training, validation, and holdout splits.
5. Split by whole tenant groups, never individual action rows. Freeze a split
   manifest and dataset digest before candidate evaluation.
6. Preserve failures, blocked decisions, recoveries, and inconclusive
   verification. Exclude neither failures nor difficult examples to improve a
   score.

## Offline replay design

Replay is a **read-only decision comparison**, not environment reproduction:

1. Freeze and hash the reviewed observation/tool-context artifacts referenced by
   each held-out episode. Candidate and baseline receive the exact same
   redacted inputs and action schema.
2. Execute candidate inference only. Do not dispatch selected actions, call
   external targets, mutate desktop state, or allow a model to approve itself.
3. Compare decisions only against labels actually available: the historical
   selected action, a subsequent corrective action, or an independently
   adjudicated label. Mark the label source; historical operator/model choices
   are weak preference labels, not ground truth.
4. Do not treat the original action's receipt as proof that an unexecuted
   counterfactual action would succeed. Counterfactual action value requires a
   separately validated simulator or a controlled, authorized execution study;
   it is not inferred from offline replay.
5. Report decision agreement, correction avoidance, safety-policy compatibility,
   latency, and cost only where their labels and measurements exist. Keep
   results stratified by task, modality, model and dataset split; publish sample
   counts and uncertainty intervals.
6. Re-evaluate on a frozen holdout after any tuning. No online learning,
   automatic routing change, or production promotion follows from this
   descriptive baseline alone.

## Promotion boundary

This protocol does not authorize promotion. Before any candidate is considered,
Evaluation & Reliability must define minimum sample sizes, uncertainty-aware
non-regression criteria, task-specific success labels, and required safety
checks. Security & Safety's deterministic `ActionPolicy` remains authoritative;
no learned score can override it.

## Current evidence status

The repository contains trace-based replay metrics and synthetic/callback-based
benchmarks, but no validated, privacy-reviewed `TrainingExample` collection
was found during the audit. Therefore the present production baseline is
**not yet measurable from this contract**. The 300-episode freeze requirement
is a future cohort-eligibility floor, not evidence that such data exists. An
empty input to the aggregator returns unavailable metric values. This is
intentional and must not be presented as a zero-performance baseline.

The existing cross-layer component regression baseline is documented
separately in [`CORTEX_BASELINE_BENCHMARK.md`](./CORTEX_BASELINE_BENCHMARK.md).
Its fixture-test results must not be presented as production task-success,
grounding-accuracy, or latency metrics.
