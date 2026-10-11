# AVCS local demonstration data

`avcs-record-demo` records operator-labeled left-click examples for future
grounding-model training. It does not choose targets, move the cursor, type, or
claim task completion.

## Record a disposable window

1. Open a disposable test application window with a unique title and no private
   content.
2. From an interactive PowerShell terminal, run:

   ```powershell
   avcs-record-demo `
     --window-title "AVCS_DEMO_WINDOW" `
     --task "Complete the harmless test form" `
     --output-dir "runs/demonstrations/form-01" `
     --confirm-isolated-window
   ```

3. Enter a short description of the next control in the terminal, focus the
   named test window, and click that control. Repeat for each demonstrated
   click. Enter `:stop` in the terminal to finish.

The recorder requires an exact, unique, visible window title and a new output
directory. It records only while that window (or an owned modal) is foreground,
stores only cropped window frames and normalized left-click points, and never
records keyboard input or uploads data. Task descriptions and target labels are
saved with the examples; do not include secrets or real user content.

The output contains `manifest.json`, `samples.jsonl`, and `frames/*.png`.
Samples are raw human demonstrations, not model predictions or validation
results. The checked-in Cortex-2M checkpoint was trained on procedural colored blocks.
Its recorded synthetic validation category accuracy is 10.83%; this is not a
real-application grounding score. The synthetic trainer now requires the
explicit `--synthetic-data-only` flag and writes to
`runs/synthetic_cortex_2m` by default. These demonstrations still need to be
reviewed, split by task/window, and used to train and independently evaluate a
click-grounding model. Until then, AVCS must not claim general GUI task
capability.

## Goal-conditioned pixel grounder

`GoalConditionedGrounder` (`avcs/goal_grounder.py`) is the new System-1
grounding architecture. It consumes an RGB frame and the planner's target
phrase, then produces a dense spatial click-target heatmap. Its learned byte
encoder consumes only the supplied phrase; UI pixels are not OCR'd or replaced
with accessibility/DOM state. It does not plan multi-step tasks, type text,
dispatch input, or verify that a click succeeded.

The architecture is implemented but **has not been trained on real UI
demonstrations**. `predict()` intentionally refuses inference from a randomly
initialized model. Train only after collecting at least five separate
recording sessions; the trainer reserves complete sessions for validation and
held-out test, so frames from one session cannot leak between splits and at
least three sessions remain for training:

```powershell
avcs-train-grounder `
  --data-dir "runs/demonstrations" `
  --output-dir "runs/real_ui_grounder"
```

The same trainer is available without the installed console entry point as
`python -m avcs.training.train_grounder --data-dir "runs/demonstrations"`.

The trainer selects a checkpoint by validation loss, reloads that checkpoint,
then evaluates it once on the session-disjoint held-out test split. Its report
includes Top-1 point accuracy at 1%, 2%, and 5% of the screenshot diagonal.
The requested 85% goal is reported against Top-1 accuracy within 2% of the
diagonal; this is an explicit point-localization metric, not target-box hit
rate, task completion, or an OSWorld score. Checkpoint loading for inference
fails closed unless this held-out metric reaches 85%. The recorder labels
clicks rather than target bounds, so target-box hit rate remains unmeasured.
Keep the held-out test sessions away from model selection and collect examples
across diverse applications, resolutions, control types, and tasks before
treating the score as representative.

Once trained, a checkpoint can be loaded with `load_grounding_checkpoint`;
its output score is an uncalibrated peak probability, not a confidence
guarantee. Do not connect candidate coordinates to live input until separate
safety, calibration, and isolated acceptance work establishes that behavior.
The 85% number is a requested acceptance target, not a current AVCS or SONIC
result. End-to-end task success, per-action latency, and cost require a
separate reproducible OSWorld run with receipts and a fixed task set.

SONIC can optionally use this evaluated grounder for natural-language GUI
target coordinates. Install the backend's `avcs-grounding` extra and configure
`SONIC_AVCS_GROUNDING_CHECKPOINT` to the resulting `goal_grounder_best.pt`.
To include PyTorch in the backend image, build with
`--build-arg INSTALL_AVCS_GROUNDING=1`. Production Compose also accepts
`SONIC_INSTALL_AVCS_GROUNDING=1` at build time and
`SONIC_AVCS_GROUNDING_CHECKPOINT` at runtime.
SONIC refuses this route when the checkpoint is absent, malformed, lacks
session-disjoint held-out evaluation, or falls below 85% point accuracy. No
checkpoint is bundled; keep operator-collected demonstrations and model
weights outside source control.
