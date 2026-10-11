# AVCS local demonstration data

`avcs-record-demo` records operator-labeled left-click examples for future
grounding-model training. It does not choose targets, move the cursor, type, or
claim task completion.

## Record a disposable window

1. Open a disposable test application window with a unique title and no private
   content.
2. From an interactive PowerShell terminal, run:

   ```powershell
   python -m avcs.training.demo_recorder `
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
initialized model. Train only after collecting at least five distinct
task/window groups. The trainer reserves each complete group for one split,
so repeated recordings of the same task in the same window cannot leak across
training, validation, and held-out test; at least three groups remain for
training:

```powershell
python -m avcs.training.train_grounder `
  --data-dir "runs/demonstrations" `
  --output-dir "runs/real_ui_grounder"
```

The trainer selects a checkpoint by validation loss, reloads that checkpoint,
then evaluates it once on the task/window-disjoint held-out test split. Its report
includes Top-1 point accuracy at 1%, 2%, and 5% of the screenshot diagonal.
The requested 85% goal is reported against Top-1 accuracy within 2% of the
diagonal; this is an explicit point-localization metric, not target-box hit
rate, task completion, or an OSWorld score. Checkpoint loading for inference
fails closed unless this held-out metric reaches 85%. The recorder labels
clicks rather than target bounds, so target-box hit rate remains unmeasured.
Keep the held-out test sessions away from model selection and collect examples
across diverse applications, resolutions, control types, and tasks before
treating the score as representative. Query-confidence calibration selects a
threshold using validation predictions and checks precision on held-out
predictions; it requires at least 10 accepted examples on each split and 90%
held-out precision. A run that misses either the point-accuracy or
confidence-calibration gate exits unsuccessfully and is not runtime-ready.

The checkpoint's query score is the spatial softmax peak, gated by a
validation-selected threshold whose empirical precision is measured on the
held-out split. This calibration is meaningful only for data representative
of the intended applications and tasks. Checkpoint metadata records the
reported metrics and coordinate frame, but is not cryptographic proof that
the weights came from the claimed data or evaluation; use trusted training
outputs and retain the corresponding dataset and report. Do not connect
candidate coordinates to live input until separate safety and isolated
acceptance work establishes that behavior.
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
held-out evaluation or confidence calibration, or falls below
the stated gates. No checkpoint is bundled; keep operator-collected
demonstrations and model weights outside source control.
