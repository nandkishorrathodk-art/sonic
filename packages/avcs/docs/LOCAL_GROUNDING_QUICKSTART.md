# Collecting local UI grounding demonstrations on Windows

The goal-conditioned grounder is not ready to use on arbitrary desktop
controls until it has a real, held-out evaluated checkpoint. The repository
does not currently contain that checkpoint. Do not treat synthetic rectangles,
unit tests, or a successful recorder run as evidence of real-application
accuracy.

## Start an isolated demonstration window

Use one interactive PowerShell window for the target. Keep it open while
recording:

```powershell
Set-Location C:\Users\nandk\sonic\AVCS
New-Item -ItemType Directory -Force -Path runs\demonstrations | Out-Null
& "C:\Users\nandk\AppData\Local\Python\pythoncore-3.14-64\python.exe" `
  -m avcs.testing.interactive_test_window `
  --state-file "runs\demonstrations\test-window-state.json" `
  --title "AVCS_DEMO_WINDOW"
```

The test window is a controlled target for harmless demonstrations. It does not
open automatically when the recorder starts.

## Record demonstrations

Open a second interactive PowerShell window. The title must exactly match the
visible target window title, and the output directory must be new:

```powershell
Set-Location C:\Users\nandk\sonic\AVCS
& "C:\Users\nandk\AppData\Local\Python\pythoncore-3.14-64\python.exe" `
  -m avcs.training.demo_recorder `
  --window-title "AVCS_DEMO_WINDOW" `
  --task "Click the requested control in the isolated test window" `
  --target-label "test target button" `
  --output-dir "runs\demonstrations\session-01" `
  --confirm-isolated-window
```

With `--target-label`, that label is reused for each click, so you can focus the
isolated target and record repeated examples without entering a label in the
terminal before every click. To label different controls in the same session,
omit `--target-label`; enter each target label in the recorder terminal, press
Enter, then focus the target window and click that control. The recorder prints
a specific reason when a click is ignored. Stop the recorder by entering
`:stop` (including the colon) in its terminal. Starting only the recorder, or
entering a title for a window that is not already open, will not create a
target window.

Record only harmless, operator-approved interactions. Avoid personal data,
credentials, and sensitive application content. The recorder stores local
frames and click coordinates; review the output before training or sharing it.

## Training and readiness

Training data must come from genuine UI demonstrations, not generated
rectangles. Keep related captures in the same group so one application/task
does not leak between training and held-out evaluation. Review labels and
coordinates before training.

Only configure a resulting checkpoint for SONIC if the trainer reports
`runtime_ready: true` and the checkpoint passes the held-out point-localization
and query-confidence gates. These are grounding metrics, not proof of target
hit rate or end-to-end task completion. Without an evaluated checkpoint, SONIC
must continue to abstain from ungrounded semantic clicks. Text entry also
remains blocked until a separately evaluated local focus-verification head is
available.

## Focus-verification head

Text entry needs a separate local focus head that answers "is this point a
focused text-entry control?" from the current frame alone. It is trained the same
way as the grounder, from recorded demonstrations:

- Record focus demonstrations: run the dedicated recorder inside the Docker VNC
  desktop (`avcs-record-focus`, or
  `python -m avcs.training.focus_demo_recorder`). It reuses the click recorder's
  isolation guarantees (exact foreground window, cropped frames, no keystroke
  capture) and adds the human `focused` ground truth:

  ```bash
  python -m avcs.training.focus_demo_recorder \
    --window-title "AVCS_DEMO_WINDOW" \
    --task "Label focused vs unfocused controls" \
    --output-dir "runs/focus_demos/session-01" \
    --target "search field:focused" \
    --target "Cancel:not-focused" \
    --confirm-isolated-window
  ```

  Each session directory it writes holds a `manifest.json` (`status: "complete"`,
  `task`, `target_window_title`) and a `records.jsonl` whose entries carry
  `schema_version: 1`, `target_label`, `frame`, a `point_px` pixel coordinate in
  the cropped window frame, and a boolean `focused` label. The `focused` value is
  a human ground-truth label -- it is never inferred from the OS or a model,
  because a classifier cannot be trained against its own guess. Group each
  app/task into its own session so training and held-out evaluation stay
  leakage-resistant. Use `--target "<label>:<focused>"` to pre-arm a sequence of
  clicks, or run without `--target` to type the label and answer the focused y/n
  prompt before each click.
- Train: `python -m avcs.training.train_focus_verifier --data-dir <root>
  --output-dir runs/focus_verifier`. The trainer writes
  `focus_verifier_best.pt` and a `training_report.json`.
- Readiness: the report is `runtime_ready: true` only when held-out precision is
  at least 0.80 over at least 30 accepted samples. Otherwise the checkpoint is
  still written but the loader refuses it.
- Configure: set `SONIC_AVCS_FOCUS_CHECKPOINT` to the checkpoint path. A
  configured path that is missing, untrained, or below the gate keeps text entry
  blocked; readiness is proven from checkpoint metadata, never from the path.
