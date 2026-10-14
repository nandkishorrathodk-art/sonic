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
