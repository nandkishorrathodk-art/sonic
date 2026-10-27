# Grounding checkpoint: Docker recorder -> Kaggle training -> SONIC

End-to-end path to give SONIC semantic GUI targeting ("click the Save button")
using the **local** grounder, with no cloud vision API.

The local grounder (`avcs.goal_grounder.GoalConditionedGrounder`) is a ~776K
parameter model that runs on CPU at inference. It is the model
`load_grounding_checkpoint` gates on, and it is what `SONIC_AVCS_GROUNDING_CHECKPOINT`
must point at. There is no shipped checkpoint in the repository; you produce one
from your own demonstrations.

> Honesty rule: a checkpoint is usable only if it was trained on **recorded human
> clicks**. The loader enforces this from the checkpoint's `dataset_provenance`.
> Synthetic data is for pipeline smoke tests and is rejected by the loader.

## 1. Record demonstrations on the Docker workstation (Linux)

The Docker workstation runs X11 on `DISPLAY=:99`. The VNC recorder attaches to
that display, captures cropped window frames, and records one left-click per
label you type.

```bash
# Inside an environment that can reach the workstation's X11 display.
export DISPLAY=:99

python -m avcs.training.vnc_demo_recorder \
  --task "Complete the test form in the isolated window" \
  --window-title "<exact visible window title>" \
  --output-dir sessions/form-01
```

The recorder requires the exact, unique, visible window title and a **new**
output directory. It prints "Label armed" when ready; then left-click the
control in the VNC desktop. Enter `:stop` to finish.

Repeat per application/task so each becomes its own session directory. Grouping
matters: `split_sessions` keeps sessions disjoint across train/validation/test,
so one session must not mix unrelated tasks.

Each session writes:

```
sessions/form-01/
  manifest.json    # schema_version, status, task, target_window_title, sample_count
  samples.jsonl    # one mouse_click record per sample
  frames/*.png     # cropped window frame for each sample
```

The recorder never records keyboard input and never uploads data.

## 2. Train on Kaggle (GPU)

Open `packages/avcs/notebooks/train_grounder_kaggle.ipynb` on Kaggle.

1. Upload your `sessions/` directory as a Kaggle dataset.
2. Set `DATA_DIR` to the dataset path (e.g. `/kaggle/input/sonic-ui-clicks`).
3. Run all cells. Step 5 trains; Step 6 verifies against the production loader.

If `DATA_DIR` is left unset the notebook generates a synthetic dataset, which
proves the pipeline runs but produces a checkpoint the loader **rejects**. That
is intentional.

Local equivalent (CPU is fine for the small model):

```bash
python -m avcs.training.train_grounder \
  --data-dir sessions/ \
  --output-dir runs/real_ui_grounder \
  --epochs 30 --batch-size 32 --input-size 256
```

Outputs:

- `goal_grounder_best.pt` — checkpoint with `dataset_provenance`
  `recorded_ui_click_demonstrations`.
- `training_report.json` — held-out metrics plus `runtime_ready`.

`runtime_ready` is true only when the held-out point accuracy meets the
acceptance threshold **and** the data carried real recorder provenance.

## 3. Wire the checkpoint into SONIC

```bash
export SONIC_AVCS_GROUNDING_CHECKPOINT=/path/to/goal_grounder_best.pt
```

With it set, a semantic `GUI_CLICK` target ("Save button") resolves through the
local grounder instead of being blocked. Without it, semantic targets fail
closed; that is the current default.

Text entry additionally needs a focus-verification checkpoint
(`SONIC_AVCS_FOCUS_CHECKPOINT`); see the focus section of
`LOCAL_GROUNDING_QUICKSTART.md`.

## Synthetic data (smoke tests only)

To exercise the whole pipeline without a desktop:

```bash
python -m avcs.training.synthetic_sessions \
  --output-dir /tmp/synth_demos --sessions 12 --controls-per-session 6
python -m avcs.training.train_grounder \
  --data-dir /tmp/synth_demos --output-dir /tmp/synth_run \
  --epochs 3 --batch-size 8 --input-size 64
```

The generator marks sessions `synthetic_rendered_controls`. The trainer prints a
warning, stamps that provenance into the checkpoint, sets `runtime_ready: false`,
and `load_grounding_checkpoint` refuses the result. Synthetic data can never be
mistaken for recorded demonstrations.
