"""Build the Kaggle notebook that trains the local AVCS goal grounder.

Run this to regenerate ``packages/avcs/notebooks/train_grounder_kaggle.ipynb``
after editing the cell sources below::

    python packages/avcs/notebooks/build_train_grounder_notebook.py

Keeping the notebook generated (rather than hand-edited JSON) avoids fragile
escaping and makes the content reviewable as ordinary Python strings.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "train_grounder_kaggle.ipynb"

MARKDOWN_INTRO = """# AVCS Goal Grounder — Kaggle GPU Training

Trains the **existing local** `GoalConditionedGrounder` (the model SONIC already
loads) on recorded UI click demonstrations, so semantic GUI targeting works
without any cloud vision API.

## What this notebook is

- **Model**: `avcs.goal_grounder.GoalConditionedGrounder` (~776K params, runs on
  CPU at inference). Not a large VLM — this is the checkpoint
  `load_grounding_checkpoint` already gates on.
- **Data**: demonstration sessions produced by `avcs.training.vnc_demo_recorder`
  against the Docker desktop (one human click per sample).
- **Output**: `goal_grounder_best.pt`, carrying the real-demonstration
  provenance that `load_grounding_checkpoint` requires.

## What this notebook is NOT

- It does **not** train a Qwen/VLM. Generic cloud VLMs were measured and do not
  ground (see `AGENTS.md`, "Cloud VLM grounding bridge").
- It does **not** invent demonstrations. If you have no recorded sessions, the
  synthetic generator produces a *smoke-test* dataset that is explicitly marked
  `synthetic_rendered_controls` and is **rejected** by the production loader.

## Getting real demonstration data

Record on a machine with the Docker workstation + X11:

```bash
python -m avcs.training.vnc_demo_recorder --task "<what you are doing>" \\
    --window-title "<exact window title>" --output-dir sessions/<name>
```

Type a target label in the terminal, press Enter, then left-click the control in
the VNC desktop. Repeat, then `:stop`. Upload the `sessions/` directory as a
Kaggle dataset and point `DATA_DIR` at it.
"""

CELL_GPU = '''# Step 1 — Check GPU environment
import torch

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"Count   : {torch.cuda.device_count()}")
else:
    print("WARNING: no GPU detected; training will run on CPU (slow).")
'''

CELL_CLONE = '''# Step 2 — Clone the SONIC repository (training code lives in packages/avcs)
import os
import subprocess

REPO_URL = os.environ.get("SONIC_REPO_URL", "https://github.com/nandkishorrathodk-art/sonic.git")
REPO_DIR = "/kaggle/working/sonic"

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

AVCS_DIR = os.path.join(REPO_DIR, "packages", "avcs")
os.chdir(AVCS_DIR)
print("AVCS dir:", AVCS_DIR)
print(subprocess.run(["git", "-C", REPO_DIR, "log", "-1", "--oneline"],
                     capture_output=True, text=True).stdout.strip())
'''

CELL_DEPS = '''# Step 3 — Dependencies (torch/PIL/numpy are preinstalled on Kaggle)
import importlib
for name in ("torch", "PIL", "numpy"):
    importlib.import_module(name)
    print(f"{name}: ok")
'''

CELL_DATA = '''# Step 4 — Point at your demonstration sessions
#
# REAL DATA (preferred): upload your recorder sessions as a Kaggle dataset and
# set DATA_DIR to its path, e.g. "/kaggle/input/sonic-ui-clicks".
#
# SMOKE TEST (no real data): leave DATA_DIR unset. A synthetic dataset is
# generated, which is marked 'synthetic_rendered_controls' and will be REJECTED
# by load_grounding_checkpoint. It only proves the pipeline runs.

import os
import shutil
import subprocess

REAL_DATA_DIR = os.environ.get("DATA_DIR", "").strip()
SMOKE = not REAL_DATA_DIR

if SMOKE:
    DATA_DIR = "/kaggle/working/synthetic_sessions"
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    subprocess.run(
        ["python", "-m", "avcs.training.synthetic_sessions",
         "--output-dir", DATA_DIR, "--sessions", "24",
         "--controls-per-session", "8", "--seed", "0"],
        check=True,
    )
    print("\\nSMOKE TEST MODE: synthetic data. The checkpoint will NOT be "
          "production-usable (loader rejects synthetic provenance).")
else:
    DATA_DIR = REAL_DATA_DIR
    print("REAL DATA MODE:", DATA_DIR)

# Count sessions so the split is known to be viable.
session_count = sum(
    1 for root, _dirs, files in os.walk(DATA_DIR) if "samples.jsonl" in files
)
print(f"sessions found: {session_count}")
assert session_count >= 3, "need at least 3 disjoint sessions for a train/val/test split"
'''

CELL_TRAIN = '''# Step 5 — Train the local goal grounder
#
# Small model, so epochs are cheap. Increase --epochs / --input-size for a
# stronger checkpoint. GPU is used automatically when available.

import subprocess

OUTPUT_DIR = "/kaggle/working/runs/real_ui_grounder"
cmd = [
    "python", "-m", "avcs.training.train_grounder",
    "--data-dir", DATA_DIR,
    "--output-dir", OUTPUT_DIR,
    "--epochs", "30",
    "--batch-size", "32",
    "--input-size", "256",
    "--overwrite",
]
print(" ".join(cmd))
result = subprocess.run(cmd, text=True)
print("exit code:", result.returncode)
'''

CELL_VERIFY = '''# Step 6 — Verify the checkpoint against the production loader contract
import json
from pathlib import Path

import torch

from avcs.goal_grounder import load_grounding_checkpoint

CKPT = Path("/kaggle/working/runs/real_ui_grounder/goal_grounder_best.pt")
REPORT = Path("/kaggle/working/runs/real_ui_grounder/training_report.json")

raw = torch.load(CKPT, map_location="cpu", weights_only=True)
report = json.loads(REPORT.read_text())

print(f"dataset_provenance : {raw['dataset_provenance']}")
print(f"training_samples   : {raw['training_samples']}")
print(f"validation_samples : {raw['validation_samples']}")
print(f"held_out_test      : {raw['held_out_test_samples']}")
print(f"runtime_ready      : {report['runtime_ready']}")
print(f"held-out metric    : {report['held_out_test_metrics']}")

# This is the real gate: the production loader must accept the checkpoint.
try:
    model = load_grounding_checkpoint(CKPT, device="cpu")
    print("\\nPRODUCTION LOADER: ACCEPTED — checkpoint is usable by SONIC.")
except Exception as exc:  # noqa: BLE001 - surface the exact rejection reason
    print(f"\\nPRODUCTION LOADER: REJECTED — {exc}")
    if report["dataset_provenance"] != "recorded_ui_click_demonstrations":
        print("This is expected for the synthetic smoke test. Train on real "
              "recorded sessions to obtain a production checkpoint.")
'''

CELL_INFER = '''# Step 7 — Smoke-test inference on a held-out frame
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from avcs.goal_grounder import encode_queries, letterbox_rgb

if "model" in dir():
    # Pick any frame from the dataset and ask the model to find a control.
    frame_path = next(Path(DATA_DIR).rglob("frames/*.png"))
    with Image.open(frame_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    tensor, _transform = letterbox_rgb(rgb, model.input_size)
    image_batch = tensor.unsqueeze(0)
    query = encode_queries(["button 0"])
    with torch.inference_mode():
        logits = model(image_batch, query)
    flat = logits[0, 0].flatten()
    index = int(flat.argmax())
    height, width = logits.shape[-2:]
    x = ((index % width) + 0.5) / width
    y = ((index // width) + 0.5) / height
    print(f"frame     : {frame_path.name}")
    print(f"predicted : normalized ({x:.3f}, {y:.3f})")
    print("This is an untrained-quality smoke check, not an accuracy claim.")
else:
    print("Model not loaded (checkpoint rejected). Skipping inference.")
'''

CELL_SAVE = '''# Step 8 — Package the checkpoint for download
import shutil

archive = shutil.make_archive(
    "/kaggle/working/sonic_grounder_checkpoint", "zip",
    "/kaggle/working/runs/real_ui_grounder",
)
print("Wrote:", archive)
print("\\nTo use in SONIC: unzip and set")
print("  SONIC_AVCS_GROUNDING_CHECKPOINT=<path>/goal_grounder_best.pt")
'''

CELLS = [
    ("markdown", MARKDOWN_INTRO),
    ("code", CELL_GPU),
    ("code", CELL_CLONE),
    ("code", CELL_DEPS),
    ("code", CELL_DATA),
    ("code", CELL_TRAIN),
    ("code", CELL_VERIFY),
    ("code", CELL_INFER),
    ("code", CELL_SAVE),
]


def build_notebook() -> dict:
    cells = []
    for cell_type, source in CELLS:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }


def main() -> int:
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1) + "\n", encoding="utf-8"
    )
    print(f"Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
