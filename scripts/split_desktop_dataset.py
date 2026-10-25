"""Split a labeled desktop capture manifest into train / held-out sets.

The split is by *capture round and scene instance* so frames from the same
screen state never leak across the boundary; a held-out score measured on
near-duplicate training frames would be meaningless.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--held-out-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = data["frames"]

    # Group by (label, note) = one real screen state; split whole groups so
    # near-duplicate frames never leak across the boundary.  Stratify per label
    # so every class is represented in BOTH splits whenever it has >= 2 groups.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in frames:
        groups[(f["label"], f.get("note", ""))].append(f)

    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in groups:
        by_label[key[0]].append(key)

    rng = random.Random(args.seed)
    held_keys: set[tuple[str, str]] = set()
    for label, keys in sorted(by_label.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        if len(keys) >= 2:
            n_held = max(1, round(len(keys) * args.held_out_frac))
            n_held = min(n_held, len(keys) - 1)  # keep >=1 group for training
        else:
            n_held = 0  # a single-instance class must stay in training
        held_keys.update(keys[:n_held])

    train, held = [], []
    for key, items in groups.items():
        (held if key in held_keys else train).extend(items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split, note in (("train", train, "training split"),
                              ("heldout", held, "held-out split")):
        (args.out_dir / f"{name}.json").write_text(
            json.dumps({"provenance": data.get("provenance", "real_desktop_capture"),
                        "split": note,
                        "frames": split}, indent=2),
            encoding="utf-8",
        )
        print(f"{name}: {len(split)} frames")

    tr_labels = sorted({f['label'] for f in train})
    he_labels = sorted({f['label'] for f in held})
    print(f"train labels   : {tr_labels}")
    print(f"heldout labels : {he_labels}")
    missing = sorted(set(he_labels) - set(tr_labels))
    if missing:
        print(f"WARNING: held-out labels absent from train: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
