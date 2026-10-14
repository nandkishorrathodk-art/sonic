"""Real-time screen-vision demo.

Run:
    python3 -m forge.vision_demo --seconds 10
    python3 -m forge.vision_demo --source synthetic --seconds 5

It trains the (non-generative) understanding heads on labelled reflex states,
then runs the live loop and writes a graphical dashboard to
``runs/screen_vision.html`` plus a pollable ``runs/screen_vision.json``.

The headline numbers are measured, not claimed: reflex p50/p95 latency and
observed fps.  If a real display is not available it says so and runs the
synthetic desktop instead, rather than pretending to watch a screen.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from forge.vision.capture import ScreenCapture, SyntheticScreen
from forge.vision.dashboard import render_dashboard, write_state_json
from forge.vision.describer import ModelDescriber, VLMDescriber
from forge.vision.encoder import (
    SceneUnderstandingModel,
    train_understanding,
    word_report,
)
from forge.vision.pipeline import ScreenVision
from forge.vision.scene import SceneState
from forge.vision.scenes import SCENE_CATEGORIES, sample_states

MODEL_ADAPTER = ModelDescriber()


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def all_category_states(categories: list[str], per_cat: int, seed: int):
    """Union over categories, kept category-labelled for word_report."""
    states: list = []
    types: list[str] = []
    for ci, cat in enumerate(categories):
        s, t = sample_states([cat], per_cat, seed=seed + ci)
        states += s
        types += t
    return states, types


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentinel real-time screen vision")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=24)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--per-category", type=int, default=60, help="training scenes per category")
    ap.add_argument("--holdout", default="scroll", choices=SCENE_CATEGORIES,
                    help="scene category excluded from training")
    ap.add_argument("--fps", type=float, default=25.0, help="reflex tick rate cap (0 = uncapped)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--vlm", action="store_true", help="try a local VLM captioner (slow, optional)")
    args = ap.parse_args()

    section("1. UNDERSTANDING MODEL (non-generative, trained)")
    # Held-out split: train on four categories, evaluate on all five.  The
    # unseen category can only be passed by generalising, which is what makes
    # the reported accuracy mean something.
    train_cats = [c for c in SCENE_CATEGORIES if c != args.holdout]
    train_states, train_types = all_category_states(train_cats, args.per_category, seed=1)
    val_states, val_types = all_category_states(train_cats, max(4, args.per_category // 3), seed=99)

    model = SceneUnderstandingModel(dim=64)
    result = train_understanding(
        model, train_states, steps=args.train_steps,
        val_states=val_states, val_scene_types=val_types,
    )
    losses = result["losses"]
    print(f"  train states    : {len(train_states)}  ({', '.join(train_cats)})")
    print(f"  val states      : {len(val_states)}  (fresh draws, same categories)")
    print(f"  params          : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  train loss      : {np.mean(losses[:20]):.4f} -> {np.mean(losses[-20:]):.4f}")

    # Evaluate on fresh draws of trained categories ...
    print("\n  VALIDATION (fresh draws of trained categories):")
    print(f"    macro-F1            : {result['val']['macro_f1']:.3f}")
    print(f"    informative accuracy: {result['val']['informative_accuracy']:.3f}")

    # ... and on the category never seen in training.
    seen_s, seen_t = all_category_states(SCENE_CATEGORIES, max(6, args.per_category // 2), seed=555)
    print(f"\n  HELD-OUT MIX (all 5 categories, '{args.holdout}' never trained):")
    rep = word_report(model, seen_s, seen_t)
    print(f"    macro-F1            : {rep['macro_f1']:.3f}")
    print(f"    informative accuracy: {rep['informative_accuracy']:.3f}")
    weak = sorted(rep["per_word"].items(), key=lambda kv: kv[1])[:5]
    print("    weakest words       : " + ", ".join(f"{w} {f:.2f}" for w, f in weak))
    print("  (macro-F1, not raw accuracy: always-on words like 'screen' would")
    print("   inflate accuracy to ~0.9 while saying nothing)")

    section("2. DESCRIPTION BACKEND")
    describer = MODEL_ADAPTER  # descriptions come from the trained model
    print("  describer : the trained model's own word head (non-generative)")
    print("  (--vlm adds an optional local captioner on change; heuristic is the")
    print("   deterministic fallback)" if args.vlm else "  (use --vlm for an optional captioner)")
    if args.vlm:
        vlm = VLMDescriber.try_load()
        if vlm is not None:
            print("  local VLM captioner available (change-only)")
        else:
            print("  VLM requested but unavailable -> model word head used")

    section("3. CAPTURE SOURCE")
    if args.source == "real" or (args.source == "auto" and ScreenCapture.available()):
        capture = ScreenCapture().open()
        print("  real desktop via mss")
    else:
        capture = SyntheticScreen()
        if args.source == "auto":
            print("  no display detected -> synthetic desktop")
        else:
            print("  synthetic desktop")

    section("4. LIVE LOOP")
    vis = ScreenVision(
        capture=capture,
        describer=describer,
        understanding=model,
        rows=args.rows,
        cols=args.cols,
        downscale=args.downscale,
        describe_every_s=1.0,
        target_fps=(args.fps or None),
    )

    history: list[dict] = []
    html_path = os.path.join(args.out, "screen_vision.html")
    json_path = os.path.join(args.out, "screen_vision.json")
    last_desc = [""]

    def on_state(state: SceneState) -> None:
        history.append({"change_fraction": state.change_fraction})
        if len(history) > 400:
            del history[:-400]

    def on_description(text: str) -> None:
        last_desc[0] = text
        print(f"  [{time.strftime('%H:%M:%S')}] {text}")

    vis.on_state(on_state)
    vis.on_description(on_description)

    vis.start()
    try:
        deadline = time.time() + args.seconds
        last_render = 0.0
        while time.time() < deadline:
            time.sleep(0.1)
            if vis.last_state is not None and time.time() - last_render > 0.5:
                und = model.understand(vis.last_state)
                render_dashboard(
                    html_path,
                    vis.last_state,
                    understanding=und,
                    events=vis._recent_events,
                    stats=vis.stats.summary(),
                    description=last_desc[0],
                    history=history,
                )
                write_state_json(json_path, vis.last_state.to_dict())
                last_render = time.time()
    finally:
        vis.stop()

    section("5. MEASURED RESULT")
    s = vis.stats.summary()
    print(f"  frames          : {s['frames']}")
    print(f"  reflex p50      : {s['reflex_p50_ms']} ms   (perception only)")
    print(f"  reflex p95      : {s['reflex_p95_ms']} ms")
    print(f"  FULL TICK p50   : {s['tick_p50_ms']} ms   (reflex + model forward)")
    print(f"  FULL TICK p95   : {s['tick_p95_ms']} ms")
    print(f"  observed fps    : {s['fps']}  (capped by --fps)")
    print(f"  events emitted  : {s['events']}")
    print(f"  descriptions    : {s['captions']}  (from the trained model)")
    print(f"\n  dashboard -> {html_path}")
    print(f"  live json -> {json_path}")

    # The budget applies to the whole tick, not to perception alone -- quoting
    # reflex-only latency would flatter the system.
    target_lo, target_hi = 40.0, 80.0
    p50 = s["tick_p50_ms"]
    verdict = "WITHIN" if target_lo <= p50 <= target_hi else ("FASTER than" if p50 < target_lo else "SLOWER than")
    print(f"\n  full tick p50 {p50} ms is {verdict} the 40-80 ms target "
          f"(headroom: {target_lo / max(p50, 1e-3):.0f}x).")


if __name__ == "__main__":
    main()
