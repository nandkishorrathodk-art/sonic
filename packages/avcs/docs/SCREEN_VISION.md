# Real-time screen vision: what this is, and what it honestly is not

The goal: a model that watches a computer screen the way a human eye does,
continuously, from raw pixels alone, and says what is happening in natural
language.

This document states plainly which half of that is real-time, which half is
generative and slow, and where the limits are. The same discipline the rest of
the repo applies to its claims applies here.

## The constraint that shapes everything

On this machine (4 CPU cores, no GPU), a language model cannot look at every
frame. A single forward pass of even a small transformer is tens to hundreds of
milliseconds; at 25 fps you have a 40 ms budget. So "real-time" and
"generative description" cannot live on the same clock.

The design answers that by splitting the job in two:

```
pixels ──▶ reflex perception (NON-generative) ──▶ typed SceneState + confidence
                     every tick, ~2-3 ms                     │
                                                            ├─▶ events (natural language, from thresholds)
                                                            │
frame ───▶ optional VLM captioner (GENERATIVE) ◀────────────┘  change-only, ~1 s
```

* The **reflex layer** is non-generative. It does array math on a downscaled
  luminance grid and emits a typed `SceneState`. No text model, no sampling. It
  runs at the capture rate.
* The **understanding head** is a small trained network over the reflex grids.
  It predicts a scene type, an activity/change/overlay vector, a *calibrated
  confidence*, and -- importantly -- a multi-label distribution over a closed
  description vocabulary. The sentence is rendered from the words the model
  actually predicted, so a wrong sentence is a wrong *prediction*, not a wrong
  template.
* A vision-language model can be plugged in as an optional captioner, but it
  runs *on change only* and is allowed to lag.

This is the System-1 / System-2 split, made concrete by a latency budget rather
than by taste.

## What the reflex layer computes, from pixels only

No OCR. No accessibility tree. No accessibility APIs. No NSI. No structured
element detection. Only what the pixels contain:

| Quantity | How |
|---|---|
| layout | luminance + edge density, pooled into tiles |
| saliency | per-tile deviation from the frame mean |
| motion | frame-to-frame luminance delta, pooled into tiles |
| change | which tiles crossed a change threshold |
| regions | connected busy tiles, classed panel / band / patch by shape |
| overlay | a compact block of strong change, not a whole-scene swap |
| novelty | cosine distance to the closest remembered frame |
| identity | average perceptual hash of the frame |

## Measured on this host

Full System-1 tick (reflex perception **plus** the understanding forward pass),
16x24 tile grid, 4x downscale, synthetic 640x360 desktop:

| metric | value |
|---|---|
| reflex p50 | ~2.5 ms |
| reflex p95 | ~3.2 ms |
| full tick p50 | ~3.7 ms |
| full tick p95 | ~4.6 ms |
| observed fps | 25 (capped by `--fps`); uncapped the tick allows ~250+ |

The 40-80 ms budget applies to the **whole tick**, not to perception alone, so
the reported figure is reflex + model forward. Quoting reflex-only latency would
flatter the system by roughly a third.

The number that must not be confused with the tick is *description* latency: a
local VLM caption is order 1 second on CPU. Descriptions are change-triggered
and run on a separate thread; the dashboard reports tick fps, not caption fps.

## Honest evaluation

The first version reported a training loss falling to ~0.02 and a per-word
accuracy of ~0.9. Both were misleading, for two different reasons, and both are
now fixed:

1. **Training loss on a single set is memorisation, not understanding.** The
   demo now trains on four scene categories and evaluates on fresh randomised
   draws *and* on a fifth category that is excluded from training entirely
   (`--holdout`). Reported numbers are held-out.
2. **Raw per-word accuracy is dominated by always-on words.** `screen`,
   `mid-tone`, `detailed` and `patch` are present in almost every frame, so a
   model that never discriminates still scores ~0.9. The report now leads with
   **macro-F1** over discriminative words and an *informative accuracy* that
   excludes the always-on set, plus the weakest individual words.

Representative run (240 training states, 600 steps, `--holdout terminal`):

| metric | validation (fresh draws) | held-out mix (unseen category) |
|---|---|---|
| macro-F1 | 0.58 | 0.50 |
| informative accuracy | 0.91 | 0.90 |
| weakest words | flat, bottom, left, right, band -> F1 0.00 | |

The honest read: the model reliably gets tone, detail and coarse structure, and
is **unreliable on directional and motion words** -- exactly the hardest part.
A macro-F1 near 0.5 is a real perception model that has learned something, not a
finished one.

## What this is not

* **Not a trained large vision model.** The understanding heads are ~26k
  parameters and train on labelled reflex states in seconds. There is no
  pretraining on image datasets here.
* **It cannot read text.** Without OCR it can see that text is present
  (high-edge regions, surfaced as the `text` vocabulary word) but not what it
  says. That is a deliberate consequence of the "pixels only" mandate; a VLM
  backend is the way to read text, and it is opt-in and slow.
* **The evaluation is synthetic.** The scene generators (`scenes.py`) vary
  window size, position, luminance and speed, and withhold whole categories, so
  the test catches positional memorisation -- but it is not a substitute for
  evaluation on recorded real desktops. That requires a display.
* **Description is change-triggered, not per-frame.** This is the honest
  tradeoff that makes the loop real-time. Per-frame rich description is not
  achievable at this scale.

## Adding a real desktop source

`ScreenCapture` uses `mss` and is already the default when a display exists;
`open_capture()` falls back to the synthetic desktop otherwise. If capture
returns a black or empty frame, the cause is almost always no `$DISPLAY` (a
headless shell), not the code. `FrameSource(provider)` wraps any callable that
returns an `(H, W, 3)` uint8 array, for webcams or video files.

## Running it

```bash
python3 -m forge.vision_demo --seconds 10              # real desktop if a display exists
python3 -m forge.vision_demo --source synthetic        # headless development
python3 -m forge.vision_demo --vlm                     # try a local VLM captioner
```

Outputs `runs/screen_vision.html` (graphical dashboard, inline SVG, no external
assets) and `runs/screen_vision.json` (pollable typed state).
