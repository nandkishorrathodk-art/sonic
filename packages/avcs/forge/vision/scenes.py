"""Scene generators with randomised parameters and held-out categories.

Why this exists: the first version trained and evaluated on one fixed synthetic
desktop, so a falling training loss proved only that the model could memorise
one scene.  To say anything about generalisation, train and evaluation must
differ in ways that matter.

Two axes of variation:

* **Parameter jitter** inside a category -- window sizes, positions, base
  luminance, cursor speed.  Train and val draws differ, so a model that keyed
  on exact pixel positions is caught.
* **Held-out categories** -- whole scene archetypes that are excluded from
  training and used only for whichever of them appears at evaluation.  A model
  can only pass these by generalising, not by recall.  Held-out categories are
  deliberately parameter-sampled from the union of *seen* categories, so the
  test isolates "new arrangement of known primitives" from "brand new
  primitive".

Honest scope: this is still synthetic.  It raises the bar and it catches
positional memorisation, but it is not a substitute for evaluation on recorded
real desktops.  That needs a real capture, which requires a display.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

SCENE_CATEGORIES = [
    "desktop",   # bright panels on a mid background
    "terminal",  # dark, fine text-like structure, low motion
    "media",     # large smooth colour field, high motion
    "dialog",    # dim base with a compact bright overlay
    "scroll",    # one long textured band moving vertically
]


def _background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    base = int(rng.integers(10, 90))
    img = np.full((h, w, 3), base, dtype=np.uint8)
    grad = np.linspace(0, rng.integers(5, 40), w, dtype=np.uint8)[None, :]
    img[..., 0] = np.clip(img[..., 0].astype(int) + grad, 0, 255)
    return img


def _desktop(h: int, w: int, rng, tick: int) -> np.ndarray:
    img = _background(h, w, rng)
    for _ in range(rng.integers(2, 4)):
        r0 = int(rng.integers(0, h // 2))
        c0 = int(rng.integers(0, w // 2))
        bh = int(rng.integers(h // 5, h // 2))
        bw = int(rng.integers(w // 5, w // 2))
        img[r0:r0 + bh, c0:c0 + bw] = int(rng.integers(140, 235))
    cx = int(w * 0.5 + 0.35 * w * np.sin(tick * 0.4))
    cy = int(h * 0.5 + 0.3 * h * np.cos(tick * 0.4))
    img[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5] = 255
    return img


def _terminal(h: int, w: int, rng, tick: int) -> np.ndarray:
    img = np.full((h, w, 3), int(rng.integers(8, 25)), dtype=np.uint8)
    # Fine horizontal text-like runs of varying brightness.
    for row in range(10, h - 10, int(rng.integers(9, 14))):
        length = int(rng.integers(w // 4, w - 20))
        val = int(rng.integers(120, 220))
        img[row:row + 5, 10:10 + length] = val
    if tick % 40 < 20:
        img[h - 20:h - 12, 10:22] = 230  # blinking cursor
    return img


def _media(h: int, w: int, rng, tick: int) -> np.ndarray:
    # Large smooth colour field with a moving object (video-like).
    yy, xx = np.mgrid[0:h, 0:w]
    phase = tick * 0.15
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = (127 + 120 * np.sin(xx / w * 3 + phase)).astype(np.uint8)
    img[..., 1] = (127 + 100 * np.sin(yy / h * 3 + phase)).astype(np.uint8)
    img[..., 2] = 140
    obj_x = int(w * 0.5 + 0.4 * w * np.sin(tick * 0.5))
    img[h // 3:2 * h // 3, max(0, obj_x - 30):obj_x + 30] = 20
    return img


def _dialog(h: int, w: int, rng, tick: int) -> np.ndarray:
    img = _background(h, w, rng) // 2
    bw = int(rng.integers(w // 3, w // 2))
    bh = int(rng.integers(h // 4, h // 3))
    c0 = (w - bw) // 2
    r0 = (h - bh) // 2
    img[r0:r0 + bh, c0:c0 + bw] = int(rng.integers(200, 245))
    img[r0:r0 + 6, c0:c0 + bw] = 60  # title bar
    return img


def _scroll(h: int, w: int, rng, tick: int) -> np.ndarray:
    img = _background(h, w, rng)
    offset = (tick * 8) % h
    band = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(0, h, 12):
        band[row:row + 6] = int(rng.integers(150, 230))
    shifted = np.roll(band, offset, axis=0)
    img = np.clip(img.astype(int) + shifted.astype(int), 0, 255).astype(np.uint8)
    return img


_GENERATORS: dict[str, Callable] = {
    "desktop": _desktop,
    "terminal": _terminal,
    "media": _media,
    "dialog": _dialog,
    "scroll": _scroll,
}


class SceneGenerator:
    """Renders a randomised scene of a given category, advancing per grab()."""

    def __init__(self, category: str, width: int = 320, height: int = 180, seed: int = 0) -> None:
        if category not in _GENERATORS:
            raise ValueError(f"unknown scene category {category!r}")
        self.category = category
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.tick = 0

    def grab(self) -> np.ndarray:
        frame = _GENERATORS[self.category](self.height, self.width, self.rng, self.tick)
        self.tick += 1
        return frame


def sample_states(
    categories: list[str],
    n_per_category: int,
    rows: int = 16,
    cols: int = 24,
    downscale: int = 4,
    seed: int = 0,
    width: int = 320,
    height: int = 180,
):
    """Generate reflex states for the given scene categories.

    Each category uses a distinct RNG seed so a train draw and a val draw of the
    same category differ in sizes, positions and luminance.
    """
    from forge.vision.reflex import ReflexPerception

    states = []
    types = []
    for ci, cat in enumerate(categories):
        gen = SceneGenerator(cat, width=width, height=height, seed=seed + 1000 * ci)
        per = ReflexPerception(rows=rows, cols=cols, downscale=downscale)
        for i in range(n_per_category):
            frame = gen.grab()
            # Occasionally mix in a full-scene swap to exercise the busy class.
            if i % 17 == 0:
                frame = 255 - frame
            states.append(per.perceive(frame, frame_index=i))
            types.append(cat)
    return states, types
