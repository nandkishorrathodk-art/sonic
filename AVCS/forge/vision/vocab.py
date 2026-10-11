"""A small closed vocabulary the description head predicts over.

The point of a closed vocabulary is that the description becomes a *learned
prediction* rather than a branch of hand-written ``if`` statements.  The model
outputs a multi-label distribution over these words (multi-label, not softmax,
because several apply at once: a screen can be ``dark`` and ``detailed`` and
``moving``).  A separate, deterministic renderer turns the predicted word set
into a grammatical sentence.

This is still non-generative in the sense that matters for latency: one forward
pass, no autoregressive sampling, no language model.  It is *not* a fixed
template: which words fire is decided by the trained network.

The word set is deliberately small and closed so it can be trained on a CPU in
seconds and evaluated exactly.  It is not a claim of open-domain captioning.
"""
from __future__ import annotations

from forge.vision.scene import SceneState

# Order defines the index; never reorder without retraining (indices are labels).
VOCAB: list[str] = [
    "screen",      # 0  always
    "dark",        # 1
    "bright",      # 2
    "mid-tone",    # 3
    "flat",        # 4
    "detailed",    # 5
    "still",       # 6
    "moving",      # 7
    "whole",       # 8
    "changed",     # 9
    "overlay",     # 10
    "small",       # 11
    "contrast",    # 12
    "movement",    # 13
    "top",         # 14
    "bottom",      # 15
    "left",        # 16
    "right",       # 17
    "centre",      # 18
    "panel",       # 19
    "band",        # 20
    "patch",       # 21
    "structured",  # 22
    "text",        # 23  fine dense structure (text-like), no OCR
    "idle",        # 24
    "active",      # 25
    "busy",        # 26
]
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}

# Regions with edge density above this carry fine, dense structure.  That is a
# pixel-level signature shared by text, icons and noise -- it says "something
# intricate is here", not "I read it".
TEXT_EDGE = 0.34


def _where_word(frac_r: float, frac_c: float) -> str:
    if frac_r < 0.34:
        vert = "top"
    elif frac_r < 0.67:
        return "centre" if 0.34 <= frac_c < 0.67 else ("left" if frac_c < 0.34 else "right")
    else:
        vert = "bottom"
    horiz = "left" if frac_c < 0.34 else ("right" if frac_c >= 0.67 else "centre")
    return f"{vert} {horiz}".split()[-1] if horiz == "centre" else f"{vert} {horiz}"


def _tones(state: SceneState) -> list[str]:
    out = []
    if state.brightness < 0.25:
        out.append("dark")
    elif state.brightness > 0.75:
        out.append("bright")
    else:
        out.append("mid-tone")
    out.append("flat" if state.contrast < 0.08 else "detailed")
    return out


def _activity(state: SceneState) -> list[str]:
    if state.scene_changed:
        return ["whole", "changed"]
    if state.overlay_suspected:
        return ["overlay", "small", "contrast"]
    if state.peak_motion > 0.10:
        return ["moving", "movement"]
    if state.peak_motion > 0.03:
        return ["moving"]
    return ["still"]


def _structure(state: SceneState) -> list[str]:
    out: list[str] = []
    if state.regions:
        out.append("structured")
        for r in state.regions[:3]:
            if r.kind not in out:
                out.append(r.kind)
        if any(r.edge_density > TEXT_EDGE for r in state.regions):
            out.append("text")
    return out


def _direction(state: SceneState) -> list[str]:
    if state.motion_centroid is None or state.peak_motion <= 0.03:
        return []
    r, c = state.motion_centroid
    fr, fc = r / state.rows, c / state.cols
    vert = "top" if fr < 0.34 else ("bottom" if fr >= 0.67 else None)
    horiz = "left" if fc < 0.34 else ("right" if fc >= 0.67 else None)
    words = [w for w in (vert, horiz) if w]
    return words or ["centre"]


def tokens_for_state(state: SceneState, scene_type: str | None = None) -> list[str]:
    """The target word set for a frame (the teacher's labels).

    Used to train the head; at inference the head predicts these directly from
    the grids, and the renderer phrases whatever it predicted.
    """
    words = ["screen"]
    words += _tones(state)
    words += _activity(state)
    words += _structure(state)
    words += _direction(state)
    if scene_type:
        words.append(scene_type)
    # de-dup, keep vocab order for stable targets
    seen = set()
    ordered = []
    for w in words:
        if w in VOCAB_INDEX and w not in seen:
            seen.add(w)
            ordered.append(w)
    return ordered


def tokens_to_vector(tokens: list[str]) -> list[float]:
    vec = [0.0] * len(VOCAB)
    for t in tokens:
        if t in VOCAB_INDEX:
            vec[VOCAB_INDEX[t]] = 1.0
    return vec


def render_tokens(tokens: list[str]) -> str:
    """Deterministic sentence from a predicted word set.

    Phrasing lives here, once.  It never invents a fact the word set does not
    contain, so a wrong description is a wrong *prediction*, not a wrong
    template.
    """
    s = set(tokens)
    if "screen" not in s and not tokens:
        return "(no description)"

    parts: list[str] = []
    tone = next((t for t in ("dark", "bright", "mid-tone") if t in s), "mid-tone")
    detail = "flat" if "flat" in s else "detailed"
    parts.append(f"A {tone}, {detail} screen.")

    kinds = [k for k in ("panel", "band", "patch") if k in s]
    if kinds:
        txt = " Fine text-like structure is present." if "text" in s else ""
        parts.append(f"Structure reads as {', '.join(kinds)}.{txt}")
    elif "structured" in s:
        parts.append("The screen is visually structured.")

    if "whole" in s and "changed" in s:
        parts.append("The whole view changed at once.")
    elif "overlay" in s:
        parts.append("A small, high-contrast overlay sits on top.")
    elif "movement" in s:
        dirs = [d for d in ("top", "bottom", "left", "right", "centre") if d in s]
        where = f" in the {' '.join(dirs)}" if dirs else ""
        parts.append(f"Clear movement{where}.")
    elif "moving" in s:
        parts.append("Something is moving gently.")
    elif "still" in s:
        parts.append("The screen is essentially still.")

    st = next((t for t in ("idle", "active", "busy", "overlay") if t in s), None)
    if st:
        parts.append(f"State reads as {st}.")
    return " ".join(parts)
