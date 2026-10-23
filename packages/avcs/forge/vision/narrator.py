"""Live narration: turn a frame stream into human-like commentary for an LLM.

AVCS perceives pixels; the LLM plans.  This module is the *voice* in between.
It watches the same frame stream the reflex layer consumes and speaks only when
something a human would notice has happened -- the way a sighted colleague
describes a screen to someone on a call.

What it can honestly report, and what it cannot
-----------------------------------------------
It never reads text.  There is no OCR and no semantic UI tree here, so it will
never say "the dialog says 'Connection Failed'".  What it *can* report from
pixels alone is:

* scene state -- tone, structure and activity, from the reflex grids
* what changed, in coarse human wording ("a compact block appeared top-centre")
* the dominant colour of a newly appeared region ("a red box")
* the outcome of an action it was told about -- a visible effect, or none

Silence is a first-class output.  While nothing changes the narrator says
nothing, so a planning LLM is not flooded with "still the same screen" every
tick.  This is the difference between a commentary stream and a log.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forge.vision.scene import Region, SceneEvent, SceneState

# --- colour naming ---------------------------------------------------------

# Hue bands (degrees) -> coarse colour word.  Order matters: the first band a
# hue falls into wins, so red wraps around 0/360 and is listed first and last.
_HUE_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 15.0, "red"),
    (15.0, 45.0, "orange"),
    (45.0, 70.0, "yellow"),
    (70.0, 165.0, "green"),
    (165.0, 200.0, "cyan"),
    (200.0, 260.0, "blue"),
    (260.0, 320.0, "purple"),
    (320.0, 345.0, "magenta"),
    (345.0, 360.0, "red"),
)

# A patch this dark or this unsaturated has no meaningful hue.
_ACHROMATIC_SAT = 0.18
_DARK_VALUE = 0.16
_BRIGHT_VALUE = 0.86
_BRIGHT_SAT = 0.12


def _rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Return (hue_degrees, saturation, value) for RGB components in [0, 1]."""
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if delta <= 1e-9:
        return 0.0, 0.0, mx
    sat = delta / mx if mx > 1e-9 else 0.0
    if mx == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif mx == g:
        hue = 60.0 * (((b - r) / delta) + 2.0)
    else:
        hue = 60.0 * (((r - g) / delta) + 4.0)
    return hue % 360.0, sat, mx


def dominant_color_name(rgb: np.ndarray) -> str:
    """Coarse colour word for the mean colour of an RGB patch.

    Achromatic patches resolve to ``black``/``white``/``grey`` rather than a
    hue, so a dark-grey dialog is never described as a colour it is not.
    """
    if not isinstance(rgb, np.ndarray) or rgb.size == 0:
        return "unknown"
    f = rgb.astype(np.float32)
    if f.ndim == 3:
        f = f.reshape(-1, f.shape[-1])
    if f.shape[-1] < 3:
        return "unknown"
    # Ignore near-black background pixels so a coloured box on a dark desktop
    # is named by the box, not by the surrounding void.
    luma = 0.299 * f[:, 0] + 0.587 * f[:, 1] + 0.114 * f[:, 2]
    keep = f[luma > 25.0]
    if keep.shape[0] < max(4, f.shape[0] // 20):
        keep = f
    r, g, b = (float(keep[:, i].mean()) / 255.0 for i in range(3))
    hue, sat, val = _rgb_to_hsv(r, g, b)
    if val <= _DARK_VALUE:
        return "black"
    if sat <= _ACHROMATIC_SAT:
        if val >= _BRIGHT_VALUE:
            return "white"
        return "grey"
    if sat <= _BRIGHT_SAT and val >= _BRIGHT_VALUE:
        return "white"
    for lo, hi, name in _HUE_BANDS:
        if lo <= hue < hi:
            return name
    return "grey"


# --- narration events ------------------------------------------------------


@dataclass
class Narration:
    """One thing the eye says to the brain.

    ``kind`` is the channel the LLM should route on:

    * ``screen``  -- a description of the current view (state, not a change)
    * ``change``  -- something moved or appeared
    * ``action``  -- the result of an action the eye was told about
    * ``alert``   -- something a human would interrupt for (popup, no effect)
    * ``waiting`` -- the screen is busy and the eye is deliberately quiet

    ``text`` is the sentence.  ``urgency`` is ``info`` | ``notice`` | ``alert``
    so a caller can decide whether to interrupt the LLM mid-plan.
    """

    kind: str
    text: str
    urgency: str = "info"
    frame_index: int = 0
    where: str | None = None
    color: str | None = None
    salience: float = 0.0
    # Pixel-space rectangle ``{"x", "y", "w", "h"}`` for the region this line is
    # about, when there is one.  This is the eye's fovea: a caller may crop the
    # frame to it and hand the small patch to a vision model, instead of
    # shipping a whole screenshot.  ``None`` when the line is about the whole
    # screen or nothing in particular.
    focus: dict[str, int] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
            "urgency": self.urgency,
            "frame_index": self.frame_index,
            "salience": round(self.salience, 3),
        }
        if self.where:
            out["where"] = self.where
        if self.color:
            out["color"] = self.color
        if self.focus:
            out["focus"] = dict(self.focus)
        if self.data:
            out["data"] = self.data
        return out


def _where_word(frac_r: float, frac_c: float) -> str:
    """Coarse human position word for normalised (row, col) fractions."""
    vert = "top" if frac_r < 0.34 else "middle" if frac_r < 0.67 else "bottom"
    horiz = "left" if frac_c < 0.34 else "centre" if frac_c < 0.67 else "right"
    if vert == "middle" and horiz == "centre":
        return "centre"
    return f"{vert} {horiz}"


def rgb_tiles(frame: np.ndarray, rows: int, cols: int) -> np.ndarray | None:
    """Block-average an RGB frame into a ``(rows, cols, 3)`` float tile grid."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    f = frame[:, :, :3].astype(np.float32)
    h, w = f.shape[:2]
    th, tw = max(h // rows, 1), max(w // cols, 1)
    f = f[: th * rows, : tw * cols]
    if f.shape[0] < rows or f.shape[1] < cols:
        return None
    return f.reshape(rows, th, cols, tw, 3).mean(axis=(1, 3))


# RGB tile distance (0-1 scale per channel) above which a tile counts as
# colour-changed.  Luma-based reflex change is blind to a saturated box whose
# luminance happens to match its background -- a red dialog on a grey desktop --
# so the narrator compares colour as well.
CHROMA_TILE = 0.10
CHROMA_BLOCK_MIN_TILES = 3


def chromatic_change_mask(
    prev_rgb_tiles: np.ndarray | None,
    cur_rgb_tiles: np.ndarray | None,
    threshold: float = CHROMA_TILE,
) -> np.ndarray | None:
    """Boolean tile mask where mean RGB moved more than ``threshold``.

    Returns ``None`` when either grid is unavailable, so callers fall back to
    the luma-based reflex verdict rather than guessing.
    """
    if prev_rgb_tiles is None or cur_rgb_tiles is None:
        return None
    if prev_rgb_tiles.shape != cur_rgb_tiles.shape:
        return None
    delta = np.abs(cur_rgb_tiles - prev_rgb_tiles).max(axis=-1) / 255.0
    return delta > threshold


def _largest_changed_block(
    changed: np.ndarray, min_tiles: int = 3
) -> tuple[int, int, int, int] | None:
    """Bounding box (r0, c0, r1, c1) of the largest 4-connected changed block."""
    if changed is None or changed.size == 0 or not changed.any():
        return None
    rows, cols = changed.shape
    seen = np.zeros_like(changed, dtype=bool)
    best: list[tuple[int, int]] | None = None
    for r in range(rows):
        for c in range(cols):
            if not changed[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and changed[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if best is None or len(comp) > len(best):
                best = comp
    if best is None or len(best) < min_tiles:
        return None
    rs = [p[0] for p in best]
    cs = [p[1] for p in best]
    return min(rs), min(cs), max(rs), max(cs)


def _region_color(frame: np.ndarray | None, state: SceneState, region: Region) -> str:
    """Dominant colour of a tile-space region, sampled from the raw frame."""
    if frame is None or frame.ndim != 3 or frame.shape[-1] < 3:
        return "unknown"
    y0 = int(region.row / state.rows * state.height)
    y1 = int((region.row + region.height) / state.rows * state.height)
    x0 = int(region.col / state.cols * state.width)
    x1 = int((region.col + region.width) / state.cols * state.width)
    patch = frame[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
    if patch.size == 0:
        return "unknown"
    return dominant_color_name(patch)


def _block_color(
    frame: np.ndarray | None, state: SceneState, block: tuple[int, int, int, int]
) -> str:
    if frame is None or frame.ndim != 3 or frame.shape[-1] < 3:
        return "unknown"
    r0, c0, r1, c1 = block
    y0 = int(r0 / state.rows * state.height)
    y1 = int((r1 + 1) / state.rows * state.height)
    x0 = int(c0 / state.cols * state.width)
    x1 = int((c1 + 1) / state.cols * state.width)
    patch = frame[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
    if patch.size == 0:
        return "unknown"
    return dominant_color_name(patch)


# Padding around a focus rectangle, as a fraction of the block's own size.  A
# dialog is read with a little margin, the way a human's gaze lands on the box
# and a bit around it rather than exactly on its border.
FOCUS_PAD_FRACTION = 0.30
# Never hand back a sliver: a focus region smaller than this is grown to it, so
# a one-tile change does not produce a crop too small to carry any context.
FOCUS_MIN_PIXELS = 48


def focus_rect_pixels(
    state: SceneState,
    block: tuple[int, int, int, int] | None,
    pad_fraction: float = FOCUS_PAD_FRACTION,
    min_pixels: int = FOCUS_MIN_PIXELS,
) -> tuple[int, int, int, int] | None:
    """Convert a tile-space block into a padded pixel-space focus rectangle.

    Returns ``(x0, y0, x1, y1)`` clamped to the frame, or ``None`` when there is
    nothing to focus on.  This is the region a caller would crop and hand to a
    vision model -- the eye's fovea -- so the numbers are pixels, not tiles.
    """
    if block is None or state.rows <= 0 or state.cols <= 0:
        return None
    if state.width <= 0 or state.height <= 0:
        return None
    r0, c0, r1, c1 = block
    x0 = int(c0 / state.cols * state.width)
    x1 = int((c1 + 1) / state.cols * state.width)
    y0 = int(r0 / state.rows * state.height)
    y1 = int((r1 + 1) / state.rows * state.height)

    # Grow to the minimum, centred on the block, so the crop carries context.
    if x1 - x0 < min_pixels:
        mid = (x0 + x1) // 2
        x0, x1 = mid - min_pixels // 2, mid + min_pixels // 2
    if y1 - y0 < min_pixels:
        mid = (y0 + y1) // 2
        y0, y1 = mid - min_pixels // 2, mid + min_pixels // 2

    # Then pad, so a popup is read with a margin.
    pad_x = int((x1 - x0) * pad_fraction)
    pad_y = int((y1 - y0) * pad_fraction)
    x0, y0 = x0 - pad_x, y0 - pad_y
    x1, y1 = x1 + pad_x, y1 + pad_y

    # Clamp into the frame.  An off-screen focus is a bug, not a guess: the
    # caller crops with these numbers, so they must be real pixels.
    x0 = max(0, min(x0, state.width - 1))
    y0 = max(0, min(y0, state.height - 1))
    x1 = max(x0 + 1, min(x1, state.width))
    y1 = max(y0 + 1, min(y1, state.height))
    return x0, y0, x1, y1


def crop_focus(
    frame: np.ndarray | None, focus: dict[str, Any] | None
) -> np.ndarray | None:
    """Crop a frame to a focus rectangle produced by :func:`focus_rect_pixels`.

    ``focus`` is the ``{"x", "y", "w", "h"}`` mapping carried on a
    :class:`Narration`.  Returns ``None`` when the frame is unusable or the
    rectangle does not land inside it, so a caller never sends a bogus crop.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return None
    if not isinstance(focus, dict):
        return None
    try:
        x = int(focus["x"])
        y = int(focus["y"])
        w = int(focus["w"])
        h = int(focus["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame[y0:y1, x0:x1]
    return patch if patch.size else None


class LiveNarrator:
    """Turns a frame stream into sparse, human-like commentary.

    Usage::

        narrator = LiveNarrator()
        for frame in stream:
            state = reflex.perceive(frame)
            events = events_from_states(prev, state)
            line = narrator.observe(frame, state, events)
            if line:            # None == nothing worth saying
                send_to_llm(line.text)

    The narrator never dispatches an action and never reads text.  It is a
    sensor, not a motor.
    """

    def __init__(
        self,
        min_speak_interval_s: float = 0.0,
        describe_scene_on_first_frame: bool = True,
    ) -> None:
        self.min_speak_interval_s = max(0.0, float(min_speak_interval_s))
        self.describe_scene_on_first_frame = bool(describe_scene_on_first_frame)
        self._last_spoken_at: float = -1e9
        self._last_signature: tuple[Any, ...] | None = None
        self._prev_rgb_tiles: np.ndarray | None = None
        self._frames_seen = 0
        self._lines_spoken = 0

    # -- introspection ------------------------------------------------------

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def lines_spoken(self) -> int:
        return self._lines_spoken

    @property
    def silence_ratio(self) -> float:
        """Fraction of frames that produced no line.  High is healthy."""
        if self._frames_seen == 0:
            return 1.0
        return 1.0 - (self._lines_spoken / self._frames_seen)

    def reset(self) -> None:
        self._last_spoken_at = -1e9
        self._last_signature = None
        self._prev_rgb_tiles = None
        self._frames_seen = 0
        self._lines_spoken = 0

    # -- observation --------------------------------------------------------

    def observe(
        self,
        frame: np.ndarray | None,
        state: SceneState,
        events: Sequence[SceneEvent] | None = None,
        active_window: str = "",
    ) -> Narration | None:
        """One line of commentary for this frame, or ``None`` for silence.

        Silence is returned when nothing a human would notice has changed, or
        when the change was already described.
        """
        self._frames_seen += 1
        events = list(events or [])

        # Colour tiles for this frame, and where colour moved relative to the
        # previous frame.  Luma-based reflex change is blind to a saturated box
        # whose luminance matches its background, so the narrator tracks colour
        # separately instead of relying on the reflex verdict alone.
        cur_tiles = rgb_tiles(frame, state.rows, state.cols) if frame is not None else None
        chroma_mask = chromatic_change_mask(self._prev_rgb_tiles, cur_tiles)
        if cur_tiles is not None:
            self._prev_rgb_tiles = cur_tiles

        chroma_block: tuple[int, int, int, int] | None = None
        if chroma_mask is not None:
            chroma_block = _largest_changed_block(chroma_mask, CHROMA_BLOCK_MIN_TILES)

        signature = self._signature(state, active_window, chroma_block)
        first = self._last_signature is None
        if first and not self.describe_scene_on_first_frame:
            self._last_signature = signature
            return None
        if not first and signature == self._last_signature:
            return None
        if not first and (state.timestamp - self._last_spoken_at) < self.min_speak_interval_s:
            return None

        line = self._phrase(state, events, frame, active_window, first, chroma_block)
        if line is None:
            return None
        self._last_signature = signature
        self._last_spoken_at = state.timestamp
        self._lines_spoken += 1
        return line

    # -- action feedback ----------------------------------------------------

    def action_result(
        self,
        action: str,
        target: str = "",
        visible_effect: bool | None = None,
        state: SceneState | None = None,
        latency_ms: float = 0.0,
        detail: str = "",
    ) -> Narration:
        """Phrase the outcome of an action the caller already dispatched.

        ``visible_effect`` is the caller's verified verdict -- ``True`` when the
        frame demonstrably changed, ``False`` when it did not, ``None`` when the
        caller could not tell.  An unverified action is reported as unverified;
        it is never upgraded to success.
        """
        where = ""
        if state is not None and state.motion_centroid is not None:
            r, c = state.motion_centroid
            where = _where_word(r / state.rows, c / state.cols)

        label = (target or action or "action").strip()
        frame_index = state.frame_index if state is not None else 0

        if visible_effect is True:
            text = f"{action} on '{label}': the screen responded"
            text += f" ({where})" if where else ""
            text += "."
            if state is not None and state.overlay_suspected:
                text += " A compact overlay appeared as a result."
            return Narration(
                kind="action",
                text=text,
                urgency="info",
                frame_index=frame_index,
                where=where or None,
                salience=state.peak_motion if state is not None else 0.0,
                data={"action": action, "target": label, "verified": True, "latency_ms": round(latency_ms, 2)},
            )

        if visible_effect is False:
            text = f"{action} on '{label}': no visible effect -- the screen did not change."
            if detail:
                text += f" {detail}"
            return Narration(
                kind="alert",
                text=text,
                urgency="alert",
                frame_index=frame_index,
                data={"action": action, "target": label, "verified": False, "latency_ms": round(latency_ms, 2)},
            )

        text = f"{action} on '{label}': dispatched, effect not verified."
        return Narration(
            kind="action",
            text=text,
            urgency="notice",
            frame_index=frame_index,
            data={"action": action, "target": label, "verified": None, "latency_ms": round(latency_ms, 2)},
        )

    def waiting(self, what: str = "the screen is still busy") -> Narration:
        """An explicit 'hold on' line, for when a caller wants to say so."""
        return Narration(kind="waiting", text=f"Still {what}.", urgency="info")

    # -- phrasing -----------------------------------------------------------

    def _signature(
        self,
        state: SceneState,
        active_window: str,
        chroma_block: tuple[int, int, int, int] | None = None,
    ) -> tuple[Any, ...]:
        """What counts as 'worth saying again'.

        Coarse on purpose: a two-tile cursor blink must not produce a line, but
        a new overlay, a whole-screen transition, or a coloured block appearing
        must.
        """
        return (
            active_window,
            state.scene_hash,
            state.overlay_suspected,
            state.scene_changed,
            round(state.peak_motion, 1),
            len(state.regions),
            chroma_block,
        )

    def _phrase(
        self,
        state: SceneState,
        events: list[SceneEvent],
        frame: np.ndarray | None,
        active_window: str,
        first: bool,
        chroma_block: tuple[int, int, int, int] | None = None,
    ) -> Narration | None:
        parts: list[str] = []
        urgency = "info"
        kind = "screen"
        where: str | None = None
        color: str | None = None
        # The tile block this line is *about*, if any.  Used to compute the
        # pixel-space focus rectangle a caller can crop to.
        focus_block: tuple[int, int, int, int] | None = None

        # 1. Where we are.  The window title is the only semantic hint the
        #    pixel path is given, and it comes from the caller, not from OCR.
        if active_window:
            parts.append(f"Active window: {active_window}.")
        elif first:
            parts.append("Screen visible.")

        # 2. What kind of scene it is.
        tone = "dark" if state.brightness < 0.25 else "bright" if state.brightness > 0.75 else "mid-tone"
        flat = "flat" if state.contrast < 0.08 else "detailed"
        parts.append(f"A {tone}, {flat} screen.")

        # 3. The event that matters most, in a human's order of noticing.
        block = _largest_changed_block(
            state.changed_grid.astype(bool) if state.changed_grid is not None else None
        )
        # A coloured block appearing is the single most useful thing to report:
        # it is what a human calls out first ("arre, ek laal box aaya").  It can
        # be invisible to the luma reflex when the new colour's luminance matches
        # its background, so the chroma block takes priority over the reflex
        # overlay verdict.
        if chroma_block is not None and not state.scene_changed:
            named = _block_color(frame, state, chroma_block)
            where = _where_word(
                ((chroma_block[0] + chroma_block[2]) / 2.0) / state.rows,
                ((chroma_block[1] + chroma_block[3]) / 2.0) / state.cols,
            )
            focus_block = chroma_block
            if named not in ("unknown", "grey", "black", "white"):
                kind = "alert"
                urgency = "alert"
                color = named
                parts.append(
                    f"A {named} box appeared in the {where} on top of the scene"
                    " -- something popped up."
                )
            else:
                kind = "change"
                parts.append(
                    f"A new block appeared in the {where} on top of the scene."
                )
        elif state.scene_changed:
            kind = "change"
            parts.append("The whole view changed at once -- like switching apps or pages.")
        elif state.overlay_suspected:
            kind = "alert"
            urgency = "alert"
            color = _block_color(frame, state, block) if block else "unknown"
            where = None
            focus_block = block
            if block:
                where = _where_word(
                    ((block[0] + block[2]) / 2.0) / state.rows,
                    ((block[1] + block[3]) / 2.0) / state.cols,
                )
            pos = f" {where}" if where else ""
            col = f" A {color} box" if color not in ("unknown", "grey") else " A small, high-contrast box"
            parts.append(f"{col} appeared{pos} on top of the scene -- something popped up.")
        elif state.peak_motion > 0.10:
            kind = "change"
            where = None
            if state.motion_centroid is not None:
                r, c = state.motion_centroid
                where = _where_word(r / state.rows, c / state.cols)
                # Focus on where the motion is, so a caller can look there.
                rr = int(r)
                cc = int(c)
                focus_block = (max(0, rr - 1), max(0, cc - 1), rr + 1, cc + 1)
            parts.append(f"There is clear movement{' in the ' + where if where else ''}.")
        elif state.peak_motion > 0.03:
            kind = "change"
            parts.append("Something is moving gently on screen.")
        elif first:
            parts.append("The screen is essentially still.")
        else:
            # A non-first frame with no notable event is not worth a line.
            return None

        # 4. Structure, so the planner knows the layout is busy or bare.
        if state.regions:
            kinds: list[str] = []
            for region in state.regions[:3]:
                if region.kind not in kinds:
                    kinds.append(region.kind)
            if kinds:
                parts.append(f"Structure reads as {', '.join(kinds)}.")

        # 5. Colour of the most prominent region, when it is a real colour.
        if color is None and state.regions:
            top = state.regions[0]
            named = _region_color(frame, state, top)
            if named not in ("unknown", "grey", "black", "white"):
                color = named
                r, c = top.center
                parts.append(
                    f"The most prominent block is {named}, in the {_where_word(r / state.rows, c / state.cols)}."
                )

        # 6. Typed events, appended verbatim -- they are already human wording.
        for ev in events[:2]:
            text = getattr(ev, "text", "")
            if text:
                parts.append(f"{getattr(ev, 'kind', 'event').replace('_', ' ')}: {text}.")

        if not parts:
            return None
        focus: dict[str, int] | None = None
        rect = focus_rect_pixels(state, focus_block)
        if rect is not None:
            x0, y0, x1, y1 = rect
            focus = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
        return Narration(
            kind=kind,
            text=" ".join(parts),
            urgency=urgency,
            frame_index=state.frame_index,
            where=where,
            color=color,
            salience=max(state.peak_motion, state.change_fraction),
            focus=focus,
        )
