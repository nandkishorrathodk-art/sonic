"""Reflex perception: raw pixels -> typed visual state, every tick.

This is the System-1 half of the model.  It is *non-generative*: it never
produces free-form text, only a structured :class:`~forge.vision.scene.SceneState`
and a calibrated confidence.  That is what makes a 40-80 ms tick budget
possible -- a single pass of cheap array math over a downscaled luminance
grid, with no language model in the loop.

What it computes, all from pixels:

* layout    -- coarse luminance structure + edge density, pooled into tiles
* saliency  -- per-tile deviation from the frame's mean luminance
* motion    -- frame-to-frame luminance delta, pooled into tiles
* change    -- which tiles actually changed (thresholded motion)
* regions   -- connected busy tiles, classified by shape into panel/band/patch
* overlay   -- a compact, high-contrast change that did not move the whole scene
* novelty   -- how unlike the recent past this frame is (context memory)
* hash      -- perceptual hash of the frame, for cheap scene identity

The thresholds are deliberately explicit constants rather than magic numbers
buried in the code, because the repo's whole point is that claims are
measurable and inspectable.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

from forge.vision.scene import Region, SceneEvent, SceneState

# --- thresholds (tuned on the synthetic desktop; exposed for tests) ---
EDGE_BUSY = 0.18          # tile edge density above this counts as "structured"
SALIENCE_BUSY = 0.35      # tile salience above this counts as "stands out"
CHANGE_TILE = 0.06        # tile motion above this counts as "changed"
SCENE_CHANGE_FRACTION = 0.35  # fraction of changed tiles => whole scene changed
OVERLAY_CHANGE_FRACTION = 0.22  # small change can still be an overlay if compact
OVERLAY_CONTRAST = 0.30   # compact change must also be a strong change to qualify
OVERLAY_MIN_TILES = 3     # below this a change is a cursor/blink, not an overlay
MOTION_TILE = 0.03        # tile motion above this counts as "moving"


def to_gray(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float luminance in [0, 1]."""
    f = frame.astype(np.float32)
    if f.ndim == 2:
        return f / 255.0
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) / 255.0


def downsample(gray: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a luminance image by an integer factor."""
    if factor <= 1:
        return gray
    h = gray.shape[0] // factor * factor
    w = gray.shape[1] // factor * factor
    if h == 0 or w == 0:
        return gray
    return gray[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def pool_tiles(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Average-pool a grid into exactly (rows, cols)."""
    h, w = grid.shape
    if h < rows or w < cols:
        # Pad by edge replication so tiny frames still produce a full grid.
        pad_h = max(0, rows - h)
        pad_w = max(0, cols - w)
        grid = np.pad(grid, ((0, pad_h), (0, pad_w)), mode="edge")
        h, w = grid.shape
    th, tw = h // rows, w // cols
    th = max(th, 1)
    tw = max(tw, 1)
    out = grid[: th * rows, : tw * cols].reshape(rows, th, cols, tw).mean(axis=(1, 3))
    return out


def edge_map(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude (Sobel-lite via finite differences), in [0, 1]."""
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(np.diff(g, axis=1))
    gy[1:, :] = np.abs(np.diff(g, axis=0))
    return np.clip(gx + gy, 0.0, 1.0)


def perceptual_hash(grid: np.ndarray, size: int = 8) -> str:
    """Average-hash of a luminance grid -> 16-char hex string."""
    pooled = pool_tiles(grid, size, size)
    bits = (pooled >= pooled.mean()).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def _connected_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected components of a boolean tile mask (iterative flood fill)."""
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_region(height: int, width: int, rows: int, cols: int) -> str:
    aspect = width / max(height, 1)
    area_frac = (height * width) / float(rows * cols)
    if aspect >= 2.5 and area_frac < 0.35:
        return "band"
    if area_frac >= 0.15 and aspect < 2.5:
        return "panel"
    return "patch"


class ReflexPerception:
    """Turns a stream of raw frames into a stream of typed scene states."""

    def __init__(
        self,
        rows: int = 16,
        cols: int = 24,
        downscale: int = 4,
        context_size: int = 24,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.downscale = downscale
        self._prev_luma: Optional[np.ndarray] = None
        self._prev_grid: Optional[np.ndarray] = None
        self._context: deque[np.ndarray] = deque(maxlen=context_size)
        self._hashes: deque[str] = deque(maxlen=context_size)

    def reset(self) -> None:
        self._prev_luma = None
        self._prev_grid = None
        self._context.clear()
        self._hashes.clear()

    # -- core ---------------------------------------------------------------

    def perceive(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: Optional[float] = None,
    ) -> SceneState:
        t0 = time.perf_counter()
        ts = time.time() if timestamp is None else timestamp
        height, width = frame.shape[0], frame.shape[1]

        gray = to_gray(frame)
        small = downsample(gray, self.downscale)
        luma = pool_tiles(small, self.rows, self.cols)
        edges = pool_tiles(edge_map(small), self.rows, self.cols)
        salience = np.abs(luma - float(luma.mean()))

        # Motion / change relative to the previous frame.
        if self._prev_grid is None:
            motion = np.zeros_like(luma)
            changed = np.zeros_like(luma, dtype=bool)
            global_motion = 0.0
            peak_motion = 0.0
        else:
            motion = np.abs(luma - self._prev_grid)
            changed = motion > CHANGE_TILE
            global_motion = float(motion.mean())
            peak_motion = float(motion.max())

        change_fraction = float(changed.mean())
        scene_changed = change_fraction >= SCENE_CHANGE_FRACTION
        motion_centroid = self._centroid(motion)
        novelty = self._novelty(luma)
        scene_hash = perceptual_hash(luma)

        regions = self._regions(luma, edges, salience)
        overlay = self._overlay(changed, motion, luma, change_fraction)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        state = SceneState(
            frame_index=frame_index,
            timestamp=ts,
            width=width,
            height=height,
            rows=self.rows,
            cols=self.cols,
            luma_grid=luma,
            edge_grid=edges,
            salience_grid=salience,
            motion_grid=motion,
            changed_grid=changed.astype(np.float32),
            brightness=float(luma.mean()),
            contrast=float(luma.std()),
            change_fraction=change_fraction,
            global_motion=global_motion,
            peak_motion=peak_motion,
            novelty=novelty,
            scene_hash=scene_hash,
            regions=regions,
            overlay_suspected=overlay,
            scene_changed=scene_changed,
            motion_centroid=motion_centroid,
            latency_ms=latency_ms,
        )

        self._prev_luma = gray
        self._prev_grid = luma
        self._context.append(luma.flatten())
        self._hashes.append(scene_hash)
        return state

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _centroid(motion: np.ndarray) -> Optional[tuple[float, float]]:
        total = float(motion.sum())
        if total <= 1e-6:
            return None
        rows, cols = motion.shape
        rr = np.arange(rows)[:, None] * motion
        cc = np.arange(cols)[None, :] * motion
        return (float(rr.sum() / total), float(cc.sum() / total))

    def _novelty(self, luma: np.ndarray) -> float:
        if not self._context:
            return 1.0
        flat = luma.flatten()
        ref = np.stack(self._context, axis=0)
        # Cosine distance to the closest remembered frame.
        ref_magnitudes = np.linalg.norm(ref, axis=1, keepdims=True)
        current_magnitude = float(np.linalg.norm(flat))
        if current_magnitude <= 1e-6:
            similarities = np.where(ref_magnitudes[:, 0] <= 1e-6, 1.0, 0.0)
            return float(np.clip(1.0 - float(similarities.max()), 0.0, 1.0))
        ref_norm = ref / (ref_magnitudes + 1e-6)
        cur_norm = flat / (current_magnitude + 1e-6)
        sims = ref_norm @ cur_norm
        return float(np.clip(1.0 - float(sims.max()), 0.0, 1.0))

    def _regions(
        self, luma: np.ndarray, edges: np.ndarray, salience: np.ndarray
    ) -> list[Region]:
        busy = (edges > EDGE_BUSY) | (salience > SALIENCE_BUSY)
        out: list[Region] = []
        for comp in _connected_regions(busy):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            mask = np.zeros_like(luma, dtype=bool)
            for (rr, cc) in comp:
                mask[rr, cc] = True
            out.append(
                Region(
                    row=r0,
                    col=c0,
                    height=h,
                    width=w,
                    mean_luma=float(luma[mask].mean()),
                    edge_density=float(edges[mask].mean()),
                    salience=float(salience[mask].mean()),
                    kind=_classify_region(h, w, self.rows, self.cols),
                )
            )
        out.sort(key=lambda r: r.area, reverse=True)
        return out[:8]

    def _overlay(
        self,
        changed: np.ndarray,
        motion: np.ndarray,
        luma: np.ndarray,
        change_fraction: float,
    ) -> bool:
        if change_fraction <= 1e-6 or change_fraction >= OVERLAY_CHANGE_FRACTION:
            return False
        comps = _connected_regions(changed)
        if not comps:
            return False
        biggest = max(comps, key=len)
        if len(biggest) < OVERLAY_MIN_TILES:
            return False  # a one-tile change is a cursor or a blink
        mask = np.zeros_like(luma, dtype=bool)
        for (rr, cc) in biggest:
            mask[rr, cc] = True
        # Compact block of strong change = something drawn on top.  Contrast is
        # measured on the *change*, not on absolute luminance, because an
        # overlay is defined by differing from what was underneath.
        compact = len(biggest) / float(changed.size) < 0.5
        change_strength = float(motion[mask].mean())
        return compact and change_strength >= OVERLAY_CONTRAST


def events_from_states(
    prev: Optional[SceneState], cur: SceneState, salience_floor: float = 0.05
) -> list[SceneEvent]:
    """Translate the difference between two states into natural-language events.

    Kept separate from perception so the reflex tick stays pure array math and
    the wording can evolve without touching the hot path.
    """
    ev: list[SceneEvent] = []
    if cur.scene_changed:
        ev.append(
            SceneEvent(
                kind="scene_change",
                text="the whole screen changed at once (new view or app switch)",
                timestamp=cur.timestamp,
                frame_index=cur.frame_index,
                salience=cur.change_fraction,
            )
        )
    if cur.overlay_suspected:
        ev.append(
            SceneEvent(
                kind="overlay",
                text="something small and high-contrast appeared over the scene",
                timestamp=cur.timestamp,
                frame_index=cur.frame_index,
                salience=cur.change_fraction,
            )
        )
    if prev is not None and not cur.scene_changed:
        appeared = [r for r in cur.regions if r.salience > 0.5]
        # Local motion is what a human notices, so gate on the peak tile delta,
        # not the frame-wide mean -- a cursor moving over a static desktop has a
        # near-zero mean and a large peak.
        if cur.peak_motion > salience_floor and cur.motion_centroid is not None:
            r, c = cur.motion_centroid
            where = _where(r / cur.rows, c / cur.cols)
            ev.append(
                SceneEvent(
                    kind="move",
                    text=f"movement in the {where} of the screen",
                    timestamp=cur.timestamp,
                    frame_index=cur.frame_index,
                    salience=cur.peak_motion,
                )
            )
        if appeared and cur.novelty > 0.35:
            top = appeared[0]
            where = _where(top.center[0] / cur.rows, top.center[1] / cur.cols)
            ev.append(
                SceneEvent(
                    kind="appear",
                    text=f"a {top.kind} of visual activity is prominent in the {where}",
                    timestamp=cur.timestamp,
                    frame_index=cur.frame_index,
                    salience=top.salience,
                )
            )
    return ev


def _where(frac_r: float, frac_c: float) -> str:
    """Turn normalised coordinates into coarse human wording."""
    if frac_r < 0.34:
        vert = "top"
    elif frac_r < 0.67:
        vert = "middle"
    else:
        vert = "bottom"
    if frac_c < 0.34:
        horiz = "left"
    elif frac_c < 0.67:
        horiz = "centre"
    else:
        horiz = "right"
    if horiz == "centre" and vert == "middle":
        return "centre"
    return f"{vert} {horiz}"
