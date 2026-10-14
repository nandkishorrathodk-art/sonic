"""Typed scene state produced from raw pixels.

Everything here is derived from luminance grids and frame-to-frame differences
computed on the pixel buffer alone.  No OCR, no element coordinates from the
OS, no accessibility tree, no structured UI detection: the same information a
human eye receives, quantised into a coarse grid.

The dataclasses are intentionally plain so they serialise straight to JSON for
the live dashboard and for tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Region:
    """A connected block of tiles the reflex layer considers visually busy.

    ``row``/``col``/``height``/``width`` are in tile units, not pixels, because
    the reflex layer never commits to pixel coordinates -- it reasons about
    coarse spatial structure.
    """

    row: int
    col: int
    height: int
    width: int
    mean_luma: float
    edge_density: float
    salience: float
    kind: str  # "panel" | "band" | "patch"

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def center(self) -> tuple[float, float]:
        return (self.row + self.height / 2.0, self.col + self.width / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "height": self.height,
            "width": self.width,
            "mean_luma": round(self.mean_luma, 3),
            "edge_density": round(self.edge_density, 4),
            "salience": round(self.salience, 3),
            "kind": self.kind,
        }


@dataclass
class SceneState:
    """A single perceived frame of the screen.

    All grids are ``(rows, cols)`` float arrays in [0, 1] unless noted.  This is
    the unit the reflex loop produces every tick and the unit the describer and
    the learned encoder consume.
    """

    frame_index: int
    timestamp: float
    width: int
    height: int
    rows: int
    cols: int

    luma_grid: np.ndarray
    edge_grid: np.ndarray
    salience_grid: np.ndarray
    motion_grid: np.ndarray
    changed_grid: np.ndarray

    brightness: float
    contrast: float
    change_fraction: float
    global_motion: float
    peak_motion: float
    novelty: float
    scene_hash: str

    regions: list[Region] = field(default_factory=list)
    overlay_suspected: bool = False
    scene_changed: bool = False
    motion_centroid: Optional[tuple[float, float]] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Compact JSON-ready view (grids quantised to keep payloads small)."""

        def grid(g: np.ndarray) -> list[list[float]]:
            return [[round(float(v), 3) for v in row] for row in g]

        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "width": self.width,
            "height": self.height,
            "rows": self.rows,
            "cols": self.cols,
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "change_fraction": round(self.change_fraction, 4),
            "global_motion": round(self.global_motion, 4),
            "peak_motion": round(self.peak_motion, 4),
            "novelty": round(self.novelty, 3),
            "scene_hash": self.scene_hash,
            "overlay_suspected": self.overlay_suspected,
            "scene_changed": self.scene_changed,
            "motion_centroid": (
                [round(self.motion_centroid[0], 2), round(self.motion_centroid[1], 2)]
                if self.motion_centroid
                else None
            ),
            "regions": [r.to_dict() for r in self.regions],
            "latency_ms": round(self.latency_ms, 2),
            "luma_grid": grid(self.luma_grid),
            "motion_grid": grid(self.motion_grid),
            "salience_grid": grid(self.salience_grid),
        }


@dataclass
class SceneEvent:
    """A discrete thing that happened on screen, in natural language.

    Events are emitted only when the reflex layer crosses a threshold, so the
    expensive describer runs on change rather than on every frame.
    """

    kind: str  # "appear" | "vanish" | "move" | "scene_change" | "overlay" | "idle"
    text: str
    timestamp: float
    frame_index: int
    salience: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "timestamp": round(self.timestamp, 4),
            "frame_index": self.frame_index,
            "salience": round(self.salience, 3),
        }
