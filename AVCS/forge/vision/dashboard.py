"""Self-contained live dashboard for the screen-vision loop.

Writes one HTML file that renders the perceived scene as a graphical grid:
luminance structure, motion heat, salience, and the current description, all in
inline SVG/CSS with no external assets -- the same convention the rest of the
repo's report generator follows.

The dashboard is deliberately dependency-free so it opens in any browser,
including on the machine that is being watched.
"""
from __future__ import annotations

import html
import json
import os
import time
from typing import Optional

import numpy as np

from forge.vision.encoder import Understanding
from forge.vision.scene import SceneEvent, SceneState

_CSS = """
:root { --bg:#0b0f14; --panel:#121821; --ink:#e6edf3; --dim:#8b98a5;
        --hot:#f97316; --cool:#38bdf8; --good:#4ade80; --grid:#1e2833; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
header { padding:14px 18px; border-bottom:1px solid var(--grid);
         display:flex; gap:24px; align-items:baseline; flex-wrap:wrap; }
h1 { font-size:15px; margin:0; letter-spacing:.06em; }
.stat { color:var(--dim); } .stat b { color:var(--ink); }
main { padding:18px; display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:12px; }
.card h2 { font-size:12px; margin:0 0 8px; color:var(--dim); font-weight:600;
           text-transform:uppercase; letter-spacing:.08em; }
.grid svg { width:100%; height:auto; display:block; image-rendering:pixelated; }
.desc { grid-column:1 / -1; }
.desc p { margin:0 0 8px; }
.events li { margin:2px 0; color:var(--dim); }
.events b { color:var(--hot); }
.badge { display:inline-block; padding:1px 7px; border-radius:99px;
         background:#1c2733; color:var(--cool); font-size:11px; }
"""


def _svg_grid(grid: np.ndarray, palette: str = "luma", cell: int = 22) -> str:
    rows, cols = grid.shape
    w, h = cols * cell, rows * cell
    rects = []
    g = np.clip(grid.astype(np.float32), 0.0, 1.0)
    for r in range(rows):
        for c in range(cols):
            v = float(g[r, c])
            if palette == "luma":
                shade = int(20 + v * 225)
                fill = f"rgb({shade},{shade},{shade})"
            elif palette == "motion":
                fill = f"rgb({int(30 + v * 225)},{int(40 - v * 30)},{int(60 - v * 40)})"
            else:  # salience
                fill = f"rgb({int(20 + v * 235)},{int(120 + v * 80)},{int(200 - v * 150)})"
            rects.append(f'<rect x="{c * cell}" y="{r * cell}" width="{cell}" height="{cell}" fill="{fill}"/>')
    return f'<svg viewBox="0 0 {w} {h}" shape-rendering="crispEdges">{"".join(rects)}</svg>'


def _regions_svg(state: SceneState, cell: int = 22) -> str:
    w, h = state.cols * cell, state.rows * cell
    boxes = []
    for reg in state.regions:
        x, y = reg.col * cell, reg.row * cell
        bw, bh = reg.width * cell, reg.height * cell
        color = {"panel": "#38bdf8", "band": "#4ade80", "patch": "#f97316"}.get(reg.kind, "#e6edf3")
        boxes.append(
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<text x="{x + 4}" y="{y + 14}" fill="{color}" font-size="11">{html.escape(reg.kind)}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}">{"".join(boxes)}</svg>'


def render_dashboard(
    path: str,
    state: SceneState,
    understanding: Optional[Understanding] = None,
    events: Optional[list[SceneEvent]] = None,
    stats: Optional[dict] = None,
    description: str = "",
    history: Optional[list[dict]] = None,
) -> str:
    events = events or []
    stats = stats or {}
    history = history or []

    stat_bits = "".join(
        f'<span class="stat">{html.escape(str(k))} <b>{html.escape(str(v))}</b></span>'
        for k, v in stats.items()
    )
    und_badge = ""
    if understanding is not None:
        und_badge = (
            f'<span class="badge">{html.escape(understanding.scene_type)} '
            f'· conf {understanding.confidence:.2f}</span>'
        )
    ev_items = "".join(
        f'<li><b>{html.escape(e.kind)}</b> {html.escape(e.text)}</li>' for e in events
    ) or "<li>no events this tick</li>"

    spark = ""
    if history:
        vals = [h.get("change_fraction", 0.0) for h in history]
        spark = _sparkline(vals)

    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sentinel Screen Vision</title>
<style>{_CSS}</style></head><body>
<header>
  <h1>SENTINEL · LIVE SCREEN VISION</h1>
  {und_badge}
  {stat_bits}
</header>
<main>
  <div class="card"><h2>Luminance structure (layout)</h2>
    <div class="grid">{_svg_grid(state.luma_grid, "luma")}</div></div>
  <div class="card"><h2>Detected regions</h2>
    <div class="grid">{_regions_svg(state)}</div></div>
  <div class="card"><h2>Motion heat</h2>
    <div class="grid">{_svg_grid(state.motion_grid, "motion")}</div></div>
  <div class="card"><h2>Salience</h2>
    <div class="grid">{_svg_grid(state.salience_grid, "salience")}</div></div>
  <div class="card desc"><h2>What is happening</h2>
    <p>{html.escape(description or "(waiting for description)")}</p>
    <h2>Events</h2><ul class="events">{ev_items}</ul>
    {spark}
    <p class="stat">updated {time.strftime('%H:%M:%S')} · frame {state.frame_index}
       · reflex {state.latency_ms:.1f} ms · hash <b>{state.scene_hash}</b></p>
  </div>
</main></body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def _sparkline(values: list[float], w: int = 640, h: int = 60) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * (w - 10) + 5
        y = h - 5 - (v - lo) / span * (h - 15)
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<h2>Change over time</h2><svg viewBox="0 0 {w} {h}">'
        f'<polyline fill="none" stroke="#f97316" stroke-width="2" points="{" ".join(pts)}"/></svg>'
    )


def write_state_json(path: str, payload: dict) -> str:
    """Sidecar JSON so a web front-end can poll without reparsing the HTML."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path
