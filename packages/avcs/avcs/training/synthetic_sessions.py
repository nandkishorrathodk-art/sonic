"""Generate synthetic AVCS grounding demonstration sessions.

The real recorder (``vnc_demo_recorder``) produces sessions from a live VNC
desktop, one human click at a time. That needs a running X11 desktop, so it
cannot run in a headless CI box or on a machine without Docker.

This generator emits sessions in the *exact* schema ``discover_demonstrations``
requires, so the whole training pipeline (loader -> dataset -> ``train_grounder``
-> checkpoint) can be exercised without a desktop. It draws a labelled UI
control onto a rendered frame and records the click point on that control.

Provenance is explicit and honest: sessions produced here are marked
``provenance: "synthetic_rendered_controls"`` in the manifest, never the
``recorded_ui_click_demonstrations`` value the real recorder uses. A checkpoint
trained on synthetic sessions must not be presented as a real-demonstration
checkpoint; the generator refuses to be confused with the recorder.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

SCHEMA_VERSION = 1
SYNTHETIC_PROVENANCE = "synthetic_rendered_controls"

# A small, generic control vocabulary. These are UI *roles*, not vendor or
# application names, so the generator stays free of hardcoded products.
_CONTROL_KINDS = (
    ("button", (0, 120, 215), (255, 255, 255)),
    ("input", (255, 255, 255), (90, 90, 90)),
    ("link", (32, 34, 38), (90, 170, 255)),
    ("checkbox", (230, 230, 230), (20, 20, 20)),
    ("tab", (60, 62, 70), (235, 235, 235)),
)

_BACKGROUNDS = (
    (32, 34, 38),
    (245, 246, 248),
    (24, 26, 32),
    (238, 236, 232),
)

_FRAME_SIZE = (640, 400)


def _render_session_frames(
    rng: random.Random,
    session_dir: Path,
    control_count: int,
    frame_size: tuple[int, int] = _FRAME_SIZE,
) -> list[dict]:
    """Render one frame per control and return their click records."""
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    width, height = frame_size
    background = rng.choice(_BACKGROUNDS)
    records: list[dict] = []

    # Place controls on a coarse grid so they never overlap and each click
    # point is unambiguous inside its own control rectangle.
    columns = 3
    cell_w = width // columns
    cell_h = 90
    margin = 18

    for index in range(control_count):
        kind, fill, text_color = rng.choice(_CONTROL_KINDS)
        col = index % columns
        row = index // columns
        x0 = col * cell_w + margin
        y0 = 70 + row * cell_h + margin
        x1 = x0 + cell_w - 2 * margin
        y1 = y0 + 44
        if x1 <= x0 or y1 <= y0 or y1 >= height:
            continue

        image = Image.new("RGB", frame_size, background)
        draw = ImageDraw.Draw(image)
        # A title bar so frames are not a single flat colour.
        draw.rectangle([0, 0, width, 32], fill=(48, 50, 56))
        draw.text((12, 10), "Synthetic UI fixture", fill=(230, 230, 230))
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(120, 120, 120))
        draw.text((x0 + 8, y0 + 14), f"{kind} {index}", fill=text_color)

        frame_name = f"frames/frame_{index:03d}.png"
        image.save(session_dir / frame_name)

        # Click the centre of the drawn control, in [0, 1) normalized space.
        cx = (x0 + x1) / 2.0 / width
        cy = (y0 + y1) / 2.0 / height
        cx = min(max(cx, 0.0), 0.9999)
        cy = min(max(cy, 0.0), 0.9999)

        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "target_label": f"{kind} {index}",
                "frame": frame_name,
                "action": {
                    "type": "mouse_click",
                    "button": "left",
                    "point_norm": [round(cx, 6), round(cy, 6)],
                },
            }
        )
    return records


def write_session(
    root: Path,
    name: str,
    *,
    task: str,
    target_window_title: str,
    control_count: int = 6,
    seed: int = 0,
) -> Path:
    """Write one session directory in the recorder's on-disk schema."""
    session_dir = root / name
    if session_dir.exists():
        raise FileExistsError(f"Session already exists: {session_dir}")

    rng = random.Random(seed)
    records = _render_session_frames(rng, session_dir, control_count)
    if not records:
        raise ValueError(f"Session {name!r} produced no controls")

    samples_path = session_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "stopped",
        "task": task,
        "target_window_title": target_window_title,
        "started_at_utc": now,
        "finished_at_utc": now,
        "sample_count": len(records),
        "capture_scope": "cropped target window only",
        "keyboard_recording": False,
        "network_upload": False,
        "provenance": SYNTHETIC_PROVENANCE,
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return session_dir


def generate_dataset(
    root: str | Path,
    *,
    sessions: int = 12,
    controls_per_session: int = 6,
    seed: int = 0,
) -> Path:
    """Generate ``sessions`` disjoint synthetic sessions under ``root``.

    Each session gets a distinct task and window title so the session-disjoint
    split in ``split_sessions`` can keep train/validation/test apart.
    """
    if sessions < 3:
        raise ValueError("At least 3 sessions are required for a train/val/test split")
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    for index in range(sessions):
        write_session(
            root_path,
            f"session-{index:03d}",
            task=f"synthetic task {index}",
            target_window_title=f"Synthetic fixture window {index}",
            control_count=controls_per_session,
            seed=seed + index,
        )
    return root_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic AVCS grounding demonstration sessions that "
            "match the recorder schema, for pipeline smoke tests and headless "
            "training. Not a substitute for real recorded demonstrations."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Where sessions are written.")
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--controls-per-session", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = generate_dataset(
        args.output_dir,
        sessions=args.sessions,
        controls_per_session=args.controls_per_session,
        seed=args.seed,
    )
    total = args.sessions * args.controls_per_session
    print(
        f"Wrote {args.sessions} synthetic sessions (~{total} click samples) to {root}.\n"
        "Provenance is 'synthetic_rendered_controls'; this is NOT recorded "
        "human demonstration data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
