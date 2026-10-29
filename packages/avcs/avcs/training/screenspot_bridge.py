"""Bridge ScreenSpot-v2 human click annotations into the recorder dataset format.

The goal-conditioned grounder trains only on sessions produced by the local
click recorder (`avcs.training.demo_recorder`), because that is the only source
whose supervision is a real operator click on a real screen. ScreenSpot-v2 is a
public research dataset of the same kind of supervision: each row pairs a real
screenshot with a natural-language instruction and the human-annotated bounding
box of the control the instruction refers to.

This module converts those rows into recorder-format sessions so the existing,
unmodified trainer can consume them. It does not weaken the trainer's checks;
it satisfies them with real annotations.

Honest limits, stated so a reader does not over-claim the result:

* The instruction is used as the ``target_label`` and the box centre as the
  click point. ScreenSpot labels a target *region*; a point at its centre is a
  faithful but not identical substitute for a recorded click.
* The dataset is third-party screen content, not this workstation. A checkpoint
  trained on it generalises across common desktop/web/mobile UI but has not
  observed the operator's own applications.
* Provenance is recorded as the dataset's own identity, never as
  ``recorded_ui_click_demonstrations``. The trainer and the runtime loader both
  gate on that string, so a ScreenSpot checkpoint is intentionally NOT accepted
  as a production grounder. It is a baseline for measuring progress, and the
  honest path to a production grounder remains real recorder sessions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

SCREENSPOT_PROVENANCE = "screenspot_v2_human_click_annotations"

# ScreenSpot names its sources; each becomes its own task/window group so the
# trainer's leakage-free split keeps a source on one side of the boundary.
_SPLIT_FILES = {
    "desktop": "screenspot_desktop_v2.json",
    "web": "screenspot_web_v2.json",
    "mobile": "screenspot_mobile_v2.json",
}

# A frame is referenced by every row that annotates it. Sessions are grouped by
# source, so a repeated frame stays inside one session.
_MAX_TARGET_LABEL_BYTES = 96


def _target_label(instruction: str) -> str:
    """Return a query string the grounder's byte encoder accepts.

    ``encode_queries`` rejects an empty query and one longer than
    ``MAX_QUERY_BYTES``; the dataset contains a few very long instructions, so
    an over-long label is truncated on a UTF-8 boundary rather than dropped.
    """
    label = " ".join(instruction.split()).strip()
    if not label:
        raise ValueError("instruction is empty after normalisation")
    encoded = label.encode("utf-8")
    if len(encoded) <= _MAX_TARGET_LABEL_BYTES:
        return label
    truncated = encoded[:_MAX_TARGET_LABEL_BYTES].decode("utf-8", errors="ignore")
    return truncated.strip()


def _bbox_centre_norm(
    bbox: Sequence[float], width: int, height: int
) -> tuple[float, float] | None:
    """Convert a ScreenSpot ``[x, y, w, h]`` box to a normalised centre point.

    Returns ``None`` for a box that is degenerate or lies outside the image, so
    a malformed annotation is skipped instead of becoming a wrong label.
    """
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    try:
        x, y, w, h = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        return None
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    if not (0.0 <= cx < 1.0 and 0.0 <= cy < 1.0):
        return None
    return cx, cy


def convert_screenspot(
    *,
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    link_frames: bool = True,
) -> dict[str, object]:
    """Write recorder-format sessions for every ScreenSpot split found.

    Each source becomes one session directory containing ``frames/``,
    ``samples.jsonl`` and ``manifest.json``. The manifest carries the real
    dataset identity as its provenance, so nothing downstream can mistake this
    for operator-recorded data.
    """
    from PIL import Image

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {annotations_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[dict[str, object]] = []
    skipped_total = 0

    for source, filename in sorted(_SPLIT_FILES.items()):
        annotation_path = annotations_dir / filename
        if not annotation_path.is_file():
            print(f"skip {source}: {annotation_path} is absent", file=sys.stderr)
            continue
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError(f"{annotation_path} must contain a JSON list")

        # One session per screenshot. The trainer splits train/validation/test
        # by (task, window title) group, so a per-image session guarantees the
        # same screenshot never appears on two sides of the split, even when it
        # carries several instructions.
        by_image: dict[str, list[dict[str, object]]] = {}
        skipped = 0
        for row in rows:
            if not isinstance(row, dict):
                skipped += 1
                continue
            image_name = row.get("img_filename") or row.get("img_path")
            instruction = row.get("instruction") or ""
            bbox = row.get("bbox")
            if not isinstance(image_name, str) or not image_name:
                skipped += 1
                continue
            source_image = images_dir / image_name
            if not source_image.is_file():
                skipped += 1
                continue
            try:
                label = _target_label(str(instruction))
            except ValueError:
                skipped += 1
                continue
            try:
                with Image.open(source_image) as image:
                    width, height = image.size
            except OSError:
                skipped += 1
                continue
            if not isinstance(bbox, (list, tuple)):
                skipped += 1
                continue
            point = _bbox_centre_norm(bbox, width, height)
            if point is None:
                skipped += 1
                continue
            by_image.setdefault(image_name, []).append(
                {
                    "schema_version": 1,
                    "task": f"screenpot {source} instruction grounding",
                    "target_label": label,
                    "frame": "frames/current.png",
                    "action": {
                        "type": "mouse_click",
                        "button": "left",
                        "point_norm": [round(point[0], 6), round(point[1], 6)],
                    },
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                }
            )

        for image_name, records in sorted(by_image.items()):
            image_stem = Path(image_name).stem
            session_dir = output_dir / f"screenspot_{source}_{image_stem}"
            frames_dir = session_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            source_image = images_dir / image_name
            target_frame = frames_dir / "current.png"
            if not target_frame.exists():
                if link_frames:
                    try:
                        target_frame.hardlink_to(source_image)
                    except OSError:
                        target_frame.write_bytes(source_image.read_bytes())
                else:
                    target_frame.write_bytes(source_image.read_bytes())

            (session_dir / "samples.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            manifest = {
                "schema_version": 1,
                "status": "stopped",
                "capture_scope": "cropped target window only",
                "keyboard_recording": False,
                "network_upload": False,
                "target_window_title": f"ScreenSpot-v2 {source} {image_stem}",
                "task": f"screenpot {source} instruction grounding",
                "sample_count": len(records),
                "provenance": SCREENSPOT_PROVENANCE,
                "source_dataset": "OS-Copilot/ScreenSpot-v2",
                "converted_at_utc": datetime.now(UTC).isoformat(),
            }
            (session_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            sessions.append(
                {
                    "session": str(session_dir),
                    "source": source,
                    "samples": len(records),
                    "skipped": 0,
                }
            )
        skipped_total += skipped

    summary = {
        "provenance": SCREENSPOT_PROVENANCE,
        "output_dir": str(output_dir),
        "sessions": sessions,
        "total_samples": sum(int(item["samples"]) for item in sessions),
        "total_skipped": skipped_total,
        "note": (
            "Dataset-derived baseline. Provenance is intentionally not "
            "'recorded_ui_click_demonstrations', so the runtime will not accept "
            "this checkpoint as a production grounder."
        ),
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--copy-frames",
        action="store_true",
        help="Copy frames instead of hardlinking (use across filesystems).",
    )
    args = parser.parse_args(argv)
    summary = convert_screenspot(
        annotations_dir=args.annotations_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        link_frames=not args.copy_frames,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
