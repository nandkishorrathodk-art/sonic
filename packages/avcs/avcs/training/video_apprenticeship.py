"""Apprenticeship from watched video: turn a screen recording into AVCS data.

A human learns a desktop tool by *watching* someone use it. This module gives
AVCS the same input path: point it at a screen-recording video (a YouTube
download, an OBS capture, or a VNC session recording) and it produces training
material without anyone hand-labelling frames.

What is honestly derivable from raw video
-----------------------------------------
A video is *unlabelled pixels*. It does not contain ground-truth UI categories,
affordance masks, or click targets. Claiming otherwise would be fabrication.
Two signals ARE genuinely present and are extracted here:

1. **Frames** - raw screen pixels. These feed the *self-supervised* cortex
   objective (temporal predictive coding): "predict the next frame's latent".
   That objective needs no labels at all, so video is a legitimate source.
2. **Cursor trajectory** - the mouse pointer is a small, high-contrast,
   temporally-coherent blob. Tracking its motion recovers *where the
   demonstrator moved and clicked*. Pauses + direction reversals are candidate
   click events, which become (frame, point) apprenticeship samples.

What is NOT derivable, and is therefore never invented here:
- the semantic label of the clicked element ("Submit", "File menu"),
- the UI category of a frame,
- dense affordance masks.

Those remain ``None``/``unlabeled`` on the records, and downstream training
must use only the objectives that can consume them. A record that has a cursor
point but no label is still useful for click-location pretraining; it is not
useful for goal-conditioned grounding, and the code says so rather than
silently substituting a guessed label.

Frame extraction uses ``imageio-ffmpeg`` (a self-contained ffmpeg binary), so
no system ffmpeg is required. Network access is the caller's concern; this
module only reads a local file the caller already obtained.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from forge.vision.servo import locate_cursor

try:  # imageio-ffmpeg ships a bundled ffmpeg binary; prefer it when present.
    import imageio_ffmpeg
except Exception:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None


class VideoIngestionError(RuntimeError):
    """Raised when a video cannot be read; never returns fake frames."""


@dataclass(frozen=True)
class FrameRecord:
    """One decoded frame plus any honestly-derivable signal."""

    index: int
    timestamp_s: float
    path: Path
    cursor_point_norm: tuple[float, float] | None = None
    cursor_confidence: float = 0.0


@dataclass
class VideoIngestResult:
    """Everything extracted from one video, with provenance."""

    source: str
    width: int
    height: int
    fps: float
    duration_s: float
    frames: list[FrameRecord] = field(default_factory=list)
    click_events: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "click_events": self.click_events,
            "notes": self.notes,
            "provenance": "watched_video_unlabeled_pixels",
        }


def ffmpeg_executable() -> str:
    """Return an ffmpeg binary path, failing honestly when none is available."""
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # pragma: no cover - defensive
            pass
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    raise VideoIngestionError(
        "No ffmpeg available. Install the 'imageio-ffmpeg' package (bundles a "
        "binary) or a system ffmpeg to ingest video."
    )


def probe_video(path: str | Path) -> dict:
    """Read width/height/fps/duration without decoding every frame."""
    video = Path(path)
    if not video.is_file():
        raise VideoIngestionError(f"Video file does not exist: {video}")
    ffmpeg = ffmpeg_executable()
    # ffmpeg prints stream info to stderr; parse the first video stream line.
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video)],
        capture_output=True,
        text=True,
    )
    info: dict = {}
    for line in proc.stderr.splitlines():
        stripped = line.strip()
        if "Video:" not in stripped or "fps" not in stripped:
            continue
        info["codec_line"] = stripped
        for token in stripped.split(","):
            token = token.strip()
            if "x" in token and token.split("x")[0].isdigit():
                parts = token.split(" ")[0].split("x")
                if len(parts) == 2 and parts[1].split(".")[0].isdigit():
                    info["width"] = int(parts[0])
                    info["height"] = int(parts[1].split(".")[0])
            if token.endswith("fps"):
                try:
                    info["fps"] = float(token.split(" ")[0])
                except ValueError:
                    pass
        break
    for line in proc.stderr.splitlines():
        if line.strip().startswith("Duration:"):
            stamp = line.split("Duration:")[1].split(",")[0].strip()
            try:
                h, m, s = stamp.split(":")
                info["duration_s"] = round(int(h) * 3600 + int(m) * 60 + float(s), 3)
            except ValueError:
                pass
            break
    if "width" not in info or "height" not in info:
        raise VideoIngestionError(
            f"Could not determine video dimensions for {video}; is it a video file?"
        )
    return info


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    *,
    fps: float = 1.0,
    max_frames: int = 300,
    scale: Optional[int] = None,  # noqa: UP045
) -> list[Path]:
    """Decode frames to PNG at ``fps`` (frames per second of *source* time).

    ``fps`` below the source rate subsamples (good for training diversity and
    disk use). ``max_frames`` bounds the run so a long video cannot fill the
    disk. Returns the written paths in order.
    """
    if fps <= 0:
        raise VideoIngestionError("fps must be positive")
    if max_frames <= 0:
        raise VideoIngestionError("max_frames must be positive")
    video = Path(path)
    if not video.is_file():
        raise VideoIngestionError(f"Video file does not exist: {video}")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_executable()

    vf = f"fps={fps}"
    if scale:
        # Keep aspect ratio; -2 makes the other dimension even for the encoder.
        vf += f",scale={int(scale)}:-2"
    pattern = str(target / "frame_%06d.png")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", vf,
        "-frames:v", str(int(max_frames)),
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoIngestionError(
            f"ffmpeg frame extraction failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:300]}"
        )
    frames = sorted(target.glob("frame_*.png"))
    if not frames:
        raise VideoIngestionError(
            f"ffmpeg produced no frames for {video}; the video may be corrupt "
            "or shorter than one frame interval."
        )
    return frames


def _read_png_rgb(path: Path) -> np.ndarray:
    """Read a PNG to an (H, W, 3) uint8 array without a heavy image dep."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception as exc:  # pragma: no cover - defensive
        raise VideoIngestionError(f"Could not read frame {path}: {exc}") from exc


def detect_cursor_point(
    frame: np.ndarray,
    *,
    patch: int = 15,
) -> Optional[tuple[float, float, float]]:  # noqa: UP045
    """Locate the mouse cursor as the most salient small high-contrast blob.

    Thin wrapper over :func:`forge.vision.servo.locate_cursor`, which owns the
    implementation. It lives there (in the torch-free servo module) so the live
    eye-hand loop and offline video ingestion share one detector instead of two
    that could drift apart. The only difference here is the error type, kept for
    backwards compatibility with this module's callers and tests.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise VideoIngestionError("Expected an (H, W, 3) RGB frame")
    return locate_cursor(frame, patch=patch)


def _trajectory_click_events(
    points: Sequence[tuple[float, float] | None],  # noqa: UP007
    *,
    pause_threshold: float = 0.01,
    min_confidence: float = 0.25,
) -> list[int]:
    """Indices where the cursor *stopped* - the best video-only click proxy.

    A click is a momentary pause: the pointer arrives, holds for at least one
    sampled frame, then leaves. Direction reversal alone is too noisy (drags and
    aim corrections look identical), so we key on "held within a small radius
    across consecutive samples". These are candidate events, never confirmed
    clicks - the caller must treat them as weak supervision.
    """
    events: list[int] = []
    for i in range(1, len(points) - 1):
        prev, cur, nxt = points[i - 1], points[i], points[i + 1]
        if prev is None or cur is None or nxt is None:
            continue
        held = (
            abs(cur[0] - prev[0]) <= pause_threshold
            and abs(cur[1] - prev[1]) <= pause_threshold
            and abs(nxt[0] - cur[0]) <= pause_threshold
            and abs(nxt[1] - cur[1]) <= pause_threshold
        )
        if held:
            events.append(i)
    # Collapse runs of held frames into a single event at the run's midpoint.
    collapsed: list[int] = []
    for idx in events:
        if collapsed and idx - collapsed[-1] <= 2:
            continue
        collapsed.append(idx)
    return collapsed


def ingest_video(
    path: str | Path,
    out_dir: str | Path,
    *,
    fps: float = 1.0,
    max_frames: int = 300,
    scale: Optional[int] = None,  # noqa: UP045
    detect_cursor: bool = True,
    min_cursor_confidence: float = 0.25,
) -> VideoIngestResult:
    """Extract frames + cursor-derived click candidates from a watched video.

    ``detect_cursor`` can be disabled to skip the (per-frame) salience scan when
    only raw frames are wanted for self-supervised training.
    """
    info = probe_video(path)
    frame_paths = extract_frames(
        path, out_dir, fps=fps, max_frames=max_frames, scale=scale
    )
    result = VideoIngestResult(
        source=str(path),
        width=int(info.get("width", 0)),
        height=int(info.get("height", 0)),
        fps=float(info.get("fps", 0.0)),
        duration_s=float(info.get("duration_s", 0.0)),
        notes=[
            "Frames are unlabeled screen pixels (no category/affordance labels).",
            "Cursor points and click events are weak video-derived signals.",
        ],
    )
    sample_interval = 1.0 / fps if fps > 0 else 1.0
    cursor_points: list[tuple[float, float] | None] = []  # noqa: UP007
    for idx, frame_path in enumerate(frame_paths):
        point = None
        confidence = 0.0
        if detect_cursor:
            rgb = _read_png_rgb(frame_path)
            detected = detect_cursor_point(rgb)
            if detected is not None and detected[2] >= min_cursor_confidence:
                point = (detected[0], detected[1])
                confidence = detected[2]
        cursor_points.append(point)
        result.frames.append(
            FrameRecord(
                index=idx,
                timestamp_s=round(idx * sample_interval, 3),
                path=frame_path,
                cursor_point_norm=point,
                cursor_confidence=confidence,
            )
        )

    for event_idx in _trajectory_click_events(cursor_points):
        record = result.frames[event_idx]
        result.click_events.append(
            {
                "frame_index": event_idx,
                "timestamp_s": record.timestamp_s,
                "point_norm": list(record.cursor_point_norm or (0.0, 0.0)),
                "confidence": record.cursor_confidence,
                "source": "video_cursor_pause",
                "confirmed": False,
            }
        )
    return result


def write_ingest_manifest(result: VideoIngestResult, out_dir: str | Path) -> Path:
    """Persist the ingest result next to the frames (provenance for training)."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "video_ingest.json"
    payload = result.to_dict()
    payload["frames"] = [
        {
            "index": f.index,
            "timestamp_s": f.timestamp_s,
            "path": f.path.name,
            "cursor_point_norm": list(f.cursor_point_norm) if f.cursor_point_norm else None,
            "cursor_confidence": f.cursor_confidence,
        }
        for f in result.frames
    ]
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def iter_frame_batches(
    frames: Sequence[FrameRecord],
    batch_size: int,
    *,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Yield normalized (B, 3, H, W) float32 batches from extracted frames.

    Pairs are returned in temporal order so the cortex's predictive-coding
    objective can consume consecutive frames as (t, t+1) without labels.
    """
    if batch_size <= 0:
        raise VideoIngestionError("batch_size must be positive")
    buffer: list[np.ndarray] = []
    for record in frames:
        rgb = _read_png_rgb(record.path)
        # Resize to the model's expected geometry with nearest neighbour to keep
        # this dependency-light; the cortex retina crops foveal/peripheral views.
        resized = _nearest_resize(rgb, width, height)
        buffer.append(resized.astype(np.float32).transpose(2, 0, 1) / 255.0)
        if len(buffer) >= batch_size:
            yield np.stack(buffer, axis=0)
            buffer = []
    if buffer:
        yield np.stack(buffer, axis=0)


def _nearest_resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize to (height, width) using numpy only."""
    src_h, src_w = rgb.shape[:2]
    if (src_h, src_w) == (height, width):
        return rgb
    y_idx = (np.arange(height) * src_h // max(height, 1)).clip(0, src_h - 1)
    x_idx = (np.arange(width) * src_w // max(width, 1)).clip(0, src_w - 1)
    return rgb[y_idx][:, x_idx]


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045
    """CLI: turn a watched video into AVCS training frames + a manifest."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Ingest a screen-recording video into AVCS training data. Produces "
            "frames and cursor-derived click candidates. Frames are unlabeled "
            "pixels; no UI category or affordance labels are invented."
        )
    )
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for extracted frames and video_ingest.json.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Source frames per second to sample (lower = fewer, more diverse frames).",
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--scale",
        type=int,
        default=0,
        help="Optional width to downscale frames to (0 keeps source size).",
    )
    parser.add_argument(
        "--no-cursor",
        action="store_true",
        help="Skip cursor detection (frames-only, for self-supervised pretraining).",
    )
    args = parser.parse_args(argv)

    try:
        result = ingest_video(
            args.video,
            args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            scale=args.scale or None,
            detect_cursor=not args.no_cursor,
        )
        manifest = write_ingest_manifest(result, args.output_dir)
    except VideoIngestionError as exc:
        print(f"Ingestion failed without fabricating data: {exc}", file=sys.stderr)
        return 1
    print(
        f"Ingested {result.frame_count} frame(s) from {result.source} "
        f"({result.width}x{result.height} @ {result.fps} fps). "
        f"{len(result.click_events)} cursor-pause click candidate(s). "
        f"Manifest: {manifest}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
