"""Tests for video apprenticeship: honest AVCS learning from watched video.

These tests build real videos with the bundled ffmpeg and verify the module
extracts real frames and cursor-derived click candidates, and that it refuses
to invent labels it does not have.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from avcs.training.video_apprenticeship import (
    VideoIngestionError,
    detect_cursor_point,
    extract_frames,
    ingest_video,
    iter_frame_batches,
    probe_video,
    write_ingest_manifest,
)
from PIL import Image


def _has_imageio() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401

        return True
    except Exception:
        return False


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _make_video(tmp_path: Path, positions, *, fps=4) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i, (x, y) in enumerate(positions):
        frame = np.full((180, 320, 3), 40, dtype=np.uint8)
        frame[:, :, 1] = 45
        frame[y - 3 : y + 4, x - 3 : x + 4] = (255, 255, 255)
        Image.fromarray(frame).save(src / f"f{i:04d}.png")
    out = tmp_path / "video.mp4"
    subprocess.run(
        [
            _ffmpeg(), "-y", "-framerate", str(fps),
            "-i", str(src / "f%04d.png"),
            "-pix_fmt", "yuv420p", str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out


@pytest.mark.skipif(not _has_imageio(), reason="imageio-ffmpeg unavailable")
def test_probe_reads_dimensions_and_fps(tmp_path):
    video = _make_video(tmp_path, [(20, 20), (60, 40), (100, 60)])
    info = probe_video(video)
    assert info["width"] == 320
    assert info["height"] == 180
    assert info["fps"] == 4.0


@pytest.mark.skipif(not _has_imageio(), reason="imageio-ffmpeg unavailable")
def test_probe_missing_file_fails_loudly(tmp_path):
    with pytest.raises(VideoIngestionError):
        probe_video(tmp_path / "nope.mp4")


@pytest.mark.skipif(not _has_imageio(), reason="imageio-ffmpeg unavailable")
def test_extract_frames_writes_real_pngs(tmp_path):
    video = _make_video(tmp_path, [(10, 10)] * 8)
    frames = extract_frames(video, tmp_path / "frames", fps=4.0, max_frames=20)
    assert len(frames) >= 2
    assert all(p.suffix == ".png" for p in frames)
    assert all(p.stat().st_size > 0 for p in frames)


@pytest.mark.skipif(not _has_imageio(), reason="imageio-ffmpeg unavailable")
def test_max_frames_bounds_extraction(tmp_path):
    video = _make_video(tmp_path, [(10, 10)] * 12)
    frames = extract_frames(video, tmp_path / "frames", fps=4.0, max_frames=5)
    assert len(frames) == 5


@pytest.mark.skipif(not _has_imageio(), reason="imageio-ffmpeg unavailable")
def test_ingest_extracts_cursor_and_click_candidates(tmp_path):
    # Cursor moves, then holds for several frames, then moves away.
    positions = [
        (20, 20), (60, 40), (120, 80), (200, 140),
        (200, 140), (200, 140), (200, 140),
        (120, 60), (60, 30), (20, 10),
    ]
    video = _make_video(tmp_path, positions, fps=4)
    result = ingest_video(video, tmp_path / "out", fps=4.0, max_frames=50)
    assert result.frame_count == len(positions)
    labeled = [f for f in result.frames if f.cursor_point_norm]
    assert len(labeled) == len(positions)
    # The held run must surface as at least one candidate click.
    assert len(result.click_events) >= 1
    event = result.click_events[0]
    assert event["confirmed"] is False
    assert event["source"] == "video_cursor_pause"
    # The detected point must be near the true held cursor location (0.625, 0.78).
    x, y = event["point_norm"]
    assert abs(x - 200 / 320) < 0.12
    assert abs(y - 140 / 180) < 0.12


@pytest.mark.skipif(not _has_imageio(), reason="imageio-ffmpeg unavailable")
def test_manifest_is_written_and_json_serializable(tmp_path):
    video = _make_video(tmp_path, [(30, 30), (90, 60), (150, 90)])
    result = ingest_video(video, tmp_path / "out", fps=4.0, max_frames=20)
    manifest = write_ingest_manifest(result, tmp_path / "out")
    payload = json.loads(manifest.read_text())
    assert payload["frame_count"] == result.frame_count
    assert payload["provenance"] == "watched_video_unlabeled_pixels"
    # No fabricated labels leak into the manifest.
    for frame in payload["frames"]:
        assert "category" not in frame
        assert "affordance" not in frame


def test_detect_cursor_point_returns_none_on_flat_frame():
    flat = np.full((120, 200, 3), 50, dtype=np.uint8)
    assert detect_cursor_point(flat) is None


def test_detect_cursor_point_locates_bright_blob():
    frame = np.full((120, 200, 3), 30, dtype=np.uint8)
    frame[70:76, 150:156] = (255, 255, 255)
    detected = detect_cursor_point(frame)
    assert detected is not None
    x, y, conf = detected
    assert abs(x - 153 / 200) < 0.06
    assert abs(y - 73 / 120) < 0.06
    assert conf > 0.5


def test_detect_cursor_rejects_bad_shape():
    with pytest.raises(VideoIngestionError):
        detect_cursor_point(np.zeros((10, 10), dtype=np.uint8))


@pytest.mark.skipif(not _has_imageio(), reason="imageio-ffmpeg unavailable")
def test_iter_frame_batches_yields_normalized_batches(tmp_path):
    video = _make_video(tmp_path, [(10, 10)] * 6)
    result = ingest_video(video, tmp_path / "out", fps=4.0, max_frames=20)
    batches = list(
        iter_frame_batches(result.frames, 4, width=64, height=36)
    )
    assert batches
    for batch in batches:
        assert batch.shape[1:] == (3, 36, 64)
        assert batch.min() >= 0.0 and batch.max() <= 1.0


def test_ffmpeg_missing_raises_honest_error(monkeypatch):
    import avcs.training.video_apprenticeship as mod

    monkeypatch.setattr(mod, "imageio_ffmpeg", None)
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    with pytest.raises(VideoIngestionError):
        mod.ffmpeg_executable()


def test_ingest_rejects_missing_file(tmp_path):
    with pytest.raises(VideoIngestionError):
        ingest_video(tmp_path / "missing.mp4", tmp_path / "out")
