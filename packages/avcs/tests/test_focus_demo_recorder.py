"""Tests for the AVCS focus-demonstration recorder.

The recorder exists so a focus checkpoint can actually be trained from real
data.  These tests prove the isolation/safety gates AND the data contract: the
bytes it writes must be exactly what ``discover_focus_demonstrations`` and the
focus trainer consume.  That round trip is the defect this module closes.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from avcs.training.focus_dataset import discover_focus_demonstrations
from avcs.training.focus_demo_recorder import (
    VNCFocusDemoRecorder,
    build_focus_record,
    cropped_point,
    parse_focus_target,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("search field:focused", ("search field", True)),
        ("Cancel:not-focused", ("Cancel", False)),
        ("Submit:YES", ("Submit", True)),
        ("Submit: n ", ("Submit", False)),
        ("a:b label:true", ("a:b label", True)),
    ],
)
def test_parse_focus_target_accepts_explicit_ground_truth(raw, expected):
    assert parse_focus_target(raw) == expected


@pytest.mark.parametrize("raw", ["no-colon", "label:maybe", ":focused", "x:"])
def test_parse_focus_target_rejects_ambiguous_or_empty(raw):
    with pytest.raises(ValueError):
        parse_focus_target(raw)


def test_cropped_point_maps_into_window_and_rejects_outside():
    assert cropped_point((15, 25), (10, 20, 40, 40)) == (5, 5)
    assert cropped_point((5, 5), (10, 20, 40, 40)) is None
    assert cropped_point((40, 25), (10, 20, 40, 40)) is None


def test_build_focus_record_has_dataset_schema_and_no_text():
    record = build_focus_record(
        target_label="search field",
        frame_path="frames/abc.png",
        point_px=(3, 4),
        focused=True,
        recorded_at_utc="2026-10-27T00:00:00+00:00",
    )
    assert record["schema_version"] == 1
    assert record["point_px"] == [3, 4]
    assert record["focused"] is True
    # A focus demonstration never captures keystrokes or typed text.
    assert set(record) == {
        "schema_version",
        "target_label",
        "frame",
        "point_px",
        "focused",
        "recorded_at_utc",
    }


def test_build_focus_record_requires_boolean_label():
    with pytest.raises(TypeError):
        build_focus_record(
            target_label="x",
            frame_path="frames/a.png",
            point_px=(1, 1),
            focused="yes",  # type: ignore[arg-type]
            recorded_at_utc="t",
        )


def test_recorder_requires_explicit_isolated_window_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.focus_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    with pytest.raises(ValueError, match="confirm-isolated-window"):
        VNCFocusDemoRecorder(
            window_title="Test",
            task="Label focus controls",
            output_dir=tmp_path / "demo",
            confirm_isolated_window=False,
        )


def test_recorder_prearms_targets_supplied_on_command_line(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.focus_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    recorder = VNCFocusDemoRecorder(
        window_title="Test",
        task="Label focus controls",
        output_dir=tmp_path / "demo",
        confirm_isolated_window=True,
        targets=[(" search field ", True), ("Cancel", False)],
        capture_factory=object(),
    )
    assert recorder._targets.get_nowait() == ("search field", True)
    assert recorder._targets.get_nowait() == ("Cancel", False)


def test_recorder_rejects_empty_prearmed_label(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.focus_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    with pytest.raises(ValueError, match="1 to 160"):
        VNCFocusDemoRecorder(
            window_title="Test",
            task="Label focus controls",
            output_dir=tmp_path / "demo",
            confirm_isolated_window=True,
            targets=[("  ", True)],
            capture_factory=object(),
        )


def _armed_recorder(monkeypatch, tmp_path, targets):
    monkeypatch.setattr("avcs.training.focus_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    session_dir = tmp_path / "session"
    (session_dir / "frames").mkdir(parents=True)
    recorder = VNCFocusDemoRecorder(
        window_title="Test",
        task="Label focus controls",
        output_dir=session_dir,
        confirm_isolated_window=True,
        targets=targets,
        capture_factory=object(),
    )
    recorder._records_file = (session_dir / "records.jsonl").open("x", encoding="utf-8")
    recorder._latest_snapshot = (
        np.full((20, 30, 3), (255, 0, 0), dtype=np.uint8),
        (10, 20, 40, 40),
        (10, 20, 40, 40),
        time.monotonic(),
    )
    monkeypatch.setattr(
        recorder,
        "_target_window_matches_snapshot",
        lambda rect: rect == (10, 20, 40, 40),
    )
    return recorder, session_dir


def test_rejected_click_does_not_consume_armed_target(monkeypatch, tmp_path):
    recorder, session_dir = _armed_recorder(
        monkeypatch, tmp_path, [("search field", True)]
    )
    # Pointer outside the target window must be ignored without consuming a label.
    recorder._record_click((5, 5))
    assert recorder._targets.qsize() == 1
    assert recorder._sample_count == 0

    recorder._record_click((25, 30))
    recorder._records_file.close()
    assert recorder._targets.empty()
    assert recorder._sample_count == 1
    assert len(list((session_dir / "frames").glob("*.png"))) == 1


def test_stale_frame_click_is_ignored(monkeypatch, tmp_path):
    recorder, _ = _armed_recorder(monkeypatch, tmp_path, [("search field", True)])
    recorder._latest_snapshot = (
        np.full((20, 30, 3), (255, 0, 0), dtype=np.uint8),
        (10, 20, 40, 40),
        (10, 20, 40, 40),
        time.monotonic() - 10.0,
    )
    recorder._record_click((25, 30))
    recorder._records_file.close()
    assert recorder._sample_count == 0
    assert recorder._targets.qsize() == 1


def test_recorder_output_round_trips_into_focus_dataset(monkeypatch, tmp_path):
    """The written session must be consumable by the real focus dataset loader."""
    recorder, session_dir = _armed_recorder(
        monkeypatch, tmp_path, [("search field", True), ("Cancel", False)]
    )
    recorder._record_click((25, 30))  # inside -> focused sample
    recorder._record_click((11, 21))  # inside -> not-focused sample
    recorder._records_file.close()
    recorder._started_at_utc = "2026-10-27T00:00:00+00:00"
    recorder._write_manifest("complete")

    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sample_count"] == 2
    assert manifest["focused_sample_count"] == 1
    assert manifest["keyboard_recording"] is False
    assert manifest["network_upload"] is False

    # The real trainer-side loader must accept the session unchanged.
    sessions = discover_focus_demonstrations(tmp_path)
    records = sessions["session"]
    assert len(records) == 2
    assert records[0].focused is True
    assert records[0].point_px == (15, 10)
    assert records[1].focused is False
    assert records[1].point_px == (1, 1)


def test_incomplete_session_is_rejected_by_loader(monkeypatch, tmp_path):
    recorder, session_dir = _armed_recorder(
        monkeypatch, tmp_path, [("search field", True)]
    )
    recorder._record_click((25, 30))
    recorder._records_file.close()
    recorder._started_at_utc = "2026-10-27T00:00:00+00:00"
    recorder._write_manifest("failed")

    with pytest.raises(ValueError, match="not complete"):
        discover_focus_demonstrations(tmp_path)
