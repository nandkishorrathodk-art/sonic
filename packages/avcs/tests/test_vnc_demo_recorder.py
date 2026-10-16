import io
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from avcs.training.demonstrations import discover_demonstrations
from avcs.training.vnc_demo_recorder import (
    VNCClickDemoRecorder,
    _is_left_raw_button_press,
    _parse_raw_button_press,
    _parse_window_listing,
    _window_id,
)


def test_parses_window_geometry_and_titles_without_app_specific_mapping():
    windows = _parse_window_listing(
        "0x03e00007  0  10  20  640  480  host  Generic Test Window\n"
        "invalid row\n"
    )

    assert windows == [
        {
            "window_id": 65011719,
            "title": "Generic Test Window",
            "rect": (10, 20, 650, 500),
        }
    ]


def test_normalizes_decimal_active_window_id():
    assert _window_id("65011719\n") == 65011719


@pytest.mark.parametrize(
    ("header", "detail", "expected"),
    [
        ("EVENT type 15 (RawButtonPress)", "detail: 1", True),
        ("EVENT type 16 (RawButtonRelease)", "detail: 1", False),
        ("EVENT type 15 (RawButtonPress)", "detail: 3", False),
        ("EVENT type 4 (ButtonPress)", "detail: 1", False),
    ],
)
def test_only_left_raw_button_press_is_recordable(header, detail, expected):
    assert _is_left_raw_button_press(header, detail) is expected


def test_raw_button_press_uses_coordinates_from_same_xinput_event():
    assert _parse_raw_button_press(
        [
            "EVENT type 15 (RawButtonPress)\n",
            "    device: 6 (10)\n",
            "    detail: 1\n",
            "    root: 1654.00/723.00\n",
            "    event: 512.00/370.00\n",
        ]
    ) == (1654, 723)


def test_raw_button_press_without_root_coordinates_is_not_grounded():
    assert _parse_raw_button_press(
        [
            "EVENT type 15 (RawButtonPress)\n",
            "    detail: 1\n",
            "    valuators:\n",
            "    0: 512.00\n",
            "    raw values:\n",
            "    0: 512.00\n",
        ]
    ) is None


def test_xinput_raw_click_falls_back_to_pointer_position(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.vnc_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    recorder = VNCClickDemoRecorder(
        window_title="Test",
        task="Click a control",
        output_dir=tmp_path / "demo",
        confirm_isolated_window=True,
        capture_factory=object(),
    )
    recorder._event_process = SimpleNamespace(
        stdout=io.StringIO(
            "EVENT type 15 (RawButtonPress)\n"
            "    device: 6 (10)\n"
            "    detail: 1\n"
            "    valuators:\n"
            "    0: 512.00\n"
            "    raw values:\n"
            "    0: 512.00\n"
            "EVENT type 16 (RawButtonRelease)\n"
            "    detail: 1\n"
        )
    )
    pointer_samples = []
    monkeypatch.setattr(
        recorder,
        "_record_current_pointer_click",
        lambda: pointer_samples.append("queried-at-click"),
    )

    recorder._read_xinput_events()

    assert pointer_samples == ["queried-at-click"]


def test_recorder_requires_explicit_isolated_window_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.vnc_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")

    with pytest.raises(ValueError, match="confirm-isolated-window"):
        VNCClickDemoRecorder(
            window_title="Test",
            task="Click the test button",
            output_dir=tmp_path / "demo",
            confirm_isolated_window=False,
        )


def test_recorder_prearms_labels_supplied_on_command_line(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.vnc_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")

    recorder = VNCClickDemoRecorder(
        window_title="Test",
        task="Click the test button",
        output_dir=tmp_path / "demo",
        confirm_isolated_window=True,
        target_labels=[" Test button ", "Submit"],
        capture_factory=object(),
    )

    assert recorder._labels.get_nowait() == "Test button"
    assert recorder._labels.get_nowait() == "Submit"


def test_recorder_rejects_empty_command_line_target_label(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.vnc_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")

    with pytest.raises(ValueError, match="Target labels"):
        VNCClickDemoRecorder(
            window_title="Test",
            task="Click the test button",
            output_dir=tmp_path / "demo",
            confirm_isolated_window=True,
            target_labels=["  "],
            capture_factory=object(),
        )


def test_rejected_click_does_not_consume_prearmed_label(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.vnc_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    session_dir = tmp_path / "session"
    (session_dir / "frames").mkdir(parents=True)
    recorder = VNCClickDemoRecorder(
        window_title="Test",
        task="Click the visible button",
        output_dir=session_dir,
        confirm_isolated_window=True,
        target_labels=["Visible button"],
        capture_factory=object(),
    )
    recorder._samples_file = (session_dir / "samples.jsonl").open(
        "x", encoding="utf-8"
    )
    recorder._latest_snapshot = (
        np.full((20, 30, 3), (255, 0, 0), dtype=np.uint8),
        (10, 20, 40, 40),
        time.monotonic(),
    )
    monkeypatch.setattr(
        recorder,
        "_target_window_matches_snapshot",
        lambda rect: rect == (10, 20, 40, 40),
    )
    pointer = [5, 5]
    recorder._record_click(tuple(pointer))
    assert recorder._labels.qsize() == 1
    assert recorder._sample_count == 0

    pointer[:] = [25, 30]
    recorder._record_click(tuple(pointer))
    recorder._samples_file.close()

    assert recorder._labels.empty()
    assert recorder._sample_count == 1
    assert len(list((session_dir / "frames").glob("*.png"))) == 1


def test_click_is_recorded_if_target_loses_focus_after_click(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.vnc_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    session_dir = tmp_path / "session"
    (session_dir / "frames").mkdir(parents=True)
    recorder = VNCClickDemoRecorder(
        window_title="Test",
        task="Launch the selected application",
        output_dir=session_dir,
        confirm_isolated_window=True,
        target_labels=["Application result"],
        capture_factory=object(),
    )
    recorder._samples_file = (session_dir / "samples.jsonl").open(
        "x", encoding="utf-8"
    )
    recorder._latest_snapshot = (
        np.full((20, 30, 3), (255, 0, 0), dtype=np.uint8),
        (10, 20, 40, 40),
        time.monotonic(),
    )
    monkeypatch.setattr(recorder, "_active_window", lambda: None)
    monkeypatch.setattr(
        recorder,
        "_target_window_matches_snapshot",
        lambda rect: rect == (10, 20, 40, 40),
    )
    recorder._record_click((25, 30))
    recorder._samples_file.close()

    records = (session_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["target_label"] == "Application result"


def test_recorder_writes_dataset_compatible_rgb_sample(monkeypatch, tmp_path):
    monkeypatch.setattr("avcs.training.vnc_demo_recorder.sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":99")
    session_dir = tmp_path / "session"
    recorder = VNCClickDemoRecorder(
        window_title="Test",
        task="Click the visible button",
        output_dir=session_dir,
        confirm_isolated_window=True,
        capture_factory=object(),
    )
    session_dir.mkdir()
    (session_dir / "frames").mkdir()
    recorder._started_at_utc = "2026-10-13T00:00:00+00:00"
    recorder._samples_file = (session_dir / "samples.jsonl").open(
        "x", encoding="utf-8"
    )
    recorder._latest_snapshot = (
        np.full((20, 30, 3), (255, 0, 0), dtype=np.uint8),
        (10, 20, 40, 40),
        time.monotonic(),
    )
    monkeypatch.setattr(
        recorder,
        "_target_window_matches_snapshot",
        lambda rect: rect == (10, 20, 40, 40),
    )
    recorder._labels.put("Visible button")

    recorder._record_click((25, 30))
    recorder._samples_file.close()
    recorder._write_manifest("stopped")

    sessions = discover_demonstrations(tmp_path)
    records = next(iter(sessions.values()))
    assert len(records) == 1
    assert records[0].target_label == "Visible button"
    assert records[0].point_norm == (0.5, 0.5)
    with records[0].frame_path.open("rb") as frame_file:
        from PIL import Image

        frame = np.asarray(Image.open(frame_file).convert("RGB"))
    assert tuple(frame[0, 0]) == (255, 0, 0)
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 1
