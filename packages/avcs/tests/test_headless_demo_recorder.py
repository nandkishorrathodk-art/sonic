"""Tests for the headless demonstration recorder.

The live path needs Xvfb/xdotool/wmctrl/xinput and a real X server, so it is
exercised only when those tools are present (they are skipped otherwise, never
faked). The pure helpers are always tested.
"""

from __future__ import annotations

import shutil

import pytest
from avcs.training.headless_demo_recorder import (
    HeadlessSessionSpec,
    _default_specs,
    discover_terminal,
    record_demonstrations,
)

_HAS_X11_TOOLING = all(
    shutil.which(name) for name in ("Xvfb", "xdotool", "wmctrl", "xinput")
)


class TestSessionSpec:
    def test_labels_are_derived_from_click_points(self):
        spec = HeadlessSessionSpec(
            window_title="W", task="click the button", click_points=[(1, 2), (3, 4)]
        )
        assert spec.resolved_labels() == [
            "click the button - control A",
            "click the button - control B",
        ]

    def test_explicit_labels_take_precedence(self):
        spec = HeadlessSessionSpec(
            window_title="W",
            task="t",
            click_points=[(1, 2)],
            labels=["custom label"],
        )
        assert spec.resolved_labels() == ["custom label"]

    def test_default_specs_are_six_distinct_groups(self):
        specs = _default_specs()
        assert len(specs) == 6
        titles = {s.window_title for s in specs}
        tasks = {s.task for s in specs}
        assert len(titles) == 6, "window titles must be unique for leakage-free splits"
        assert len(tasks) == 6, "tasks must be unique for leakage-free splits"
        for spec in specs:
            assert spec.click_points, "every spec must click something real"


class TestTerminalDiscovery:
    def test_missing_requested_app_raises(self):
        with pytest.raises(RuntimeError, match="not installed"):
            discover_terminal("definitely-not-a-real-binary-xyz")

    def test_found_terminal_is_returned(self):
        if shutil.which("xterm") is None:
            pytest.skip("no terminal emulator installed")
        assert discover_terminal("xterm").endswith("xterm")


class TestRecordingErrors:
    def test_no_specs_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="At least one session spec"):
            record_demonstrations(tmp_path / "out", [])


@pytest.mark.skipif(
    not _HAS_X11_TOOLING, reason="headless X11 tooling (Xvfb/xdotool/wmctrl) absent"
)
@pytest.mark.skipif(
    shutil.which("openbox") is None
    and shutil.which("fluxbox") is None
    and shutil.which("metacity") is None
    and shutil.which("xfwm4") is None
    and shutil.which("icewm") is None,
    reason="no window manager available for wmctrl to report the active window",
)
@pytest.mark.skipif(shutil.which("xterm") is None, reason="no terminal emulator")
def test_live_recording_produces_loadable_demonstrations(tmp_path):
    """End-to-end: record on a real headless display, then load the result.

    This is the genuine article -- real X server, real pointer events, real
    frames -- so the loader's ``recorded_ui_click_demonstrations`` provenance is
    honest. It is skipped where the tooling is absent rather than mocked.
    """
    from avcs.training.demonstrations import discover_demonstrations

    specs = _default_specs()
    results = record_demonstrations(
        tmp_path / "dataset", specs, display=":97", app_command="xterm"
    )
    assert sum(results.values()) > 0, f"expected real samples, got {results}"

    sessions = discover_demonstrations(tmp_path / "dataset")
    assert len(sessions) >= 1
    records = [r for recs in sessions.values() for r in recs]
    assert records, "the recorder must produce loadable records"
    assert {r.provenance for r in records} == {"recorded_ui_click_demonstrations"}
    for record in records:
        assert record.target_label, "every sample carries the label it was given"
        assert record.frame_path.is_file(), "every sample references a real frame"
