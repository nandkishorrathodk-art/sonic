"""Synthetic session generator + honest provenance propagation.

The generator exists so the grounding pipeline can be exercised without a live
X11 desktop. The critical property is that synthetic data must never be able to
masquerade as real recorded demonstrations: the provenance has to travel from
the session manifest, through the loader, into the checkpoint.
"""

from __future__ import annotations

import json

import pytest
from avcs.training.demonstrations import (
    RECORDED_PROVENANCE,
    DemonstrationClickDataset,
    discover_demonstrations,
    split_sessions,
)
from avcs.training.synthetic_sessions import (
    SYNTHETIC_PROVENANCE,
    generate_dataset,
    write_session,
)


class TestSyntheticGenerator:
    def test_writes_sessions_the_loader_accepts(self, tmp_path):
        generate_dataset(tmp_path / "demos", sessions=4, controls_per_session=3)
        sessions = discover_demonstrations(tmp_path / "demos")
        assert len(sessions) == 4

    def test_session_matches_recorder_manifest_contract(self, tmp_path):
        session = write_session(
            tmp_path,
            "s1",
            task="t",
            target_window_title="w",
            control_count=2,
        )
        manifest = json.loads((session / "manifest.json").read_text())
        assert manifest["schema_version"] == 1
        assert manifest["status"] == "stopped"
        assert manifest["capture_scope"] == "cropped target window only"
        assert manifest["keyboard_recording"] is False
        assert manifest["network_upload"] is False
        assert manifest["sample_count"] == 2

    def test_click_points_are_normalized_in_unit_interval(self, tmp_path):
        session = write_session(
            tmp_path, "s2", task="t", target_window_title="w", control_count=5
        )
        for line in (session / "samples.jsonl").read_text().splitlines():
            record = json.loads(line)
            x, y = record["action"]["point_norm"]
            assert 0.0 <= x < 1.0
            assert 0.0 <= y < 1.0

    def test_every_sample_frame_exists(self, tmp_path):
        session = write_session(
            tmp_path, "s3", task="t", target_window_title="w", control_count=4
        )
        for line in (session / "samples.jsonl").read_text().splitlines():
            frame = json.loads(line)["frame"]
            assert (session / frame).is_file()

    def test_too_few_sessions_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            generate_dataset(tmp_path / "demos", sessions=2)

    def test_existing_session_dir_is_not_overwritten(self, tmp_path):
        write_session(tmp_path, "dup", task="t", target_window_title="w")
        with pytest.raises(FileExistsError):
            write_session(tmp_path, "dup", task="t", target_window_title="w")


class TestProvenanceHonesty:
    def test_generated_sessions_are_marked_synthetic(self, tmp_path):
        generate_dataset(tmp_path / "demos", sessions=3)
        sessions = discover_demonstrations(tmp_path / "demos")
        provenances = {
            record.provenance
            for records in sessions.values()
            for record in records
        }
        assert provenances == {SYNTHETIC_PROVENANCE}
        assert SYNTHETIC_PROVENANCE != RECORDED_PROVENANCE

    def test_manifest_without_provenance_defaults_to_recorded(self, tmp_path):
        """A recorder session (which omits the field) still reads as recorded."""
        session = tmp_path / "legacy"
        (session / "frames").mkdir(parents=True)
        from PIL import Image

        Image.new("RGB", (32, 32)).save(session / "frames" / "f.png")
        (session / "samples.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "target_label": "Save",
                    "frame": "frames/f.png",
                    "action": {
                        "type": "mouse_click",
                        "button": "left",
                        "point_norm": [0.5, 0.5],
                    },
                }
            )
            + "\n"
        )
        (session / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "stopped",
                    "task": "t",
                    "target_window_title": "w",
                    "sample_count": 1,
                    "capture_scope": "cropped target window only",
                    "keyboard_recording": False,
                    "network_upload": False,
                }
            )
        )
        sessions = discover_demonstrations(tmp_path)
        records = next(iter(sessions.values()))
        assert records[0].provenance == RECORDED_PROVENANCE

    def test_invalid_provenance_is_rejected(self, tmp_path):
        generate_dataset(tmp_path / "demos", sessions=3)
        manifest_path = tmp_path / "demos" / "session-000" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"] = "   "
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            discover_demonstrations(tmp_path / "demos")

    def test_split_works_on_generated_sessions(self, tmp_path):
        generate_dataset(tmp_path / "demos", sessions=6)
        sessions = discover_demonstrations(tmp_path / "demos")
        train, validation, test = split_sessions(sessions, seed=1)
        assert train and validation and test
        dataset = DemonstrationClickDataset(train, input_size=64)
        sample = dataset[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["target_heatmap"].shape == (32, 32)
