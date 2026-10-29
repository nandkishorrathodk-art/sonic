"""Tests for the ScreenSpot-v2 -> recorder-format dataset bridge.

The bridge exists so the grounder trainer can consume a public dataset of real
human click annotations without weakening any of the trainer's provenance
checks. These tests pin the contract that matters:

* converted sessions are readable by the *real* loader,
* provenance is never laundered into the operator-recording identity, and
* malformed annotations are skipped rather than becoming wrong labels.
"""

from __future__ import annotations

import json

import pytest
from avcs.training.demonstrations import (
    RECORDED_PROVENANCE,
    discover_demonstrations,
    split_sessions,
)
from avcs.training.screenspot_bridge import (
    SCREENSPOT_PROVENANCE,
    _bbox_centre_norm,
    _target_label,
    convert_screenspot,
)
from PIL import Image


def _write_image(path, size=(640, 480)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (12, 34, 56)).save(path)


def _write_annotations(annotations_dir, name, rows) -> None:
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / name).write_text(json.dumps(rows), encoding="utf-8")


def _make_dataset(tmp_path):
    annotations = tmp_path / "ann"
    images = tmp_path / "img"
    out = tmp_path / "out"
    for index in range(6):
        _write_image(images / f"shot_{index}.png")
    rows = [
        {
            "img_filename": f"shot_{index}.png",
            "bbox": [100, 100, 40, 40],
            "instruction": f"click control {index}",
            "data_type": "icon",
            "data_source": "windows",
        }
        for index in range(6)
    ]
    _write_annotations(annotations, "screenspot_desktop_v2.json", rows)
    return annotations, images, out


class TestTargetLabel:
    def test_normalises_whitespace(self):
        assert _target_label("  click   the  button ") == "click the button"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _target_label("   ")

    def test_truncates_over_long_labels_to_the_encoder_budget(self):
        label = _target_label("x" * 500)
        assert 0 < len(label.encode("utf-8")) <= 96

    def test_truncation_stays_valid_utf8(self):
        label = _target_label("é" * 200)
        label.encode("utf-8")  # must not raise


class TestBboxCentre:
    def test_centre_of_a_valid_box(self):
        assert _bbox_centre_norm([100, 50, 40, 20], 640, 480) == pytest.approx(
            (120 / 640, 60 / 480)
        )

    @pytest.mark.parametrize(
        "bbox",
        [
            [100, 100, 0, 40],  # zero width
            [100, 100, 40, -4],  # negative height
            [-5, 10, 40, 40],  # starts off-image
            [600, 10, 100, 40],  # overflows width
            [10, 400, 40, 200],  # overflows height
            [1, 2, 3],  # wrong arity
        ],
    )
    def test_rejects_degenerate_or_out_of_bounds(self, bbox):
        assert _bbox_centre_norm(bbox, 640, 480) is None


class TestConversion:
    def test_converted_sessions_load_in_the_real_loader(self, tmp_path):
        annotations, images, out = _make_dataset(tmp_path)
        summary = convert_screenspot(
            annotations_dir=annotations, images_dir=images, output_dir=out
        )
        assert summary["total_samples"] == 6
        sessions = discover_demonstrations(out)
        assert sum(len(records) for records in sessions.values()) == 6

    def test_split_is_possible_from_converted_sessions(self, tmp_path):
        annotations, images, out = _make_dataset(tmp_path)
        convert_screenspot(
            annotations_dir=annotations, images_dir=images, output_dir=out
        )
        sessions = discover_demonstrations(out)
        train, validation, test = split_sessions(sessions, seed=1)
        assert train and validation and test
        assert len(train) + len(validation) + len(test) == 6

    def test_provenance_is_never_laundered_as_recorded(self, tmp_path):
        annotations, images, out = _make_dataset(tmp_path)
        convert_screenspot(
            annotations_dir=annotations, images_dir=images, output_dir=out
        )
        manifests = list(out.glob("*/manifest.json"))
        assert manifests
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["provenance"] == SCREENSPOT_PROVENANCE
            assert manifest["provenance"] != RECORDED_PROVENANCE

    def test_each_image_becomes_its_own_session(self, tmp_path):
        annotations, images, out = _make_dataset(tmp_path)
        convert_screenspot(
            annotations_dir=annotations, images_dir=images, output_dir=out
        )
        sessions = discover_demonstrations(out)
        # One session per screenshot keeps a frame on one side of the split.
        assert len(sessions) == 6

    def test_malformed_rows_are_skipped_not_mislabelled(self, tmp_path):
        annotations, images, out = _make_dataset(tmp_path)
        rows = json.loads(
            (annotations / "screenspot_desktop_v2.json").read_text(encoding="utf-8")
        )
        rows.append(
            {"img_filename": "missing.png", "bbox": [1, 2, 3, 4], "instruction": "x"}
        )
        rows.append({"img_filename": "shot_0.png", "bbox": [1, 2], "instruction": "x"})
        rows.append(
            {"img_filename": "shot_1.png", "bbox": [1, 2, 3, 4], "instruction": "  "}
        )
        _write_annotations(annotations, "screenspot_desktop_v2.json", rows)
        summary = convert_screenspot(
            annotations_dir=annotations, images_dir=images, output_dir=out
        )
        assert summary["total_samples"] == 6
        assert summary["total_skipped"] == 3

    def test_multiple_instructions_per_image_stay_in_one_session(self, tmp_path):
        annotations, images, out = _make_dataset(tmp_path)
        rows = json.loads(
            (annotations / "screenspot_desktop_v2.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "img_filename": "shot_0.png",
                "bbox": [200, 200, 40, 40],
                "instruction": "a different control",
            }
        )
        _write_annotations(annotations, "screenspot_desktop_v2.json", rows)
        summary = convert_screenspot(
            annotations_dir=annotations, images_dir=images, output_dir=out
        )
        assert summary["total_samples"] == 7
        sessions = discover_demonstrations(out)
        assert len(sessions) == 6
        counts = sorted(len(records) for records in sessions.values())
        assert counts[-1] == 2

    def test_missing_annotation_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_screenspot(
                annotations_dir=tmp_path / "nope",
                images_dir=tmp_path,
                output_dir=tmp_path / "out",
            )

    def test_missing_images_directory_raises(self, tmp_path):
        annotations, _images, out = _make_dataset(tmp_path)
        with pytest.raises(FileNotFoundError):
            convert_screenspot(
                annotations_dir=annotations,
                images_dir=tmp_path / "nope",
                output_dir=out,
            )

    def test_copy_mode_produces_independent_frames(self, tmp_path):
        annotations, images, out = _make_dataset(tmp_path)
        convert_screenspot(
            annotations_dir=annotations,
            images_dir=images,
            output_dir=out,
            link_frames=False,
        )
        frame = next(out.glob("*/frames/current.png"))
        source = images / "shot_0.png"
        assert frame.read_bytes() == source.read_bytes()
        assert frame.stat().st_ino != source.stat().st_ino
