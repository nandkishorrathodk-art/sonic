"""Affordance supervision from measured click outcomes + its honesty gate.

These tests assert the properties that make "AVCS learns where to click" real:

* the dorsal head's readiness is derived from recorded training provenance, not
  from the checkpoint file existing;
* a checkpoint whose head was never trained (or whose held-out accuracy is at
  chance) reports hotspots as unavailable;
* click outcomes become sparse point supervision, and every unobserved region
  stays ignored rather than being labelled non-interactive;
* a real fine-tune reduces loss and measures a held-out score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from forge.vision.cortex.config import ModelConfig
from forge.vision.cortex.data.outcome_affordance import (
    AFFORDANCE_GRID,
    IGNORE_INDEX,
    OutcomeAffordanceDataset,
    build_point_mask,
)
from forge.vision.cortex.model import ArtificialVisualCortex
from forge.vision.cortex.training.affordance_provenance import (
    MIN_HELD_OUT_POINT_ACCURACY,
    affordance_head_trained,
    read_affordance_provenance,
    write_affordance_provenance,
)
from forge.vision.cortex.training.affordance_trainer import AffordanceTrainer
from PIL import Image


def _write_cortex(path: Path) -> Path:
    model = ArtificialVisualCortex(ModelConfig())
    torch.save(model.state_dict(), path)
    return path


def _make_manifest(tmp_path: Path, count: int = 24) -> Path:
    """A tiny click-outcome manifest with real interactive points."""
    from forge.vision.cortex.data.synthetic import ProceduralScreenGenerator

    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    generator = ProceduralScreenGenerator(width=640, height=360)
    samples = []
    for idx in range(count):
        data = generator.generate(seed=5000 + idx)
        Image.fromarray(data.frame).save(frames / f"f{idx:03d}.png")
        if data.interactive_points and idx % 3 != 2:
            x, y, cls = data.interactive_points[0]
            action, changed = (cls, True)
        else:
            x, y, action, changed = (0.5, 0.5, "clickable", False)
        samples.append({
            "frame": f"frames/f{idx:03d}.png",
            "click": [x, y],
            "action": action,
            "changed": changed,
            "group": f"g{idx // 4}",
            "query": "button" if changed else "",
            "category": data.category,
        })
    manifest = tmp_path / "samples.json"
    manifest.write_text(
        json.dumps({"provenance": "measured_click_outcomes", "samples": samples}),
        encoding="utf-8",
    )
    return manifest


# -- point mask -------------------------------------------------------------


def test_point_mask_is_sparse_and_ignores_everything_else():
    mask = build_point_mask((0.5, 0.5), class_index=1, radius=2)
    supervised = mask != IGNORE_INDEX
    # A small disk is labelled; the vast majority of the frame is ignored.
    assert 0 < int(supervised.sum()) < mask.size * 0.1
    assert set(np.unique(mask[supervised]).tolist()) == {1}


def test_point_mask_marks_the_clicked_cell_not_the_centre():
    mask = build_point_mask((0.9, 0.1), class_index=2, radius=1)
    gx, gy = int(round(0.9 * (AFFORDANCE_GRID[1] - 1))), int(round(0.1 * (AFFORDANCE_GRID[0] - 1)))
    assert mask[gy, gx] == 2


def test_point_mask_rejects_out_of_range_class():
    with pytest.raises(ValueError):
        build_point_mask((0.5, 0.5), class_index=99)


# -- dataset ----------------------------------------------------------------


def test_dataset_labels_positive_click_with_action_class(tmp_path):
    manifest = _make_manifest(tmp_path)
    dataset = OutcomeAffordanceDataset(manifest)
    positive = next(dataset[i] for i in range(len(dataset)) if dataset.records[i]["changed"])
    supervised = positive["affordance_mask"] != IGNORE_INDEX
    assert bool(positive["grounding_valid"]) is True
    assert int(supervised.sum()) > 0


def test_dataset_labels_negative_click_as_static(tmp_path):
    manifest = _make_manifest(tmp_path)
    dataset = OutcomeAffordanceDataset(manifest)
    negative = next(
        dataset[i] for i in range(len(dataset)) if not dataset.records[i]["changed"]
    )
    supervised = negative["affordance_mask"] != IGNORE_INDEX
    assert set(np.unique(negative["affordance_mask"][supervised]).tolist()) == {0}


def test_dataset_skips_samples_without_a_measured_outcome(tmp_path):
    manifest = _make_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    # Remove the outcome from one sample: it must be dropped, not assumed.
    del payload["samples"][0]["changed"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    dataset = OutcomeAffordanceDataset(manifest)
    assert len(dataset) == len(payload["samples"]) - 1


def test_dataset_grounding_masked_without_a_named_query(tmp_path):
    manifest = _make_manifest(tmp_path)
    dataset = OutcomeAffordanceDataset(manifest)
    unqueried = next(
        dataset[i] for i in range(len(dataset)) if not dataset.records[i]["query"]
    )
    assert bool(unqueried["grounding_valid"]) is False


def test_dataset_raises_when_nothing_is_measurable(tmp_path):
    manifest = tmp_path / "empty.json"
    manifest.write_text(json.dumps({"samples": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        OutcomeAffordanceDataset(manifest)


# -- provenance gate --------------------------------------------------------


def test_gate_fails_closed_without_a_sidecar(tmp_path):
    checkpoint = _write_cortex(tmp_path / "cortex.pt")
    assert affordance_head_trained(checkpoint) is False


def test_gate_fails_closed_at_chance_accuracy(tmp_path):
    checkpoint = _write_cortex(tmp_path / "cortex.pt")
    write_affordance_provenance(
        checkpoint,
        dataset_provenance="test",
        samples=24,
        positive_samples=16,
        negative_samples=8,
        held_out_samples=8,
        held_out_iou=0.2,
        held_out_point_accuracy=MIN_HELD_OUT_POINT_ACCURACY - 0.05,
    )
    assert affordance_head_trained(checkpoint) is False


def test_gate_fails_closed_without_a_measured_accuracy(tmp_path):
    checkpoint = _write_cortex(tmp_path / "cortex.pt")
    write_affordance_provenance(
        checkpoint,
        dataset_provenance="test",
        samples=24,
        positive_samples=16,
        negative_samples=8,
        held_out_samples=8,
        held_out_iou=0.2,
        held_out_point_accuracy=None,
    )
    assert affordance_head_trained(checkpoint) is False


def test_gate_passes_above_chance_accuracy(tmp_path):
    checkpoint = _write_cortex(tmp_path / "cortex.pt")
    write_affordance_provenance(
        checkpoint,
        dataset_provenance="test",
        samples=24,
        positive_samples=16,
        negative_samples=8,
        held_out_samples=8,
        held_out_iou=0.6,
        held_out_point_accuracy=0.75,
    )
    assert affordance_head_trained(checkpoint) is True
    payload = read_affordance_provenance(checkpoint)
    assert payload["held_out_point_accuracy"] == 0.75


def test_provenance_requires_a_named_dataset(tmp_path):
    checkpoint = _write_cortex(tmp_path / "cortex.pt")
    with pytest.raises(ValueError):
        write_affordance_provenance(
            checkpoint,
            dataset_provenance="",
            samples=5,
            positive_samples=3,
            negative_samples=2,
            held_out_samples=2,
            held_out_iou=0.1,
            held_out_point_accuracy=0.9,
        )


# -- trainer ----------------------------------------------------------------


def test_trainer_fine_tunes_the_dorsal_head_and_measures_held_out(tmp_path):
    manifest = _make_manifest(tmp_path)
    base = _write_cortex(tmp_path / "base.pt")
    trainer = AffordanceTrainer(base, epochs=6, seed=3)
    result = trainer.train(manifest, tmp_path / "affordance.pt", work_dir=tmp_path)

    assert result.trained is True
    assert result.samples > 0
    assert result.held_out_samples > 0
    # Loss must actually fall; a run that did not learn is not a training run.
    assert result.history[-1]["loss"] < result.history[0]["loss"]
    # A held-out score is measured, even if the small set cannot generalise.
    assert result.held_out_point_accuracy is not None
    assert 0.0 <= result.held_out_point_accuracy <= 1.0


def test_trainer_writes_provenance_next_to_the_checkpoint(tmp_path):
    manifest = _make_manifest(tmp_path)
    base = _write_cortex(tmp_path / "base.pt")
    trainer = AffordanceTrainer(base, epochs=3, seed=3)
    result = trainer.train(manifest, tmp_path / "affordance.pt", work_dir=tmp_path)
    payload = read_affordance_provenance(tmp_path / "affordance.pt")
    assert payload is not None
    assert payload["dataset_provenance"] == "click_outcome_supervision"
    assert payload["held_out_point_accuracy"] == result.held_out_point_accuracy


def test_trainer_refuses_when_no_samples_are_measurable(tmp_path):
    manifest = tmp_path / "empty.json"
    manifest.write_text(json.dumps({"samples": []}), encoding="utf-8")
    base = _write_cortex(tmp_path / "base.pt")
    trainer = AffordanceTrainer(base, epochs=2)
    result = trainer.train(manifest, tmp_path / "out.pt", work_dir=tmp_path)
    assert result.trained is False
    assert result.reason


def test_trainer_leaves_the_category_head_untouched(tmp_path):
    manifest = _make_manifest(tmp_path)
    base = _write_cortex(tmp_path / "base.pt")
    before = torch.load(base, map_location="cpu")
    trainer = AffordanceTrainer(base, epochs=3, seed=3)
    trainer.train(manifest, tmp_path / "affordance.pt", work_dir=tmp_path)
    after = torch.load(tmp_path / "affordance.pt", map_location="cpu")

    changed = [
        key for key in before
        if not torch.equal(before[key], after[key])
    ]
    # Only dorsal parameters may differ; the frozen trunk and ventral head must
    # be identical so a sparse click set cannot degrade the category model.
    assert changed, "expected the dorsal head to change"
    assert all(key.startswith("dorsal.") for key in changed), changed
