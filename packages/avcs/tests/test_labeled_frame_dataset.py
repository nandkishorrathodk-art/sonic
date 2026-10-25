"""LabeledFrameDataset must use real labels and never invent supervision.

The dataset consumes a capture manifest of real desktop frames with an
operator-observed scene label. Unknown labels must be masked, and the
affordance/click heads must stay masked rather than trained on guesses.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from forge.vision.cortex.core.ventral import UI_CATEGORIES  # noqa: E402
from forge.vision.cortex.data.labeled_dataset import (  # noqa: E402
    LabeledFrameDataset,
    create_labeled_dataloader,
)


def _write_manifest(tmp_path: Path, labels: list[str]) -> Path:
    from PIL import Image

    frames = []
    for index, label in enumerate(labels):
        name = f"frame_{index}.png"
        Image.new("RGB", (64, 36), (index * 10 % 255, 40, 80)).save(tmp_path / name)
        frames.append({"frame": name, "label": label})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return manifest


def test_real_labels_map_to_category_indices(tmp_path):
    known = UI_CATEGORIES[0]
    manifest = _write_manifest(tmp_path, [known, known])

    dataset = LabeledFrameDataset(manifest, width=64, height=36)

    assert len(dataset) == 2
    assert dataset.labeled_ratio == 1.0
    sample = dataset[0]
    assert int(sample["category_idx"]) == UI_CATEGORIES.index(known)


def test_unknown_label_is_masked_not_guessed(tmp_path):
    manifest = _write_manifest(tmp_path, ["not-a-real-category"])

    dataset = LabeledFrameDataset(manifest, width=64, height=36)

    assert dataset.labeled_ratio == 0.0
    assert int(dataset[0]["category_idx"]) == -1


def test_affordance_targets_are_masked_not_fabricated(tmp_path):
    manifest = _write_manifest(tmp_path, [UI_CATEGORIES[0], UI_CATEGORIES[0]])

    sample = LabeledFrameDataset(manifest, width=64, height=36)[0]

    affordance = sample["affordance_mask"]
    assert affordance.shape == (32, 32)
    assert bool((affordance == -100).all()), "affordance must stay masked"


def test_temporal_pair_comes_from_the_same_scene(tmp_path):
    manifest = _write_manifest(tmp_path, [UI_CATEGORIES[0], UI_CATEGORIES[0]])

    dataset = LabeledFrameDataset(manifest, width=64, height=36)
    first = dataset[0]["frame_next"]
    second = dataset[1]["frame_next"]

    # Frame 0 pairs with frame 1 and vice versa (same scene, no self-pair).
    assert not torch.equal(first, second)


def test_empty_manifest_fails_closed(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        LabeledFrameDataset(manifest)


def test_dataloader_batches_real_frames(tmp_path):
    manifest = _write_manifest(tmp_path, [UI_CATEGORIES[0]] * 4)

    loader = create_labeled_dataloader(manifest, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch["frame_t"].shape[0] == 2
    assert batch["category_idx"].tolist() == [
        UI_CATEGORIES.index(UI_CATEGORIES[0])
    ] * 2
