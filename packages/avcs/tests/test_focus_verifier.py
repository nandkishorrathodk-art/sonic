"""Tests for the AVCS local focus-verification head and its honesty gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from avcs.focus_verifier import (
    FOCUS_ACCEPTANCE_METRIC,
    FOCUS_CHECKPOINT_SCHEMA,
    FOCUS_DATASET_PROVENANCE,
    FOCUS_MODEL_TYPE,
    FocusVerifier,
    focus_patch_bounds,
    focus_patch_rgb,
    load_focus_checkpoint,
)
from avcs.training.focus_dataset import (
    FocusDemoDataset,
    discover_focus_demonstrations,
    focus_collate,
    split_focus_sessions,
)
from avcs.training.train_focus_verifier import _select_threshold


def _render_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    frame[60:100, 40:280] = 235  # a bright text-field band
    frame += rng.integers(0, 6, size=frame.shape, dtype=np.uint8)
    return frame


def _write_session(root: Path, name: str, task: str, title: str, count: int = 12) -> None:
    from PIL import Image

    session = root / name
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"status": "complete", "task": task, "target_window_title": title})
    )
    records = []
    for i in range(count):
        frame = _render_frame(i)
        focused = bool(i % 2 == 0)
        point = [160, 80] if focused else [10 + i, 220]
        Image.fromarray(frame).save(session / f"frame_{i:03d}.png")
        records.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": task,
                    "target_label": "search field",
                    "frame": f"frame_{i:03d}.png",
                    "point_px": point,
                    "focused": focused,
                }
            )
        )
    (session / "records.jsonl").write_text("\n".join(records) + "\n")


def _make_root(tmp_path: Path, sessions: int = 6) -> Path:
    root = tmp_path / "demos"
    root.mkdir()
    for s in range(sessions):
        _write_session(root, f"session-{s:02d}", f"task-{s}", f"win-{s}")
    return root


def test_focus_patch_bounds_clamps_to_frame_edges():
    assert focus_patch_bounds(5, 5, 100, 100, patch_fraction=0.5) == (0, 0, 50, 50)
    left, top, right, bottom = focus_patch_bounds(99, 99, 100, 100, patch_fraction=0.5)
    assert (left, top) == (50, 50) and (right, bottom) == (100, 100)


def test_focus_patch_rejects_out_of_frame_points():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="inside the frame"):
        focus_patch_rgb(frame, 25, 5)


def test_focus_patch_rejects_non_uint8_frames():
    frame = np.zeros((20, 20, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        focus_patch_rgb(frame, 5, 5)


def test_focus_model_predict_requires_trained_checkpoint():
    model = FocusVerifier(input_size=32)
    frame = _render_frame(0)
    with pytest.raises(RuntimeError, match="no demonstration-trained checkpoint"):
        model.predict(frame, 160, 80, "search field")


def test_focus_model_backpropagates():
    model = FocusVerifier(feature_dim=32, input_size=32)
    images = torch.rand(2, 3, 32, 32)
    from avcs.goal_grounder import encode_queries

    logits = model(images, encode_queries(["field", "button"]))
    assert logits.shape == (2,)
    logits.square().mean().backward()
    assert model.head[0].weight.grad is not None


def test_discover_and_split_are_session_disjoint(tmp_path):
    root = _make_root(tmp_path)
    sessions = discover_focus_demonstrations(root)
    assert len(sessions) == 6
    train, val, test = split_focus_sessions(sessions, seed=1)
    train_sessions = {r.session_dir.name for r in train}
    val_sessions = {r.session_dir.name for r in val}
    test_sessions = {r.session_dir.name for r in test}
    assert not (train_sessions & val_sessions)
    assert not (train_sessions & test_sessions)
    assert not (val_sessions & test_sessions)


def test_split_requires_three_sessions(tmp_path):
    root = tmp_path / "demos"
    root.mkdir()
    _write_session(root, "session-00", "task-0", "win-0")
    _write_session(root, "session-01", "task-1", "win-1")
    with pytest.raises(ValueError, match="at least three recording sessions"):
        split_focus_sessions(discover_focus_demonstrations(root))


def test_dataset_rejects_frame_path_traversal(tmp_path):
    root = tmp_path / "demos"
    root.mkdir()
    session = root / "session-00"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps({"status": "complete", "task": "t", "target_window_title": "w"})
    )
    (session / "records.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "t",
                "target_label": "field",
                "frame": "../../../etc/passwd",
                "point_px": [1, 1],
                "focused": True,
            }
        )
    )
    with pytest.raises(ValueError, match="missing or outside its session"):
        discover_focus_demonstrations(root)


def test_focus_collate_shapes(tmp_path):
    root = _make_root(tmp_path)
    sessions = discover_focus_demonstrations(root)
    records = next(iter(sessions.values()))
    dataset = FocusDemoDataset(records, input_size=32)
    batch = [dataset[i] for i in range(4)]
    patches, queries, labels = focus_collate(batch)
    assert patches.shape == (4, 3, 32, 32)
    assert queries.shape[0] == 4
    assert labels.shape == (4,)


def test_threshold_selection_requires_minimum_coverage():
    scores = [0.9] * 5 + [0.1] * 5
    hits = [1] * 5 + [0] * 5
    threshold, accepted, precision = _select_threshold(
        scores, hits, required_precision=0.8
    )
    # Only 5 samples clear the threshold, below the minimum coverage, so the
    # fallback returns the highest-precision point rather than a fake pass.
    assert accepted < 30
    assert precision == 1.0


def test_loader_rejects_non_focus_provenance(tmp_path):
    model = FocusVerifier(input_size=32)
    checkpoint_path = tmp_path / "fake.pt"
    torch.save(
        {
            "schema_version": FOCUS_CHECKPOINT_SCHEMA,
            "model_type": FOCUS_MODEL_TYPE,
            "dataset_provenance": "synthetic_rectangles",
            "training_samples": 10,
            "validation_samples": 5,
            "held_out_test_samples": 40,
            "input_size": 32,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="recorded focus demonstration provenance"):
        load_focus_checkpoint(checkpoint_path)


def test_loader_rejects_below_precision_threshold(tmp_path):
    model = FocusVerifier(input_size=32)
    checkpoint_path = tmp_path / "weak.pt"
    torch.save(
        {
            "schema_version": FOCUS_CHECKPOINT_SCHEMA,
            "model_type": FOCUS_MODEL_TYPE,
            "dataset_provenance": FOCUS_DATASET_PROVENANCE,
            "training_samples": 10,
            "validation_samples": 5,
            "held_out_test_samples": 40,
            "input_size": 32,
            "decision_threshold": 0.5,
            "evaluation": {
                "metric": FOCUS_ACCEPTANCE_METRIC,
                "precision": 0.5,
                "samples": 40,
                "passed": False,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="held-out precision contract"):
        load_focus_checkpoint(checkpoint_path)


def test_loader_accepts_passing_checkpoint_and_predicts(tmp_path):
    model = FocusVerifier(input_size=32)
    checkpoint_path = tmp_path / "good.pt"
    torch.save(
        {
            "schema_version": FOCUS_CHECKPOINT_SCHEMA,
            "model_type": FOCUS_MODEL_TYPE,
            "dataset_provenance": FOCUS_DATASET_PROVENANCE,
            "training_samples": 40,
            "validation_samples": 20,
            "held_out_test_samples": 40,
            "input_size": 32,
            "decision_threshold": 0.5,
            "evaluation": {
                "metric": FOCUS_ACCEPTANCE_METRIC,
                "precision": 0.95,
                "samples": 40,
                "passed": True,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    loaded = load_focus_checkpoint(checkpoint_path)
    assert loaded.is_trained is True
    frame = _render_frame(0)
    prediction = loaded.predict(frame, 160, 80, "search field")
    assert 0.0 <= prediction.score <= 1.0
    assert prediction.threshold == 0.5
    assert len(prediction.patch_bounds) == 4


def test_end_to_end_training_reports_runtime_ready_flag(tmp_path):
    from avcs.training.train_focus_verifier import train_focus_verifier

    root = _make_root(tmp_path)
    report = train_focus_verifier(
        data_dir=str(root),
        output_dir=str(tmp_path / "out"),
        epochs=3,
        input_size=32,
    )
    # The flag must reflect the real held-out gate, not be hardcoded true.
    assert report["runtime_ready"] == report["evaluation"]["passed"]
    assert report["dataset_provenance"] == FOCUS_DATASET_PROVENANCE


class TestTorchConstructionSerialization:
    """Model construction must not be corrupted by a concurrent dtype change.

    ``transformers``' ``from_pretrained`` temporarily sets the process-global
    torch default dtype.  Without serialization a concurrently built model can
    be created with the wrong parameter dtype, which surfaces as
    "Input type (torch.FloatTensor) and weight type (CPUBFloat16Type)".
    """

    def test_build_model_restores_default_dtype(self):
        import torch
        from forge.vision.torch_build import build_model

        original = torch.get_default_dtype()

        def _mutating_factory():
            torch.set_default_dtype(torch.bfloat16)
            return torch.nn.Linear(2, 2)

        build_model(_mutating_factory)
        assert torch.get_default_dtype() == original

    def test_build_model_serializes_concurrent_constructions(self):
        import threading
        import time

        from forge.vision.torch_build import build_model

        active = 0
        max_active = 0
        lock = threading.Lock()

        def _factory():
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return object()

        threads = [threading.Thread(target=build_model, args=(_factory,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert max_active == 1, "model construction was not serialized"
