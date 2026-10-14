import json

import numpy as np
import pytest
import torch
from avcs.goal_grounder import (
    GROUNDING_ACCEPTANCE_METRIC,
    GROUNDING_CHECKPOINT_SCHEMA,
    GoalConditionedGrounder,
    click_heatmap,
    encode_queries,
    letterbox_rgb,
    load_grounding_checkpoint,
)
from avcs.training.demonstrations import (
    DemonstrationClickDataset,
    discover_demonstrations,
    split_sessions,
)
from avcs.training.train_grounder import _batch_metrics, _select_confidence_threshold
from PIL import Image


def test_query_encoder_uses_byte_tokens_and_rejects_empty_text():
    tokens = encode_queries(["Save", "é"])

    assert tokens.shape == (2, 96)
    assert tokens[0, :4].tolist() == [84, 98, 119, 102]
    assert tokens[1, :2].tolist() == [196, 170]
    with pytest.raises(ValueError, match="must not be empty"):
        encode_queries(["  "])
    with pytest.raises(ValueError, match="at most 4 UTF-8 bytes"):
        encode_queries(["button"], max_length=4)


def test_model_predict_requires_real_demonstration_checkpoint():
    model = GoalConditionedGrounder()
    with pytest.raises(RuntimeError, match="no demonstration-trained checkpoint"):
        model.predict(np.zeros((40, 80, 3), dtype=np.uint8), "save button")


def test_goal_conditioned_model_emits_dense_map_and_backpropagates():
    model = GoalConditionedGrounder(feature_dim=64, input_size=64)
    image = torch.rand(2, 3, 64, 64)
    query = encode_queries(["save button", "close window"])

    logits = model(image, query)
    assert logits.shape == (2, 1, 32, 32)
    logits.square().mean().backward()
    assert model.query_encoder.embedding.weight.grad is not None
    assert model.visual_encoder.stem.conv.weight.grad is not None


def test_letterbox_preserves_aspect_and_coordinate_round_trip():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, transform = letterbox_rgb(frame, input_size=100)

    assert tensor.shape == (3, 100, 100)
    assert transform.original_to_canvas((0.75, 0.25)) == pytest.approx((0.75, 0.375))
    assert transform.canvas_to_original((0.75, 0.375)) == pytest.approx((0.75, 0.25))


def test_click_heatmap_is_normalized_and_peaks_near_labeled_click():
    target = click_heatmap((0.7, 0.25), (32, 32))

    assert float(target.sum()) == pytest.approx(1.0)
    y, x = divmod(int(target.argmax()), target.shape[1])
    assert abs((x + 0.5) / 32 - 0.7) < 0.04
    assert abs((y + 0.5) / 32 - 0.25) < 0.04


def test_grounder_top1_point_accuracy_uses_screen_diagonal_tolerance():
    logits = torch.full((2, 1, 4, 4), -10.0)
    logits[0, 0, 1, 1] = 10.0
    logits[1, 0, 0, 0] = 10.0
    targets = torch.full((2, 1, 4, 4), 1.0 / 16.0)
    points = torch.tensor([[0.375, 0.375], [0.875, 0.875]])
    letterbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 2)

    metrics = _batch_metrics(logits, targets, points, letterbox)

    assert metrics["top1_point_accuracy_within_0.02_diagonal"].item() == pytest.approx(0.5)
    assert metrics["top1_point_accuracy_within_0.05_diagonal"].item() == pytest.approx(0.5)
    assert metrics["mean_normalized_click_distance"].item() == pytest.approx(0.375)


def test_confidence_threshold_requires_minimum_validation_coverage():
    threshold, accepted, precision = _select_confidence_threshold(
        [0.9] * 9,
        [True] * 9,
        required_precision=0.9,
    )

    assert (threshold, accepted, precision) == (1.0, 0, 0.0)


def test_confidence_threshold_selects_widest_validation_set_at_required_precision():
    threshold, accepted, precision = _select_confidence_threshold(
        [0.2] * 10 + [0.5] * 10 + [0.8] * 10,
        [True] * 10 + [False] * 10 + [True] * 10,
        required_precision=0.9,
    )

    assert threshold == 0.8
    assert accepted == 10
    assert precision == 1.0


def _write_session(root, name, point=(0.5, 0.5), *, task=None, window_title=None):
    session = root / name
    (session / "frames").mkdir(parents=True)
    Image.new("RGB", (120, 60), color=(40, 50, 60)).save(session / "frames" / "frame.png")
    record = {
        "schema_version": 1,
        "target_label": "Save button",
        "frame": "frames/frame.png",
        "action": {"type": "mouse_click", "button": "left", "point_norm": list(point)},
    }
    (session / "samples.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "stopped",
        "task": task or f"click the labeled test button in {name}",
        "target_window_title": window_title or f"AVCS disposable test fixture {name}",
        "sample_count": 1,
        "capture_scope": "cropped target window only",
        "keyboard_recording": False,
        "network_upload": False,
    }
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return session


def test_demonstration_split_keeps_recording_sessions_disjoint(tmp_path):
    _write_session(tmp_path, "session-a")
    _write_session(tmp_path, "session-b", (0.25, 0.75))
    _write_session(tmp_path, "session-c", (0.75, 0.25))
    _write_session(tmp_path, "session-d", (0.1, 0.9))
    _write_session(tmp_path, "session-e", (0.9, 0.1))
    discovered = discover_demonstrations(tmp_path)
    train, validation, test = split_sessions(discovered, seed=3)

    assert len(train) == 3
    assert len(validation) == len(test) == 1
    assert len({record.session_dir for record in train + validation + test}) == 5
    split_groups = [
        {(record.task.casefold(), record.target_window_title.casefold()) for record in split}
        for split in (train, validation, test)
    ]
    assert not split_groups[0] & split_groups[1]
    assert not split_groups[0] & split_groups[2]
    assert not split_groups[1] & split_groups[2]
    sample = DemonstrationClickDataset(train, input_size=64)[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["target_heatmap"].shape == (32, 32)
    assert sample["query_tokens"].shape == (96,)


def test_demonstration_split_rejects_repeated_task_window_sessions(tmp_path):
    for index in range(5):
        _write_session(
            tmp_path,
            f"session-{index}",
            task="complete the same form",
            window_title="same application window",
        )

    with pytest.raises(ValueError, match="distinct task/window groups"):
        split_sessions(discover_demonstrations(tmp_path))


def test_dataset_rejects_frame_path_traversal(tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("RGB", (10, 10)).save(outside)
    session = tmp_path / "session"
    session.mkdir()
    sample = {
        "schema_version": 1,
        "target_label": "button",
        "frame": "../outside.png",
        "action": {"type": "mouse_click", "button": "left", "point_norm": [0.5, 0.5]},
    }
    (session / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "stopped",
                "task": "fixture task",
                "target_window_title": "fixture",
                "sample_count": 1,
                "capture_scope": "cropped target window only",
                "keyboard_recording": False,
                "network_upload": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Frame path escapes"):
        discover_demonstrations(tmp_path)


def test_session_split_requires_multiple_recording_sessions(tmp_path):
    _write_session(tmp_path, "only-session")
    with pytest.raises(ValueError, match="At least 5 distinct task/window groups"):
        split_sessions(discover_demonstrations(tmp_path))


def test_checkpoint_loader_rejects_non_demonstration_provenance(tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    torch.save(
        {
            "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": "synthetic_colored_blocks",
            "training_samples": 100,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="dataset_provenance"):
        load_grounding_checkpoint(checkpoint)


def test_checkpoint_loader_accepts_demonstration_trainer_shape_and_predicts(tmp_path):
    source_model = GoalConditionedGrounder(input_size=64)
    checkpoint = tmp_path / "grounder.pt"
    torch.save(
        {
            "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": "recorded_ui_click_demonstrations",
            "training_samples": 12,
            "validation_samples": 3,
            "held_out_test_samples": 20,
            "evaluation": {
                "metric": GROUNDING_ACCEPTANCE_METRIC,
                "value": 0.90,
                "minimum": 0.85,
                "passed": True,
                "samples": 20,
            },
            "coordinate_frame": "cropped_target_window",
            "query_confidence_calibration": {
                "method": "validation_selected_peak_score_threshold",
                "evaluation": {
                    "metric": "empirical_precision_at_threshold",
                    "samples": 20,
                    "accepted_samples": 10,
                    "threshold": 0.2,
                    "precision": 0.9,
                    "required_precision": 0.9,
                    "minimum_accepted_samples": 10,
                    "passed": True,
                },
            },
            "input_size": 64,
            "model_state_dict": source_model.state_dict(),
        },
        checkpoint,
    )

    loaded = load_grounding_checkpoint(checkpoint)
    result = loaded.predict(np.zeros((48, 96, 3), dtype=np.uint8), "Save button")

    assert loaded.is_trained is True
    assert result.heatmap.shape == (32, 32)
    assert result.point_norm is None or all(0 <= coordinate < 1 for coordinate in result.point_norm)


def test_checkpoint_loader_rejects_grounder_below_held_out_accuracy_target(tmp_path):
    source_model = GoalConditionedGrounder(input_size=64)
    checkpoint = tmp_path / "below_target.pt"
    torch.save(
        {
            "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": "recorded_ui_click_demonstrations",
            "training_samples": 12,
            "validation_samples": 3,
            "held_out_test_samples": 3,
            "evaluation": {
                "metric": GROUNDING_ACCEPTANCE_METRIC,
                "value": 0.84,
                "minimum": 0.85,
                "passed": False,
                "samples": 3,
            },
            "input_size": 64,
            "model_state_dict": source_model.state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="does not meet the held-out evaluation contract"):
        load_grounding_checkpoint(checkpoint)
