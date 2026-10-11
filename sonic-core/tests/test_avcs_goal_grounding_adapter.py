from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from sonic.computer_use.avcs_perception import AVCSPerception


def _png_bytes(width=200, height=100, color=(10, 20, 30)):
    image = Image.new("RGB", (width, height), color=color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _checkpoint_metadata(
    *,
    samples=20,
    accuracy=0.9,
    passed=True,
    metric="top1_point_accuracy_within_0.02_diagonal",
):
    return {
        "schema_version": 2,
        "model_type": "avcs_goal_conditioned_grounder",
        "dataset_provenance": "recorded_ui_click_demonstrations",
        "training_samples": 100,
        "validation_samples": 20,
        "held_out_test_samples": 20,
        "evaluation": {
            "metric": metric,
            "samples": samples,
            "value": accuracy,
            "passed": passed,
        },
        "query_confidence_calibration": {
            "method": "held_out_empirical_precision",
            "evaluation": {
                "metric": "empirical_precision_at_threshold",
                "samples": 20,
                "threshold": 0.8,
                "precision": 0.95,
                "required_precision": 0.9,
                "passed": True,
            },
        },
    }


class _StubGrounder:
    def __init__(
        self,
        point_norm=(0.75, 0.5),
        metadata=None,
        is_trained=True,
        peak_probability=0.95,
    ):
        self.checkpoint_metadata = metadata or _checkpoint_metadata()
        self.is_trained = is_trained
        self.point_norm = point_norm
        self.peak_probability = peak_probability
        self.frame_shape = None
        self.query = None

    def predict(self, frame, query):
        self.frame_shape = frame.shape
        self.query = query
        return SimpleNamespace(
            point_norm=self.point_norm,
            peak_probability=self.peak_probability,
        )


def _configured_perception(monkeypatch, tmp_path, *, model=None):
    checkpoint = tmp_path / "evaluated.pt"
    checkpoint.touch()
    monkeypatch.setenv("SONIC_AVCS_GROUNDING_CHECKPOINT", str(checkpoint))
    perception = AVCSPerception()
    grounder = model or _StubGrounder()
    perception._load_goal_grounder = lambda: grounder
    return perception, grounder


def test_goal_grounder_uses_checkpoint_gated_prediction_for_latest_pixel_frame(
    monkeypatch, tmp_path
):
    perception, model = _configured_perception(monkeypatch, tmp_path)
    png_bytes = _png_bytes()
    state = perception.perceive_png(png_bytes)

    result = perception.ground_target_png(
        png_bytes, "Save the document", width=200, height=100
    )

    assert {key: value for key, value in result.items() if key != "latency_ms"} == {
        "engine": "avcs_goal_grounder",
        "x": 150,
        "y": 50,
        "point_norm": [0.75, 0.5],
        "frame_index": state["frame_index"],
        "scene_hash": state["scene_hash"],
        "held_out_point_accuracy": 0.9,
        "query_score": 0.95,
        "query_score_threshold": 0.8,
        "checkpoint_metadata": _checkpoint_metadata(),
    }
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0
    assert model.frame_shape == (100, 200, 3)
    assert model.query == "Save the document"
    assert perception.goal_grounder_ready is True


def test_goal_grounder_not_ready_before_valid_checkpoint_load(monkeypatch):
    monkeypatch.delenv("SONIC_AVCS_GROUNDING_CHECKPOINT", raising=False)
    perception = AVCSPerception()

    assert perception.goal_grounder_configured is False
    assert perception.goal_grounder_ready is False
    with pytest.raises(RuntimeError, match="is not configured"):
        perception.ground_target_png(_png_bytes(), "Save", width=200, height=100)


def test_configured_but_missing_checkpoint_does_not_claim_readiness(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("SONIC_AVCS_GROUNDING_CHECKPOINT", str(missing))
    perception = AVCSPerception()

    assert perception.goal_grounder_configured is True
    assert perception.goal_grounder_ready is False
    perception.perceive_png(_png_bytes())
    with pytest.raises(FileNotFoundError, match="does not exist"):
        perception.ground_target_png(_png_bytes(), "Save", width=200, height=100)
    assert perception.goal_grounder_ready is False


def test_goal_grounder_requires_explicit_checkpoint_configuration(monkeypatch):
    monkeypatch.delenv("SONIC_AVCS_GROUNDING_CHECKPOINT", raising=False)
    perception = AVCSPerception()

    with pytest.raises(RuntimeError, match="is not configured"):
        perception.ground_target_png(_png_bytes(), "Save", width=200, height=100)


@pytest.mark.parametrize(
    ("metadata", "is_trained"),
    [
        (_checkpoint_metadata(passed=False), True),
        (_checkpoint_metadata(accuracy=0.84), True),
        (_checkpoint_metadata(metric="training_accuracy"), True),
        (_checkpoint_metadata(samples=0), True),
        (_checkpoint_metadata(samples=19), True),
        (_checkpoint_metadata(), False),
        ({**_checkpoint_metadata(), "training_samples": True}, True),
    ],
)
def test_goal_grounder_rejects_unvalidated_or_untrained_checkpoint(
    monkeypatch, tmp_path, metadata, is_trained
):
    model = _StubGrounder(metadata=metadata, is_trained=is_trained)
    perception, _ = _configured_perception(monkeypatch, tmp_path, model=model)
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    with pytest.raises(ValueError, match="checkpoint|evaluation"):
        perception.ground_target_png(png_bytes, "Save", width=200, height=100)
    assert perception.goal_grounder_ready is False
    assert model.frame_shape is None


def test_goal_grounder_rejects_frame_dimension_mismatch(monkeypatch, tmp_path):
    perception, _ = _configured_perception(monkeypatch, tmp_path)
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    with pytest.raises(ValueError, match="dimensions do not match"):
        perception.ground_target_png(png_bytes, "Save", width=100, height=100)


def test_goal_grounder_requires_pixels_from_latest_perceived_frame(monkeypatch, tmp_path):
    perception, model = _configured_perception(monkeypatch, tmp_path)
    old_frame = _png_bytes(color=(10, 20, 30))
    new_frame = _png_bytes(color=(40, 50, 60))
    perception.perceive_png(old_frame)
    latest_state = perception.perceive_png(new_frame)

    with pytest.raises(RuntimeError, match="latest perceived frame"):
        perception.ground_target_png(old_frame, "Save", width=200, height=100)
    result = perception.ground_target_png(new_frame, "Save", width=200, height=100)

    assert result["frame_index"] == latest_state["frame_index"]
    assert result["scene_hash"] == latest_state["scene_hash"]
    assert model.frame_shape == (100, 200, 3)


@pytest.mark.parametrize(
    ("width", "height", "query"),
    [
        (True, 100, "Save"),
        (200, False, "Save"),
        (0, 100, "Save"),
        (200, 100, ""),
        (200, 100, "é" * 49),
    ],
)
def test_goal_grounder_rejects_invalid_dimensions_and_queries(
    monkeypatch, tmp_path, width, height, query
):
    perception, _ = _configured_perception(monkeypatch, tmp_path)
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    with pytest.raises(ValueError):
        perception.ground_target_png(png_bytes, query, width=width, height=height)


@pytest.mark.parametrize(
    "point",
    [(1.0, 0.5), (-0.01, 0.5), (float("nan"), 0.5), (True, 0.5), (0.5,)],
)
def test_goal_grounder_rejects_invalid_model_points(monkeypatch, tmp_path, point):
    perception, _ = _configured_perception(
        monkeypatch, tmp_path, model=_StubGrounder(point_norm=point)
    )
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    with pytest.raises(ValueError, match="point"):
        perception.ground_target_png(png_bytes, "Save", width=200, height=100)


def test_goal_grounder_returns_no_target_when_local_prediction_is_inconclusive(
    monkeypatch, tmp_path
):
    perception, model = _configured_perception(
        monkeypatch, tmp_path, model=_StubGrounder(point_norm=None)
    )
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    result = perception.ground_target_png(
        png_bytes, "Save the document", width=200, height=100
    )

    assert result is None
    assert model.frame_shape == (100, 200, 3)
    assert model.query == "Save the document"


@pytest.mark.parametrize("score", [None, float("nan"), -0.1, 0.79, 1.01, True])
def test_goal_grounder_abstains_without_valid_held_out_query_score(
    monkeypatch, tmp_path, score
):
    perception, _ = _configured_perception(
        monkeypatch,
        tmp_path,
        model=_StubGrounder(peak_probability=score),
    )
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    result = perception.ground_target_png(
        png_bytes, "Save the document", width=200, height=100
    )

    assert result is None


def test_goal_grounder_rejects_checkpoint_without_calibrated_confidence_gate(
    monkeypatch, tmp_path
):
    metadata = _checkpoint_metadata()
    metadata.pop("query_confidence_calibration")
    model = _StubGrounder(metadata=metadata)
    perception, _ = _configured_perception(monkeypatch, tmp_path, model=model)
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    with pytest.raises(ValueError, match="per-query confidence calibration"):
        perception.ground_target_png(png_bytes, "Save", width=200, height=100)
    assert perception.goal_grounder_ready is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("precision", 0.89),
        ("passed", False),
        ("samples", 19),
    ],
)
def test_goal_grounder_rejects_unvalidated_query_score_operating_point(
    monkeypatch, tmp_path, field, value
):
    metadata = _checkpoint_metadata()
    metadata["query_confidence_calibration"]["evaluation"][field] = value
    model = _StubGrounder(metadata=metadata)
    perception, _ = _configured_perception(monkeypatch, tmp_path, model=model)
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    with pytest.raises(ValueError, match="per-query confidence calibration"):
        perception.ground_target_png(png_bytes, "Save", width=200, height=100)
    assert perception.goal_grounder_ready is False


def test_goal_grounder_rejects_frame_change_during_local_prediction(
    monkeypatch, tmp_path
):
    perception, model = _configured_perception(monkeypatch, tmp_path)
    png_bytes = _png_bytes()
    perception.perceive_png(png_bytes)

    def predict_then_advance_frame(frame, query):
        model.frame_shape = frame.shape
        model.query = query
        perception.perceive_png(_png_bytes(color=(40, 50, 60)))
        return SimpleNamespace(
            point_norm=model.point_norm,
            peak_probability=model.peak_probability,
        )

    model.predict = predict_then_advance_frame

    with pytest.raises(RuntimeError, match="frame changed while target grounding"):
        perception.ground_target_png(
            png_bytes, "Save the document", width=200, height=100
        )


def test_reflex_state_is_versioned_pixel_only_and_local():
    perception = AVCSPerception()
    frame = _png_bytes()

    first = perception.perceive_png(frame)
    second = perception.perceive_png(frame)

    assert first["engine"] == second["engine"] == "avcs_reflex"
    assert second["frame_index"] == first["frame_index"] + 1
    assert second["width"] == 200
    assert second["height"] == 100
    assert "ocr" not in second
    assert "text" not in second
    assert "detected_controls" not in second
