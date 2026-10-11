from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from sonic.computer_use.avcs_perception import AVCSPerception


def _png_bytes(width=200, height=100):
    image = Image.new("RGB", (width, height), color=(10, 20, 30))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class _StubGrounder:
    checkpoint_metadata = {"evaluation": {"value": 0.9}}

    def __init__(self):
        self.frame_shape = None
        self.query = None

    def predict(self, frame, query):
        self.frame_shape = frame.shape
        self.query = query
        return SimpleNamespace(point_norm=(0.75, 0.5))


def test_goal_grounder_uses_checkpoint_gated_pixel_prediction(monkeypatch, tmp_path):
    checkpoint = tmp_path / "evaluated.pt"
    checkpoint.touch()
    monkeypatch.setenv("SONIC_AVCS_GROUNDING_CHECKPOINT", str(checkpoint))
    perception = AVCSPerception()
    model = _StubGrounder()
    perception._load_goal_grounder = lambda: model

    result = perception.ground_target_png(_png_bytes(), "Save the document", width=200, height=100)

    assert {key: value for key, value in result.items() if key != "latency_ms"} == {
        "engine": "avcs_goal_grounder",
        "x": 150,
        "y": 50,
        "point_norm": [0.75, 0.5],
        "held_out_point_accuracy": 0.9,
    }
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0
    assert model.frame_shape == (100, 200, 3)
    assert model.query == "Save the document"


def test_goal_grounder_requires_explicit_checkpoint_configuration(monkeypatch):
    monkeypatch.delenv("SONIC_AVCS_GROUNDING_CHECKPOINT", raising=False)
    perception = AVCSPerception()

    with pytest.raises(RuntimeError, match="is not configured"):
        perception.ground_target_png(_png_bytes(), "Save", width=200, height=100)


def test_goal_grounder_rejects_frame_dimension_mismatch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "evaluated.pt"
    checkpoint.touch()
    monkeypatch.setenv("SONIC_AVCS_GROUNDING_CHECKPOINT", str(checkpoint))
    perception = AVCSPerception()

    with pytest.raises(ValueError, match="dimensions do not match"):
        perception.ground_target_png(_png_bytes(), "Save", width=100, height=100)
