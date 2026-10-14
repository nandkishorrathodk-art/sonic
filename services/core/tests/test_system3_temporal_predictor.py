from __future__ import annotations

import numpy as np
import pytest

from sonic.computer_use.system3_predictor import System3TemporalPredictor


@pytest.mark.no_live_infra
def test_system3_reports_initialization_and_stable_persistence():
    predictor = System3TemporalPredictor()
    latent = np.full((16, 24), 0.4, dtype=np.float32)

    first = predictor.observe(latent, frame_index=1)
    second = predictor.observe(latent, frame_index=2)

    assert first["prediction_available"] is False
    assert first["stability_state"] == "initializing"
    assert second["prediction_available"] is True
    assert second["surprise_score"] == 0.0
    assert second["stability_state"] == "stable"
    assert second["action_authorization"] is False


@pytest.mark.no_live_infra
def test_system3_marks_large_unexpected_scene_change_as_advisory_surprise():
    predictor = System3TemporalPredictor()

    predictor.observe(np.zeros((16, 24), dtype=np.float32), frame_index=1)
    changed = predictor.observe(np.ones((16, 24), dtype=np.float32), frame_index=2)

    assert changed["surprise_score"] == 1.0
    assert changed["stability_state"] == "surprise_event"
    assert changed["trained"] is False
    assert changed["action_authorization"] is False


@pytest.mark.no_live_infra
@pytest.mark.parametrize(
    ("latent", "frame_index", "message"),
    [
        (np.zeros((0, 0), dtype=np.float32), 1, "two-dimensional"),
        (np.full((2, 2), np.nan, dtype=np.float32), 1, "finite"),
        (np.full((2, 2), 1.1, dtype=np.float32), 1, r"\[0, 1\]"),
        (np.zeros((2, 2), dtype=np.float32), True, "positive integer"),
    ],
)
def test_system3_rejects_invalid_pixels_or_frame_identity(latent, frame_index, message):
    with pytest.raises(ValueError, match=message):
        System3TemporalPredictor().observe(latent, frame_index=frame_index)


@pytest.mark.no_live_infra
def test_system3_rejects_replayed_or_out_of_order_frames():
    predictor = System3TemporalPredictor()
    latent = np.zeros((2, 2), dtype=np.float32)
    predictor.observe(latent, frame_index=3)

    with pytest.raises(ValueError, match="increase monotonically"):
        predictor.observe(latent, frame_index=3)
