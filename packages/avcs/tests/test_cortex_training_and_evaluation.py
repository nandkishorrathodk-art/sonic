"""AVCS cortex ML-layer honesty + evaluation tests.

These assert the properties that make the "AVCS learns to see" claim real
rather than decorative:

* the model is a real PyTorch network whose forward pass produces the head
  shapes the runtime consumes;
* training genuinely reduces loss and the checkpoint round-trips;
* held-out evaluation measures accuracy on seeds disjoint from training;
* weights resolution/download fails closed instead of pretending to fetch
  pre-trained weights that do not exist.
"""

from __future__ import annotations

import json

import pytest
import torch
from forge.vision.cortex.config import ModelConfig
from forge.vision.cortex.model import ArtificialVisualCortex
from forge.vision.cortex.training.evaluate import (
    EVAL_SEED_BASE,
    TRAIN_SEED_BASE,
    evaluate_cortex,
    load_cortex_checkpoint,
    write_evaluation_sidecar,
)


def test_cortex_is_a_real_network_with_expected_head_shapes():
    model = ArtificialVisualCortex(ModelConfig())
    params = sum(p.numel() for p in model.parameters())
    assert params > 100_000, "cortex should be a real network, not a stub"

    out = model(torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
    assert out["affordance_logits"].shape[1] == 5  # static/clickable/typeable/scrollable/draggable
    assert out["category_logits"].shape[1] == 8  # UI_CATEGORIES
    assert out["scene_latent"].shape[0] == 1
    assert out["temporal_memory"].shape[0] == 1


def test_train_and_heldout_split_seeds_are_disjoint():
    """Held-out evaluation must never reuse a training seed."""
    assert EVAL_SEED_BASE != TRAIN_SEED_BASE
    # A whole training run at 512 samples covers seeds [0, 512); the eval base
    # is far beyond that, so no overlap is possible.
    assert EVAL_SEED_BASE > 100_000


def test_evaluation_reports_real_metrics_and_is_deterministic():
    torch.manual_seed(0)
    model = ArtificialVisualCortex(ModelConfig())
    first = evaluate_cortex(model, samples=4)
    second = evaluate_cortex(model, samples=4)
    # Same weights + same seeds => identical accuracy/IoU measurements (not
    # random numbers). Latency is wall-clock and is therefore excluded.
    assert first.category_accuracy == second.category_accuracy
    assert first.affordance_pixel_accuracy == second.affordance_pixel_accuracy
    assert first.affordance_mean_iou == second.affordance_mean_iou
    assert 0.0 <= first.category_accuracy <= 1.0
    assert 0.0 <= first.affordance_pixel_accuracy <= 1.0
    assert 0.0 <= first.affordance_mean_iou <= 1.0
    assert first.latency_ms_mean > 0.0, "real inference latency must be measured"


def test_training_reduces_loss_and_checkpoint_round_trips(tmp_path):
    """A short real training run must reduce loss and reload exactly."""
    from forge.vision.cortex.config import CortexConfig
    from forge.vision.cortex.training.trainer import AVCSCloudTrainer

    cfg = CortexConfig()
    cfg.training.epochs = 3
    cfg.training.device = "cpu"
    cfg.training.use_amp = False
    trainer = AVCSCloudTrainer(config=cfg)
    trainer.dataloader = __import__(
        "forge.vision.cortex.data.dataset", fromlist=["create_dataloader"]
    ).create_dataloader(num_samples=24, batch_size=8)

    first = trainer.train_epoch(1)["loss"]
    for epoch in (2, 3):
        last = trainer.train_epoch(epoch)["loss"]
    assert last < first, f"training did not reduce loss ({first} -> {last})"

    checkpoint = tmp_path / "cortex.pt"
    torch.save(trainer.model.state_dict(), checkpoint)
    reloaded = load_cortex_checkpoint(checkpoint)
    # Compare the persisted tensors directly: a forward pass would advance
    # BatchNorm running statistics and make an equality check meaningless.
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key]), f"mismatch at {key}"


def test_load_missing_checkpoint_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cortex_checkpoint(tmp_path / "nope.pt")


def test_load_shape_mismatch_fails_closed(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"unexpected": torch.zeros(1)}, bad)
    with pytest.raises(Exception):
        load_cortex_checkpoint(bad)


def test_evaluation_sidecar_records_provenance(tmp_path):
    model = ArtificialVisualCortex(ModelConfig())
    evaluation = evaluate_cortex(model, samples=2)
    checkpoint = tmp_path / "cortex.pt"
    torch.save(model.state_dict(), checkpoint)

    sidecar = write_evaluation_sidecar(
        checkpoint, evaluation, dataset_provenance="synthetic_procedural_screens"
    )
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    # A metric without provenance is not evidence.
    assert payload["dataset_provenance"] == "synthetic_procedural_screens"
    assert payload["evaluation"]["samples"] == 2


def test_no_fabricated_pretrained_weights_url():
    """The module must not advertise a download that cannot work."""
    from avcs import weights

    assert weights.DEFAULT_WEIGHTS_URL is None
    with pytest.raises(RuntimeError):
        weights.download_weights(dest_path="/tmp/should_not_exist.pt")


def test_weights_path_returns_none_when_nothing_is_installed(monkeypatch):
    from avcs import weights

    monkeypatch.delenv("AVCS_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "/nonexistent-avcs-cache")
    monkeypatch.setattr(weights, "get_default_cache_dir", lambda: __import__("pathlib").Path("/nonexistent-avcs-cache"))
    assert weights.get_weights_path() is None


def test_load_avcs_weights_reports_false_without_a_checkpoint():
    """No checkpoint => False (caller must not treat the cortex as trained)."""
    from avcs.weights import load_avcs_weights

    cortex = ArtificialVisualCortex(ModelConfig())
    assert load_avcs_weights(cortex, path="/nonexistent/weights.pt") is False
