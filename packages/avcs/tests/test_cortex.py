"""Comprehensive Unit Tests for Artificial Visual Cortex for Screens (AVCS)."""
from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
import torch

from forge.vision.cortex import (
    AFFORDANCE_CLASSES,
    UI_CATEGORIES,
    AVCSMultiTaskLoss,
    ArtificialVisualCortex,
    CortexConfig,
    CorticalBackbone,
    DorsalAffordanceStream,
    LanguageModulation,
    PredictiveCodingEngine,
    ProceduralScreenGenerator,
    RetinalFoveation,
    ScreenDataset,
    VentralSemanticStream,
    create_dataloader,
    export_torchscript,
)


def test_retinal_foveation_shapes():
    retina = RetinalFoveation(peripheral_size=(128, 128), foveal_size=(128, 128))
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    view = retina.extract_views(frame, target_fixation=(0.25, 0.75))

    assert view.peripheral.shape == (1, 3, 128, 128)
    assert view.foveal.shape == (1, 3, 128, 128)
    assert 0.0 <= view.fixation[0] <= 1.0
    assert 0.0 <= view.fixation[1] <= 1.0


def test_cortical_backbone_fusion():
    backbone = CorticalBackbone(in_channels=3, feature_dim=64)
    p = torch.randn(2, 3, 128, 128)
    f = torch.randn(2, 3, 128, 128)

    fused, feat_p, feat_f = backbone(p, f)
    assert fused.shape == (2, 64, 8, 8)
    assert feat_p.shape == (2, 64, 8, 8)
    assert feat_f.shape == (2, 64, 8, 8)


def test_dorsal_affordance_stream():
    dorsal = DorsalAffordanceStream(in_dim=64, num_classes=len(AFFORDANCE_CLASSES))
    feats = torch.randn(1, 64, 8, 8)
    logits = dorsal(feats)
    # Upsampled 2x to 16x16
    assert logits.shape == (1, len(AFFORDANCE_CLASSES), 16, 16)

    probs = dorsal.get_affordance_map(feats)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 16, 16), atol=1e-4)

    hotspots = dorsal.find_action_hotspots(feats, action_type="clickable", top_k=3, threshold=0.0)
    assert len(hotspots) <= 3


def test_ventral_semantic_stream():
    ventral = VentralSemanticStream(in_dim=64, latent_dim=128)
    feats = torch.randn(2, 64, 8, 8)
    out = ventral(feats)

    assert out["latent"].shape == (2, 128)
    assert out["category_logits"].shape == (2, len(UI_CATEGORIES))
    assert out["text_density"].shape == (2, 1)

    cat, conf = ventral.get_primary_category(feats[0:1])
    assert cat in UI_CATEGORIES
    assert 0.0 <= conf <= 1.0


def test_language_modulation():
    mod = LanguageModulation(visual_dim=64, text_dim=128)
    feats = torch.randn(1, 64, 8, 8)
    text = torch.randn(1, 128)

    rel_map, coords = mod(feats, text)
    assert rel_map.shape == (1, 1, 8, 8)
    assert torch.allclose(rel_map.sum(dim=[-2, -1]), torch.tensor([[1.0]]), atol=1e-4)
    assert coords.shape == (1, 2)
    assert 0.0 <= coords[0, 0].item() <= 1.0
    assert 0.0 <= coords[0, 1].item() <= 1.0


def test_predictive_coding_surprise():
    engine = PredictiveCodingEngine(latent_dim=128)
    z1 = torch.randn(1, 128)
    mem1, s1, logits1 = engine(z1)

    # Identical latent: surprise should be low
    mem2, s2, logits2 = engine(z1)
    # Radically different latent: surprise should spike
    z_diff = -z1 * 2.0
    mem3, s3, logits3 = engine(z_diff)

    assert s3.item() > s2.item()


def test_procedural_screen_generator():
    gen = ProceduralScreenGenerator(width=320, height=180)
    data = gen.generate(category="editor_code", seed=42)

    assert data.frame.shape == (180, 320, 3)
    assert data.affordance_mask.shape == (180, 320)
    assert data.category == "editor_code"
    assert data.category_idx == UI_CATEGORIES.index("editor_code")
    # Verify non-empty affordance mask
    assert np.any(data.affordance_mask > 0)


def test_screen_dataset_and_dataloader():
    dl = create_dataloader(num_samples=10, batch_size=2, shuffle=False)
    batch = next(iter(dl))

    assert "frame_t" in batch
    assert "frame_next" in batch
    assert "affordance_mask" in batch
    assert "category_idx" in batch
    assert batch["frame_t"].shape == (2, 3, 360, 640)
    assert batch["affordance_mask"].shape == (2, 32, 32)


def test_multitask_loss():
    loss_fn = AVCSMultiTaskLoss()
    B = 2
    dummy_out = {
        "temporal_memory": torch.randn(B, 128),
        "affordance_logits": torch.randn(B, 5, 32, 32),
        "category_logits": torch.randn(B, len(UI_CATEGORIES)),
        "query_coords": torch.rand(B, 2),
    }
    latent_next = torch.randn(B, 128)
    aff_targets = torch.randint(0, 5, (B, 32, 32))
    cat_targets = torch.randint(0, len(UI_CATEGORIES), (B,))
    coord_targets = torch.rand(B, 2)

    losses = loss_fn(dummy_out, latent_next, aff_targets, cat_targets, coord_targets)
    assert "loss" in losses
    assert losses["loss"].item() > 0.0


def test_avcs_end_to_end_perceive():
    cfg = CortexConfig()
    cfg.model.feature_dim = 64
    cfg.model.latent_dim = 128
    cfg.model.retina.peripheral_size = (64, 64)
    cfg.model.retina.foveal_size = (64, 64)

    cortex = ArtificialVisualCortex(cfg)
    frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    query = torch.randn(1, 128)

    state = cortex.perceive(frame, text_embedding=query)
    assert state.primary_category in UI_CATEGORIES
    assert state.visual_latent.shape == (128,)
    assert state.query_target is not None
    assert state.inference_ms > 0.0
    assert isinstance(state.to_dict(), dict)


def test_export_torchscript():
    cfg = CortexConfig()
    cfg.model.feature_dim = 32
    cfg.model.latent_dim = 64
    cfg.model.retina.peripheral_size = (64, 64)
    cfg.model.retina.foveal_size = (64, 64)

    cortex = ArtificialVisualCortex(cfg)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        exported = export_torchscript(cortex, save_path=path)
        assert os.path.exists(exported)
        assert os.path.getsize(exported) > 0
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_visual_servoing():
    from forge.vision.cortex import VisualServoingEngine

    servo = VisualServoingEngine(foveal_contact_radius=0.03)

    # Gaze is at (0.8, 0.8), cursor is at (0.2, 0.2)
    state_far = servo.compute_motor_flow(gaze_target=(0.8, 0.8), current_cursor=(0.2, 0.2))
    assert not state_far.contact_acquired
    assert state_far.motor_velocity[0] > 0.0 # Velocity points right
    assert state_far.motor_velocity[1] > 0.0 # Velocity points down
    assert state_far.distance_to_target > 0.5

    # Gaze and cursor aligned within foveal contact radius
    state_near = servo.compute_motor_flow(gaze_target=(0.50, 0.50), current_cursor=(0.51, 0.51))
    assert state_near.contact_acquired
    assert state_near.motor_velocity == (0.0, 0.0)


def test_screen_eye_sdk():
    from forge.vision.cortex import ScreenEye
    from forge.vision.capture import SyntheticScreen

    screen = SyntheticScreen(width=320, height=180)
    eye = ScreenEye(fps=25, source=screen)

    state = eye.perceive_now()
    assert state is not None
    assert state.primary_category in UI_CATEGORIES
    assert isinstance(state.motor_flow, tuple)
    assert isinstance(state.contact_acquired, bool)

    # Test hand-eye steering
    vel, aligned = eye.steer_hand(current_cursor=(0.1, 0.1))
    assert isinstance(vel, tuple)
    assert isinstance(aligned, bool)


def test_real_ui_and_unified_dataset():
    from forge.vision.cortex.data import RealUIDataset, ScreenDataset, UnifiedScreenDataset

    real_ds = RealUIDataset(synthetic_fallback_count=10)
    assert len(real_ds) == 10
    sample = real_ds[0]
    assert "frame_t" in sample
    assert "affordance_mask" in sample
    assert "category_idx" in sample

    synth_ds = ScreenDataset(num_samples=10)
    unified = UnifiedScreenDataset(synthetic_dataset=synth_ds, real_dataset=real_ds, synthetic_ratio=0.5)
    assert len(unified) == 20
    u_sample = unified[3]
    assert u_sample["frame_t"].shape[0] == 3


def test_visual_hud_renderer():
    from forge.vision.cortex.dashboard import AVCSVisualHUD
    from forge.vision.cortex import ArtificialVisualCortex, VisualServoingEngine
    from forge.vision.capture import SyntheticScreen

    screen = SyntheticScreen(width=320, height=180)
    frame = screen.grab()

    cortex = ArtificialVisualCortex()
    state = cortex.perceive(frame)

    servoing = VisualServoingEngine()
    motor = servoing.compute_motor_flow(gaze_target=state.fixation, current_cursor=(0.1, 0.1))

    hud = AVCSVisualHUD(canvas_size=(480, 270))
    img = hud.render(frame, state, motor, dt_ms=25.0, fps=40.0)
    assert img.size == (480, 270)


def test_realistic_desktop_renderer():
    from forge.vision.cortex.benchmark_real_desktop import RealisticDesktopRenderer

    renderer = RealisticDesktopRenderer(width=320, height=180)
    vscode = renderer.render_vscode()
    assert vscode.frame.shape == (180, 320, 3)
    assert vscode.category_target == "editor_code"

    chrome = renderer.render_chrome()
    assert chrome.frame.shape == (180, 320, 3)
    assert chrome.category_target == "browser_webpage"


