"""Tests for Scaled 2.0M Parameter Cortical Visual Foundation Model (Cortex-2M)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from avcs.cortex_2m import (
    AffordanceHead2M,
    CorticalBackbone2M,
    HumanEye2M,
    LanguageGrounder2M,
    UnifiedCortex2M,
    UNIVERSAL_AFFORDANCE_CLASSES,
    UNIVERSAL_APP_CATEGORIES,
    VentralStream2M,
)
from forge.vision.capture import SyntheticScreen


def test_universal_ontology_counts():
    """Validates that universal desktop ontology covers 32 apps and 24 affordance geometries."""
    assert len(UNIVERSAL_APP_CATEGORIES) == 32
    assert len(UNIVERSAL_AFFORDANCE_CLASSES) == 24
    assert "editor_code" in UNIVERSAL_APP_CATEGORIES
    assert "cad_3d_modeling" in UNIVERSAL_APP_CATEGORIES
    assert "data_spreadsheet" in UNIVERSAL_APP_CATEGORIES
    assert "terminal_shell" in UNIVERSAL_APP_CATEGORIES
    assert "button_primary" in UNIVERSAL_AFFORDANCE_CLASSES
    assert "code_editor_line" in UNIVERSAL_AFFORDANCE_CLASSES
    assert "spreadsheet_cell" in UNIVERSAL_AFFORDANCE_CLASSES


def test_cortex_2m_parameter_scale():
    """Asserts that UnifiedCortex2M scales to ~2.40M parameters."""
    cortex = UnifiedCortex2M(feature_dim=256)
    total_params = cortex.total_parameters

    # Verify scaling between 2.2M and 2.6M
    assert 2_200_000 <= total_params <= 2_600_000, f"Expected ~2.4M params, got {total_params:,}"

    backbone_p = sum(p.numel() for p in cortex.backbone.parameters())
    affordance_p = sum(p.numel() for p in cortex.affordance.parameters())
    ventral_p = sum(p.numel() for p in cortex.ventral.parameters())
    grounder_p = sum(p.numel() for p in cortex.grounder.parameters())

    assert backbone_p > 700_000
    assert affordance_p > 1_000_000
    assert ventral_p > 200_000
    assert grounder_p > 100_000


def test_cortex_2m_forward_pass():
    """Tests forward pass tensor dimensions."""
    cortex = UnifiedCortex2M(feature_dim=256)
    cortex.eval()

    x = torch.randn(2, 3, 128, 128)
    text_emb = torch.randn(2, 128)

    with torch.no_grad():
        out = cortex(x, text_emb=text_emb)

    assert "feat" in out
    assert out["feat"].shape == (2, 256, 8, 8)

    assert "affordance_logits" in out
    assert out["affordance_logits"].shape == (2, 24, 8, 8)

    assert "visual_latent_raw" in out
    assert out["visual_latent_raw"].shape == (2, 256)

    assert "visual_latent" in out
    assert out["visual_latent"].shape == (2, 256)

    assert "category_logits" in out
    assert out["category_logits"].shape == (2, 32)

    assert "grounding_map" in out
    assert out["grounding_map"].shape == (2, 1, 8, 8)


def test_language_grounder_uses_training_embedding_dimension():
    grounder = LanguageGrounder2M(visual_dim=256, text_dim=128)
    feat = torch.randn(1, 256, 8, 8)
    assert grounder(feat, torch.randn(1, 128)).shape == (1, 1, 8, 8)
    with pytest.raises(ValueError, match="text_emb must have shape"):
        grounder(feat, torch.randn(1, 256))


def test_human_eye_2m_requires_complete_weights_for_semantics(tmp_path):
    class StaticSource:
        def grab(self):
            return np.full((64, 96, 3), 80, dtype=np.uint8)

    missing_path = tmp_path / "missing.pt"
    eye = HumanEye2M(
        source=StaticSource(),
        weights_path=str(missing_path),
    )
    result = eye.perceive()
    assert result["weights_loaded"] is False
    assert result["primary_category"] == "unknown_window"
    assert result["category_confidence"] == 0.0
    assert result["affordances"] == []
    with pytest.raises(RuntimeError, match="complete Cortex-2M checkpoint"):
        eye.perceive(text_emb=torch.randn(1, 128))
    eye.close()

    with pytest.raises(FileNotFoundError, match="weights are required"):
        HumanEye2M(source=StaticSource(), weights_path=str(missing_path), require_weights=True)


def test_human_eye_2m_rejects_partial_checkpoint(tmp_path):
    class StaticSource:
        def grab(self):
            return np.zeros((32, 48, 3), dtype=np.uint8)

    checkpoint_path = tmp_path / "partial.pt"
    torch.save({"model_state_dict": {}}, checkpoint_path)
    with pytest.raises(RuntimeError, match="Failed to load complete Cortex-2M checkpoint"):
        HumanEye2M(source=StaticSource(), weights_path=str(checkpoint_path))


def test_human_eye_2m_static_frame_still_produces_text_grounding(tmp_path):
    class StaticSource:
        def grab(self):
            frame = np.zeros((96, 128, 3), dtype=np.uint8)
            frame[20:60, 30:90] = 200
            return frame

    checkpoint_path = tmp_path / "complete.pt"
    torch.save(
        {"model_state_dict": UnifiedCortex2M().state_dict()},
        checkpoint_path,
    )
    eye = HumanEye2M(source=StaticSource(), weights_path=str(checkpoint_path))
    query = torch.randn(1, 128)
    first = eye.perceive(text_emb=query)
    second = eye.perceive(text_emb=query)
    assert first["weights_loaded"] is True
    assert first["grounding_map"].shape == (1, 1, 8, 8)
    assert second["is_static"] is True
    assert second["grounding_map"].shape == (1, 1, 8, 8)
    eye.close()


def test_cortex_2m_affordance_hotspots():
    """Tests hotspot extraction on synthetic feature map."""
    cortex = UnifiedCortex2M(feature_dim=256)
    cortex.eval()

    feat = torch.randn(1, 256, 8, 8)
    hotspots = cortex.affordance.find_hotspots(feat, top_k=3, threshold=0.0)

    assert len(hotspots) <= 3
    for x, y, aff_name, conf in hotspots:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert aff_name in UNIVERSAL_AFFORDANCE_CLASSES
        assert 0.0 <= conf <= 1.0


def test_human_eye_2m_perception(tmp_path):
    """Tests honest untrained-state reporting in the perception loop."""
    class StaticSource:
        def __init__(self):
            self.frame = np.full((480, 640, 3), 60, dtype=np.uint8)
            # Add static shapes
            self.frame[100:200, 150:350] = 200
        def grab(self):
            return self.frame

    source = StaticSource()
    eye = HumanEye2M(source=source, weights_path=str(tmp_path / "missing-cortex-2m-test.pt"))

    # Frame 1: Dynamic initial pass (runs full 2.4M forward)
    res1 = eye.perceive()
    assert res1["frame_index"] == 1
    assert not res1["is_static"]
    assert res1["primary_category"] == "unknown_window"
    assert res1["category_confidence"] == 0.0
    assert res1["parameters"] > 2_200_000
    assert res1["affordances"] == []
    assert res1["weights_loaded"] is False

    # Frame 2: Trigger static screen early exit (<0.2ms)
    res2 = eye.perceive()
    assert res2["frame_index"] == 2
    assert res2["is_static"]
    eye.close()


def test_lateral_cross_stream_flow():
    """Verifies that dorsal affordance context informs ventral recognition via lateral projections."""
    cortex = UnifiedCortex2M(feature_dim=256)
    cortex.eval()

    feat = torch.randn(2, 256, 8, 8)
    aff_ctx1 = torch.zeros(2, 24, 8, 8)
    aff_ctx2 = torch.ones(2, 24, 8, 8) * 5.0

    with torch.no_grad():
        latent1, logits1 = cortex.ventral(feat, dorsal_context=aff_ctx1)
        latent2, logits2 = cortex.ventral(feat, dorsal_context=aff_ctx2)

    # Different dorsal affordance distributions must yield distinct modulated ventral outputs
    assert not torch.allclose(latent1, latent2, atol=1e-3)
    assert not torch.allclose(logits1, logits2, atol=1e-3)


def test_anti_collapse_variance_regularization():
    """Verifies mathematical properties of VICReg variance loss and normalized visual latents."""
    from avcs.training.vicreg_safeguard import vicreg_variance_loss, normalize_embedding

    # 1. Collapsed state (small variance batch is penalized)
    low_var_latent = torch.full((16, 256), 0.2) + torch.randn(16, 256) * 1e-3
    loss_collapsed = vicreg_variance_loss(low_var_latent, target_std=1.0)
    assert loss_collapsed.item() > 0.95, f"Expected heavy penalty on low variance, got {loss_collapsed.item()}"

    # 2. Diverse state (standard normal samples with unit variance)
    diverse_latent = torch.randn(64, 256)
    loss_diverse = vicreg_variance_loss(diverse_latent, target_std=1.0)
    assert loss_diverse.item() < 0.25, f"Expected low penalty on diverse representations, got {loss_diverse.item()}"

    # 3. Model outputs both unnormalized projection and L2-normalized embedding
    cortex = UnifiedCortex2M(feature_dim=256)
    x = torch.randn(4, 3, 128, 128)
    out = cortex(x)

    # visual_latent_raw has unconstrained norm
    raw_norms = torch.norm(out["visual_latent_raw"], p=2, dim=-1)
    assert not torch.allclose(raw_norms, torch.ones_like(raw_norms), atol=1e-3)

    # visual_latent is strictly unit-L2 normalized
    unit_norms = torch.norm(out["visual_latent"], p=2, dim=-1)
    assert torch.allclose(unit_norms, torch.ones_like(unit_norms), atol=1e-5)
