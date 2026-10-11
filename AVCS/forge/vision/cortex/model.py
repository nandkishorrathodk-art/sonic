"""Artificial Visual Cortex for Screens (AVCS) Unified Model.

Coordinates Retinal Foveation, Dual-path Backbone, Dorsal Stream (Affordances),
Ventral Stream (Semantics), Language Modulation, and Predictive Coding.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from forge.vision.cortex.config import CortexConfig, ModelConfig
from forge.vision.cortex.core.backbone import CorticalBackbone
from forge.vision.cortex.core.dorsal import AFFORDANCE_CLASSES, DorsalAffordanceStream
from forge.vision.cortex.core.fovea import RetinalFoveation, RetinalView
from forge.vision.cortex.core.modulation import LanguageModulation
from forge.vision.cortex.core.predictive import (
    STABILITY_STATES,
    PredictiveCodingEngine,
    decode_stability,
)
from forge.vision.cortex.core.servoing import VisualMotorState, VisualServoingEngine
from forge.vision.cortex.core.ventral import UI_CATEGORIES, VentralSemanticStream


@dataclass
class CortexState:
    """Rich, typed visual perception state emitted every tick."""
    visual_latent: np.ndarray             # (latent_dim,) compact semantic embedding
    primary_category: str                 # "browser_webpage", "editor_code", etc.
    category_confidence: float
    text_density: float                   # [0, 1] text saturation
    stability_state: str                  # "stable", "animating", "transitioning", "surprise_event"
    surprise_score: float                 # Temporal prediction error
    fixation: Tuple[float, float]         # Current gaze point (x, y) in [0, 1]
    motor_flow: Tuple[float, float] = (0.0, 0.0) # Closed-loop hand-eye steering vector (dx, dy)
    contact_acquired: bool = False        # True when hand/cursor aligns with gaze
    action_hotspots: List[Tuple[float, float, str, float]] = None # Interactive affordances
    affordance_map: Optional[np.ndarray] = None           # (H_grid, W_grid) dense discrete affordance mask
    query_target: Optional[Tuple[float, float]] = None    # Language grounded gaze target
    query_relevance_map: Optional[np.ndarray] = None
    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_category": self.primary_category,
            "category_confidence": round(self.category_confidence, 3),
            "text_density": round(self.text_density, 3),
            "stability_state": self.stability_state,
            "surprise_score": round(self.surprise_score, 4),
            "fixation": (round(self.fixation[0], 3), round(self.fixation[1], 3)),
            "motor_flow": (round(self.motor_flow[0], 3), round(self.motor_flow[1], 3)),
            "contact_acquired": self.contact_acquired,
            "action_hotspots": [
                (round(x, 3), round(y, 3), act, round(c, 3))
                for x, y, act, c in (self.action_hotspots or [])
            ],
            "query_target": (
                (round(self.query_target[0], 3), round(self.query_target[1], 3))
                if self.query_target else None
            ),
            "inference_ms": round(self.inference_ms, 2),
        }


class ArtificialVisualCortex(nn.Module):
    """The Complete Artificial Visual Cortex for Screens (AVCS)."""

    def __init__(self, config: Optional[ModelConfig | CortexConfig] = None) -> None:
        super().__init__()
        if config is None:
            self.cfg = ModelConfig()
        elif isinstance(config, CortexConfig):
            self.cfg = config.model
        else:
            self.cfg = config

        # 1. Retinal Foveation (Sensor view extraction)
        self.retina = RetinalFoveation(
            peripheral_size=self.cfg.retina.peripheral_size,
            foveal_size=self.cfg.retina.foveal_size,
            foveal_coverage_ratio=self.cfg.retina.foveal_coverage_ratio,
            saccade_smoothing=self.cfg.retina.saccade_smoothing,
        )

        # 2. Cortical Neural Backbone
        self.backbone = CorticalBackbone(
            in_channels=self.cfg.in_channels,
            feature_dim=self.cfg.feature_dim,
        )

        # 3. Dorsal Stream ("WHERE/HOW" Affordances)
        self.dorsal = DorsalAffordanceStream(
            in_dim=self.cfg.feature_dim,
            num_classes=self.cfg.num_affordance_classes,
        )

        # 4. Ventral Stream ("WHAT" Semantics)
        self.ventral = VentralSemanticStream(
            in_dim=self.cfg.feature_dim,
            latent_dim=self.cfg.latent_dim,
        )

        # 5. Top-down Language Modulation
        self.language = LanguageModulation(
            visual_dim=self.cfg.feature_dim,
            text_dim=self.cfg.latent_dim,
        )

        # 6. Temporal Predictive Coding (World Model)
        self.predictive = PredictiveCodingEngine(
            latent_dim=self.cfg.latent_dim,
        )

        # 7. Biological Hand-Eye Servoing Engine
        self.servoing = VisualServoingEngine()

    def reset_temporal_memory(self) -> None:
        """Reset internal temporal world model."""
        self.predictive.reset()

    def forward(
        self,
        peripheral: torch.Tensor,
        foveal: torch.Tensor,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Single-pass forward execution."""
        # 1. Visual Feature Extraction & Multi-scale Fusion
        fused_feat, feat_p, feat_f = self.backbone(peripheral, foveal)

        # 2. Dorsal Stream: Affordance Maps
        affordance_logits = self.dorsal(fused_feat)

        # 3. Ventral Stream: Category & Scene Latent
        ventral_out = self.ventral(fused_feat)
        scene_latent = ventral_out["latent"]

        # 4. Predictive Coding: Temporal Surprise & Stability
        updated_memory, surprise_tensor, stability_logits = (
            self.predictive(scene_latent)
        )

        # 5. Optional Language Modulation (Zero-generation grounding)
        rel_map, query_coords = None, None
        if text_embedding is not None:
            rel_map, query_coords = self.language(fused_feat, text_embedding)

        return {
            "fused_feat": fused_feat,
            "affordance_logits": affordance_logits,
            "category_logits": ventral_out["category_logits"],
            "text_density": ventral_out["text_density"],
            "scene_latent": scene_latent,
            "temporal_memory": updated_memory,
            "surprise_score": surprise_tensor,
            "stability_logits": stability_logits,
            "relevance_map": rel_map,
            "query_coords": query_coords,
        }

    @torch.no_grad()
    def perceive(
        self,
        frame: np.ndarray | torch.Tensor,
        target_fixation: Optional[Tuple[float, float]] = None,
        cursor_position: Optional[Tuple[float, float]] = None,
        text_embedding: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
    ) -> CortexState:
        """High-level real-time perception interface."""
        t0 = time.perf_counter()

        if device is None:
            device = next(self.parameters()).device

        # 1. Retinal Foveation View Extraction
        retinal_view = self.retina.extract_views(frame, target_fixation=target_fixation)
        periph = retinal_view.peripheral.to(device)
        fov = retinal_view.foveal.to(device)

        if text_embedding is not None:
            text_embedding = text_embedding.to(device)

        # 2. Execute Neural Forward Pass
        out = self.forward(periph, fov, text_embedding=text_embedding)

        # 3. Extract High-Level Readouts
        cat_probs = torch.softmax(out["category_logits"], dim=-1)[0]
        cat_idx = int(torch.argmax(cat_probs).item())
        cat_name = UI_CATEGORIES[cat_idx] if cat_idx < len(UI_CATEGORIES) else "unknown"
        cat_conf = float(cat_probs[cat_idx].item())

        text_dense = float(out["text_density"][0].item())
        surprise = max(0.0, min(1.0, float(out["surprise_score"].mean().item())))
        stability = decode_stability(
            out["stability_logits"],
            surprise_score=surprise,
            prev_state=getattr(self.predictive, "current_state", None),
        )
        self.predictive.current_state = stability

        # 4. Extract Top Action Hotspots (Clickable / Typeable)
        hotspots: List[Tuple[float, float, str, float]] = []
        for act in ["clickable", "typeable"]:
            spots = self.dorsal.find_action_hotspots(
                out["fused_feat"], action_type=act, top_k=2, threshold=0.25
            )
            for x, y, conf in spots:
                hotspots.append((x, y, act, conf))

        # 5. Extract Language Grounding (if query passed)
        query_target = None
        rel_map_np = None
        if out["query_coords"] is not None:
            coords = out["query_coords"][0].cpu().numpy()
            query_target = (float(coords[0]), float(coords[1]))
            rel_map_np = out["relevance_map"][0, 0].cpu().numpy()

        # 6. Biological Hand-Eye Coordination & Visual Servoing
        motor_flow = (0.0, 0.0)
        contact_acquired = False
        target_gaze = query_target or retinal_view.fixation
        if cursor_position is not None:
            servo_state = self.servoing.compute_motor_flow(
                gaze_target=target_gaze,
                current_cursor=cursor_position,
                confidence=cat_conf,
            )
            motor_flow = servo_state.motor_velocity
            contact_acquired = servo_state.contact_acquired

        # Dense affordance argmax map (H_grid, W_grid)
        aff_map_np = torch.argmax(out["affordance_logits"][0], dim=0).cpu().numpy()

        dt = (time.perf_counter() - t0) * 1000.0

        return CortexState(
            visual_latent=out["scene_latent"][0].cpu().numpy(),
            primary_category=cat_name,
            category_confidence=cat_conf,
            text_density=text_dense,
            stability_state=stability,
            surprise_score=surprise,
            fixation=retinal_view.fixation,
            motor_flow=motor_flow,
            contact_acquired=contact_acquired,
            action_hotspots=hotspots,
            affordance_map=aff_map_np,
            query_target=query_target,
            query_relevance_map=rel_map_np,
            inference_ms=dt,
        )

    def load_weights(self, weights_path: str, strict: bool = False) -> None:
        """Loads trained weights checkpoint into the cortex modules."""
        state = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(state, strict=strict)

