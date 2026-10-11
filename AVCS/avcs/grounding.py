"""Language Gaze Grounding & Dual-Fovea Acuity Engine.

Implements Phase 2 of the AVCS Universal Real-Time Human Eye Platform:
1. Open-Vocabulary Language Grounding: Maps natural language queries
   ('Submit Order', 'search bar in top left', 'close icon') directly to spatial
   gaze coordinates in <3ms using character n-gram text projection and spatial
   cross-correlation.
2. Dual-Fovea Acuity Engine: Replicates biological foveation by extracting a
   high-resolution zoom crop centered on fixation gaze to classify element type
   (button, input, icon, tab) and regress exact bounding box boundaries in <2ms.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.vision.cortex.core.modulation import (
    CharNGramEncoder,
    LanguageModulation,
    dsnt_soft_argmax,
)

FOVEAL_ELEMENT_TYPES = [
    "button",
    "input_field",
    "icon",
    "tab",
    "link",
    "text_label",
    "container",
]


@dataclasses.dataclass(frozen=True)
class GroundingResult:
    """Result of top-down natural language gaze grounding."""
    query: str
    target_coord: Tuple[float, float]       # Normalized (x, y) in [0, 1]
    confidence: float                       # Saliency confidence in [0, 1]
    relevance_map: Optional[np.ndarray]     # 2D spatial correlation map
    snapped_to_affordance: bool             # True if snapped to nearest conscious affordance
    latency_ms: float                       # Processing latency in milliseconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "target_coord": (round(self.target_coord[0], 3), round(self.target_coord[1], 3)),
            "confidence": round(self.confidence, 3),
            "snapped_to_affordance": self.snapped_to_affordance,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclasses.dataclass(frozen=True)
class FovealInspection:
    """High-acuity central foveation analysis on fixation target."""
    centroid: Tuple[float, float]           # Screen coordinate (fx, fy) in [0, 1]
    element_kind: str                       # 'button', 'input_field', 'icon', 'tab', etc.
    confidence: float                       # Classification confidence
    bbox: Tuple[float, float, float, float] # Global screen bounds [xmin, ymin, xmax, ymax]
    contrast_score: float                   # Local sharpness / visual edge density
    foveal_crop_shape: Tuple[int, int]      # Resolution of extracted crop (H, W)
    latency_ms: float                       # Inspection latency in milliseconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centroid": (round(self.centroid[0], 3), round(self.centroid[1], 3)),
            "element_kind": self.element_kind,
            "confidence": round(self.confidence, 3),
            "bbox": [round(b, 3) for b in self.bbox],
            "contrast_score": round(self.contrast_score, 3),
            "latency_ms": round(self.latency_ms, 2),
        }


class LanguageGazeGrounder(nn.Module):
    """Top-down semantic gaze grounder with character n-grams & spatial modulation."""

    def __init__(self, visual_dim: int = 128, text_dim: int = 128) -> None:
        super().__init__()
        self.text_encoder = CharNGramEncoder(out_dim=text_dim)
        self.modulator = LanguageModulation(visual_dim=visual_dim, text_dim=text_dim)
        self._query_cache: Dict[str, torch.Tensor] = {}

    def get_query_embedding(self, query: str, device: torch.device) -> torch.Tensor:
        """Encodes or retrieves cached query embedding."""
        clean_q = query.strip().lower()
        if clean_q in self._query_cache:
            return self._query_cache[clean_q].to(device)

        with torch.inference_mode():
            emb = self.text_encoder(clean_q, device=device)
            # Cache up to 128 common queries
            if len(self._query_cache) < 128:
                self._query_cache[clean_q] = emb.cpu()
            return emb

    @torch.inference_mode()
    def ground(
        self,
        visual_features: torch.Tensor,
        query: str,
        affordances: Optional[List[Tuple[float, float, str, float]]] = None,
        snap_distance_threshold: float = 0.12,
    ) -> GroundingResult:
        """Grounds a text query onto visual features in <3ms on CPU."""
        t0 = time.perf_counter()
        device = visual_features.device
        text_emb = self.get_query_embedding(query, device=device)

        # 1. Forward spatial modulation
        rel_map, target_coords = self.modulator(visual_features, text_emb)
        # target_coords: (1, 2)
        raw_x = float(target_coords[0, 0].item())
        raw_y = float(target_coords[0, 1].item())

        # 2. Extract confidence & relevance map
        rel_np = rel_map[0, 0].cpu().numpy()
        peak_val = float(np.max(rel_np))
        mean_val = float(np.mean(rel_np))
        # Peak-to-average ratio as confidence metric
        confidence = float(min(1.0, max(0.1, (peak_val - mean_val) * 8.0)))

        # 3. Affordance snapping: snap to nearest conscious affordance hotspot if close
        snapped = False
        final_x, final_y = raw_x, raw_y
        if affordances:
            best_dist = float("inf")
            best_hotspot = None
            for hx, hy, _, h_conf in affordances:
                d = math.hypot(hx - raw_x, hy - raw_y)
                if d < snap_distance_threshold and d < best_dist:
                    best_dist = d
                    best_hotspot = (hx, hy, h_conf)

            if best_hotspot is not None:
                final_x, final_y = best_hotspot[0], best_hotspot[1]
                snapped = True
                confidence = min(1.0, confidence * 1.25)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return GroundingResult(
            query=query,
            target_coord=(final_x, final_y),
            confidence=confidence,
            relevance_map=rel_np,
            snapped_to_affordance=snapped,
            latency_ms=dt_ms,
        )


class FovealAcuityHead(nn.Module):
    """Lightweight convolutional neural network for high-acuity foveal inspection."""

    def __init__(self, in_channels: int = 3, num_classes: int = len(FOVEAL_ELEMENT_TYPES)) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, num_classes)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        logits = self.classifier(feat)
        bbox = self.bbox_regressor(feat)
        return logits, bbox


class FovealAcuityEngine:
    """Extracts dynamic high-resolution foveal zoom crops and executes fine-grained inspection."""

    def __init__(
        self,
        device: str = "cpu",
        fovea_radius_frac: float = 0.08,
        input_size: Tuple[int, int] = (96, 96),
    ) -> None:
        self.device = torch.device(device)
        self.radius_frac = fovea_radius_frac
        self.input_size = input_size
        self.head = FovealAcuityHead().to(self.device).eval()

    def crop_fovea(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Extracts high-resolution un-downsampled foveal crop around fixation point."""
        H, W = frame.shape[:2]
        cx, cy = centroid
        px = int(cx * W)
        py = int(cy * H)

        rw = int(max(24, W * self.radius_frac))
        rh = int(max(24, H * self.radius_frac))

        x1 = max(0, px - rw)
        y1 = max(0, py - rh)
        x2 = min(W, px + rw)
        y2 = min(H, py + rh)

        crop = frame[y1:y2, x1:x2]
        norm_crop_box = (x1 / W, y1 / H, x2 / W, y2 / H)
        return crop, norm_crop_box

    @torch.inference_mode()
    def inspect(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
    ) -> FovealInspection:
        """Performs biological foveal inspection on target coordinate (<2ms)."""
        t0 = time.perf_counter()
        crop, crop_box = self.crop_fovea(frame, centroid)

        if crop.size == 0:
            return FovealInspection(
                centroid=centroid,
                element_kind="empty",
                confidence=0.0,
                bbox=(centroid[0], centroid[1], centroid[0], centroid[1]),
                contrast_score=0.0,
                foveal_crop_shape=(0, 0),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Compute visual edge sharpness / contrast score
        gray = np.mean(crop, axis=-1)
        # Approximate gradient magnitude
        gy, gx = np.gradient(gray)
        contrast = float(np.mean(np.hypot(gx, gy)) / 255.0)

        # 2. Resize foveal crop to tensor input
        t_crop = torch.from_numpy(crop).float() / 255.0
        if t_crop.ndim == 2:
            t_crop = torch.stack([t_crop] * 3, dim=-1)
        elif t_crop.shape[-1] > 3:
            t_crop = t_crop[..., :3]

        t_crop = t_crop.permute(2, 0, 1).unsqueeze(0).to(self.device)
        t_in = F.interpolate(t_crop, size=self.input_size, mode="bilinear", align_corners=False)

        # 3. Model forward pass
        logits, reg_bbox = self.head(t_in)
        probs = torch.softmax(logits, dim=-1)[0]
        cat_idx = int(torch.argmax(probs).item())
        element_kind = (
            FOVEAL_ELEMENT_TYPES[cat_idx]
            if cat_idx < len(FOVEAL_ELEMENT_TYPES)
            else "button"
        )
        conf = float(probs[cat_idx].item())

        # 4. Map local crop bounding box back to global normalized screen coordinates
        rel_x1, rel_y1, rel_x2, rel_y2 = reg_bbox[0].tolist()
        bx1 = min(rel_x1, rel_x2)
        bx2 = max(rel_x1, rel_x2)
        by1 = min(rel_y1, rel_y2)
        by2 = max(rel_y1, rel_y2)

        cb_x1, cb_y1, cb_x2, cb_y2 = crop_box
        cb_w = cb_x2 - cb_x1
        cb_h = cb_y2 - cb_y1

        global_x1 = max(0.0, min(1.0, cb_x1 + bx1 * cb_w))
        global_y1 = max(0.0, min(1.0, cb_y1 + by1 * cb_h))
        global_x2 = max(0.0, min(1.0, cb_x1 + bx2 * cb_w))
        global_y2 = max(0.0, min(1.0, cb_y1 + by2 * cb_h))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return FovealInspection(
            centroid=centroid,
            element_kind=element_kind,
            confidence=conf,
            bbox=(global_x1, global_y1, global_x2, global_y2),
            contrast_score=contrast,
            foveal_crop_shape=(crop.shape[0], crop.shape[1]),
            latency_ms=dt_ms,
        )

