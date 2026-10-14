"""Experimental compact visual backbone and spatial heads for AVCS.

The checked-in checkpoint was trained on procedurally generated colored blocks
and has not demonstrated general desktop grounding. This module does not provide
a task policy or prove reliable mouse/keyboard operation.
"""
from __future__ import annotations

import collections
import math
import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.human_eye import (
    ConvBNAct,
    FastAffordanceHead,
    InvertedResidual,
    MagnocellularMotionEngine,
    MovingObject,
    SmoothPursuitTracker,
)
from avcs.memory import VisualMemoryManager, UIMLTree, VisualDiff
from forge.vision.capture import ScreenCapture, SyntheticScreen, open_capture

# Universal 32 Desktop Application Categories (Beyond Web Browsers)
UNIVERSAL_APP_CATEGORIES: List[str] = [
    # Developer & Technical Tools
    "editor_code",          # VS Code, IntelliJ, Sublime Text, Vim
    "terminal_shell",       # PowerShell, Command Prompt, Bash, Alacritty
    "git_diff_viewer",      # GitKraken, SourceTree, Git GUI
    "database_client",      # DBeaver, pgAdmin, DataGrip
    # Office & Productivity
    "data_spreadsheet",     # Excel, LibreOffice Calc, Google Sheets
    "document_writer",      # Word, LibreOffice Writer, Google Docs
    "presentation_slides",  # PowerPoint, Keynote
    "pdf_reader",           # Acrobat Reader, SumatraPDF
    # Creative, CAD & Engineering
    "graphic_canvas",       # Photoshop, GIMP, Paint.NET
    "vector_design",        # Figma, Illustrator, Inkscape
    "cad_3d_modeling",      # Blender, AutoCAD, Fusion 360, SolidWorks
    "video_editor",         # Premiere Pro, DaVinci Resolve, Kdenlive
    "audio_daw",            # Ableton Live, FL Studio, Audacity
    "media_player",         # VLC, Spotify, Windows Media Player
    # OS & Core System Utilities
    "file_explorer",        # Windows Explorer, Finder, Nautilus
    "task_manager",         # Windows Task Manager, Process Explorer, Activity Monitor
    "system_settings",      # Windows Settings, Control Panel, System Preferences
    "installer_wizard",     # Setup wizards, MSI packages, package managers
    "notification_hub",     # Action Center, system tray flyouts
    # Web, Collaboration & Communication
    "browser_web",          # Chrome, Edge, Firefox, Brave
    "email_client",         # Outlook, Thunderbird
    "chat_collaboration",   # Slack, Microsoft Teams, Discord
    "video_conference",     # Zoom, Google Meet
    # Tools, Utilities & Virtualization
    "calculator",           # Windows Calculator, SpeedCrunch
    "archive_tool",         # 7-Zip, WinRAR
    "game_viewport",        # 2D/3D games, DirectX/Vulkan game viewports
    "hardware_monitor",     # HWiNFO, CPU-Z, Task Manager Performance
    "virtual_machine",      # VMware, VirtualBox, Hyper-V
    "remote_desktop",       # RDP client, AnyDesk, TeamViewer
    "desktop_canvas",       # Desktop wallpaper and icon arrangement
    "modal_dialog",         # Popups, confirmation dialogs, file pickers
    "unknown_window",       # Fallback generic application
]

# Universal 24 Affordance Geometry Classes
UNIVERSAL_AFFORDANCE_CLASSES: List[str] = [
    "static_background",        # 0: Non-interactive surface
    "button_primary",           # 1: Highlighted CTA button
    "button_secondary",         # 2: Standard flat button
    "icon_tool",                # 3: Toolbar icon / quick action
    "tab_header",               # 4: Workspace or document tab
    "menu_item",                # 5: Menu bar or context menu item
    "dropdown_select",          # 6: Dropdown selector / combobox
    "text_field_single",        # 7: Single-line input / search box
    "text_field_multiline",     # 8: Textarea / editor body
    "code_editor_line",         # 9: Syntax line in IDE
    "spreadsheet_cell",         # 10: Grid cell in table or spreadsheet
    "checkbox_toggle",          # 11: Binary checkbox
    "radio_option",             # 12: Single-select radio button
    "slider_track",             # 13: Volume or adjustment slider
    "splitter_pane",            # 14: Resizable dock or pane divider
    "scrollbar_thumb",          # 15: Scrollbar thumb or track
    "canvas_anchor_handle",     # 16: Transform node on vector/3D canvas
    "tree_node_expand",         # 17: Treeview branch chevron or expander
    "table_column_header",      # 18: Sortable column title
    "modal_scrim_dismiss",      # 19: Background overlay or close button
    "tooltip_hover",            # 20: Information popover
    "notification_close",       # 21: Notification dismiss button
    "draggable_titlebar",       # 22: Window header bar for moving
    "hyperlink",                # 23: Inline clickable text link
]


class CorticalBackbone2M(nn.Module):
    """Deep multi-scale visual backbone (~776k parameters)."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Stem: 128x128 -> 64x64
        self.stem = ConvBNAct(in_channels, 32, stride=2)
        # Stage 1: 64x64 -> 32x32
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=2),
            InvertedResidual(64, 64, stride=1, expand_ratio=2),
        )
        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=2),
            InvertedResidual(128, 128, stride=1, expand_ratio=2),
        )
        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            InvertedResidual(128, 256, stride=2, expand_ratio=2),
            InvertedResidual(256, 256, stride=1, expand_ratio=2),
        )
        # Spatial Projector to conscious visual feature representation
        self.proj = ConvBNAct(256, feature_dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 128, 128) -> (B, 256, 8, 8)."""
        return self.forward_pyramid(x)[-1]

    def forward_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale activations at 1/2, 1/4, 1/8, and 1/16 resolution."""
        x = self.stem(x)
        stem = x
        x = self.stage1(x)
        stage1 = x
        x = self.stage2(x)
        stage2 = x
        x = self.stage3(x)
        return [stem, stage1, stage2, x, self.proj(x)]


class AffordanceHead2M(nn.Module):
    """Dense 24-class spatial affordance recognition head (~1.19M parameters)."""

    def __init__(self, in_dim: int = 256, num_classes: int = len(UNIVERSAL_AFFORDANCE_CLASSES)) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            ConvBNAct(in_dim, 256, kernel_size=3, padding=1),
            ConvBNAct(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, 256, H, W) -> (B, num_classes, H, W)."""
        return self.head(feat)

    @torch.no_grad()
    def find_hotspots(
        self,
        feat: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.02,
        logits: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, str, float]]:
        """Identifies peak affordances across all 24 geometry classes."""
        if logits is None:
            logits = self.forward(feat)[0]  # (num_classes, H, W)
        elif logits.dim() == 4:
            logits = logits[0]
        probs = torch.softmax(logits, dim=0)
        num_c, H, W = probs.shape

        hotspots: List[Tuple[float, float, str, float]] = []
        for c in range(1, num_c):  # Skip 0 (static background)
            class_map = probs[c]
            max_val = float(torch.max(class_map))
            if max_val >= threshold:
                flat_idx = int(torch.argmax(class_map))
                py = flat_idx // W
                px = flat_idx % W
                nx = (px + 0.5) / float(W)
                ny = (py + 0.5) / float(H)
                class_name = UNIVERSAL_AFFORDANCE_CLASSES[c]
                hotspots.append((nx, ny, class_name, max_val))

        hotspots.sort(key=lambda h: h[3], reverse=True)
        return hotspots[:top_k]


class VentralStream2M(nn.Module):
    """Universal 32-app categorization and 256-d Re-ID embedding stream (~271k parameters)."""

    def __init__(
        self,
        in_dim: int = 256,
        latent_dim: int = 256,
        num_categories: int = len(UNIVERSAL_APP_CATEGORIES),
        dorsal_classes: int = len(UNIVERSAL_AFFORDANCE_CLASSES),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Lateral projection: integrates dorsal affordance summary into ventral recognition
        self.dorsal_lateral = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dorsal_classes, 64),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim + 64, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_categories)

    def forward(
        self,
        feat: torch.Tensor,
        dorsal_context: Optional[torch.Tensor] = None,
        return_raw: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Returns (normalized_visual_latent_embedding, category_logits)
        or (raw_latent, normalized_latent, category_logits) if return_raw is True.
        """
        pool_feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        if dorsal_context is not None:
            d_ctx = self.dorsal_lateral(dorsal_context)
            fused = torch.cat([pool_feat, d_ctx], dim=1)
        else:
            dummy_ctx = torch.zeros(pool_feat.size(0), 64, device=feat.device, dtype=feat.dtype)
            fused = torch.cat([pool_feat, dummy_ctx], dim=1)

        raw_latent = self.fc[2:](fused)  # Skip the Flatten/Pool and run Linear->GELU->Linear
        # Anti-collapse L2-normalization for metric contrastive embedding
        norm_latent = F.normalize(raw_latent, p=2, dim=-1)
        cat_logits = self.classifier(norm_latent)
        if return_raw:
            return raw_latent, norm_latent, cat_logits
        return norm_latent, cat_logits


class LanguageGrounder2M(nn.Module):
    """Legacy grounding head for synthetic 128-dimensional hashed text features."""

    def __init__(self, visual_dim: int = 256, text_dim: int = 128) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.GELU(),
            nn.Linear(256, visual_dim),
        )
        self.cross_attn = nn.Conv2d(visual_dim, visual_dim, kernel_size=1)
        self.heatmap = nn.Conv2d(visual_dim, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Map a matching batch of text embeddings to spatial relevance."""
        if text_emb.ndim != 2 or text_emb.shape[1] != self.text_dim:
            raise ValueError(
                f"text_emb must have shape (batch, {self.text_dim}); "
                f"got {tuple(text_emb.shape)}"
            )
        if text_emb.shape[0] != feat.shape[0]:
            raise ValueError(
                f"text_emb batch ({text_emb.shape[0]}) must match "
                f"feature batch ({feat.shape[0]})"
            )
        t = self.text_proj(text_emb).unsqueeze(-1).unsqueeze(-1)
        modulated = self.cross_attn(feat * t)
        return self.heatmap(modulated)


class UnifiedCortex2M(nn.Module):
    """Legacy experimental multi-head model; not a complete computer-use model."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = CorticalBackbone2M(in_channels=3, feature_dim=feature_dim)
        self.affordance = AffordanceHead2M(in_dim=feature_dim)
        self.ventral = VentralStream2M(in_dim=feature_dim, latent_dim=feature_dim)
        # The training dataset emits normalized 128-dimensional text features.
        self.grounder = LanguageGrounder2M(visual_dim=feature_dim, text_dim=128)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        # 1. Dorsal stream computes spatial affordance field
        aff_logits = self.affordance(feat)
        # 2. Lateral Cross-Stream Information Flow: Dorsal spatial density informs Ventral recognition
        raw_latent, norm_latent, cat_logits = self.ventral(feat, dorsal_context=aff_logits, return_raw=True)
        out = {
            "feat": feat,
            "affordance_logits": aff_logits,
            "visual_latent_raw": raw_latent,
            "visual_latent": norm_latent,
            "category_logits": cat_logits,
        }
        if text_emb is not None:
            out["grounding_map"] = self.grounder(feat, text_emb)
        return out


class HumanEye2M:
    """Universal 2.40M Biological Human Eye Platform for Screen AI Agents.

    Integrates:
    - Magnocellular optical flow engine with static screen bypass (<0.15ms)
    - UnifiedCortex2M deep visual feature representation (~2.40M parameters)
    - Universal 32-application scene recognition
    - 24-class spatial affordance hotspot detection
    - 256-d visual Re-ID episodic memory management
    """

    def __init__(
        self,
        source: Optional[ScreenCapture | str] = None,
        device: str = "cpu",
        input_size: Tuple[int, int] = (128, 128),
        threaded_capture: bool = True,
        weights_path: Optional[str] = None,
        require_weights: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.input_size = input_size

        if str(self.device) == "cpu" and torch.get_num_threads() > 4:
            try:
                torch.set_num_threads(4)
            except Exception:
                pass

        if source is None or source == "desktop":
            self.source = open_capture(prefer_real=True, threaded=threaded_capture)
        elif source == "synthetic":
            self.source = SyntheticScreen()
        elif hasattr(source, "grab"):
            self.source = source
        else:
            self.source = open_capture(prefer_real=True, threaded=threaded_capture)

        self.motion_engine = MagnocellularMotionEngine(grid_rows=16, grid_cols=24)
        self.cortex = UnifiedCortex2M(feature_dim=256).to(self.device)
        self.cortex.eval()
        if str(self.device) == "cpu":
            self.cortex = self.cortex.to(memory_format=torch.channels_last)

        workspace_root = Path(__file__).resolve().parent.parent
        configured_path = weights_path or os.environ.get("AVCS_CORTEX_2M_WEIGHTS")
        candidates = (
            [Path(configured_path).expanduser()]
            if configured_path
            else [workspace_root / "runs" / "cortex_2m" / "cortex_2m_best.pt"]
        )
        checkpoint_path = next((path for path in candidates if path.is_file()), None)
        self.weights_loaded = False
        self.weights_path: Optional[Path] = None
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state = (
                    checkpoint["model_state_dict"]
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                    else checkpoint
                )
                if not isinstance(state, dict):
                    raise ValueError("checkpoint must contain a model state dictionary")
                self.cortex.load_state_dict(state, strict=True)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Failed to load complete Cortex-2M checkpoint "
                    f"'{checkpoint_path}': {exc}"
                ) from exc
            self.weights_loaded = True
            self.weights_path = checkpoint_path.resolve()
        elif require_weights:
            requested = configured_path or str(candidates[0])
            raise FileNotFoundError(
                f"Cortex-2M weights are required but were not found: {requested}"
            )

        self.pursuit = SmoothPursuitTracker()
        self.memory = VisualMemoryManager()

        self.frame_count: int = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=100)
        self._last_state: Optional[Any] = None
        self._cached_feat: Optional[torch.Tensor] = None
        self._cached_affordances: List[Tuple[float, float, str, float]] = []
        self._cached_cat_name: str = "desktop_canvas"
        self._cached_cat_conf: float = 1.0

    @property
    def total_parameters(self) -> int:
        return self.cortex.total_parameters

    def foveal_crop(
        self,
        raw_frame: np.ndarray,
        center_norm: Tuple[float, float],
        crop_size_px: int = 256,
        target_size: Tuple[int, int] = (96, 96),
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extracts an unscaled native-resolution patch around gaze fixation for high-acuity foveal inspection.

        Returns:
            (resized_crop_rgb, (x0, y0, x1, y1) in physical screen coordinates)
        """
        H, W = raw_frame.shape[:2]
        cx = int(center_norm[0] * W)
        cy = int(center_norm[1] * H)
        r = crop_size_px // 2
        x0 = max(0, min(W - crop_size_px, cx - r))
        y0 = max(0, min(H - crop_size_px, cy - r))
        x1 = min(W, x0 + crop_size_px)
        y1 = min(H, y0 + crop_size_px)

        crop = raw_frame[y0:y1, x0:x1]
        pil_crop = Image.fromarray(crop).resize(target_size, resample=Image.BILINEAR)
        return np.array(pil_crop), (x0, y0, x1, y1)

    def perceive(self, text_emb: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Perceive the latest frame; semantic outputs require a complete checkpoint."""
        t_start = time.perf_counter()
        if text_emb is not None:
            if not isinstance(text_emb, torch.Tensor):
                raise TypeError("text_emb must be a torch.Tensor.")
            text_emb = text_emb.to(device=self.device, dtype=torch.float32)
        raw_frame = self.source.grab()

        # 1. Biological Magnocellular Motion Flow
        motion_field, mean_motion, moving_objects, surprise = (
            self.motion_engine.process(raw_frame)
        )

        # 2. Static Screen Early Exit (<0.15ms on idle screens)
        is_static = (mean_motion < 0.005) and (self.frame_count > 0)
        if is_static:
            grounding_map = None
            if text_emb is not None:
                if not self.weights_loaded or self._cached_feat is None:
                    raise RuntimeError(
                        "Cannot produce a grounded prediction without a complete "
                        "Cortex-2M checkpoint."
                    )
                with torch.inference_mode():
                    grounding_map = self.cortex.grounder(self._cached_feat, text_emb)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self._latencies.append(elapsed_ms)
            self.frame_count += 1
            return {
                "frame_index": self.frame_count,
                "latency_ms": elapsed_ms,
                "fps": 1000.0 / max(1e-4, elapsed_ms),
                "is_static": True,
                "motion_magnitude": mean_motion,
                "primary_category": self._cached_cat_name,
                "category_confidence": self._cached_cat_conf,
                "affordances": self._cached_affordances,
                "moving_objects": moving_objects,
                "parameters": self.total_parameters,
                "weights_loaded": self.weights_loaded,
                "weights_path": str(self.weights_path) if self.weights_path else None,
                "grounding_map": grounding_map,
            }

        grounding_map = None
        if self.weights_loaded:
            # Area resize keeps the entire desktop, including taskbars and edges.
            pil_frame = Image.fromarray(raw_frame).resize(
                (self.input_size[1], self.input_size[0]), resample=Image.BILINEAR
            )
            arr = np.array(pil_frame, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            if str(self.device) == "cpu":
                tensor = tensor.to(memory_format=torch.channels_last)

            with torch.inference_mode():
                out = self.cortex(tensor, text_emb=text_emb)
                feat = out["feat"]
                cat_logits = out["category_logits"][0]
                cat_idx = int(torch.argmax(cat_logits))
                cat_conf = float(torch.softmax(cat_logits, dim=0)[cat_idx])
                cat_name = UNIVERSAL_APP_CATEGORIES[cat_idx]
                affordances = self.cortex.affordance.find_hotspots(
                    feat, top_k=5, logits=out["affordance_logits"]
                )
                grounding_map = out.get("grounding_map")
            self._cached_feat = feat
        else:
            if text_emb is not None:
                raise RuntimeError(
                    "Cannot produce a grounded prediction without a complete "
                    "Cortex-2M checkpoint."
                )
            feat = None
            cat_name = "unknown_window"
            cat_conf = 0.0
            affordances = []

        self._cached_affordances = affordances
        self._cached_cat_name = cat_name
        self._cached_cat_conf = cat_conf

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._latencies.append(elapsed_ms)
        self.frame_count += 1

        return {
            "frame_index": self.frame_count,
            "latency_ms": elapsed_ms,
            "fps": 1000.0 / max(1e-4, elapsed_ms),
            "is_static": False,
            "motion_magnitude": mean_motion,
            "primary_category": cat_name,
            "category_confidence": cat_conf,
            "affordances": affordances,
            "moving_objects": moving_objects,
            "parameters": self.total_parameters,
            "weights_loaded": self.weights_loaded,
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "grounding_map": grounding_map,
        }

    def close(self) -> None:
        if hasattr(self.source, "close") and callable(self.source.close):
            self.source.close()
