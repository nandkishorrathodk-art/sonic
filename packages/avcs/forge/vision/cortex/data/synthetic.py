"""Synthetic Procedural Desktop Generator with Ground-Truth Affordances.

Generates realistic desktop screens and produces corresponding:
- UI Category Label (editor, browser, terminal, modal, etc.)
- Dense Ground-Truth Affordance Masks (static, clickable, typeable, scrollable)
- Interactive Elements with (x, y) target coordinates
- Temporal sequences (consecutive frames with animated cursor or modal emergence).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from forge.vision.cortex.core.dorsal import AFFORDANCE_CLASSES
from forge.vision.cortex.core.ventral import UI_CATEGORIES


@dataclass
class SyntheticScreenData:
    frame: np.ndarray             # (H, W, 3) uint8 RGB
    affordance_mask: np.ndarray   # (H, W) int64 class indices [0..4]
    category: str                 # e.g. "editor_code", "modal_dialog"
    category_idx: int             # index into UI_CATEGORIES
    interactive_points: List[Tuple[float, float, str]] # (norm_x, norm_y, class_name)


class ProceduralScreenGenerator:
    """Renders structured desktop scenes with pixel-level ground truth."""

    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height

    def generate(self, category: str | None = None, seed: int | None = None) -> SyntheticScreenData:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if category is None or category not in UI_CATEGORIES:
            category = random.choice(UI_CATEGORIES)

        cat_idx = UI_CATEGORIES.index(category)

        # Base canvas
        canvas = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        mask = np.zeros((self.height, self.width), dtype=np.int64) # 0: static
        points: List[Tuple[float, float, str]] = []

        # 1. Render Window Header (Titlebar)
        h_bar = 28
        canvas[:h_bar, :] = (45, 45, 48)
        # Window control buttons (clickable)
        for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            bx = 16 + i * 18
            by = 14
            canvas[by - 5:by + 5, bx - 5:bx + 5] = col
            mask[by - 5:by + 5, bx - 5:bx + 5] = 1 # clickable
            points.append((bx / self.width, by / self.height, "clickable"))

        # 2. Render Category-Specific Elements
        if category == "editor_code":
            # Code area
            canvas[h_bar:, :60] = (30, 30, 30) # Line numbers gutter
            canvas[h_bar:, 60:] = (24, 24, 24) # Editor background
            mask[h_bar:, 60:] = 2              # typeable area
            # Fake code lines
            for y in range(h_bar + 16, self.height - 20, 16):
                lw = random.randint(80, self.width - 120)
                canvas[y:y + 4, 75:75 + lw] = random.choice([(86, 156, 214), (220, 220, 170), (206, 145, 120)])

        elif category == "terminal_shell":
            canvas[h_bar:, :] = (12, 12, 12) # Dark terminal
            mask[h_bar:, :] = 2              # typeable
            # Terminal prompt
            canvas[h_bar + 15:h_bar + 25, 20:120] = (80, 250, 123)
            # Cursor
            cx, cy = 130, h_bar + 15
            canvas[cy:cy + 10, cx:cx + 6] = (255, 255, 255)

        elif category == "modal_dialog":
            # Dimmed background
            canvas[h_bar:, :] = (20, 20, 20)
            # Modal box
            mw, mh = int(self.width * 0.5), int(self.height * 0.5)
            mx = (self.width - mw) // 2
            my = (self.height - mh) // 2
            canvas[my:my + mh, mx:mx + mw] = (50, 50, 54)

            # Confirm button (clickable)
            btn_w, btn_h = 70, 26
            bx = mx + mw - btn_w - 20
            by = my + mh - btn_h - 16
            canvas[by:by + btn_h, bx:bx + btn_w] = (0, 122, 255)
            mask[by:by + btn_h, bx:bx + btn_w] = 1 # clickable
            points.append(((bx + btn_w / 2) / self.width, (by + btn_h / 2) / self.height, "clickable"))

        elif category in ("browser_webpage", "form_input"):
            # Web address bar or search box (typeable)
            ax, ay, aw, ah = 80, 36, self.width - 160, 24
            canvas[ay:ay + ah, ax:ax + aw] = (255, 255, 255)
            mask[ay:ay + ah, ax:ax + aw] = 2 # typeable
            points.append(((ax + aw / 2) / self.width, (ay + ah / 2) / self.height, "typeable"))

            # Submit / Search Button (clickable)
            bx, by, bw, bh = ax + aw - 60, ay + 2, 56, 20
            canvas[by:by + bh, bx:bx + bw] = (0, 122, 204)
            mask[by:by + bh, bx:bx + bw] = 1 # clickable
            points.append(((bx + bw / 2) / self.width, (by + bh / 2) / self.height, "clickable"))

            # Scrollable main feed
            canvas[ay + ah + 15:, 50:self.width - 50] = (40, 42, 48)
            mask[ay + ah + 15:, 50:self.width - 50] = 3 # scrollable

        else: # settings_panel, desktop_canvas, media_playback
            # Sidebar navigation (clickable items)
            canvas[h_bar:, :120] = (37, 37, 38)
            for sy in range(h_bar + 16, self.height - 40, 32):
                canvas[sy:sy + 18, 16:104] = (62, 62, 66)
                mask[sy:sy + 18, 16:104] = 1 # clickable
                points.append((60 / self.width, (sy + 9) / self.height, "clickable"))

        return SyntheticScreenData(
            frame=canvas,
            affordance_mask=mask,
            category=category,
            category_idx=cat_idx,
            interactive_points=points,
        )

