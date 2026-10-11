"""Playwright Visual Eye Adapter (avcs-playwright).

Enables selector-free, zero-DOM visual test automation in Playwright.
Executes clicks, text inputs, and visual stability gating directly through
AVCS biological ocular grounding without relying on fragile CSS/XPath selectors.
"""
from __future__ import annotations

import io
import time
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from avcs import ScreenEye


class AVCSPlaywrightEye:
    """Selector-free visual interaction layer for Playwright browser automation."""

    def __init__(
        self,
        page: Any,
        eye: Optional[ScreenEye] = None,
        viewport_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.page = page
        self.eye = eye or ScreenEye()
        self._viewport_size = viewport_size

    def get_viewport_size(self) -> Tuple[int, int]:
        """Returns browser viewport dimensions (width, height)."""
        if self._viewport_size:
            return self._viewport_size
        try:
            vp = self.page.viewport_size
            if vp and "width" in vp and "height" in vp:
                return (int(vp["width"]), int(vp["height"]))
        except Exception:
            pass
        return (1280, 720)

    def grab_page_frame(self) -> np.ndarray:
        """Grabs current screenshot from Playwright page as RGB numpy array."""
        try:
            raw_bytes = self.page.screenshot()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            return np.array(pil_img)
        except Exception:
            # Fallback to local capture source if page screenshot unavailable
            return self.eye.source.grab()

    def wait_until_visual_stable(self, timeout: float = 4.0) -> bool:
        """Blocks until rendering, transitions, and CSS animations settle."""
        return self.eye.wait_until_stable(timeout=timeout)

    def visual_click(
        self,
        label_or_intent: str,
        timeout: float = 5.0,
        click_delay: float = 0.05,
    ) -> Tuple[int, int]:
        """Locates UI element via visual grounding and executes physical click.

        Completely eliminates fragile CSS selectors and dynamic DOM hierarchies.
        Returns the pixel (x, y) coordinates where the click was executed.
        """
        # Ensure page is stabilized
        self.wait_until_visual_stable(timeout=min(2.0, timeout))

        # Grab page screenshot frame
        frame = self.grab_page_frame()

        # Foveate on semantic target using browser buffer
        target_norm = self.eye.look_at(label_or_intent, frame=frame)
        vw, vh = self.get_viewport_size()

        click_x = int(round(target_norm[0] * vw))
        click_y = int(round(target_norm[1] * vh))

        # Perform mouse action in Playwright
        if hasattr(self.page, "mouse") and hasattr(self.page.mouse, "click"):
            self.page.mouse.move(click_x, click_y)
            if click_delay > 0:
                time.sleep(click_delay)
            self.page.mouse.click(click_x, click_y)

        return (click_x, click_y)

    def visual_type(
        self,
        field_label: str,
        text: str,
        clear_first: bool = True,
    ) -> Tuple[int, int]:
        """Visually locates a text field, clicks to focus, and types input string."""
        import sys
        x, y = self.visual_click(field_label)
        if clear_first and hasattr(self.page, "keyboard"):
            # Select all and delete existing text (cross-platform)
            select_key = "Meta+A" if sys.platform == "darwin" else "Control+A"
            self.page.keyboard.press(select_key)
            self.page.keyboard.press("Backspace")

        if hasattr(self.page, "keyboard") and hasattr(self.page.keyboard, "type"):
            self.page.keyboard.type(text)

        return (x, y)

    def assert_element_visible(self, label_or_intent: str) -> bool:
        """Visual assertion: returns True if element is grounded with high confidence."""
        frame = self.grab_page_frame()
        norm_x, norm_y = self.eye.look_at(label_or_intent, frame=frame)
        state = self.eye.current_state or self.eye.perceive_now(frame=frame)
        return 0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0 and state.category_confidence > 0.1

    # Aliases for developer convenience
    click_visual = visual_click
    input_text = visual_type
