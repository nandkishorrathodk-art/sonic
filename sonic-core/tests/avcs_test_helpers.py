"""Small valid workstation-frame fixtures for AVCS contract tests."""

import base64
import io

from PIL import Image


def png_base64(width: int = 1280, height: int = 800, color: str = "black") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
