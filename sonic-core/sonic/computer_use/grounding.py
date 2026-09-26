"""
SONIC A-SEA — Computer-Use Visual Grounding Engine
===================================================
Adapted and enhanced from open-computer-use (OS-Atlas & ShowUI).
Translates natural language UI queries (for example, a visible address bar,
a submit control, or a terminal window) into precise pixel coordinates (x, y)
on the active desktop screen, avoiding coordinate guessing or hallucination.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
from typing import Any

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from sonic.logger import get_logger

logger = get_logger(__name__)

# Compatibility dictionary kept empty per Mandate 4 (no static percentage landmarks)
_COMMON_UI_LANDMARKS: dict[str, tuple[float, float]] = {}


def _parse_grounding_coords(
    text: str,
    width: int = 1280,
    height: int = 800,
) -> tuple[int, int] | None:
    """Parse grounding model coordinate text, scaling [0, 1000] tagged boxes to screen dimensions."""
    return extract_bbox_midpoint(text, width=width, height=height)


def extract_bbox_midpoint(
    bbox_response: Any,
    width: int = 1280,
    height: int = 800,
    is_normalized_1000: bool | None = None,
    *,
    screen_width: int | None = None,
    screen_height: int | None = None,
    coordinate_space: str | None = None,
) -> tuple[int, int] | None:
    """Extract (x, y) pixel midpoint from multimodal model grounding output.
    
    Supports:
    - Direct (x, y) tuple or list of numbers
    - JSON objects: {"x": 500, "y": 400} or {"point": [500, 400]}
    - Box tags: <|box_start|>(x1, y1, x2, y2)<|box_end|>
    - Normalized coordinates [0, 1000] (standard UI grounding format)
    - Normalized floats [0.0, 1.0]
    - Direct pixel pairs [x, y]
    """
    if not bbox_response:
        return None
    if screen_width is not None:
        width = screen_width
    if screen_height is not None:
        height = screen_height

    if isinstance(bbox_response, str):
        upper_resp = bbox_response.upper()
        if "NOT_FOUND" in upper_resp or "NOT FOUND" in upper_resp:
            return None

    if coordinate_space is not None:
        normalized_space = coordinate_space.strip().lower()
        if normalized_space in {"normalized", "normalized_1000", "model_1000"}:
            is_normalized_1000 = True
        elif normalized_space in {"pixel", "screen_pixel", "absolute"}:
            is_normalized_1000 = False
        else:
            raise ValueError(f"unsupported coordinate space: {coordinate_space}")
    pt = _extract_bbox_midpoint_impl(bbox_response, width, height, is_normalized_1000)
    if pt == (0, 0):
        return None
    return pt


def _extract_bbox_midpoint_impl(
    bbox_response: Any,
    width: int = 1280,
    height: int = 800,
    is_normalized_1000: bool | None = None,
) -> tuple[int, int] | None:
    if not bbox_response:
        return None

    if isinstance(bbox_response, str) and "{" in bbox_response and "}" in bbox_response:
        try:
            import json
            json_match = re.search(r"\{.*?\}", bbox_response, re.DOTALL)
            if json_match:
                parsed_obj = json.loads(json_match.group(0))
                if isinstance(parsed_obj, dict):
                    if "x" in parsed_obj and "y" in parsed_obj:
                        return _extract_bbox_midpoint_impl(
                            (float(parsed_obj["x"]), float(parsed_obj["y"])),
                            width,
                            height,
                            is_normalized_1000,
                        )
                    if "point" in parsed_obj and isinstance(parsed_obj["point"], (list, tuple)) and len(parsed_obj["point"]) >= 2:
                        return _extract_bbox_midpoint_impl(
                            (float(parsed_obj["point"][0]), float(parsed_obj["point"][1])),
                            width,
                            height,
                            is_normalized_1000,
                        )
        except Exception:
            pass

    if isinstance(bbox_response, (tuple, list)):
        try:
            if len(bbox_response) >= 4:
                x1, y1, x2, y2 = float(bbox_response[0]), float(bbox_response[1]), float(bbox_response[2]), float(bbox_response[3])
                if max(x1, y1, x2, y2) <= 1.0:
                    mid_x = int(((x1 + x2) / 2.0) * width)
                    mid_y = int(((y1 + y2) / 2.0) * height)
                elif is_normalized_1000 is True or (
                    is_normalized_1000 is None and max(x1, y1, x2, y2) <= 1000 and (x2 > width or y2 > height)
                ):
                    mid_x = int(((x1 + x2) / 2.0 / 1000.0) * width)
                    mid_y = int(((y1 + y2) / 2.0 / 1000.0) * height)
                else:
                    mid_x = int((x1 + x2) / 2.0)
                    mid_y = int((y1 + y2) / 2.0)
                mid_pt = (min(max(0, mid_x), max(0, width - 1)), min(max(0, mid_y), max(0, height - 1)))
                return mid_pt
            elif len(bbox_response) >= 2:
                x, y = float(bbox_response[0]), float(bbox_response[1])
                if max(x, y) <= 1.0:
                    px = int(x * width)
                    py = int(y * height)
                elif is_normalized_1000 is True or (is_normalized_1000 is None and (x > width or y > height)):
                    px = int((x / 1000.0) * width)
                    py = int((y / 1000.0) * height)
                else:
                    px, py = int(x), int(y)
                pt = (min(max(0, px), max(0, width - 1)), min(max(0, py), max(0, height - 1)))
                return pt
        except (ValueError, TypeError):
            pass

    if not isinstance(bbox_response, str):
        bbox_response = str(bbox_response)

    tagged_box = ("<|box_start|>" in bbox_response) or ("<box>" in bbox_response)
    match = re.search(r"(?:<\|box_start\|>|<box>)(.*?)(?:<\|box_end\|>|</box>)", bbox_response, re.DOTALL)
    inner_text = match.group(1) if match else bbox_response

    numbers = [float(num) for num in re.findall(r"\d+\.?\d*", inner_text)]
    if not numbers:
        return None

    if len(numbers) >= 4:
        x1, y1, x2, y2 = numbers[0], numbers[1], numbers[2], numbers[3]
        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        if max(x1, y1, x2, y2) <= 1.0:
            mid_x = int(raw_x * width)
            mid_y = int(raw_y * height)
        elif (
            is_normalized_1000 is True
            or (
                tagged_box
                and 0.0 <= raw_x <= 1000.0
                and 0.0 <= raw_y <= 1000.0
                and (raw_x > 1.0 or raw_y > 1.0)
            )
            or (
                is_normalized_1000 is None and max(x1, y1, x2, y2) <= 1000 and (x2 > width or y2 > height)
            )
        ):
            # Explicitly normalized [0, 1000] scale
            mid_x = int((raw_x / 1000.0) * width)
            mid_y = int((raw_y / 1000.0) * height)
        else:
            mid_x = int(raw_x)
            mid_y = int(raw_y)
        mid_pt = (max(0, min(width - 1, mid_x)), max(0, min(height - 1, mid_y)))
        return mid_pt

    elif len(numbers) >= 2:
        raw_x, raw_y = numbers[0], numbers[1]
        if max(raw_x, raw_y) <= 1.0:
            px = int(raw_x * width)
            py = int(raw_y * height)
        elif (
            is_normalized_1000 is True
            or (
                tagged_box
                and 0.0 <= raw_x <= 1000.0
                and 0.0 <= raw_y <= 1000.0
                and (raw_x > 1.0 or raw_y > 1.0)
            )
            or (
                is_normalized_1000 is None and max(raw_x, raw_y) <= 1000 and (raw_x > width or raw_y > height)
            )
        ):
            px = int((raw_x / 1000.0) * width)
            py = int((raw_y / 1000.0) * height)
        else:
            px = int(raw_x)
            py = int(raw_y)
        pt = (max(0, min(width - 1, px)), max(0, min(height - 1, py)))
        return pt

    return None


def draw_action_marker(
    image_b64: str,
    coordinates: tuple[int, int],
    label: str = "CLICK",
    color: str = "#00ffcc",
    radius: int = 14,
) -> str:
    """Draw a visual target indicator and action label on a base64 screenshot."""
    if not _HAS_PIL or not image_b64:
        return image_b64
    try:
        raw_bytes = base64.b64decode(image_b64.split(",")[-1])
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(image)

        x, y = coordinates
        bbox = [x - radius, y - radius, x + radius, y + radius]

        # Use distinct styling for right-click / double-click
        ring_color = color
        if "RIGHT" in label.upper():
            ring_color = "#ff3366"
        elif "DOUBLE" in label.upper():
            ring_color = "#ffaa00"

        # Draw outer pulse ring + filled aim point
        draw.ellipse([x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4], outline=ring_color, width=2)
        draw.ellipse(bbox, fill=(0, 255, 204, 180), outline="black", width=2)

        # Draw crosshairs
        draw.line([(x - radius - 8, y), (x + radius + 8, y)], fill=ring_color, width=2)
        draw.line([(x, y - radius - 8), (x, y + radius + 8)], fill=ring_color, width=2)

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning("draw_action_marker_failed", error=str(e))
        return image_b64


def _invoke_grounding_fn(
    grounding_fn: Any,
    query: str,
    screenshot_b64: str,
    width: int,
    height: int,
) -> Any:
    import inspect
    try:
        sig = inspect.signature(grounding_fn)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if has_varargs or len(params) >= 4:
            return grounding_fn(query, screenshot_b64, width, height)
        return grounding_fn(query, screenshot_b64)
    except (ValueError, TypeError):
        try:
            return grounding_fn(query, screenshot_b64, width, height)
        except TypeError:
            return grounding_fn(query, screenshot_b64)


def resolve_ui_target(
    query: str,
    screenshot_b64: str | None = None,
    width: int = 1280,
    height: int = 800,
    grounding_fn: Any | None = None,
    allow_landmarks: bool = True,
    screen_width: int | None = None,
    screen_height: int | None = None,
    perception_bus: Any | None = None,
    coordinate_space: str | None = None,
) -> tuple[int, int] | None:
    """Resolves a natural language UI target query to absolute screen coordinates (x, y).

    Resolution pipeline:
    0. In-memory fast path via PerceptionBus if available (< 0.1ms).
    1. Direct numeric coordinate check (if query is already "640,400").
    2. Dynamic Multimodal Grounding Function (vision LLM / ShowUI / OS-Atlas) if provided.
    Strictly forbids falling back to arbitrary mock percentage landmarks or blind clicking.
    """
    if not query:
        return None

    # 0. High-speed in-memory resolution via PerceptionBus
    if perception_bus is not None and hasattr(perception_bus, "resolve_coordinates"):
        coords = perception_bus.resolve_coordinates(query)
        if coords is not None:
            return coords

    if screen_width is not None:
        width = screen_width
    if screen_height is not None:
        height = screen_height

    clean_query = query.strip().lower()

    # 1. Direct numeric coordinates
    if "," in clean_query:
        parts = clean_query.split(",")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            if coordinate_space is None and screenshot_b64:
                return None
            coords = (int(parts[0].strip()), int(parts[1].strip()))
            if coordinate_space and coordinate_space.strip().lower() in {"normalized", "normalized_1000", "model_1000"}:
                return (
                    int(coords[0] * width / 1000),
                    int(coords[1] * height / 1000),
                )
            return coords

    # 2. Dynamic Grounding function (multimodal vision model callback)
    if grounding_fn and screenshot_b64:
        try:
            grounding_output = _invoke_grounding_fn(grounding_fn, query, screenshot_b64, width, height)
            if grounding_output:
                if isinstance(grounding_output, (tuple, list)) and len(grounding_output) >= 2:
                    return int(grounding_output[0]), int(grounding_output[1])
                output_space = grounding_output.get("coordinate_space") if isinstance(grounding_output, dict) else coordinate_space
                coords = extract_bbox_midpoint(
                    grounding_output,
                    width,
                    height,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    coordinate_space=output_space,
                )
                if coords:
                    return coords
        except Exception as exc:
            logger.warning("grounding_fn_resolution_failed", query=query, error=str(exc))

    return None


async def resolve_ui_target_async(
    query: str,
    screenshot_b64: str | None = None,
    width: int = 1280,
    height: int = 800,
    grounding_fn: Any | None = None,
    allow_landmarks: bool = False,
    screen_width: int | None = None,
    screen_height: int | None = None,
    perception_bus: Any | None = None,
    coordinate_space: str | None = None,
) -> tuple[int, int] | None:
    """Asynchronous variant of resolve_ui_target supporting coroutine grounding functions."""
    if not query:
        return None

    # 0. High-speed in-memory resolution via PerceptionBus
    if perception_bus is not None and hasattr(perception_bus, "resolve_coordinates"):
        coords = perception_bus.resolve_coordinates(query)
        if coords is not None:
            return coords

    if screen_width is not None:
        width = screen_width
    if screen_height is not None:
        height = screen_height

    clean_query = query.strip().lower()

    # 1. Direct numeric coordinates
    if "," in clean_query:
        parts = clean_query.split(",")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            if coordinate_space is None and screenshot_b64:
                return None
            coords = (int(parts[0].strip()), int(parts[1].strip()))
            if coordinate_space and coordinate_space.strip().lower() in {"normalized", "normalized_1000", "model_1000"}:
                return (
                    int(coords[0] * width / 1000),
                    int(coords[1] * height / 1000),
                )
            return coords

    # 2. Dynamic Grounding function (sync or async)
    if grounding_fn and screenshot_b64:
        try:
            import inspect
            res = _invoke_grounding_fn(grounding_fn, query, screenshot_b64, width, height)
            if inspect.isawaitable(res):
                grounding_output = await res
            else:
                grounding_output = res
            if grounding_output:
                if isinstance(grounding_output, (tuple, list)) and len(grounding_output) >= 2:
                    return int(grounding_output[0]), int(grounding_output[1])
                output_space = grounding_output.get("coordinate_space") if isinstance(grounding_output, dict) else coordinate_space
                coords = extract_bbox_midpoint(
                    grounding_output,
                    width,
                    height,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    coordinate_space=output_space,
                )
                if coords:
                    return coords
        except Exception as exc:
            logger.warning("grounding_fn_async_resolution_failed", query=query, error=str(exc))

    return None


async def query_multimodal_grounding(
    llm_router: Any,
    query: str,
    screenshot_b64: str,
    width: int = 1280,
    height: int = 800,
) -> tuple[int, int] | None:
    """Ground a visual UI query using the configured multimodal vision model (e.g. moonshotai/kimi-k3)."""
    if not llm_router or not screenshot_b64:
        return None
    from sonic.llm.prompts import grounding_user_prompt
    from sonic.llm.schemas import ImageContent, LLMRequest, Message, MessageRole

    raw_b64 = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64
    images = [ImageContent(base64=raw_b64, media_type="image/png")]
    prompt = grounding_user_prompt(query, width, height)
    req = LLMRequest(
        messages=[
            Message(role=MessageRole.USER, content=prompt, images=images)
        ],
        task_type="vision",
        max_tokens=64,
        temperature=0.1,
    )
    try:
        res = await asyncio.wait_for(llm_router.complete(req), timeout=8.0)
        return extract_bbox_midpoint(res.content, width=width, height=height)
    except Exception as exc:
        logger.warning("multimodal_grounding_model_query_failed", query=query, error=str(exc))
        return None


def crop_toolbar_region(
    screenshot_b64: str,
    bbox: tuple[int, int, int, int] | None = None,
    width: int = 1280,
    height: int = 800,
    crop_height_ratio: float | None = None,
) -> tuple[str, tuple[int, int]] | str:
    """
    Hierarchical micro-crop targeting of high-density UI toolbars (e.g. dense GUI components, navigation bars, buttons).

    Args:
        screenshot_b64: Base64-encoded PNG screenshot of the desktop.
        bbox: Optional (x1, y1, x2, y2) bounding box to crop. Defaults to the top toolbar band (0, 0, width, min(240, height)).
        width: Desktop screen width.
        height: Desktop screen height.
        crop_height_ratio: Optional fraction of screen height to crop (returns raw b64 str if provided).

    Returns:
        Tuple of (cropped_screenshot_b64, (offset_x, offset_y)) by default, or cropped_screenshot_b64 if crop_height_ratio is specified.
        If PIL is unavailable or decoding fails, returns (screenshot_b64, (0, 0)).
    """
    if not _HAS_PIL or not screenshot_b64:
        return (screenshot_b64, (0, 0)) if crop_height_ratio is None else screenshot_b64

    crop_h = int(height * crop_height_ratio) if crop_height_ratio is not None else min(240, height)
    crop_box = bbox or (0, 0, width, crop_h)
    x1, y1, x2, y2 = crop_box

    try:
        raw_b64 = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64
        raw_bytes = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(raw_bytes))
        cropped = image.crop((x1, y1, x2, y2))

        buffer = io.BytesIO()
        cropped.convert("RGB").save(buffer, format="PNG")
        cropped_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        if crop_height_ratio is not None:
            return cropped_b64
        return cropped_b64, (x1, y1)
    except Exception as exc:
        logger.warning("crop_toolbar_region_failed", error=str(exc))
        return (screenshot_b64, (0, 0)) if crop_height_ratio is None else screenshot_b64


def map_crop_to_screen(
    local_coords: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[int, int]:
    """Translate coordinates detected inside a micro-crop back to absolute desktop screen coordinates."""
    return local_coords[0] + offset[0], local_coords[1] + offset[1]
