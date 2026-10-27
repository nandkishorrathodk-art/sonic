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
import json
import math
import os
import re
import threading
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
_AVCS_PERCEPTION_CACHE: dict[str, Any] = {}
_AVCS_PERCEPTION_LOCK = threading.Lock()


def _parse_grounding_coords(
    text: str,
    width: int = 1920,
    height: int = 1080,
) -> tuple[int, int] | None:
    """Parse grounding model coordinate text, scaling [0, 1000] tagged boxes to screen dimensions."""
    return extract_bbox_midpoint(
        text,
        width=width,
        height=height,
        coordinate_space="normalized_1000",
    )


def extract_bbox_midpoint(
    bbox_response: Any,
    width: int = 1920,
    height: int = 1080,
    is_normalized_1000: bool | None = None,
    *,
    screen_width: int | None = None,
    screen_height: int | None = None,
    coordinate_space: str | None = None,
) -> tuple[int, int] | None:
    """Extract an explicitly described point or box midpoint in screen pixels.
    
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
            force_pixels = False
        elif normalized_space in {"pixel", "screen_pixel", "absolute"}:
            is_normalized_1000 = False
            force_pixels = True
        else:
            raise ValueError(f"unsupported coordinate space: {coordinate_space}")
    else:
        force_pixels = False
    pt = _extract_bbox_midpoint_impl(
        bbox_response,
        width,
        height,
        is_normalized_1000,
        force_pixels=force_pixels,
    )
    return pt


def _extract_bbox_midpoint_impl(
    bbox_response: Any,
    width: int = 1280,
    height: int = 800,
    is_normalized_1000: bool | None = None,
    *,
    force_pixels: bool = False,
) -> tuple[int, int] | None:
    if not bbox_response:
        return None

    def _screen_point(x: int, y: int) -> tuple[int, int] | None:
        if 0 <= x < width and 0 <= y < height:
            return x, y
        return None

    if isinstance(bbox_response, str) and "{" in bbox_response and "}" in bbox_response:
        try:
            json_match = re.search(r"\{.*?\}", bbox_response, re.DOTALL)
            if json_match:
                parsed_obj = json.loads(json_match.group(0))
                if isinstance(parsed_obj, dict):
                    if "x" in parsed_obj and "y" in parsed_obj:
                        if any(
                            isinstance(parsed_obj[key], bool)
                            or not isinstance(parsed_obj[key], (int, float))
                            or not math.isfinite(parsed_obj[key])
                            for key in ("x", "y")
                        ):
                            return None
                        return _extract_bbox_midpoint_impl(
                            (float(parsed_obj["x"]), float(parsed_obj["y"])),
                            width,
                            height,
                            is_normalized_1000,
                            force_pixels=force_pixels,
                        )
                    if "point" in parsed_obj and isinstance(parsed_obj["point"], (list, tuple)) and len(parsed_obj["point"]) >= 2:
                        if any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                            for value in parsed_obj["point"][:2]
                        ):
                            return None
                        return _extract_bbox_midpoint_impl(
                            (float(parsed_obj["point"][0]), float(parsed_obj["point"][1])),
                            width,
                            height,
                            is_normalized_1000,
                            force_pixels=force_pixels,
                        )
        except Exception:
            pass

    if isinstance(bbox_response, (tuple, list)):
        try:
            if len(bbox_response) not in (2, 4) or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in bbox_response
            ):
                return None
            if len(bbox_response) == 4:
                x1, y1, x2, y2 = float(bbox_response[0]), float(bbox_response[1]), float(bbox_response[2]), float(bbox_response[3])
                if x1 > x2 or y1 > y2:
                    return None
                if force_pixels:
                    mid_x = int((x1 + x2) / 2.0)
                    mid_y = int((y1 + y2) / 2.0)
                elif max(x1, y1, x2, y2) <= 1.0:
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
                return _screen_point(mid_x, mid_y)
            else:
                x, y = float(bbox_response[0]), float(bbox_response[1])
                if force_pixels:
                    px, py = int(x), int(y)
                elif max(x, y) <= 1.0:
                    px = int(x * width)
                    py = int(y * height)
                elif is_normalized_1000 is True or (
                    is_normalized_1000 is None
                    and max(x, y) <= 1000
                    and (x > width or y > height)
                ):
                    px = int((x / 1000.0) * width)
                    py = int((y / 1000.0) * height)
                else:
                    px, py = int(x), int(y)
                return _screen_point(px, py)
        except (ValueError, TypeError, OverflowError):
            pass

    if not isinstance(bbox_response, str):
        bbox_response = str(bbox_response)

    tagged_box = ("<|box_start|>" in bbox_response) or ("<box>" in bbox_response)
    match = re.search(r"(?:<\|box_start\|>|<box>)(.*?)(?:<\|box_end\|>|</box>)", bbox_response, re.DOTALL)
    inner_text = match.group(1) if match else bbox_response

    numbers = [float(num) for num in re.findall(r"-?\d+\.?\d*", inner_text)]
    if not numbers:
        return None

    if len(numbers) >= 4:
        x1, y1, x2, y2 = numbers[0], numbers[1], numbers[2], numbers[3]
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x1 > x2 or y1 > y2:
            return None
        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        if force_pixels:
            mid_x = int(raw_x)
            mid_y = int(raw_y)
        elif max(x1, y1, x2, y2) <= 1.0:
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
        return _screen_point(mid_x, mid_y)

    elif len(numbers) >= 2:
        raw_x, raw_y = numbers[0], numbers[1]
        if not math.isfinite(raw_x) or not math.isfinite(raw_y):
            return None
        if force_pixels:
            px = int(raw_x)
            py = int(raw_y)
        elif max(raw_x, raw_y) <= 1.0:
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
                is_normalized_1000 is not False
                and 0.0 <= raw_x <= 1000.0
                and 0.0 <= raw_y <= 1000.0
                and (raw_x > 1.0 or raw_y > 1.0)
            )
        ):
            px = int((raw_x / 1000.0) * width)
            py = int((raw_y / 1000.0) * height)
        else:
            px = int(raw_x)
            py = int(raw_y)
        return _screen_point(px, py)

    return None


def _direct_screen_point(
    x: Any,
    y: Any,
    width: int,
    height: int,
    coordinate_space: str | None,
) -> tuple[int, int] | None:
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(x, bool)
        or not isinstance(x, int)
        or isinstance(y, bool)
        or not isinstance(y, int)
    ):
        return None

    normalized_space = (coordinate_space or "pixel").strip().lower()
    if normalized_space in {"normalized", "normalized_1000", "model_1000"}:
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            return None
        x = int(x * width / 1000)
        y = int(y * height / 1000)
    elif normalized_space not in {"pixel", "screen_pixel", "absolute"}:
        return None

    if 0 <= x < width and 0 <= y < height:
        return x, y
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


def _resolve_avcs_target(
    query: str,
    screenshot_b64: str,
    width: int,
    height: int,
    window_geometry: tuple[int, int, int, int] | None = None,
) -> tuple[int, int] | None:
    """Ground a workstation screenshot locally, only with a validated AVCS checkpoint."""
    checkpoint = os.environ.get("SONIC_AVCS_GROUNDING_CHECKPOINT", "").strip()
    if not checkpoint or not query or width <= 0 or height <= 0:
        return None

    try:
        with _AVCS_PERCEPTION_LOCK:
            perception = _AVCS_PERCEPTION_CACHE.get(checkpoint)
            if perception is None:
                from sonic.computer_use.avcs_perception import AVCSPerception

                perception = AVCSPerception()
                if not perception.goal_grounder_configured:
                    return None
                _AVCS_PERCEPTION_CACHE[checkpoint] = perception

        encoded_image = screenshot_b64.split(",", 1)[-1]
        image_bytes = base64.b64decode(encoded_image, validate=True)
        result = perception.ground_target_png(
            image_bytes,
            query,
            width,
            height,
            window_geometry=window_geometry,
        )
        if not isinstance(result, dict) or result.get("engine") != "avcs_goal_grounder":
            return None
        if result.get("ambiguous") is True or result.get("not_found") is True:
            return None
        if getattr(perception, "goal_grounder_ready", False) is not True:
            return None

        checkpoint_metadata = result.get("checkpoint_metadata")
        evaluation = (
            checkpoint_metadata.get("evaluation")
            if isinstance(checkpoint_metadata, dict)
            else None
        )
        if not isinstance(evaluation, dict):
            return None
        from avcs.goal_grounder import GROUNDING_ACCEPTANCE_METRIC, GROUNDING_ACCEPTANCE_THRESHOLD

        evaluation_value = evaluation.get("value")
        evaluation_samples = evaluation.get("samples")
        if (
            evaluation.get("metric") != GROUNDING_ACCEPTANCE_METRIC
            or evaluation.get("passed") is not True
            or isinstance(evaluation_value, bool)
            or not isinstance(evaluation_value, (int, float))
            or not math.isfinite(evaluation_value)
            or evaluation_value < GROUNDING_ACCEPTANCE_THRESHOLD
            or isinstance(evaluation_samples, bool)
            or not isinstance(evaluation_samples, int)
            or evaluation_samples <= 0
        ):
            return None
        accuracy = result.get("held_out_point_accuracy")
        if (
            isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(accuracy)
            or accuracy != evaluation_value
        ):
            return None
        confidence_calibration = checkpoint_metadata.get(
            "query_confidence_calibration"
        )
        confidence_evaluation = (
            confidence_calibration.get("evaluation")
            if isinstance(confidence_calibration, dict)
            else None
        )
        query_score = result.get("query_score")
        query_score_threshold = result.get("query_score_threshold")
        measured_precision = (
            confidence_evaluation.get("precision")
            if isinstance(confidence_evaluation, dict)
            else None
        )
        required_precision = (
            confidence_evaluation.get("required_precision")
            if isinstance(confidence_evaluation, dict)
            else None
        )
        if (
            not isinstance(confidence_calibration, dict)
            or not isinstance(confidence_calibration.get("method"), str)
            or not confidence_calibration["method"].strip()
            or not isinstance(confidence_evaluation, dict)
            or confidence_evaluation.get("metric")
            != "empirical_precision_at_threshold"
            or confidence_evaluation.get("passed") is not True
            or isinstance(confidence_evaluation.get("samples"), bool)
            or not isinstance(confidence_evaluation.get("samples"), int)
            or confidence_evaluation.get("samples") <= 0
            or confidence_evaluation.get("threshold") != query_score_threshold
            or isinstance(measured_precision, bool)
            or not isinstance(measured_precision, (int, float))
            or not math.isfinite(float(measured_precision))
            or isinstance(required_precision, bool)
            or not isinstance(required_precision, (int, float))
            or not math.isfinite(float(required_precision))
            or not 0 < float(required_precision) <= 1
            or float(measured_precision) < float(required_precision)
            or isinstance(query_score, bool)
            or not isinstance(query_score, (int, float))
            or not math.isfinite(float(query_score))
            or not 0 <= float(query_score) <= 1
            or isinstance(query_score_threshold, bool)
            or not isinstance(query_score_threshold, (int, float))
            or not math.isfinite(float(query_score_threshold))
            or not 0 < float(query_score_threshold) <= 1
            or float(query_score) < float(query_score_threshold)
        ):
            return None
        x, y = result.get("x"), result.get("y")
        if (
            isinstance(x, bool)
            or not isinstance(x, int)
            or isinstance(y, bool)
            or not isinstance(y, int)
        ):
            return None
        return _direct_screen_point(x, y, width, height, "pixel")
    except Exception as exc:
        logger.warning("avcs_grounding_unavailable", error=str(exc))
        return None


def _parse_vlm_grounding_reply(
    response: str, width: int, height: int
) -> tuple[int, int] | None:
    """Parse a VLM grounding reply in normalized [0, 1000] space to pixels.

    Accepts the machine-readable shapes ``{"x":..,"y":..}`` (normalized
    0-1000), ``{"point":[x,y]}``, ``[x,y]``, and UI-TARS style
    ``<|box_start|>(x1,y1,x2,y2)<|box_end|>``. Returns ``None`` when the model
    reports ``NOT_FOUND`` or returns anything ambiguous. Out-of-range values
    are rejected rather than clamped.
    """
    if not response:
        return None
    text = response.strip()
    if not text or text.upper() == "NOT_FOUND":
        return None

    number = r"-?\d+(?:\.\d+)?"
    box_match = re.search(
        rf"<\|box_start\|>\s*\(?\s*({number})\s*,\s*({number})\s*,"
        rf"\s*({number})\s*,\s*({number})\s*\)?\s*<\|box_end\|>",
        text,
        re.IGNORECASE,
    )
    if box_match:
        x1, y1, x2, y2 = (float(v) for v in box_match.groups())
        nx, ny = (x1 + x2) / 2, (y1 + y2) / 2
        # UI-TARS boxes are already in pixel space of the image sent.
        return _direct_screen_point(int(round(nx)), int(round(ny)), width, height, "pixel")

    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        match = re.search(r"\[[^\[\]]*\]", text)
    if not match:
        return None
    try:
        decoded = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(decoded, dict) and decoded.get("not_found") is True:
        return None
    if isinstance(decoded, dict) and "x" in decoded and "y" in decoded:
        point = (decoded["x"], decoded["y"])
    elif (
        isinstance(decoded, dict)
        and isinstance(decoded.get("point"), list)
        and len(decoded["point"]) == 2
    ):
        point = tuple(decoded["point"])
    elif isinstance(decoded, list) and len(decoded) == 2:
        point = tuple(decoded)
    else:
        return None

    try:
        nx, ny = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(nx) and math.isfinite(ny)):
        return None
    if not (0 <= nx <= 1000 and 0 <= ny <= 1000):
        return None
    px = int(round(nx / 1000.0 * width))
    py = int(round(ny / 1000.0 * height))
    return _direct_screen_point(px, py, width, height, "pixel")


async def _resolve_vlm_target(
    query: str,
    screenshot_b64: str,
    width: int,
    height: int,
    router: Any | None = None,
) -> tuple[int, int] | None:
    """Ground a target with the configured vision model (VLM) route.

    This is the cloud/API bridge: it sends the live frame plus the target
    phrase to the ``vision`` model route and parses a normalized point from the
    reply. It is used only when no evaluated local AVCS grounder checkpoint is
    configured. The caller (SONIC) still owns the decision and the dispatch;
    this function only answers "where is X?".
    """
    if router is None:
        return None

    try:
        from sonic.llm.prompts import grounding_user_prompt
        from sonic.llm.schemas import ImageContent, LLMRequest, Message, MessageRole
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("vlm_grounding_import_failed", error=str(exc))
        return None

    encoded = screenshot_b64.split(",", 1)[-1]
    try:
        base64.b64decode(encoded, validate=True)
    except Exception:
        return None

    prompt = grounding_user_prompt(query, width, height)
    request = LLMRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content=prompt,
                images=[ImageContent(base64=encoded, media_type="image/png")],
            )
        ],
        task_type="vision",
        max_tokens=128,
        temperature=0.0,
    )
    try:
        response = await router.complete(request)
    except Exception as exc:
        logger.warning("vlm_grounding_call_failed", error=str(exc))
        return None

    reply = getattr(response, "content", "") or ""
    point = _parse_vlm_grounding_reply(reply, width, height)
    if point is None:
        logger.info("vlm_grounding_no_target", query=query, reply=reply[:200])
    return point


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

    Screenshot-based semantic grounding uses only the local AVCS checkpoint.
    Without an explicitly configured, valid evaluated checkpoint, it fails closed.
    Explicit pixel or explicitly labeled normalized coordinates remain available.

    ``grounding_fn`` and ``perception_bus`` remain in the signature for
    compatibility, but are never used for semantic grounding. Arbitrary
    callbacks or non-AVCS control inventories are not trusted grounding
    sources for workstation pixels.
    """
    if not query:
        return None

    if screen_width is not None:
        width = screen_width
    if screen_height is not None:
        height = screen_height

    clean_query = query.strip().lower()

    # 1. Direct numeric coordinates
    if "," in clean_query:
        parts = clean_query.split(",")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return _direct_screen_point(
                int(parts[0].strip()),
                int(parts[1].strip()),
                width,
                height,
                coordinate_space,
            )

    if screenshot_b64:
        return _resolve_avcs_target(query, screenshot_b64, width, height)

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
    window_geometry: tuple[int, int, int, int] | None = None,
) -> tuple[int, int] | None:
    """Asynchronous variant; semantic screenshot grounding remains local AVCS-only."""
    if not query:
        return None

    if screen_width is not None:
        width = screen_width
    if screen_height is not None:
        height = screen_height

    clean_query = query.strip().lower()

    # 1. Direct numeric coordinates
    if "," in clean_query:
        parts = clean_query.split(",")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return _direct_screen_point(
                int(parts[0].strip()),
                int(parts[1].strip()),
                width,
                height,
                coordinate_space,
            )

    if screenshot_b64:
        return await asyncio.to_thread(
            _resolve_avcs_target,
            query,
            screenshot_b64,
            width,
            height,
            window_geometry,
        )

    return None


def _parse_normalized_grounding_response(
    response: str,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    """Accept only the declared machine-readable grounding formats."""
    text = response.strip()
    if text.upper() == "NOT_FOUND":
        return None

    number = r"-?\d+(?:\.\d+)?"
    box_match = re.fullmatch(
        rf"<\|box_start\|>\s*\(?\s*({number})\s*,\s*({number})\s*,"
        rf"\s*({number})\s*,\s*({number})\s*\)?\s*<\|box_end\|>",
        text,
        re.IGNORECASE,
    )
    if box_match:
        x1, y1, x2, y2 = (float(value) for value in box_match.groups())
        point = ((x1 + x2) / 2, (y1 + y2) / 2)
    else:
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

        if isinstance(decoded, dict) and decoded.get("not_found") is True:
            return None
        if isinstance(decoded, dict) and "x" in decoded and "y" in decoded:
            point = (decoded["x"], decoded["y"])
        elif (
            isinstance(decoded, dict)
            and isinstance(decoded.get("point"), list)
            and len(decoded["point"]) == 2
        ):
            point = tuple(decoded["point"])
        elif isinstance(decoded, list) and len(decoded) == 2:
            point = tuple(decoded)
        else:
            return None

    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in point
    ):
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1000 for value in (x, y)):
        return None

    pixel = (int(round(x * width / 1000)), int(round(y * height / 1000)))
    if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        return None
    return pixel


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


def crop_active_window_region(
    screenshot_b64: str,
    geometry: tuple[int, int, int, int] | list[int] | dict[str, Any] | Any | None = None,
    width: int = 1280,
    height: int = 800,
    padding: int = 0,
    return_offset: bool | None = None,
) -> tuple[str, tuple[int, int]] | str:
    """
    Crop the active window region (x, y, w, h) from the desktop screenshot for high-resolution foveated vision.

    Args:
        screenshot_b64: Base64-encoded PNG screenshot of the desktop.
        geometry: Active window bounding box as (x, y, w, h), [x, y, w, h], dict with x, y, width/w, height/h,
                  or an object with x, y, width, height or bbox attributes.
        width: Fallback desktop screen width if image cannot be parsed.
        height: Fallback desktop screen height if image cannot be parsed.
        padding: Optional padding in pixels to expand the cropped window region.
        return_offset: If True, returns (cropped_b64, (offset_x, offset_y)).
                       If False, returns cropped_b64 string.
                       If None (default), returns a tuple (cropped_b64, (offset_x, offset_y)).

    Returns:
        tuple of (cropped_screenshot_b64, (offset_x, offset_y)) or cropped_screenshot_b64 string.
    """
    if not _HAS_PIL or not screenshot_b64 or geometry is None:
        return (screenshot_b64, (0, 0)) if (return_offset is not False) else screenshot_b64

    # Extract x, y, w, h from various formats
    x, y, w, h = 0, 0, 0, 0
    if hasattr(geometry, "bbox"):
        b = geometry.bbox
        x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    elif hasattr(geometry, "x") and hasattr(geometry, "y") and hasattr(geometry, "width") and hasattr(geometry, "height"):
        x, y, w, h = int(geometry.x), int(geometry.y), int(geometry.width), int(geometry.height)
    elif isinstance(geometry, dict):
        x = int(geometry.get("x", 0))
        y = int(geometry.get("y", 0))
        w = int(geometry.get("w", geometry.get("width", 0)))
        h = int(geometry.get("h", geometry.get("height", 0)))
    elif isinstance(geometry, (tuple, list)) and len(geometry) >= 4:
        x, y, w, h = int(geometry[0]), int(geometry[1]), int(geometry[2]), int(geometry[3])
    else:
        return (screenshot_b64, (0, 0)) if (return_offset is not False) else screenshot_b64

    if w <= 0 or h <= 0:
        return (screenshot_b64, (0, 0)) if (return_offset is not False) else screenshot_b64

    try:
        raw_b64 = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64
        raw_bytes = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(raw_bytes))
        img_w, img_h = image.size

        # Clamp crop box to image dimensions with padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        if x2 <= x1 or y2 <= y1:
            return (screenshot_b64, (0, 0)) if (return_offset is not False) else screenshot_b64

        cropped = image.crop((x1, y1, x2, y2))
        buffer = io.BytesIO()
        cropped.convert("RGB").save(buffer, format="PNG")
        cropped_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if return_offset is False:
            return cropped_b64
        return cropped_b64, (x1, y1)
    except Exception as exc:
        logger.warning("crop_active_window_region_failed", error=str(exc))
        return (screenshot_b64, (0, 0)) if (return_offset is not False) else screenshot_b64


def map_crop_to_screen(
    local_coords: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[int, int]:
    """Translate coordinates detected inside a micro-crop back to absolute desktop screen coordinates."""
    return local_coords[0] + offset[0], local_coords[1] + offset[1]
