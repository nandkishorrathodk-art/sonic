"""AVCS pixel evidence and dashboard snapshot adapters for workstation routes."""

from __future__ import annotations

import base64
import math
import os
import time
from io import BytesIO
from typing import Any

from fastapi import HTTPException, status

_capability_probe: Any = None
_capability_probe_key: tuple[str, ...] = ()


def _capability_probe_config_key() -> tuple[str, ...]:
    """Env vars that change what the capability probe would report."""
    import os

    return (
        os.environ.get("SONIC_AVCS_CORTEX_CHECKPOINT", "").strip(),
        os.environ.get("SONIC_AVCS_GROUNDING_CHECKPOINT", "").strip(),
        os.environ.get("SONIC_AVCS_FOCUS_CHECKPOINT", "").strip(),
    )


def _shared_capability_probe() -> Any:
    """One AVCSPerception for capability reporting.

    Constructing a fresh instance per poll is wasteful once a focus checkpoint
    is configured, because loading a checkpoint is expensive. The probe is
    read-only capability reporting, so a process-wide instance is safe.

    It is re-created when the checkpoint configuration changes. The instance
    resolves its checkpoint paths in ``__init__``, so a cached instance would
    otherwise keep reporting a checkpoint that has since been unset (or miss a
    newly configured one), which is exactly the honesty surface an operator
    relies on.
    """
    global _capability_probe, _capability_probe_key
    key = _capability_probe_config_key()
    if _capability_probe is None or key != _capability_probe_key:
        from sonic.computer_use.avcs_perception import AVCSPerception

        _capability_probe = AVCSPerception()
        _capability_probe_key = key
    return _capability_probe


def dashboard_screen_payload(screen: Any) -> dict[str, Any]:
    """Encode raw frames only for explicit dashboard snapshot responses."""
    payload = screen.model_dump()
    raw_rgb = getattr(screen, "raw_rgb", b"")
    width = getattr(screen, "width", 0)
    height = getattr(screen, "height", 0)
    if not payload.get("screenshot_base64") and raw_rgb:
        if (
            not isinstance(raw_rgb, bytes)
            or not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
            or width <= 0
            or height <= 0
            or len(raw_rgb) != width * height * 3
        ):
            raise ValueError("Dashboard snapshot requires a valid RGB frame and dimensions")

        from PIL import Image

        png = BytesIO()
        Image.frombytes("RGB", (width, height), raw_rgb).save(png, format="PNG")
        payload["screenshot_base64"] = base64.b64encode(png.getvalue()).decode("ascii")
    return payload


def avcs_pixel_state_summary(avcs_state: Any) -> dict[str, Any]:
    """Keep only coarse pixel-derived fields suitable for workstation prompts."""
    if not isinstance(avcs_state, dict) or avcs_state.get("engine") != "avcs_reflex":
        return {}

    frame_index = avcs_state.get("frame_index")
    width = avcs_state.get("width")
    height = avcs_state.get("height")
    scene_hash = avcs_state.get("scene_hash")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index <= 0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or not isinstance(scene_hash, str)
        or not scene_hash.strip()
    ):
        return {}

    summary: dict[str, Any] = {
        "engine": "avcs_reflex",
        "frame_index": frame_index,
        "width": width,
        "height": height,
        "scene_hash": scene_hash.strip()[:128],
    }
    for field_name in (
        "brightness",
        "contrast",
        "change_fraction",
        "global_motion",
        "peak_motion",
        "novelty",
        "latency_ms",
    ):
        value = avcs_state.get(field_name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            summary[field_name] = value
    for field_name in ("scene_changed", "overlay_suspected"):
        value = avcs_state.get(field_name)
        if isinstance(value, bool):
            summary[field_name] = value

    centroid = avcs_state.get("motion_centroid")
    if (
        isinstance(centroid, (list, tuple))
        and len(centroid) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in centroid
        )
    ):
        summary["motion_centroid"] = [float(value) for value in centroid]

    regions = avcs_state.get("regions")
    if isinstance(regions, list):
        safe_regions = []
        for region in regions[:8]:
            if not isinstance(region, dict):
                continue
            safe_region = {}
            for field_name in (
                "row",
                "col",
                "height",
                "width",
                "mean_luma",
                "edge_density",
                "salience",
            ):
                value = region.get(field_name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    safe_region[field_name] = value
            if safe_region:
                safe_regions.append(safe_region)
        summary["regions"] = safe_regions
    return summary


def fresh_avcs_pixel_evidence(
    screen: Any, *, max_age_ms: float = 2_000
) -> dict[str, Any] | None:
    """Return bounded AVCS evidence only for a fresh, identified pixel frame."""
    screenshot = getattr(screen, "screenshot_base64", None)
    raw_rgb = getattr(screen, "raw_rgb", b"")
    raw_frame_id = getattr(screen, "frame_id", None)
    width = getattr(screen, "width", None)
    height = getattr(screen, "height", None)
    raw_pixels_valid = (
        isinstance(raw_rgb, bytes)
        and bool(raw_rgb)
        and isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
        and len(raw_rgb) == width * height * 3
        and isinstance(raw_frame_id, int)
        and not isinstance(raw_frame_id, bool)
        and raw_frame_id > 0
    )
    if (
        not (
            isinstance(screenshot, str)
            and screenshot.strip()
            or raw_pixels_valid
        )
        or getattr(screen, "desktop_state", None) != "INTERACTIVE"
    ):
        return None

    capture_age_ms = getattr(screen, "capture_age_ms", None)
    if (
        not isinstance(capture_age_ms, (int, float))
        or isinstance(capture_age_ms, bool)
        or not math.isfinite(float(capture_age_ms))
        or capture_age_ms < 0
    ):
        capture_timestamp_ns = getattr(screen, "capture_timestamp_ns", None)
        if (
            not isinstance(capture_timestamp_ns, int)
            or isinstance(capture_timestamp_ns, bool)
            or capture_timestamp_ns < 0
        ):
            return None
        capture_age_ms = (time.time_ns() - capture_timestamp_ns) / 1_000_000
    else:
        capture_timestamp_ns = getattr(screen, "capture_timestamp_ns", None)
        if (
            isinstance(capture_timestamp_ns, int)
            and not isinstance(capture_timestamp_ns, bool)
            and capture_timestamp_ns >= 0
        ):
            timestamp_age_ms = (time.time_ns() - capture_timestamp_ns) / 1_000_000
            if not math.isfinite(timestamp_age_ms) or timestamp_age_ms < 0:
                return None
            capture_age_ms = max(float(capture_age_ms), timestamp_age_ms)

    if capture_age_ms > max_age_ms:
        return None

    avcs_state = getattr(screen, "avcs_state", None)
    if raw_pixels_valid and (
        not isinstance(avcs_state, dict)
        or avcs_state.get("source_frame_id") != raw_frame_id
    ):
        return None
    summary = avcs_pixel_state_summary(avcs_state)
    if not summary:
        return None
    frame_index = summary["frame_index"]
    if (
        getattr(screen, "perception_version", None) != frame_index
        or getattr(screen, "width", None) != summary["width"]
        or getattr(screen, "height", None) != summary["height"]
    ):
        return None

    return {
        "frame_index": frame_index,
        "scene_hash": summary["scene_hash"],
        "width": summary["width"],
        "height": summary["height"],
        "capture_age_ms": float(capture_age_ms),
        "summary": summary,
    }


async def require_fresh_avcs_evidence(
    computer: Any,
    workspace_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Capture and require a recent AVCS-processed frame before GUI dispatch."""
    screen = await computer.screenshot(workspace_id=workspace_id)
    evidence = fresh_avcs_pixel_evidence(screen)
    if evidence is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Desktop action blocked: fresh, versioned AVCS pixel evidence "
                "is unavailable."
            ),
        )
    return screen, evidence


def _awareness_snapshot() -> dict[str, Any]:
    """Report continuous-awareness state without starting or disturbing it."""
    try:
        from sonic.computer_use.awareness import awareness_stats

        return awareness_stats()
    except Exception:
        return {"running": False}


def avcs_capability_state() -> dict[str, Any]:
    """Report AVCS tool availability separately from GUI model readiness.

    AVCS is a deterministic, read-only reflex tool in the workstation agent.
    It does not provide semantic targets, focus verification, or task success.
    The vision route check only confirms that an image-capable provider/model is
    configured; it does not prove localization quality or GUI task completion.
    """
    capability: dict[str, Any] = {
        "reflex_perception": False,
        "avcs_observe_available": False,
        "live_narration": False,
        "semantic_grounding": False,
        "focus_verification": False,
        "vision_route_configured": False,
        "gui_actions_ready": False,
        "grounding_checkpoint_configured": False,
        "focus_checkpoint_configured": False,
        "learned_cortex_configured": False,
        "learned_cortex_ready": False,
        "affordance_head_trained": False,
        "affordance_outcome_samples": 0,
        "continuous_awareness": False,
        "awareness_stats": {},
        "apprenticeship_enabled": False,
        "apprenticeship_samples": 0,
        "apprenticeship_labels": {},
        "vision_route_ready": False,
        "grounding_source": "none",
        "focus_verification_source": "none",
        "reason": "",
    }
    try:
        _shared_capability_probe()
        capability["reflex_perception"] = True
        capability["avcs_observe_available"] = True
    except Exception as exc:  # pragma: no cover - defensive capability probe
        capability["reason"] = f"AVCS reflex tool unavailable: {exc}"

    try:
        from sonic.api.routes.llm import get_model_router
        router = get_model_router()
        resolved_ok = False
        try:
            router._resolve_provider(
                task_type="computer_use",
                require_images=True,
            )
            resolved_ok = True
        except Exception:
            pass

        rule = getattr(router, "routing_rules", {}).get("computer_use", {}) if hasattr(router, "routing_rules") else {}
        provider_name = rule.get("provider") if isinstance(rule, dict) else None
        model = rule.get("model") if isinstance(rule, dict) else None
        providers = getattr(router, "providers", {})
        provider = providers.get(provider_name) if provider_name and isinstance(providers, dict) else None
        has_rule = bool(rule and provider and hasattr(router, "_model_supports_images") and router._model_supports_images(model, provider))

        if resolved_ok or has_rule or (hasattr(router, "_resolve_provider") and not hasattr(router, "routing_rules")):
            capability["vision_route_configured"] = True

        capability["vision_route_ready"] = router.vision_route_ready()
    except Exception as exc:
        if not capability["reason"]:
            capability["reason"] = f"No configured image-capable computer-use route: {exc}"

    capability["continuous_awareness"] = bool(_awareness_snapshot().get("running"))
    capability["awareness_stats"] = _awareness_snapshot()
    capability["apprenticeship_enabled"] = (
        os.environ.get("SONIC_AVCS_APPRENTICESHIP") == "1"
    )
    if capability["vision_route_configured"]:
        capability["reason"] = (
            "Image-capable computer-use route is configured. AVCS is optional; "
            "GUI localization quality and task success remain unverified."
        )
    elif not capability["reason"]:
        capability["reason"] = (
            "No configured image-capable computer-use route. GUI reasoning requires "
            "a vision model; AVCS reflex data is not a semantic substitute."
        )
    return capability

