"""Live AVCS narration for the workstation agent.

This is SONIC's side of the "talking eye" contract: AVCS perceives pixels
locally and *says* what it sees in plain language, and the reasoning model
plans from those words instead of from full screenshots.

Design rules
------------
* **Local only.** No pixels and no frames leave the process.  The narrator
  returns sentences; the caller decides what to forward to a model.
* **Never invents content.** AVCS has no OCR and no semantic UI tree, so the
  narrator reports coarse visual structure -- tone, motion, position, colour,
  and change -- and nothing about what text says.
* **Silence is normal.** When the screen has not visibly changed the narrator
  says nothing, so a planning model is not flooded with restatements.
* **Advisory, never authoritative.** A narration is not proof an action
  succeeded.  Callers still verify with their own receipts.

The narrator is optional: if the AVCS package or a frame is unavailable the
helper returns empty results and the agent keeps its existing behaviour.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bounded so a long mission cannot grow this without limit.
_MAX_RECENT_LINES = 12


class AVCSNarrator:
    """Thin wrapper around ``forge.vision.narrator.LiveNarrator``.

    It owns one :class:`LiveNarrator` per workspace and converts SONIC's
    ``ScreenObservation`` shape into the frame/state the narrator expects.
    Every method fails soft: narration is an enhancement, and losing it must
    never break an action.
    """

    def __init__(self) -> None:
        self._narrators: dict[str, Any] = {}
        self._recent: dict[str, list[dict[str, Any]]] = {}
        self._available: Optional[bool] = None

    # -- capability ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when the AVCS narrator module can be imported."""
        if self._available is None:
            try:
                from forge.vision.narrator import LiveNarrator  # noqa: F401

                self._available = True
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("avcs_narrator_unavailable", error=str(exc))
                self._available = False
        return bool(self._available)

    def _narrator_for(self, workspace_id: str) -> Any:
        key = workspace_id or "default"
        narrator = self._narrators.get(key)
        if narrator is None:
            from forge.vision.narrator import LiveNarrator

            narrator = LiveNarrator()
            self._narrators[key] = narrator
            self._recent[key] = []
        return narrator

    # -- observation --------------------------------------------------------

    @staticmethod
    def frame_from_screen(screen: Any) -> Optional[np.ndarray]:
        """Decode a ``ScreenObservation``'s raw RGB into an (H, W, 3) array.

        Returns ``None`` when raw pixels are unavailable or malformed; the
        caller then simply gets no narration for that frame.
        """
        raw = getattr(screen, "raw_rgb", b"") or b""
        width = getattr(screen, "width", 0)
        height = getattr(screen, "height", 0)
        if (
            not raw
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
            or len(raw) != width * height * 3
        ):
            return None
        try:
            return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        except Exception:  # pragma: no cover - defensive
            return None

    def observe_screen(
        self,
        workspace_id: str,
        screen: Any,
        events: Optional[list[Any]] = None,
        active_window: str = "",
    ) -> Optional[dict[str, Any]]:
        """One narration line for a fresh screen, or ``None`` for silence."""
        if not self.available:
            return None
        state = getattr(screen, "avcs_state", None)
        if not isinstance(state, dict) or state.get("engine") != "avcs_reflex":
            return None
        frame = self.frame_from_screen(screen)
        if frame is None:
            return None
        try:
            from forge.vision.scene import SceneState

            scene = _scene_state_from_dict(state, SceneState)
            narrator = self._narrator_for(workspace_id)
            line = narrator.observe(
                frame, scene, list(events or []), active_window=active_window or ""
            )
        except Exception as exc:
            logger.warning("avcs_narration_failed", error=str(exc))
            return None
        if line is None:
            return None
        payload = line.to_dict()
        recent = self._recent.setdefault(workspace_id or "default", [])
        recent.append(payload)
        del recent[:-_MAX_RECENT_LINES]
        return payload

    # -- action feedback ----------------------------------------------------

    def action_result(
        self,
        workspace_id: str,
        action: str,
        target: str = "",
        visible_effect: Optional[bool] = None,
        screen: Any = None,
        latency_ms: float = 0.0,
        detail: str = "",
    ) -> Optional[dict[str, Any]]:
        """Phrase the outcome of an action the agent already dispatched."""
        if not self.available:
            return None
        state = getattr(screen, "avcs_state", None) if screen is not None else None
        try:
            from forge.vision.scene import SceneState

            scene = (
                _scene_state_from_dict(state, SceneState)
                if isinstance(state, dict) and state.get("engine") == "avcs_reflex"
                else None
            )
            narrator = self._narrator_for(workspace_id)
            line = narrator.action_result(
                action,
                target=target,
                visible_effect=visible_effect,
                state=scene,
                latency_ms=latency_ms,
                detail=detail,
            )
        except Exception as exc:
            logger.warning("avcs_action_narration_failed", error=str(exc))
            return None
        payload = line.to_dict()
        recent = self._recent.setdefault(workspace_id or "default", [])
        recent.append(payload)
        del recent[:-_MAX_RECENT_LINES]
        return payload

    # -- read-out -----------------------------------------------------------

    def recent_lines(self, workspace_id: str) -> list[dict[str, Any]]:
        """Most recent narration lines, oldest first (bounded)."""
        return list(self._recent.get(workspace_id or "default", []))

    def format_for_prompt(self, workspace_id: str, limit: int = 3) -> str:
        """Render recent narration as a bounded prompt block.

        Empty string when there is nothing to say, so callers can append it
        unconditionally without adding noise.
        """
        lines = self.recent_lines(workspace_id)[-max(1, int(limit)):]
        if not lines:
            return ""
        rendered = []
        for item in lines:
            urgency = str(item.get("urgency", "info"))
            marker = {"alert": "!", "notice": "~"}.get(urgency, "-")
            rendered.append(f"  {marker} {item.get('text', '')}")
        return "\n".join(rendered)

    def reset(self, workspace_id: str = "") -> None:
        """Forget narration state for a workspace (or all of them)."""
        if workspace_id:
            self._narrators.pop(workspace_id, None)
            self._recent.pop(workspace_id, None)
        else:
            self._narrators.clear()
            self._recent.clear()


def _scene_state_from_dict(state: dict[str, Any], scene_cls: Any) -> Any:
    """Rebuild a ``SceneState`` from the serialized AVCS state.

    The reflex grids are sent to the agent as plain lists; the narrator needs
    the typed object.  Missing grids degrade to the scalar fields the narrator
    actually reads (brightness, motion, regions), so a partial state still
    yields an honest, if coarser, line.
    """
    rows = int(state.get("rows") or 0)
    cols = int(state.get("cols") or 0)
    if rows <= 0 or cols <= 0:
        raise ValueError("AVCS state is missing tile dimensions")

    def grid(name: str, default: float) -> np.ndarray:
        raw = state.get(name)
        if isinstance(raw, list) and raw:
            try:
                arr = np.asarray(raw, dtype=np.float32)
                if arr.shape == (rows, cols):
                    return arr
            except Exception:
                pass
        return np.full((rows, cols), float(default), dtype=np.float32)

    regions: list[Any] = []
    try:
        from forge.vision.scene import Region

        for item in state.get("regions") or []:
            if not isinstance(item, dict):
                continue
            regions.append(
                Region(
                    row=int(item.get("row", 0)),
                    col=int(item.get("col", 0)),
                    height=int(item.get("height", 1)),
                    width=int(item.get("width", 1)),
                    mean_luma=float(item.get("mean_luma", 0.0)),
                    edge_density=float(item.get("edge_density", 0.0)),
                    salience=float(item.get("salience", 0.0)),
                    kind=str(item.get("kind", "patch")),
                )
            )
    except Exception:  # pragma: no cover - defensive
        regions = []

    centroid = state.get("motion_centroid")
    motion_centroid = None
    if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
        try:
            motion_centroid = (float(centroid[0]), float(centroid[1]))
        except (TypeError, ValueError):
            motion_centroid = None

    return scene_cls(
        frame_index=int(state.get("frame_index") or 0),
        timestamp=float(state.get("timestamp") or 0.0),
        width=int(state.get("width") or 0),
        height=int(state.get("height") or 0),
        rows=rows,
        cols=cols,
        luma_grid=grid("luma_grid", state.get("brightness", 0.0)),
        edge_grid=grid("edge_grid", 0.0),
        salience_grid=grid("salience_grid", 0.0),
        motion_grid=grid("motion_grid", 0.0),
        changed_grid=grid("changed_grid", 0.0),
        brightness=float(state.get("brightness") or 0.0),
        contrast=float(state.get("contrast") or 0.0),
        change_fraction=float(state.get("change_fraction") or 0.0),
        global_motion=float(state.get("global_motion") or 0.0),
        peak_motion=float(state.get("peak_motion") or 0.0),
        novelty=float(state.get("novelty") or 0.0),
        scene_hash=str(state.get("scene_hash") or ""),
        regions=regions,
        overlay_suspected=bool(state.get("overlay_suspected")),
        scene_changed=bool(state.get("scene_changed")),
        motion_centroid=motion_centroid,
        latency_ms=float(state.get("latency_ms") or 0.0),
    )
