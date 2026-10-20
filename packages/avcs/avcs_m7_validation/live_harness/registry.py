"""Registry mapping all 12 Milestone 7/8 benchmark scenarios to their live harnesses."""
from __future__ import annotations

from typing import Dict, List, Tuple, Type

from avcs_m7_validation.catalog import SCENARIOS
from avcs_m7_validation.live_harness.contracts import BaseLiveHarness
from avcs_m7_validation.live_harness.scenarios import (
    AccessibilityMissingHarness,
    BrowserCanvasMoveHarness,
    BrowserDialogDelayedHarness,
    CustomCanvasDragHarness,
    CustomCanvasSelectHarness,
    LegacyFormValidationHarness,
    NativeContextMenuHarness,
    NativeDpiRepositionHarness,
    NativeFileSaveAsHarness,
    NativeSimilarButtonsHarness,
    OverlayStaleTargetHarness,
    RemoteDesktopFocusLossHarness,
)

# Canonical 12-scenario binding map
LIVE_HARNESS_REGISTRY: Dict[str, Type[BaseLiveHarness]] = {
    "native_file_save_as": NativeFileSaveAsHarness,
    "native_similar_buttons": NativeSimilarButtonsHarness,
    "native_context_menu": NativeContextMenuHarness,
    "native_dpi_reposition": NativeDpiRepositionHarness,
    "custom_canvas_select": CustomCanvasSelectHarness,
    "custom_canvas_drag": CustomCanvasDragHarness,
    "browser_canvas_move": BrowserCanvasMoveHarness,
    "browser_dialog_delayed": BrowserDialogDelayedHarness,
    "legacy_form_validation": LegacyFormValidationHarness,
    "accessibility_missing": AccessibilityMissingHarness,
    "remote_desktop_focus_loss": RemoteDesktopFocusLossHarness,
    "overlay_stale_target": OverlayStaleTargetHarness,
}


def get_live_harness(scenario_id: str) -> BaseLiveHarness:
    """Instantiates the live harness registered for scenario_id, failing closed if unmapped."""
    cls = LIVE_HARNESS_REGISTRY.get(scenario_id)
    if not cls:
        raise KeyError(
            f"Scenario '{scenario_id}' has no live application harness registered. "
            f"Available scenarios: {sorted(list(LIVE_HARNESS_REGISTRY.keys()))}"
        )
    return cls()


def list_supported_scenarios() -> List[str]:
    """Returns a sorted list of all scenario IDs with registered live harnesses."""
    return sorted(list(LIVE_HARNESS_REGISTRY.keys()))


def validate_catalog_coverage() -> Tuple[bool, List[str]]:
    """Checks whether every scenario enumerated in catalog.py has a live application harness."""
    catalog_ids = {s.scenario_id for s in SCENARIOS}
    registered_ids = set(LIVE_HARNESS_REGISTRY.keys())
    missing = sorted(list(catalog_ids - registered_ids))
    return len(missing) == 0, missing

