"""Real-time, pixel-only screen vision.

Pure visual awareness of a desktop: capture -> reflex perception -> typed scene
state -> optional description.  No OCR, no accessibility tree, no structured UI
detection -- only the pixels a human eye would receive.

Attributes are resolved lazily (PEP 562).  The workstation perception hot path
imports ``forge.vision.reflex`` directly on every fresh process; importing that
submodule must not drag in the training/encoder stack (``torch``) that this
package also re-exports.  A consumer that asks for e.g. ``ReflexPerception`` by
name still gets it -- the defining submodule is imported on first access.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public name -> defining submodule.  Kept in sync with ``__all__`` below.
_EXPORTS: dict[str, str] = {
    "CaptureError": "capture",
    "FrameSource": "capture",
    "ScreenCapture": "capture",
    "SyntheticScreen": "capture",
    "open_capture": "capture",
    "HeuristicDescriber": "describer",
    "ModelDescriber": "describer",
    "VLMDescriber": "describer",
    "SceneUnderstandingModel": "encoder",
    "Understanding": "encoder",
    "train_understanding": "encoder",
    "word_report": "encoder",
    "ArtificialVisualCortex": "cortex",
    "CortexState": "cortex",
    "DorsalAffordanceStream": "cortex",
    "LanguageModulation": "cortex",
    "PredictiveCodingEngine": "cortex",
    "RetinalFoveation": "cortex",
    "VentralSemanticStream": "cortex",
    "ScreenVision": "pipeline",
    "ReflexPerception": "reflex",
    "events_from_states": "reflex",
    "Region": "scene",
    "SceneEvent": "scene",
    "SceneState": "scene",
    "SCENE_CATEGORIES": "scenes",
    "SceneGenerator": "scenes",
    "sample_states": "scenes",
    "VOCAB": "vocab",
    "render_tokens": "vocab",
    "tokens_for_state": "vocab",
}


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "ArtificialVisualCortex",
    "CortexState",
    "RetinalFoveation",
    "DorsalAffordanceStream",
    "VentralSemanticStream",
    "LanguageModulation",
    "PredictiveCodingEngine",
    "CaptureError",
    "ScreenCapture",
    "SyntheticScreen",
    "FrameSource",
    "open_capture",
    "HeuristicDescriber",
    "ModelDescriber",
    "VLMDescriber",
    "SceneUnderstandingModel",
    "Understanding",
    "train_understanding",
    "word_report",
    "ScreenVision",
    "ReflexPerception",
    "events_from_states",
    "Region",
    "SceneEvent",
    "SceneState",
    "SCENE_CATEGORIES",
    "SceneGenerator",
    "sample_states",
    "VOCAB",
    "render_tokens",
    "tokens_for_state",
]
