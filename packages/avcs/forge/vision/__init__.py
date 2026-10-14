"""Real-time, pixel-only screen vision.

Pure visual awareness of a desktop: capture -> reflex perception -> typed scene
state -> optional description.  No OCR, no accessibility tree, no structured UI
detection -- only the pixels a human eye would receive.
"""
from forge.vision.capture import (
    CaptureError,
    FrameSource,
    ScreenCapture,
    SyntheticScreen,
    open_capture,
)
from forge.vision.describer import HeuristicDescriber, ModelDescriber, VLMDescriber
from forge.vision.encoder import (
    SceneUnderstandingModel,
    Understanding,
    train_understanding,
    word_report,
)
from forge.vision.cortex import (
    ArtificialVisualCortex,
    CortexState,
    DorsalAffordanceStream,
    LanguageModulation,
    PredictiveCodingEngine,
    RetinalFoveation,
    VentralSemanticStream,
)
from forge.vision.pipeline import ScreenVision
from forge.vision.reflex import ReflexPerception, events_from_states
from forge.vision.scene import Region, SceneEvent, SceneState
from forge.vision.scenes import SCENE_CATEGORIES, SceneGenerator, sample_states
from forge.vision.vocab import VOCAB, render_tokens, tokens_for_state

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
