"""AVCS: Artificial Visual Cortex for Screens (Official Developer SDK).

Real-Time Non-Generative Biological Visual Perception Foundation for Screen AI Agents.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path regardless of execution working directory
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from forge.vision.cortex.sdk import ScreenEye
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState
from forge.vision.cortex.config import CortexConfig
from forge.vision.cortex.core.servoing import (
    VisualServoingEngine as VisualServoingController,
    VisualMotorState as MotorServoState,
)
from forge.vision.cortex.core import (
    AFFORDANCE_CLASSES,
    UI_CATEGORIES,
    STABILITY_STATES,
    RetinalFoveation,
    CorticalBackbone,
    DorsalAffordanceStream,
    VentralSemanticStream,
    PredictiveCodingEngine,
)
from forge.vision.capture import open_capture, ScreenCapture, SyntheticScreen
from avcs.weights import (
    download_weights,
    get_weights_path,
    load_avcs_weights,
)
from avcs.stream import (
    NeuroPacket,
    NeuroStreamBus,
    RealTimeNeuroStreamer,
    GameReflexController,
)
from avcs.drivers import (
    BaseAVCSDriver,
    StepResult,
    AnthropicAVCSDriver,
    OpenAIAVCSDriver,
    OllamaAVCSDriver,
)
from avcs.human_eye import (
    FastAffordanceHead,
    HumanEye,
    HumanEyeState,
    MagnocellularMotionEngine,
    MovingObject,
    RealTimeCorticalBackbone,
    SmoothPursuitTracker,
)
from avcs.grounding import (
    FovealAcuityEngine,
    FovealAcuityHead,
    FovealInspection,
    GroundingResult,
    LanguageGazeGrounder,
)
from avcs.memory import (
    AffordanceType,
    BoundingBox,
    EntityLifecycle,
    PersistentUIRegistry,
    SemanticSceneTreeExtractor,
    UIMLElement,
    UIMLTree,
    VisualDiff,
    VisualDiffEngine,
    VisualMemoryManager,
    VisualMemorySnapshot,
)
from avcs.servoing import (
    NeuromuscularServoingController,
    NeuromuscularServoingResult,
    NeuromuscularTrajectoryPoint,
)
from avcs.verifier import (
    VerificationResult,
    VerificationStatus,
    VisualRippleVerifier,
)
from avcs.ipc import (
    IPCPerceptionFrame,
    SharedMemoryBroadcaster,
    SharedMemoryReader,
)
from avcs.auth.server import EnterpriseLicensingServer
from avcs.events import (
    VisualEvent,
    VisualEventStreamer,
    VisualEventType,
)
from avcs.sdk import (
    HumanEyeClient,
    eye,
)
from avcs.mcp import AVCSMCPServer, run_mcp_server
from avcs.cortex_2m import (
    AffordanceHead2M,
    CorticalBackbone2M,
    HumanEye2M,
    LanguageGrounder2M,
    UnifiedCortex2M,
    UNIVERSAL_AFFORDANCE_CLASSES,
    UNIVERSAL_APP_CATEGORIES,
    VentralStream2M,
)

__version__ = "0.2.0"
__author__ = "Praxis Neural"
__license__ = "Apache-2.0"

__all__ = [
    # Biological Real-Time Human Eye Intelligence Model & SDK
    "HumanEye",
    "HumanEyeState",
    "HumanEyeClient",
    "eye",
    "MovingObject",
    "MagnocellularMotionEngine",
    "RealTimeCorticalBackbone",
    "FastAffordanceHead",
    "SmoothPursuitTracker",
    # Scaled 2.0M Parameter Cortical Foundation Model
    "UnifiedCortex2M",
    "HumanEye2M",
    "CorticalBackbone2M",
    "AffordanceHead2M",
    "VentralStream2M",
    "LanguageGrounder2M",
    "UNIVERSAL_APP_CATEGORIES",
    "UNIVERSAL_AFFORDANCE_CLASSES",
    # Phase 2: Language Grounding & Foveation
    "LanguageGazeGrounder",
    "FovealAcuityEngine",
    "FovealAcuityHead",
    "GroundingResult",
    "FovealInspection",
    "VisualEvent",
    "VisualEventStreamer",
    "VisualEventType",
    # Phase 3: Semantic DOM-less Scene Tree & Multi-Turn Episodic Memory
    "UIMLTree",
    "UIMLElement",
    "BoundingBox",
    "VisualDiff",
    "VisualMemorySnapshot",
    "VisualMemoryManager",
    "SemanticSceneTreeExtractor",
    "PersistentUIRegistry",
    "VisualDiffEngine",
    "AffordanceType",
    "EntityLifecycle",
    # Phase 4: Closed-Loop Physical Hand-Eye Coordination & Visual Verification
    "NeuromuscularServoingController",
    "NeuromuscularServoingResult",
    "NeuromuscularTrajectoryPoint",
    "VisualRippleVerifier",
    "VerificationResult",
    "VerificationStatus",
    # Phase 5: High-Speed IPC & Enterprise Licensing
    "IPCPerceptionFrame",
    "SharedMemoryBroadcaster",
    "SharedMemoryReader",
    "EnterpriseLicensingServer",
    # Developer SDK Class
    "ScreenEye",
    # Core Neural Model & State
    "ArtificialVisualCortex",
    "CortexConfig",
    "CortexState",
    # Motor Servoing
    "VisualServoingController",
    "MotorServoState",
    # Ontologies & Streams
    "AFFORDANCE_CLASSES",
    "UI_CATEGORIES",
    "STABILITY_STATES",
    "RetinalFoveation",
    "CorticalBackbone",
    "DorsalAffordanceStream",
    "VentralSemanticStream",
    "PredictiveCodingEngine",
    # Capture Utilities
    "open_capture",
    "ScreenCapture",
    "SyntheticScreen",
    # Checkpoint & Weights Management
    "get_weights_path",
    "load_avcs_weights",
    "download_weights",
    # Real-Time Neuro-Streaming & Reflex
    "NeuroPacket",
    "NeuroStreamBus",
    "RealTimeNeuroStreamer",
    "GameReflexController",
    # LLM Drivers & MCP Server
    "BaseAVCSDriver",
    "StepResult",
    "AnthropicAVCSDriver",
    "OpenAIAVCSDriver",
    "OllamaAVCSDriver",
    "AVCSMCPServer",
    "run_mcp_server",
    # Native Desktop GUI
    "launch_gui",
    "AVCSDesktopApp",
    # Metadata
    "__version__",
]

def launch_gui() -> None:
    """Launches the AVCS Desktop Station GUI application."""
    from avcs.gui import launch_gui as _launch
    _launch()


def __getattr__(name: str):
    if name == "AVCSDesktopApp":
        from avcs.gui import AVCSDesktopApp
        return AVCSDesktopApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
