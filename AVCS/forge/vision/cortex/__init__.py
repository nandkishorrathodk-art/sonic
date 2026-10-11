"""Artificial Visual Cortex for Screens (AVCS).

Non-generative, real-time multimodal perception system for computer screens.
Inspired by biological primate visual neuroscience.
"""
from forge.vision.cortex.config import CortexConfig, ModelConfig, RetinaConfig, TrainingConfig
from forge.vision.cortex.core import (
    AFFORDANCE_CLASSES,
    STABILITY_STATES,
    UI_CATEGORIES,
    CorticalBackbone,
    DorsalAffordanceStream,
    LanguageModulation,
    PredictiveCodingEngine,
    RetinalFoveation,
    RetinalView,
    VentralSemanticStream,
    VisualMotorState,
    VisualServoingEngine,
)
from forge.vision.cortex.data import (
    ProceduralScreenGenerator,
    ScreenDataset,
    SyntheticScreenData,
    create_dataloader,
)
from forge.vision.cortex.inference import (
    AVCSInferenceEngine,
    export_onnx,
    export_torchscript,
)
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.cortex.training import (
    AVCSCloudTrainer,
    AVCSMultiTaskLoss,
)

__all__ = [
    # Top-Level SDK, Model & State
    "ScreenEye",
    "ArtificialVisualCortex",
    "CortexState",
    "CortexConfig",
    "ModelConfig",
    "RetinaConfig",
    "TrainingConfig",
    # Core Streams & Servoing
    "RetinalFoveation",
    "RetinalView",
    "CorticalBackbone",
    "DorsalAffordanceStream",
    "VentralSemanticStream",
    "LanguageModulation",
    "PredictiveCodingEngine",
    "VisualServoingEngine",
    "VisualMotorState",
    # Ontologies
    "AFFORDANCE_CLASSES",
    "UI_CATEGORIES",
    "STABILITY_STATES",
    # Data
    "ProceduralScreenGenerator",
    "SyntheticScreenData",
    "ScreenDataset",
    "create_dataloader",
    # Training
    "AVCSCloudTrainer",
    "AVCSMultiTaskLoss",
    # Inference & Export
    "AVCSInferenceEngine",
    "export_torchscript",
    "export_onnx",
]
