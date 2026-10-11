"""AVCS Multi-Turn Visual Memory & Semantic DOM-less Scene Tree (Phase 3).

Provides:
- SemanticSceneTreeExtractor (UIML Scene Tree from pure pixels)
- PersistentUIRegistry (Hungarian Re-ID and spatial tracking)
- VisualDiffEngine (Causal delta transitions Delta_visual)
- VisualMemoryManager (Episodic memory saving >93% of vision tokens)
"""
from __future__ import annotations

from avcs.memory.diff_engine import VisualDiffEngine
from avcs.memory.manager import VisualMemoryManager
from avcs.memory.registry import PersistentUIRegistry
from avcs.memory.tree_extractor import SemanticSceneTreeExtractor
from avcs.memory.types import (
    AffordanceType,
    BoundingBox,
    EntityLifecycle,
    UIMLElement,
    UIMLTree,
    VisualDiff,
    VisualMemorySnapshot,
)

__all__ = [
    "AffordanceType",
    "EntityLifecycle",
    "BoundingBox",
    "UIMLElement",
    "UIMLTree",
    "VisualDiff",
    "VisualMemorySnapshot",
    "SemanticSceneTreeExtractor",
    "PersistentUIRegistry",
    "VisualDiffEngine",
    "VisualMemoryManager",
]
