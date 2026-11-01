"""
SONIC-REDA — Autonomous Computer-Use Engine (Phase 14)
======================================================
Unified exports for Phase 14 Autonomous Computer-Using Engineer.

Attributes are resolved lazily (PEP 562).  The workstation perception hot path
imports ``sonic.computer_use.avcs_perception`` on every fresh process; because
Python runs a package ``__init__`` before any submodule, an eager ``__init__``
here drags the whole agent/benchmark/ledger stack (and its imports) into that
perception-only path.  A consumer that asks for one of these names by attribute
still gets it -- the defining submodule is imported on first access.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public name -> defining submodule.  Kept in sync with ``__all__`` below.
_EXPORTS: dict[str, str] = {
    "ActionExecutionStatus": "models",
    "ComputerWorldObservation": "models",
    "ComputerActionPlan": "models",
    "ComputerDecisionTrace": "models",
    "ComputerAutonomyLevel": "models",
    "EngineeringMissionMode": "models",
    "ComputerActionType": "models",
    "ComputerUseMetrics": "models",
    "ComputerUseAgent": "agent",
    "BenchmarkTaskTrials": "benchmark",
    "MultiTrialBenchmarkSuite": "benchmark",
    "MultiTrialResult": "benchmark",
    "TrialMeasurement": "benchmark",
    "MotorReflexes": "motor",
    "HackerScratchpad": "scratchpad",
    "WireTelemetryEngine": "wire_telemetry",
    "Checkpoint": "checkpoint",
    "CheckpointManager": "checkpoint",
    "LocalWorkspaceFileStore": "checkpoint",
    "crop_active_window_region": "grounding",
    "crop_toolbar_region": "grounding",
    "map_crop_to_screen": "grounding",
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
    "ActionExecutionStatus",
    "ComputerWorldObservation",
    "ComputerActionPlan",
    "ComputerDecisionTrace",
    "ComputerAutonomyLevel",
    "EngineeringMissionMode",
    "ComputerActionType",
    "ComputerUseMetrics",
    "ComputerUseAgent",
    "BenchmarkTaskTrials",
    "MultiTrialBenchmarkSuite",
    "MultiTrialResult",
    "TrialMeasurement",
    "MotorReflexes",
    "HackerScratchpad",
    "WireTelemetryEngine",
    "Checkpoint",
    "CheckpointManager",
    "LocalWorkspaceFileStore",
    "crop_active_window_region",
    "crop_toolbar_region",
    "map_crop_to_screen",
]
