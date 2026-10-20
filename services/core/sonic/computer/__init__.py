"""
SONIC-REDA — Autonomous Computer & Engineering Workspace (Phase 13)
=====================================================================
Unified execution body for Desktop GUI, Terminal, Filesystem,
code-server IDE, Browser, Applications, Git, and Services.

Attributes are resolved lazily (PEP 562).  The computer-use hot path
(``sonic.computer_use.agent`` -> ``from sonic.computer.models import ...``)
runs a package ``__init__`` before any submodule, so importing this package on
every fresh process must not drag in ``bootstrap``/``headless``/``benchmark``
(which pull ``httpx`` and the subprocess stack).  A consumer that asks for one
of those names by attribute still gets it -- the defining submodule is imported
on first access.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public name -> defining submodule.  Kept in sync with ``__all__`` below.
_EXPORTS: dict[str, str] = {
    "AutonomousEngineerBenchmark": "benchmark",
    "ComputerBenchmarkMetrics": "benchmark",
    "WorkstationBootstrapEngine": "bootstrap",
    "HeadlessComputeProvider": "headless",
    "ApplicationPolicy": "models",
    "ComputerAuditEvent": "models",
    "ComputerProfile": "models",
    "ComputerRiskLevel": "models",
    "ComputerSession": "models",
    "ComputerSessionMode": "models",
    "ComputerState": "models",
    "ComputerWorkspace": "models",
    "ComputerWorkspaceStatus": "models",
    "ComputerWorkspaceType": "models",
    "FileEntry": "models",
    "GitStatusInfo": "models",
    "GUIAction": "models",
    "GUIActionType": "models",
    "ProcessInfo": "models",
    "ScreenObservation": "models",
    "ServiceInfo": "models",
    "ComputerProvider": "provider",
    "UnifiedComputerProvider": "provider",
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
    "ApplicationPolicy",
    "AutonomousEngineerBenchmark",
    "ComputerAuditEvent",
    "ComputerBenchmarkMetrics",
    "ComputerProfile",
    "ComputerProvider",
    "ComputerRiskLevel",
    "ComputerSession",
    "ComputerSessionMode",
    "ComputerState",
    "ComputerWorkspace",
    "ComputerWorkspaceStatus",
    "ComputerWorkspaceType",
    "FileEntry",
    "GitStatusInfo",
    "GUIAction",
    "GUIActionType",
    "HeadlessComputeProvider",
    "ProcessInfo",
    "ScreenObservation",
    "ServiceInfo",
    "UnifiedComputerProvider",
    "WorkstationBootstrapEngine",
]

