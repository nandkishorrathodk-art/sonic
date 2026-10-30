"""
SONIC — Hierarchical Deterministic Convergence Engine (HDCE) Re-exports
=======================================================================

Re-exports `ActionContract`, `StateDelta`, `ActionReceipt`, and
`ActionContractValidator` from `sonic.computer_use.action_contracts` so
mission orchestration (`MissionDirector`, `execute_milestone`, etc.) can
validate state-first convergence and pre/postconditions cleanly.
"""

from __future__ import annotations

from sonic.computer_use.action_contracts import (
    ActionContract,
    ActionContractValidator,
    ActionReceipt,
    ContractValidation,
    StateDelta,
)

__all__ = [
    "ActionContract",
    "StateDelta",
    "ActionReceipt",
    "ContractValidation",
    "ActionContractValidator",
]

