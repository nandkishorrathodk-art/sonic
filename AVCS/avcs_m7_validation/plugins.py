from __future__ import annotations
import importlib
import os
from typing import Any, Protocol
from .contracts import AgentTask, ExecutionResult, Mode, OracleResult, Scenario

class ExecutionAdapter(Protocol):
    def execute(self, *, run_id: str, task: AgentTask, mode: Mode, seed: int, timeout_s: float) -> ExecutionResult: ...

class IndependentOracle(Protocol):
    def evaluate(self, *, run_id: str, scenario: Scenario, seed: int) -> OracleResult: ...

class MissingPlugin(RuntimeError):
    pass

def _load(spec: str, expected: str) -> Any:
    if ":" not in spec:
        raise ValueError("Plugin spec must use module:factory format")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    plugin = factory()
    if not callable(getattr(plugin, expected, None)):
        raise TypeError(f"Plugin must implement {expected}()")
    return plugin

def load_live_plugins() -> tuple[ExecutionAdapter, IndependentOracle]:
    adapter_spec = os.getenv("AVCS_M7_ADAPTER", "").strip()
    oracle_spec = os.getenv("AVCS_M7_ORACLE", "").strip()
    if not adapter_spec or not oracle_spec:
        raise MissingPlugin(
            "Live validation is disabled. Set AVCS_M7_ADAPTER=module:factory and "
            "AVCS_M7_ORACLE=module:factory. The oracle must independently inspect the "
            "final application state and must not expose expected state to the agent."
        )
    return _load(adapter_spec, "execute"), _load(oracle_spec, "evaluate")
