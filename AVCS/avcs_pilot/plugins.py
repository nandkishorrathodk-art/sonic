from __future__ import annotations

import importlib
from typing import Any


def load_factory(spec: str) -> Any:
    """Load a plugin factory from ``package.module:callable`` syntax."""
    if not spec or ":" not in spec:
        raise ValueError("Plugin must use module.path:factory syntax")
    module_name, attr = spec.split(":", 1)
    if not module_name or not attr:
        raise ValueError("Plugin must use module.path:factory syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise TypeError(f"Plugin factory is not callable: {spec}")
    return factory()
