"""Serialize torch model construction across threads.

``transformers``' ``from_pretrained`` calls ``torch.set_default_dtype`` around
model construction and restores it afterwards (see
``transformers/modeling_utils.py``).  That default is **process-global**, so a
background thread loading a VLM while another thread builds a model can make
the second model's parameters come out in the wrong dtype -- the classic
"Input type (torch.FloatTensor) and weight type (CPUBFloat16Type)" failure.

SONIC loads AVCS models on more than one thread (a background VLM captioner and
the foreground cortex), so construction must be serialized and the default
dtype restored.  Use :func:`build_model` for any torch model construction that
could race.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

# Held only around model construction, never around inference.
_BUILD_LOCK = threading.RLock()


def build_model[T](factory: Callable[[], T]) -> T:
    """Run ``factory`` under the shared construction lock with dtype restored.

    The torch default dtype is captured before and restored after, so a
    concurrent ``from_pretrained`` cannot leak a different default into the
    caller's model parameters.
    """
    import torch

    with _BUILD_LOCK:
        previous = torch.get_default_dtype()
        try:
            return factory()
        finally:
            if torch.get_default_dtype() != previous:
                torch.set_default_dtype(previous)


def build_lock() -> Any:
    """The shared construction lock (for callers needing a wider critical section)."""
    return _BUILD_LOCK
