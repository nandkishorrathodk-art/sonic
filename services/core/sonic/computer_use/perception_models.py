"""Shared immutable contracts for computer-use perception state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerceptionSnapshot:
    """Versioned state shared by all workers for a single computer."""

    version: int
    screen_hash: str
    width: int
    height: int
    active_window: str
    windows: tuple[str, ...]
    processes: tuple[str, ...]
    visible_text: str
    controls: tuple[str, ...]
    browser_state: dict[str, Any]
    observed_at_ns: int
