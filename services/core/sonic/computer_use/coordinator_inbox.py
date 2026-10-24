"""Operator/coordinator messaging into a running worker.

Parallel workers run in their own contexts and do NOT talk to each other. The
operator (or a coordinating agent) may still need to steer one specific worker
mid-run -- e.g. "stop touching file X", "also check the edge case in test Y".

This module is a small process-wide inbox keyed by the run-unique
``agent_uid``. A message posted here is drained by the owning worker on its
next reasoning step and rendered as an explicit operator instruction. It is a
delivery mechanism only: it never fabricates a result, never marks a worker
complete, and the worker's own evidence + the independent Judge still decide
whether the work succeeded.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

_MAX_PENDING = 20
_MAX_MESSAGE_CHARS = 2000


class CoordinatorInbox:
    """Thread-safe, bounded inbox of pending operator messages per worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, deque[dict[str, Any]]] = {}

    def post(self, agent_uid: str, message: str, *, sender: str = "operator") -> dict[str, Any]:
        """Queue a message for a worker. Returns the stored record.

        Empty messages are rejected rather than silently queued, so a caller
        cannot appear to have steered a worker without sending anything.
        """
        uid = str(agent_uid or "").strip()
        text = " ".join(str(message or "").split())
        if not uid:
            raise ValueError("agent_uid is required to message a worker")
        if not text:
            raise ValueError("message text is required")
        record = {
            "sender": str(sender or "operator")[:80],
            "message": text[:_MAX_MESSAGE_CHARS],
        }
        with self._lock:
            bucket = self._pending.setdefault(uid, deque(maxlen=_MAX_PENDING))
            bucket.append(record)
        return record

    def drain(self, agent_uid: str) -> list[dict[str, Any]]:
        """Remove and return all pending messages for a worker."""
        uid = str(agent_uid or "").strip()
        if not uid:
            return []
        with self._lock:
            bucket = self._pending.pop(uid, None)
        return list(bucket) if bucket else []

    def pending_count(self, agent_uid: str) -> int:
        with self._lock:
            bucket = self._pending.get(str(agent_uid or "").strip())
        return len(bucket) if bucket else 0


_INBOX = CoordinatorInbox()


def get_coordinator_inbox() -> CoordinatorInbox:
    """Return the process-wide coordinator inbox."""
    return _INBOX


def reset_coordinator_inbox() -> None:
    """Clear the process-wide inbox (test isolation / shutdown)."""
    global _INBOX
    _INBOX = CoordinatorInbox()
