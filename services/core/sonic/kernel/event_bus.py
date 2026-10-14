"""
SONIC v2 — Typed Event Bus
===========================
The asynchronous communication backbone connecting Central Research Brain,
Specialists, Safety Kernel, and Observation feeds.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sonic.logger import get_logger

logger = get_logger(__name__)


class EventTopic(StrEnum):
    OBSERVATION = "observation"
    EXPERIMENT_REQUEST = "experiment_request"
    EXPERIMENT_RESULT = "experiment_result"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    FINDING_CANDIDATE = "finding_candidate"
    FINDING_VERIFIED = "finding_verified"
    DEAD_END_DETECTED = "dead_end_detected"
    SAFETY_AUDIT = "safety_audit"
    MISSION_STATE = "mission_state"


@dataclass(frozen=True)
class Event:
    """Immutable event published across the system."""
    topic: EventTopic
    sender: str
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


EventHandler = Callable[[Event], Any]


class EventBus:
    """Thread-safe, synchronous and asynchronous pub/sub event bus."""

    def __init__(self):
        self._subscribers: dict[EventTopic, list[EventHandler]] = defaultdict(list)
        self._event_log: list[Event] = []
        self._lock = threading.RLock()

    def subscribe(self, topic: EventTopic, handler: EventHandler) -> None:
        with self._lock:
            if handler not in self._subscribers[topic]:
                self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: EventTopic, handler: EventHandler) -> None:
        with self._lock:
            if handler in self._subscribers[topic]:
                self._subscribers[topic].remove(handler)

    def publish(self, topic: EventTopic, sender: str, payload: dict[str, Any]) -> Event:
        """Publishes an event synchronously to all registered handlers."""
        event = Event(topic=topic, sender=sender, payload=payload)
        with self._lock:
            self._event_log.append(event)
            handlers = list(self._subscribers[topic])
        for handler in handlers:
            try:
                res = handler(event)
                if inspect.isawaitable(res):
                    try:
                        loop = asyncio.get_running_loop()
                        task = asyncio.ensure_future(res, loop=loop)
                        task.add_done_callback(self._log_async_handler_failure)
                    except RuntimeError:
                        async def await_handler() -> Any:
                            return await res

                        asyncio.run(await_handler())
            except Exception as exc:
                logger.error("event_handler_failed", topic=topic.value, sender=sender, error=str(exc))
        return event

    def get_events(self, topic: EventTopic | None = None) -> list[Event]:
        with self._lock:
            if topic:
                return [e for e in self._event_log if e.topic == topic]
            return list(self._event_log)

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()
            self._event_log.clear()

    @staticmethod
    def _log_async_handler_failure(task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.error("async_event_handler_failed", error=str(exc))
