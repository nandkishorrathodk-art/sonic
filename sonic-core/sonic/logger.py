"""
SONIC-REDA — Logging Utility
===============================
Provides a structured logger with fallback to standard library logging
if structlog is not yet installed.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import structlog

    def get_logger(name: str = "sonic"):
        return structlog.get_logger(name)

except ImportError:
    # Standard library logging fallback wrapper with kwarg support
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    class FallbackLogger:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def _format_msg(self, event: str, kwargs: dict[str, Any]) -> str:
            if kwargs:
                extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
                return f"{event} | {extra}"
            return event

        def _emit(self, method: Any, event: str, *args: Any, **kwargs: Any):
            message = event % args if args else event
            extra = {key: value for key, value in kwargs.items() if key not in {"exc_info", "stack_info", "stacklevel"}}
            method(self._format_msg(message, extra), **{
                key: value for key, value in kwargs.items()
                if key in {"exc_info", "stack_info", "stacklevel"}
            })

        def info(self, event: str, *args: Any, **kwargs: Any):
            self._emit(self._logger.info, event, *args, **kwargs)

        def warning(self, event: str, *args: Any, **kwargs: Any):
            self._emit(self._logger.warning, event, *args, **kwargs)

        def error(self, event: str, *args: Any, **kwargs: Any):
            self._emit(self._logger.error, event, *args, **kwargs)

        def critical(self, event: str, *args: Any, **kwargs: Any):
            self._emit(self._logger.critical, event, *args, **kwargs)

        def debug(self, event: str, *args: Any, **kwargs: Any):
            self._emit(self._logger.debug, event, *args, **kwargs)

    def get_logger(name: str = "sonic"):
        return FallbackLogger(name)
