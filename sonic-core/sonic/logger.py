"""
SONIC-REDA — Logging Utility
===============================
Provides a structured logger with fallback to standard library logging
if structlog is not yet installed. Enforces strict secret scrubbing on all logs.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

# ==============================================================================
# Sensitive Data Sanitization
# ==============================================================================

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:token|secret|password|passwd|api[_-]?key|auth|bearer|credential|jwt|private[_-]?key)",
    re.IGNORECASE,
)

_SECRET_VALUE_PATTERN = re.compile(
    r"(?:gsk_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9_\-]{20,}|nvapi-[a-zA-Z0-9_\-]{20,}|Bearer\s+[a-zA-Z0-9_\-\.]+)",
    re.IGNORECASE,
)


def _sanitize_log_value(val: Any) -> Any:
    """Mask sensitive values, dictionary keys, or list elements."""
    if isinstance(val, str):
        return _SECRET_VALUE_PATTERN.sub("[REDACTED]", val)
    if isinstance(val, dict):
        return {
            k: ("[REDACTED]" if _SENSITIVE_KEY_PATTERN.search(str(k)) else _sanitize_log_value(v))
            for k, v in val.items()
        }
    if isinstance(val, (list, tuple)):
        return [_sanitize_log_value(v) for v in val]
    return val


def _sanitize_structlog_event(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to redact sensitive keys and values."""
    for k in list(event_dict.keys()):
        if _SENSITIVE_KEY_PATTERN.search(str(k)):
            event_dict[k] = "[REDACTED]"
        else:
            event_dict[k] = _sanitize_log_value(event_dict[k])
    return event_dict


# ==============================================================================
# Logger Implementation
# ==============================================================================

try:
    import structlog

    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                _sanitize_structlog_event,
                structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )

    def get_logger(name: str = "sonic"):
        return structlog.get_logger(name)

except ImportError:
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    class FallbackLogger:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def _format_msg(self, event: str, kwargs: dict[str, Any]) -> str:
            clean_event = _SECRET_VALUE_PATTERN.sub("[REDACTED]", str(event))
            if kwargs:
                clean_kwargs = {
                    k: ("[REDACTED]" if _SENSITIVE_KEY_PATTERN.search(str(k)) else _sanitize_log_value(v))
                    for k, v in kwargs.items()
                }
                extra = " ".join(f"{k}={v}" for k, v in clean_kwargs.items())
                return f"{clean_event} | {extra}"
            return clean_event

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
