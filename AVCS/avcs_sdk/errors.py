"""Typed SDK errors and dispatch-state semantics."""
from __future__ import annotations

from enum import Enum


class AVCSException(Exception):
    """Base exception for SDK-level contract and lifecycle errors."""


class ClientNotConnectedError(AVCSException):
    pass


class ObservationNotKnownError(AVCSException):
    """Raised when a target refers to an observation unknown to this client."""


class StaleTargetError(AVCSException):
    """Raised for an expired or no-longer-current target reference."""


class DuplicateActionIdError(AVCSException):
    """One action ID was reused for a different request payload."""


class BackendContractError(AVCSException):
    """Backend returned a value that violates the SDK protocol."""


class DispatchState(str, Enum):
    NOT_DISPATCHED = "not_dispatched"
    MAY_HAVE_DISPATCHED = "may_have_dispatched"


class BackendActionError(AVCSException):
    """Backend error that makes the dispatch state explicit.

    MAY_HAVE_DISPATCHED must be interpreted as ambiguous. The SDK must never
    retry it automatically, because the UI might already have acted.
    """

    def __init__(self, message: str, *, dispatch_state: DispatchState, error_code: str = "BACKEND_ACTION_ERROR") -> None:
        super().__init__(message)
        self.dispatch_state = dispatch_state
        self.error_code = error_code
