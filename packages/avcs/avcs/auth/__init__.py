"""AVCS Authentication & Cryptographic Licensing System.

Provides enterprise token verification, API key authentication, and hardware-bound
licensing for commercial AVCS deployments.
"""
from avcs.auth.license import (
    InvalidLicenseError,
    LicenseBindingError,
    LicenseError,
    LicenseExpiredError,
    LicenseManager,
    LicensePayload,
    LicenseTamperedError,
    LicenseTier,
    get_machine_fingerprint,
)

__all__ = [
    "LicenseManager",
    "LicensePayload",
    "LicenseTier",
    "LicenseError",
    "LicenseExpiredError",
    "LicenseTamperedError",
    "LicenseBindingError",
    "InvalidLicenseError",
    "get_machine_fingerprint",
]

