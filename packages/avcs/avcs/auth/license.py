"""Cryptographic License Key & API Token Authentication Engine.

Protects AVCS intellectual property, enforces commercial subscription tiers,
verifies offline/air-gapped cryptographic signatures, and prevents unauthorized node execution.
"""
from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
import hmac
import json
import os
import platform
import socket
import time
import uuid
from typing import Any, Dict, List, Optional


class LicenseTier(str, enum.Enum):
    """Commercial subscription tiers."""
    TRIAL = "trial"
    DEVELOPER = "developer"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class LicenseError(Exception):
    """Base exception for all licensing and authentication failures."""


class LicenseExpiredError(LicenseError):
    """Raised when subscription license has passed its expiration date."""


class LicenseTamperedError(LicenseError):
    """Raised when license token payload or signature has been modified."""


class LicenseBindingError(LicenseError):
    """Raised when node machine fingerprint does not match licensed machine."""


class InvalidLicenseError(LicenseError):
    """Raised when token format is malformed or invalid."""


@dataclasses.dataclass(frozen=True)
class LicensePayload:
    """Decoded cryptographic license properties."""
    license_id: str
    org_id: str
    tier: str
    max_seats: int
    created_at: float
    expires_at: float
    features: List[str]
    machine_fingerprint: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "license_id": self.license_id,
            "org_id": self.org_id,
            "tier": self.tier,
            "max_seats": self.max_seats,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "features": list(self.features),
            "machine_fingerprint": self.machine_fingerprint,
        }


def get_machine_fingerprint() -> str:
    """Generates a stable hardware fingerprint hash for node binding."""
    raw = f"{socket.gethostname()}:{platform.system()}:{uuid.getnode()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class LicenseManager:
    """Cryptographic licensing engine supporting online and air-gapped HMAC signatures."""

    DEFAULT_MASTER_KEY = "praxis_avcs_master_root_2026_enterprise_signing_authority"

    def __init__(self, signing_key: Optional[str] = None) -> None:
        self.signing_key = (
            signing_key
            or os.environ.get("AVCS_SIGNING_KEY")
            or self.DEFAULT_MASTER_KEY
        ).encode("utf-8")

    def generate_license(
        self,
        org_id: str,
        tier: LicenseTier = LicenseTier.PRO,
        duration_days: int = 365,
        max_seats: int = 1,
        features: Optional[List[str]] = None,
        bind_to_machine: bool = False,
        machine_fingerprint: Optional[str] = None,
    ) -> str:
        """Issues a cryptographically signed commercial license string."""
        now = time.time()
        lic_id = f"LIC_{uuid.uuid4().hex[:12].upper()}"
        feats = features or ["60fps", "win32_motor", "visual_servoing", "stream_bus"]

        target_fp = machine_fingerprint or (get_machine_fingerprint() if bind_to_machine else None)

        payload_dict = {
            "license_id": lic_id,
            "org_id": org_id,
            "tier": tier.value if isinstance(tier, LicenseTier) else str(tier),
            "max_seats": max_seats,
            "created_at": now,
            "expires_at": now + (duration_days * 86400.0),
            "features": feats,
            "machine_fingerprint": target_fp,
        }

        payload_bytes = json.dumps(payload_dict, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")

        sig = hmac.new(self.signing_key, payload_bytes, hashlib.sha256).hexdigest()
        return f"AVCS-{payload_b64}.{sig}"

    def generate_license_key(self, *args, **kwargs) -> str:
        """Alias for generate_license."""
        return self.generate_license(*args, **kwargs)

    def verify_license(
        self,
        token: str,
        enforce_machine_binding: bool = False,
        expected_machine_fingerprint: Optional[str] = None,
    ) -> LicensePayload:
        """Validates license integrity, expiration date, and hardware binding."""
        clean_token = token.strip()
        if not clean_token.startswith("AVCS-") or "." not in clean_token:
            raise InvalidLicenseError("Malformed license key: missing AVCS prefix or signature separator.")

        token_body = clean_token[5:]
        parts = token_body.split(".", 1)
        if len(parts) != 2:
            raise InvalidLicenseError("Malformed license key structure.")

        payload_b64, signature = parts
        # Restore padding
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding

        try:
            payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
            data = json.loads(payload_bytes.decode("utf-8"))
        except Exception as e:
            raise LicenseTamperedError(f"Failed to decode license payload: {str(e)}") from e

        # Cryptographic verification
        expected_sig = hmac.new(self.signing_key, payload_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            raise LicenseTamperedError("Cryptographic signature check failed: unauthorized or altered license.")

        # Expiration check
        expires_at = float(data.get("expires_at", 0))
        if time.time() > expires_at:
            raise LicenseExpiredError(f"License expired at timestamp {expires_at}.")

        # Hardware binding check
        bound_fp = data.get("machine_fingerprint")
        if bound_fp and (enforce_machine_binding or expected_machine_fingerprint is not None):
            current_fp = expected_machine_fingerprint or get_machine_fingerprint()
            if current_fp != bound_fp:
                raise LicenseBindingError(
                    f"License is hardware-bound to machine '{bound_fp}', but running on '{current_fp}'."
                )

        return LicensePayload(
            license_id=data["license_id"],
            org_id=data["org_id"],
            tier=data["tier"],
            max_seats=int(data.get("max_seats", 1)),
            created_at=float(data["created_at"]),
            expires_at=expires_at,
            features=list(data.get("features", [])),
            machine_fingerprint=bound_fp,
        )

    def generate_api_key(
        self,
        org_id: str,
        tier: LicenseTier = LicenseTier.DEVELOPER,
        is_live: bool = True,
    ) -> str:
        """Generates a client API key (e.g. avcs_live_... or avcs_test_...)."""
        prefix = "avcs_live_" if is_live else "avcs_test_"
        now = time.time()
        payload = {
            "org_id": org_id,
            "tier": tier.value if isinstance(tier, LicenseTier) else str(tier),
            "iat": now,
        }
        raw_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")
        sig = hmac.new(self.signing_key, raw_bytes, hashlib.sha256).hexdigest()[:24]
        return f"{prefix}{payload_b64}_{sig}"

    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validates API key format and cryptographic signature."""
        key = api_key.strip()
        if not (key.startswith("avcs_live_") or key.startswith("avcs_test_")):
            raise InvalidLicenseError("Invalid API key prefix. Expected 'avcs_live_' or 'avcs_test_'.")

        body = key[10:]
        if "_" not in body:
            raise InvalidLicenseError("Malformed API key structure.")

        payload_b64, sig = body.rsplit("_", 1)
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding

        try:
            raw_bytes = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            raise LicenseTamperedError(f"Corrupt API key payload: {e}") from e

        expected_sig = hmac.new(self.signing_key, raw_bytes, hashlib.sha256).hexdigest()[:24]
        if not hmac.compare_digest(sig, expected_sig):
            raise LicenseTamperedError("API key signature verification failed.")

        return data

