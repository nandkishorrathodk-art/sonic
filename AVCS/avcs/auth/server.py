"""AVCS Enterprise Licensing & Hardware Entitlement Server.

Provides offline and network-based license issuance, hardware fingerprint verification,
and enterprise billing reconciliation for commercial multi-seat deployments.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional

from avcs.auth.license import (
    LicenseManager,
    LicensePayload,
    LicenseTier,
    get_machine_fingerprint,
)


class EnterpriseLicensingServer:
    """Enterprise licensing engine for issuance and node validation."""

    def __init__(self, master_signing_key: Optional[str] = None) -> None:
        self.master_key = master_signing_key or LicenseManager.DEFAULT_MASTER_KEY
        self.manager = LicenseManager(signing_key=self.master_key)

    def issue_enterprise_license(
        self,
        org_id: str,
        tier: LicenseTier = LicenseTier.ENTERPRISE,
        duration_days: int = 365,
        max_seats: int = 50,
        bind_to_machine: Optional[str] = None,
        custom_features: Optional[list[str]] = None,
    ) -> str:
        """Issues an air-gapped cryptographic license key."""
        features = custom_features or [
            "realtime_60fps",
            "foveal_acuity_engine",
            "domless_uiml_tree",
            "visual_diff_engine",
            "neuromuscular_servoing",
            "shared_memory_ipc",
            "commercial_redistribution",
        ]
        return self.manager.generate_license_key(
            org_id=org_id,
            tier=tier,
            duration_days=duration_days,
            max_seats=max_seats,
            machine_fingerprint=bind_to_machine,
            features=features,
        )

    def verify_node(
        self,
        license_token: str,
        expected_machine_fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validates node entitlement and machine binding."""
        fp = expected_machine_fingerprint or get_machine_fingerprint()
        payload = self.manager.verify_license(license_token, expected_machine_fingerprint=fp)
        return {
            "valid": True,
            "license_id": payload.license_id,
            "org_id": payload.org_id,
            "tier": payload.tier,
            "max_seats": payload.max_seats,
            "expires_in_days": round((payload.expires_at - time.time()) / 86400.0, 1),
            "features": payload.features,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Enterprise Licensing Tool")
    parser.add_argument("--issue", action="store_true", help="Issue a new commercial license")
    parser.add_argument("--verify", type=str, help="Verify an existing license token")
    parser.add_argument("--org", type=str, default="enterprise_client", help="Organization ID")
    parser.add_argument("--tier", type=str, default="enterprise", choices=["trial", "developer", "pro", "enterprise"])
    parser.add_argument("--days", type=int, default=365, help="Validity duration in days")
    parser.add_argument("--seats", type=int, default=10, help="Maximum seats")
    parser.add_argument("--bind-current-machine", action="store_true", help="Cryptographically bind to current machine")
    args = parser.parse_args()

    server = EnterpriseLicensingServer()

    if args.issue:
        target_fp = get_machine_fingerprint() if args.bind_current_machine else None
        tier = LicenseTier(args.tier)
        token = server.issue_enterprise_license(
            org_id=args.org,
            tier=tier,
            duration_days=args.days,
            max_seats=args.seats,
            bind_to_machine=target_fp,
        )
        print("=" * 60)
        print(f"  AVCS ENTERPRISE LICENSE KEY GENERATED")
        print("=" * 60)
        print(f"Organization: {args.org}")
        print(f"Tier:         {args.tier.upper()}")
        print(f"Duration:     {args.days} days")
        print(f"Hardware FP:  {target_fp or 'UNBOUND (Portable)'}")
        print("-" * 60)
        print(token)
        print("=" * 60)
    elif args.verify:
        try:
            res = server.verify_node(args.verify)
            print("=" * 60)
            print("  LICENSE VERIFICATION: VALID")
            print("=" * 60)
            print(json.dumps(res, indent=2))
        except Exception as exc:
            print(f"Verification Failed: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        current_fp = get_machine_fingerprint()
        print(f"Current Machine Hardware Fingerprint: {current_fp}")


if __name__ == "__main__":
    main()

