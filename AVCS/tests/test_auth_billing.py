"""Tests for AVCS Authentication, Licensing, and Usage Metering."""
from __future__ import annotations

import time
import pytest

from avcs.auth import (
    InvalidLicenseError,
    LicenseBindingError,
    LicenseExpiredError,
    LicenseManager,
    LicensePayload,
    LicenseTamperedError,
    LicenseTier,
    get_machine_fingerprint,
)
from avcs.billing import RateCard, SessionUsageReceipt, UsageMeter


def test_license_generation_and_verification():
    mgr = LicenseManager(signing_key="test_signing_key_secret_12345")

    # 1. Valid pro license
    token = mgr.generate_license(org_id="org_acme_corp", tier=LicenseTier.PRO, duration_days=30)
    assert token.startswith("AVCS-")
    assert "." in token

    payload = mgr.verify_license(token)
    assert isinstance(payload, LicensePayload)
    assert payload.org_id == "org_acme_corp"
    assert payload.tier == "pro"
    assert not payload.is_expired
    assert payload.max_seats == 1

    # 2. Tampered license payload
    parts = token.split(".", 1)
    tampered = parts[0][:-2] + "AA." + parts[1]
    with pytest.raises((LicenseTamperedError, InvalidLicenseError)):
        mgr.verify_license(tampered)

    # 3. Expired license
    expired_token = mgr.generate_license(org_id="org_old", duration_days=-1)
    with pytest.raises(LicenseExpiredError):
        mgr.verify_license(expired_token)

    # 4. Machine bound license
    fp_token = mgr.generate_license(org_id="org_bound", bind_to_machine=True)
    payload_bound = mgr.verify_license(fp_token, enforce_machine_binding=True)
    assert payload_bound.machine_fingerprint == get_machine_fingerprint()


def test_api_key_generation_and_verification():
    mgr = LicenseManager(signing_key="test_api_key_secret_54321")

    # Live key
    live_key = mgr.generate_api_key(org_id="org_enterprise", tier=LicenseTier.ENTERPRISE, is_live=True)
    assert live_key.startswith("avcs_live_")
    data = mgr.verify_api_key(live_key)
    assert data["org_id"] == "org_enterprise"
    assert data["tier"] == "enterprise"

    # Test key
    test_key = mgr.generate_api_key(org_id="org_dev", is_live=False)
    assert test_key.startswith("avcs_test_")
    test_data = mgr.verify_api_key(test_key)
    assert test_data["org_id"] == "org_dev"

    # Tampered key
    tampered_key = live_key[:-3] + "xyz"
    with pytest.raises((LicenseTamperedError, InvalidLicenseError)):
        mgr.verify_api_key(tampered_key)


def test_usage_metering_and_savings():
    rate_card = RateCard(
        perception_per_minute_usd=0.006,
        motor_action_usd=0.002,
        query_usd=0.001,
        vlm_baseline_cost_per_action_usd=0.05,
    )
    meter = UsageMeter(org_id="org_meter_test", tier="pro", rate_card=rate_card)

    meter.start()
    time.sleep(0.05)
    meter.record_frames(100)
    meter.record_action("click")
    meter.record_action("click")
    meter.record_action("type_text")
    meter.record_query()
    meter.record_surprise()

    snapshot = meter.get_snapshot()
    assert snapshot["frames_processed"] == 100
    assert snapshot["motor_actions"] == 3
    assert snapshot["visual_queries"] == 1
    assert snapshot["surprises_detected"] == 1
    assert snapshot["action_breakdown"]["click"] == 2
    assert snapshot["action_breakdown"]["type_text"] == 1

    savings = meter.calculate_savings()
    assert savings["vlm_baseline_cost_usd"] > 0
    assert savings["net_saved_usd"] > 0
    assert savings["estimated_tokens_saved"] > 0

    meter.pause()


def test_signed_billing_receipt():
    meter = UsageMeter(org_id="org_receipt_test", tier="enterprise")
    meter.start()
    meter.record_frames(500)
    meter.record_action("click")
    meter.record_query()

    receipt = meter.generate_receipt()
    assert isinstance(receipt, SessionUsageReceipt)
    assert receipt.receipt_id.startswith("RCPT_")
    assert receipt.frames_processed == 500
    assert receipt.motor_actions == 1
    assert receipt.visual_queries == 1
    assert receipt.signature != ""

    # Validate signature verification
    assert UsageMeter.verify_receipt(receipt) is True

    # Tamper with receipt
    tampered_dict = receipt.to_dict()
    tampered_dict["motor_actions"] = 999
    assert UsageMeter.verify_receipt(tampered_dict) is False

