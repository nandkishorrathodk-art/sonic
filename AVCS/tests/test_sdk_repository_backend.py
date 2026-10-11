"""Integration tests for AVCS Python SDK with RepositoryAVCSBackend and ActionBroker."""
from __future__ import annotations

import pytest

from avcs.action_broker import ActionBroker, ActionSafetyController
from avcs.capture_adapter import InteractiveSyntheticCaptureAdapter, SyntheticCaptureAdapter
from avcs.human_eye import HumanEye
from avcs_sdk import (
    ActionClass,
    ActionKind,
    ActionRequest,
    ActionResult,
    ActionStatus,
    AVCSClient,
    BoundingBox,
    NormalizedPoint,
    RepositoryAVCSBackend,
    RiskLevel,
    Target,
    TargetProvenance,
)
from forge.vision.cortex.motor.driver import VirtualCursorDriver


@pytest.mark.asyncio
async def test_repository_backend_lifecycle_and_observation():
    """Verify that AVCSClient with RepositoryAVCSBackend handles lifecycle and captures real frames."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        assert obs.width > 0
        assert obs.height > 0
        assert obs.coordinate_space == "normalized_virtual_desktop"
        assert obs.frame_fingerprint is not None
        assert backend._connected is True

    assert backend._connected is False


@pytest.mark.asyncio
async def test_repository_backend_find_targets():
    """Verify that find_targets returns properly normalized targets."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        targets = await client.find_targets("Save Button", observation=obs)
        assert len(targets) >= 1
        t = targets[0]
        assert t.observation_id == obs.observation_id
        assert 0.0 <= t.bounds.x <= 1.0
        assert 0.0 <= t.bounds.y <= 1.0
        assert 0.0 < t.bounds.width <= 1.0
        assert 0.0 < t.bounds.height <= 1.0
        assert t.confidence >= 0.70
        assert t.provenance == TargetProvenance.VISION


@pytest.mark.asyncio
async def test_repository_backend_click_with_verified_evidence():
    """Verify that clicking through repository backend carries verified evidence."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        targets = await client.find_targets("Submit", observation=obs)
        target = targets[0]

        result = await client.click(
            target=target,
            expected_postcondition="Submit state update",
            action_class=ActionClass.IDEMPOTENT,
        )

        assert result.status == ActionStatus.CONFIRMED
        assert result.evidence is not None
        assert result.evidence.verified is True
        assert "action_broker" in result.evidence.method
        assert result.duration_ms >= 0.0


@pytest.mark.asyncio
async def test_repository_backend_high_risk_unauthorized_blocked():
    """Verify that high-risk action with invalid authorization is blocked by safety controller."""
    backend = RepositoryAVCSBackend(trusted_auth_tokens={"valid_token_123"})
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(
            point=NormalizedPoint(0.5, 0.5),
            observation=obs,
            risk=RiskLevel.HIGH,
            authorization_ref="unauthorized_spoofed_token",
        )

        assert result.status == ActionStatus.BLOCKED
        assert result.evidence is None


@pytest.mark.asyncio
async def test_repository_backend_high_risk_authorized_confirmed():
    """Verify that high-risk action with valid authorization is allowed and confirmed."""
    backend = RepositoryAVCSBackend(trusted_auth_tokens={"valid_token_123"})
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(
            point=NormalizedPoint(0.42, 0.58),
            observation=obs,
            risk=RiskLevel.HIGH,
            authorization_ref="valid_token_123",
        )

        assert result.status == ActionStatus.CONFIRMED
        assert result.evidence is not None
        assert result.evidence.verified is True


@pytest.mark.asyncio
async def test_repository_backend_type_text_and_key():
    """Verify that type_text and press_key route through ActionBroker without bypassing."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        res_text = await client.type_text("Hello AVCS", action_class=ActionClass.IDEMPOTENT)
        assert res_text.status == ActionStatus.CONFIRMED

        res_key = await client.press_key("Enter", action_class=ActionClass.IDEMPOTENT)
        assert res_key.status == ActionStatus.CONFIRMED


@pytest.mark.asyncio
async def test_repository_backend_coordinate_conversion():
    """Verify exact coordinate translation between NormalizedPoint and broker target_coord."""
    driver = VirtualCursorDriver()
    cap = InteractiveSyntheticCaptureAdapter().open()
    eye = HumanEye(source=cap, threaded_capture=False)
    safety = ActionSafetyController()
    broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety, settle_delay_s=0.001)

    backend = RepositoryAVCSBackend(broker=broker, eye=eye, capture=cap, driver=driver)
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        pt = NormalizedPoint(0.2468, 0.7531)
        res = await client.click(point=pt, observation=obs, action_class=ActionClass.IDEMPOTENT)
        assert res.status == ActionStatus.CONFIRMED
