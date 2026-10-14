"""Real-Time Usage Metering & Billing Engine.

Tracks continuous visual perception runtime, motor actions, token savings against
naive cloud VLM screenshot polling, and generates cryptographically signed billing receipts.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional


@dataclasses.dataclass(frozen=True)
class RateCard:
    """Pricing rates for AVCS commercial consumption."""
    perception_per_minute_usd: float = 0.005   # $0.30 / hour of active 60Hz eye perception
    motor_action_usd: float = 0.002            # $0.002 per hardware click/stroke
    query_usd: float = 0.001                   # $0.001 per visual search / affordance query
    vlm_baseline_cost_per_action_usd: float = 0.05  # Standard cloud screenshot VLM cost
    vlm_tokens_per_screenshot: int = 1600      # Standard vision tokens per full HD screenshot


@dataclasses.dataclass(frozen=True)
class SessionUsageReceipt:
    """Immutable, cryptographically verifiable usage receipt for billing reconciliation."""
    receipt_id: str
    session_id: str
    org_id: str
    tier: str
    started_at: float
    ended_at: float
    duration_seconds: float
    frames_processed: int
    motor_actions: int
    visual_queries: int
    surprises_detected: int
    total_cost_usd: float
    estimated_cost_saved_usd: float
    estimated_tokens_saved: int
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class UsageMeter:
    """Thread-safe usage meter tracking real-time perception runtime and actions."""

    DEFAULT_BILLING_KEY = "praxis_avcs_billing_authority_secret_2026"

    def __init__(
        self,
        org_id: str = "org_default",
        tier: str = "pro",
        rate_card: Optional[RateCard] = None,
        signing_key: Optional[str] = None,
    ) -> None:
        self.session_id: str = f"sess_{uuid.uuid4().hex[:12]}"
        self.org_id: str = org_id
        self.tier: str = tier
        self.rate_card: RateCard = rate_card or RateCard()
        self.signing_key: bytes = (
            signing_key
            or os.environ.get("AVCS_BILLING_KEY")
            or self.DEFAULT_BILLING_KEY
        ).encode("utf-8")

        self._lock = threading.Lock()
        self.started_at: float = time.time()
        self.last_active_time: float = time.monotonic()
        self.active_runtime_seconds: float = 0.0
        self.is_active: bool = False

        self.frames_processed: int = 0
        self.motor_actions: int = 0
        self.visual_queries: int = 0
        self.surprises_detected: int = 0
        self.action_breakdown: Dict[str, int] = {}

    def start(self) -> None:
        """Starts or resumes active perception time tracking."""
        with self._lock:
            if not self.is_active:
                self.is_active = True
                self.last_active_time = time.monotonic()

    def pause(self) -> None:
        """Pauses perception time tracking (e.g., when eye enters idle state)."""
        with self._lock:
            if self.is_active:
                now = time.monotonic()
                self.active_runtime_seconds += max(0.0, now - self.last_active_time)
                self.is_active = False

    def record_frames(self, count: int = 1) -> None:
        """Records processed real-time perception frames and updates runtime."""
        with self._lock:
            now = time.monotonic()
            if self.is_active:
                self.active_runtime_seconds += max(0.0, now - self.last_active_time)
                self.last_active_time = now
            self.frames_processed += count

    def record_action(self, action_type: str = "click") -> None:
        """Records a physical motor action (mouse click, keystroke, etc.)."""
        with self._lock:
            self.motor_actions += 1
            self.action_breakdown[action_type] = self.action_breakdown.get(action_type, 0) + 1

    def record_query(self) -> None:
        """Records a visual search or affordance query."""
        with self._lock:
            self.visual_queries += 1

    def record_surprise(self) -> None:
        """Records a surprise trigger event (scene change / modal popup)."""
        with self._lock:
            self.surprises_detected += 1

    def get_current_runtime(self) -> float:
        """Returns total active perception runtime in seconds."""
        with self._lock:
            current = self.active_runtime_seconds
            if self.is_active:
                current += max(0.0, time.monotonic() - self.last_active_time)
            return current

    def calculate_cost(self) -> float:
        """Calculates current accrued cost in USD based on active rate card."""
        runtime_sec = self.get_current_runtime()
        with self._lock:
            runtime_min = runtime_sec / 60.0
            perception_cost = runtime_min * self.rate_card.perception_per_minute_usd
            motor_cost = self.motor_actions * self.rate_card.motor_action_usd
            query_cost = self.visual_queries * self.rate_card.query_usd
            return round(perception_cost + motor_cost + query_cost, 6)

    def calculate_savings(self) -> Dict[str, Any]:
        """Calculates estimated token and dollar savings compared to cloud screenshot VLMs."""
        with self._lock:
            actions = self.motor_actions + self.visual_queries
            # Without AVCS, an agent would have to send a full screenshot for each observation step
            vlm_baseline_cost = actions * self.rate_card.vlm_baseline_cost_per_action_usd
            total_tokens_saved = actions * self.rate_card.vlm_tokens_per_screenshot

        actual_cost = self.calculate_cost()
        net_saved_usd = max(0.0, vlm_baseline_cost - actual_cost)

        return {
            "vlm_baseline_cost_usd": round(vlm_baseline_cost, 4),
            "actual_avcs_cost_usd": actual_cost,
            "net_saved_usd": round(net_saved_usd, 4),
            "estimated_tokens_saved": total_tokens_saved,
            "savings_percentage": round((net_saved_usd / max(0.0001, vlm_baseline_cost)) * 100.0, 1)
            if vlm_baseline_cost > 0 else 0.0,
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns real-time usage metrics snapshot."""
        savings = self.calculate_savings()
        runtime_sec = self.get_current_runtime()
        with self._lock:
            return {
                "session_id": self.session_id,
                "org_id": self.org_id,
                "tier": self.tier,
                "active_runtime_seconds": round(runtime_sec, 2),
                "frames_processed": self.frames_processed,
                "motor_actions": self.motor_actions,
                "visual_queries": self.visual_queries,
                "surprises_detected": self.surprises_detected,
                "action_breakdown": dict(self.action_breakdown),
                "cost_usd": savings["actual_avcs_cost_usd"],
                "net_saved_usd": savings["net_saved_usd"],
                "tokens_saved": savings["estimated_tokens_saved"],
            }

    def generate_receipt(self) -> SessionUsageReceipt:
        """Issues an immutable, cryptographically signed receipt for billing reconciliation."""
        self.pause()
        now = time.time()
        runtime_sec = self.get_current_runtime()
        cost = self.calculate_cost()
        savings = self.calculate_savings()

        receipt_id = f"RCPT_{uuid.uuid4().hex[:12].upper()}"

        with self._lock:
            payload = {
                "receipt_id": receipt_id,
                "session_id": self.session_id,
                "org_id": self.org_id,
                "tier": self.tier,
                "started_at": self.started_at,
                "ended_at": now,
                "duration_seconds": round(runtime_sec, 2),
                "frames_processed": self.frames_processed,
                "motor_actions": self.motor_actions,
                "visual_queries": self.visual_queries,
                "surprises_detected": self.surprises_detected,
                "total_cost_usd": cost,
                "estimated_cost_saved_usd": savings["net_saved_usd"],
                "estimated_tokens_saved": savings["estimated_tokens_saved"],
            }

        raw_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = hmac.new(self.signing_key, raw_bytes, hashlib.sha256).hexdigest()

        return SessionUsageReceipt(
            receipt_id=receipt_id,
            session_id=self.session_id,
            org_id=self.org_id,
            tier=self.tier,
            started_at=self.started_at,
            ended_at=now,
            duration_seconds=round(runtime_sec, 2),
            frames_processed=payload["frames_processed"],
            motor_actions=payload["motor_actions"],
            visual_queries=payload["visual_queries"],
            surprises_detected=payload["surprises_detected"],
            total_cost_usd=cost,
            estimated_cost_saved_usd=savings["net_saved_usd"],
            estimated_tokens_saved=savings["estimated_tokens_saved"],
            signature=signature,
        )

    @classmethod
    def verify_receipt(
        cls,
        receipt: SessionUsageReceipt | Dict[str, Any],
        signing_key: Optional[str] = None,
    ) -> bool:
        """Validates cryptographic integrity of a billing receipt."""
        key_bytes = (
            signing_key
            or os.environ.get("AVCS_BILLING_KEY")
            or cls.DEFAULT_BILLING_KEY
        ).encode("utf-8")

        if isinstance(receipt, SessionUsageReceipt):
            data = receipt.to_dict()
        else:
            data = dict(receipt)

        claimed_sig = data.pop("signature", None)
        if not claimed_sig:
            return False

        raw_bytes = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        expected_sig = hmac.new(key_bytes, raw_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(claimed_sig, expected_sig)

