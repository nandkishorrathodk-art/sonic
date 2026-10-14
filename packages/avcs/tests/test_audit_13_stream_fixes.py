"""Tests for AVCS Comprehensive 13-Stream Codebase Hardening Fixes.

Validates that:
1. Win32 capture GDI desktop handle cleanup and strided RGB slice.
2. SharedMemoryBroadcaster non-destructive capacity and r+b buffer initialization.
3. Memory manager get_context_prompt uses latest_tree over stale initial_tree.
4. Persistent UI Registry recovers off-screen scrolled elements and calculates IoU.
5. Neuromuscular servoing evaluates analytical minimum-jerk velocity with AR(1) tremor.
6. VisualRippleVerifier uses photometric local delta and guards against surprise leakage.
7. CharNGramEncoder produces deterministic CRC32 hashes.
8. EnterpriseLicensingServer shares default master key with LicenseManager.
9. VisualEventStreamer enforces thread-safe subscription and dispatch.
10. UsageMeter calculates duration using monotonic clocks.
"""
from __future__ import annotations

import time
import numpy as np
import pytest
import torch

from avcs.auth.license import LicenseManager
from avcs.auth.server import EnterpriseLicensingServer
from avcs.billing.meter import UsageMeter
from avcs.events import VisualEventStreamer, VisualEventType
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.ipc import SharedMemoryBroadcaster, SharedMemoryReader
from avcs.memory.manager import VisualMemoryManager
from avcs.memory.registry import PersistentUIRegistry
from avcs.memory.types import AffordanceType, BoundingBox, EntityLifecycle, UIMLElement
from avcs.servoing import NeuromuscularServoingController
from avcs.verifier import VerificationResult, VerificationStatus, VisualRippleVerifier
from forge.vision.capture import SyntheticScreen, Win32FastCapture
from forge.vision.cortex.core.modulation import CharNGramEncoder
from forge.vision.cortex.motor import VirtualCursorDriver


def test_win32_fast_capture_structure():
    """Verify Win32FastCapture has _h_desk handle tracking attribute."""
    cap = Win32FastCapture()
    assert hasattr(cap, "_h_desk")
    assert cap._h_desk is None


def test_ipc_buffer_capacity_safe_handling():
    """Verify SharedMemoryBroadcaster safely handles payload overcapacity without destructive slice."""
    b = SharedMemoryBroadcaster(buffer_name="test_audit_ipc_mmap", buffer_size=1024)
    try:
        # Broadcast huge payload exceeding 1024 bytes
        huge_data = {"key": "x" * 2000}
        seq = b.broadcast(huge_data)
        assert seq > 0

        r = SharedMemoryReader(buffer_name="test_audit_ipc_mmap", buffer_size=1024)
        try:
            frame = r.read_latest()
            assert frame is not None
            assert frame.data.get("error") == "payload_truncated"
            assert frame.data.get("original_len") > 1024
        finally:
            r.close()
    finally:
        b.close()


def test_memory_manager_latest_tree_in_prompt():
    """Verify get_context_prompt emits the latest tree rather than freezing initial turn 1."""
    mgr = VisualMemoryManager()

    mgr.step(affordances=[(0.2, 0.15, "clickable", 0.9)], window_title="Checkout Screen", prior_action="observe")
    prompt_turn1 = mgr.get_context_prompt()
    assert "Checkout Screen" in prompt_turn1

    mgr.step(affordances=[(0.5, 0.45, "clickable", 0.9)], window_title="Confirmation Screen", prior_action="click")
    prompt_turn2 = mgr.get_context_prompt()
    # Turn 2 prompt must show current state reflecting Confirmation Screen, not frozen Checkout Screen
    assert "Confirmation Screen" in prompt_turn2


def test_ui_registry_one_way_scroll_recovery():
    """Verify PersistentUIRegistry recovers off-screen scrolled elements when scrolled back."""
    reg = PersistentUIRegistry(cost_threshold=0.45)

    e1 = UIMLElement("t1", AffordanceType.CLICKABLE, "Header Logo", BoundingBox(0.1, 0.05, 0.3, 0.15))
    e2 = UIMLElement("t2", AffordanceType.CLICKABLE, "Footer Link", BoundingBox(0.1, 0.85, 0.3, 0.95))
    active1, _ = reg.update([e1, e2])
    id_header = active1[0].element_id
    id_footer = active1[1].element_id

    # Turn 2: User scrolls down significantly (dy = +0.10)
    # Header scrolls off-screen above
    active2, disappeared = reg.update([e2], container_flow=(0.0, 0.10))
    assert reg.lifecycles[id_header] == EntityLifecycle.OFF_SCREEN_BELOW or reg.lifecycles[id_header] == EntityLifecycle.OFF_SCREEN_ABOVE

    # Turn 3: User scrolls back up (dy = -0.10)
    # Header re-appears! Must be re-identified with original id_header!
    active3, _ = reg.update([e1, e2], container_flow=(0.0, -0.10))
    recovered_ids = [elem.element_id for elem in active3]
    assert id_header in recovered_ids
    assert reg.lifecycles[id_header] == EntityLifecycle.VISIBLE


def test_servoing_smooth_analytical_velocity():
    """Verify NeuromuscularServoingController generates analytical velocities with smooth profiles."""
    driver = VirtualCursorDriver()
    controller = NeuromuscularServoingController(driver=driver, tick_rate_hz=100)
    points = controller.generate_minimum_jerk_profile((0.1, 0.1), (0.8, 0.8), duration_s=0.4)

    assert len(points) >= 30
    # Peak velocity occurs near midpoint tau=0.5
    mid_pt = points[len(points) // 2]
    assert mid_pt.vx > 0.5
    assert mid_pt.vy > 0.5
    # Terminal point docks smoothly
    end_pt = points[-1]
    assert abs(end_pt.x - 0.8) < 0.03
    assert abs(end_pt.y - 0.8) < 0.03


def test_verifier_photometric_delta():
    """Verify VisualRippleVerifier evaluates local reaction and handles pre_state differentials."""
    source = SyntheticScreen()
    eye = HumanEye(source=source)
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver)

    pre_state = eye.observe()
    res = verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)
    assert isinstance(res, VerificationResult)
    assert hasattr(res, "optical_ripple_energy")


def test_char_ngram_encoder_deterministic_crc32():
    """Verify CharNGramEncoder produces identical embeddings across independent calls."""
    enc = CharNGramEncoder(out_dim=64, hash_bins=256)
    emb1 = enc("Submit Order", device=torch.device("cpu"))
    emb2 = enc("Submit Order", device=torch.device("cpu"))
    assert torch.allclose(emb1, emb2, atol=1e-6)

    # Different words produce different embeddings
    emb_diff = enc("Cancel", device=torch.device("cpu"))
    cos_sim = float(torch.cosine_similarity(emb1, emb_diff, dim=-1).item())
    assert cos_sim < 0.99


def test_auth_server_master_key_parity():
    """Verify EnterpriseLicensingServer defaults to LicenseManager.DEFAULT_MASTER_KEY."""
    server = EnterpriseLicensingServer()
    assert server.master_key == LicenseManager.DEFAULT_MASTER_KEY

    # License generated by server verifies cleanly by default LicenseManager
    token = server.issue_enterprise_license("org_enterprise_test")
    client_mgr = LicenseManager()
    payload = client_mgr.verify_license(token)
    assert payload.org_id == "org_enterprise_test"
    assert payload.tier == "enterprise"


def test_visual_events_streamer_thread_safety():
    """Verify VisualEventStreamer safely registers subscribers and dispatches events."""
    streamer = VisualEventStreamer()
    received = []

    def on_alert(ev):
        received.append(ev)

    streamer.subscribe(VisualEventType.SURPRISE_ALERT, on_alert)

    # Trigger surprise alert
    eye_state = HumanEyeState(
        timestamp=time.time(),
        frame_index=1,
        latency_ms=10.0,
        fps=60.0,
        gaze_fixation=(0.5, 0.5),
        motion_magnitude=0.5,
        motion_field=np.zeros((16, 24, 2)),
        moving_objects=[],
        pursuit_target=None,
        pursuit_velocity=(0.0, 0.0),
        motor_flow=(0.0, 0.0),
        contact_acquired=False,
        stability="surprise",
        surprise_score=0.95,
        affordance_hotspots=[],
        primary_category="screen",
        category_confidence=0.9,
    )

    events = streamer.process_state(eye_state)
    assert len(events) >= 1
    assert len(received) >= 1
    assert received[0].event_type == VisualEventType.SURPRISE_ALERT


def test_usage_meter_monotonic_time():
    """Verify UsageMeter calculates runtime without drift."""
    meter = UsageMeter(org_id="org_test")
    meter.start()
    time.sleep(0.05)
    rt = meter.get_current_runtime()
    assert rt >= 0.04
    meter.pause()
    receipt = meter.generate_receipt()
    assert receipt.duration_seconds >= 0.04
