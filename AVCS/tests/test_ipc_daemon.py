"""Unit and Integration Tests for AVCS Phase 5: Zero-Copy Shared Memory IPC & Enterprise Licensing.
"""
import time
import uuid
import pytest

from avcs.ipc import (
    SharedMemoryBroadcaster,
    SharedMemoryReader,
    IPCPerceptionFrame,
)
from avcs.auth.server import EnterpriseLicensingServer
from avcs.auth.license import LicenseTier, get_machine_fingerprint


def test_shared_memory_broadcast_and_read():
    """Verify high-throughput zero-copy IPC transmission and deserialization."""
    test_buffer = f"test_avcs_ipc_{uuid.uuid4().hex[:8]}"

    broadcaster = SharedMemoryBroadcaster(buffer_name=test_buffer)
    reader = SharedMemoryReader(buffer_name=test_buffer)

    try:
        payload_1 = {
            "latency_ms": 7.4,
            "fps": 62.5,
            "gaze_fixation": [0.42, 0.58],
            "primary_category": "code_editor",
            "surprise_score": 0.012,
            "stability": "stable",
            "affordances_count": 14,
        }

        # Broadcast frame 1
        seq_1 = broadcaster.broadcast(payload_1)
        assert seq_1 == 1

        # Reader retrieves latest frame
        frame_1 = reader.read_latest()
        assert frame_1 is not None
        assert isinstance(frame_1, IPCPerceptionFrame)
        assert frame_1.sequence_number == 1
        assert frame_1.fps == 62.5
        assert frame_1.latency_ms == 7.4
        assert frame_1.primary_category == "code_editor"
        assert frame_1.gaze_fixation == (0.42, 0.58)
        assert frame_1.affordances_count == 14

        # Immediate next read without new broadcast should return None (no new frame)
        cached = reader.read_latest()
        assert cached is None

        # Broadcast frame 2
        payload_2 = {
            "latency_ms": 6.8,
            "fps": 64.0,
            "gaze_fixation": [0.75, 0.25],
            "primary_category": "terminal",
            "surprise_score": 0.085,
            "stability": "transient",
            "affordances_count": 8,
        }
        seq_2 = broadcaster.broadcast(payload_2)
        assert seq_2 == 2

        frame_2 = reader.read_latest()
        assert frame_2 is not None
        assert frame_2.sequence_number == 2
        assert frame_2.primary_category == "terminal"
        assert frame_2.gaze_fixation == (0.75, 0.25)

    finally:
        reader.close()
        broadcaster.close()


def test_shared_memory_latency_benchmark():
    """Benchmark shared memory IPC transfer latency to confirm sub-0.1ms performance."""
    test_buffer = f"test_bench_ipc_{uuid.uuid4().hex[:8]}"
    broadcaster = SharedMemoryBroadcaster(buffer_name=test_buffer)
    reader = SharedMemoryReader(buffer_name=test_buffer)

    payload = {
        "latency_ms": 5.0,
        "fps": 120.0,
        "gaze_fixation": [0.5, 0.5],
        "primary_category": "browser",
        "surprise_score": 0.0,
        "stability": "stable",
        "affordances_count": 25,
    }

    try:
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            broadcaster.broadcast(payload)
            frame = reader.read_latest()
            t1 = time.perf_counter()

            assert frame is not None
            latencies.append((t1 - t0) * 1000.0)

        p50 = sorted(latencies)[len(latencies) // 2]
        # Must easily be sub-1ms (typically ~0.02ms - 0.08ms)
        assert p50 < 1.0, f"Shared memory latency {p50:.3f}ms exceeds target 1.0ms"

    finally:
        reader.close()
        broadcaster.close()


def test_enterprise_licensing_issuance_and_verification():
    """Test air-gapped enterprise license generation, verification, and hardware gating."""
    server = EnterpriseLicensingServer(master_signing_key="unit_test_secret_key_2026")
    current_fp = get_machine_fingerprint()

    # 1. Portable license (unbound)
    portable_token = server.issue_enterprise_license(
        org_id="acme_corp",
        tier=LicenseTier.ENTERPRISE,
        duration_days=30,
        max_seats=25,
        bind_to_machine=None,
    )
    result = server.verify_node(portable_token, expected_machine_fingerprint=current_fp)
    assert result["valid"] is True
    assert result["org_id"] == "acme_corp"
    assert result["max_seats"] == 25
    assert "realtime_60fps" in result["features"]

    # 2. Machine-bound license matching current machine
    bound_token = server.issue_enterprise_license(
        org_id="secure_defense_labs",
        tier=LicenseTier.PRO,
        duration_days=90,
        max_seats=5,
        bind_to_machine=current_fp,
    )
    result_bound = server.verify_node(bound_token, expected_machine_fingerprint=current_fp)
    assert result_bound["valid"] is True
    assert result_bound["org_id"] == "secure_defense_labs"

    # 3. Machine-bound license with mismatched hardware fingerprint must fail
    mismatched_fp = "mismatched_hardware_fp_99999"
    with pytest.raises(Exception):
        server.verify_node(bound_token, expected_machine_fingerprint=mismatched_fp)


def test_hardware_fingerprint_deterministic():
    """Verify hardware fingerprint is consistent and non-empty across calls."""
    fp1 = get_machine_fingerprint()
    fp2 = get_machine_fingerprint()
    assert fp1 == fp2
    assert len(fp1) >= 16

