"""Unit tests for AVCS Real-Time Neuro-Streaming & Game Reflex Controller."""
from __future__ import annotations

import time
import numpy as np
import pytest

from avcs.stream import (
    NeuroPacket,
    NeuroStreamBus,
    RealTimeNeuroStreamer,
    GameReflexController,
)
from forge.vision.cortex.motor import VirtualCursorDriver
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


def test_neuro_packet_compact_packing():
    pkt = NeuroPacket(
        timestamp=1700000000.123,
        seq=42,
        fixation=(0.35, 0.65),
        motor_flow=(0.01, -0.02),
        contact_acquired=True,
        primary_category="editor_code",
        stability_state="stable_scene",
        surprise_score=0.12,
        inference_ms=3.7,
        action_hotspots=[(0.5, 0.5, "clickable", 0.9)],
    )

    data = pkt.pack_compact()
    assert len(data) == 40  # 40-byte C-ABI natural alignment

    unpacked = NeuroPacket.unpack_compact(data)
    assert unpacked["seq"] == 42
    assert abs(unpacked["fixation"][0] - 0.35) < 1e-4
    assert abs(unpacked["fixation"][1] - 0.65) < 1e-4
    assert abs(unpacked["motor_flow"][0] - 0.01) < 1e-4
    assert unpacked["contact_acquired"] is True
    assert abs(unpacked["surprise_score"] - 0.12) < 1e-4

    # Backwards compatibility test with legacy 33-byte payload
    legacy_33 = data[:33]
    unpacked_legacy = NeuroPacket.unpack_compact(legacy_33)
    assert unpacked_legacy["seq"] == 42
    assert abs(unpacked_legacy["surprise_score"] - 0.12) < 1e-4


def test_neuro_stream_bus_pub_sub_and_ring_buffer():
    bus = NeuroStreamBus(buffer_size=10)
    received = []

    unsub = bus.subscribe(lambda p: received.append(p.seq))

    for i in range(15):
        pkt = NeuroPacket(
            timestamp=time.time(),
            seq=i,
            fixation=(0.5, 0.5),
            motor_flow=(0.0, 0.0),
            contact_acquired=False,
            primary_category="browser_webpage",
            stability_state="stable_scene",
            surprise_score=0.05,
            inference_ms=3.5,
        )
        bus.publish(pkt)

    assert len(received) == 15
    assert bus.packet_count == 15
    # Ring buffer keeps only last 10
    window = bus.get_window(10)
    assert len(window) == 10
    assert window[-1].seq == 14
    assert window[0].seq == 5

    unsub()
    # After unsubscribe, no more calls
    bus.publish(window[-1])
    assert len(received) == 15


def test_realtime_streamer_and_reflex_controller():
    eye = ScreenEye(source=SyntheticScreen(width=320, height=240), device="cpu")
    bus = NeuroStreamBus(buffer_size=50)
    streamer = RealTimeNeuroStreamer(eye=eye, bus=bus)

    driver = VirtualCursorDriver(failsafe=False)
    controller = GameReflexController(eye=eye, bus=bus, driver=driver, force_virtual=True)

    # Test key hold and release
    controller.hold_key("w")
    assert "w" in controller._held_keys
    controller.release_key("w")
    assert "w" not in controller._held_keys

    # Test aim adjustment
    driver.move_to_normalized(0.1, 0.1)
    new_pos = controller.aim_towards((0.5, 0.5), gain=0.5, max_step=0.2)
    assert new_pos[0] > 0.1
    assert new_pos[1] > 0.1

    # Test surprise reflex hook
    reflex_triggered = []

    @controller.on_surprise_reflex(threshold=0.2)
    def on_surprise(pkt):
        reflex_triggered.append(pkt.seq)

    # Step perception and verify exactly ONE packet published per tick
    count_before = bus.packet_count
    p1 = streamer.step_once()
    assert p1.seq >= 1
    assert bus.packet_count == count_before + 1  # Deduplication verification

    # Test NaN aim safety guard (must not drift or raise)
    pos_before = driver.get_position_normalized()
    pos_after = controller.aim_towards((float("nan"), float("nan")), gain=0.5, max_step=0.2)
    assert pos_after == pos_before

    # Test driver held keys synchronization
    controller.hold_key("shift")
    assert "shift" in driver.held_keys
    controller.release_key("shift")
    assert "shift" not in driver.held_keys

    # Manually fire a surprise packet to verify reflex trigger
    surprise_pkt = NeuroPacket(
        timestamp=time.time(),
        seq=999,
        fixation=(0.5, 0.5),
        motor_flow=(0.0, 0.0),
        contact_acquired=False,
        primary_category="game_viewport",
        stability_state="surprise_event",
        surprise_score=0.85,
        inference_ms=3.6,
    )
    bus.publish(surprise_pkt)
    assert 999 in reflex_triggered

    controller.close()
    eye.close()
