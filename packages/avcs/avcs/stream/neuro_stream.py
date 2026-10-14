"""Real-Time Biological Neuro-Streaming Engine.

Ultra-low-latency 60-120 FPS sensory-motor telemetry broadcasting for AI models,
real-time gaming controllers, and continuous video tracking agents.
"""
from __future__ import annotations

import collections
import dataclasses
import struct
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np

from forge.vision.cortex.model import CortexState
from forge.vision.cortex.sdk import ScreenEye


@dataclasses.dataclass(frozen=True)
class NeuroPacket:
    """High-frequency (60-120 Hz) biological sensory-motor telemetry packet."""
    timestamp: float
    seq: int
    fixation: Tuple[float, float]
    motor_flow: Tuple[float, float]
    contact_acquired: bool
    primary_category: str
    stability_state: str
    surprise_score: float
    inference_ms: float
    action_hotspots: List[Tuple[float, float, str, float]] = dataclasses.field(default_factory=list)
    visual_latent: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "seq": self.seq,
            "fixation": [round(self.fixation[0], 4), round(self.fixation[1], 4)],
            "motor_flow": [round(self.motor_flow[0], 4), round(self.motor_flow[1], 4)],
            "contact_acquired": self.contact_acquired,
            "primary_category": self.primary_category,
            "stability_state": self.stability_state,
            "surprise_score": round(self.surprise_score, 4),
            "inference_ms": round(self.inference_ms, 2),
            "action_hotspots": [
                [round(x, 4), round(y, 4), act, round(conf, 3)]
                for x, y, act, conf in self.action_hotspots
            ],
        }

    COMPACT_FMT_40 = "<dI5fB7x"
    COMPACT_FMT_33 = "<dI5fB"

    def pack_compact(self) -> bytes:
        """Serializes core coordinates and flow into 40-byte C-ABI aligned binary for IPC."""
        return struct.pack(
            self.COMPACT_FMT_40,
            float(self.timestamp),
            int(self.seq),
            float(self.fixation[0]),
            float(self.fixation[1]),
            float(self.motor_flow[0]),
            float(self.motor_flow[1]),
            float(self.surprise_score),
            1 if self.contact_acquired else 0,
        )

    @classmethod
    def unpack_compact(cls, data: bytes) -> Dict[str, Any]:
        """Unpacks compact binary stream telemetry (supports 40-byte aligned and legacy 33-byte)."""
        fmt = cls.COMPACT_FMT_40 if len(data) >= 40 else cls.COMPACT_FMT_33
        size = struct.calcsize(fmt)
        if len(data) < size:
            raise ValueError(f"Data length {len(data)} is too short for format {fmt} (needs {size} bytes)")
        ts, seq, fx, fy, flowx, flowy, surprise, contact = struct.unpack(fmt, data[:size])
        return {
            "timestamp": ts,
            "seq": seq,
            "fixation": (fx, fy),
            "motor_flow": (flowx, flowy),
            "surprise_score": surprise,
            "contact_acquired": bool(contact),
        }


class NeuroStreamBus:
    """Thread-safe zero-copy circular ring buffer & pub-sub broadcaster for 60Hz telemetry."""

    def __init__(self, buffer_size: int = 120) -> None:
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self._buffer: Deque[NeuroPacket] = collections.deque(maxlen=buffer_size)
        self._subscribers: List[Callable[[NeuroPacket], None]] = []
        self._packet_count = 0
        self._last_tick_time = time.time()
        self._fps = 0.0

    def publish(self, packet: NeuroPacket) -> None:
        with self._lock:
            self._buffer.append(packet)
            self._packet_count += 1
            now = packet.timestamp
            dt = now - self._last_tick_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps = 0.9 * self._fps + 0.1 * inst_fps if self._fps > 0 else inst_fps
            self._last_tick_time = now
            subs = list(self._subscribers)

        # Non-blocking notification to subscribers
        for sub in subs:
            try:
                sub(packet)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[NeuroPacket], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def get_latest(self) -> Optional[NeuroPacket]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_window(self, count: int = 10) -> List[NeuroPacket]:
        with self._lock:
            items = list(self._buffer)
            return items[-count:] if len(items) >= count else items

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packet_count


class RealTimeNeuroStreamer:
    """Connects AVCS ScreenEye perception ticks directly to the high-rate NeuroStreamBus."""

    def __init__(self, eye: ScreenEye, bus: Optional[NeuroStreamBus] = None) -> None:
        self.eye = eye
        self.bus = bus or NeuroStreamBus()
        self._seq = 0
        self._active = False
        self._setup_listener()

    def _create_packet(self, state: CortexState) -> NeuroPacket:
        return NeuroPacket(
            timestamp=time.time(),
            seq=self._seq,
            fixation=state.fixation,
            motor_flow=state.motor_flow,
            contact_acquired=state.contact_acquired,
            primary_category=state.primary_category,
            stability_state=state.stability_state,
            surprise_score=state.surprise_score,
            inference_ms=state.inference_ms,
            action_hotspots=list(state.action_hotspots or []),
            visual_latent=state.visual_latent,
        )

    def _setup_listener(self) -> None:
        def on_screen_state(state: CortexState) -> None:
            self._seq += 1
            packet = self._create_packet(state)
            self.bus.publish(packet)

        self.eye.on_change(on_screen_state)

    def start(self) -> RealTimeNeuroStreamer:
        self._active = True
        self.eye.start()
        return self

    def stop(self) -> None:
        self._active = False
        self.eye.stop()

    def step_once(self) -> NeuroPacket:
        """Forces a single synchronous perception tick and broadcasts the resulting packet."""
        prev_seq = self._seq
        state = self.eye.perceive_now()
        latest = self.bus.get_latest()
        if latest is not None and self._seq > prev_seq:
            return latest

        self._seq += 1
        packet = self._create_packet(state)
        self.bus.publish(packet)
        return packet
