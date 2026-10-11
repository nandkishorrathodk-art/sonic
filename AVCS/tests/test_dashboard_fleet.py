"""Unit tests for Multi-Agent Fleet Stream Aggregator and Prometheus metrics."""
from __future__ import annotations

import json
import pytest

from dashboard.fleet_manager import AVCSFleetManager
from avcs import SyntheticScreen, ScreenEye


def test_fleet_manager_registration_and_telemetry():
    """Verify fleet manager registers agent nodes and aggregates telemetry."""
    manager = AVCSFleetManager()

    eye1 = ScreenEye(source=SyntheticScreen(width=640, height=360), fps=10)
    eye2 = ScreenEye(source=SyntheticScreen(width=640, height=360), fps=10)

    node1 = manager.register_agent("agent-01", "VS Code Coder", eye=eye1)
    node2 = manager.register_agent("agent-02", "Browser Web QA", eye=eye2)

    assert len(manager.agents) == 2
    assert manager.primary_agent_id == "agent-01"

    # Start and tick briefly
    manager.start_fleet()
    import time
    time.sleep(0.3)
    manager.stop_fleet()

    # Verify frame capture and telemetry
    telemetry = manager.get_fleet_telemetry()
    assert telemetry["total_agents"] == 2
    assert len(telemetry["agents"]) == 2

    # Verify Prometheus exposition output
    metrics = manager.get_prometheus_metrics()
    assert "avcs_agent_frames_total" in metrics
    assert "avcs_agent_tokens_saved" in metrics
    assert "avcs_agent_surprise" in metrics
    assert 'agent="agent-01"' in metrics
    assert 'agent="agent-02"' in metrics
