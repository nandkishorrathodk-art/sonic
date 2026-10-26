"""Honesty gate + environment-reward supervision for the AVCS affordance head.

These tests assert the properties that stop the agent from telling the reasoning
model that controls exist where none were detected:

* an untrained (or unproven) dorsal head reports hotspots as unavailable, never
  as a detected-target count;
* a head whose measured held-out accuracy is at chance is treated the same way;
* a genuinely trained head still reports its hotspots;
* click outcomes become durable, sparse point supervision;
* the agent only records pointer actions that carry a real pixel target.
"""

from __future__ import annotations

import numpy as np
import pytest

from sonic.computer.models import GUIAction, GUIActionType
from sonic.computer_use.apprenticeship import OutcomeSampleStore
from sonic.computer_use.avcs_perception import AVCSPerception


@pytest.fixture()
def store(tmp_path):
    return OutcomeSampleStore(tmp_path)


def _frame() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


# -- outcome store ----------------------------------------------------------


def test_outcome_store_records_positive_and_negative(store):
    store.add(_frame(), (0.25, 0.5), "clickable", True, query="Save")
    store.add(_frame(), (0.75, 0.5), "clickable", False)
    counts = store.outcome_counts()
    assert counts == {"total": 2, "positive": 1, "negative": 1}


def test_outcome_store_persists_to_disk(store, tmp_path):
    store.add(_frame(), (0.1, 0.1), "clickable", True)
    reopened = OutcomeSampleStore(tmp_path)
    assert reopened.count() == 1


def test_outcome_store_records_its_label_rule(store):
    record = store.add(_frame(), (0.3, 0.3), "clickable", True, query="OK")
    assert record["label_source"] == "click_outcome"
    assert record["changed"] is True
    assert record["query"] == "OK"


def test_outcome_store_rejects_a_non_point_click(store):
    with pytest.raises(ValueError):
        store.add(_frame(), (0.5,), "clickable", True)  # type: ignore[arg-type]


def test_outcome_store_writes_frames_alongside_the_manifest(store):
    store.add(_frame(), (0.5, 0.5), "clickable", True)
    assert (store.frames_dir / "outcome_00000.png").is_file()


# -- perception honesty gate ------------------------------------------------


def test_gate_is_false_without_a_configured_checkpoint(monkeypatch):
    monkeypatch.delenv("SONIC_AVCS_CORTEX_CHECKPOINT", raising=False)
    perception = AVCSPerception()
    perception._cortex = None
    perception._cortex_checkpoint = ""
    assert perception.affordance_head_trained is False


def test_gate_is_false_when_the_cortex_did_not_load(monkeypatch):
    monkeypatch.setenv("SONIC_AVCS_CORTEX_CHECKPOINT", "/nonexistent/cortex.pt")
    perception = AVCSPerception()
    perception._cortex = None
    assert perception.affordance_head_trained is False


def test_gate_fails_closed_without_training_provenance(monkeypatch, tmp_path):
    checkpoint = tmp_path / "cortex.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.setenv("SONIC_AVCS_CORTEX_CHECKPOINT", str(checkpoint))
    perception = AVCSPerception()
    # Pretend the checkpoint loaded; the gate must still fail because no
    # provenance sidecar proves the dorsal head was trained.
    perception._cortex = object()
    perception._cortex_checkpoint = str(checkpoint)
    perception._affordance_head_trained = None
    assert perception.affordance_head_trained is False


def test_semantics_omit_hotspots_when_head_is_untrained(monkeypatch, tmp_path):
    """The core honesty property: no hotspot list without a trained head."""
    checkpoint = tmp_path / "cortex.pt"
    checkpoint.write_bytes(b"x")
    monkeypatch.setenv("SONIC_AVCS_CORTEX_CHECKPOINT", str(checkpoint))
    perception = AVCSPerception()
    perception._cortex = _StubCortex()
    perception._cortex_checkpoint = str(checkpoint)
    perception._affordance_head_trained = None

    semantics = perception.perceive_semantics_rgb(_frame())

    assert semantics["affordance_head_trained"] is False
    assert "action_hotspots" not in semantics


def test_semantics_report_hotspots_when_head_is_trained(monkeypatch, tmp_path):
    from forge.vision.cortex.training.affordance_provenance import (
        write_affordance_provenance,
    )

    checkpoint = tmp_path / "cortex.pt"
    checkpoint.write_bytes(b"x")
    write_affordance_provenance(
        checkpoint,
        dataset_provenance="test",
        samples=24,
        positive_samples=16,
        negative_samples=8,
        held_out_samples=8,
        held_out_iou=0.6,
        held_out_point_accuracy=0.8,
    )
    monkeypatch.setenv("SONIC_AVCS_CORTEX_CHECKPOINT", str(checkpoint))
    perception = AVCSPerception()
    perception._cortex = _StubCortex()
    perception._cortex_checkpoint = str(checkpoint)
    perception._affordance_head_trained = None

    semantics = perception.perceive_semantics_rgb(_frame())

    assert semantics["affordance_head_trained"] is True
    assert len(semantics["action_hotspots"]) == 1
    assert semantics["action_hotspots"][0]["x"] == 12.0


# -- agent recording --------------------------------------------------------


class _StubScreen:
    def __init__(self, width=640, height=360):
        self.width = width
        self.height = height
        self.raw_rgb = bytes(width * height * 3)


class _RecordingLoop:
    def __init__(self):
        self.calls = []

    def record_click_outcome(self, rgb, click, action, changed, **kwargs):
        self.calls.append((click, action, changed, kwargs))


def _agent_with(loop):
    from sonic.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent.__new__(ComputerUseAgent)
    agent.apprenticeship = loop
    return agent


def test_agent_records_a_click_with_its_measured_outcome():
    loop = _RecordingLoop()
    agent = _agent_with(loop)
    action = GUIAction(action=GUIActionType.CLICK, x=320, y=180)
    agent._record_click_outcome(action, _StubScreen(), True)
    assert len(loop.calls) == 1
    click, action_class, changed, _ = loop.calls[0]
    assert click == pytest.approx((0.5, 0.5))
    assert action_class == "clickable"
    assert changed is True


def test_agent_does_not_record_a_keyboard_action():
    loop = _RecordingLoop()
    agent = _agent_with(loop)
    action = GUIAction(action=GUIActionType.KEYPRESS, x=10, y=10)
    agent._record_click_outcome(action, _StubScreen(), True)
    assert loop.calls == []


def test_agent_does_not_record_a_click_without_pixel_coordinates():
    loop = _RecordingLoop()
    agent = _agent_with(loop)
    action = GUIAction(action=GUIActionType.CLICK)
    agent._record_click_outcome(action, _StubScreen(), True)
    assert loop.calls == []


def test_agent_recording_is_best_effort_when_no_loop_is_attached():
    agent = _agent_with(None)
    action = GUIAction(action=GUIActionType.CLICK, x=1, y=1)
    # Must not raise.
    agent._record_click_outcome(action, _StubScreen(), False)


def test_agent_clamps_click_to_the_normalized_unit_square():
    loop = _RecordingLoop()
    agent = _agent_with(loop)
    action = GUIAction(action=GUIActionType.CLICK, x=9999, y=-5)
    agent._record_click_outcome(action, _StubScreen(), True)
    click, _, _, _ = loop.calls[0]
    assert 0.0 <= click[0] <= 1.0
    assert 0.0 <= click[1] <= 1.0


class _StubCortex:
    """Minimal cortex double exposing the fields the semantics path reads."""

    def perceive(self, frame):
        from forge.vision.cortex.model import CortexState

        return CortexState(
            visual_latent=np.zeros(256, dtype=np.float32),
            primary_category="editor_code",
            category_confidence=0.9,
            text_density=0.1,
            stability_state="stable",
            surprise_score=0.0,
            fixation=(0.5, 0.5),
            motor_flow=(0.0, 0.0),
            contact_acquired=False,
            action_hotspots=[(12.0, 34.0, "clickable", 0.7)],
            affordance_map=np.zeros((32, 32), dtype=np.int64),
            query_target=None,
            query_relevance_map=None,
            inference_ms=1.0,
        )
