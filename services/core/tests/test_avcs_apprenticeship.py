"""AVCS apprenticeship: the eye learning from the being's own work.

These tests assert the honesty contract, not just the mechanics:

* a label comes from real freedesktop metadata, and an unlabelable window is
  SKIPPED rather than assigned a guessed class;
* the split keeps near-duplicate frames out of held-out data;
* a candidate is deployed only when it genuinely beats the current checkpoint on
  held-out frames -- a regression keeps the existing eye.

The training path is exercised with a real tiny model so the loop's gate is
tested against real numbers, and a stub is used to force a *worse* candidate so
the rejection branch is reachable deterministically.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from sonic.computer_use.apprenticeship import (
    UI_CATEGORIES,
    ApprenticeshipStore,
    AVCSApprenticeshipLoop,
    DesktopSceneLabeler,
    _grouped_split,
    freedesktop_categories_to_ui,
    parse_desktop_categories,
)


# --------------------------------------------------------------------------
# Honest labeling
# --------------------------------------------------------------------------

class TestDesktopCategories:
    def test_parse_categories(self):
        text = "[Desktop Entry]\nName=X\nCategories=GTK;System;TerminalEmulator;\n"
        assert parse_desktop_categories(text) == ["GTK", "System", "TerminalEmulator"]

    def test_parse_missing_categories(self):
        assert parse_desktop_categories("[Desktop Entry]\nName=X\n") == []

    @pytest.mark.parametrize(
        ("categories", "expected"),
        [
            (["Network", "WebBrowser"], "browser_webpage"),
            (["GTK", "System", "TerminalEmulator"], "terminal_shell"),
            (["Development"], "editor_code"),
            (["System", "FileTools", "FileManager"], "form_input"),
            (["Settings", "DesktopSettings"], "settings_panel"),
            (["AudioVideo", "Player"], "media_playback"),
        ],
    )
    def test_standard_categories_map(self, categories, expected):
        assert freedesktop_categories_to_ui(categories) == expected

    def test_unmapped_category_returns_none(self):
        # RemoteAccess is a real category with no cortex analogue: no label,
        # never a guessed default.
        assert freedesktop_categories_to_ui(["Network", "RemoteAccess"]) is None

    def test_empty_categories_returns_none(self):
        assert freedesktop_categories_to_ui([]) is None


class TestDesktopSceneLabeler:
    _SCAN = (
        "xfce4-terminal\tXfce4-terminal\txfce4-terminal\t\tGTK;System;TerminalEmulator;\n"
        "firefox-esr\tfirefox\t/usr/lib/firefox-esr/firefox-esr\t\tNetwork;WebBrowser;\n"
        "remote\t\t\t\tNetwork;RemoteAccess;\n"
        "org.xfce.mousepad\t\tmousepad\t\tUtility;TextEditor;GTK;\n"
        "org.xfce.mousepad-settings\t\tmousepad --preferences\t\tXFCE;GTK;Settings;DesktopSettings;\n"
    )

    def _labeler(self) -> DesktopSceneLabeler:
        async def _exec(_cmd: str) -> str:
            return self._SCAN

        return DesktopSceneLabeler(exec_runner=_exec)

    def test_wm_class_resolves_to_label(self):
        labeler = self._labeler()
        label, source = asyncio.run(
            labeler.label_for("xfce4-terminal.Xfce4-terminal", True)
        )
        assert label == "terminal_shell"
        assert source.startswith("freedesktop_categories:")

    def test_instance_class_second_token_matches(self):
        labeler = self._labeler()
        label, _ = asyncio.run(labeler.label_for("firefox.Firefox", True))
        assert label == "browser_webpage"

    def test_exec_binary_resolves_when_startup_wm_class_absent(self):
        """A desktop entry without StartupWMClass still labels via its Exec binary.

        org.xfce.mousepad.desktop publishes no StartupWMClass; keying only on
        basename/StartupWMClass left its frames unlabeled, so the eye could
        never learn from that application.
        """
        labeler = self._labeler()
        label, source = asyncio.run(labeler.label_for("mousepad", True))
        assert label == "editor_code"
        assert source.startswith("freedesktop_categories:")

    def test_exec_collision_prefers_the_application_not_its_settings_dialog(self):
        """Two entries sharing an Exec binary must resolve to the app itself."""
        labeler = self._labeler()
        label, _ = asyncio.run(labeler.label_for("mousepad", True))
        assert label == "editor_code"  # not settings_panel

    def test_unmapped_window_yields_no_label(self):
        labeler = self._labeler()
        label, source = asyncio.run(labeler.label_for("weird.Thing", True))
        assert label is None
        assert source == "unmapped_wm_class"

    def test_bare_desktop_is_desktop_canvas(self):
        labeler = self._labeler()
        label, source = asyncio.run(labeler.label_for("", False))
        assert label == "desktop_canvas"
        assert source == "no_active_window"

    def test_focused_window_without_class_is_not_labeled(self):
        labeler = self._labeler()
        label, source = asyncio.run(labeler.label_for("", True))
        assert label is None
        assert source == "no_wm_class"


# --------------------------------------------------------------------------
# Durable store
# --------------------------------------------------------------------------

class TestApprenticeshipStore:
    def test_add_persists_frame_and_manifest(self, tmp_path):
        store = ApprenticeshipStore(root=tmp_path)
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        sample = store.add(rgb, "terminal_shell", "Xfce4-terminal", "freedesktop")
        assert (tmp_path / sample.frame).is_file()
        manifest = json.loads((tmp_path / "samples.json").read_text())
        assert manifest["label_rule"] == "freedesktop_categories_from_wm_class"
        assert manifest["samples"][0]["label"] == "terminal_shell"
        assert store.count() == 1
        assert store.label_counts() == {"terminal_shell": 1}


# --------------------------------------------------------------------------
# Split discipline
# --------------------------------------------------------------------------

class TestGroupedSplit:
    def test_no_group_leaks_across_split(self):
        samples = [
            {"label": lab, "wm_class": wm, "frame": f"frames/{lab}_{wm}_{i}.png"}
            for lab in ("terminal_shell", "browser_webpage")
            for wm in ("a", "b")
            for i in range(4)
        ]
        train, held = _grouped_split(samples, 0.25, 7)
        train_groups = {(s["label"], s["wm_class"]) for s in train}
        held_groups = {(s["label"], s["wm_class"]) for s in held}
        assert train_groups & held_groups == set()
        assert {s["label"] for s in held} <= {s["label"] for s in train}

    def test_single_group_class_stays_in_training(self):
        samples = [
            {"label": "desktop_canvas", "wm_class": "x", "frame": "frames/a.png"},
            {"label": "terminal_shell", "wm_class": "y", "frame": "frames/b.png"},
            {"label": "terminal_shell", "wm_class": "z", "frame": "frames/c.png"},
        ]
        train, held = _grouped_split(samples, 0.5, 3)
        assert any(s["label"] == "desktop_canvas" for s in train)
        assert all(s["label"] != "desktop_canvas" for s in held)


# --------------------------------------------------------------------------
# Capture: only honest labels are stored
# --------------------------------------------------------------------------

class TestCaptureRound:
    def test_unlabelable_frames_are_skipped_not_guessed(self, tmp_path):
        async def _exec(_cmd: str) -> str:
            return (
                "xfce4-terminal\tXfce4-terminal\txfce4-terminal\t\tGTK;System;TerminalEmulator;\n"
                "remote\t\t\t\tNetwork;RemoteAccess;\n"
            )

        store = ApprenticeshipStore(root=tmp_path)
        loop = AVCSApprenticeshipLoop(
            store=store, labeler=DesktopSceneLabeler(exec_runner=_exec)
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        seq = [
            (frame, "xfce4-terminal.Xfce4-terminal", True),  # stored
            (frame, "remote.Remote", True),  # unmapped -> skipped
            (frame, "", False),  # bare desktop -> stored
        ]

        async def source():
            return seq.pop(0) if seq else (None, "", False)

        stored = asyncio.run(loop.capture_round(source, max_samples=3, interval_s=0.0))
        assert stored == 2
        counts = store.label_counts()
        assert counts == {"terminal_shell": 1, "desktop_canvas": 1}
        assert "form_input" not in counts  # nothing guessed

    def test_no_frame_means_no_sample(self, tmp_path):
        store = ApprenticeshipStore(root=tmp_path)
        loop = AVCSApprenticeshipLoop(store=store)

        async def source():
            return None, "", False

        stored = asyncio.run(loop.capture_round(source, max_samples=2, interval_s=0.0))
        assert stored == 0
        assert store.count() == 0


# --------------------------------------------------------------------------
# Learn gate
# --------------------------------------------------------------------------

def _seed_real_frames(store: ApprenticeshipStore, labels=("terminal_shell", "browser_webpage")):
    """Write distinct synthetic-but-real PNG frames with honest labels."""
    from PIL import Image

    store.frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    samples = []
    idx = 0
    for lab in labels:
        for wm in ("a", "b", "c", "d", "e"):
            for _ in range(4):
                name = f"frame_{idx:05d}.png"
                # Distinguishable per-label structure so a real model can learn
                # within a few CPU epochs.
                base = 20 if lab == "terminal_shell" else 220
                arr = np.full((360, 640, 3), base, dtype=np.uint8)
                arr += rng.integers(0, 8, arr.shape, dtype=np.uint8)
                Image.fromarray(arr).save(store.frames_dir / name)
                samples.append(
                    {
                        "frame": f"frames/{name}",
                        "label": lab,
                        "wm_class": wm,
                        "label_source": "test",
                        "captured_at": 0.0,
                    }
                )
                idx += 1
    store.manifest_path.write_text(json.dumps({"samples": samples}))


class TestLearnGate:
    def test_insufficient_samples_is_honest_no_op(self, tmp_path):
        store = ApprenticeshipStore(root=tmp_path)
        loop = AVCSApprenticeshipLoop(store=store)
        result = asyncio.run(loop.learn())
        assert result.promoted is False
        assert "insufficient" in result.reason
        assert not loop.checkpoint_path.exists()

    def test_worse_candidate_is_rejected(self, tmp_path, monkeypatch):
        store = ApprenticeshipStore(root=tmp_path)
        _seed_real_frames(store)
        loop = AVCSApprenticeshipLoop(store=store, epochs=12)

        # Deploy a real checkpoint first.
        first = asyncio.run(loop.learn())
        assert first.promoted is True
        deployed = loop.checkpoint_path
        assert deployed.is_file()

        # Force a candidate that never learned: the gate must reject it.
        def _untrained(_manifest):
            import torch

            from forge.vision.cortex.config import ModelConfig
            from forge.vision.cortex.model import ArtificialVisualCortex

            out = loop.store.root / "runs" / "candidate.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ArtificialVisualCortex(ModelConfig()).state_dict(), out)

        monkeypatch.setattr(loop, "_train_candidate", _untrained)
        second = asyncio.run(loop.learn())
        assert second.promoted is False
        assert second.candidate_accuracy <= (second.deployed_accuracy or 0.0)
        # The deployed checkpoint is untouched by a rejected candidate.
        assert deployed.is_file()

    def test_improving_candidate_is_promoted_and_sidecar_written(self, tmp_path):
        store = ApprenticeshipStore(root=tmp_path)
        _seed_real_frames(store)
        loop = AVCSApprenticeshipLoop(store=store, epochs=12)
        result = asyncio.run(loop.learn())
        assert result.promoted is True
        assert result.candidate_accuracy > (1.0 / len(UI_CATEGORIES))
        assert loop.checkpoint_path.is_file()
        sidecar = loop.checkpoint_path.with_suffix(".eval.json")
        assert sidecar.is_file()
        data = json.loads(sidecar.read_text())
        assert data["samples"] == result.heldout_frames
        assert "apprenticeship" in data["dataset_provenance"]


class TestStatusSurface:
    def test_status_reports_real_state(self, tmp_path):
        store = ApprenticeshipStore(root=tmp_path)
        loop = AVCSApprenticeshipLoop(store=store)
        status = loop.status.to_dict()
        assert status["samples"] == 0
        assert status["promoted"] == 0
        assert status["ui_categories"] == list(UI_CATEGORIES)


class TestCortexCheckpointResolution:
    """A relative configured path must still load the real checkpoint.

    The backend is normally launched from ``services/core`` while ``.env``
    paths are written relative to the repo root; resolving only the CWD would
    silently load nothing and misreport the eye as untrained.
    """

    def test_absolute_existing_path_is_used(self, tmp_path, monkeypatch):
        from sonic.computer_use.avcs_perception import AVCSPerception

        ckpt = tmp_path / "cortex.pt"
        ckpt.write_bytes(b"x")
        monkeypatch.setenv("SONIC_AVCS_CORTEX_CHECKPOINT", str(ckpt))
        assert AVCSPerception._resolve_cortex_checkpoint() == str(ckpt.resolve())

    def test_relative_path_resolves_from_repo_root(self, monkeypatch):
        from pathlib import Path

        from sonic.computer_use.avcs_perception import AVCSPerception

        # sonic_data/avcs/cortex_real.pt is written relative to the repo root.
        monkeypatch.setenv(
            "SONIC_AVCS_CORTEX_CHECKPOINT", "sonic_data/avcs/cortex_real.pt"
        )
        resolved = AVCSPerception._resolve_cortex_checkpoint()
        repo_root = Path(__file__).resolve().parents[3]
        if (repo_root / "sonic_data/avcs/cortex_real.pt").is_file():
            assert resolved and Path(resolved).is_file()

    def test_missing_path_is_not_silently_substituted(self, tmp_path, monkeypatch):
        from sonic.computer_use.avcs_perception import AVCSPerception

        monkeypatch.setenv(
            "SONIC_AVCS_CORTEX_CHECKPOINT", str(tmp_path / "nope.pt")
        )
        # Returns the configured value so cortex_ready stays False (fail-closed)
        # rather than quietly loading some other checkpoint.
        resolved = AVCSPerception._resolve_cortex_checkpoint()
        assert resolved.endswith("nope.pt")

    def test_learned_checkpoint_used_when_unset(self, tmp_path, monkeypatch):
        from sonic.computer_use.avcs_perception import AVCSPerception

        monkeypatch.delenv("SONIC_AVCS_CORTEX_CHECKPOINT", raising=False)
        monkeypatch.setenv("SONIC_AVCS_APPRENTICESHIP_DIR", str(tmp_path))
        learned = tmp_path / "cortex_learned.pt"
        learned.write_bytes(b"x")
        assert AVCSPerception._resolve_cortex_checkpoint() == str(learned)
