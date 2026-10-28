"""Tests for verifiable AVCS model bundle packaging.

The point of these tests is the honesty gate: a checkpoint with no named
provenance, an unloadable file, or a tampered bundle must NOT be reported as
packed/valid. A packer that accepts anything is worse than no packer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from avcs.model_bundle import (
    BUNDLE_SCHEMA_VERSION,
    load_bundle,
    pack_model,
    verify_bundle,
)
from forge.vision.cortex.config import ModelConfig
from forge.vision.cortex.model import ArtificialVisualCortex


def _write_checkpoint(path: Path) -> Path:
    """Write a real, strict-loadable cortex state_dict."""
    model = ArtificialVisualCortex(ModelConfig())
    torch.save(model.state_dict(), path)
    return path


def _write_sidecar(checkpoint: Path, provenance: str | None) -> Path:
    sidecar = checkpoint.with_suffix(".eval.json")
    payload = {
        "checkpoint": checkpoint.name,
        "samples": 32,
        "category_accuracy": 0.75,
    }
    if provenance is not None:
        payload["dataset_provenance"] = provenance
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def test_pack_produces_manifest_with_checksums_and_provenance(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")

    result = pack_model(checkpoint, tmp_path / "out", archive=True)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["dataset_provenance"] == "real desktop capture: heldout.json"
    assert manifest["strict_load_verified"] is True
    assert {entry["name"] for entry in manifest["files"]} == {
        "eye.pt",
        "eye.eval.json",
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert result.archive_path is not None and result.archive_path.is_file()
    assert result.checkpoint_sha256 == next(
        entry["sha256"] for entry in manifest["files"] if entry["name"] == "eye.pt"
    )


def test_pack_refuses_checkpoint_without_evaluation_sidecar(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")

    with pytest.raises(FileNotFoundError, match="no evaluation sidecar"):
        pack_model(checkpoint, tmp_path / "out")


def test_pack_refuses_sidecar_without_named_provenance(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, None)

    with pytest.raises(ValueError, match="names no dataset_provenance"):
        pack_model(checkpoint, tmp_path / "out")


def test_pack_refuses_unloadable_checkpoint(tmp_path):
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_bytes(b"not a real torch state dict")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")

    with pytest.raises(Exception):
        pack_model(checkpoint, tmp_path / "out")


def test_pack_refuses_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pack_model(tmp_path / "absent.pt", tmp_path / "out")


def test_verify_bundle_accepts_intact_bundle(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")
    result = pack_model(checkpoint, tmp_path / "out")

    ok, problems = verify_bundle(result.bundle_dir)

    assert ok is True
    assert problems == []


def test_verify_bundle_detects_tampered_checkpoint(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")
    result = pack_model(checkpoint, tmp_path / "out")

    bundled = result.bundle_dir / "eye.pt"
    bundled.write_bytes(bundled.read_bytes() + b"\x00")

    ok, problems = verify_bundle(result.bundle_dir)

    assert ok is False
    assert any("checksum mismatch" in problem for problem in problems)


def test_verify_bundle_detects_missing_file(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")
    result = pack_model(checkpoint, tmp_path / "out")

    (result.bundle_dir / "eye.eval.json").unlink()

    ok, problems = verify_bundle(result.bundle_dir)

    assert ok is False
    assert any("missing bundled file" in problem for problem in problems)


def test_verify_bundle_rejects_directory_without_manifest(tmp_path):
    ok, problems = verify_bundle(tmp_path)

    assert ok is False
    assert any("missing manifest" in problem for problem in problems)


def test_pack_is_repeatable_into_the_same_output_dir(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")

    first = pack_model(checkpoint, tmp_path / "out")
    stale = first.bundle_dir / "stale.txt"
    stale.write_text("leftover", encoding="utf-8")

    second = pack_model(checkpoint, tmp_path / "out")

    assert second.bundle_dir == first.bundle_dir
    assert not stale.exists()
    ok, problems = verify_bundle(second.bundle_dir)
    assert ok is True, problems


def test_load_bundle_returns_model_and_manifest(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")
    result = pack_model(checkpoint, tmp_path / "out")

    model, manifest = load_bundle(result.bundle_dir)

    assert manifest["dataset_provenance"] == "real desktop capture: heldout.json"
    assert isinstance(model, ArtificialVisualCortex)
    assert not model.training


def test_load_bundle_refuses_tampered_bundle(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "eye.pt")
    _write_sidecar(checkpoint, "real desktop capture: heldout.json")
    result = pack_model(checkpoint, tmp_path / "out")
    bundled = result.bundle_dir / "eye.pt"
    bundled.write_bytes(bundled.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="unverifiable model bundle"):
        load_bundle(result.bundle_dir)
