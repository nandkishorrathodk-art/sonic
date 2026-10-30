"""Pack a trained AVCS checkpoint into a verifiable, shippable bundle.

A checkpoint file on its own is not shippable evidence: nothing records which
data it was measured on, whether it still loads, or whether the bytes an
operator received are the bytes that were evaluated. ``pack_model`` therefore
refuses to package a checkpoint that cannot strict-load or that has no
evaluation sidecar with named provenance, and writes a ``manifest.json`` holding
a SHA-256 for every bundled file so ``verify_bundle`` can prove integrity later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA_VERSION = 1
DEFAULT_MODEL_TYPE = "avcs_cortex"


@dataclass(frozen=True)
class PackResult:
    """Where a packed bundle landed and the checksum of its checkpoint."""

    bundle_dir: Path
    manifest_path: Path
    checkpoint_sha256: str
    archive_path: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_info() -> dict[str, str]:
    info = {"python": platform.python_version()}
    try:
        import torch

        info["torch"] = str(torch.__version__)
    except Exception:
        pass
    try:
        from avcs import __version__ as avcs_version

        info["avcs"] = str(avcs_version)
    except Exception:
        pass
    return info


def _read_evaluation_sidecar(checkpoint: Path) -> tuple[Path, dict]:
    sidecar = checkpoint.with_suffix(".eval.json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"no evaluation sidecar for {checkpoint.name}; a metric without named "
            "provenance is not evidence, so the checkpoint is not shippable "
            f"(expected {sidecar.name})"
        )
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid evaluation sidecar {sidecar}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"evaluation sidecar {sidecar} must be a JSON object")
    provenance = data.get("dataset_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            f"evaluation sidecar {sidecar} names no dataset_provenance; refusing to "
            "package a model whose numbers are not tied to data"
        )
    return sidecar, data


def _verify_strict_load(checkpoint: Path) -> None:
    try:
        from forge.vision.cortex.training.evaluate import load_cortex_checkpoint
    except Exception as exc:  # pragma: no cover - import environment problem
        raise RuntimeError(
            f"cannot import the cortex loader to verify {checkpoint.name}: {exc}"
        ) from exc
    load_cortex_checkpoint(checkpoint)


def pack_model(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    name: str | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    archive: bool = False,
) -> PackResult:
    """Bundle a loadable, evaluated checkpoint with provenance and checksums."""
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    _verify_strict_load(checkpoint)
    sidecar, sidecar_data = _read_evaluation_sidecar(checkpoint)

    bundle_name = name or f"{checkpoint.stem}-bundle"
    bundle_dir = Path(output_dir).expanduser() / bundle_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    bundled_checkpoint = bundle_dir / checkpoint.name
    bundled_sidecar = bundle_dir / sidecar.name
    shutil.copy2(checkpoint, bundled_checkpoint)
    shutil.copy2(sidecar, bundled_sidecar)

    files = [
        {
            "name": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (bundled_checkpoint, bundled_sidecar)
    ]

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle": bundle_name,
        "model_type": model_type,
        "created_at": time.time(),
        "source_checkpoint": str(checkpoint),
        "checkpoint_file": bundled_checkpoint.name,
        "dataset_provenance": sidecar_data["dataset_provenance"].strip(),
        "strict_load_verified": True,
        "evaluation": sidecar_data,
        "files": files,
        "runtime": _runtime_info(),
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path = None
    if archive:
        archive_path = Path(output_dir).expanduser() / f"{bundle_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_name)

    return PackResult(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        checkpoint_sha256=next(
            entry["sha256"]
            for entry in files
            if entry["name"] == bundled_checkpoint.name
        ),
        archive_path=archive_path,
    )


def verify_bundle(bundle_dir: str | Path) -> tuple[bool, list[str]]:
    """Re-check a bundle's manifest and every recorded checksum."""
    bundle = Path(bundle_dir).expanduser()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return False, [f"missing manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"invalid manifest JSON: {exc}"]
    if not isinstance(manifest, dict):
        return False, ["manifest must be a JSON object"]

    problems: list[str] = []
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        problems.append(f"unexpected schema_version: {manifest.get('schema_version')!r}")
    if not str(manifest.get("dataset_provenance", "")).strip():
        problems.append("manifest names no dataset_provenance")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        problems.append("manifest lists no files")
        return False, problems
    for entry in files:
        if not isinstance(entry, dict) or "name" not in entry:
            problems.append(f"malformed file entry: {entry!r}")
            continue
        target = bundle / str(entry["name"])
        if not target.is_file():
            problems.append(f"missing bundled file: {entry['name']}")
            continue
        actual = _sha256(target)
        if actual != entry.get("sha256"):
            problems.append(
                f"checksum mismatch for {entry['name']}: "
                f"expected {entry.get('sha256')}, found {actual}"
            )
    return (not problems), problems


def load_bundle(bundle_dir: str | Path, *, device: str = "cpu") -> tuple[Any, dict]:
    """Verify a bundle, then load its checkpoint. Returns ``(model, manifest)``.

    Verification happens first: a caller must never load weights from a bundle
    whose bytes do not match its manifest, or a corrupted/hand-edited checkpoint
    would silently be trusted as the evaluated model.
    """
    ok, problems = verify_bundle(bundle_dir)
    if not ok:
        raise ValueError(
            "refusing to load an unverifiable model bundle: " + "; ".join(problems)
        )
    bundle = Path(bundle_dir).expanduser()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    from forge.vision.cortex.training.evaluate import load_cortex_checkpoint

    model = load_cortex_checkpoint(bundle / str(manifest["checkpoint_file"]), device=device)
    return model, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pack or verify a shippable AVCS model bundle"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="bundle a loadable, evaluated checkpoint")
    pack.add_argument("--checkpoint", required=True)
    pack.add_argument("--output-dir", default="runs/model_bundles")
    pack.add_argument("--name", default=None)
    pack.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    pack.add_argument("--archive", action="store_true")

    verify = sub.add_parser("verify", help="re-check a bundle's checksums")
    verify.add_argument("--bundle", required=True)

    args = parser.parse_args(argv)
    if args.command == "pack":
        result = pack_model(
            args.checkpoint,
            args.output_dir,
            name=args.name,
            model_type=args.model_type,
            archive=args.archive,
        )
        print(f"Packed {result.bundle_dir}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Checkpoint SHA-256: {result.checkpoint_sha256}")
        if result.archive_path:
            print(f"Archive: {result.archive_path}")
        return 0

    ok, problems = verify_bundle(args.bundle)
    if ok:
        print(f"Bundle OK: {args.bundle}")
        return 0
    print(f"Bundle FAILED: {args.bundle}")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
