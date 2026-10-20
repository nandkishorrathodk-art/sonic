"""Regression guard for the ``forge.vision`` package import contract.

The workstation perception hot path imports ``forge.vision.reflex`` on every
fresh process.  If the package ``__init__`` eagerly imports the encoder stack,
``torch`` is dragged into that path (~1.1 s of pure import latency) even though
the reflex path never uses it.  These tests run in fresh subprocesses so the
result is independent of whatever the pytest session has already imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


def test_reflex_import_does_not_load_torch():
    result = _run(
        "import sys; import forge.vision.reflex; "
        "print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "NO_TORCH", (
        "importing forge.vision.reflex must not import torch; the package "
        "__init__ is eagerly importing the encoder stack again"
    )


def test_package_star_import_still_exposes_public_api():
    result = _run(
        "import forge.vision as fv; "
        "missing = [n for n in fv.__all__ if not hasattr(fv, n)]; "
        "print('OK' if not missing else 'MISSING:' + ','.join(missing))"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK", result.stdout


def test_encoder_symbol_resolves_lazily():
    result = _run(
        "import sys; from forge.vision import SceneUnderstandingModel; "
        "print(SceneUnderstandingModel.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SceneUnderstandingModel"


def test_unknown_attribute_raises_attribute_error():
    result = _run(
        "import forge.vision as fv; "
        "\ntry:\n    fv.NotARealSymbol\nexcept AttributeError:\n    print('RAISED')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RAISED"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
