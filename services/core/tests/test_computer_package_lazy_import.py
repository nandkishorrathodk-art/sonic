"""Regression guard for the ``sonic.computer`` package import contract.

The computer-use hot path (``sonic.computer_use.agent`` ->
``from sonic.computer.models import ...``) runs this package ``__init__`` before
any submodule on every fresh process.  If the package ``__init__`` eagerly
imports ``bootstrap``/``headless``/``benchmark``, the ``httpx`` client and the
subprocess stack are dragged into that path even though the reflex/perception
hot path never uses them.  These tests run in fresh subprocesses so the result
is independent of whatever the pytest session has already imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_CORE_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_CORE_ROOT),
    )


def test_models_import_does_not_load_httpx_or_heavy_submodules():
    result = _run(
        "import sys; import sonic.computer.models as m; "
        "print('HEAVY' if ('httpx' in sys.modules "
        "or 'sonic.computer.headless' in sys.modules "
        "or 'sonic.computer.bootstrap' in sys.modules "
        "or 'sonic.computer.benchmark' in sys.modules) else 'LIGHT')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LIGHT", (
        "importing sonic.computer.models must not import httpx/headless/"
        "bootstrap/benchmark; the package __init__ is eagerly importing "
        "them again"
    )


def test_lazy_public_api_still_resolves():
    result = _run(
        "import sonic.computer as c; "
        "names = ['ComputerProvider', 'UnifiedComputerProvider', "
        "'ScreenObservation', 'HeadlessComputeProvider', "
        "'WorkstationBootstrapEngine', 'AutonomousEngineerBenchmark']; "
        "missing = [n for n in names if not hasattr(c, n)]; "
        "print('OK' if not missing else 'MISSING:' + ','.join(missing))"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK", result.stdout


def test_dir_lists_public_names():
    result = _run(
        "import sonic.computer as c; "
        "print('YES' if 'ScreenObservation' in dir(c) else 'NO')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "YES"


def test_unknown_attribute_raises_attribute_error():
    result = _run(
        "import sonic.computer as c\n"
        "try:\n    c.NotARealSymbol\n"
        "except AttributeError:\n    print('RAISED')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RAISED"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
