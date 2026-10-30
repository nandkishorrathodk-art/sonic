"""
SONIC-REDA — Shared pytest configuration.

Honest environment gating: integration tests that require a live Docker daemon
or live Daytona Cloud sandboxes are SKIPPED (not failed) when those
dependencies are unavailable in the current environment. This keeps the suite
honest — no false-green from a skipped assertion, and no false-red from an
environment the test was never able to exercise.

Opt-in live integration:
    - Docker-daemon tests run only when `docker info` succeeds.
    - Live Daytona tests run only when SONIC_RUN_LIVE_DAYTONA=1 is set (they
      are slow and subject to cloud quota/concurrency limits; default off).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess

import pytest


def _docker_daemon_available() -> bool:
    """True only if the docker binary exists, the daemon answers `docker info`,
    and the isolated sandbox network `sonic-sandbox-net` exists.
    """
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        if proc.returncode != 0 or not bool(proc.stdout.strip()):
            return False
        net_proc = subprocess.run(
            ["docker", "network", "inspect", "sonic-sandbox-net"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return net_proc.returncode == 0
    except Exception:
        return False


# Resolved once per session.
_DOCKER_OK: bool | None = None


def _docker_ok() -> bool:
    global _DOCKER_OK
    if _DOCKER_OK is None:
        _DOCKER_OK = _docker_daemon_available()
    return _DOCKER_OK


# Module-level symbols that indicate a test exercises a live compute provider.
_DOCKER_SYMBOLS = ("DockerProvider", "UnifiedComputerProvider")
_DAYTONA_SYMBOLS = ("DaytonaComputerProvider",)


def _module_has_any(module, names) -> bool:
    if module is None:
        return False
    return any(getattr(module, n, None) is not None for n in names)


def _has_marker(request, name: str) -> bool:
    try:
        return name in {m.name for m in request.node.iter_markers()}
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _gate_live_compute(request):
    """Skip tests whose module imports a live compute provider when it is
    unavailable — UNLESS the test opts out via ``@pytest.mark.no_live_infra``,
    which marks it as exercising only the provider's state/safety/code-path
    logic with no live sandbox dependency.
    """
    if _has_marker(request, "no_live_infra"):
        return
    module = request.module
    if module is None:
        return
    if _module_has_any(module, _DOCKER_SYMBOLS) and not _docker_ok():
        pytest.skip("Docker daemon unavailable — live compute integration test skipped")
    if _module_has_any(module, _DAYTONA_SYMBOLS) and not os.environ.get("SONIC_RUN_LIVE_DAYTONA"):
        pytest.skip("Live Daytona integration test (set SONIC_RUN_LIVE_DAYTONA=1 to run)")


@pytest.fixture(autouse=True)
def isolate_all_sonic_test_state(tmp_path, monkeypatch):
    """Automatically isolate all test state (workstations, memory, being, craft)
    to a unique temporary directory for every single test in the test suite.
    Prevents any test from polluting production data or other tests.
    """
    tmp_data_dir = tmp_path / "sonic_data"
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    state_file = str(tmp_data_dir / "workstations.json")
    db_file = str(tmp_data_dir / "sonic_test.db")

    monkeypatch.setenv("SONIC_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("SONIC_WORKSTATION_STATE_FILE", state_file)
    monkeypatch.setenv("SONIC_WORKSTATION_STATE_PATH", state_file)
    monkeypatch.setenv("SONIC_MEMORY_DB_PATH", db_file)
    monkeypatch.setenv("SONIC_BEING_DB_PATH", db_file)

    def _reset_all_singletons():
        try:
            from sonic.api.routes.workstation import reset_workstation_state
            reset_workstation_state()
        except Exception:
            pass
        try:
            from sonic.memory.router import reset_memory_singleton
            reset_memory_singleton()
        except Exception:
            pass
        try:
            from sonic.memory.vector import reset_vector_memory_singleton
            reset_vector_memory_singleton()
        except Exception:
            pass
        try:
            from sonic.being.identity import reset_being_singleton
            reset_being_singleton()
        except Exception:
            pass
        try:
            from sonic.config import get_settings
            get_settings.cache_clear()
        except Exception:
            pass

    _reset_all_singletons()
    try:
        yield
    finally:
        _reset_all_singletons()


def pytest_configure(config):
    config.addinivalue_line("markers", "no_live_infra: provider state/safety logic only, no live sandbox")
