"""The grounder trainer must not exhaust container shared memory.

A hardcoded ``num_workers=2`` crashed with "unable to allocate shared memory"
under the Docker default ``/dev/shm`` of 64 MB -- i.e. training failed to start
in exactly the container it runs in. The worker count is now derived from the
available shared memory.
"""

from __future__ import annotations

from avcs.training.train_grounder import (
    _MIN_SHM_FOR_WORKERS_BYTES,
    _resolve_loader_workers,
)


class TestLoaderWorkerResolution:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("SONIC_AVCS_DATALOADER_WORKERS", "1")
        assert _resolve_loader_workers() == 1

    def test_env_override_is_clamped_non_negative(self, monkeypatch):
        monkeypatch.setenv("SONIC_AVCS_DATALOADER_WORKERS", "-5")
        assert _resolve_loader_workers() == 0

    def test_invalid_env_override_is_ignored(self, monkeypatch):
        monkeypatch.setenv("SONIC_AVCS_DATALOADER_WORKERS", "not-a-number")
        assert _resolve_loader_workers() >= 0

    def test_small_shm_disables_workers(self, monkeypatch):
        """The 64 MB Docker default must fall back to in-process loading."""
        monkeypatch.delenv("SONIC_AVCS_DATALOADER_WORKERS", raising=False)

        class _SmallShm:
            f_bavail = (64 * 1024 * 1024) // 4096
            f_frsize = 4096

        monkeypatch.setattr(
            "avcs.training.train_grounder.os.statvfs", lambda _p: _SmallShm()
        )
        assert _resolve_loader_workers() == 0

    def test_large_shm_allows_workers(self, monkeypatch):
        monkeypatch.delenv("SONIC_AVCS_DATALOADER_WORKERS", raising=False)

        class _LargeShm:
            f_bavail = (2 * 1024 * 1024 * 1024) // 4096
            f_frsize = 4096

        monkeypatch.setattr(
            "avcs.training.train_grounder.os.statvfs", lambda _p: _LargeShm()
        )
        assert _resolve_loader_workers() == 2

    def test_missing_shm_fails_safe_to_in_process(self, monkeypatch):
        monkeypatch.delenv("SONIC_AVCS_DATALOADER_WORKERS", raising=False)

        def _boom(_path):
            raise OSError("no /dev/shm")

        monkeypatch.setattr("avcs.training.train_grounder.os.statvfs", _boom)
        assert _resolve_loader_workers() == 0

    def test_threshold_is_generous_enough_for_batches(self):
        # The 64 MB Docker default is far below the threshold; a real dev box is
        # far above it. The threshold must sit between, not at, those values.
        assert _MIN_SHM_FOR_WORKERS_BYTES > 64 * 1024 * 1024
        assert _MIN_SHM_FOR_WORKERS_BYTES <= 2 * 1024 * 1024 * 1024
