from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_staging_compose_references_existing_build_and_config_paths() -> None:
    compose_path = REPO_ROOT / "infra" / "docker-compose.staging.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    for service in ("sonic-api", "sonic-worker", "sonic-dashboard"):
        build = compose["services"][service]["build"]
        context = (compose_path.parent / build["context"]).resolve()
        assert context.is_dir(), service
        assert (context / build["dockerfile"]).is_file(), service

    caddy_mount = compose["services"]["edge-proxy"]["volumes"][0].split(":", 1)[0]
    assert (compose_path.parent / caddy_mount).is_file()
    for service in ("sonic-api", "sonic-worker"):
        config_mounts = compose["services"][service]["volumes"]
        assert any(
            (compose_path.parent / mount.split(":", 1)[0]).resolve()
            == (REPO_ROOT / "configs").resolve()
            for mount in config_mounts
        )


def test_production_runtime_data_uses_a_declared_persistent_volume() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8"))
    core = compose["services"]["sonic-core"]

    assert "sonic_data_prod:/app/sonic_data" in core["volumes"]
    assert "sonic_data_prod" in compose["volumes"]
    assert "DATABASE_URL=sqlite+aiosqlite:////app/sonic_data/sonic_data.db" in core["environment"]
    assert "SONIC_MEMORY_DB_PATH=/app/sonic_data/sonic_data.db" in core["environment"]
    assert "SONIC_WORKSTATION_STATE_PATH=/app/sonic_data/workstations.json" in core["environment"]
