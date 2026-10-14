from __future__ import annotations

import base64
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from sonic.being.identity import (
    BeingStore,
    VaultKeyUnavailableError,
    VaultLegacyDataError,
    get_being_store,
    get_or_create_being,
    reset_being_singleton,
)
from sonic.being.vault import BeingVault


@pytest.fixture(autouse=True)
def _isolated_being_store(tmp_path, monkeypatch):
    db_path = str(tmp_path / "being.db")
    monkeypatch.setenv("SONIC_BEING_DB_PATH", db_path)
    monkeypatch.setenv("SONIC_MEMORY_DB_PATH", db_path)
    monkeypatch.setenv(
        "SONIC_BEING_VAULT_KEY",
        base64.urlsafe_b64encode(b"test-only-key-material-32-bytes!").decode("ascii"),
    )
    reset_being_singleton()
    yield
    reset_being_singleton()


def test_vault_encrypts_sensitive_values_at_rest_and_retrieves_them(tmp_path):
    db_path = str(tmp_path / "vault.db")
    vault = BeingVault("being-vault-test", db_path=db_path)
    vault.store_credential(
        "service.test",
        username="operator",
        secret="vault-secret-value",
        token="vault-token-value",
        profile_url="https://service.test/profile?token=also-sensitive",
    )
    vault.save_browser_state(
        {"cookies": [{"name": "session", "value": "browser-session-secret"}]},
        profile_key="primary",
    )

    with sqlite3.connect(db_path) as db:
        credential = db.execute(
            "SELECT secret, token, profile_url FROM being_vault"
        ).fetchone()
        state = db.execute(
            "SELECT storage_state_json FROM being_browser_state"
        ).fetchone()[0]
    persisted = " ".join((*credential, state))
    assert "vault-secret-value" not in persisted
    assert "vault-token-value" not in persisted
    assert "also-sensitive" not in persisted
    assert "browser-session-secret" not in persisted
    assert all(value.startswith("enc:v1:") for value in (*credential, state))

    vault = BeingVault("being-vault-test", db_path=db_path)
    retrieved = vault.get_credential("service.test")
    assert retrieved is not None
    assert retrieved["secret"] == "vault-secret-value"
    assert retrieved["token"] == "vault-token-value"
    assert retrieved["profile_url"] == "https://service.test/profile?token=also-sensitive"
    assert vault.load_browser_state("primary") == {
        "cookies": [{"name": "session", "value": "browser-session-secret"}]
    }


def test_vault_requires_configured_key_and_does_not_accept_wrong_key(tmp_path, monkeypatch):
    vault = BeingVault("being-vault-test", db_path=str(tmp_path / "vault.db"))
    vault.store_credential("service.test", secret="encrypted-secret")
    monkeypatch.delenv("SONIC_BEING_VAULT_KEY")
    with pytest.raises(VaultKeyUnavailableError, match="SONIC_BEING_VAULT_KEY"):
        vault.get_credential("service.test")

    monkeypatch.setenv(
        "SONIC_BEING_VAULT_KEY",
        base64.urlsafe_b64encode(b"w" * 32).decode("ascii"),
    )
    with pytest.raises(RuntimeError, match="authentication or decoding failed"):
        vault.get_credential("service.test")


def test_legacy_plaintext_is_not_returned_until_explicit_migration(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    store = BeingStore(db_path=db_path)
    store._db.execute(
        "INSERT INTO being_vault (being_id, service, username, secret, token, profile_url, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("being-legacy", "service.test", "operator", "legacy-secret", "legacy-token",
         "https://service.test/?token=legacy", "now"),
    )
    store._db.execute(
        "INSERT INTO being_browser_state (being_id, profile_key, storage_state_json, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("being-legacy", "primary", '{"cookies":[{"value":"legacy-cookie"}]}', "now"),
    )
    store._db.commit()

    vault = BeingVault("being-legacy", store=store)
    with pytest.raises(VaultLegacyDataError, match="explicitly"):
        vault.get_credential("service.test")
    with pytest.raises(VaultLegacyDataError, match="explicitly"):
        vault.load_browser_state("primary")
    with sqlite3.connect(db_path) as db:
        assert db.execute(
            "SELECT secret FROM being_vault WHERE being_id = ?", ("being-legacy",)
        ).fetchone()[0] == "legacy-secret"

    assert vault.migrate_legacy_data() == {"credentials": 1, "browser_states": 1}
    assert vault.get_credential("service.test") == {
        "being_id": "being-legacy",
        "service": "service.test",
        "username": "operator",
        "secret": "legacy-secret",
        "token": "legacy-token",
        "profile_url": "https://service.test/?token=legacy",
        "updated_at": "now",
    }
    assert vault.load_browser_state("primary") == {
        "cookies": [{"value": "legacy-cookie"}]
    }
    store.close()


def test_vault_migration_requires_key_before_touching_legacy_rows(tmp_path, monkeypatch):
    db_path = str(tmp_path / "legacy-no-key.db")
    store = BeingStore(db_path=db_path)
    store._db.execute(
        "INSERT INTO being_vault (being_id, service, username, secret, token, profile_url, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("being-legacy", "service.test", "operator", "secret", "", "", "now"),
    )
    store._db.commit()
    monkeypatch.delenv("SONIC_BEING_VAULT_KEY")
    with pytest.raises(VaultKeyUnavailableError):
        store.migrate_legacy_vault_data()
    assert store._db.execute("SELECT secret FROM being_vault").fetchone()[0] == "secret"
    store.close()


def test_concurrent_stores_atomically_create_one_being_per_tenant(tmp_path):
    db_path = str(tmp_path / "race.db")
    stores = [BeingStore(db_path=db_path), BeingStore(db_path=db_path)]
    barrier = threading.Barrier(2)

    def provision(store: BeingStore):
        barrier.wait()
        return store.get_or_create_being("race-tenant")[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        beings = list(pool.map(provision, stores))

    assert beings[0].being_id == beings[1].being_id
    with sqlite3.connect(db_path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM beings WHERE tenant_id = ?", ("race-tenant",)
        ).fetchone()[0] == 1
    for store in stores:
        store.close()


@pytest.mark.parametrize("operation", ["idle", "mission"])
def test_being_mind_persistence_errors_propagate(operation, monkeypatch):
    being = get_or_create_being(f"mind-{operation}")
    store = get_being_store()
    mind = store.get_mind(being.being_id)

    def fail_save(_mind):
        raise OSError("database unavailable")

    monkeypatch.setattr(store, "save_mind", fail_save)
    with pytest.raises(OSError, match="database unavailable"):
        if operation == "idle":
            mind.record_idle_cycle(was_novel=True, learned_fact="persist this")
        else:
            mind.record_mission_outcome("complete task", success=True)


def test_mind_mutations_persist_through_the_constructing_store(tmp_path, monkeypatch):
    singleton_path = str(tmp_path / "singleton.db")
    local_path = str(tmp_path / "local-store.db")
    monkeypatch.setenv("SONIC_BEING_DB_PATH", singleton_path)
    reset_being_singleton()
    singleton = get_being_store()
    local_store = BeingStore(db_path=local_path)
    being_id = "being-owned-by-local-store"

    # Ensure a global singleton exists; mind mutations must still use their owner.
    mind = local_store.get_mind(being_id)
    mind.record_idle_cycle(was_novel=True, learned_fact="local idle fact")
    mind.record_mission_outcome(
        "local mission",
        success=True,
        facts=["local mission fact"],
    )
    local_store.close()

    reopened_store = BeingStore(db_path=local_path)
    persisted = reopened_store.get_mind(being_id)
    assert persisted.idle_cycles_run == 1
    assert persisted.goals_pursued == 2
    assert persisted.goals_completed == 0
    assert persisted.learned_facts == ["local idle fact", "local mission fact"]

    persisted.record_mission_outcome(
        "verified mission",
        success=True,
        summary="Independent verifier confirmed the expected state.",
        verification_receipt_id="receipt-verified-1",
    )
    assert persisted.goals_completed == 1

    assert singleton._db.execute(
        "SELECT COUNT(*) FROM being_minds WHERE being_id = ?", (being_id,)
    ).fetchone()[0] == 0
    singleton.close()
    reopened_store.close()
