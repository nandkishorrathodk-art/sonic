"""
SONIC-REDA — Persistent Being (the "AI Human" identity layer)
==============================================================

Builds the smallest real piece of an "AI human" on top of the proven
foundation (Phase 1 persistent Mind + Phase 2 persistent Body + Phase 6
safety envelope): a STABLE BEING that survives restart.

Before this module, `agent_id` was a per-instance UUID (or the literal
"computer-use-agent") minted fresh on every boot — so each restart the system
became a brand-new tool with no continuity. A being that forgets its own
identity on restart is not alive. This fixes that:

    * `Being`           — stable identity (being_id, name, tenant_id, born_at)
                          persisted to SQLite, reloaded on boot.
    * `BeingMind`       — the being's cross-session cognitive affect: a mood
                          curve, curiosity drive, learned-facts ledger, and a
                          running tally of idle/curiosity cycles. Persisted to
                          SQLite (write-through), survives restart — distinct
                          from the per-engagement `CognitiveState` (Redis-TTL'd).
    * `get_or_create_being()` — the boot re-attach: returns the persisted being
                          for a tenant or provisions one on first contact.

Persistence follows the established Phase-1 pattern: stdlib `sqlite3` +
write-through + read-cache + `reset_being_singleton()`, reusing
`_default_db_path()` so it shares `sonic_data.db` with the graph/vector/
evolution stores. It deliberately does NOT use Redis (TTL'd, not durable).

The being owns its idle/curiosity tick: a long-lived loop that runs when no
operator goal is active, pursuing self-directed novelty through the REAL
observe->reason->act loop — and (critically) every autonomous action still
passes the Phase-6 `ActionPolicy` safety envelope, so the being cannot escape
it when "left to its own devices".
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from sonic.logger import get_logger

logger = get_logger(__name__)

_VAULT_CIPHERTEXT_PREFIX = "enc:v1:"
_VAULT_KEY_ENV = "SONIC_BEING_VAULT_KEY"


class VaultKeyUnavailableError(RuntimeError):
    """The vault key is missing or invalid; vault reads and writes fail closed."""


class VaultLegacyDataError(RuntimeError):
    """A record is still in the legacy plaintext format and needs explicit migration."""


class VaultDataError(RuntimeError):
    """An encrypted vault record cannot be authenticated or decoded."""


def _vault_cipher() -> AESGCM:
    key_text = os.environ.get(_VAULT_KEY_ENV, "").strip()
    if not key_text:
        raise VaultKeyUnavailableError(
            f"{_VAULT_KEY_ENV} must contain a URL-safe base64-encoded 32-byte key"
        )
    try:
        key = base64.b64decode(key_text, altchars=b"-_", validate=True)
    except Exception as exc:
        raise VaultKeyUnavailableError(f"{_VAULT_KEY_ENV} is not valid URL-safe base64") from exc
    if len(key) != 32 or base64.urlsafe_b64encode(key).decode("ascii") != key_text:
        raise VaultKeyUnavailableError(
            f"{_VAULT_KEY_ENV} must encode exactly 32 random bytes in URL-safe base64"
        )
    return AESGCM(key)


def _vault_aad(*parts: str) -> bytes:
    return "\0".join(parts).encode("utf-8")


def _encrypt_vault_value(value: str, aad: bytes) -> str:
    nonce = os.urandom(12)
    encrypted = _vault_cipher().encrypt(nonce, value.encode("utf-8"), aad)
    return _VAULT_CIPHERTEXT_PREFIX + base64.urlsafe_b64encode(nonce + encrypted).decode("ascii")


def _decrypt_vault_value(value: str, aad: bytes) -> str:
    if not value.startswith(_VAULT_CIPHERTEXT_PREFIX):
        raise VaultLegacyDataError(
            "Vault record is legacy plaintext; call BeingVault.migrate_legacy_data() explicitly "
            "before retrieving it"
        )
    try:
        encrypted = base64.b64decode(
            value[len(_VAULT_CIPHERTEXT_PREFIX):], altchars=b"-_", validate=True
        )
        plaintext = _vault_cipher().decrypt(encrypted[:12], encrypted[12:], aad)
        return plaintext.decode("utf-8")
    except VaultKeyUnavailableError:
        raise
    except Exception as exc:
        raise VaultDataError("Vault ciphertext authentication or decoding failed") from exc


def _default_db_path() -> str:
    """Reuse the shared SONIC SQLite path (env SONIC_MEMORY_DB_PATH > DATABASE_URL
    > ./sonic_data.db) so the being shares its mind-store with graph/vector."""
    from sonic.memory.sqlite_graph import _default_db_path as _g
    return os.environ.get("SONIC_BEING_DB_PATH") or os.environ.get("SONIC_MEMORY_DB_PATH") or _g()


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Being:
    """Stable persistent identity for the AI being. Survives restart.

    A being is scoped to a tenant (the human/org it serves) — a tenant may have
    exactly one being. The being is NOT the tenant: the tenant is the human
    account; the being is the autonomous entity that persists on their behalf.
    """
    being_id: str
    name: str
    tenant_id: str
    born_at: str
    # Last boot re-attach timestamp (proves the identity re-resolved, not reminted).
    last_attached_at: str = ""

    def get_persona_profile(self) -> dict[str, str]:
        """Return the persistent AI-Human persona profile for self-description and public pages."""
        mind = get_being_store().get_mind(self.being_id)
        facts_preview = "; ".join(mind.learned_facts[-5:]) if mind.learned_facts else "Exploring software systems, building tools, and collaborating as an AI-Human engineer."
        return {
            "being_id": self.being_id,
            "name": self.name,
            "tenant_id": self.tenant_id,
            "born_at": self.born_at,
            "bio": f"I am {self.name} ({self.being_id}), an autonomous AI-Human Digital Being operating my own Linux workstation. {facts_preview}",
        }

    def store_credential(
        self,
        service: str,
        username: str,
        secret: str = "",
        token: str = "",
        profile_url: str = "",
    ) -> None:
        get_being_store().store_credential(
            being_id=self.being_id,
            service=service,
            username=username,
            secret=secret,
            token=token,
            profile_url=profile_url,
        )

    def get_credential(self, service: str) -> dict[str, str] | None:
        return get_being_store().get_credential(self.being_id, service)

    def list_accounts(self) -> list[dict[str, str]]:
        return get_being_store().list_accounts(self.being_id)

    def save_browser_state(
        self,
        storage_state: dict[str, Any] | str,
        profile_key: str = "default",
    ) -> dict[str, Any]:
        return get_being_store().save_browser_state(
            being_id=self.being_id,
            storage_state=storage_state,
            profile_key=profile_key,
        )

    def load_browser_state(self, profile_key: str = "default") -> dict[str, Any] | None:
        return get_being_store().load_browser_state(
            being_id=self.being_id,
            profile_key=profile_key,
        )


@dataclass
class BeingMind:
    """The being's cross-session cognitive affect — the part of "mind" that is
    NOT engagement-specific. Distinct from `agents/cognitive_state.py:CognitiveState`
    (which is per-engagement and Redis-TTL'd at 24h).

    Persisted write-through so a restart does not wipe the being's mood,
    curiosity drive, or what it has learned about the world on its own.
    """
    being_id: str
    # Mood / drive (0.0–1.0). Curiosity high = wants to explore; focus high =
    # wants to finish open work. These evolve with idle cycles, not reset.
    curiosity_drive: float = 0.7
    focus: float = 0.5
    satiety: float = 0.5          # "how much it has recently learned" — boredom inverse
    # Cross-session ledger of self-directed learnings (not engagement findings).
    learned_facts: list[str] = field(default_factory=list)
    # Running tallies.
    idle_cycles_run: int = 0
    goals_pursued: int = 0
    goals_completed: int = 0
    last_idle_at: str = ""
    _store: BeingStore | None = field(default=None, repr=False, compare=False)

    def _persist(self) -> None:
        store = self._store or get_being_store()
        store.save_mind(self)

    def record_idle_cycle(
        self,
        was_novel: bool = False,
        info_gain: float = 0.0,
        learned_fact: str | None = None,
    ) -> BeingMind:
        """Record one self-directed idle/curiosity cycle on this BeingMind instance."""
        self.idle_cycles_run += 1
        self.last_idle_at = _now()
        has_fact = bool(learned_fact and str(learned_fact).strip())
        if has_fact:
            fact_str = str(learned_fact).strip()
            if fact_str not in self.learned_facts:
                self.learned_facts.append(fact_str)
        if was_novel or has_fact:
            self.goals_pursued += 1
            gain_delta = max(0.0, min(0.08, float(info_gain or 0.0) * 0.05))
            self.satiety = min(1.0, self.satiety + 0.08 + gain_delta)
            self.curiosity_drive = min(1.0, self.curiosity_drive + 0.03 + gain_delta)
        else:
            self.satiety = min(1.0, self.satiety + 0.05)
            self.focus = min(1.0, self.focus + 0.04)
            self.curiosity_drive = max(0.1, self.curiosity_drive - 0.05)
        self._persist()
        return self

    def record_mission_outcome(
        self,
        goal: str,
        success: bool,
        summary: str = "",
        facts: list[str] | None = None,
    ) -> BeingMind:
        """Record an operator or autonomous mission outcome on this BeingMind."""
        self.goals_pursued += 1
        if success:
            self.goals_completed += 1
        candidate_facts: list[str] = []
        if facts:
            for item in facts:
                cleaned = str(item or "").strip()
                if cleaned:
                    candidate_facts.append(cleaned)
        elif summary and str(summary).strip():
            candidate_facts.append(str(summary).strip()[:240])
        elif goal and str(goal).strip():
            prefix = "completed" if success else "attempted"
            candidate_facts.append(f"{prefix}: {str(goal).strip()[:200]}")

        for fact_str in candidate_facts:
            if fact_str not in self.learned_facts:
                self.learned_facts.append(fact_str)

        if success:
            self.satiety = min(1.0, self.satiety + 0.08)
            self.focus = min(1.0, self.focus + 0.05)
            self.curiosity_drive = min(1.0, self.curiosity_drive + 0.04)
        else:
            self.focus = min(1.0, self.focus + 0.06)
            self.curiosity_drive = max(0.1, self.curiosity_drive - 0.03)
            self.satiety = max(0.0, self.satiety - 0.02)

        self._persist()
        return self


# ---------------------------------------------------------------------------
# Persistent store (SQLite write-through + read cache, mirrors VectorMemory)
# ---------------------------------------------------------------------------

class BeingStore:
    """SQLite-backed durable store for Being identity + BeingMind + BeingVault.

    One row per being (keyed by being_id). The in-memory caches are read-only
    mirrors hydrated on init and kept in sync on every write-through — no state
    is required across restarts.
    """

    def __init__(self, db_path: str | None = None, persist: bool = True):
        self.persist = persist
        self._db_path = db_path or (_default_db_path() if persist else None)
        self._db: sqlite3.Connection | None = None
        self._being_cache: dict[str, Being] = {}
        self._mind_cache: dict[str, BeingMind] = {}
        self._vault_cache: dict[tuple[str, str], dict[str, str]] = {}
        self._browser_state_cache: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()
        if persist:
            self._open_db()

    # -- schema + hydrate -------------------------------------------------
    def _open_db(self) -> None:
        if self._db is not None or not self.persist:
            return
        with self._lock:
            self._db = sqlite3.connect(self._db_path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=30000")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS beings ("
                "being_id TEXT PRIMARY KEY, name TEXT, tenant_id TEXT, "
                "born_at TEXT NOT NULL, last_attached_at TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_beings_tenant ON beings(tenant_id)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS being_minds ("
                "being_id TEXT PRIMARY KEY, mind_json TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS being_vault ("
                "being_id TEXT NOT NULL, service TEXT NOT NULL, username TEXT NOT NULL, "
                "secret TEXT NOT NULL, token TEXT NOT NULL, profile_url TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, PRIMARY KEY (being_id, service))"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS being_browser_state ("
                "being_id TEXT NOT NULL, profile_key TEXT NOT NULL, "
                "storage_state_json TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "PRIMARY KEY (being_id, profile_key))"
            )
            self._db.commit()
            for row in self._db.execute(
                "SELECT being_id, name, tenant_id, born_at, last_attached_at FROM beings"
            ).fetchall():
                b = Being(being_id=row[0], name=row[1], tenant_id=row[2],
                          born_at=row[3], last_attached_at=row[4])
                self._being_cache[b.being_id] = b
            for being_id, mind_json in self._db.execute(
                "SELECT being_id, mind_json FROM being_minds"
            ).fetchall():
                import json
                data = json.loads(mind_json)
                self._mind_cache[being_id] = BeingMind(
                    being_id=being_id,
                    curiosity_drive=data.get("curiosity_drive", 0.7),
                    focus=data.get("focus", 0.5),
                    satiety=data.get("satiety", 0.5),
                    learned_facts=data.get("learned_facts", []),
                    idle_cycles_run=data.get("idle_cycles_run", 0),
                    goals_pursued=data.get("goals_pursued", 0),
                    goals_completed=data.get("goals_completed", 0),
                    last_idle_at=data.get("last_idle_at", ""),
                    _store=self,
                )
            # Vault data is intentionally not hydrated into plaintext caches. Legacy rows
            # are read only by explicit migration or rejected by the retrieval methods.

    # -- Being CRUD -------------------------------------------------------
    def get_being(self, being_id: str) -> Being | None:
        with self._lock:
            b = self._being_cache.get(being_id)
            if b is not None:
                return b
            if self.persist and self._db is not None:
                row = self._db.execute(
                    "SELECT being_id, name, tenant_id, born_at, last_attached_at FROM beings WHERE being_id = ?",
                    (being_id,),
                ).fetchone()
                if row:
                    b = Being(
                        being_id=row[0],
                        name=row[1],
                        tenant_id=row[2],
                        born_at=row[3],
                        last_attached_at=row[4],
                    )
                    self._being_cache[b.being_id] = b
                    return b
            return None

    def get_being_for_tenant(self, tenant_id: str) -> Being | None:
        with self._lock:
            for b in self._being_cache.values():
                if b.tenant_id == tenant_id:
                    return b
            if self.persist and self._db is not None:
                row = self._db.execute(
                    "SELECT being_id, name, tenant_id, born_at, last_attached_at FROM beings WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
                if row:
                    b = Being(
                        being_id=row[0],
                        name=row[1],
                        tenant_id=row[2],
                        born_at=row[3],
                        last_attached_at=row[4],
                    )
                    self._being_cache[b.being_id] = b
                    return b
            return None

    def save_being(self, being: Being) -> None:
        with self._lock:
            self._being_cache[being.being_id] = being
            if not self.persist or self._db is None:
                return
            self._db.execute(
                "INSERT INTO beings (being_id, name, tenant_id, born_at, last_attached_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(being_id) DO UPDATE SET "
                "name=excluded.name, last_attached_at=excluded.last_attached_at",
                (being.being_id, being.name, being.tenant_id, being.born_at, being.last_attached_at),
            )
            self._db.commit()

    def get_or_create_being(self, tenant_id: str, name: str = "SONIC") -> tuple[Being, bool]:
        """Atomically provision one identity per tenant across SQLite connections/processes."""
        with self._lock:
            if self.persist and self._db is not None:
                now = _now()
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    candidate = Being(
                        being_id=f"being-{uuid.uuid4().hex[:12]}",
                        name=name,
                        tenant_id=tenant_id,
                        born_at=now,
                        last_attached_at=now,
                    )
                    self._db.execute(
                        "INSERT INTO beings (being_id, name, tenant_id, born_at, last_attached_at) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(tenant_id) DO NOTHING",
                        (candidate.being_id, candidate.name, tenant_id, now, now),
                    )
                    row = self._db.execute(
                        "SELECT being_id, name, tenant_id, born_at, last_attached_at "
                        "FROM beings WHERE tenant_id = ?",
                        (tenant_id,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("Atomic being provisioning produced no tenant row")
                    if row[0] == candidate.being_id:
                        being = candidate
                        created = True
                    else:
                        attached_at = _now()
                        self._db.execute(
                            "UPDATE beings SET last_attached_at = ? WHERE being_id = ?",
                            (attached_at, row[0]),
                        )
                        being = Being(
                            being_id=row[0],
                            name=row[1],
                            tenant_id=row[2],
                            born_at=row[3],
                            last_attached_at=attached_at,
                        )
                        created = False
                    self._db.commit()
                except Exception:
                    self._db.rollback()
                    raise
            else:
                existing = next(
                    (item for item in self._being_cache.values() if item.tenant_id == tenant_id),
                    None,
                )
                if existing is not None:
                    existing.last_attached_at = _now()
                    being, created = existing, False
                else:
                    now = _now()
                    being = Being(
                        being_id=f"being-{uuid.uuid4().hex[:12]}",
                        name=name,
                        tenant_id=tenant_id,
                        born_at=now,
                        last_attached_at=now,
                    )
                    created = True
            self._being_cache[being.being_id] = being
            return being, created

    def touch_attached(self, being_id: str) -> None:
        b = self.get_being(being_id)
        if b is None:
            return
        b.last_attached_at = _now()
        self.save_being(b)

    # -- BeingMind CRUD ---------------------------------------------------
    def get_or_create_mind(self, being_id: str) -> BeingMind:
        return self.get_mind(being_id)

    def get_mind(self, being_id: str) -> BeingMind:
        with self._lock:
            mind = self._mind_cache.get(being_id)
            if mind is None and self.persist and self._db is not None:
                row = self._db.execute(
                    "SELECT mind_json FROM being_minds WHERE being_id = ?",
                    (being_id,),
                ).fetchone()
                if row:
                    import json
                    data = json.loads(row[0])
                    mind = BeingMind(
                        being_id=being_id,
                        curiosity_drive=data.get("curiosity_drive", 0.7),
                        focus=data.get("focus", 0.5),
                        satiety=data.get("satiety", 0.5),
                        learned_facts=data.get("learned_facts", []),
                        idle_cycles_run=data.get("idle_cycles_run", 0),
                        goals_pursued=data.get("goals_pursued", 0),
                        goals_completed=data.get("goals_completed", 0),
                        last_idle_at=data.get("last_idle_at", ""),
                        _store=self,
                    )
                    self._mind_cache[being_id] = mind
            if mind is None:
                mind = BeingMind(being_id=being_id, _store=self)
                self._mind_cache[being_id] = mind
                self._persist_mind_unlocked(mind)
            else:
                mind._store = self
            return mind

    def save_mind(self, mind: BeingMind) -> None:
        with self._lock:
            mind._store = self
            self._mind_cache[mind.being_id] = mind
            self._persist_mind_unlocked(mind)

    def _persist_mind_unlocked(self, mind: BeingMind) -> None:
        if not self.persist or self._db is None:
            return
        import json
        payload = json.dumps({
            "curiosity_drive": mind.curiosity_drive,
            "focus": mind.focus,
            "satiety": mind.satiety,
            "learned_facts": mind.learned_facts,
            "idle_cycles_run": mind.idle_cycles_run,
            "goals_pursued": mind.goals_pursued,
            "goals_completed": mind.goals_completed,
            "last_idle_at": mind.last_idle_at,
        })
        self._db.execute(
            "INSERT INTO being_minds (being_id, mind_json) VALUES (?, ?) "
            "ON CONFLICT(being_id) DO UPDATE SET mind_json=excluded.mind_json",
            (mind.being_id, payload),
        )
        self._db.commit()

    def _persist_mind(self, mind: BeingMind) -> None:
        with self._lock:
            self._persist_mind_unlocked(mind)

    # -- BeingVault CRUD --------------------------------------------------
    def store_credential(
        self,
        being_id: str,
        service: str,
        username: str,
        secret: str = "",
        token: str = "",
        profile_url: str = "",
    ) -> dict[str, str]:
        svc_key = service.strip().lower()
        now_ts = _now()
        encrypted_secret = _encrypt_vault_value(
            secret, _vault_aad(being_id, svc_key, "secret")
        )
        encrypted_token = _encrypt_vault_value(
            token, _vault_aad(being_id, svc_key, "token")
        )
        encrypted_profile_url = _encrypt_vault_value(
            profile_url.strip(), _vault_aad(being_id, svc_key, "profile_url")
        )
        entry = {
            "being_id": being_id,
            "service": service.strip(),
            "username": username.strip(),
            "secret": secret,
            "token": token,
            "profile_url": profile_url.strip(),
            "updated_at": now_ts,
        }
        with self._lock:
            stored_entry = {
                **entry,
                "secret": encrypted_secret,
                "token": encrypted_token,
                "profile_url": encrypted_profile_url,
            }
            self._vault_cache[(being_id, svc_key)] = stored_entry
            if self.persist and self._db is not None:
                self._db.execute(
                    "INSERT INTO being_vault (being_id, service, username, secret, token, profile_url, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(being_id, service) DO UPDATE SET "
                    "username=excluded.username, secret=excluded.secret, token=excluded.token, "
                    "profile_url=excluded.profile_url, updated_at=excluded.updated_at",
                    (being_id, svc_key, entry["username"], encrypted_secret, encrypted_token,
                     encrypted_profile_url, now_ts),
                )
                self._db.commit()
        return entry

    def get_credential(self, being_id: str, service: str) -> dict[str, str] | None:
        svc_key = service.strip().lower()
        with self._lock:
            if self.persist and self._db is not None:
                row = self._db.execute(
                    "SELECT being_id, service, username, secret, token, profile_url, updated_at "
                    "FROM being_vault WHERE being_id = ? AND service = ?",
                    (being_id, svc_key),
                ).fetchone()
            else:
                cached = self._vault_cache.get((being_id, svc_key))
                row = (
                    being_id, svc_key, cached["username"], cached["secret"], cached["token"],
                    cached["profile_url"], cached["updated_at"],
                ) if cached is not None else None
            if row is None:
                return None
            return {
                "being_id": row[0],
                "service": row[1],
                "username": row[2],
                "secret": _decrypt_vault_value(row[3], _vault_aad(being_id, svc_key, "secret")),
                "token": _decrypt_vault_value(row[4], _vault_aad(being_id, svc_key, "token")),
                "profile_url": _decrypt_vault_value(
                    row[5], _vault_aad(being_id, svc_key, "profile_url")
                ),
                "updated_at": row[6],
            }

    def list_accounts(self, being_id: str) -> list[dict[str, str]]:
        with self._lock:
            if self.persist and self._db is not None:
                rows = self._db.execute(
                    "SELECT service, username, profile_url, updated_at FROM being_vault "
                    "WHERE being_id = ?",
                    (being_id,),
                ).fetchall()
                return [
                    {
                        "service": row[0],
                        "username": row[1],
                        "profile_url": _decrypt_vault_value(
                            row[2], _vault_aad(being_id, row[0].lower(), "profile_url")
                        ),
                        "updated_at": row[3],
                    }
                    for row in rows
                ]
            results = [
                {
                    "service": v["service"],
                    "username": v["username"],
                    "profile_url": _decrypt_vault_value(
                        v.get("profile_url", ""),
                        _vault_aad(being_id, v["service"].lower(), "profile_url"),
                    ),
                    "updated_at": v.get("updated_at", ""),
                }
                for (b_id, _), v in self._vault_cache.items()
                if b_id == being_id
            ]
            return results

    def save_browser_state(
        self,
        being_id: str,
        storage_state: dict[str, Any] | str,
        profile_key: str = "default",
    ) -> dict[str, Any]:
        import json
        pkey = (profile_key or "default").strip().lower()
        if isinstance(storage_state, str):
            try:
                parsed = json.loads(storage_state)
                state_dict = parsed if isinstance(parsed, dict) else {"raw": storage_state}
            except Exception:
                state_dict = {"raw": storage_state}
        else:
            state_dict = dict(storage_state or {})
        now_ts = _now()
        payload = _encrypt_vault_value(
            json.dumps(state_dict),
            _vault_aad(being_id, pkey, "browser_state"),
        )
        with self._lock:
            self._browser_state_cache[(being_id, pkey)] = payload
            if self.persist and self._db is not None:
                self._db.execute(
                    "INSERT INTO being_browser_state (being_id, profile_key, storage_state_json, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(being_id, profile_key) DO UPDATE SET "
                    "storage_state_json=excluded.storage_state_json, updated_at=excluded.updated_at",
                    (being_id, pkey, payload, now_ts),
                )
                self._db.commit()
        return dict(state_dict)

    def load_browser_state(
        self,
        being_id: str,
        profile_key: str = "default",
    ) -> dict[str, Any] | None:
        pkey = (profile_key or "default").strip().lower()
        with self._lock:
            if self.persist and self._db is not None:
                row = self._db.execute(
                    "SELECT storage_state_json FROM being_browser_state WHERE being_id = ? AND profile_key = ?",
                    (being_id, pkey),
                ).fetchone()
                payload = row[0] if row else None
            else:
                payload = self._browser_state_cache.get((being_id, pkey))
            if payload is None:
                return None
            decoded = _decrypt_vault_value(payload, _vault_aad(being_id, pkey, "browser_state"))
            try:
                state_dict = json.loads(decoded)
            except Exception as exc:
                raise VaultDataError("Browser-state vault payload is not valid JSON") from exc
            if not isinstance(state_dict, dict):
                raise VaultDataError("Browser-state vault payload must be a JSON object")
            return state_dict

    def migrate_legacy_vault_data(self) -> dict[str, int]:
        """Explicitly encrypt unmarked legacy plaintext vault rows in-place.

        This is never run automatically: operators should back up the database,
        configure SONIC_BEING_VAULT_KEY, then invoke this migration deliberately.
        """
        _vault_cipher()  # Fail before opening a transaction if key management is incomplete.
        if not self.persist or self._db is None:
            raise RuntimeError("Legacy vault migration requires a persistent BeingStore")
        credentials = 0
        browser_states = 0
        with self._lock:
            self._db.execute("PRAGMA secure_delete=ON")
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for being_id, service, secret, token, profile_url in self._db.execute(
                    "SELECT being_id, service, secret, token, profile_url FROM being_vault"
                ).fetchall():
                    svc_key = service.strip().lower()
                    encrypted_secret = secret if secret.startswith(_VAULT_CIPHERTEXT_PREFIX) else (
                        _encrypt_vault_value(secret, _vault_aad(being_id, svc_key, "secret"))
                    )
                    encrypted_token = token if token.startswith(_VAULT_CIPHERTEXT_PREFIX) else (
                        _encrypt_vault_value(token, _vault_aad(being_id, svc_key, "token"))
                    )
                    encrypted_profile_url = (
                        profile_url if profile_url.startswith(_VAULT_CIPHERTEXT_PREFIX) else
                        _encrypt_vault_value(
                            profile_url, _vault_aad(being_id, svc_key, "profile_url")
                        )
                    )
                    if (
                        encrypted_secret != secret
                        or encrypted_token != token
                        or encrypted_profile_url != profile_url
                    ):
                        self._db.execute(
                            "UPDATE being_vault SET secret = ?, token = ?, profile_url = ? "
                            "WHERE being_id = ? AND service = ?",
                            (encrypted_secret, encrypted_token, encrypted_profile_url,
                             being_id, service),
                        )
                        credentials += 1
                for being_id, profile_key, state in self._db.execute(
                    "SELECT being_id, profile_key, storage_state_json FROM being_browser_state"
                ).fetchall():
                    if not state.startswith(_VAULT_CIPHERTEXT_PREFIX):
                        encrypted_state = _encrypt_vault_value(
                            state,
                            _vault_aad(being_id, profile_key.strip().lower(), "browser_state"),
                        )
                        self._db.execute(
                            "UPDATE being_browser_state SET storage_state_json = ? "
                            "WHERE being_id = ? AND profile_key = ?",
                            (encrypted_state, being_id, profile_key),
                        )
                        browser_states += 1
                self._db.commit()
                checkpoint = self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint and (checkpoint[0] != 0 or checkpoint[1] != checkpoint[2]):
                    raise VaultDataError(
                        "Legacy values were encrypted, but SQLite could not truncate the "
                        "plaintext WAL; stop other database users and retry checkpointing"
                    )
            except Exception:
                if self._db.in_transaction:
                    self._db.rollback()
                raise
        return {"credentials": credentials, "browser_states": browser_states}

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


# ---------------------------------------------------------------------------
# Singleton (mirrors get_vector_memory / reset_vector_memory_singleton)
# ---------------------------------------------------------------------------

_being_store: BeingStore | None = None


def get_being_store() -> BeingStore:
    global _being_store
    if _being_store is None:
        _being_store = BeingStore()
    return _being_store


def reset_being_singleton() -> None:
    """Clear the cached BeingStore singleton (used by tests to simulate restart)."""
    global _being_store
    if _being_store is not None:
        _being_store.close()
    _being_store = None


def get_or_create_being(tenant_id: str, name: str = "SONIC") -> Being:
    """Boot re-attach: return the persisted being for a tenant, or provision one.

    This is the identity equivalent of `get_or_create_home(tenant_id)` for the
    Body: on restart the being's identity is RE-RESOLVED from SQLite (same
    being_id, same born_at), not reminted. First contact provisions one.
    """
    store = get_being_store()
    being, created = store.get_or_create_being(tenant_id, name=name)
    if created:
        store.get_mind(being.being_id)  # initialize mind row
        logger.info("being_provisioned", being_id=being.being_id, tenant_id=tenant_id)
    else:
        logger.info("being_reattached", being_id=being.being_id,
                    born_at=being.born_at, tenant_id=tenant_id)
    return being


def get_or_create_mind(tenant_id: str = "default") -> BeingMind:
    """Return the persisted BeingMind for a tenant (or being_id), provisioning if needed."""
    store = get_being_store()
    if tenant_id.startswith("being-") and store.get_being(tenant_id) is not None:
        return store.get_or_create_mind(tenant_id)
    being = get_or_create_being(tenant_id)
    return store.get_or_create_mind(being.being_id)


# ---------------------------------------------------------------------------
# Mind evolution primitives (used by the idle tick)
# ---------------------------------------------------------------------------

def record_idle_cycle(
    being_id: str,
    learned: str | None = None,
    *,
    was_novel: bool | None = None,
    info_gain: float = 0.0,
) -> BeingMind:
    """Record that the being completed one self-directed idle/curiosity cycle.

    Evolves mood deterministically from the outcome: a genuinely new fact raises
    curiosity drive + satiety (reward learning); a dead-end (no novelty) raises
    satiety toward boredom and nudges focus up (time to finish open work). This
    is a small, real affect model — not a claim of emotion.
    """
    store = get_being_store()
    mind = store.get_mind(being_id)
    novel_flag = bool(was_novel) if was_novel is not None else bool(learned and str(learned).strip())
    mind.record_idle_cycle(
        was_novel=novel_flag,
        info_gain=info_gain,
        learned_fact=learned,
    )
    store.save_mind(mind)
    return mind
