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

import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from typing import Any

from sonic.logger import get_logger

logger = get_logger(__name__)


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
    last_idle_at: str = ""

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
        if _being_store is not None:
            try:
                _being_store.save_mind(self)
            except Exception:
                pass
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
        self._browser_state_cache: dict[tuple[str, str], dict[str, Any]] = {}
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
                    last_idle_at=data.get("last_idle_at", ""),
                )
            for row in self._db.execute(
                "SELECT being_id, service, username, secret, token, profile_url, updated_at FROM being_vault"
            ).fetchall():
                self._vault_cache[(row[0], row[1].lower())] = {
                    "being_id": row[0],
                    "service": row[1],
                    "username": row[2],
                    "secret": row[3],
                    "token": row[4],
                    "profile_url": row[5],
                    "updated_at": row[6],
                }
            for b_id, p_key, state_json, _ in self._db.execute(
                "SELECT being_id, profile_key, storage_state_json, updated_at FROM being_browser_state"
            ).fetchall():
                try:
                    import json
                    parsed_state = json.loads(state_json)
                    if isinstance(parsed_state, dict):
                        self._browser_state_cache[(b_id, p_key.lower())] = parsed_state
                except Exception:
                    pass

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

    def touch_attached(self, being_id: str) -> None:
        b = self.get_being(being_id)
        if b is None:
            return
        b.last_attached_at = _now()
        self.save_being(b)

    # -- BeingMind CRUD ---------------------------------------------------
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
                        last_idle_at=data.get("last_idle_at", ""),
                    )
                    self._mind_cache[being_id] = mind
            if mind is None:
                mind = BeingMind(being_id=being_id)
                self._mind_cache[being_id] = mind
                self._persist_mind_unlocked(mind)
            return mind

    def save_mind(self, mind: BeingMind) -> None:
        with self._lock:
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
            self._vault_cache[(being_id, svc_key)] = entry
            if self.persist and self._db is not None:
                self._db.execute(
                    "INSERT INTO being_vault (being_id, service, username, secret, token, profile_url, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(being_id, service) DO UPDATE SET "
                    "username=excluded.username, secret=excluded.secret, token=excluded.token, "
                    "profile_url=excluded.profile_url, updated_at=excluded.updated_at",
                    (being_id, svc_key, entry["username"], secret, token, entry["profile_url"], now_ts),
                )
                self._db.commit()
        return entry

    def get_credential(self, being_id: str, service: str) -> dict[str, str] | None:
        svc_key = service.strip().lower()
        with self._lock:
            cached = self._vault_cache.get((being_id, svc_key))
            if cached is not None:
                return dict(cached)
            if self.persist and self._db is not None:
                row = self._db.execute(
                    "SELECT being_id, service, username, secret, token, profile_url, updated_at "
                    "FROM being_vault WHERE being_id = ? AND service = ?",
                    (being_id, svc_key),
                ).fetchone()
                if row:
                    entry = {
                        "being_id": row[0],
                        "service": row[1],
                        "username": row[2],
                        "secret": row[3],
                        "token": row[4],
                        "profile_url": row[5],
                        "updated_at": row[6],
                    }
                    self._vault_cache[(being_id, svc_key)] = entry
                    return dict(entry)
            return None

    def list_accounts(self, being_id: str) -> list[dict[str, str]]:
        with self._lock:
            results = [
                {
                    "service": v["service"],
                    "username": v["username"],
                    "profile_url": v.get("profile_url", ""),
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
        payload = json.dumps(state_dict)
        with self._lock:
            self._browser_state_cache[(being_id, pkey)] = state_dict
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
        import json
        pkey = (profile_key or "default").strip().lower()
        with self._lock:
            cached = self._browser_state_cache.get((being_id, pkey))
            if cached is not None:
                return dict(cached)
            if self.persist and self._db is not None:
                row = self._db.execute(
                    "SELECT storage_state_json FROM being_browser_state WHERE being_id = ? AND profile_key = ?",
                    (being_id, pkey),
                ).fetchone()
                if row:
                    try:
                        state_dict = json.loads(row[0])
                        if isinstance(state_dict, dict):
                            self._browser_state_cache[(being_id, pkey)] = state_dict
                            return dict(state_dict)
                    except Exception:
                        pass
            return None

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
    being = store.get_being_for_tenant(tenant_id)
    if being is None:
        being = Being(
            being_id=f"being-{uuid.uuid4().hex[:12]}",
            name=name,
            tenant_id=tenant_id,
            born_at=_now(),
            last_attached_at=_now(),
        )
        store.save_being(being)
        store.get_mind(being.being_id)  # initialize mind row
        logger.info("being_provisioned", being_id=being.being_id, tenant_id=tenant_id)
    else:
        store.touch_attached(being.being_id)
        logger.info("being_reattached", being_id=being.being_id,
                    born_at=being.born_at, tenant_id=tenant_id)
    return being


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
