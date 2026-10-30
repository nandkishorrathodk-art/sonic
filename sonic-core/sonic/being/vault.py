"""
SONIC — Persistent BeingVault (Credentials, Accounts & Browser Session State)
=============================================================================

Provides a clean `BeingVault` facade wrapping `BeingStore` so the AI-Human
Digital Being can persist and reload platform accounts, credentials, tokens,
and browser `storage_state` profiles across restarts.
"""

from __future__ import annotations

from typing import Any

from sonic.being.identity import Being, BeingStore, get_being_store


class BeingVault:
    """Durable credential and browser-session vault for a persistent Being."""

    def __init__(
        self,
        being_id: str | Being = "default",
        store: BeingStore | None = None,
        db_path: str | None = None,
    ) -> None:
        if isinstance(being_id, Being):
            self.being_id = being_id.being_id
        else:
            self.being_id = str(being_id or "default").strip() or "default"
        if store is not None:
            self._store = store
        elif db_path is not None:
            self._store = BeingStore(db_path=db_path, persist=True)
        else:
            self._store = None

    @property
    def store(self) -> BeingStore:
        return self._store if self._store is not None else get_being_store()

    def _resolve_being_id(self, being_id: str | None = None) -> str:
        return (str(being_id).strip() if being_id else self.being_id) or "default"

    def store_credential(
        self,
        service: str,
        username: str = "",
        secret: str = "",
        token: str = "",
        profile_url: str = "",
        *,
        being_id: str | None = None,
    ) -> dict[str, str]:
        """Persist a service credential/account for the being."""
        return self.store.store_credential(
            being_id=self._resolve_being_id(being_id),
            service=service,
            username=username,
            secret=secret,
            token=token,
            profile_url=profile_url,
        )

    def get_credential(
        self,
        service: str,
        *,
        being_id: str | None = None,
    ) -> dict[str, str] | None:
        """Retrieve a persisted service credential for the being."""
        return self.store.get_credential(
            being_id=self._resolve_being_id(being_id),
            service=service,
        )

    def list_accounts(self, *, being_id: str | None = None) -> list[dict[str, str]]:
        """List all accounts/services stored in the vault for the being."""
        return self.store.list_accounts(self._resolve_being_id(being_id))

    def save_browser_state(
        self,
        storage_state_or_being_id: dict[str, Any] | str,
        profile_key_or_state: dict[str, Any] | str = "default",
        profile_key: str = "default",
        *,
        being_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist browser storage_state (cookies/localStorage) for a profile."""
        if isinstance(storage_state_or_being_id, dict):
            b_id = self._resolve_being_id(being_id)
            state: dict[str, Any] | str = storage_state_or_being_id
            if profile_key != "default":
                pkey = profile_key
            elif isinstance(profile_key_or_state, str):
                pkey = profile_key_or_state
            else:
                pkey = "default"
        elif isinstance(profile_key_or_state, dict):
            b_id = str(storage_state_or_being_id).strip() or self._resolve_being_id(being_id)
            state = profile_key_or_state
            pkey = profile_key
        else:
            b_id = self._resolve_being_id(being_id)
            state = storage_state_or_being_id
            pkey = profile_key if profile_key != "default" else str(profile_key_or_state or "default")

        return self.store.save_browser_state(
            being_id=b_id,
            storage_state=state,
            profile_key=pkey,
        )

    def load_browser_state(
        self,
        profile_key: str = "default",
        *,
        being_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Load persisted browser storage_state for a profile."""
        b_id = self._resolve_being_id(being_id)
        res = self.store.load_browser_state(being_id=b_id, profile_key=profile_key)
        if res is None and being_id is None and profile_key != "default":
            # Support callers passing load_browser_state(being_id) with default profile_key
            alt = self.store.load_browser_state(being_id=profile_key, profile_key="default")
            if alt is not None:
                return alt
        return res


__all__ = ["BeingVault"]
