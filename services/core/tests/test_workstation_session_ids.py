"""Server-minted workstation session ids.

Sessions used to be minted by the browser as ``mission-<8 hex>``, which the
backend neither produced nor validated. Two problems followed:

* the id could not be trusted to be URL- or shell-safe (a ``#`` in a query
  string silently truncates at the fragment delimiter);
* there was no way to create a session server-side at all, so the client owned
  naming while the server owned storage.

These tests pin the contract:

* ``POST /workstation/session`` mints ``ws-<16 hex>`` and persists it;
* an invalid id is rejected (400) rather than stored;
* a colliding id is rejected (409) rather than silently returning someone
  else's session;
* the display handle is ``workstation#<token>`` while the stored id stays
  ``ws-<token>`` -- the ``#`` never enters the id;
* ``default`` stays reserved and existing (legacy) ids keep working.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

WS_ID_RE = re.compile(r"^ws-[0-9a-f]{16}$")


@pytest.fixture
def client():
    from sonic.api.main import app

    return TestClient(app)


def _login(client: TestClient, tenant_id: str = "default") -> str:
    res = client.post(
        "/auth/login",
        json={
            "email": f"u@{tenant_id}.com",
            "name": "U",
            "role": "operator",
            "tenant_id": tenant_id,
        },
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ==========================================================
# Minting
# ==========================================================


def test_create_session_mints_a_safe_id_and_persists_it(client):
    token = _login(client)
    res = client.post("/workstation/session", headers=_auth(token))
    assert res.status_code == 200, res.text
    body = res.json()

    session_id = body["session_id"]
    assert WS_ID_RE.match(session_id), f"unexpected id shape: {session_id}"
    # The id itself must never carry the display delimiter.
    assert "#" not in session_id
    assert body["display_name"] == f"workstation#{session_id[len('ws-'):]}"
    assert body["status"] == "IDLE"

    # It must be reachable as a real session.
    listed = client.get("/workstation/sessions", headers=_auth(token)).json()
    assert session_id in {s["session_id"] for s in listed}


def test_two_creates_produce_distinct_sessions(client):
    token = _login(client)
    first = client.post("/workstation/session", headers=_auth(token)).json()["session_id"]
    second = client.post("/workstation/session", headers=_auth(token)).json()["session_id"]
    assert first != second, "each create must mint a fresh id"


def test_new_session_state_is_initialised(client):
    token = _login(client)
    session_id = client.post("/workstation/session", headers=_auth(token)).json()["session_id"]
    state = client.get(
        f"/workstation/state?session_id={session_id}", headers=_auth(token)
    ).json()
    assert state["session_id"] == session_id
    # A brand-new session must not inherit another session's desktop.
    assert state["desktop"]["workspace_id"] == ""


# ==========================================================
# Validation
# ==========================================================


@pytest.mark.parametrize(
    "bad_id",
    [
        "workstation#abc",   # URL fragment delimiter
        "ws-abc;rm -rf /",   # shell metacharacter
        "../escape",         # path traversal
        "under_score",
        "sp ace",
        "a" * 65,            # too long
    ],
)
def test_invalid_ids_are_rejected_not_stored(client, bad_id):
    token = _login(client)
    res = client.post(
        "/workstation/session", json={"session_id": bad_id}, headers=_auth(token)
    )
    assert res.status_code == 400, f"{bad_id!r} should be rejected: {res.text}"

    listed = client.get("/workstation/sessions", headers=_auth(token)).json()
    assert bad_id not in {s["session_id"] for s in listed}


def test_invalid_session_id_is_rejected_on_read_routes(client):
    """Validation also guards ids arriving on ordinary query routes."""
    token = _login(client)
    res = client.get(
        "/workstation/state?session_id=workstation%23abc", headers=_auth(token)
    )
    assert res.status_code == 400


def test_default_is_reserved(client):
    token = _login(client)
    res = client.post(
        "/workstation/session", json={"session_id": "default"}, headers=_auth(token)
    )
    assert res.status_code == 400
    assert "reserved" in res.json()["detail"].lower()


def test_colliding_id_is_rejected_not_reused(client):
    token = _login(client)
    session_id = client.post("/workstation/session", headers=_auth(token)).json()["session_id"]

    again = client.post(
        "/workstation/session", json={"session_id": session_id}, headers=_auth(token)
    )
    assert again.status_code == 409, "a duplicate id must not silently resolve to an existing session"


def test_legacy_session_ids_still_resolve(client):
    """Ids minted before this change (e.g. mission-xxxxxxxx) keep working."""
    token = _login(client)
    legacy = "mission-abc12345"
    state = client.get(
        f"/workstation/state?session_id={legacy}", headers=_auth(token)
    )
    assert state.status_code == 200, state.text
    assert state.json()["session_id"] == legacy
    # Legacy ids keep their historical label rather than a ws- handle.
    listed = client.get("/workstation/sessions", headers=_auth(token)).json()
    entry = next(s for s in listed if s["session_id"] == legacy)
    assert entry["session_name"] == f"Session {legacy[-8:]}"


# ==========================================================
# Auth
# ==========================================================


def test_create_requires_auth(client):
    assert client.post("/workstation/session").status_code == 401


def test_created_session_is_tenant_scoped(client):
    token_a = _login(client, tenant_id="tenant-a")
    token_b = _login(client, tenant_id="tenant-b")
    session_id = client.post("/workstation/session", headers=_auth(token_a)).json()["session_id"]

    # Tenant B must not see or reach tenant A's new session.
    listed_b = client.get("/workstation/sessions", headers=_auth(token_b)).json()
    assert session_id not in {s["session_id"] for s in listed_b}

    state_b = client.get(
        f"/workstation/state?session_id={session_id}", headers=_auth(token_b)
    )
    # It resolves only as tenant B's own (empty) session, never tenant A's data.
    assert state_b.status_code == 200
    assert state_b.json()["desktop"]["workspace_id"] == ""


# ==========================================================
# Display handle helpers
# ==========================================================


def test_display_name_helper_never_leaks_the_hash_into_the_id():
    from sonic.api.routes.workstation import _session_display_name

    assert _session_display_name("ws-0123456789abcdef") == "workstation#0123456789abcdef"
    assert _session_display_name("default") == "New conversation"
    # Legacy ids keep their old label.
    assert _session_display_name("mission-abc12345") == "Session abc12345"


def test_validator_accepts_existing_ids_and_rejects_unsafe_ones():
    from sonic.api.routes.workstation import _is_valid_session_id, _new_session_id

    # Every id shape already in use must keep working.
    assert _is_valid_session_id("default")
    assert _is_valid_session_id("mission-abc12345")
    assert _is_valid_session_id("desktop-vision-True")   # mixed case is safe
    assert _is_valid_session_id(_new_session_id())
    # Unsafe shapes are rejected.
    assert not _is_valid_session_id("workstation#abc")   # URL fragment
    assert not _is_valid_session_id("")
    assert not _is_valid_session_id("a b")
    assert not _is_valid_session_id("ws-abc;id")         # shell metacharacter
    assert not _is_valid_session_id("../../etc/passwd")  # path traversal
