"""
Unit and Integration Tests for User Authentication and Per-User Data Isolation.

Covers:
1. Registration: success, duplicate username (409), weak password (422),
   invalid username characters (422).
2. Login: valid credentials return a JWT, wrong password (401), unknown
   user (401).
3. Token handling: /auth/me with valid token, missing token (401), malformed
   token (401), and 401 on protected endpoints without a token.
4. Per-user data isolation: notes and article history are visible only to
   their owner; cross-user read/update/delete is reported as 404.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_db
from database.notes_history import NotesHistoryManager


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_auth.db"
    manager = NotesHistoryManager(db_path=db_file)
    yield manager
    manager.close()


@pytest.fixture
def client(test_db):
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def register(client, username, password):
    return client.post(
        "/api/v1/auth/register", json={"username": username, "password": password}
    )


def login(client, username, password):
    return client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )


def authed(client, token):
    """Copy of the client whose default headers carry the given bearer token."""
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


# =============================================================================
# REGISTRATION
# =============================================================================

def test_register_success(client):
    res = register(client, "alice", "password123")
    assert res.status_code == 201
    body = res.json()
    assert body["username"] == "alice"
    assert isinstance(body["id"], int)
    assert "password" not in body and "password_hash" not in body


def test_register_duplicate_username_rejected(client):
    assert register(client, "alice", "password123").status_code == 201
    res = register(client, "Alice", "different123")
    assert res.status_code == 409


def test_register_short_password_rejected(client):
    res = register(client, "bob", "short")
    assert res.status_code == 422


def test_register_invalid_username_chars_rejected(client):
    res = register(client, "bad user!", "password123")
    assert res.status_code == 422


# =============================================================================
# LOGIN
# =============================================================================

def test_login_success_returns_token(client):
    register(client, "carol", "password123")
    res = login(client, "carol", "password123")
    assert res.status_code == 200
    body = res.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["username"] == "carol"


def test_login_wrong_password(client):
    register(client, "dave", "password123")
    assert login(client, "dave", "wrongpassword").status_code == 401


def test_login_unknown_user(client):
    assert login(client, "ghost", "password123").status_code == 401


# =============================================================================
# TOKEN HANDLING & PROTECTED ENDPOINTS
# =============================================================================

def test_me_with_valid_token(client):
    register(client, "erin", "password123")
    token = login(client, "erin", "password123").json()["access_token"]
    res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["username"] == "erin"


def test_me_without_token(client):
    assert client.get("/api/v1/auth/me").status_code == 401


def test_malformed_token_rejected(client):
    res = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert res.status_code == 401


def test_protected_endpoints_require_token(client):
    # Clear any default header contamination; this fixture's client has none.
    for method, path, payload in [
        ("post", "/api/v1/summarize", {"text": "A sentence here."}),
        ("post", "/api/v1/vocabulary/extract", {"text": "A sentence here."}),
        ("post", "/api/v1/evaluate", {"candidate_summary": "a", "reference_summary": "b"}),
        ("get", "/api/v1/notes", None),
        ("get", "/api/v1/history", None),
        ("get", "/api/v1/speech/voices", None),
    ]:
        if method == "post":
            res = client.post(path, json=payload)
        else:
            res = client.get(path)
        assert res.status_code == 401, f"{method.upper()} {path} returned {res.status_code}"


def test_health_and_root_stay_public(client):
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


# =============================================================================
# PER-USER DATA ISOLATION
# =============================================================================

def make_user(client, username):
    register(client, username, "password123")
    token = login(client, username, "password123").json()["access_token"]
    return authed(client, token)


def reset_headers(client):
    client.headers.pop("Authorization", None)


def test_notes_isolated_between_users(client):
    alice = make_user(client, "alice")
    created = alice.post("/api/v1/notes", json={"content": "Alice's private note", "term": "satellite"})
    assert created.status_code == 201
    note_id = created.json()["id"]

    reset_headers(client)
    bob = make_user(client, "bob")
    listed = bob.get("/api/v1/notes")
    assert listed.status_code == 200
    assert listed.json()["total"] == 0
    assert listed.json()["items"] == []

    assert bob.get(f"/api/v1/notes/{note_id}").status_code == 404
    upd = bob.put(f"/api/v1/notes/{note_id}", json={"content": "hijacked"})
    assert upd.status_code == 404
    assert bob.delete(f"/api/v1/notes/{note_id}").status_code == 404

    # Alice still sees her note untouched.
    reset_headers(client)
    alice = make_user(client, "alice")
    got = alice.get(f"/api/v1/notes/{note_id}")
    assert got.status_code == 200
    assert got.json()["content"] == "Alice's private note"


def test_history_isolated_between_users(client):
    alice = make_user(client, "alice")
    created = alice.post(
        "/api/v1/history",
        json={"summary": "Alice summary.", "method": "textrank", "title": "Alice article"},
    )
    assert created.status_code == 201
    history_id = created.json()["id"]

    reset_headers(client)
    bob = make_user(client, "bob")
    listed = bob.get("/api/v1/history")
    assert listed.status_code == 200
    assert listed.json()["total"] == 0

    assert bob.get(f"/api/v1/history/{history_id}").status_code == 404
    assert bob.delete(f"/api/v1/history/{history_id}").status_code == 404

    reset_headers(client)
    alice = make_user(client, "alice")
    assert alice.get(f"/api/v1/history/{history_id}").status_code == 200


def test_note_cannot_link_to_other_users_history(client):
    alice = make_user(client, "alice")
    history = alice.post(
        "/api/v1/history", json={"summary": "Alice summary.", "method": "tfidf"}
    )
    history_id = history.json()["id"]

    reset_headers(client)
    bob = make_user(client, "bob")
    res = bob.post(
        "/api/v1/notes",
        json={"content": "Linking to Alice's history", "article_history_id": history_id},
    )
    assert res.status_code == 400


def test_search_respects_user_isolation(client):
    alice = make_user(client, "alice")
    alice.post("/api/v1/history", json={"summary": "quantum computing advances", "method": "textrank"})

    reset_headers(client)
    bob = make_user(client, "bob")
    res = bob.get("/api/v1/history/search", params={"q": "quantum"})
    assert res.status_code == 200
    assert res.json()["total"] == 0
