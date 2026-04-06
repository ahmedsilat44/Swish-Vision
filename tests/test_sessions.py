import pytest


def get_auth_token(client):
    client.post("/api/register", json={"name": "Test", "email": "sess@example.com", "password": "Password1"})
    res = client.post("/api/login", json={"email": "sess@example.com", "password": "Password1"})
    return res.json()["access_token"]


class TestSessions:
    def test_list_sessions_unauthenticated(self, client):
        res = client.get("/api/sessions/")
        assert res.status_code == 403

    def test_list_sessions_empty(self, client):
        token = get_auth_token(client)
        res = client.get("/api/sessions/", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json() == []

    def test_get_nonexistent_session(self, client):
        token = get_auth_token(client)
        res = client.get("/api/sessions/9999", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 404
