import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def get_auth_token():
    client.post("/api/register", json={"name": "Test", "email": "sess@example.com", "password": "Password1"})
    res = client.post("/api/login", json={"email": "sess@example.com", "password": "Password1"})
    return res.json()["access_token"]


class TestSessions:
    def test_list_sessions_unauthenticated(self):
        res = client.get("/api/sessions/")
        assert res.status_code == 403

    def test_list_sessions_empty(self):
        token = get_auth_token()
        res = client.get("/api/sessions/", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json() == []

    def test_get_nonexistent_session(self):
        token = get_auth_token()
        res = client.get("/api/sessions/9999", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 404
