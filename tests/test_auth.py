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


class TestRegistration:
    def test_register_success(self):
        res = client.post("/api/register", json={"name": "Test", "email": "test@example.com", "password": "Password1"})
        assert res.status_code == 201
        assert "access_token" in res.json()

    def test_register_duplicate_email(self):
        client.post("/api/register", json={"name": "A", "email": "dup@example.com", "password": "Password1"})
        res = client.post("/api/register", json={"name": "B", "email": "dup@example.com", "password": "Password1"})
        assert res.status_code == 409

    def test_register_weak_password(self):
        res = client.post("/api/register", json={"name": "Test", "email": "weak@example.com", "password": "short"})
        assert res.status_code == 422


class TestLogin:
    def test_login_success(self):
        client.post("/api/register", json={"name": "Test", "email": "login@example.com", "password": "Password1"})
        res = client.post("/api/login", json={"email": "login@example.com", "password": "Password1"})
        assert res.status_code == 200
        assert "access_token" in res.json()

    def test_login_invalid_credentials(self):
        res = client.post("/api/login", json={"email": "noone@example.com", "password": "Wrong123"})
        assert res.status_code == 401
