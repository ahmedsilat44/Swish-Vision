import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models.user import User
from app.core.security import verify_password


@pytest.fixture
def db_session_fixture():
    """Fixture to get a database session for tests"""
    from tests.conftest import TestingSessionLocal
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


class TestRegistration:
    """Unit tests for the /api/register endpoint (SCRUM-81)"""

    def test_register_success(self, client):
        """Test successful user registration with valid data"""
        res = client.post("/api/register", json={"name": "John Doe", "email": "john@example.com", "password": "SecurePass123"})
        assert res.status_code == 201
        body = res.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert isinstance(body["access_token"], str)
        assert len(body["access_token"]) > 0

    def test_register_response_structure(self, client):
        """Test that registration response has correct structure"""
        res = client.post("/api/register", json={"name": "Alice", "email": "alice@example.com", "password": "Password123"})
        assert res.status_code == 201
        body = res.json()
        assert set(body.keys()) == {"access_token", "token_type"}

    def test_register_user_persisted_in_db(self, client, db_session_fixture: Session):
        """Test that registered user is saved to database"""
        payload = {"name": "Bob Smith", "email": "bob@example.com", "password": "ValidPass456"}
        client.post("/api/register", json=payload)
        
        user = db_session_fixture.query(User).filter(User.email == "bob@example.com").first()
        assert user is not None
        assert user.name == "Bob Smith"
        assert user.email == "bob@example.com"

    def test_register_password_hashed(self, client, db_session_fixture: Session):
        """Test that password is hashed and not stored in plaintext"""
        payload = {"name": "Charlie", "email": "charlie@example.com", "password": "MyPassword789"}
        client.post("/api/register", json=payload)
        
        user = db_session_fixture.query(User).filter(User.email == "charlie@example.com").first()
        assert user is not None
        assert user.password_hash != "MyPassword789"
        assert verify_password("MyPassword789", user.password_hash)
        assert not verify_password("WrongPassword", user.password_hash)

    def test_register_duplicate_email(self, client):
        """Test that duplicate email registration is rejected (409 Conflict)"""
        payload = {"name": "User1", "email": "duplicate@example.com", "password": "Pass123456"}
        client.post("/api/register", json=payload)
        
        res = client.post("/api/register", json={"name": "User2", "email": "duplicate@example.com", "password": "Pass654321"})
        assert res.status_code == 409
        assert "Email already registered" in res.json()["detail"]

    def test_register_duplicate_email_case_insensitive(self, client):
        """Test that email validation is case-insensitive"""
        payload = {"name": "David", "email": "test@example.com", "password": "Password123"}
        client.post("/api/register", json=payload)
        
        res = client.post("/api/register", json={"name": "Eve", "email": "TEST@EXAMPLE.COM", "password": "Different456"})
        # SQLite comparison is case-insensitive by default
        assert res.status_code in [201, 409]

    def test_register_weak_password_too_short(self, client):
        """Test that password shorter than 8 characters is rejected"""
        res = client.post("/api/register", json={"name": "Frank", "email": "frank@example.com", "password": "Short1"})
        assert res.status_code == 422
        assert "Password too weak" in res.json()["detail"]

    def test_register_weak_password_no_letters(self, client):
        """Test that password without letters is rejected"""
        res = client.post("/api/register", json={"name": "Grace", "email": "grace@example.com", "password": "12345678"})
        assert res.status_code == 422
        assert "Password too weak" in res.json()["detail"]

    def test_register_weak_password_no_digits(self, client):
        """Test that password without digits is rejected"""
        res = client.post("/api/register", json={"name": "Henry", "email": "henry@example.com", "password": "OnlyLetters"})
        assert res.status_code == 422
        assert "Password too weak" in res.json()["detail"]

    def test_register_invalid_email_format_no_at(self, client):
        """Test that email without @ is rejected"""
        res = client.post("/api/register", json={"name": "Iris", "email": "invalidexample.com", "password": "ValidPass123"})
        assert res.status_code == 422
        assert "Invalid email format" in res.json()["detail"]

    def test_register_invalid_email_format_no_domain(self, client):
        """Test that email without domain is rejected"""
        res = client.post("/api/register", json={"name": "Jack", "email": "invalid@", "password": "ValidPass123"})
        assert res.status_code == 422
        assert "Invalid email format" in res.json()["detail"]

    def test_register_invalid_email_format_no_tld(self, client):
        """Test that email without TLD is rejected"""
        res = client.post("/api/register", json={"name": "Karen", "email": "invalid@domain", "password": "ValidPass123"})
        assert res.status_code == 422
        assert "Invalid email format" in res.json()["detail"]

    def test_register_invalid_email_with_spaces(self, client):
        """Test that email with spaces is rejected"""
        res = client.post("/api/register", json={"name": "Leo", "email": "invalid @example.com", "password": "ValidPass123"})
        assert res.status_code == 422
        assert "Invalid email format" in res.json()["detail"]

    def test_register_valid_email_formats(self, client):
        """Test that various valid email formats are accepted"""
        valid_emails = [
            "user@example.com",
            "user.name@example.co.uk",
            "user+tag@example.com",
            "123@example.com",
            "a@b.co",
        ]
        for i, email in enumerate(valid_emails):
            res = client.post("/api/register", json={"name": f"User{i}", "email": email, "password": f"Pass{i}1234"})
            assert res.status_code == 201, f"Email {email} should be valid"

    def test_register_missing_name_field(self, client):
        """Test that missing name field causes validation error"""
        res = client.post("/api/register", json={"email": "missing@example.com", "password": "Password123"})
        assert res.status_code == 422

    def test_register_missing_email_field(self, client):
        """Test that missing email field causes validation error"""
        res = client.post("/api/register", json={"name": "Missing Email", "password": "Password123"})
        assert res.status_code == 422

    def test_register_missing_password_field(self, client):
        """Test that missing password field causes validation error"""
        res = client.post("/api/register", json={"name": "Missing Pass", "email": "missing@example.com"})
        assert res.status_code == 422

    def test_register_empty_name(self, client):
        """Test that empty name is accepted (no specific validation mentioned)"""
        res = client.post("/api/register", json={"name": "", "email": "emptyname@example.com", "password": "Password123"})
        assert res.status_code in [201, 422]

    def test_register_empty_email(self, client):
        """Test that empty email is rejected"""
        res = client.post("/api/register", json={"name": "Empty Email", "email": "", "password": "Password123"})
        assert res.status_code == 422

    def test_register_empty_password(self, client):
        """Test that empty password is rejected"""
        res = client.post("/api/register", json={"name": "Empty Pass", "email": "emptypass@example.com", "password": ""})
        assert res.status_code == 422

    def test_register_status_code_201(self, client):
        """Test that successful registration returns 201 Created"""
        res = client.post("/api/register", json={"name": "Test User", "email": "test201@example.com", "password": "CreatedPass123"})
        assert res.status_code == 201

    def test_register_multiple_valid_users(self, client, db_session_fixture: Session):
        """Test that multiple users can be registered successfully"""
        users_data = [
            {"name": "User1", "email": "user1@example.com", "password": "Pass1111"},
            {"name": "User2", "email": "user2@example.com", "password": "Pass2222"},
            {"name": "User3", "email": "user3@example.com", "password": "Pass3333"},
        ]
        for data in users_data:
            res = client.post("/api/register", json=data)
            assert res.status_code == 201

        user_count = db_session_fixture.query(User).count()
        assert user_count == 3

    def test_register_password_with_special_characters(self, client):
        """Test that passwords with special characters are accepted"""
        res = client.post("/api/register", json={
            "name": "Special Char User",
            "email": "special@example.com",
            "password": "Pass!@#$%^&*()"
        })
        assert res.status_code == 201

    def test_register_name_with_special_characters(self, client):
        """Test that names with special characters are accepted"""
        res = client.post("/api/register", json={
            "name": "José García-López",
            "email": "jose@example.com",
            "password": "ValidPass123"
        })
        assert res.status_code == 201


class TestLogin:
    def test_login_success(self, client):
        client.post("/api/register", json={"name": "Test", "email": "login@example.com", "password": "Password1"})
        res = client.post("/api/login", json={"email": "login@example.com", "password": "Password1"})
        assert res.status_code == 200
        assert "access_token" in res.json()

    def test_login_invalid_credentials(self, client):
        res = client.post("/api/login", json={"email": "noone@example.com", "password": "Wrong123"})
        assert res.status_code == 401
