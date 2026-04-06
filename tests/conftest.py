"""Pytest configuration and fixtures for testing"""
import pytest
import tempfile
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Setup: Create test engine using a temporary file
test_db_fd, test_db_path = tempfile.mkstemp(suffix=".db")
os.close(test_db_fd)
TEST_SQLALCHEMY_DATABASE_URL = f"sqlite:///{test_db_path}"

test_engine = create_engine(
    TEST_SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Import after engine is created
from app.database import Base, get_db
from app.models.user import User
from app.models.session import SessionModel
from app.models.report import Report
from app.models.shot_event import ShotEvent
from app.models.angle_frame import AngleFrame

# Create tables in test database
Base.metadata.create_all(bind=test_engine)

# Now import the app
from app.main import app
import app.main as _app_main

# Patch the engine referenced in app.main's lifespan so that
# TestClient startup (which triggers lifespan) uses the test DB
# instead of the production SQL Server engine.
_app_main.engine = test_engine

# Override the get_db dependency to use test database
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def client():
    """Fixture that provides a test client"""
    return TestClient(app)


@pytest.fixture
def db_session():
    """Fixture for direct database access in tests"""
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture(autouse=True)
def clear_db_before_test():
    """Clear database before each test"""
    with test_engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.execute(text("PRAGMA foreign_keys = ON"))
    yield


# Cleanup at the end
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db():
    """Delete test database file after all tests"""
    yield
    try:
        os.unlink(test_db_path)
    except Exception as e:
        pass
