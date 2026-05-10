"""Tests for middleware behaviour, specifically security headers and HTTPS redirect."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.middleware import setup_middleware


def _make_app() -> FastAPI:
    """Return a minimal FastAPI app with a single health endpoint."""
    mini = FastAPI()

    @mini.get("/health")
    def health():
        return {"ok": True}

    return mini


# ── HSTS header ───────────────────────────────────────────────────────────────

def test_hsts_header_present_in_production(monkeypatch):
    """Strict-Transport-Security must be set when ENV == 'production'."""
    from app.config import settings
    monkeypatch.setattr(settings, "ENV", "production")

    app = _make_app()
    setup_middleware(app)

    # TestClient follows HTTPS redirects by default; raise_server_exceptions
    # keeps transport errors visible while still returning 3xx responses.
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("https://testserver/health")

    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers
    assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
    assert "includeSubDomains" in response.headers["Strict-Transport-Security"]


def test_hsts_header_absent_in_development():
    """Strict-Transport-Security must NOT be set in non-production environments."""
    # settings.ENV defaults to "development" – no monkeypatching needed.
    app = _make_app()
    setup_middleware(app)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert "Strict-Transport-Security" not in response.headers


# ── HTTPS redirect ────────────────────────────────────────────────────────────

def test_http_redirects_to_https_in_production(monkeypatch):
    """Plain-HTTP requests must be redirected to HTTPS when ENV == 'production'.

    Note: In real deployments behind a TLS-terminating reverse proxy this
    middleware can cause redirect loops unless the ASGI server is started with
    --proxy-headers so that X-Forwarded-Proto is honoured.  See middleware.py
    for the full explanation.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "ENV", "production")

    app = _make_app()
    setup_middleware(app)

    # Disable redirect following so we can assert the 307 response itself.
    client = TestClient(app, follow_redirects=False, raise_server_exceptions=False)
    response = client.get("http://testserver/health")

    assert response.status_code in (301, 307, 308)
    location = response.headers.get("location", "")
    assert location.startswith("https://"), (
        f"Expected redirect to https://, got Location: {location!r}"
    )


def test_no_https_redirect_in_development():
    """HTTP requests must NOT be redirected when ENV != 'production'."""
    app = _make_app()
    setup_middleware(app)

    client = TestClient(app, follow_redirects=False)
    response = client.get("http://testserver/health")

    assert response.status_code == 200


# ── Other security headers ────────────────────────────────────────────────────

def test_static_security_headers_always_present():
    """X-Content-Type-Options, X-Frame-Options, X-XSS-Protection must be set
    regardless of environment."""
    app = _make_app()
    setup_middleware(app)

    client = TestClient(app)
    response = client.get("/health")

    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"


def test_unhandled_error_keeps_cors_headers_for_allowed_origin():
    """Even on unexpected 500s, allowed browser origins should still receive CORS headers."""
    app = FastAPI()

    @app.get("/boom")
    def boom():
        raise RuntimeError("unexpected")

    setup_middleware(app)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 500
    assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
    assert response.headers.get("Access-Control-Allow-Credentials") == "true"
