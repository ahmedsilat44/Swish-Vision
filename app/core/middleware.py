from fastapi import FastAPI
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "media-src 'self';"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if settings.ENV == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def setup_middleware(app: FastAPI):
    # HTTPS redirect only in production.
    # WARNING: HTTPSRedirectMiddleware determines the scheme from the raw
    # incoming request. If this app runs behind a reverse proxy or ingress that
    # terminates TLS (nginx, a cloud load-balancer, Kubernetes ingress, etc.),
    # the app will always see plain-HTTP requests, causing an infinite redirect
    # loop.  Preferred approach: handle the HTTP→HTTPS redirect at the
    # proxy/ingress layer.  If you must do it in-app, run the ASGI server with
    # proxy-header trust enabled, e.g.:
    #   uvicorn app.main:app --proxy-headers --forwarded-allow-ips=<proxy-ip>
    # so that X-Forwarded-Proto is honoured before this middleware inspects the
    # scheme.
    if settings.ENV == "production":
        app.add_middleware(HTTPSRedirectMiddleware)

    # CORS – allow React dev server and localhost origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(SecurityHeadersMiddleware)
