from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import engine, Base
from app.api import auth, sessions, dashboard
from app.core.middleware import setup_middleware
import app.models.user  # noqa: F401 – ensure tables are registered with Base
import app.models.session  # noqa: F401
import app.models.revoked_token  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="SwishVision API", version="1.0.0", lifespan=lifespan)

# Middleware (includes CORS for React dev server)
setup_middleware(app)

# API routes
app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(dashboard.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
