import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import engine, Base
from app.config import settings
from app.api import auth, sessions, dashboard
from app.core.middleware import setup_middleware
import app.models  # noqa: F401 – importing the package registers all models with Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    yield


app = FastAPI(title="SwishVision API", version="1.0.0", lifespan=lifespan)

# Middleware (includes CORS for React dev server)
setup_middleware(app)

# API routes
app.include_router(auth.router)
app.include_router(auth.auth_router)
app.include_router(sessions.router)
app.include_router(dashboard.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
