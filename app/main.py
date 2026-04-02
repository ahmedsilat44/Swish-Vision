from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import engine, Base
from app.api import auth, sessions, dashboard
from app.core.middleware import setup_middleware


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
