from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

from core.config import settings
from db.database import db_state, init_db
from api.endpoints import health, ingestion, auth
from services.session_manager import session_manager
from services.ai_pipeline import _pipeline_available

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        db_state.client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=2000)
        await init_db()
    except Exception as e:
        print(f"Warning: Could not connect to MongoDB on startup. {e}")
        db_state.client = None
    
    # Init AI Pipeline (happens implicitly on import, but we can log)
    print(f"AI Pipeline Loaded: {_pipeline_available}")
    
    # Start session manager cleanup task
    cleanup_task = asyncio.create_task(session_manager.cleanup_idle_sessions())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    if db_state.client:
        db_state.client.close()

app = FastAPI(
    title=settings.app_name,
    description="SIH 26104 - AI Powered Real Time Voice Cloning Detection System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    # Configured via settings.cors_origins (default ["*"] for local dev).
    # Override in .env: CORS_ORIGINS=["https://your-frontend.com"]
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(ingestion.router, tags=["Ingestion"])
# The old endpoints depend on heavy ML libs that have been moved to the ML team's domain.
# They have been successfully deleted along with their legacy services.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)