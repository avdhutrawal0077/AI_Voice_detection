from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

from core.config import settings
from db.database import db_state, init_db
from api.endpoints import health, ingestion
from services.session_manager import session_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db_state.client = AsyncIOMotorClient(settings.mongodb_uri)
    await init_db()
    
    # Start session manager cleanup task
    cleanup_task = asyncio.create_task(session_manager.cleanup_idle_sessions())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    db_state.client.close()

app = FastAPI(
    title=settings.app_name,
    description="SIH 26104 - AI Powered Real Time Voice Cloning Detection System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permissive for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(ingestion.router, tags=["Ingestion"])
# The old endpoints depend on heavy ML libs that have been moved to the ML team's domain.
# app.include_router(analyze.router, prefix="/api/v1", tags=["Analyze"])
# app.include_router(stream.router, prefix="/api/v1", tags=["Stream"])
# app.include_router(results.router, prefix="/api/v1", tags=["Results"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)