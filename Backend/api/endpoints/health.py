from fastapi import APIRouter, status
from db.database import db_state
from core.config import settings

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    health_status = {"api": "ok"}
    try:
        # Ping mongo
        if db_state.client:
            await db_state.client.admin.command('ping')
            health_status["database"] = "ok"
            health_status["database_name"] = settings.mongodb_db_name
        else:
            health_status["database"] = "unreachable"
    except Exception as e:
        health_status["database"] = "unreachable"
        health_status["db_error"] = str(e)
        
    return health_status