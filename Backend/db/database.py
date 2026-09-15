from motor.motor_asyncio import AsyncIOMotorClient
import logging
from core.config import settings

logger = logging.getLogger(__name__)

class DatabaseState:
    client: AsyncIOMotorClient = None
    db = None

db_state = DatabaseState()

async def init_db():
    try:
        db_state.db = db_state.client[settings.mongodb_db_name]
        
        # Create indexes for performance
        await db_state.db.sessions.create_index("session_id", unique=True)
        await db_state.db.sessions.create_index("start_time")
        
        await db_state.db.window_results.create_index("session_id")
        await db_state.db.window_results.create_index([("session_id", 1), ("window_index", 1)])
        
        logger.info("MongoDB initialized and indexes created.")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        raise

def get_db():
    return db_state.db