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
        if db_state.client is None:
            logger.warning("MongoDB client is None. Proceeding in in-memory mode.")
            return

        # Try a ping to confirm connection
        await db_state.client.admin.command('ping')
        
        db_state.db = db_state.client[settings.mongodb_db_name]
        
        await db_state.db.sessions.create_index("session_id", unique=True)
        await db_state.db.sessions.create_index("start_time")
        
        await db_state.db.window_results.create_index("session_id")
        await db_state.db.window_results.create_index([("session_id", 1), ("chunk_id", 1)])
        
        await db_state.db.users.create_index("email", unique=True)
        
        logger.info("MongoDB initialized and indexes created.")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}. Falling back to in-memory mode (no persistence).")
        db_state.db = None
        db_state.client = None

def get_db():
    return db_state.db