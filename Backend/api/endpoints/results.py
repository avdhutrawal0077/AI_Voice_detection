from fastapi import APIRouter, HTTPException
from typing import List
from db.database import get_db

router = APIRouter()

@router.get("/results", response_model=List[dict])
async def list_sessions(skip: int = 0, limit: int = 20):
    db = get_db()
    cursor = db.sessions.find({}, {"_id": 0}).sort("start_time", -1).skip(skip).limit(limit)
    sessions = await cursor.to_list(length=limit)
    return sessions

@router.get("/results/{session_id}")
async def get_session_details(session_id: str):
    db = get_db()
    session = await db.sessions.find_one({"session_id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    cursor = db.window_results.find({"session_id": session_id}, {"_id": 0, "features_summary": 0}).sort("window_index", 1)
    windows = await cursor.to_list(length=1000)
    
    session["windows"] = windows
    return session