from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import ValidationError
from datetime import datetime, timezone
import json
import logging

from models.schemas import AudioChunk, ErrorResponse, StreamResponse, SessionSummaryResponse
from services.session_manager import session_manager
from core.config import settings
from db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/audio/{session_id}")
async def websocket_audio(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = session_manager.get_or_create_session(session_id)
    
    # Bug 3: Create Session Lifecycle in DB
    db = get_db()
    if db is not None:
        try:
            await db.sessions.update_one(
                {"session_id": session_id},
                {"$setOnInsert": {
                    "session_id": session_id,
                    "start_time": datetime.now(timezone.utc),
                    "status": "in_progress",
                    "source_type": "websocket_stream"
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to create/upsert session {session_id} in DB: {e}")

    try:
        while True:
            # The client sends JSON string payloads
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
                chunk = AudioChunk(**payload)
            except json.JSONDecodeError:
                await _send_error(websocket, session_id, "Invalid JSON payload")
                continue
            except ValidationError as e:
                await _send_error(websocket, session_id, "Validation Error", details=str(e))
                continue
                
            # Verify session ID matches
            if chunk.session_id != session_id:
                await _send_error(websocket, session_id, "Session ID mismatch in payload")
                continue

            # Add chunk and buffer
            is_sequential = session.add_chunk(chunk)
            if not is_sequential:
                # We could send a specific warning event back, but for now we just buffer out-of-order chunks
                pass
                
            # Check buffer and trigger ML inference
            responses = await session.process_ready_chunks()
            for response in responses:
                await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket for session {session_id}: {e}")
    finally:
        # Bug 3: Complete Session Lifecycle in DB via SessionManager
        await session_manager.finalize_session(session_id)

async def _send_error(websocket: WebSocket, session_id: str, error: str, details: str = None):
    error_resp = ErrorResponse(
        session_id=session_id,
        error=error,
        details=details,
        timestamp=datetime.now(timezone.utc)
    )
    await websocket.send_text(error_resp.model_dump_json())

@router.post("/audio/chunk", response_model=list[StreamResponse])
async def rest_audio_chunk(chunk: AudioChunk):
    """
    REST fallback endpoint for testing. 
    Accepts the exact same JSON body as the WebSocket.
    Returns any inference results that were triggered by this chunk.
    """
    session = session_manager.get_or_create_session(chunk.session_id)
    session.add_chunk(chunk)
    
    try:
        responses = await session.process_ready_chunks()
        return responses
    except Exception as e:
        logger.error(f"Error in REST chunk processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str):
    """
    Returns the final summary for a session.
    """
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available (running in memory mode)")
        
    session_doc = await db.sessions.find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if session_doc.get("status") != "completed":
        # If it's still in progress, we can return the current state, 
        # but for simplicity, let's wait until it's finalized or just return what we have.
        pass
        
    # Also fetch window results for the timeline
    cursor = db.window_results.find({"session_id": session_id}).sort("timestamp", 1)
    results = await cursor.to_list(length=1000)
    
    total_chunks = session_doc.get("total_windows", 0)
    final_verdict = session_doc.get("final_verdict", "UNCERTAIN")
    avg_p_fake = session_doc.get("final_p_fake", 0.0)
    status = session_doc.get("status", "in_progress")
    
    timestamps = [r["timestamp"].timestamp() if hasattr(r["timestamp"], "timestamp") else r["timestamp"] for r in results]
    window_scores = [r.get("smoothed_p_fake", r.get("p_fake", 0.0)) for r in results]
    
    return SessionSummaryResponse(
        session_id=session_id,
        total_chunks=total_chunks,
        final_verdict=final_verdict,
        average_confidence=avg_p_fake, # mapped to average_confidence for legacy frontend compat
        timestamps=timestamps,
        window_scores=window_scores,
        status=status
    )
