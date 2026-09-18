from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import ValidationError
from datetime import datetime, timezone
import json
import logging

from models.schemas import AudioChunk, ErrorResponse, StreamResponse, SessionSummaryResponse
from services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/audio/{session_id}")
async def websocket_audio(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = session_manager.get_or_create_session(session_id)
    
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
        session_manager.remove_session(session_id)

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
    from db.database import get_db
    
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available (running in memory mode)")
        
    cursor = db.window_results.find({"session_id": session_id}).sort("timestamp", 1)
    results = await cursor.to_list(length=1000)
    
    if not results:
        raise HTTPException(status_code=404, detail="Session not found or no data")
        
    total_chunks = len(results)
    avg_confidence = sum(r["confidence"] for r in results) / total_chunks
    final_verdict = "AI" if avg_confidence > 0.5 else "HUMAN"
    
    timestamps = [r["timestamp"].timestamp() if hasattr(r["timestamp"], "timestamp") else r["timestamp"] for r in results]
    window_scores = [r["p_fake"] for r in results]
    
    return SessionSummaryResponse(
        session_id=session_id,
        total_chunks=total_chunks,
        final_verdict=final_verdict,
        average_confidence=avg_confidence,
        timestamps=timestamps,
        window_scores=window_scores,
        status="completed"
    )
