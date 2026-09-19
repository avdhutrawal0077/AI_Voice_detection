from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import ValidationError
from datetime import datetime, timezone
import json
import logging

from models.schemas import (
    AudioChunk, ErrorResponse, StreamResponse,
    SessionSummaryResponse, FinalResultResponse
)
from services.session_manager import session_manager
from core.config import settings
from db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/audio/{session_id}")
async def websocket_audio(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = session_manager.get_or_create_session(session_id)

    # Record session start in DB (upsert — safe on reconnect)
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

    explicit_end = False  # tracks whether the frontend sent END_SESSION

    try:
        while True:
            data = await websocket.receive_text()

            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await _send_error(websocket, session_id, "Invalid JSON payload")
                continue

            # ── Control messages ───────────────────────────────────────────────
            if payload.get("type") == "control":
                action = payload.get("action")
                if action == "END_SESSION":
                    logger.info(f"Session {session_id}: END_SESSION received.")
                    explicit_end = True

                    # Finalize session and get the authoritative result
                    agg = await session_manager.finalize_session(session_id)

                    # Build and send FINAL_RESULT to the frontend
                    windows_failed = 0
                    if session is not None:
                        windows_failed = getattr(session, 'windows_failed', 0)

                    final_resp = FinalResultResponse(
                        session_id=session_id,
                        event="final_result",
                        verdict=agg["final_verdict"] if agg else "UNCERTAIN",
                        p_fake=agg["final_p_fake"] if agg else 0.0,
                        confidence=agg["final_confidence"] if agg else 0.0,
                        total_windows=agg["total_windows"] if agg else 0,
                        windows_failed=windows_failed,
                        timestamp=datetime.now(timezone.utc)
                    )
                    await websocket.send_text(final_resp.model_dump_json())
                    break  # Frontend will close the socket after receiving this

                else:
                    logger.warning(f"Session {session_id}: Unknown control action '{action}'.")
                continue

            # ── Audio chunk messages ───────────────────────────────────────────
            try:
                chunk = AudioChunk(**payload)
            except ValidationError as e:
                await _send_error(websocket, session_id, "Validation Error", details=str(e))
                continue

            if chunk.session_id != session_id:
                await _send_error(websocket, session_id, "Session ID mismatch in payload")
                continue

            session.add_chunk(chunk)
            responses = await session.process_ready_chunks()
            for response in responses:
                await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket for session {session_id}: {e}")
    finally:
        # Safety net: if the frontend disconnected without sending END_SESSION,
        # finalize the session here. finalize_session() is idempotent so this
        # is harmless if END_SESSION was already handled above.
        if not explicit_end:
            logger.info(f"Session {session_id}: finalizing in safety-net finally block.")
            await session_manager.finalize_session(session_id)


async def _send_error(websocket: WebSocket, session_id: str, error: str, details: str = None):
    error_resp = ErrorResponse(
        session_id=session_id,
        error=error,
        details=details,
        timestamp=datetime.now(timezone.utc)
    )
    try:
        await websocket.send_text(error_resp.model_dump_json())
    except Exception:
        pass  # socket may already be closed


@router.post("/audio/chunk", response_model=list[StreamResponse])
async def rest_audio_chunk(chunk: AudioChunk):
    """
    REST fallback endpoint for testing.
    Accepts the exact same JSON body as the WebSocket.
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
    Returns the final summary for a completed session.
    """
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available (running in memory mode)")

    session_doc = await db.sessions.find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")

    cursor = db.window_results.find({"session_id": session_id}).sort("timestamp", 1)
    results = await cursor.to_list(length=1000)

    total_windows   = session_doc.get("total_windows", 0)
    final_verdict   = session_doc.get("final_verdict", "UNCERTAIN")
    avg_confidence  = session_doc.get("final_confidence", 0.0)
    status          = session_doc.get("status", "in_progress")

    timestamps    = [
        r["timestamp"].timestamp() if hasattr(r["timestamp"], "timestamp") else r["timestamp"]
        for r in results
    ]
    window_scores = [r.get("smoothed_p_fake", r.get("p_fake", 0.0)) for r in results]

    return SessionSummaryResponse(
        session_id=session_id,
        total_windows=total_windows,
        final_verdict=final_verdict,
        average_confidence=avg_confidence,
        timestamps=timestamps,
        window_scores=window_scores,
        status=status
    )
