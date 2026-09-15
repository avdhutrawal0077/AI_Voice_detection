from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import uuid
import numpy as np
from datetime import datetime, timezone
import json
from db.database import get_db
from core.config import settings
from services.audio_processing import is_speech_present, extract_features
from services.ml_dummy import model_instance
from services.aggregation import aggregate_scores

router = APIRouter()

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    db = get_db()
    session_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    
    await db.sessions.insert_one({
        "session_id": session_id,
        "source_type": "stream",
        "start_time": start_time,
        "status": "in_progress",
        "total_windows_processed": 0
    })

    await websocket.send_json({"session_id": session_id, "status": "connected"})
    
    buffer = bytearray()
    window_bytes = int(settings.sample_rate * settings.window_duration_sec * 2) # 16-bit PCM = 2 bytes per sample
    hop_bytes = int(settings.sample_rate * settings.hop_duration_sec * 2)
    window_index = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)
            
            while len(buffer) >= window_bytes:
                chunk = buffer[:window_bytes]
                buffer = buffer[hop_bytes:] # Slide window
                
                # Convert PCM16 to float32
                audio_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                
                if not is_speech_present(audio_array):
                    continue
                    
                features = extract_features(audio_array)
                prediction = model_instance.predict(features)
                
                # Write immediately to DB
                doc = {
                    "session_id": session_id,
                    "window_index": window_index,
                    "timestamp": datetime.now(timezone.utc),
                    "verdict": prediction["verdict"],
                    "risk_score": prediction["risk_score"],
                    "confidence": prediction["confidence"],
                    "features_summary": features.tolist(),
                    "evidence": prediction["evidence"]
                }
                await db.window_results.insert_one(doc)
                
                # Fetch partial history to update the running aggregate
                cursor = db.window_results.find({"session_id": session_id})
                all_windows = await cursor.to_list(length=1000)
                current_verdict, current_risk = aggregate_scores(all_windows)
                
                await websocket.send_json({
                    "event": "window_processed",
                    "window_index": window_index,
                    "window_verdict": prediction["verdict"],
                    "window_risk_score": prediction["risk_score"],
                    "aggregate_verdict": current_verdict,
                    "aggregate_risk_score": current_risk
                })
                
                window_index += 1

    except WebSocketDisconnect:
        # Client disconnected cleanly or network drop
        pass
    finally:
        # Finalize Session
        cursor = db.window_results.find({"session_id": session_id})
        all_windows = await cursor.to_list(length=1000)
        final_verdict, final_risk = aggregate_scores(all_windows)
        
        await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "status": "completed",
                "end_time": datetime.now(timezone.utc),
                "final_verdict": final_verdict,
                "final_risk_score": final_risk,
                "total_windows_processed": window_index
            }}
        )