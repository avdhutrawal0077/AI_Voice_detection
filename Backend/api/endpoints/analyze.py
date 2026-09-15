from fastapi import APIRouter, UploadFile, File, HTTPException, status
import uuid
from datetime import datetime, timezone
from db.database import get_db
from services.audio_processing import load_audio_bytes, generate_windows, is_speech_present, extract_features
from services.ml_dummy import model_instance
from services.aggregation import aggregate_scores
from models.schemas import SessionStatus, Verdict, AnalysisResponse, WindowResultResponse

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV, MP3, FLAC, or OGG.")
        
    db = get_db()
    session_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    
    try:
        audio_bytes = await file.read()
        audio_array = load_audio_bytes(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    windows = list(generate_windows(audio_array))
    
    window_docs = []
    response_windows = []
    
    for idx, window in enumerate(windows):
        if not is_speech_present(window):
            continue # Skip silent windows
            
        features = extract_features(window)
        prediction = model_instance.predict(features)
        
        doc = {
            "session_id": session_id,
            "window_index": idx,
            "timestamp": datetime.now(timezone.utc),
            "verdict": prediction["verdict"],
            "risk_score": prediction["risk_score"],
            "confidence": prediction["confidence"],
            "features_summary": features.tolist(),
            "evidence": prediction["evidence"]
        }
        window_docs.append(doc)
        
        response_windows.append(WindowResultResponse(
            window_index=idx,
            verdict=prediction["verdict"],
            risk_score=prediction["risk_score"],
            confidence=prediction["confidence"]
        ))

    if window_docs:
        await db.window_results.insert_many(window_docs)
        
    final_verdict, final_risk = aggregate_scores(window_docs)
    
    session_doc = {
        "session_id": session_id,
        "source_type": "upload",
        "original_filename": file.filename,
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc),
        "status": SessionStatus.COMPLETED,
        "final_verdict": final_verdict,
        "final_risk_score": final_risk,
        "total_windows_processed": len(window_docs)
    }
    
    await db.sessions.insert_one(session_doc)
    
    return AnalysisResponse(
        session_id=session_id,
        status=SessionStatus.COMPLETED,
        final_verdict=final_verdict,
        final_risk_score=final_risk,
        windows=response_windows
    )