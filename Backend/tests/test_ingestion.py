import pytest
import json
from fastapi.testclient import TestClient
from main import app
from models.schemas import AudioChunk
from services.session_manager import session_manager
from core.config import settings
import asyncio

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_sessions():
    session_manager.sessions.clear()
    yield
    session_manager.sessions.clear()

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["api"] == "ok"

def test_rest_audio_chunk_valid():
    payload = {
        "session_id": "test_session_rest",
        "chunk_id": 1,
        "pcm": [0.0] * 16000,
        "sample_rate": 16000,
        "format": "mono_float32"
    }
    response = client.post("/audio/chunk", json=payload)
    assert response.status_code == 200
    # One chunk is exactly 1 second, our window is 1.0 sec, so we expect 1 window processed
    data = response.json()
    assert len(data) == 1
    assert data[0]["event"] == "window_processed"
    assert data[0]["result"]["label"] == "NOT_IMPLEMENTED"

def test_rest_audio_chunk_invalid_format():
    payload = {
        "session_id": "test_session_rest",
        "chunk_id": 1,
        "pcm": [0.0] * 16000,
        "sample_rate": 16000,
        "format": "invalid_format"
    }
    response = client.post("/audio/chunk", json=payload)
    assert response.status_code == 422 # Validation error

def test_websocket_ingestion():
    with client.websocket_connect("/ws/audio/test_session_ws") as websocket:
        payload = {
            "session_id": "test_session_ws",
            "chunk_id": 1,
            "pcm": [0.0] * 16000,
            "sample_rate": 16000,
            "format": "mono_float32"
        }
        websocket.send_text(json.dumps(payload))
        
        # We expect a window_processed event
        response = websocket.receive_json()
        assert response["event"] == "window_processed"
        assert response["session_id"] == "test_session_ws"
        assert response["result"]["label"] == "NOT_IMPLEMENTED"
        
        # Test out of order chunk
        payload["chunk_id"] = 3
        websocket.send_text(json.dumps(payload))
        # Shouldn't trigger a window immediately because chunk 2 is missing
        # But wait, wait_for might block in a real scenario, here we just do a non-blocking check if possible, or send chunk 2
        payload["chunk_id"] = 2
        websocket.send_text(json.dumps(payload))
        
        # Now chunk 2 AND chunk 3 will be processed.
        # Chunk 2 gives another 1s, which will slide by hop_size (0.5s) and create two more windows (from chunk 1+2 overlapping)
        # Then chunk 3 gives another 1s.
        response_2 = websocket.receive_json()
        assert response_2["event"] == "window_processed"

def test_session_cleanup():
    session = session_manager.get_or_create_session("timeout_test")
    assert "timeout_test" in session_manager.sessions
    
    # Force timeout
    import datetime
    session.last_activity = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=settings.session_timeout_sec + 10)
    
    # Run a single step of cleanup manually
    now = datetime.datetime.now(datetime.timezone.utc)
    expired = []
    for sid, sess in session_manager.sessions.items():
        if (now - sess.last_activity).total_seconds() > settings.session_timeout_sec:
            expired.append(sid)
    for sid in expired:
        session_manager.remove_session(sid)
        
    assert "timeout_test" not in session_manager.sessions
