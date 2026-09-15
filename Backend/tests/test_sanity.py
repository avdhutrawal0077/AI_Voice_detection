import pytest
import numpy as np
import soundfile as sf
import io
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def sample_wav_bytes():
    # Generate 3 seconds of 440Hz sine wave as PCM 16kHz WAV
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV', subtype='PCM_16')
    return buffer.getvalue()

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["api"] == "ok"

def test_analyze_and_fetch(sample_wav_bytes):
    # 1. Upload
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", sample_wav_bytes, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "completed"
    
    session_id = data["session_id"]
    
    # 2. Fetch session details
    res_get = client.get(f"/api/v1/results/{session_id}")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["session_id"] == session_id
    assert "windows" in get_data