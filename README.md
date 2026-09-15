# Voice Integrity Verification Backend

Backend for SIH 26104: AI Powered Real Time Voice Cloning Detection System.

This backend provides a high-performance, real-time ingestion pipeline via WebSockets and REST.

## 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Run the Server

```bash
python main.py
```
The server will start at `http://0.0.0.0:8000`.

## 3. For the ML Team 🚀

The backend handles transport, session management, buffering, and validation. **Do not** write ML code in the transport layer.

All ML code belongs in **`services/ml_interface.py`**.

When a session has accumulated enough audio (configurable via `window_duration_sec` in `core/config.py`), the backend automatically calls:

```python
async def run_inference(audio_window: np.ndarray, sample_rate: int, session_id: str) -> InferenceResult:
```

### What you receive:
- `audio_window`: A 1D float32 `numpy` array of PCM samples representing exactly 1 window of audio.
- `sample_rate`: The integer sample rate (e.g., 16000).

### What you must return:
An `InferenceResult` object (defined in `models/schemas.py`). For example:
```python
return InferenceResult(
    label="AI",
    confidence=0.98,
    evidence={"mfcc_mean": 1.23, "note": "Likely synthetic - strongest evidence in high frequency band"}
)
```

## 4. API Endpoints

### WebSocket: `/ws/audio/{session_id}`
Client sends chunks as JSON strings:
```json
{
    "session_id": "test_session_001",
    "chunk_id": 1,
    "pcm": [0.0, 0.001, ...], 
    "sample_rate": 16000,
    "format": "mono_float32"
}
```

The server responds with:
```json
{
    "session_id": "test_session_001",
    "event": "window_processed",
    "chunk_range": [1],
    "result": {
        "label": "NOT_IMPLEMENTED",
        "confidence": 0.0,
        "evidence": {}
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "status": "success"
}
```

### REST Fallback: `POST /audio/chunk`
Same request JSON body. Returns an array of processed windows if any were triggered by the chunk.
