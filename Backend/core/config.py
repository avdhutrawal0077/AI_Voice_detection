from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    app_name: str = "Voice Integrity Verification API"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "voice_detection"

    # ── Canonical Streaming Pipeline Configuration ────────────────────────
    # AudioWorklet chunk:     250 ms  (configured in frontend)
    # Inference window:       2.0 sec (inference_window_sec)
    # Inference hop (50%):    derived — always inference_window_sec / 2
    # Model input (padded):   4.0 sec (TARGET_SAMPLES=64000 in AI_Pipeline)
    # Sample rate:            16,000 Hz
    sample_rate:          int   = 16000
    inference_window_sec: float = 2.0

    @property
    def hop_duration_sec(self) -> float:
        """50% overlap — hop is always derived from inference_window_sec, never independently configured."""
        return self.inference_window_sec / 2.0

    # ── Verdict Thresholds (mirrors AI_Pipeline/models/thresholds.json) ───
    # p_fake >= high_threshold → AI verdict
    # p_fake <= low_threshold  → HUMAN verdict
    # between                  → UNCERTAIN
    high_threshold: float = 0.71
    low_threshold:  float = 0.29

    # ── Session Config ─────────────────────────────────────────────────────
    session_timeout_sec:      float      = 60.0
    max_pcm_size_elements:    int        = 160000  # max 10 sec at 16kHz per chunk
    max_pcm_buffer_sec:       float      = 30.0    # max total PCM buffered per session
    max_pending_chunks:       int        = 50      # max out-of-order chunks held in memory
    max_chunk_id_gap:         int        = 100     # max accepted gap between chunk IDs
    allowed_audio_formats:    List[str]  = ["mono_float32"]

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Set to specific origins in production via .env: CORS_ORIGINS=["https://yourdomain.com"]
    cors_origins:             List[str]  = ["*"]

    # ── AI Pipeline Path ───────────────────────────────────────────────────
    ai_pipeline_dir:  str  = "../AI_Pipeline"
    use_real_pipeline: bool = True
    use_mock_pipeline: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()