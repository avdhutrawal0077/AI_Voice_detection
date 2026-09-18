from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    app_name: str = "Voice Integrity Verification API"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "voice_detection"

    # ── Canonical Streaming Pipeline Configuration ────────────────────────
    # AudioWorklet chunk:     250 ms  (configured in frontend)
    # Inference window:       2.0 sec (inference_window_sec)
    # Inference hop (50%):    1.0 sec (hop_duration_sec)
    # Model input (padded):   4.0 sec (TARGET_SAMPLES=64000 in AI_Pipeline)
    # Sample rate:            16,000 Hz
    sample_rate:          int   = 16000
    inference_window_sec: float = 2.0
    
    @property
    def hop_duration_sec(self) -> float:
        """Architecturally mandated 50% overlap."""
        return self.inference_window_sec / 2.0

    # ── Verdict Thresholds (mirrors AI_Pipeline/models/thresholds.json) ───
    # p_fake >= high_threshold → AI verdict
    # p_fake <= low_threshold  → HUMAN verdict
    # between                  → UNCERTAIN
    high_threshold: float = 0.71
    low_threshold:  float = 0.29

    # ── Session Config ─────────────────────────────────────────────────────
    session_timeout_sec:      float      = 60.0
    max_pcm_size_elements:    int        = 160000  # max 10 sec at 16kHz
    allowed_audio_formats:    List[str]  = ["mono_float32"]

    # ── AI Pipeline Path ───────────────────────────────────────────────────
    ai_pipeline_dir:  str  = "../AI_Pipeline"
    use_real_pipeline: bool = True
    use_mock_pipeline: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()