from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    app_name: str = "Voice Integrity Verification API"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "voice_detection"
    sample_rate: int = 16000
    window_duration_sec: float = 1.0
    hop_duration_sec: float = 0.5
    
    # New streaming backend config
    session_timeout_sec: float = 60.0
    max_pcm_size_elements: int = 160000  # E.g., max 10 seconds of audio at 16kHz
    allowed_audio_formats: List[str] = ["mono_float32"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()