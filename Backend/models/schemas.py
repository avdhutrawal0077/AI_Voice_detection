from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class Verdict(str, Enum):
    HUMAN = "HUMAN"
    AI = "AI"
    UNCERTAIN = "UNCERTAIN"

class SessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class WindowResultDB(BaseModel):
    session_id: str
    window_index: int
    timestamp: datetime
    verdict: Verdict
    risk_score: float
    confidence: float
    features_summary: List[float] = Field(description="1D array summary of MFCC/Mel features")
    evidence: Dict[str, Any]

class SessionDB(BaseModel):
    session_id: str
    source_type: str 
    original_filename: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SessionStatus
    final_verdict: Optional[Verdict] = None
    final_risk_score: Optional[float] = None
    total_windows_processed: int = 0

class WindowResultResponse(BaseModel):
    window_index: int
    verdict: Verdict
    risk_score: float
    confidence: float

class AnalysisResponse(BaseModel):
    session_id: str
    status: SessionStatus
    final_verdict: Optional[Verdict]
    final_risk_score: Optional[float]
    windows: List[WindowResultResponse]

# ── Streaming Backend Schemas ─────────────────────────────────────────────────
from core.config import settings

class AudioChunk(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the session")
    chunk_id: int = Field(..., description="Sequential chunk ID to track order (starts at 1)")
    pcm: List[float] = Field(..., description="Audio samples as float32 array")
    sample_rate: int = Field(..., description="Sample rate — must be 16000 Hz")
    format: str = Field(..., description="Audio format (e.g., mono_float32)")
    # VAD metadata: true=speech, false=silence, null=VAD unavailable.
    # All chunks (speech and silence) are sent; this field is informational only.
    is_speech: Optional[bool] = Field(None, description="Silero VAD classification for this chunk")

    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v != settings.sample_rate:
            raise ValueError(f"Sample rate must be exactly {settings.sample_rate} Hz")
        return v

    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        if v not in settings.allowed_audio_formats:
            raise ValueError(f"Format {v} is not allowed. Allowed: {settings.allowed_audio_formats}")
        return v

    @field_validator('pcm')
    @classmethod
    def validate_pcm_length(cls, v):
        if not v:
            raise ValueError("PCM data cannot be empty")
        if len(v) > settings.max_pcm_size_elements:
            raise ValueError(f"PCM data too large. Max elements allowed: {settings.max_pcm_size_elements}")
        return v

class AcousticFeatures(BaseModel):
    pitch_f0_hz: float
    jitter_percent: float
    shimmer_percent: float
    hnr_db: float
    spectral_centroid_hz: float
    rms_energy_db: float
    speech_rate_syll_per_sec: float
    pause_ratio_percent: float

class BranchTiming(BaseModel):
    model_a_ms: float
    model_b_ms: float
    model_c_ms: float

class PipelineResponse(BaseModel):
    verdict: str = Field(..., description="Authoritative verdict derived from smoothed_p_fake")
    confidence: float = Field(..., description="Confidence score based on smoothed_p_fake")
    uncertain: bool = Field(..., description="Whether the smoothed result is uncertain")
    p_fake: float = Field(..., description="Raw probability from the fusion model")
    smoothed_p_fake: float = Field(..., description="Temporally smoothed probability")
    windows_seen: int = Field(..., description="Number of windows seen by the smoother")
    p_a: float
    p_b: float
    processing_time_ms: float
    branch_timing: BranchTiming
    acoustic_features: AcousticFeatures

class StreamResponse(BaseModel):
    session_id: str
    event: str
    chunk_id: Optional[int] = None
    result: Optional[PipelineResponse] = None
    timestamp: datetime
    status: str

class FinalResultResponse(BaseModel):
    """
    Sent by the backend in response to an END_SESSION control message.
    This is the single authoritative result for the session — the frontend
    must use this (not the last streaming window result) for the final report.
    """
    session_id: str
    event: str = "final_result"
    verdict: str = Field(..., description="HUMAN | AI | UNCERTAIN")
    p_fake: float = Field(..., description="Average smoothed_p_fake across all windows")
    confidence: float = Field(..., description="Final confidence (=p_fake for AI, =1-p_fake for HUMAN)")
    total_windows: int
    windows_failed: int = 0
    timestamp: datetime

class SessionSummaryResponse(BaseModel):
    session_id: str
    total_windows: int          # was total_chunks — renamed to be consistent
    final_verdict: str
    average_confidence: float
    timestamps: List[float]
    window_scores: List[float]
    status: str

class ErrorResponse(BaseModel):
    session_id: Optional[str] = None
    error: str
    details: Optional[str] = None
    timestamp: datetime