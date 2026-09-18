import sys
import os
import logging
from typing import Dict, List, Optional
from core.config import settings

logger = logging.getLogger(__name__)

# Add AI Pipeline to sys.path so it can be imported
pipeline_path = os.path.abspath(settings.ai_pipeline_dir)
if pipeline_path not in sys.path:
    sys.path.insert(0, pipeline_path)
    
src_path = os.path.join(pipeline_path, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try to import the real pipeline
try:
    if settings.use_real_pipeline:
        from wave_spectrum_inference import WaveSpectrumDetector, TemporalSmoother
        _detector = WaveSpectrumDetector()
        _pipeline_available = True
        logger.info("AI Pipeline loaded successfully.")
    else:
        _pipeline_available = False
        logger.info("AI Pipeline is disabled in config.")
except ImportError as e:
    _pipeline_available = False
    logger.error(f"Failed to import AI Pipeline: {e}. Falling back to mock implementation.")
except Exception as e:
    _pipeline_available = False
    logger.error(f"Error initializing AI Pipeline: {e}. Falling back to mock implementation.")


class PipelineService:
    _session_smoothers: Dict[str, any] = {}
    
    @classmethod
    def get_smoother(cls, session_id: str):
        if not _pipeline_available:
            return None
        if session_id not in cls._session_smoothers:
            cls._session_smoothers[session_id] = TemporalSmoother(high=settings.high_threshold, low=settings.low_threshold)
        return cls._session_smoothers[session_id]
        
    @classmethod
    def clear_session(cls, session_id: str):
        if session_id in cls._session_smoothers:
            del cls._session_smoothers[session_id]
            
    @classmethod
    def predict(cls, pcm_list: List[float], session_id: str) -> dict:
        if not _pipeline_available:
            return cls._mock_predict(pcm_list, session_id)
            
        try:
            # 1. Run inference
            result = _detector.predict(pcm_list)
            
            # 2. Update smoother
            smoother = cls.get_smoother(session_id)
            smoothed = smoother.update(result["p_fake"])
            
            # 3. Merge and return as dict mapping to PipelineResponse
            return {
                "verdict": smoothed["verdict"],
                "confidence": smoothed["confidence"],
                "uncertain": smoothed["uncertain"],
                "p_fake": result["p_fake"],
                "smoothed_p_fake": smoothed["smoothed_p_fake"],
                "windows_seen": smoothed["windows_seen"],
                "p_a": result["p_a"],
                "p_b": result["p_b"],
                "processing_time_ms": result["processing_time_ms"],
                "branch_timing": result["branch_timing"],
                "acoustic_features": result["acoustic_features"]
            }
        except Exception as e:
            logger.error(f"Error during AI inference: {e}")
            raise e
            
    @classmethod
    def _mock_predict(cls, pcm_list: List[float], session_id: str) -> dict:
        if not getattr(settings, 'use_mock_pipeline', False):
            raise RuntimeError("Real AI pipeline is unavailable and mock pipeline is disabled.")
            
        # Generate mock dynamic confidence scores
        import numpy as np
        base_score = 0.85 + (np.random.random() * 0.1) # 0.85 to 0.95
        
        return {
            "verdict": "AI",
            "confidence": base_score,
            "uncertain": False,
            "p_fake": base_score,
            "smoothed_p_fake": base_score,
            "windows_seen": 1,
            "p_a": base_score - 0.02,
            "p_b": base_score + 0.01,
            "processing_time_ms": 25.0,
            "branch_timing": {
                "model_a_ms": 10.0,
                "model_b_ms": 12.0,
                "model_c_ms": 3.0
            },
            "acoustic_features": {
                "pitch_f0_hz": 120.0,
                "jitter_percent": 1.2,
                "shimmer_percent": 3.4,
                "hnr_db": 15.0,
                "spectral_centroid_hz": 1500.0,
                "rms_energy_db": -20.0,
                "speech_rate_syll_per_sec": 4.5,
                "pause_ratio_percent": 10.0
            }
        }
