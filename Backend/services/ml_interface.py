import numpy as np
from typing import Dict, Any, List
from models.schemas import PipelineResponse
from services.ai_pipeline import PipelineService

class InferenceEngine:
    """
    Interface for the ML team to plug in their models.
    The backend handles WebSocket transport, windowing, and session management.
    When a full window of audio is ready, this class is called.
    """
    
    @classmethod
    async def run_inference(cls, audio_window: np.ndarray, sample_rate: int, session_id: str) -> PipelineResponse:
        """
        Runs the actual ML detection model on the provided audio window.
        """
        # Convert np.ndarray to list of floats for the pipeline
        pcm_list = audio_window.tolist()
        
        # Offload CPU/GPU inference to a worker thread so we don't block the FastAPI event loop.
        # This allows the WebSocket to continue handling ping/pong and END_SESSION messages.
        import asyncio
        result_dict = await asyncio.to_thread(PipelineService.predict, pcm_list, session_id)
        
        # Parse into PipelineResponse
        return PipelineResponse(**result_dict)

    @classmethod
    def get_session_history(cls, session_id: str) -> List[float]:
        # Temporarily mock or remove this if unused by session_manager
        return []
        
    @classmethod
    def clear_session(cls, session_id: str):
        PipelineService.clear_session(session_id)
