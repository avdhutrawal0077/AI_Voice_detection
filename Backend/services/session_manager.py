import asyncio
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

from core.config import settings
from models.schemas import AudioChunk, StreamResponse
from services.ml_interface import InferenceEngine
from db.database import get_db

logger = logging.getLogger(__name__)

class AudioSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.expected_chunk_id = 1
        self.ready_chunks: List[AudioChunk] = []
        self.last_activity = datetime.now(timezone.utc)
        self.pending_chunks: Dict[int, AudioChunk] = {}
        self.processed_windows = 0
        self._pcm_buffer: List[float] = []
        self._inference_window_samples = int(settings.inference_window_sec * settings.sample_rate)
        
    def add_chunk(self, chunk: AudioChunk) -> bool:
        """
        Adds a chunk to the buffer. Returns True if successfully added, False if out of order.
        """
        self.last_activity = datetime.now(timezone.utc)
        
        if chunk.chunk_id == self.expected_chunk_id:
            self._process_sequential(chunk)
            return True
        elif chunk.chunk_id > self.expected_chunk_id:
            logger.warning(f"Session {self.session_id}: Out of order chunk received. Expected {self.expected_chunk_id}, got {chunk.chunk_id}")
            self.pending_chunks[chunk.chunk_id] = chunk
            return False
        else:
            logger.warning(f"Session {self.session_id}: Received old chunk {chunk.chunk_id} when expected {self.expected_chunk_id}")
            return False

    def _process_sequential(self, chunk: AudioChunk):
        self.ready_chunks.append(chunk)
        self.expected_chunk_id += 1
        
        # Check if we can now process any pending chunks
        while self.expected_chunk_id in self.pending_chunks:
            next_chunk = self.pending_chunks.pop(self.expected_chunk_id)
            self.ready_chunks.append(next_chunk)
            self.expected_chunk_id += 1

    async def process_ready_chunks(self) -> List[StreamResponse]:
        """
        Processes chunks by accumulating them into a buffer and triggering ML interface
        once the threshold is met.
        Returns a list of StreamResponse objects.
        """
        responses = []
        while self.ready_chunks:
            chunk = self.ready_chunks.pop(0)
            self._pcm_buffer.extend(chunk.pcm)
            
            # Check if we have enough samples for inference (e.g. 32000 for 2s at 16kHz)
            if len(self._pcm_buffer) >= self._inference_window_samples:
                # Extract window
                window_pcm = self._pcm_buffer[:self._inference_window_samples]
                # Keep remainder
                self._pcm_buffer = self._pcm_buffer[self._inference_window_samples:]
                
                # Run Inference
                try:
                    result = await InferenceEngine.run_inference(
                        audio_window=np.array(window_pcm, dtype=np.float32),
                        sample_rate=chunk.sample_rate,
                        session_id=self.session_id
                    )
                except Exception as e:
                    logger.error(f"Inference failed for session {self.session_id}: {e}")
                    continue

                response = StreamResponse(
                    session_id=self.session_id,
                    event="window_processed",
                    chunk_id=chunk.chunk_id,
                    result=result,
                    timestamp=datetime.now(timezone.utc),
                    status="success"
                )
                responses.append(response)
                
                # Save to MongoDB
                try:
                    db = get_db()
                    if db is not None:
                        await db.window_results.insert_one({
                            "session_id": self.session_id,
                            "chunk_id": chunk.chunk_id,
                            "timestamp": response.timestamp,
                            "verdict": result.verdict,
                            "confidence": result.confidence,
                            "p_fake": result.p_fake,
                            "p_a": result.p_a,
                            "p_b": result.p_b,
                            "processing_time_ms": result.processing_time_ms,
                            "branch_timing": result.branch_timing.model_dump(),
                            "acoustic_features": result.acoustic_features.model_dump(),
                            "temporal_smoothing": result.temporal_smoothing.model_dump()
                        })
                except Exception as e:
                    logger.error(f"Failed to save window result to DB: {e}")
                    
                self.processed_windows += 1
                
        return responses

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, AudioSession] = {}
        
    def get_or_create_session(self, session_id: str) -> AudioSession:
        if session_id not in self.sessions:
            logger.info(f"Creating new session {session_id}")
            self.sessions[session_id] = AudioSession(session_id)
        return self.sessions[session_id]
        
    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            logger.info(f"Removing session {session_id}")
            del self.sessions[session_id]
            InferenceEngine.clear_session(session_id)

    async def cleanup_idle_sessions(self):
        while True:
            now = datetime.now(timezone.utc)
            expired = []
            for sid, session in self.sessions.items():
                if (now - session.last_activity).total_seconds() > settings.session_timeout_sec:
                    expired.append(sid)
            for sid in expired:
                logger.warning(f"Session {sid} timed out and is being cleaned up.")
                self.remove_session(sid)
            await asyncio.sleep(10) # Check every 10 seconds

# Global singleton
session_manager = SessionManager()
