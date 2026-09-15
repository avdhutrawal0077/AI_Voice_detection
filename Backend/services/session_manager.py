import asyncio
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

from core.config import settings
from models.schemas import AudioChunk, InferenceResult, StreamResponse
from services.ml_interface import InferenceEngine

logger = logging.getLogger(__name__)

class AudioSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.expected_chunk_id = 1
        self.buffer = np.array([], dtype=np.float32)
        self.last_activity = datetime.now(timezone.utc)
        self.window_size_elements = int(settings.sample_rate * settings.window_duration_sec)
        self.hop_size_elements = int(settings.sample_rate * settings.hop_duration_sec)
        # We can store pending out-of-order chunks in a dict: {chunk_id: chunk_data}
        self.pending_chunks: Dict[int, AudioChunk] = {}
        self.processed_windows = 0
        
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
        self.buffer = np.concatenate((self.buffer, np.array(chunk.pcm, dtype=np.float32)))
        self.expected_chunk_id += 1
        
        # Check if we can now process any pending chunks
        while self.expected_chunk_id in self.pending_chunks:
            next_chunk = self.pending_chunks.pop(self.expected_chunk_id)
            self.buffer = np.concatenate((self.buffer, np.array(next_chunk.pcm, dtype=np.float32)))
            self.expected_chunk_id += 1

    async def extract_and_infer_windows(self) -> List[StreamResponse]:
        """
        Extracts full windows from the buffer and runs inference on them.
        Returns a list of StreamResponse objects to be sent back to the client.
        """
        responses = []
        while len(self.buffer) >= self.window_size_elements:
            # Extract window
            window = self.buffer[:self.window_size_elements]
            # Slide buffer
            self.buffer = self.buffer[self.hop_size_elements:]
            
            # Run Inference
            try:
                result = await InferenceEngine.run_inference(
                    audio_window=window,
                    sample_rate=settings.sample_rate,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Inference failed for session {self.session_id}: {e}")
                result = InferenceResult(label="ERROR", evidence={"error": str(e)})

            response = StreamResponse(
                session_id=self.session_id,
                event="window_processed",
                chunk_range=[self.expected_chunk_id - 1], # Approximation, ideally we'd map samples back to chunk_ids
                result=result,
                timestamp=datetime.now(timezone.utc),
                status="success"
            )
            responses.append(response)
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
