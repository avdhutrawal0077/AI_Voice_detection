import asyncio
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

from core.config import settings
from models.schemas import AudioChunk, StreamResponse, FinalResultResponse
from services.ml_interface import InferenceEngine
from db.database import get_db

logger = logging.getLogger(__name__)



class AudioSession:
    def __init__(self, session_id: str):
        self.session_id       = session_id
        self.expected_chunk_id = 1
        self.ready_chunks: List[AudioChunk] = []
        self.last_activity    = datetime.now(timezone.utc)
        self.pending_chunks: Dict[int, AudioChunk] = {}
        self.processed_windows = 0
        self.windows_failed   = 0

        self._pcm_buffer: List[float] = []
        self._inference_window_samples = int(settings.inference_window_sec * settings.sample_rate)
        self._inference_hop_samples    = int(settings.hop_duration_sec    * settings.sample_rate)
        self._max_pcm_buffer_samples   = int(settings.max_pcm_buffer_sec * settings.sample_rate)

        # Per-session lock: prevents process_ready_chunks() from being called concurrently
        # (e.g. if two WebSocket connections somehow share the same session object).
        self._lock = asyncio.Lock()

        # Finalization guard: once True, finalize_session() is a no-op.
        self._finalized = False

    # ── Chunk ingestion ────────────────────────────────────────────────────────

    def add_chunk(self, chunk: AudioChunk) -> bool:
        """
        Adds a chunk to the ready/pending queue.
        Returns True if the chunk was accepted and sequential, False otherwise.
        """
        self.last_activity = datetime.now(timezone.utc)

        if chunk.chunk_id == self.expected_chunk_id:
            self._process_sequential(chunk)
            return True

        elif chunk.chunk_id > self.expected_chunk_id:
            gap = chunk.chunk_id - self.expected_chunk_id
            if gap > MAX_CHUNK_ID_GAP:
                logger.warning(
                    f"Session {self.session_id}: Chunk ID {chunk.chunk_id} is "
                    f"{gap} ahead of expected {self.expected_chunk_id} (gap > {settings.max_chunk_id_gap}). "
                    f"Rejecting to prevent memory exhaustion."
                )
                return False
            if len(self.pending_chunks) >= settings.max_pending_chunks:
                # Evict the lowest pending chunk ID to make room
                oldest_id = min(self.pending_chunks.keys())
                logger.warning(
                    f"Session {self.session_id}: Pending chunk buffer full ({settings.max_pending_chunks}). "
                    f"Evicting oldest pending chunk {oldest_id}."
                )
                self.pending_chunks.pop(oldest_id)
            logger.warning(
                f"Session {self.session_id}: Out-of-order chunk. "
                f"Expected {self.expected_chunk_id}, got {chunk.chunk_id}."
            )
            self.pending_chunks[chunk.chunk_id] = chunk
            return False

        else:
            logger.warning(
                f"Session {self.session_id}: Received old chunk {chunk.chunk_id} "
                f"when expected {self.expected_chunk_id}. Discarding."
            )
            return False

    def _process_sequential(self, chunk: AudioChunk):
        self.ready_chunks.append(chunk)
        self.expected_chunk_id += 1

        # Drain any out-of-order chunks that are now in sequence
        while self.expected_chunk_id in self.pending_chunks:
            next_chunk = self.pending_chunks.pop(self.expected_chunk_id)
            self.ready_chunks.append(next_chunk)
            self.expected_chunk_id += 1

    # ── Inference ──────────────────────────────────────────────────────────────

    async def add_and_process_chunk(self, chunk: AudioChunk) -> List[StreamResponse]:
        """
        Atomically adds a chunk to the queue and triggers processing.
        Protected by a per-session asyncio.Lock to prevent concurrent mutation.
        """
        async with self._lock:
            self.add_chunk(chunk)
            return await self._process_ready_chunks_locked()

    async def process_ready_chunks(self) -> List[StreamResponse]:
        """
        Legacy entrypoint. Prefer add_and_process_chunk for atomic ingestion.
        """
        async with self._lock:
            return await self._process_ready_chunks_locked()

    async def _process_ready_chunks_locked(self) -> List[StreamResponse]:
        responses = []
        while self.ready_chunks:
            chunk = self.ready_chunks.pop(0)

            # Guard against unbounded PCM buffer growth
            incoming = chunk.pcm
            if len(self._pcm_buffer) + len(incoming) > self._max_pcm_buffer_samples:
                excess = (len(self._pcm_buffer) + len(incoming)) - self._max_pcm_buffer_samples
                logger.error(
                    f"Session {self.session_id}: PCM buffer would exceed "
                    f"{settings.max_pcm_buffer_sec}s limit. Trimming {excess} oldest samples."
                )
                self._pcm_buffer = self._pcm_buffer[excess:]

            self._pcm_buffer.extend(incoming)

            # Fire inference once the window threshold is met
            while len(self._pcm_buffer) >= self._inference_window_samples:
                window_pcm = self._pcm_buffer[:self._inference_window_samples]
                # 50% overlap: keep the second half as context for the next window
                self._pcm_buffer = self._pcm_buffer[self._inference_hop_samples:]

                try:
                    result = await InferenceEngine.run_inference(
                        audio_window=np.array(window_pcm, dtype=np.float32),
                        sample_rate=chunk.sample_rate,
                        session_id=self.session_id
                    )
                except Exception as e:
                    self.windows_failed += 1
                    logger.error(f"Inference failed for session {self.session_id}: {e}")
                    # Notify frontend of the failure so it isn't silently lost
                    from models.schemas import ErrorResponse
                    err_resp = StreamResponse(
                        session_id=self.session_id,
                        event="inference_error",
                        chunk_id=chunk.chunk_id,
                        result=None,
                        timestamp=datetime.now(timezone.utc),
                        status="error"
                    )
                    responses.append(err_resp)
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

                # Persist window result to MongoDB
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
                            "smoothed_p_fake": result.smoothed_p_fake,
                            "windows_seen": result.windows_seen,
                            "processing_time_ms": result.processing_time_ms,
                            "branch_timing": result.branch_timing.model_dump(),
                            "acoustic_features": result.acoustic_features.model_dump()
                        })
                except Exception as e:
                    logger.error(f"Failed to save window result to DB: {e}")

                self.processed_windows += 1

        return responses


# ── Session Manager ────────────────────────────────────────────────────────────

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

    @staticmethod
    def _aggregate_results(results: list) -> dict:
        """
        Single authoritative aggregation function.
        All final verdict computation must go through here — no duplicate logic elsewhere.

        Returns dict with: final_verdict, final_p_fake, final_confidence, total_windows.
        """
        if not results:
            return {
                "final_verdict": "UNCERTAIN",
                "final_p_fake": 0.0,
                "final_confidence": 0.0,
                "total_windows": 0
            }

        avg_p_fake = sum(r.get("smoothed_p_fake", r.get("p_fake", 0.0)) for r in results) / len(results)

        if avg_p_fake >= settings.high_threshold:
            final_verdict = "AI"
            final_confidence = round(avg_p_fake, 4)
        elif avg_p_fake <= settings.low_threshold:
            final_verdict = "HUMAN"
            final_confidence = round(1.0 - avg_p_fake, 4)
        else:
            final_verdict = "UNCERTAIN"
            # Distance from the nearest threshold as a confidence proxy
            dist_to_high = settings.high_threshold - avg_p_fake
            dist_to_low  = avg_p_fake - settings.low_threshold
            band_width    = settings.high_threshold - settings.low_threshold
            final_confidence = round(max(dist_to_high, dist_to_low) / (band_width / 2), 4)

        return {
            "final_verdict": final_verdict,
            "final_p_fake": round(avg_p_fake, 4),
            "final_confidence": final_confidence,
            "total_windows": len(results)
        }

    async def finalize_session(self, session_id: str) -> Optional[dict]:
        """
        Aggregates window results, writes the authoritative verdict to MongoDB,
        and cleans up in-memory state.

        Idempotent: if the session was already finalized, returns None immediately.
        Returns the aggregation dict on success.
        """
        session = self.sessions.get(session_id)
        if session is not None and session._finalized:
            logger.info(f"Session {session_id} already finalized — skipping.")
            return None

        if session is not None:
            session._finalized = True

        db = get_db()
        agg = None
        if db is not None:
            try:
                results = await db.window_results.find({"session_id": session_id}).to_list(length=None)
                agg = self._aggregate_results(results)

                windows_failed = session.windows_failed if session else 0

                await db.sessions.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "status": "completed",
                        "end_time": datetime.now(timezone.utc),
                        "final_verdict": agg["final_verdict"],
                        "final_p_fake": agg["final_p_fake"],
                        "final_confidence": agg["final_confidence"],
                        "total_windows": agg["total_windows"],
                        "windows_failed": windows_failed
                    }}
                )
                logger.info(
                    f"Session {session_id} finalized: verdict={agg['final_verdict']}, "
                    f"avg_p_fake={agg['final_p_fake']}, windows={agg['total_windows']}"
                )
            except Exception as e:
                logger.error(f"Failed to finalize session {session_id} in DB: {e}")

        self.remove_session(session_id)
        return agg

    async def cleanup_idle_sessions(self):
        while True:
            now = datetime.now(timezone.utc)
            expired = []
            for sid, session in self.sessions.items():
                if (now - session.last_activity).total_seconds() > settings.session_timeout_sec:
                    expired.append(sid)
            for sid in expired:
                logger.warning(f"Session {sid} timed out — finalizing.")
                await self.finalize_session(sid)
            await asyncio.sleep(10)


# Global singleton
session_manager = SessionManager()
