# Wave Spectrum — Final Bug & Fix List

**Branch:** `feat/true-tone-system`  
**Repository:** `avdhutrawal0077/AI_Voice_detection`

> **Scope:** This document covers backend, AI pipeline, streaming, session management, configuration, and architecture issues.  
> The React UI/build output is intentionally excluded.

---

## 🔴 A. Core Runtime / Correctness Issues

### 1. Temporal Smoothing Is Not Actually EMA

**File:** `AI_Pipeline/src/wave_spectrum_inference.py`

#### Problem

`TemporalSmoother.update()` currently calculates the smoothed score using:

```python
smoothed = float(np.mean(self.history))
```

This is a **simple moving average**, not an **Exponential Moving Average (EMA)**.

However, the architecture describes the temporal smoothing mechanism as EMA.

Therefore:

```text
Implementation ≠ Architecture
```

#### Why it matters

The smoothing behavior of the detector is different from what the architecture claims.

#### Proper Fix

If EMA is the intended design, implement:

```text
S(t) = α × P(t) + (1 − α) × S(t−1)
```

and make the following consistent:

- implementation
- configuration
- architecture documentation
- frontend terminology

**Priority:** 🔴 Critical

---

### 2. Session Summary Uses a Different Verdict Rule

**File:** `Backend/api/endpoints/ingestion.py`

#### Problem

The live AI pipeline uses calibrated thresholds from:

```text
AI_Pipeline/models/thresholds.json
```

But the session summary currently determines the final verdict using a separate rule:

```python
final_verdict = "AI" if avg_confidence > 0.5 else "HUMAN"
```

This means the live result and final session result can theoretically disagree.

For example:

```text
Live inference:
p_fake = 0.45
→ UNCERTAIN

Session summary:
average = 0.45
→ HUMAN
```

#### Why it matters

There must be **one authoritative decision system**.

Otherwise the user can see one verdict during the call and another in the final report.

#### Proper Fix

Create one central aggregation/verdict mechanism:

```text
Window predictions
        ↓
Temporal smoothing
        ↓
Session aggregation
        ↓
Final verdict
```

Both:

- live results
- session summary

must use this same logic.

**Priority:** 🔴 Critical

---

### 3. New Streaming Sessions Do Not Have a Complete Database Lifecycle

**Files:**

- `Backend/api/endpoints/ingestion.py`
- `Backend/services/session_manager.py`
- `Backend/db/database.py`

#### Problem

The new `/ws/audio/{session_id}` pipeline processes and stores window results, but the session itself does not have the same complete lifecycle as the old streaming implementation.

The intended lifecycle is:

```text
Session created
      ↓
IN_PROGRESS
      ↓
Window results
      ↓
Final aggregation
      ↓
COMPLETED
```

The new pipeline primarily handles `window_results`.

#### Why it matters

History/report functionality needs a reliable session-level record containing:

- session ID
- start time
- end time
- status
- final verdict
- final score
- number of windows
- session metadata

#### Proper Fix

Make the new streaming session lifecycle responsible for:

1. Creating the session document.
2. Marking it `in_progress`.
3. Recording window results.
4. Calculating the final result.
5. Updating the session to `completed`.
6. Cleaning up session state.

The database should reflect the actual runtime lifecycle rather than being populated as an afterthought.

**Priority:** 🔴 Critical

---

# 🟠 B. Architecture / Codebase Consistency Issues

### 4. Two Different Backend Architectures Exist

**Files include:**

```text
Backend/api/endpoints/stream.py
Backend/api/endpoints/analyze.py
Backend/services/audio_processing.py
Backend/services/ml_dummy.py
Backend/services/aggregation.py
```

alongside the new production architecture:

```text
Backend/api/endpoints/ingestion.py
Backend/services/session_manager.py
Backend/services/ai_pipeline.py
AI_Pipeline/src/wave_spectrum_inference.py
```

#### Problem

The repository contains two generations of the system.

### Old pipeline

```text
Audio
 ↓
Old feature extraction
 ↓
ml_dummy
 ↓
Old aggregation
```

### New pipeline

```text
AudioWorklet
 ↓
Silero VAD
 ↓
WebSocket
 ↓
Session Manager
 ↓
Wave Spectrum Detector
 ↓
Branch A + B + C
 ↓
Temporal smoothing
```

The old endpoints are currently disabled in `main.py`, so this is not necessarily a runtime failure.

#### Why it matters

It creates ambiguity about which pipeline is the real production system.

#### Proper Fix

Explicitly establish:

```text
NEW WAVE SPECTRUM PIPELINE
        ↓
Production architecture
```

Then remove obsolete production code.

If mock functionality is required for development, place it explicitly under a testing/mock structure.

**Priority:** 🟠 Important

---

### 5. Branch B Documentation Does Not Match Its Actual Classifier Features

**Files:**

- `ARCHITECTURE.md`
- `AI_Pipeline/src/wave_spectrum_inference.py`

#### Problem

The architecture can imply that Branch B directly uses features such as:

- pitch
- jitter
- shimmer
- HNR
- spectral features

for the XGBoost classifier.

However, the actual `extract_branch_b()` classifier input primarily contains statistical acoustic/spectral features such as:

- MFCC mean/std
- spectral centroid
- bandwidth
- spectral rolloff
- spectral flux
- RMS
- ZCR
- etc.

The Praat-derived:

- pitch
- jitter
- shimmer
- HNR

are extracted separately for acoustic evidence/display.

#### Why it matters

The documentation should accurately describe what the trained model actually receives.

#### Proper Fix

Keep the trained model unchanged unless retraining is intentionally planned.

Document the distinction:

```text
Branch B
    ↓
Acoustic/Spectral ML Features
    ↓
XGBoost
```

and separately:

```text
Acoustic Evidence
    ↓
Pitch / Jitter / Shimmer / HNR / etc.
    ↓
Dashboard + Report
```

Do not modify the classifier simply to make the documentation match.

**Priority:** 🟠 Important

---

### 6. Detection Window vs Model Input Duration Must Be Explicit

Current design:

```text
Sample rate:             16 kHz
Streaming inference:      2 sec
Inference hop:            1 sec
Model input:              4 sec
Model samples:            64,000
```

#### Problem

The backend performs inference on a 2-second streaming window, while the AI model expects 4 seconds and therefore receives padded input.

This is not inherently a bug.

The problem is that the relationship must be explicit and tied to the training configuration.

#### Proper Fix

Document/configure the complete pipeline:

```text
AudioWorklet chunk
      ↓
250 ms
      ↓
VAD
      ↓
2-second detection window
      ↓
1-second hop
      ↓
Pad to 4 seconds
      ↓
64,000 samples
      ↓
Model inference
```

Verify that the 4-second model input matches the actual model training/preprocessing configuration.

**Priority:** 🟠 Important

---

# 🟡 C. Smaller Implementation Problems

### 7. Acoustic Feature Duration Fallback Uses an Invalid Variable

**File:**

```text
AI_Pipeline/src/wave_spectrum_inference.py
```

#### Problem

The fallback duration calculation references `sound.duration` even though the relevant `sound` object is not reliably defined in that scope.

#### Why it matters

It can produce an exception when the normal nonzero-audio path is not available.

#### Proper Fix

Calculate duration directly from the actual number of valid samples:

```text
actual_samples / sample_rate
```

and use a well-defined fallback when no samples are present.

The function should have an explicit:

```text
PCM
 ↓
valid sample count
 ↓
duration
```

data flow.

**Priority:** 🟡

---

### 8. Multiple Configuration Variables Define Window Behavior

**File:**

```text
Backend/core/config.py
```

There are settings such as:

```python
window_duration_sec = 1.0
hop_duration_sec = 1.0
```

alongside:

```python
inference_window_sec = 2.0
```

#### Problem

There are multiple concepts/configuration values describing inference windows.

This creates a risk that future code uses the wrong configuration.

#### Proper Fix

Create one canonical streaming configuration containing:

```text
sample_rate
chunk_duration
inference_window
inference_hop
model_input_duration
```

Remove obsolete window settings.

There should be one source of truth.

**Priority:** 🟡

---

### 9. Legacy `audio_processing.py` Creates a Second Audio-Processing Pipeline

**File:**

```text
Backend/services/audio_processing.py
```

It contains its own:

- audio decoding
- RMS VAD
- MFCC extraction
- Mel extraction
- window generation

while the actual production AI pipeline has its own preprocessing and feature extraction.

#### Problem

There are now two sources of truth for audio processing.

#### Proper Fix

Remove the legacy implementation when the old endpoints are removed.

If a utility is genuinely required by the production system, move it into the correct production module and give it a single owner.

**Priority:** 🟡

---

### 10. `ml_dummy.py` Remains Part of the Old Backend Path

**File:**

```text
Backend/services/ml_dummy.py
```

#### Problem

The actual system uses:

```text
WaveSpectrumDetector
```

but the old backend still contains:

```text
ml_dummy
```

#### Why it matters

A final repository should clearly distinguish:

```text
Production AI
```

from:

```text
Mock/Test AI
```

#### Proper Fix

Remove the dummy production path.

If mock predictions are required for frontend testing, move them into an explicit:

```text
tests/
mock/
development/
```

structure.

**Priority:** 🟡

---

# 🟢 D. Already Fixed / No Action Required

These were previous issues and have been rechecked.

### 11. Duplicate WebSocket implementation

**Status:** ✅ Fixed

There is now one `TrueToneSocket`.

The previous duplicate implementation and overwritten `sendChunk()` problem is gone.

---

### 12. VAD bypass

**Status:** ✅ Fixed

The direct:

```text
Chunk → WebSocket
```

path has been removed.

The intended flow is now:

```text
AudioWorklet
      ↓
Chunk Manager
      ↓
Silero VAD
      ↓
Speech Queue
      ↓
WebSocket
```

---

### 13. Session ID mismatch

**Status:** ✅ Fixed

The socket now uses the session ID supplied by the audio/session layer rather than generating an unrelated second ID.

The intended flow is:

```text
One session ID
      ↓
Frontend
      ↓
WebSocket
      ↓
Backend
      ↓
Database
```

---

### 14. 2-second Window / 50% Overlap

**Status:** ✅ Fixed

The current implementation uses:

```text
Window = 2 seconds
Hop    = 1 second
```

Therefore:

```text
Window 1: 0 ───── 2 sec
Window 2:     1 ───── 3 sec
Window 3:         2 ───── 4 sec
Window 4:             3 ───── 5 sec
```

This is the intended **50% overlap**.

---

### 15. 2-second Streaming Window → 4-second Model Input

**Status:** ✅ Consistent

The current pipeline intentionally uses:

```text
2-second detection window
        ↓
zero padding
        ↓
4-second / 64,000-sample model input
```

This should still be verified against the model's training configuration, but the current inference code is internally consistent.

---

# Final Fix Order

## Phase 1 — Correctness

```text
[ ] 1. Implement actual EMA or explicitly change architecture to moving average
[ ] 2. Create one authoritative verdict/aggregation system
[ ] 3. Implement complete streaming session database lifecycle
```

## Phase 2 — Architecture Cleanup

```text
[ ] 4. Remove legacy backend pipeline
[ ] 5. Correct Branch B documentation
[ ] 6. Explicitly define 2-sec detection → 4-sec model input
```

## Phase 3 — Code Cleanup

```text
[ ] 7. Fix acoustic duration fallback
[ ] 8. Consolidate window configuration
[ ] 9. Remove legacy audio_processing pipeline
[ ] 10. Remove/isolate ml_dummy
```

## Phase 4 — End-to-End Validation

```text
[ ] 11. Browser microphone capture
[ ] 12. AudioWorklet chunk generation
[ ] 13. Silero VAD
[ ] 14. Speech-only WebSocket transmission
[ ] 15. Correct session ID throughout
[ ] 16. 2-sec window / 1-sec hop
[ ] 17. Branch A inference
[ ] 18. Branch B inference
[ ] 19. Branch C fusion
[ ] 20. Temporal smoothing
[ ] 21. MongoDB window persistence
[ ] 22. MongoDB session persistence
[ ] 23. Live result returned to frontend
[ ] 24. Final session result
[ ] 25. History/report consistency
```

---

# Definition of "Ready to Merge"

The branch should satisfy all of these:

```text
                    WAVE SPECTRUM
                          │
                     Microphone
                          │
                    AudioWorklet
                     250 ms chunks
                          │
                     Silero VAD
                          │
                   Speech-only queue
                          │
                     WebSocket
                          │
                      FastAPI
                          │
                   Session Manager
                          │
              2 sec window / 1 sec hop
                          │
                 ┌────────┴────────┐
                 │                 │
             Branch A          Branch B
             Wav2Vec2         Acoustic/Spectral
             + XGBoost          + XGBoost
                 │                 │
                 └────────┬────────┘
                          │
                    Branch C Fusion
                          │
                   Final p(fake)
                          │
                   Temporal EMA
                          │
                  Session Aggregation
                          │
                ┌─────────┴─────────┐
                │                   │
             MongoDB            WebSocket
                │                   │
             History            Dashboard
                │
              Report
```

**The key principle:** there should be **one production audio pipeline, one inference pipeline, one verdict mechanism, one session lifecycle, and one source of truth for configuration.**

That is the standard to use before merging this branch into `main`.