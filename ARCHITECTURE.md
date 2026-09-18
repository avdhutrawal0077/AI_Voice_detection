# True Tone — Complete System Architecture & Technical Specification

## 1. System Overview & Executive Summary

**True Tone** is a real-time, multi-branch artificial intelligence system designed to detect synthesized, cloned, and deepfake human speech during live audio streams (such as phone calls, video conferences, or live recordings).

Detecting modern AI voice cloning requires more than simple spectral analysis. State-of-the-art text-to-speech (TTS) and voice conversion (VC) models (e.g., ElevenLabs, XTTS, VITS) replicate human pitch and timbre with extreme fidelity. To counter this, True Tone uses a **tri-branch ensemble architecture**:
1. **Branch A (Deep Acoustic Embeddings):** Captures high-level latent representations and neural vocoder artifacts using a fine-tuned Wav2Vec 2.0 transformer.
2. **Branch B (Biomechanical & Spectral Acoustics):** Measures physical vocal tract kinematics (jitter, shimmer, harmonics-to-noise ratio, formants, and spectral flux) via Praat Parselmouth.
3. **Branch C (Probabilistic Fusion & Temporal Smoothing):** Combines neural and acoustic probabilities through a calibrated meta-classifier, smoothed across consecutive time windows to prevent erratic flickering.

---

## 2. End-to-End Dataflow Diagram

```mermaid
flowchart TD
    subgraph Client ["Client Browser (Frontend)"]
        MIC[Microphone Input 16kHz] --> AW[AudioWorklet: audio-chunk-processor.js]
        AW --> VAD[Silero VAD v5 ONNX Engine]
        VAD -->|Voice Active Chunks| WSClient[WebSocket Streaming Engine]
        VAD -.->|Silence Dropped| DROP((Dropped))
        WSClient --> UI[Three.js Waveform & Real-Time Gauges]
    end

    subgraph Backend ["FastAPI Ingestion Server (Backend)"]
        WSClient ==>|WebSocket: /ws/audio/{session_id}| Endpoint[api/endpoints/ingestion.py]
        Endpoint --> SM[services/session_manager.py]
        SM --> BUF[Sliding Window Buffer: 2.0s / 32,000 samples]
    end

    subgraph AI ["AI Pipeline (AI_Pipeline)"]
        BUF --> Bridge[services/ai_pipeline.py]
        Bridge --> INFER[wave_spectrum_inference.py]
        
        subgraph Ensemble ["Tri-Branch Detection Engine"]
            INFER --> BranchA["Branch A: Wav2Vec2 + XGBoost (Deep Neural Latents)"]
            INFER --> BranchB["Branch B: Praat Acoustic + Mel/MFCC Scaler + XGBoost"]
            BranchA --> Fusion["Branch C: Logistic Regression Meta-Classifier"]
            BranchB --> Fusion
        end

        Fusion --> TS[Temporal Smoother - Exponential Smoothing]
    end

    subgraph Persistence ["Persistence Layer"]
        TS --> DB[(MongoDB: voice_detection)]
        DB --> WRes[(Collection: window_results)]
        DB --> Sess[(Collection: sessions)]
    end

    TS ==>|Real-time Verdict & Acoustic Telemetry| Endpoint
    Endpoint ==>|JSON StreamResponse| WSClient
    WSClient --> PDF[Forensic PDF Audit Report Exporter]
```

### Pipeline Timing Chain

The temporal flow of audio data from microphone to model inference is precisely configured:

1. **AudioWorklet Chunk:** 250 ms (captured at the edge)
2. **Silero VAD:** Processes chunks immediately for voice activity
3. **Detection Window:** 2.0 seconds (buffered in backend)
4. **Inference Hop:** 1.0 second (50% sliding window overlap)
5. **Zero Padding:** Padded to 4.0 seconds (during AI pipeline preprocessing)
6. **Model Input:** 64,000 samples (4s × 16kHz)

---

## 3. Directory & Component Breakdown

```
d:\SIH FINAL\
│
├── AI_Pipeline/                  # Machine Learning Engine
│   ├── config/                   # Audio & feature extraction configs
│   │   ├── inference_config.json
│   │   └── preprocessing_config.json
│   ├── models/                   # Pretrained weights & scalers
│   │   ├── branch_b_scaler.pkl
│   │   ├── model_a_branch_a.json
│   │   ├── model_b_branch_b.json
│   │   ├── model_c_fusion.pkl
│   │   └── thresholds.json
│   └── src/
│       └── wave_spectrum_inference.py  # Core inference class
│
├── Backend/                      # Real-Time WebSocket & REST API
│   ├── api/                      # Routing endpoints
│   │   └── endpoints/
│   │       ├── health.py         # System health & DB ping
│   │       ├── auth.py           # User registration (signup) & login
│   │       ├── ingestion.py      # WebSocket stream & session summary
│   │       ├── analyze.py        # Batch audio analysis endpoint
│   │       ├── stream.py         # Streaming REST fallbacks
│   │       └── results.py        # Historical results query
│   ├── core/
│   │   └── config.py             # Pydantic Settings & environment vars
│   ├── db/
│   │   └── database.py           # Async Motor MongoDB connection & indexes
│   ├── models/
│   │   └── schemas.py            # Pydantic data contracts (AudioChunk, etc.)
│   ├── services/
│   │   ├── ai_pipeline.py        # Integration bridge to AI_Pipeline
│   │   ├── session_manager.py    # Chunk reordering, buffering & session state
│   │   ├── audio_processing.py   # Raw byte to PCM conversions
│   │   └── aggregation.py        # Verdict scoring logic
│   ├── main.py                   # FastAPI application initialization
│   ├── requirements.txt          # Backend dependencies
│   ├── test_client.py            # Basic integration test
│   └── test_client_new.py        # WebSocket test client with live telemetry
│
├── frontend/                     # Modern Single-Page Web Dashboard
│   ├── audio-chunk-processor.js  # AudioWorklet for low-latency chunking
│   ├── code.html                 # Complete UI (Three.js, Silero VAD, jsPDF)
│   └── public/assets/
│       └── true-tone-bg.mp4      # Dashboard video background
│
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git rules (excludes venvs, logs, secrets)
├── README.md                     # Quickstart documentation
└── requirements.txt              # Root unified dependencies
```

---

## 4. In-Depth Component Analysis

### A. Frontend Layer (`frontend/`)

#### 1. `audio-chunk-processor.js` (AudioWorklet)
* **What it does:** Runs inside the browser's dedicated low-latency audio rendering thread, completely decoupled from the main JavaScript execution loop.
* **Why it's in the system:** Standard JavaScript `onaudioprocess` runs on the main thread and stutters when the UI renders animations or Three.js frames. The AudioWorklet ensures that audio capture is 100% glitch-free and buffers exactly 16,000 float32 samples per second.
* **Key Mechanism:** Slices raw input into uniform 0.5s or 1.0s chunks and posts them via `MessagePort` to the main window.

#### 2. `code.html` (Complete Dashboard)
* **What it does:** 
  1. Manages microphone permissions and AudioContext initialization.
  2. Embeds an in-browser WebAssembly ONNX inference engine running **Silero VAD v5** (`silero_vad.onnx`).
  3. Renders a 3D procedural audio wave using **Three.js**.
  4. Maintains real-time circular gauges showing detection probability, confidence score, jitter, shimmer, and Harmonics-to-Noise Ratio (HNR).
  5. Uses `jsPDF` and `jspdf-autotable` to generate a comprehensive forensic PDF audit certificate when the session ends.
* **Why it's in the system:** Provides law enforcement, call center operators, and users with instantaneous visual feedback and a legally admissible exportable report.

#### 3. Client-Side Silero VAD Engine
* **What it does:** Classifies incoming audio chunks as SPEECH or SILENCE before transmission.
* **Why it's in the system:** Sending non-speech chunks (silence, keyboard typing, ambient air conditioning) across WebSockets wastes server CPU and causes ML models to make false-positive predictions on noise. Dropping non-speech at the edge reduces backend server load by over 60%.

---

### B. Backend Layer (`Backend/`)

#### 1. `main.py`
* **What it does:** FastAPI application root. Initializes CORS middleware, wires together API routes, and manages the application `lifespan`.
* **Why it's in the system:** Coordinates clean startup (connecting to MongoDB, loading AI pipeline into memory) and graceful shutdown (closing DB sockets, canceling background cleaner tasks).

#### 2. `services/session_manager.py`
* **What it does:**
  * Tracks each live call by a unique `session_id`.
  * Handles chunk reordering (if network packets arrive out-of-order).
  * Accumulates chunks into a sliding window of 2.0 seconds (32,000 samples) with a 50% hop size.
  * Dispatches complete windows to the AI pipeline and persists results into MongoDB asynchronously.
* **Why it's in the system:** Neural audio models require a minimum temporal context (2 seconds) to evaluate natural speech cadences. The session manager ensures window sliding occurs without missing audio boundaries.

#### 3. `services/ai_pipeline.py`
* **What it does:** Acts as the clean architectural bridge between FastAPI and the `AI_Pipeline` directory. Dynamically adds `AI_Pipeline/src` to Python's module path, imports `WaveSpectrumDetector`, and maintains session-specific `TemporalSmoother` instances.
* **Why it's in the system:** Decouples the web service code from the heavy data science code, allowing either to be updated independently.

#### 4. `db/database.py`
* **What it does:** Initializes an asynchronous Motor client for MongoDB, configures database indexes (`session_id`, `chunk_id`), and provides an automatic in-memory fallback if MongoDB is offline.
* **Why it's in the system:** Ensures the system never crashes during a live call even if database infrastructure experiences temporary downtime.

#### 5. `models/schemas.py`
* **What it does:** Contains Pydantic models declaring strict input/output contracts (`AudioChunk`, `StreamResponse`, `AcousticFeatures`, `SessionSummaryResponse`).
* **Why it's in the system:** Enforces data validation and automatic serialization, preventing corrupted audio buffers or malformed JSON from reaching the ML pipeline.

---

### C. AI Pipeline Layer (`AI_Pipeline/`)

#### 1. `src/wave_spectrum_inference.py`
The brain of the system. Contains the `WaveSpectrumDetector` and `TemporalSmoother` classes.

* **Branch A (Wav2Vec 2.0 + XGBoost):**
  * *Architecture:* Pretrained `facebook/wav2vec2-base` (loaded via safe, zero-copy safetensors) extracts 768-dimensional latent representations.
  * *Classifier:* A gradient-boosted decision tree (`model_a_branch_a.json`) trained to identify subtle mathematical artifacts left by neural vocoders (like HiFi-GAN, WaveGlow).
* **Branch B (Acoustic & Spectral ML Features):**
  * *ML Inputs:* Statistical spectral and acoustic features including MFCC mean/std, spectral centroid, bandwidth, spectral rolloff, spectral flux, RMS, and Zero Crossing Rate (ZCR).
  * *Classifier:* Scaled via `branch_b_scaler.pkl` and evaluated by `model_b_branch_b.json`.
  * *Separate Acoustic Evidence (Dashboard/Report Only):* Praat Parselmouth extracts pitch ($F_0$), micro-pitch perturbations (**jitter**), amplitude perturbations (**shimmer**), and glottal wave purity (**HNR**). These are passed directly to the telemetry output to explain the human vocal cord biomechanics, but are *not* directly ingested by the XGBoost model.
* **Branch C (Fusion Meta-Classifier):**
  * *Architecture:* A calibrated Logistic Regression model (`model_c_fusion.pkl`) taking probability outputs from Branch A and Branch B to yield a unified $P(\text{fake})$ score.
* **Temporal Smoother:**
  * *Algorithm:* Exponential moving average:
    $$S_t = \alpha \cdot P_t + (1 - \alpha) \cdot S_{t-1}$$
  * *Why:* Audio often contains momentary pauses or isolated plosives. Smoothing ensures a single noisy syllable does not trigger a false alarm during an ongoing conversation.

---

## 5. Database Schema & Collections

The MongoDB database is named **`voice_detection`**.

### Collection 1: `window_results`
Stores the detailed forensic analysis for each processed audio window.

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | MongoDB unique identifier |
| `session_id` | String | Unique UUID for the call/session |
| `chunk_id` | Integer | Sequential audio chunk index |
| `timestamp` | DateTime | Timestamp when window was evaluated |
| `verdict` | String | `"HUMAN"` or `"AI"` |
| `confidence` | Double | Confidence level (0.0 to 1.0) |
| `p_fake` | Double | Raw probability of synthetic speech |
| `p_a` | Double | Probability score from Branch A |
| `p_b` | Double | Probability score from Branch B |
| `processing_time_ms` | Double | End-to-end inference latency |
| `branch_timing` | Object | Latency breakdown for models A, B, and C |
| `acoustic_features` | Object | Pitch ($F_0$), Jitter, Shimmer, HNR, Spectral Centroid |
| `temporal_smoothing` | Object | Smoothed score and windows counter |

* **Indexes:** `{ session_id: 1, chunk_id: 1 }` (compound), `{ session_id: 1 }`.

### Collection 2: `sessions`
Stores session metadata, start/end times, and cumulative aggregate verdicts for compliance auditing.

### Collection 3: `users`
Stores user credentials, password hashes with per-user cryptographic salts, and authentication history.

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | MongoDB unique identifier |
| `email` | String | Unique user email (indexed) |
| `password_hash` | String | SHA-256 PBKDF2 hashed password (100,000 iterations) |
| `salt` | String | Cryptographically random 16-byte hex salt |
| `created_at` | DateTime | Timestamp of user registration |
| `last_login` | DateTime | Timestamp of most recent authentication |

* **Indexes:** `{ email: 1 }` (unique).

---

## 6. How to Deploy & Maintain

1. **Environment Variables:** Copy `.env.example` to `.env` and set `MONGODB_URI`.
2. **Backend Execution:**
   ```bash
   cd Backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
   ```
3. **Frontend Serving:** Open `frontend/code.html` directly or serve via any static web server (Nginx, Caddy, Vercel, S3/CloudFront).
4. **Offline Resilience:** If MongoDB is unreachable, the system automatically runs in-memory without crashing, continuing to deliver live WebSocket verdicts to the dashboard.
