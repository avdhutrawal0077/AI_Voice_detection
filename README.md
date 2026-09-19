# True Tone — AI Voice Clone & Impersonation Detection

This is the final integrated codebase for **True Tone**. It features a modern browser-based frontend for real-time visualization and Voice Activity Detection (VAD), along with a powerful FastAPI backend running a Python-based machine learning pipeline for deepfake detection.

## Prerequisites

1. **Python 3.10+** (Ensure python and pip are in your PATH).
2. **MongoDB** (Ensure MongoDB is running locally on the default port 27017).

## Installation & Setup

1. **Clone or Extract the Repository:**
   Open a terminal and navigate to this project folder (`D:\SIH FINAL` or wherever you extracted it).

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   * On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   * On macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the System

You need to run the backend server and open the frontend interface.

### Step 1: Start the Backend (FastAPI + AI Pipeline)
Ensure your virtual environment is active, then navigate to the Backend folder and run the server:
```bash
cd Backend
uvicorn main:app --reload
```
You should see output indicating that the AI Pipeline loaded successfully and the server is running on `http://127.0.0.1:8000`.

### Step 2: Launch the Frontend
Open the `frontend/code.html` file in any modern web browser. 

* **To start the analysis:** Click the **Start Analysis** button. This will request microphone permissions, start the live audio capture, initialize the local VAD, and establish a WebSocket connection to the backend.
* **To end the analysis:** Click the **Complete Analysis** button. This will automatically fetch the final analysis report from the backend.

## Project Structure
- `Backend/`: FastAPI server, database logic, WebSocket endpoints, and the integration layer (`ai_pipeline.py`).
- `AI_Pipeline/`: The core Machine Learning models, configurations, and feature extraction scripts (`wave_spectrum_inference.py`).
- `frontend/`: The User Interface (`code.html`), static assets, and the AudioWorklet chunk processor.
