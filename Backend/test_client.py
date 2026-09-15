import asyncio
import websockets
import numpy as np
import httpx
import sys

async def stream_audio_ws():
    uri = "ws://localhost:8000/api/v1/ws/stream"
    print(f"Connecting to {uri}")
    
    # Generate 5 seconds of mock audio (alternating loud sine wave and silence)
    sr = 16000
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    audio_float = np.sin(2 * np.pi * 440 * t)
    audio_float[sr*2:sr*3] = 0.0 # Inject 1 sec silence
    audio_pcm16 = (audio_float * 32767).astype(np.int16)
    raw_bytes = audio_pcm16.tobytes()
    
    chunk_size = 8000 # Send 0.25s chunks
    
    async with websockets.connect(uri) as ws:
        init_msg = await ws.recv()
        print("Connected:", init_msg)
        
        for i in range(0, len(raw_bytes), chunk_size):
            chunk = raw_bytes[i:i+chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.1) # Simulate real-time streaming delay
            
            # Non-blocking read to get intermediate results
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                print("Server:", msg)
            except asyncio.TimeoutError:
                pass
                
        print("Finished streaming.")
        # Final read loop
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                print("Server Finalizing:", msg)
        except asyncio.TimeoutError:
            pass

def test_api():
    print("Fetching Health...")
    r = httpx.get("http://localhost:8000/api/v1/health")
    print(r.json())
    
    print("\nFetching recent results...")
    r = httpx.get("http://localhost:8000/api/v1/results")
    sessions = r.json()
    print(f"Found {len(sessions)} sessions.")
    if sessions:
        sid = sessions[0]['session_id']
        print(f"\nFetching specific session {sid}...")
        r = httpx.get(f"http://localhost:8000/api/v1/results/{sid}")
        print("Status Code:", r.status_code)
        print("Total Windows:", len(r.json().get('windows', [])))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ws":
        asyncio.run(stream_audio_ws())
    else:
        test_api()