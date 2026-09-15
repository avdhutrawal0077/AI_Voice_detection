import asyncio
import websockets
import json
import math
import time

async def simulate_client():
    uri = "ws://localhost:8000/ws/audio/test_session_demo"
    
    # Generate 5 seconds of a fake 440Hz sine wave as PCM float32
    sample_rate = 16000
    duration = 5
    t = [i / sample_rate for i in range(sample_rate * duration)]
    pcm_full = [math.sin(2 * math.pi * 440 * i) for i in t]
    
    # Chunk size: 1 second
    chunk_size = sample_rate * 1
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket.")
            
            chunk_id = 1
            for start in range(0, len(pcm_full), chunk_size):
                end = start + chunk_size
                chunk_pcm = pcm_full[start:end]
                
                payload = {
                    "session_id": "test_session_demo",
                    "chunk_id": chunk_id,
                    "pcm": chunk_pcm,
                    "sample_rate": sample_rate,
                    "format": "mono_float32"
                }
                
                print(f"Sending chunk {chunk_id}...")
                await websocket.send(json.dumps(payload))
                
                # Receive any responses that came back immediately
                try:
                    while True:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                        print(f"Received: {response}")
                except asyncio.TimeoutError:
                    pass
                
                chunk_id += 1
                time.sleep(0.5) # Simulate real-time streaming delay
            
            print("Finished sending all chunks.")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_client())
