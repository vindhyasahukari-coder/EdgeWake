import json
import time
from fastapi import WebSocket
from server.asr_engine import ASREngine
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.db import get_connection

# Initialize ASR engine (loads model into memory)
asr = ASREngine()

async def handle_audio_stream(websocket: WebSocket):
    # 1. Receive metadata
    metadata_msg = await websocket.receive_text()
    metadata = json.loads(metadata_msg)
    wake_timestamp = metadata.get("wake_timestamp", 0)
    
    # 2. Receive audio bytes
    audio_bytes = await websocket.receive_bytes()
    t3 = time.time() # Server received audio
    
    # 3. Process ASR
    t4 = time.time() # ASR starts
    transcript = asr.transcribe(audio_bytes)
    t5 = time.time() # Transcript returned
    
    # 4. Calculate latencies
    network_latency = (t3 - wake_timestamp) * 1000 if wake_timestamp else 0
    asr_latency = (t5 - t4) * 1000
    end_to_end_latency = (t5 - wake_timestamp) * 1000 if wake_timestamp else 0
    
    # Estimate bytes
    audio_sent_bytes = len(audio_bytes)
    # The client prints total captured, but we can't perfectly know from server side. 
    # For now we'll just log what we know. The dashboard will use this event.
    
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO events (timestamp, transcript, network_latency_ms, asr_latency_ms, total_latency_ms, audio_sent_bytes) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), transcript, network_latency, asr_latency, end_to_end_latency, audio_sent_bytes)
            )
    except Exception as e:
        print(f"DB Error: {e}")
    
    # 5. Send result back
    result = {
        "transcript": transcript,
        "network_latency_ms": network_latency,
        "asr_latency_ms": asr_latency,
        "latency_ms": end_to_end_latency
    }
    await websocket.send_json(result)
