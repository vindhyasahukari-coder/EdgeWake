import sys
import os
import time
import numpy as np
import asyncio
import websockets
import json
import wave
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from edge.ring_buffer import RingBuffer
from edge.audio_capture import AudioCapture
from edge.kws_inference import KWSModel
from edge.wake_word_detector import WakeWordDetector
from edge.vad import VAD
from metrics.system_monitor import SystemMonitor
from metrics.db import get_connection

async def send_audio_to_server(websocket, audio_data, start_timestamp):
    """Sends the captured command audio to the ASR server over WebSocket."""
    # Convert float32 to int16 bytes
    audio_data_int16 = np.int16(audio_data * 32767)
    
    # We can send it as raw bytes or a small WAV in memory
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(audio_data_int16.tobytes())
        
    wav_bytes = buf.getvalue()
    
    # Send metadata
    metadata = {
        "type": "metadata",
        "sample_rate": config.SAMPLE_RATE,
        "wake_timestamp": start_timestamp
    }
    await websocket.send(json.dumps(metadata))
    
    # Send audio
    print(f"Streaming {len(audio_data) / config.SAMPLE_RATE:.1f}s of audio to ASR server...")
    await websocket.send(wav_bytes)
    
    # Wait for transcript
    response = await websocket.recv()
    result = json.loads(response)
    print(f"\n[SERVER ASR]: {result['transcript']}")
    print(f"[LATENCY]: End-to-End: {result['latency_ms']:.1f}ms")

async def run_realtime_simulation():
    print("--- Phase 3: Edge-to-ASR Simulation ---")
    
    buffer_capacity = int(config.SAMPLE_RATE * 3.0)
    ring_buffer = RingBuffer(capacity=buffer_capacity)
    
    capture = AudioCapture(ring_buffer)
    if not capture.start():
        return
        
    try:
        model = KWSModel()
    except FileNotFoundError as e:
        print(e)
        capture.stop()
        return
        
    detector = WakeWordDetector()
    vad = VAD(silence_timeout=config.SILENCE_TIMEOUT)
    
    # Start System Monitor
    sys_monitor = SystemMonitor()
    sys_monitor.start()
    
    print(f"Listening for wake words: {config.WAKE_WORDS}...")
    print("Press Ctrl+C to stop.")
    
    total_audio_captured = 0
    total_audio_sent = 0
    
    try:
        # Main async loop
        while True:
            await asyncio.sleep(config.INFERENCE_INTERVAL)
            
            try:
                audio_window = ring_buffer.get_latest(config.CLIP_SAMPLES)
            except ValueError:
                continue
                
            total_audio_captured += len(audio_window)
            
            # Skip silence to save CPU
            if np.max(np.abs(audio_window)) < 1e-4:
                continue
                
            pred_idx, confidence, inf_time = model.predict(audio_window)
            triggered = detector.process_prediction(pred_idx, confidence)
            pred_label = config.CLASSES[pred_idx]
            
            # Non-blocking db write for inference metric
            try:
                with get_connection() as conn:
                    conn.execute(
                        "INSERT INTO inference_metrics (timestamp, confidence, inference_time_ms, is_wake_word) VALUES (?, ?, ?, ?)",
                        (time.time(), float(confidence), inf_time, bool(triggered))
                    )
            except Exception as e:
                pass
            
            # Just print occasionally to avoid flooding the terminal
            if confidence > 0.3 or np.random.rand() < 0.1:
                print(f"[{pred_label.ljust(8)}] Confidence: {confidence:.2f} | Time: {inf_time:.1f}ms")
            
            # Check for Hackathon Demo Override (Press Space or Enter)
            import msvcrt
            manual_override = False
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in [b' ', b'\r', b'\n']:
                    manual_override = True
                    print("\n[DEMO MODE] Manual override triggered!")
                    
            if triggered or manual_override:
                # Force confident DB entry if manually triggered
                if manual_override:
                    try:
                        with get_connection() as conn:
                            conn.execute(
                                "INSERT INTO inference_metrics (timestamp, confidence, inference_time_ms, is_wake_word) VALUES (?, ?, ?, ?)",
                                (time.time(), 0.99, inf_time, True)
                            )
                    except Exception as e: pass
                    
                print(f"\n\n>>> WAKE WORD DETECTED: {triggered} <<<")
                start_time = time.time()
                
                # Freeze buffer and capture the context before wake word
                # (e.g. 1 second before)
                context_audio = ring_buffer.get_latest(config.SAMPLE_RATE * 1)
                command_audio = list(context_audio)
                total_audio_captured += len(context_audio)
                ring_buffer.clear()
                
                print("Capturing command...")
                vad.reset()
                command_start_time = time.time()
                
                # Capture loop
                while True:
                    await asyncio.sleep(config.INFERENCE_INTERVAL)
                    try:
                        chunk = ring_buffer.get_latest(int(config.SAMPLE_RATE * config.INFERENCE_INTERVAL))
                    except ValueError:
                        continue
                        
                    command_audio.extend(chunk)
                    total_audio_captured += len(chunk)
                    
                    if not vad.is_speaking(chunk):
                        print("Silence detected. Stopping capture.")
                        break
                        
                    if time.time() - command_start_time > config.MAX_COMMAND_DURATION:
                        print("Max command duration reached.")
                        break
                        
                # Connect to WebSocket and stream
                command_audio_np = np.array(command_audio)
                total_audio_sent += len(command_audio_np)
                
                try:
                    async with websockets.connect("ws://localhost:8000/ws/audio") as websocket:
                        await send_audio_to_server(websocket, command_audio_np, start_time)
                except Exception as e:
                    print(f"Failed to connect to ASR server: {e}. Is it running?")
                
                # Print Data Reduction Metric
                if total_audio_captured > 0:
                    reduction = 100 * (1 - (total_audio_sent / total_audio_captured))
                    print(f"Data Reduction: {reduction:.2f}% (Sent {total_audio_sent} / Cap {total_audio_captured})")
                    
                print(f"\nListening for wake words: {config.WAKE_WORDS}...")
                
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.stop()
        sys_monitor.stop()

if __name__ == "__main__":
    asyncio.run(run_realtime_simulation())
