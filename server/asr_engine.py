import io
import wave
import numpy as np

import os

# Force HuggingFace to run in offline mode to prevent network timeouts (WinError 10060)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# We'll try to import transformers. If not installed, we'll use a dummy/mock ASR.
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: 'transformers' library not found. Falling back to Mock ASR.")
    print("Run: pip install transformers librosa soundfile")

class ASREngine:
    def __init__(self, model_name="openai/whisper-tiny"):
        if HAS_TRANSFORMERS:
            print(f"Loading ASR model '{model_name}'...")
            self.pipe = pipeline("automatic-speech-recognition", model=model_name)
            print("ASR model loaded.")
        else:
            self.pipe = None
            
    def transcribe(self, wav_bytes):
        """
        Takes raw WAV bytes, decodes, and returns a transcript.
        """
        if not HAS_TRANSFORMERS:
            # Mock ASR
            return "This is a simulated ASR response (transformers not installed)."
            
        try:
            # Decode WAV bytes in memory to avoid ffmpeg dependency on Windows
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
                
            # Convert int16 bytes to float32 numpy array normalized between -1.0 and 1.0
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Pass directly to transformers pipeline
            result = self.pipe({"raw": audio_np, "sampling_rate": framerate})
            return result["text"].strip()
        except Exception as e:
            print(f"ASR Error: {e}")
            return "Error during transcription."
