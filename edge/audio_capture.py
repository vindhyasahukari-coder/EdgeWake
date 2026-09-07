import sys
import os

try:
    import sounddevice as sd
except Exception as e:
    print(f"Error importing sounddevice (audio backend might be missing): {e}")
    sd = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class AudioCapture:
    def __init__(self, ring_buffer):
        self.ring_buffer = ring_buffer
        self.stream = None
        self.is_running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """This is called for each audio block by sounddevice."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        
        # Flatten the 1D chunk and append to ring buffer
        audio_chunk = indata[:, 0]
        self.ring_buffer.append(audio_chunk)

    def start(self):
        if sd is None:
            print("Sounddevice not available. Cannot start audio capture.")
            return False
            
        print(f"Starting audio capture at {config.SAMPLE_RATE} Hz")
        self.stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=config.CHUNK_SIZE,
            callback=self._audio_callback
        )
        self.stream.start()
        self.is_running = True
        return True

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False
        print("Audio capture stopped.")
