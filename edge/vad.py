import numpy as np

class VAD:
    """
    Simple Energy-based Voice Activity Detection.
    Keeps track of audio energy to determine if speech has stopped.
    """
    def __init__(self, energy_threshold=0.01, silence_timeout=1.5):
        self.energy_threshold = energy_threshold
        self.silence_timeout = silence_timeout
        self.silence_frames = 0
        # Assuming inference interval of 0.2s
        self.max_silence_frames = int(silence_timeout / 0.2)

    def is_speaking(self, audio_chunk):
        """
        Returns True if the current chunk contains speech, False otherwise.
        """
        # Calculate RMS energy of the chunk
        energy = np.sqrt(np.mean(np.square(audio_chunk)))
        
        if energy < self.energy_threshold:
            self.silence_frames += 1
        else:
            self.silence_frames = 0
            
        return self.silence_frames < self.max_silence_frames

    def reset(self):
        self.silence_frames = 0
