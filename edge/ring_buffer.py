import numpy as np
import threading

class RingBuffer:
    """
    Thread-safe Ring Buffer for continuous audio capture.
    Holds the last `capacity` samples of audio.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.is_full = False
        self.lock = threading.Lock()

    def append(self, data):
        """Append new audio data (1D numpy array) to the buffer."""
        with self.lock:
            n = len(data)
            if n >= self.capacity:
                # If data is larger than capacity, keep only the latest `capacity` samples
                self.buffer[:] = data[-self.capacity:]
                self.index = 0
                self.is_full = True
            else:
                # Calculate space remaining until the end of the array
                space = self.capacity - self.index
                if n <= space:
                    self.buffer[self.index : self.index + n] = data
                    self.index = (self.index + n) % self.capacity
                else:
                    self.buffer[self.index : self.capacity] = data[:space]
                    self.buffer[0 : n - space] = data[space:]
                    self.index = n - space
                
                if self.index < n:
                    self.is_full = True

    def get_latest(self, num_samples):
        """Return the latest `num_samples` from the buffer."""
        with self.lock:
            if num_samples > self.capacity:
                raise ValueError("Requested more samples than the buffer capacity.")
            
            # If buffer hasn't filled up enough, pad with zeros
            if not self.is_full and self.index < num_samples:
                out = np.zeros(num_samples, dtype=np.float32)
                out[-self.index:] = self.buffer[:self.index]
                return out
                
            if self.index >= num_samples:
                return self.buffer[self.index - num_samples : self.index].copy()
            else:
                out = np.empty(num_samples, dtype=np.float32)
                first_part = num_samples - self.index
                out[:first_part] = self.buffer[self.capacity - first_part:]
                out[first_part:] = self.buffer[:self.index]
                return out

    def clear(self):
        with self.lock:
            self.buffer.fill(0)
            self.index = 0
            self.is_full = False
