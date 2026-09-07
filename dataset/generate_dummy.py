import os
import sys
import numpy as np
import wave

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def save_wav(filename, audio_data, sample_rate):
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    audio_data_int16 = np.int16(audio_data * 32767)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 2 bytes = 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_int16.tobytes())

def generate_dummy_data(num_samples_per_class=20):
    print("Generating dummy data...")
    for label in config.CLASSES:
        label_dir = os.path.join(config.RAW_DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            filename = os.path.join(label_dir, f"dummy_{label}_{i}.wav")
            
            # Generate random noise or simple sine waves as dummy data
            duration = config.CLIP_DURATION
            t = np.linspace(0, duration, int(config.SAMPLE_RATE * duration), endpoint=False)
            
            if label == "keyword":
                # Sine wave 440Hz
                audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            else:
                # Random noise
                audio_data = np.random.randn(int(config.SAMPLE_RATE * duration)).astype(np.float32) * 0.1
                
            save_wav(filename, audio_data, config.SAMPLE_RATE)
    print("Done generating dummy data.")

if __name__ == "__main__":
    generate_dummy_data()
